"""
Single-patient inference pipeline for ASTRA.

Loads a trained model and deployment bundle, runs predictions and SHAP
explanations on individual patients without needing the full training
data pipeline or dataloaders.

Usage:
    session = InferenceSession.load("my_model")
    result = session.predict(x_ts, x_ts_cat, tab_df)
    shap_result = session.explain(x_ts, x_ts_cat, tab_df)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from astra.data.dataloader import (
    get_trajectory_lengths,
    load_deployment_bundle,
    normalize_new_patient,
)
from astra.models.hybrid.model import TSTabFusionTransformerMultiHot


# ============================================================================
# RESULT DATACLASSES
# ============================================================================

@dataclass
class InferenceResult:
    """Result of a single-patient prediction."""
    pid: Any
    probability: float                              # P(deceased_30d)
    trajectory_length: int                          # Actual data timesteps
    censor_step: Optional[int] = None               # Timestep evaluated at
    predictions_over_time: Optional[np.ndarray] = None  # [seq_len] (temporal only)


@dataclass
class SHAPResult:
    """SHAP explanation for a single patient."""
    pid: Any
    ts_shap: Dict[str, np.ndarray]                       # {channel: [seq_len]}
    cat_ts_shap: Optional[Dict[str, np.ndarray]] = None  # {category: [seq_len]}
    static_cat_shap: Optional[Dict[str, float]] = None   # {feature: importance}
    static_cont_shap: Optional[Dict[str, float]] = None  # {feature: importance}
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    eval_timestep: Optional[int] = None                  # step the model was evaluated at


# _SHAPModelWrapper and _embed_categorical_features live in
# astra.evaluation.behavior (ModelWrapperWithRawCatTS / embed_categorical_features).
# Imported lazily inside explain() to keep pipeline.py dependency-light.


# ============================================================================
# INFERENCE SESSION
# ============================================================================

class InferenceSession:
    """
    Loads a trained model and all deployment artifacts once.
    Supports repeated single-patient inference and SHAP explanation.
    """

    def __init__(self, model, bundle, device='cpu'):
        self.model = model
        self.bundle = bundle
        self.device = device
        self.is_temporal = bundle['model_params']['temporal_head']

        # Pre-load SHAP background on device
        bg = bundle['shap_background']
        if bg is not None:
            self._bg = {
                'ts': torch.from_numpy(bg['ts']).float().to(device),
                'ts_cat': torch.from_numpy(bg['ts_cat']).float().to(device),
                'cat': torch.from_numpy(bg['cat']).long().to(device),
                'cont': torch.from_numpy(bg['cont']).float().to(device),
            }
        else:
            self._bg = None

    @classmethod
    def load(cls, model_name, device=None, bundle_dir='models/deployment',
             weights_dir='models'):
        """
        Load deployment bundle and model weights.

        Args:
            model_name: Name of the model (matches training save name).
            device: 'cuda', 'cpu', or None for auto-detect.
            bundle_dir: Where deployment bundles are stored.
            weights_dir: Where .pth model weights are stored.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        bundle = load_deployment_bundle(model_name, bundle_dir)
        params = bundle['model_params']

        # Build model from saved params (no data dict needed)
        model = TSTabFusionTransformerMultiHot(
            c_in=params['c_in'],
            c_out=2,
            seq_len=params['seq_len'],
            classes=params['classes'],
            cont_names=params['cont_names'],
            ts_cat_dims=params['ts_cat_dims'],
            d_model=params['d_model'],
            n_layers=params['n_layers'],
            n_heads=params['n_heads'],
            fc_dropout=params['fc_dropout'],
            res_dropout=params['res_dropout'],
            fc_mults=params['fc_mults'],
            temporal_head=params['temporal_head'],
            causal=params['causal'],
            temporal_head_dropout=params['temporal_head_dropout'],
        )

        # Load weights (FastAI format: {'model': state_dict, ...})
        weights_path = os.path.join(weights_dir, f'{model_name}.pth')
        checkpoint = torch.load(weights_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()
        model.to(device)

        return cls(model, bundle, device)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_tensors(self, x_ts, x_ts_cat, tab_df):
        """
        Normalize raw patient data and convert to model-ready tensors.

        Args:
            x_ts: np.ndarray [n_channels, seq_len] or [1, n_channels, seq_len]
                  Raw (unnormalized) continuous time series.
            x_ts_cat: np.ndarray [n_cat_dims, seq_len] or [1, n_cat_dims, seq_len]
                      Multi-hot encoded categorical time series.
            tab_df: pd.DataFrame with one row containing static features.

        Returns:
            (x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len) — all tensors on device
        """
        if x_ts.ndim == 2:
            x_ts = x_ts[np.newaxis, ...]
        if x_ts_cat.ndim == 2:
            x_ts_cat = x_ts_cat[np.newaxis, ...]

        tab_df = tab_df.copy()
        num_cols = self.bundle['tab_feature_names']
        tab_scaler = self.bundle['tab_scaler']

        # Record which numeric columns are NaN BEFORE filling (for _na indicators)
        na_mask = {}
        if num_cols:
            for col in num_cols:
                if col in tab_df.columns:
                    na_mask[col] = bool(pd.isna(tab_df[col].iloc[0]))

        # Fill NaN in tabular numeric columns before normalization.
        # FillMissing uses median during training; we approximate with the scaler's
        # mean so that missing values become 0 after standardization.
        if num_cols and hasattr(tab_scaler, 'mean_'):
            for i, col in enumerate(num_cols):
                if col in tab_df.columns and pd.isna(tab_df[col].iloc[0]):
                    tab_df.loc[tab_df.index[0], col] = tab_scaler.mean_[i]

        # Normalize continuous TS + tabular using saved scalers
        ts_norm, tab_norm = normalize_new_patient(
            x_ts, tab_df, self.bundle
        )

        # Trajectory length (computed on raw data before normalization zeroes padding)
        traj_len = int(get_trajectory_lengths(x_ts)[0])

        # Convert to tensors
        x_ts_t = torch.from_numpy(ts_norm).float().to(self.device)

        x_ts_cat_t = torch.from_numpy(x_ts_cat).float().to(self.device)

        # Static categorical: encode via the same procs used in training
        # classes dict maps feature_name -> list of categories (index 0 = #na#)
        # FillMissing adds {col}_na boolean indicators for numeric columns with NaN
        classes = self.bundle['model_params']['classes']
        if classes:
            cat_indices = []
            for col in classes:
                class_list = list(classes[col])
                if col.endswith('_na') and col not in tab_df.columns:
                    # _na indicator: True if the original numeric column had NaN
                    orig_col = col[:-3]  # strip '_na'
                    is_na = na_mask.get(orig_col, False)
                    idx = class_list.index(is_na) if is_na in class_list else 0
                elif col in tab_df.columns:
                    val = tab_df[col].iloc[0]
                    idx = class_list.index(val) if val in class_list else 0
                else:
                    idx = 0  # unknown column → #na#
                cat_indices.append(idx)
            x_cat_t = torch.tensor([cat_indices], dtype=torch.long, device=self.device)
        else:
            x_cat_t = torch.zeros(1, 0, dtype=torch.long, device=self.device)

        # Static continuous: already normalized in tab_norm
        num_cols = self.bundle['tab_feature_names']
        if num_cols and len(num_cols) > 0:
            cont_vals = tab_norm[num_cols].values.astype(np.float32)
            x_cont_t = torch.from_numpy(cont_vals).to(self.device)
        else:
            x_cont_t = torch.zeros(1, 0, dtype=torch.float32, device=self.device)

        return x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x_ts, x_ts_cat, tab_df, censor_step=None, pid=None):
        """
        Run inference on a single patient.

        Args:
            x_ts: np.ndarray [n_channels, seq_len] — raw continuous TS
            x_ts_cat: np.ndarray [n_cat_dims, seq_len] — multi-hot categorical TS
            tab_df: pd.DataFrame with one row — static demographics
            censor_step: Optional timestep to evaluate at (temporal head only)
            pid: Optional patient identifier for the result

        Returns:
            InferenceResult
        """
        x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len = self._prepare_tensors(
            x_ts, x_ts_cat, tab_df
        )

        with torch.no_grad():
            logits = self.model((x_ts_t, (x_cat_t, x_cont_t), x_ts_cat_t))

        if self.is_temporal:
            # logits: [1, seq_len]
            probs_all = torch.sigmoid(logits).cpu().numpy()[0]  # [seq_len]
            step = min(censor_step, traj_len - 1) if censor_step is not None else traj_len - 1
            step = max(step, 0)
            probability = float(probs_all[step])
            return InferenceResult(
                pid=pid,
                probability=probability,
                trajectory_length=traj_len,
                censor_step=step,
                predictions_over_time=probs_all,
            )
        else:
            # logits: [1, 2]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probability = float(probs[1])  # class 1 = deceased
            return InferenceResult(
                pid=pid,
                probability=probability,
                trajectory_length=traj_len,
            )

    # ------------------------------------------------------------------
    # SHAP explanation
    # ------------------------------------------------------------------

    def explain(self, x_ts, x_ts_cat, tab_df, censor_step=None, pid=None):
        """
        Compute SHAP values for a single patient.

        Args:
            x_ts: np.ndarray [n_channels, seq_len] — raw continuous TS
            x_ts_cat: np.ndarray [n_cat_dims, seq_len] — multi-hot categorical TS
            tab_df: pd.DataFrame with one row — static demographics
            censor_step: Timestep to attribute (temporal head) or None (standard head, class 1)
            pid: Optional patient identifier

        Returns:
            SHAPResult
        """
        import shap
        from astra.evaluation.behavior import ModelWrapperWithRawCatTS, embed_categorical_features

        if self._bg is None:
            raise RuntimeError(
                "No SHAP background data in deployment bundle. "
                "Re-save the bundle with extract_shap_background()."
            )

        x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len = self._prepare_tensors(
            x_ts, x_ts_cat, tab_df
        )

        # Determine target step.
        # Temporal head: selects which output position to explain.
        # Non-temporal head: stored as eval_timestep for visualization cropping only
        #                    (the wrapper always returns class-1 logit regardless).
        target_step = None
        if self.is_temporal:
            step = censor_step if censor_step is not None else traj_len - 1
            target_step = min(max(step, 0), traj_len - 1)
        elif censor_step is not None:
            target_step = min(max(censor_step, 0), traj_len - 1)

        # Censor future data if requested
        if censor_step is not None and censor_step < x_ts_t.shape[2] - 1:
            x_ts_t = x_ts_t.clone()
            x_ts_cat_t = x_ts_cat_t.clone()
            x_ts_t[:, :, censor_step + 1:] = 0.0
            x_ts_cat_t[:, :, censor_step + 1:] = 0

            bg_ts = self._bg['ts'].clone()
            bg_ts_cat = self._bg['ts_cat'].clone()
            bg_ts[:, :, censor_step + 1:] = 0.0
            bg_ts_cat[:, :, censor_step + 1:] = 0
        else:
            bg_ts = self._bg['ts']
            bg_ts_cat = self._bg['ts_cat']

        # Wrapper with causal mask + temporal step targeting
        has_cat_ts = self.model.n_ts_cat > 0
        wrapped = ModelWrapperWithRawCatTS(self.model, has_cat_ts=has_cat_ts,
                                           eval_timestep=target_step if target_step is not None else -1)

        # Pre-embed static categoricals (not differentiable — treated as context)
        bg_cat_emb = embed_categorical_features(self.model, self._bg['cat'])
        sample_cat_emb = embed_categorical_features(self.model, x_cat_t)

        # Build input lists for GradientExplainer
        bg_inputs = [bg_ts, bg_ts_cat.float().requires_grad_(True)]
        sample_inputs = [x_ts_t, x_ts_cat_t.float().requires_grad_(True)]

        if bg_cat_emb is not None:
            bg_inputs.append(bg_cat_emb)
            sample_inputs.append(sample_cat_emb)
        if self._bg['cont'].shape[1] > 0:
            bg_inputs.append(self._bg['cont'])
            sample_inputs.append(x_cont_t)

        explainer = shap.GradientExplainer(wrapped, bg_inputs)
        shap_values = explainer.shap_values(sample_inputs)

        # Parse SHAP output — normalise to flat list [per_input][n_samples, ...]
        # GradientExplainer may return one of two formats depending on SHAP version:
        #   Format (a): [n_classes][n_inputs][n_samples, ...]  — outer list indexed by class
        #   Format (b): [n_inputs][n_samples, ..., n_classes]  — trailing class dim on ndarrays
        # For temporal head: model returns [batch, 1] → treated as single output (class_idx=0)
        # For standard head: model returns [batch, 2] → select class 1 (deceased)
        if isinstance(shap_values, list) and shap_values:
            class_idx = 0 if self.is_temporal else 1
            if isinstance(shap_values[0], list):
                # Format (a): select class
                shap_values = shap_values[class_idx]
            elif isinstance(shap_values[0], np.ndarray) and shap_values[0].ndim == 4:
                # Format (b): TS input [1, n_ch, seq_len, n_classes] → strip trailing class dim
                shap_values = [sv[..., class_idx] for sv in shap_values]

        # Unpack per-input SHAP values
        idx = 0
        ts_shap_raw = shap_values[idx][0]  # [n_channels, seq_len]
        idx += 1

        cat_ts_shap_raw = None
        if has_cat_ts:
            cat_ts_shap_raw = shap_values[idx][0]  # [n_cat_dims, seq_len]
            idx += 1

        cat_shap_raw = None
        if bg_cat_emb is not None:
            # [n_cat_features, d_model] — average over embedding dim
            cat_shap_raw = shap_values[idx][0].mean(axis=1)
            idx += 1

        cont_shap_raw = None
        if self._bg['cont'].shape[1] > 0:
            cont_shap_raw = shap_values[idx][0]  # [n_cont_features]
            idx += 1

        # Map to named features
        channel_names = self.bundle['ts_channel_names']
        ts_shap_dict = {
            name: ts_shap_raw[i] for i, name in enumerate(channel_names)
            if i < ts_shap_raw.shape[0]
        }

        cat_ts_shap_dict = None
        if cat_ts_shap_raw is not None:
            encoding_info = self.bundle['encoding_info']
            cat_labels = encoding_info.get('category_labels', {})
            cat_ts_shap_dict = {}
            for feat_name, (start, end) in encoding_info.get('feature_ranges', {}).items():
                labels = cat_labels.get(feat_name, [f'{feat_name}_{i}' for i in range(end - start)])
                for j, label in enumerate(labels):
                    if start + j < cat_ts_shap_raw.shape[0]:
                        cat_ts_shap_dict[label] = cat_ts_shap_raw[start + j]

        static_cat_dict = None
        if cat_shap_raw is not None:
            classes = self.bundle['model_params']['classes']
            static_cat_dict = {
                name: float(cat_shap_raw[i])
                for i, name in enumerate(classes.keys())
                if i < len(cat_shap_raw)
            }

        static_cont_dict = None
        if cont_shap_raw is not None:
            num_cols = self.bundle['tab_feature_names']
            static_cont_dict = {
                name: float(cont_shap_raw[i])
                for i, name in enumerate(num_cols)
                if i < len(cont_shap_raw)
            }

        # Top features by absolute importance (across all types)
        all_importances = []
        for name, arr in ts_shap_dict.items():
            all_importances.append((name, float(np.abs(arr).mean())))
        if static_cont_dict:
            for name, val in static_cont_dict.items():
                all_importances.append((name, abs(val)))
        if static_cat_dict:
            for name, val in static_cat_dict.items():
                all_importances.append((name, abs(val)))
        all_importances.sort(key=lambda x: x[1], reverse=True)

        return SHAPResult(
            pid=pid,
            ts_shap=ts_shap_dict,
            cat_ts_shap=cat_ts_shap_dict,
            static_cat_shap=static_cat_dict,
            static_cont_shap=static_cont_dict,
            top_features=all_importances[:20],
            eval_timestep=target_step,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict_and_explain(self, x_ts, x_ts_cat, tab_df, censor_step=None, pid=None):
        """Run both prediction and SHAP explanation."""
        pred = self.predict(x_ts, x_ts_cat, tab_df, censor_step, pid)
        shap_result = self.explain(x_ts, x_ts_cat, tab_df, censor_step, pid)
        return pred, shap_result

    def shap_to_viz_dict(self, shap_result, x_ts, x_ts_cat, tab_df):
        """
        Convert a SHAPResult into the dict format expected by
        visualize_shap_individual() from astra.evaluation.behavior.

        Args:
            shap_result: SHAPResult from self.explain().
            x_ts: np.ndarray [n_channels, seq_len] — raw continuous TS
                  (same array passed to explain()).
            x_ts_cat: np.ndarray [n_cat_dims, seq_len] — multi-hot categorical TS.
            tab_df: pd.DataFrame with one row — static demographics.

        Returns:
            (shap_dict, channel2feature, feature_names_cat, feature_names_cont)
            suitable for visualize_shap_individual(shap_dict, sample_idx=0,
                channel2feature=channel2feature, ...).
        """
        channel_names = self.bundle['ts_channel_names']

        # --- ts_shap: [1, n_channels, seq_len] ---
        seq_len = x_ts.shape[-1]
        ts_shap = np.stack(
            [np.asarray(shap_result.ts_shap.get(ch, np.zeros(seq_len))).squeeze()
             for ch in channel_names]
        )[np.newaxis, ...]

        # --- channel2feature mapping ---
        channel2feature = {i: name for i, name in enumerate(channel_names)}

        # --- Categorical TS ---
        encoding_info = self.bundle.get('encoding_info')
        cat_ts_shap = None
        cat_ts_shap_per_category = None
        if shap_result.cat_ts_shap and encoding_info:
            # Rebuild per-category array in feature_ranges order
            all_labels = []
            for feat, (start, end) in encoding_info.get('feature_ranges', {}).items():
                labels = encoding_info.get('category_labels', {}).get(
                    feat, [f'{feat}_{i}' for i in range(end - start)]
                )
                all_labels.extend(labels)

            seq_len = x_ts_cat.shape[-1]
            n_cats = len(all_labels)
            per_cat = np.zeros((n_cats, seq_len))
            for i, label in enumerate(all_labels):
                if label in shap_result.cat_ts_shap:
                    val = np.asarray(shap_result.cat_ts_shap[label]).squeeze()
                    per_cat[i] = val

            cat_ts_shap_per_category = per_cat[np.newaxis, ...]  # [1, n_cats, seq_len]
            cat_ts_shap = np.abs(per_cat).mean(axis=0)[np.newaxis, ...]  # [1, seq_len]

        # --- Static categorical ---
        classes = self.bundle['model_params'].get('classes', {})
        cat_shap = None
        cat_indices = None
        feature_names_cat = list(classes.keys()) if classes else []
        if shap_result.static_cat_shap and classes:
            cat_vals = [shap_result.static_cat_shap.get(c, 0.0) for c in classes]
            cat_shap = np.array(cat_vals)[np.newaxis, ...]  # [1, n_features]

            # Reconstruct raw cat index for test_data display
            cat_idx_list = []
            for col in classes:
                class_list = list(classes[col])
                if col in tab_df.columns:
                    val = tab_df[col].iloc[0]
                    idx = class_list.index(val) if val in class_list else 0
                else:
                    idx = 0
                cat_idx_list.append(idx)
            cat_indices = np.array(cat_idx_list)[np.newaxis, ...]

        # --- Static continuous ---
        num_cols = self.bundle.get('tab_feature_names', [])
        cont_shap = None
        cont_vals = None
        feature_names_cont = list(num_cols) if num_cols else []
        if shap_result.static_cont_shap and num_cols:
            cont_vals_list = [
                shap_result.static_cont_shap.get(c, 0.0) for c in num_cols
            ]
            cont_shap = np.array(cont_vals_list)[np.newaxis, ...]
            cont_vals = tab_df[num_cols].values.astype(np.float32) if all(
                c in tab_df.columns for c in num_cols
            ) else np.full((1, len(num_cols)), np.nan)

        # --- Build the dict ---
        if x_ts.ndim == 2:
            x_ts = x_ts[np.newaxis, ...]
        if x_ts_cat.ndim == 2:
            x_ts_cat = x_ts_cat[np.newaxis, ...]

        shap_dict = {
            'ts_shap': ts_shap,
            'cat_ts_shap': cat_ts_shap,
            'cat_ts_shap_per_category': cat_ts_shap_per_category,
            'cat_ts_shap_embedded': None,
            'cat_shap': cat_shap,
            'cat_shap_embedded': None,
            'cont_shap': cont_shap,
            'n_static_cat': len(classes),
            'eval_timestep': shap_result.eval_timestep,
            'test_data': {
                'ts': x_ts,
                'ts_cat': x_ts_cat,
                'cat': cat_indices if cat_indices is not None else np.zeros((1, 0)),
                'cont': cont_vals if cont_vals is not None else np.zeros((1, 0)),
                'y': np.array([0]),
            },
            'encoding_info': encoding_info,
        }

        return shap_dict, channel2feature, feature_names_cat, feature_names_cont

    # ------------------------------------------------------------------
    # PatientContext-based inference
    # ------------------------------------------------------------------
    
    def create_patient_context(self, raw_data):
        """Create a :class:`PatientContext` for repeated inference.

        Args:
            raw_data: Dict with patient data (same schema as
                ``prepare_single_patient``).

        Returns:
            :class:`~astra.inference.patient_context.PatientContext`
        """
        from astra.inference.patient_context import PatientContext
        return PatientContext.create(raw_data, self.bundle)

    def predict_from_context(self, context, censor_step=None):
        """Run prediction using an existing PatientContext.

        Args:
            context: A :class:`PatientContext` instance.
            censor_step: Override timestep to evaluate at (temporal head).
                Defaults to ``context.trajectory_length - 1``.

        Returns:
            :class:`InferenceResult`
        """
        step = censor_step if censor_step is not None else context.trajectory_length - 1
        return self.predict(
            context.x_ts,
            context.x_ts_cat,
            context.tab_df,
            censor_step=step,
            pid=context.pid,
        )

    def refresh_and_predict(self, context, current_time, new_data=None):
        """Update a PatientContext with new data and run prediction.

        Args:
            context: A :class:`PatientContext` instance (modified in place).
            current_time: New time horizon.
            new_data: Optional dict with new measurements to append.

        Returns:
            :class:`InferenceResult`
        """
        context.refresh(current_time, new_data)
        return self.predict_from_context(context)

    def explain_from_context(self, context, censor_step=None):
        """Compute SHAP explanation using an existing PatientContext.

        Args:
            context: A :class:`PatientContext` instance.
            censor_step: Override timestep to attribute (temporal head).

        Returns:
            :class:`SHAPResult`
        """
        step = censor_step if censor_step is not None else context.trajectory_length - 1
        return self.explain(
            context.x_ts,
            context.x_ts_cat,
            context.tab_df,
            censor_step=step,
            pid=context.pid,
        )


# ============================================================================
# HELPER: Extract a patient from existing data dict (for testing)
# ============================================================================

def extract_patient_from_data(data, pid):
    """
    Pull a single patient's raw (unnormalized) tensors from the holdout data,
    ready for use with InferenceSession.predict().

    Args:
        data: dict from prepare_data_and_dls() or cache
        pid: Patient ID to extract

    Returns:
        dict with keys: x_ts, x_ts_cat, tab_df, y, sample_idx
    """
    holdout = data["holdout"]
    pids = holdout.tab_df['PID'].tolist()
    if pid not in pids:
        raise ValueError(f"PID {pid} not found in holdout. Available: {pids[:10]}...")

    sample_idx = pids.index(pid)

    # Raw (unnormalized) continuous TS
    x_ts_raw = data["tX_raw"][sample_idx]  # [n_channels, seq_len]

    # Multi-hot categorical TS
    x_ts_cat = data["tX_multi_hot"][sample_idx]  # [n_cat_dims, seq_len]

    # Static tabular (single row)
    tab_df = holdout.tab_df.iloc[[sample_idx]].copy()

    # Target
    y = data["ty"][sample_idx]

    return {
        'x_ts': x_ts_raw,
        'x_ts_cat': x_ts_cat,
        'tab_df': tab_df,
        'y': y,
        'sample_idx': sample_idx,
        'pid': pid,
    }
