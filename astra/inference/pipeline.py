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

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import random

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

from astra.utils import cfg
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
    probability: float                              # P(deceased) or cumulative risk
    trajectory_length: int = 0                      # Actual data timesteps
    censor_step: Optional[int] = None               # Timestep evaluated at
    predictions_over_time: Optional[np.ndarray] = None  # [seq_len] (temporal only)
    survival_curve: Optional[np.ndarray] = None     # [seq_len] S(t) (survival mode only)
    label: Optional[int] = None                     # Predicted label (thresholded)
    uncertainty: Optional[float] = None             # MC Dropout std (Fix 8)


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


@dataclass
class DifferentialSHAPResult:
    """Differential SHAP between two timepoints: ΔSHAP = SHAP(T2) - SHAP(T1)."""
    pid: Any
    t1_hours: float
    t2_hours: float
    t1_step: int
    t2_step: int
    t1_probability: float
    t2_probability: float
    delta_ts_shap: Dict[str, np.ndarray]                       # {channel: [seq_len]}
    delta_cat_ts_shap: Optional[Dict[str, np.ndarray]] = None  # {category: [seq_len]}
    delta_static_cat_shap: Optional[Dict[str, float]] = None
    delta_static_cont_shap: Optional[Dict[str, float]] = None
    top_delta_features: List[Tuple[str, float]] = field(default_factory=list)
    shap_t1: Optional[SHAPResult] = None
    shap_t2: Optional[SHAPResult] = None


# SHAPModelWrapper and embed_categorical_features live in
# astra.evaluation.behavior. Imported lazily inside explain() to keep
# pipeline.py dependency-light.


# ============================================================================
# INFERENCE SESSION
# ============================================================================

class InferenceSession:
    """
    Loads a trained model and all deployment artifacts once.
    Supports repeated single-patient inference and SHAP explanation.
    """

    def __init__(self, model, bundle, device='cpu',
                 calibrators=None, calibration_method=None):
        self.model = model
        self.bundle = bundle
        self.device = device
        self.is_temporal = bundle['model_params']['temporal_head']

        self._calibrators = calibrators
        self._global_calibrator = None
        self._calibration_method = calibration_method
        if calibrators is not None:
            self._global_calibrator = calibrators.pop('global', None)

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

        logger.info("Loading model '%s' on %s", model_name, device)
        bundle = load_deployment_bundle(model_name, bundle_dir)
        bundle['model_name'] = model_name
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
            temporal_head_mult=params.get('temporal_head_mult', 0.5),
            temporal_channel_idx=params.get('temporal_channel_idx', None),
            exclude_channel_indices=params.get('exclude_channel_indices', []),
            head_pool=params.get('head_pool', 'flatten'),
            per_feature_cont_proj=params.get('per_feature_cont_proj', False),
            cat_ts_gate=params.get('cat_ts_gate', False),
            local_temporal_kernel=params.get('local_temporal_kernel', 1),
            bin_width_channel_idx=params.get('bin_width_channel_idx', None),
            bin_width_modulation=params.get('bin_width_modulation', False),
            ts_cat_profile_dims=params.get('ts_cat_profile_dims', None),
        )

        # Load weights (FastAI format: {'model': state_dict, ...})
        weights_path = os.path.join(weights_dir, f'{model_name}.pth')
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded: %d parameters, temporal_head=%s",
                     n_params, params['temporal_head'])
        logger.debug("Model params: c_in=%d seq_len=%d d_model=%d n_layers=%d",
                      params['c_in'], params['seq_len'],
                      params['d_model'], params['n_layers'])

        # Load posthoc calibrators if available
        calibrators = None
        calibration_method = None
        calibrator_dir = os.path.join(weights_dir, 'calibrators', model_name)
        if os.path.isdir(calibrator_dir):
            meta_path = os.path.join(calibrator_dir, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                calibration_method = meta.get('best_method', 'isotonic')
                calibrators = {}
                for step in meta.get('timepoints', []):
                    pkl = os.path.join(calibrator_dir, f"{calibration_method}_step{step}.pkl")
                    if os.path.exists(pkl):
                        with open(pkl, 'rb') as fh:
                            calibrators[step] = pickle.load(fh)
                global_pkl = os.path.join(calibrator_dir, f"{calibration_method}_global.pkl")
                if os.path.exists(global_pkl):
                    with open(global_pkl, 'rb') as fh:
                        calibrators['global'] = pickle.load(fh)
                n_tp = sum(1 for k in calibrators if k != 'global')
                logger.info("Loaded %s calibrators (%d per-timepoint, global=%s)",
                            calibration_method, n_tp, 'global' in calibrators)

        return cls(model, bundle, device,
                   calibrators=calibrators,
                   calibration_method=calibration_method)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_tensors(self, x_ts, x_ts_cat, tab_df, trajectory_length=None,
                         profiling=None):
        """
        Normalize raw patient data and convert to model-ready tensors.

        Args:
            x_ts: np.ndarray [n_channels, seq_len] or [1, n_channels, seq_len]
                  Raw (unnormalized) continuous time series.
            x_ts_cat: np.ndarray [n_cat_dims, seq_len] or [1, n_cat_dims, seq_len]
                      Multi-hot encoded categorical time series.
            tab_df: pd.DataFrame with one row containing static features.
            trajectory_length: Optional override for trajectory length. When
                provided (e.g. from PatientContext), used instead of detecting
                from the tensor via get_trajectory_lengths().
            profiling: Optional dict to collect sub-stage timing (for perf analysis).

        Returns:
            (x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len) — all tensors on device
        """
        from contextlib import nullcontext as _nullctx
        from astra.inference.data_prep import timed_stage

        def _ts(name):
            return timed_stage(profiling, name) if profiling is not None else _nullctx()

        if x_ts.ndim == 2:
            x_ts = x_ts[np.newaxis, ...]
        if x_ts_cat.ndim == 2:
            x_ts_cat = x_ts_cat[np.newaxis, ...]

        with _ts('prep_tab_fill'):
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
            fill_values = None
            if num_cols:
                if hasattr(tab_scaler, 'mean_'):
                    fill_values = tab_scaler.mean_
                elif hasattr(tab_scaler, 'center_'):
                    fill_values = tab_scaler.center_
            if fill_values is not None:
                for i, col in enumerate(num_cols):
                    if col in tab_df.columns and pd.isna(tab_df[col].iloc[0]):
                        tab_df.loc[tab_df.index[0], col] = fill_values[i]

        # Exclude EBM channel from trajectory length computation so forward-filled
        # predictions cannot extend the trajectory beyond clinical measurements.
        ebm_ch_idx = self.bundle.get('data_config', {}).get('ebm_channel_idx')
        traj_exclude_chs = [ebm_ch_idx] if ebm_ch_idx is not None else None

        # Normalize continuous TS + tabular using saved scalers
        with _ts('prep_normalize'):
            ts_norm, tab_norm = normalize_new_patient(
                x_ts, tab_df, self.bundle, exclude_channels=traj_exclude_chs
            )

        # Restore raw elapsed_hours — sinusoidal PE requires actual hours (0–720),
        # not the ~N(0,1) values produced by the TS scaler.
        temporal_ch_idx = self.bundle['model_params'].get('temporal_channel_idx', None)
        if temporal_ch_idx is not None:
            ts_norm[0, temporal_ch_idx, :] = x_ts[0, temporal_ch_idx, :]

        # Trajectory length: use explicit value if provided, else detect from tensor
        if trajectory_length is not None:
            traj_len = int(trajectory_length)
        else:
            traj_len = int(get_trajectory_lengths(x_ts, exclude_channels=traj_exclude_chs)[0])

        # Convert to tensors
        with _ts('prep_to_torch'):
            x_ts_t = torch.from_numpy(ts_norm).float().to(self.device)

            x_ts_cat_t = torch.from_numpy(x_ts_cat).float().to(self.device)

        # Static categorical: encode via the same procs used in training
        with _ts('prep_cat_encode'):
            classes = self.bundle['model_params']['classes']
            if classes:
                cat_indices = []
                for col in classes:
                    class_list = list(classes[col])
                    if col.endswith('_na') and col not in tab_df.columns:
                        orig_col = col[:-3]
                        is_na = na_mask.get(orig_col, False)
                        idx = class_list.index(is_na) if is_na in class_list else 0
                    elif col in tab_df.columns:
                        val = tab_df[col].iloc[0]
                        idx = class_list.index(val) if val in class_list else 0
                    else:
                        idx = 0
                    cat_indices.append(idx)
                x_cat_t = torch.tensor([cat_indices], dtype=torch.long, device=self.device)
            else:
                x_cat_t = torch.zeros(1, 0, dtype=torch.long, device=self.device)

        # Static continuous: already normalized in tab_norm
        with _ts('prep_cont_encode'):
            num_cols = self.bundle['tab_feature_names']
            if num_cols and len(num_cols) > 0:
                cont_vals = tab_norm[num_cols].values.astype(np.float32)
                x_cont_t = torch.from_numpy(cont_vals).to(self.device)
            else:
                x_cont_t = torch.zeros(1, 0, dtype=torch.float32, device=self.device)

        logger.debug("Tensors prepared: ts=%s cat=%s cont=%s ts_cat=%s traj_len=%d",
                     x_ts_t.shape, x_cat_t.shape, x_cont_t.shape,
                     x_ts_cat_t.shape, traj_len)
        return x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate(self, prob, step=None):
        """Apply posthoc calibration to a probability or array of probabilities.

        Supports three calibrator types (checked in priority order):
        1. ``TemporalCalibrator`` — per-window calibration, preferred for
           temporal models.
        2. Per-step dict of sklearn calibrators (legacy).
        3. Global sklearn calibrator fallback.
        """
        from astra.evaluation.posthoc_calibration import TemporalCalibrator

        is_scalar = isinstance(prob, (float, np.floating))
        arr = np.atleast_1d(np.asarray(prob, dtype=np.float64))

        # Prefer TemporalCalibrator if available
        if isinstance(self._calibrators, TemporalCalibrator):
            if step is not None:
                calibrated = self._calibrators.transform_at_step(arr, step)
            else:
                calibrated = arr
            return float(calibrated[0]) if is_scalar else calibrated

        if self._calibrators is None and self._global_calibrator is None:
            return prob
        from astra.evaluation.posthoc_calibration import apply_calibrator

        cal = None
        if step is not None and self._calibrators and step in self._calibrators:
            cal = self._calibrators[step]
        elif self._global_calibrator is not None:
            cal = self._global_calibrator

        if cal is None:
            return prob

        calibrated = apply_calibrator(cal, arr, self._calibration_method)
        return float(calibrated[0]) if is_scalar else calibrated

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x_ts, x_ts_cat, tab_df, censor_step=None, pid=None,
                trajectory_length=None, profiling=None):
        """
        Run inference on a single patient.

        Args:
            x_ts: np.ndarray [n_channels, seq_len] — raw continuous TS
            x_ts_cat: np.ndarray [n_cat_dims, seq_len] — multi-hot categorical TS
            tab_df: pd.DataFrame with one row — static demographics
            censor_step: Optional timestep to evaluate at (temporal head only)
            pid: Optional patient identifier for the result
            trajectory_length: Optional override for trajectory length (from PatientContext).
            profiling: Optional dict to collect sub-stage timing (for perf analysis).

        Returns:
            InferenceResult
        """
        from contextlib import nullcontext as _nullctx
        from astra.inference.data_prep import timed_stage

        def _ts(name):
            return timed_stage(profiling, name) if profiling is not None else _nullctx()

        with _ts('prep_tensors'):
            x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len = self._prepare_tensors(
                x_ts, x_ts_cat, tab_df, trajectory_length=trajectory_length,
                profiling=profiling,
            )

        # Pass trajectory lengths so the model builds a proper key_padding_mask
        # (matching training behavior from AstraMixedDataset)
        traj_lengths_t = torch.tensor([traj_len], dtype=torch.long, device=self.device)

        with _ts('model_forward'):
            with torch.no_grad():
                logits = self.model((x_ts_t, (x_cat_t, x_cont_t), x_ts_cat_t, traj_lengths_t))

        if self.is_temporal:
            # logits: [1, seq_len]
            logger.debug("Temporal logits shape: %s", logits.shape)
            seq_len = x_ts_t.shape[2]
            if censor_step is not None:
                step = min(censor_step, seq_len - 1)
            else:
                step = traj_len - 1
            step = max(step, 0)

            survival_mode = self.bundle.get('model_params', {}).get('survival_mode', False)
            if survival_mode:
                # Discrete-time survival: compute S(t) from hazard logits
                from astra.training.utils import hazards_to_survival
                surv_probs = hazards_to_survival(logits).cpu().numpy()[0]  # [seq_len]
                cum_incidence = 1.0 - surv_probs  # risk at each timestep
                probability = float(cum_incidence[step])
                logger.info("Prediction (survival): pid=%s risk=%.4f S(t)=%.4f step=%d traj_len=%d",
                            pid, probability, surv_probs[step], step, traj_len)
                return InferenceResult(
                    pid=pid,
                    probability=probability,
                    trajectory_length=traj_len,
                    censor_step=step,
                    predictions_over_time=cum_incidence,
                    survival_curve=surv_probs,
                )
            else:
                probs_all = torch.sigmoid(logits).cpu().numpy()[0]  # [seq_len]
                probs_all = self._calibrate(probs_all)
                probability = float(probs_all[step])
                logger.info("Prediction: pid=%s P(deceased)=%.4f step=%d traj_len=%d",
                            pid, probability, step, traj_len)
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
            probability = self._calibrate(float(probs[1]))
            logger.info("Prediction: pid=%s P(deceased)=%.4f traj_len=%d",
                        pid, probability, traj_len)
            return InferenceResult(
                pid=pid,
                probability=probability,
                trajectory_length=traj_len,
            )

    # ------------------------------------------------------------------
    # SHAP explanation
    # ------------------------------------------------------------------

    def explain(self, x_ts, x_ts_cat, tab_df, censor_step=None, pid=None,
                trajectory_length=None):
        """
        Compute SHAP values for a single patient.

        Args:
            x_ts: np.ndarray [n_channels, seq_len] — raw continuous TS
            x_ts_cat: np.ndarray [n_cat_dims, seq_len] — multi-hot categorical TS
            tab_df: pd.DataFrame with one row — static demographics
            censor_step: Timestep to attribute (temporal head) or None (standard head, class 1)
            pid: Optional patient identifier
            trajectory_length: Optional override for trajectory length (from PatientContext).

        Returns:
            SHAPResult
        """
        import shap
        from astra.evaluation.behavior import SHAPModelWrapper, embed_categorical_features

        logger.info("Computing SHAP explanation for pid=%s (censor_step=%s)", pid, censor_step)

        if self._bg is None:
            raise RuntimeError(
                "No SHAP background data in deployment bundle. "
                "Re-save the bundle with extract_shap_background()."
            )

        x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len = self._prepare_tensors(
            x_ts, x_ts_cat, tab_df, trajectory_length=trajectory_length
        )

        # Determine target step.
        # Temporal head: selects which output position to explain.
        # Non-temporal head: stored as eval_timestep for visualization cropping only
        #                    (the wrapper always returns class-1 logit regardless).
        seq_len = x_ts_t.shape[2]
        target_step = None
        if self.is_temporal:
            step = censor_step if censor_step is not None else traj_len - 1
            # Trust the caller's step (from time_to_step); only cap at seq_len
            target_step = min(max(step, 0), seq_len - 1)
        elif censor_step is not None:
            target_step = min(max(censor_step, 0), seq_len - 1)

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
        # Clamp traj_length to censor_step+1 so padding mask excludes censored positions
        has_cat_ts = self.model.n_ts_cat > 0
        effective_traj = min(traj_len, censor_step + 1) if censor_step is not None else traj_len
        traj_lengths_t = torch.tensor([effective_traj], dtype=torch.long, device=self.device)
        survival_mode = self.bundle.get('model_params', {}).get('survival_mode', False)

        bg_cat_onehot = embed_categorical_features(self.model, self._bg['cat'])
        sample_cat_onehot = embed_categorical_features(self.model, x_cat_t)
        has_static_cat = bg_cat_onehot is not None
        has_cont = self._bg['cont'].shape[1] > 0

        wrapped = SHAPModelWrapper(
            self.model, has_cat_ts=has_cat_ts,
            has_static_cat=has_static_cat, has_cont=has_cont,
            eval_timestep=target_step if target_step is not None else -1,
            traj_lengths=traj_lengths_t,
            survival_mode=survival_mode,
        )

        bg_inputs = [bg_ts]
        sample_inputs = [x_ts_t]
        if has_cat_ts:
            bg_inputs.append(bg_ts_cat.float().requires_grad_(True))
            sample_inputs.append(x_ts_cat_t.float().requires_grad_(True))
        if has_static_cat:
            bg_inputs.append(bg_cat_onehot)
            sample_inputs.append(sample_cat_onehot)
        if has_cont:
            bg_inputs.append(self._bg['cont'])
            sample_inputs.append(x_cont_t)

        shap_seed = cfg.get("evaluation", {}).get("shap_seed", 42)
        shap_nsamples = cfg.get("evaluation", {}).get("shap_nsamples", 200)
        explainer = shap.GradientExplainer(wrapped, bg_inputs)
        if shap_seed is not None:
            random.seed(shap_seed)
            np.random.seed(shap_seed)
            torch.manual_seed(shap_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(shap_seed)
        shap_values = explainer.shap_values(sample_inputs, nsamples=shap_nsamples)

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
        if has_static_cat:
            cat_shap_raw = np.abs(shap_values[idx][0]).sum(axis=1)
            idx += 1

        cont_shap_raw = None
        if has_cont:
            cont_shap_raw = shap_values[idx][0]  # [n_cont_features]
            idx += 1

        # Zero SHAP values beyond effective trajectory (padding + censored positions)
        if effective_traj < ts_shap_raw.shape[-1]:
            ts_shap_raw[:, effective_traj:] = 0.0
            if cat_ts_shap_raw is not None:
                cat_ts_shap_raw[:, effective_traj:] = 0.0

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

        logger.info("SHAP done for pid=%s: %d TS channels, %d top features",
                    pid, len(ts_shap_dict), min(len(all_importances), 20))
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
    # Uncertainty estimation (Fix 8: MC Dropout)
    # ------------------------------------------------------------------

    def predict_with_uncertainty(self, x_ts, x_ts_cat, tab_df,
                                  censor_step=None, pid=None,
                                  trajectory_length=None, n_samples=30):
        """
        MC Dropout prediction: run N stochastic forward passes to estimate
        prediction uncertainty. Returns mean probability and standard deviation.

        Args:
            x_ts, x_ts_cat, tab_df: Same as predict()
            censor_step: Optional timestep for temporal head
            pid: Optional patient ID
            trajectory_length: Optional trajectory length override
            n_samples: Number of MC dropout forward passes

        Returns:
            InferenceResult with uncertainty field populated
        """
        import torch.nn as nn

        x_ts_t, x_cat_t, x_cont_t, x_ts_cat_t, traj_len = self._prepare_tensors(
            x_ts, x_ts_cat, tab_df, trajectory_length=trajectory_length,
        )
        traj_lengths_t = torch.tensor([traj_len], dtype=torch.long, device=self.device)
        inputs = (x_ts_t, (x_cat_t, x_cont_t), x_ts_cat_t, traj_lengths_t)

        # Enable dropout, freeze normalization layers
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(inputs)
                preds.append(torch.sigmoid(logits))

        self.model.eval()  # Restore

        preds = torch.stack(preds)  # [n_samples, 1, ...]
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)

        if self.is_temporal and censor_step is not None:
            target_step = min(censor_step, mean_pred.shape[-1] - 1)
            prob = mean_pred[0, target_step].item()
            unc = std_pred[0, target_step].item()
        else:
            prob = mean_pred[0, 1].item() if mean_pred.dim() > 1 and mean_pred.shape[-1] == 2 else mean_pred[0].item()
            unc = std_pred[0, 1].item() if std_pred.dim() > 1 and std_pred.shape[-1] == 2 else std_pred[0].item()

        return InferenceResult(
            probability=prob,
            label=int(prob >= 0.5),
            pid=pid,
            uncertainty=unc,
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

        # Compute trajectory_length from the zero-padded x_ts so that
        # visualize_data_completeness can use it as the authoritative bound
        # rather than relying on NaN heuristics (which fail for zero-padded data).
        traj_len_ts = int(get_trajectory_lengths(x_ts)[0])

        shap_dict = {
            'ts_shap': ts_shap,
            'cat_ts_shap': cat_ts_shap,
            'cat_ts_shap_per_category': cat_ts_shap_per_category,
            'cat_shap': cat_shap,
            'cont_shap': cont_shap,
            'n_static_cat': len(classes),
            'eval_timestep': shap_result.eval_timestep,
            'trajectory_length': traj_len_ts,
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

    def predict_from_context(self, context, censor_step=None, profiling=None):
        """Run prediction using an existing PatientContext.

        Args:
            context: A :class:`PatientContext` instance.
            censor_step: Override timestep to evaluate at (temporal head).
                Defaults to ``context.trajectory_length - 1``.
            profiling: Optional dict to collect sub-stage timing (for perf analysis).

        Returns:
            :class:`InferenceResult`
        """
        step = censor_step if censor_step is not None else context.trajectory_length - 1
        traj_len = min(step + 1, context.trajectory_length) if censor_step is not None else context.trajectory_length
        return self.predict(
            context.x_ts,
            context.x_ts_cat,
            context.tab_df,
            censor_step=step,
            pid=context.pid,
            trajectory_length=traj_len,
            profiling=profiling,
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
        traj_len = min(step + 1, context.trajectory_length) if censor_step is not None else context.trajectory_length
        return self.explain(
            context.x_ts,
            context.x_ts_cat,
            context.tab_df,
            censor_step=step,
            pid=context.pid,
            trajectory_length=traj_len,
        )

    def explain_differential(self, context, t1_hours, t2_hours):
        """Compute differential SHAP between two timepoints.

        Uses the same patient context with different censor_step values
        to get SHAP at T1 and T2, then computes ΔSHAP = SHAP(T2) - SHAP(T1).

        Args:
            context: A :class:`PatientContext` advanced to at least T2.
            t1_hours: Earlier timepoint (hours after admission).
            t2_hours: Later timepoint (hours after admission).

        Returns:
            :class:`DifferentialSHAPResult`
        """
        from astra.evaluation.utils import time_to_step

        # Ensure t1 < t2
        if t1_hours > t2_hours:
            t1_hours, t2_hours = t2_hours, t1_hours

        # Use the bundle's bin config — the local YAML may have diverged
        # from the grid this model was trained on.
        data_config = self.bundle.get('data_config')
        step_t1 = time_to_step(t1_hours, 'h', data_config=data_config)
        step_t2 = time_to_step(t2_hours, 'h', data_config=data_config)
        # time_to_step extends into the open-ended 'end' interval, so an
        # hours value beyond the grid can map past the model's last step.
        max_step = self.bundle['model_params']['seq_len'] - 1
        step_t1 = min(step_t1, max_step)
        step_t2 = min(step_t2, max_step)

        logger.info("Differential SHAP: pid=%s T1=%.1fh (step %d) → T2=%.1fh (step %d)",
                     context.pid, t1_hours, step_t1, t2_hours, step_t2)

        # SHAP at both timepoints
        shap_t1 = self.explain_from_context(context, censor_step=step_t1)
        shap_t2 = self.explain_from_context(context, censor_step=step_t2)

        # Predictions at both timepoints
        pred_t1 = self.predict_from_context(context, censor_step=step_t1)
        pred_t2 = self.predict_from_context(context, censor_step=step_t2)

        # Delta for time-series channels
        all_channels = set(shap_t1.ts_shap) | set(shap_t2.ts_shap)
        seq_len = next(iter(shap_t2.ts_shap.values())).shape[0]
        delta_ts = {}
        for ch in all_channels:
            v1 = shap_t1.ts_shap.get(ch, np.zeros(seq_len))
            v2 = shap_t2.ts_shap.get(ch, np.zeros(seq_len))
            delta_ts[ch] = np.asarray(v2) - np.asarray(v1)

        # Delta for categorical TS
        delta_cat_ts = None
        if shap_t1.cat_ts_shap or shap_t2.cat_ts_shap:
            cats_t1 = shap_t1.cat_ts_shap or {}
            cats_t2 = shap_t2.cat_ts_shap or {}
            all_cats = set(cats_t1) | set(cats_t2)
            delta_cat_ts = {}
            for cat in all_cats:
                v1 = cats_t1.get(cat, np.zeros(seq_len))
                v2 = cats_t2.get(cat, np.zeros(seq_len))
                delta_cat_ts[cat] = np.asarray(v2) - np.asarray(v1)

        # Delta for static features
        delta_static_cat = None
        if shap_t1.static_cat_shap or shap_t2.static_cat_shap:
            sc1 = shap_t1.static_cat_shap or {}
            sc2 = shap_t2.static_cat_shap or {}
            delta_static_cat = {k: sc2.get(k, 0.0) - sc1.get(k, 0.0)
                                for k in set(sc1) | set(sc2)}

        delta_static_cont = None
        if shap_t1.static_cont_shap or shap_t2.static_cont_shap:
            sn1 = shap_t1.static_cont_shap or {}
            sn2 = shap_t2.static_cont_shap or {}
            delta_static_cont = {k: sn2.get(k, 0.0) - sn1.get(k, 0.0)
                                 for k in set(sn1) | set(sn2)}

        # Top features by absolute delta
        all_importances = []
        for name, arr in delta_ts.items():
            all_importances.append((name, float(np.abs(arr).mean())))
        if delta_static_cont:
            for name, val in delta_static_cont.items():
                all_importances.append((name, abs(val)))
        if delta_static_cat:
            for name, val in delta_static_cat.items():
                all_importances.append((name, abs(val)))
        all_importances.sort(key=lambda x: x[1], reverse=True)

        return DifferentialSHAPResult(
            pid=context.pid,
            t1_hours=t1_hours,
            t2_hours=t2_hours,
            t1_step=step_t1,
            t2_step=step_t2,
            t1_probability=pred_t1.probability,
            t2_probability=pred_t2.probability,
            delta_ts_shap=delta_ts,
            delta_cat_ts_shap=delta_cat_ts,
            delta_static_cat_shap=delta_static_cat,
            delta_static_cont_shap=delta_static_cont,
            top_delta_features=all_importances[:20],
            shap_t1=shap_t1,
            shap_t2=shap_t2,
        )

    def differential_shap_to_viz_dict(self, diff_result, x_ts, x_ts_cat, tab_df):
        """Convert a DifferentialSHAPResult into the viz dict format.

        Produces the same structure as :meth:`shap_to_viz_dict` but populated
        with delta values (SHAP(T2) - SHAP(T1)).  Existing plot functions
        (heatmaps, bar charts) work directly on this dict.

        Returns:
            (shap_dict, channel2feature, feature_names_cat, feature_names_cont)
        """
        # Build a temporary SHAPResult with delta values so we can delegate
        # to the existing shap_to_viz_dict.
        delta_shap = SHAPResult(
            pid=diff_result.pid,
            ts_shap=diff_result.delta_ts_shap,
            cat_ts_shap=diff_result.delta_cat_ts_shap,
            static_cat_shap=diff_result.delta_static_cat_shap,
            static_cont_shap=diff_result.delta_static_cont_shap,
            top_features=diff_result.top_delta_features,
            eval_timestep=diff_result.t2_step,
        )
        return self.shap_to_viz_dict(delta_shap, x_ts, x_ts_cat, tab_df)

    def explain_ebm(self, context, save_path=None, top_n=20, top_k_lines=5):
        """Compute and optionally visualize per-patient EBM feature importance.

        Extracts local explanations (signed per-feature contributions) from
        each EBM model available at the patient's current observation time.

        Args:
            context: A :class:`PatientContext` with ``_ebm_context`` set
                (created via ``PatientContext.from_csv`` with EBM enabled).
            save_path: Optional path to save the visualization figure.
            top_n: Max features to display (default 20).
            top_k_lines: Top features for line plots in Case C (default 5).

        Returns:
            Dict of local explanations per timeframe, or None if EBM is not
            available. See :func:`compute_ebm_local_explanations` for the
            dict structure.
        """
        from astra.inference.ebm import compute_ebm_local_explanations

        if context._ebm_context is None:
            logger.info("No EBM context in PatientContext — skipping EBM explanation.")
            return None

        ebm_ctx = context._ebm_context
        explanations = compute_ebm_local_explanations(
            raw_data=context._raw_data,
            filtered_concepts=ebm_ctx['filtered_concepts'],
            base_df=ebm_ctx['base_df'],
            cfg=ebm_ctx['cfg'],
            ebm_models_dir=ebm_ctx['ebm_models_dir'],
        )

        if not explanations:
            return None

        from astra.evaluation.behavior import visualize_ebm_patient_importance
        visualize_ebm_patient_importance(
            explanations,
            top_n=top_n,
            top_k_lines=top_k_lines,
            pid=context.pid,
            save_path=save_path,
        )

        return explanations


# ============================================================================
# STANDALONE EBM EXPLANATION (no InferenceSession needed)
# ============================================================================

def explain_ebm_standalone(
    raw_data: dict,
    filtered_concepts: Dict,
    base_df: pd.DataFrame,
    cfg: dict,
    ebm_models_dir: str = 'models/ebm',
    save_path: Optional[str] = None,
    top_n: int = 20,
    top_k_lines: int = 5,
    pid: Optional[str] = None,
) -> Optional[Dict]:
    """
    Compute and optionally visualize per-patient EBM feature importance
    without requiring an InferenceSession.

    Args:
        raw_data: Dict with patient data (must include 'admission_time', 'current_time').
        filtered_concepts: Dict mapping concept name -> filtered DataFrame.
        base_df: Single-row patient base DataFrame.
        cfg: Configuration dictionary.
        ebm_models_dir: Directory containing saved EBM deployment models.
        save_path: Optional path to save the visualization figure.
        top_n: Max features to display (default 20).
        top_k_lines: Top features for line plots in Case C (default 5).
        pid: Optional patient ID for title.

    Returns:
        Dict of local explanations per timeframe, or None if no models available.
    """
    from astra.inference.ebm import compute_ebm_local_explanations

    explanations = compute_ebm_local_explanations(
        raw_data=raw_data,
        filtered_concepts=filtered_concepts,
        base_df=base_df,
        cfg=cfg,
        ebm_models_dir=ebm_models_dir,
    )

    if not explanations:
        return None

    from astra.evaluation.behavior import visualize_ebm_patient_importance
    visualize_ebm_patient_importance(
        explanations,
        top_n=top_n,
        top_k_lines=top_k_lines,
        pid=pid,
        save_path=save_path,
    )

    return explanations


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
