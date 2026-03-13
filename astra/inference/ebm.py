"""
EBM (Explainable Boosting Machine) prediction for single-patient inference.

Computes EBM predictions at multiple time intervals and injects them into
the _ebm_pred channel of x_ts, matching the batch pipeline in dataloader.py.

Also provides per-patient local explanations (feature contributions) from
each available EBM model for interpretability visualization.

Usage (called automatically by prepare_patient_from_csv when EBM is enabled):
    from astra.inference.ebm import compute_ebm_predictions, inject_ebm_into_x_ts

    preds = compute_ebm_predictions(raw_data, filtered_concepts, base_df, cfg, models_dir)
    x_ts = inject_ebm_into_x_ts(x_ts, preds, bin_df, admission_time, bundle)

    # Per-patient feature importance:
    from astra.inference.ebm import compute_ebm_local_explanations
    explanations = compute_ebm_local_explanations(raw_data, filtered_concepts, base_df, cfg)
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)



def _get_valid_ebm_intervals(
    raw_data: dict,
    cfg: dict,
    ebm_models_dir: str,
) -> Tuple[List[float], pd.Timestamp, set]:
    """
    Determine valid EBM intervals for a patient based on elapsed time and
    available model files.

    Returns:
        (valid_intervals, admission_time, ts_cat_names) or raises if no
        models directory exists.
    """
    from astra.models.ebm.generate_ebm_feature import (
        generate_ebm_intervals,
        _model_filename,
    )

    admission_time = pd.Timestamp(raw_data['admission_time'])
    current_time = pd.Timestamp(raw_data['current_time'])
    max_elapsed_hours = (current_time - admission_time).total_seconds() / 3600

    # Get all possible intervals from config
    all_intervals = generate_ebm_intervals(cfg)

    # Filter to intervals within patient's time window and with saved models
    valid_intervals = []
    for h in all_intervals:
        if h > max_elapsed_hours:
            continue
        model_path = os.path.join(ebm_models_dir, _model_filename(h))
        if os.path.exists(model_path):
            valid_intervals.append(h)

    ts_cat_names = set(cfg.get('dataset', {}).get('ts_cat_names', []))

    return valid_intervals, admission_time, ts_cat_names


def compute_ebm_predictions(
    raw_data: dict,
    filtered_concepts: Dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    cfg: dict,
    ebm_models_dir: str = 'models/ebm',
    cached_predictions: Optional[Dict[float, float]] = None,
) -> Dict[float, float]:
    """
    Compute EBM predictions at all relevant intervals for a single patient.

    For each interval where masking_hours <= elapsed time AND a trained model
    exists, aggregates patient features and runs the EBM model.

    Args:
        raw_data: Dict with patient data (from _filtered_dfs_to_raw_data).
        filtered_concepts: Dict mapping concept name -> filtered DataFrame
            (from _filter_concepts_for_patient).
        base_df: Single-row patient base DataFrame.
        cfg: Configuration dictionary.
        ebm_models_dir: Directory containing saved EBM deployment models.
        cached_predictions: Previously computed predictions to skip.
            Intervals already present in this dict will not be recomputed.

    Returns:
        Dict with only *newly computed* predictions
        ``{masking_hours: predicted_probability}``.  Caller should merge
        with *cached_predictions* to get the full set.
    """
    if not os.path.isdir(ebm_models_dir):
        logger.warning(
            f"EBM models directory not found: {ebm_models_dir}. "
            "Leaving _ebm_pred channel empty."
        )
        return {}

    valid_intervals, admission_time, ts_cat_names = _get_valid_ebm_intervals(
        raw_data, cfg, ebm_models_dir
    )

    if not valid_intervals:
        max_elapsed = (
            pd.Timestamp(raw_data['current_time']) -
            pd.Timestamp(raw_data['admission_time'])
        ).total_seconds() / 3600
        logger.info(
            f"No EBM models available for elapsed time {max_elapsed:.1f}h. "
            f"Models dir: {ebm_models_dir}"
        )
        return {}

    # Skip intervals already in cache
    _cached = cached_predictions or {}
    intervals_to_compute = [h for h in valid_intervals if h not in _cached]

    if not intervals_to_compute:
        logger.debug("All %d EBM intervals already cached — skipping", len(valid_intervals))
        return {}

    logger.info(
        f"Computing EBM predictions at {len(intervals_to_compute)} NEW intervals "
        f"({len(_cached)} cached, elapsed: "
        f"{(pd.Timestamp(raw_data['current_time']) - admission_time).total_seconds() / 3600:.1f}h)"
    )

    predictions = {}
    for masking_hours in intervals_to_compute:
        try:
            prob = _predict_at_interval_from_raw(
                filtered_concepts=filtered_concepts,
                base_df=base_df,
                admission_time=admission_time,
                masking_hours=masking_hours,
                ts_cat_names=ts_cat_names,
                cfg=cfg,
                ebm_models_dir=ebm_models_dir,
            )
            predictions[masking_hours] = prob
        except Exception as e:
            logger.warning(
                f"EBM prediction failed at {masking_hours:.1f}h: {e}"
            )
            continue

    if predictions:
        logger.info(
            f"EBM predictions computed: {len(predictions)}/{len(valid_intervals)} intervals. "
            f"Range: [{min(predictions.values()):.3f}, {max(predictions.values()):.3f}]"
        )
    else:
        logger.info("No EBM predictions computed.")

    return predictions


def compute_ebm_local_explanations(
    raw_data: dict,
    filtered_concepts: Dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    cfg: dict,
    ebm_models_dir: str = 'models/ebm',
) -> Dict[float, Dict]:
    """
    Compute per-feature local EBM explanations at all relevant intervals.

    For each available EBM model (where masking_hours <= elapsed time),
    extracts signed per-feature contributions using InterpretML's
    explain_local(), which decomposes the prediction into additive
    feature effects: logit = intercept + sum(f_i(x_i)).

    Args:
        raw_data: Dict with patient data (from _filtered_dfs_to_raw_data).
        filtered_concepts: Dict mapping concept name -> filtered DataFrame.
        base_df: Single-row patient base DataFrame.
        cfg: Configuration dictionary.
        ebm_models_dir: Directory containing saved EBM deployment models.

    Returns:
        {masking_hours: {
            'feature_names': List[str],
            'contributions': np.ndarray,   # signed per-feature contributions
            'intercept': float,
            'predicted_prob': float,
            'feature_values': np.ndarray,  # raw feature values for context
        }}
        Empty dict if no models available.
    """
    if not os.path.isdir(ebm_models_dir):
        logger.warning(
            f"EBM models directory not found: {ebm_models_dir}. "
            "Cannot compute local explanations."
        )
        return {}

    valid_intervals, admission_time, ts_cat_names = _get_valid_ebm_intervals(
        raw_data, cfg, ebm_models_dir
    )

    if not valid_intervals:
        logger.info("No EBM models available for local explanations.")
        return {}

    logger.info(
        f"Computing EBM local explanations at {len(valid_intervals)} intervals"
    )

    explanations = {}
    for masking_hours in valid_intervals:
        try:
            model_dict, X_processed = _prepare_ebm_at_interval(
                filtered_concepts=filtered_concepts,
                base_df=base_df,
                admission_time=admission_time,
                masking_hours=masking_hours,
                ts_cat_names=ts_cat_names,
                cfg=cfg,
                ebm_models_dir=ebm_models_dir,
            )

            ebm = model_dict['model']

            # Get prediction probability
            prob = float(ebm.predict_proba(X_processed)[:, 1][0])

            # Get local explanation (additive feature contributions)
            local_exp = ebm.explain_local(X_processed)
            exp_data = local_exp.data(0)

            # Feature values may be mixed types (float for continuous,
            # str for categorical/interaction terms), so keep as object array
            raw_values = exp_data['values']
            try:
                feature_values = np.array(raw_values, dtype=float)
            except (ValueError, TypeError):
                feature_values = np.array(raw_values, dtype=object)

            explanations[masking_hours] = {
                'feature_names': list(exp_data['names']),
                'contributions': np.array(exp_data['scores'], dtype=float),
                'intercept': float(ebm.intercept_[0]),
                'predicted_prob': prob,
                'feature_values': feature_values,
            }
        except Exception as e:
            logger.warning(
                f"EBM local explanation failed at {masking_hours:.1f}h: {e}"
            )
            continue

    if explanations:
        logger.info(
            f"EBM local explanations computed: {len(explanations)}/{len(valid_intervals)} intervals"
        )
    else:
        logger.info("No EBM local explanations computed.")

    return explanations


def _prepare_ebm_at_interval(
    filtered_concepts: Dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    admission_time: pd.Timestamp,
    masking_hours: float,
    ts_cat_names: set,
    cfg: dict,
    ebm_models_dir: str,
) -> Tuple[dict, pd.DataFrame]:
    """
    Aggregate features and load EBM model at a single interval.

    Shared helper for both prediction and local explanation.

    Returns:
        (model_dict, X_processed) where model_dict contains the EBM model,
        encoder, and feature metadata, and X_processed is the preprocessed
        feature DataFrame ready for prediction/explanation.
    """
    from astra.models.ebm.generate_ebm_feature import (
        _model_filename,
        preprocess_features,
    )

    # Load EBM model
    model_path = os.path.join(ebm_models_dir, _model_filename(masking_hours))
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    # Aggregate patient features at this masking point
    X_row = _aggregate_patient_features(
        filtered_concepts=filtered_concepts,
        base_df=base_df,
        admission_time=admission_time,
        masking_hours=masking_hours,
        ts_cat_names=ts_cat_names,
        cfg=cfg,
    )

    # Remove PID column (EBM expects features only)
    id_col = cfg.get('dataset', {}).get('id_col', 'PID')
    X_features = X_row.drop(columns=[id_col], errors='ignore')

    # Determine which of our features are cat vs cont for preprocessing
    our_cat_feats = [
        c for c in X_features.columns
        if c in model_dict['expected_cat_feats']
    ]
    our_cont_feats = [
        c for c in X_features.columns
        if c in model_dict['expected_cont_feats']
    ]

    # Preprocess (handles missing features by adding zero columns)
    X_processed, _, _ = preprocess_features(
        X_features,
        cat_feats=our_cat_feats,
        cont_feats=our_cont_feats,
        encoder=model_dict['encoder'],
        fit=False,
        expected_cat_feats=model_dict['expected_cat_feats'],
        expected_cont_feats=model_dict['expected_cont_feats'],
    )

    return model_dict, X_processed


def _predict_at_interval_from_raw(
    filtered_concepts: Dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    admission_time: pd.Timestamp,
    masking_hours: float,
    ts_cat_names: set,
    cfg: dict,
    ebm_models_dir: str,
) -> float:
    """
    Aggregate features and run EBM prediction at a single interval.

    Returns:
        Predicted probability (float).
    """
    model_dict, X_processed = _prepare_ebm_at_interval(
        filtered_concepts=filtered_concepts,
        base_df=base_df,
        admission_time=admission_time,
        masking_hours=masking_hours,
        ts_cat_names=ts_cat_names,
        cfg=cfg,
        ebm_models_dir=ebm_models_dir,
    )

    # Predict
    prob = model_dict['model'].predict_proba(X_processed)[:, 1]
    return float(prob[0])


def _aggregate_patient_features(
    filtered_concepts: Dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    admission_time: pd.Timestamp,
    masking_hours: float,
    ts_cat_names: set,
    cfg: dict,
) -> pd.DataFrame:
    """
    Create a single-row feature DataFrame using the batch AggregatedDS path.

    Delegates to _create_aggregated_dataset() with a concept_cache built from
    the in-memory filtered_concepts, ensuring perfect feature parity with the
    batch EBM pipeline (same masking, aggregation, fillna, and naming logic).

    Args:
        filtered_concepts: {concept_name: DataFrame} with columns
            [PID, FEATURE, VALUE, TIMESTAMP] for continuous,
            [PID, VALUE, TIMESTAMP] for categorical.
        base_df: Single-row patient DataFrame with demographics.
        admission_time: Patient admission time (unused — AggregatedDS reads
            start from base_df directly).
        masking_hours: Hours from admission to mask data at.
        ts_cat_names: Set of categorical concept names (from cfg).
        cfg: Configuration dictionary.

    Returns:
        Single-row DataFrame with all features (including PID column).
    """
    from astra.models.ebm.generate_ebm_feature import _create_aggregated_dataset

    # Build concept_cache in the format AggregatedDS expects:
    # {concept_name: DataFrame[PID, FEATURE, VALUE, TIMESTAMP]}
    # ALL config concepts must have an entry (even if empty) to prevent
    # AggregatedDS from falling back to disk I/O for missing concepts.
    required_cols = ['PID', 'FEATURE', 'VALUE', 'TIMESTAMP']
    empty_df = pd.DataFrame(columns=required_cols)
    concept_cache = {c: empty_df for c in cfg.get('concepts', [])}

    for concept, df in filtered_concepts.items():
        if df.empty:
            continue
        df_std = df.copy()
        # Categorical concepts may lack FEATURE column — add fallback so
        # AggregatedDS can process them uniformly.
        if 'FEATURE' not in df_std.columns:
            df_std['FEATURE'] = concept
        # Keep only the standard columns (drop END_TIMESTAMP, extras)
        available = [c for c in required_cols if c in df_std.columns]
        concept_cache[concept] = df_std[available]

    X, y, cat_feats, cont_feats = _create_aggregated_dataset(
        base_df, cfg, masking_hours, concept_cache=concept_cache,
    )
    return X


def inject_ebm_into_x_ts(
    x_ts: np.ndarray,
    ebm_predictions: Dict[float, float],
    bin_df: pd.DataFrame,
    admission_time: pd.Timestamp,
    bundle: dict,
    trajectory_length: int = None,
) -> np.ndarray:
    """
    Populate the _ebm_pred channel in x_ts with forward-filled EBM predictions.

    Args:
        x_ts: [n_channels, seq_len] raw continuous time series.
        ebm_predictions: {masking_hours: probability} from compute_ebm_predictions.
        bin_df: Patient bin DataFrame with bin_start, bin_end columns.
        admission_time: Patient admission time (pd.Timestamp).
        bundle: Deployment bundle.
        trajectory_length: Number of visible bins (from current_time masking).
            If None, defaults to min(len(bin_df), seq_len) for backward compat.

    Returns:
        Modified x_ts with EBM predictions filled in.
    """
    from astra.data.ebm_features import _forward_fill_predictions

    ts_channel_names = bundle['ts_channel_names']
    if '_ebm_pred' not in ts_channel_names:
        return x_ts

    ebm_channel_idx = ts_channel_names.index('_ebm_pred')
    seq_len = x_ts.shape[1]

    if not ebm_predictions:
        logger.info("No EBM predictions to inject — channel stays at default (0.0)")
        # Ensure the channel is 0.0 (not NaN) so normalization produces 0.0
        x_ts[ebm_channel_idx, :] = 0.0
        return x_ts

    # Compute elapsed hours at bin_start — matches batch pipeline
    # (ebm_features.py:60).  Using bin_start (not midpoint) ensures the
    # forward-fill comparison ``masking_hours <= elapsed_hours`` is strictly
    # conservative: an EBM trained at masking time M is only applied from the
    # first bin that STARTS at or after M.
    admission_time = pd.Timestamp(admission_time)
    elapsed_hours = (
        (bin_df['bin_start'] - admission_time).dt.total_seconds() / 3600
    ).values

    # Use caller-provided trajectory length (visible bins) or fall back to full grid
    if trajectory_length is None:
        trajectory_length = min(len(bin_df), seq_len)
    trajectory_length = min(trajectory_length, seq_len, len(elapsed_hours))

    # Sort interval keys for forward-fill
    intervals_hours = sorted(ebm_predictions.keys())
    # NaN for positions before the first EBM interval — matches batch pipeline
    # (ebm_features.py:177).  NaN is treated as missing by normalization (→ 0.0),
    # whereas a raw 0.0 would normalize to (0 - mean) / std ≠ 0.
    default_value = np.nan

    # Forward-fill predictions to bin positions
    filled = _forward_fill_predictions(
        patient_elapsed=elapsed_hours[:trajectory_length],
        ebm_intervals_hours=intervals_hours,
        patient_preds=ebm_predictions,
        default_value=default_value,
    )

    # Inject into x_ts
    x_ts[ebm_channel_idx, :trajectory_length] = filled
    # Ensure padding beyond trajectory is 0.0
    if trajectory_length < seq_len:
        x_ts[ebm_channel_idx, trajectory_length:] = 0.0

    measured = filled[~np.isnan(filled)]
    n_measured = len(measured)
    n_nonzero = np.count_nonzero(measured)
    val_range = f"[{measured.min():.3f}, {measured.max():.3f}]" if n_measured > 0 else "N/A"
    logger.info(
        f"Injected EBM predictions into channel {ebm_channel_idx}: "
        f"{n_nonzero}/{trajectory_length} non-zero positions "
        f"({trajectory_length - n_measured} NaN pre-EBM), range {val_range}"
    )

    return x_ts
