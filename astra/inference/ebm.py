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

# Default aggregation functions (must match AggregatedDS defaults)
_AGG_FUNCS = ['first', 'last', 'min', 'max', 'mean', 'std']


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

    Returns:
        {masking_hours: predicted_probability}
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

    logger.info(
        f"Computing EBM predictions at {len(valid_intervals)} intervals "
        f"(elapsed: {(pd.Timestamp(raw_data['current_time']) - admission_time).total_seconds() / 3600:.1f}h)"
    )

    predictions = {}
    for masking_hours in valid_intervals:
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
    Create a single-row feature DataFrame matching AggregatedDS output.

    Replicates AggregatedDS.set_tab_df() + collect_and_aggregate_concepts()
    for one patient at one masking point, using in-memory filtered data
    instead of reading from disk.

    Feature naming matches AggregatedDS:
    - Continuous: {FEATURE}_{concept}_{agg_func}  (datasets.py:439)
    - Categorical: {VALUE_cleaned}_{concept}_count and _given  (datasets.py:289-290)

    Args:
        filtered_concepts: {concept_name: DataFrame} with columns
            [PID, FEATURE, VALUE, TIMESTAMP] for continuous,
            [PID, VALUE, TIMESTAMP] for categorical.
        base_df: Single-row patient DataFrame with demographics.
        admission_time: Patient admission time.
        masking_hours: Hours from admission to mask data at.
        ts_cat_names: Set of categorical concept names (from cfg).
        cfg: Configuration dictionary.

    Returns:
        Single-row DataFrame with all features.
    """
    cutoff = admission_time + pd.Timedelta(hours=masking_hours)
    row = base_df.iloc[0]

    # Start with demographics (matches AggregatedDS.set_tab_df)
    feature_row = {}
    id_col = cfg.get('dataset', {}).get('id_col', 'PID')
    feature_row[id_col] = row.get('PID', 1)

    for col in cfg.get('dataset', {}).get('num_cols', []):
        feature_row[col] = row.get(col, np.nan)
    for col in cfg.get('dataset', {}).get('cat_cols', []):
        feature_row[col] = row.get(col, np.nan)

    # Process each concept
    for concept_name, concept_df in filtered_concepts.items():
        if concept_df.empty:
            continue

        is_categorical = concept_name in ts_cat_names

        if is_categorical:
            _aggregate_categorical_concept(
                concept_df, concept_name, cutoff, feature_row
            )
        else:
            _aggregate_continuous_concept(
                concept_df, concept_name, cutoff, feature_row
            )

    return pd.DataFrame([feature_row])


def _aggregate_continuous_concept(
    concept_df: pd.DataFrame,
    concept_name: str,
    cutoff: pd.Timestamp,
    feature_row: dict,
) -> None:
    """
    Aggregate continuous concept data up to cutoff time.

    Naming: {FEATURE}_{concept_name}_{agg_func}
    Matches AggregatedDS._aggregate_numeric_cpu (datasets.py:439).
    """
    df = concept_df.copy()

    # Ensure TIMESTAMP is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['TIMESTAMP']):
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # Mask to before cutoff
    df = df[df['TIMESTAMP'] <= cutoff]
    if df.empty:
        return

    # Convert VALUE to numeric
    df['VALUE_numeric'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df = df.dropna(subset=['VALUE_numeric'])
    if df.empty:
        return

    # Sort by timestamp for first/last
    df = df.sort_values('TIMESTAMP')

    # Aggregate per FEATURE
    for feature, group in df.groupby('FEATURE'):
        vals = group['VALUE_numeric']
        if len(vals) == 0:
            continue

        for agg_func in _AGG_FUNCS:
            col_name = f'{feature}_{concept_name}_{agg_func}'

            if agg_func == 'first':
                feature_row[col_name] = float(vals.iloc[0])
            elif agg_func == 'last':
                feature_row[col_name] = float(vals.iloc[-1])
            elif agg_func == 'min':
                feature_row[col_name] = float(vals.min())
            elif agg_func == 'max':
                feature_row[col_name] = float(vals.max())
            elif agg_func == 'mean':
                feature_row[col_name] = float(vals.mean())
            elif agg_func == 'std':
                feature_row[col_name] = float(vals.std()) if len(vals) > 1 else 0.0


def _aggregate_categorical_concept(
    concept_df: pd.DataFrame,
    concept_name: str,
    cutoff: pd.Timestamp,
    feature_row: dict,
) -> None:
    """
    Aggregate categorical concept data up to cutoff time.

    Naming: {VALUE_cleaned}_{concept_name}_count and _given
    Matches AggregatedDS._aggregate_categorical_optimized (datasets.py:289-290).
    """
    df = concept_df.copy()

    # Determine timestamp column
    ts_col = 'TIMESTAMP'
    if ts_col not in df.columns:
        # ADTHaendelser may not have TIMESTAMP directly
        return

    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col])

    # Mask to before cutoff
    df = df[df[ts_col] <= cutoff]
    if df.empty:
        return

    # Drop NaN values
    df = df[df['VALUE'].notna()]
    if df.empty:
        return

    # Count occurrences per VALUE
    value_counts = df['VALUE'].value_counts()

    for value, count in value_counts.items():
        # Clean value name (matches datasets.py:286-288)
        cleaned = str(value).replace(' ', '_').replace('/', '_').replace('-', '_')
        feature_row[f'{cleaned}_{concept_name}_count'] = float(count)
        feature_row[f'{cleaned}_{concept_name}_given'] = 1.0


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

    # Compute elapsed hours per bin position
    # Same formula as ebm_features.py:48-51
    admission_time = pd.Timestamp(admission_time)
    elapsed_hours = (
        (bin_df['bin_start'] - admission_time).dt.total_seconds() / 3600
        + (bin_df['bin_end'] - bin_df['bin_start']).dt.total_seconds() / 7200
    ).values

    # Use caller-provided trajectory length (visible bins) or fall back to full grid
    if trajectory_length is None:
        trajectory_length = min(len(bin_df), seq_len)
    trajectory_length = min(trajectory_length, seq_len)

    # Sort interval keys for forward-fill
    intervals_hours = sorted(ebm_predictions.keys())
    default_value = 0.0

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

    n_nonzero = np.count_nonzero(filled)
    logger.info(
        f"Injected EBM predictions into channel {ebm_channel_idx}: "
        f"{n_nonzero}/{trajectory_length} non-zero positions, "
        f"range [{filled.min():.3f}, {filled.max():.3f}]"
    )

    return x_ts
