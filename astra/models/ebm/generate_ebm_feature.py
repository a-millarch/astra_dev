import logging
import os
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from interpret.glassbox import ExplainableBoostingClassifier

from astra.utils import get_base_df, get_train_test_split, cfg
from astra.data.datasets import AggregatedDS

logger = logging.getLogger(__name__)


def _parse_duration_to_hours(s: str) -> float:
    """Parse a duration string like '10min', '4h', '6D' to hours."""
    if s.endswith("min"):
        return float(s[:-3]) / 60
    elif s.endswith("h"):
        return float(s[:-1])
    elif s.endswith("D"):
        return float(s[:-1]) * 24
    else:
        raise ValueError(f"Cannot parse duration: {s!r}")


def generate_ebm_intervals(cfg_dict: dict) -> List[float]:
    """
    Generate EBM training intervals in hours.

    If ``ebm_feature.interval_schedule`` is defined in config, uses that to
    generate the full schedule.  Otherwise falls back to the legacy hardcoded
    early schedule + bin_intervals-derived post-72h schedule.

    The ``interval_schedule`` format mirrors ``bin_intervals``::

        interval_schedule:
          30min: '10min'   # 0 → 30min, step 10min
          4h: '1h'         # 30min → 4h, step 1h
          ...

    Each range starts at the next clean multiple of the step above the
    previous boundary so intervals land on round numbers.

    Returns:
        Sorted list of masking times in hours.
    """
    import math

    ebm_cfg = cfg_dict.get("ebm_feature", {})
    schedule = ebm_cfg.get("interval_schedule")

    if schedule:
        return _intervals_from_schedule(schedule)

    # Legacy fallback: hardcoded early + bin_intervals post-72h
    return _intervals_legacy(cfg_dict)


def _intervals_from_schedule(schedule: dict) -> List[float]:
    """Generate intervals from an explicit interval_schedule config dict."""
    import math

    sorted_keys = sorted(schedule.keys(), key=_parse_duration_to_hours)
    intervals: List[float] = []
    prev_boundary_h = 0.0

    for key in sorted_keys:
        end_h = _parse_duration_to_hours(key)
        step_h = _parse_duration_to_hours(schedule[key])

        # Start at the first clean multiple of step_h above prev_boundary_h
        first = math.ceil(prev_boundary_h / step_h) * step_h
        if first == prev_boundary_h and first > 0:
            first += step_h

        t = first
        while t <= end_h + 1e-9:  # small epsilon for float precision
            intervals.append(round(t, 6))
            t += step_h

        prev_boundary_h = end_h

    return sorted(set(intervals))


def _intervals_legacy(cfg_dict: dict) -> List[float]:
    """Legacy interval generation: hardcoded early + bin_intervals post-72h."""
    intervals = [10 / 60, 30 / 60, 1, 2, 3, 4]
    intervals += list(range(6, 73, 2))  # 6h to 72h, step 2h

    bin_intervals = cfg_dict.get("bin_intervals", {})
    bin_freq_include = set(cfg_dict.get("bin_freq_include", []))

    sorted_keys = sorted(
        [k for k in bin_intervals.keys() if k != "end"],
        key=_parse_duration_to_hours,
    )

    for i, key in enumerate(sorted_keys):
        end_h = _parse_duration_to_hours(key)

        if i > 0:
            start_h = _parse_duration_to_hours(sorted_keys[i - 1])
        else:
            start_h = 0

        if end_h <= 72:
            continue

        start_h = max(start_h, 72)

        resolution_str = bin_intervals[key]
        if resolution_str not in bin_freq_include:
            continue

        res_h = _parse_duration_to_hours(resolution_str)

        t = start_h + res_h
        while t <= end_h:
            intervals.append(t)
            t += res_h

    return sorted(set(intervals))


def _get_default_ebm_params() -> dict:
    """Default EBM hyperparameters, matching train_ebm_over_time.py."""
    return {
        "random_state": 42,
        "interactions": 3,
        "validation_size": 0.2,
        "early_stopping_rounds": 100,
        "max_leaves": 2,
        "inner_bags": 0,
    }


def _create_aggregated_dataset(
    base_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    concept_cache: Optional[dict] = None,
) -> Tuple[pd.DataFrame, np.ndarray, list, list]:
    """
    Create AggregatedDS at a masking point and extract X, y with PIDs.

    Args:
        concept_cache: Optional pre-loaded concept data (from preload_concept_cache).
            Avoids reloading concept pkls from disk for every interval.

    Returns:
        X (DataFrame with PID column), y (array), categorical_features, continuous_features
    """
    masking_point = pd.Timedelta(hours=masking_hours)

    agg_ds = AggregatedDS(
        cfg=cfg_dict,
        base_df=base_df,
        masking_point=masking_point,
        agg_funcs=["first", "last", "min", "max", "mean", "std"],
        concepts=cfg_dict["concepts"],
        default_mode=True,
        concept_cache=concept_cache,
    )

    X, y = agg_ds.get_X_y(include_id=True)
    return X, np.asarray(y), agg_ds.categorical_features, agg_ds.continuous_features


def preload_concept_cache(
    base_df: pd.DataFrame,
    cfg_dict: dict,
) -> Dict[str, pd.DataFrame]:
    """
    Pre-load and filter concept data for a patient set.

    Performs the expensive disk I/O and concept-specific filtering (e.g.,
    filter_vitals with prehospital concat) once. The returned cache can be
    passed to _create_aggregated_dataset / AggregatedDS for every masking
    interval, skipping repeated pkl reads.

    Args:
        base_df: Patient base DataFrame (trainval or holdout).
        cfg_dict: Configuration dictionary.

    Returns:
        Dict mapping concept name -> filtered DataFrame with columns
        [PID, FEATURE, VALUE, TIMESTAMP].
    """
    # Create a non-default-mode AggregatedDS just to access loading infrastructure
    agg = AggregatedDS(
        cfg=cfg_dict,
        base_df=base_df,
        default_mode=False,
        concepts=cfg_dict["concepts"],
    )
    cache = {}
    for concept in agg.concepts:
        try:
            df = agg._load_and_filter_concept(concept)
            if len(df) > 0:
                cache[concept] = df
                logger.info(f"Cached {concept}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to cache {concept}: {e}")
    return cache


def _pad_to_reference_features(
    X: pd.DataFrame,
    cat_feats: list,
    cont_feats: list,
    ref_cat_feats: list,
    ref_cont_feats: list,
) -> Tuple[pd.DataFrame, list, list]:
    """
    Ensure X has all reference features, adding zero-filled columns for missing ones.

    This guarantees all EBM models (across all masking time points) share the same
    feature space. Early time points where certain features don't exist yet get
    zero-valued columns, so the EBM can still accept those features at inference
    time for patients who do have data that early.

    Returns:
        X with all reference columns, ref_cat_feats, ref_cont_feats
    """
    missing_cont = [f for f in ref_cont_feats if f not in X.columns]
    missing_cat = [f for f in ref_cat_feats if f not in X.columns]

    if missing_cont or missing_cat:
        # Build missing columns in one shot to avoid fragmentation
        missing_data = {}
        for feat in missing_cont:
            missing_data[feat] = 0.0
        for feat in missing_cat:
            missing_data[feat] = np.nan
        if missing_data:
            missing_df = pd.DataFrame(missing_data, index=X.index)
            X = pd.concat([X, missing_df], axis=1)

    return X, ref_cat_feats, ref_cont_feats


def preprocess_features(
    X: pd.DataFrame,
    cat_feats: list,
    cont_feats: list,
    encoder: Optional[OneHotEncoder] = None,
    fit: bool = True,
    expected_cat_feats: Optional[list] = None,
    expected_cont_feats: Optional[list] = None,
) -> Tuple[pd.DataFrame, Optional[OneHotEncoder], list]:
    """
    Preprocess features with explicit one-hot encoding for categoricals.
    
    **KEY FIX**: Handles missing categorical AND continuous columns in holdout/validation data
    by adding them as zero columns to maintain consistent feature space.
    
    Args:
        X: Feature dataframe (no ID column)
        cat_feats: Categorical feature names PRESENT in this dataset
        cont_feats: Continuous feature names PRESENT in this dataset
        encoder: Fitted encoder (for transform-only mode)
        fit: Whether to fit encoder (True for train, False for val/test)
        expected_cat_feats: Expected categorical feature names (for validation/holdout)
        expected_cont_feats: Expected continuous feature names (for validation/holdout)
    
    Returns:
        X_processed: DataFrame with all continuous features
        encoder: Fitted encoder (if fit=True) or None
        feature_names: List of all feature names
    """
    
    if fit:
        # TRAINING MODE: Fit encoder on available categorical features
        # Separate continuous features (use what's present)
        X_cont = X[cont_feats].copy() if cont_feats else pd.DataFrame(index=X.index)
        
        if len(cat_feats) > 0:
            X_cat = X[cat_feats]
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',  # Ignore unknown categories
                dtype=np.float64,
            )
            X_cat_encoded = encoder.fit_transform(X_cat)
            
            # Generate feature names
            cat_feature_names = []
            for i, cat_feat in enumerate(cat_feats):
                categories = encoder.categories_[i]
                for cat in categories:
                    cat_feature_names.append(f"{cat_feat}_{cat}")
            
            X_cat_df = pd.DataFrame(
                X_cat_encoded,
                index=X.index,
                columns=cat_feature_names,
            )
            
            # Combine categorical and continuous
            X_processed = pd.concat([X_cat_df, X_cont], axis=1)
            feature_names = cat_feature_names + cont_feats
        else:
            # No categorical features
            X_processed = X_cont.copy()
            feature_names = cont_feats
            encoder = None
    else:
        # VALIDATION/HOLDOUT MODE: Transform with missing feature handling
        if encoder is None or expected_cat_feats is None or expected_cont_feats is None:
            raise ValueError("Encoder, expected_cat_feats, and expected_cont_feats must be provided when fit=False")
        
        # === HANDLE CATEGORICAL FEATURES ===
        if len(expected_cat_feats) > 0:
            # Create DataFrame with ALL expected categorical features in one shot
            cat_data = {
                feat: X[feat] if feat in X.columns else np.nan
                for feat in expected_cat_feats
            }
            X_cat_full = pd.DataFrame(cat_data, index=X.index)

            # Fill NaN with sentinel string to avoid mixed dtype (object + float)
            # causing np.isnan failures in encoder.transform/predict_proba.
            # handle_unknown='ignore' encodes unknown categories as all-zeros.
            X_cat_full = X_cat_full.fillna('#missing#')

            X_cat_encoded = encoder.transform(X_cat_full)
            
            # Generate feature names (must match training)
            cat_feature_names = []
            for i, cat_feat in enumerate(expected_cat_feats):
                categories = encoder.categories_[i]
                for cat in categories:
                    cat_feature_names.append(f"{cat_feat}_{cat}")
            
            X_cat_df = pd.DataFrame(
                X_cat_encoded,
                index=X.index,
                columns=cat_feature_names,
            )
        else:
            cat_feature_names = []
            X_cat_df = pd.DataFrame(index=X.index)
        
        # === HANDLE CONTINUOUS FEATURES ===
        # Create DataFrame with ALL expected continuous features in one shot
        cont_data = {
            feat: X[feat] if feat in X.columns else 0.0
            for feat in expected_cont_feats
        }
        X_cont = pd.DataFrame(cont_data, index=X.index)
        
        # Combine categorical and continuous
        X_processed = pd.concat([X_cat_df, X_cont], axis=1)
        feature_names = cat_feature_names + expected_cont_feats
    
    return X_processed, encoder, feature_names


def train_ebm_kfold_at_timepoint(
    train_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    n_folds: int = 5,
    ebm_params: Optional[dict] = None,
    ref_cat_feats: Optional[list] = None,
    ref_cont_feats: Optional[list] = None,
) -> Tuple[Dict[int, float], list, list, list]:
    """
    Train K-fold EBMs at one masking point to generate OOF predictions.

    Args:
        train_df: Training patients base_df (no holdout).
        cfg_dict: Configuration dictionary.
        masking_hours: Time point in hours.
        n_folds: Number of CV folds.
        ebm_params: EBM hyperparameters.
        ref_cat_feats: Reference categorical features (ensures consistent feature space
            across all time points). If None, uses only features present at this time point.
        ref_cont_feats: Reference continuous features. If None, uses only features present.

    Returns:
        oof_preds: {PID: predicted_probability} for all training patients.
        fold_models: List of (model, encoder, expected_cat_feats, expected_cont_feats) tuples.
        expected_cat_feats: List of categorical feature names (reference set if provided).
        expected_cont_feats: List of continuous feature names (reference set if provided).
    """
    if ebm_params is None:
        ebm_params = _get_default_ebm_params()

    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        train_df, cfg_dict, masking_hours
    )

    id_col = cfg_dict["dataset"]["id_col"]
    pids = X_full[id_col].values
    X_features = X_full.drop(columns=[id_col])

    # Pad to reference features if provided (ensures all time points share same feature space)
    if ref_cat_feats is not None and ref_cont_feats is not None:
        X_features, cat_feats, cont_feats = _pad_to_reference_features(
            X_features, cat_feats, cont_feats, ref_cat_feats, ref_cont_feats,
        )

    # Fit encoder on FULL dataset to ensure all categories are known
    X_processed, global_encoder, feature_names = preprocess_features(
        X_features, cat_feats, cont_feats, encoder=None, fit=True
    )

    oof_preds = {}
    fold_models = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_full)):
        X_train = X_processed.iloc[train_idx]
        y_train = y_full[train_idx]
        X_val = X_processed.iloc[val_idx]
        y_val = y_full[val_idx]
        val_pids = pids[val_idx]

        # Check class diversity
        if len(set(y_train)) < 2:
            logger.warning(
                f"  Fold {fold_idx}: insufficient class diversity in train, skipping"
            )
            for pid in val_pids:
                oof_preds[pid] = 0.0
            continue

        # Verify shapes match
        if X_train.shape[1] != X_val.shape[1]:
            logger.error(
                f"  Fold {fold_idx}: Shape mismatch! Train={X_train.shape}, Val={X_val.shape}"
            )
            for pid in val_pids:
                oof_preds[pid] = 0.0
            continue

        ebm = ExplainableBoostingClassifier(
            feature_names=feature_names,
            **ebm_params,
        )
        ebm.fit(X_train, y_train)

        y_proba = ebm.predict_proba(X_val)[:, 1]

        for pid, prob in zip(val_pids, y_proba):
            oof_preds[pid] = float(prob)

        # Store model with encoder and expected features for holdout prediction
        fold_models.append((ebm, global_encoder, cat_feats, cont_feats))

    return oof_preds, fold_models, cat_feats, cont_feats


def predict_holdout(
    holdout_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    fold_models: list,
    expected_cat_feats: list,
    expected_cont_feats: list,
    ref_cat_feats: Optional[list] = None,
    ref_cont_feats: Optional[list] = None,
) -> Dict[int, float]:
    """
    Generate averaged predictions for holdout patients across K fold models.

    Args:
        holdout_df: Holdout patients base_df.
        cfg_dict: Configuration dictionary.
        masking_hours: Time point in hours.
        fold_models: List of (model, encoder, expected_cat_feats, expected_cont_feats) tuples.
        expected_cat_feats: Categorical features expected from training.
        expected_cont_feats: Continuous features expected from training.
        ref_cat_feats: Reference categorical features for consistent feature space.
        ref_cont_feats: Reference continuous features for consistent feature space.

    Returns:
        {PID: averaged_probability}
    """
    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        holdout_df, cfg_dict, masking_hours
    )

    id_col = cfg_dict["dataset"]["id_col"]
    pids = X_full[id_col].values
    X_features = X_full.drop(columns=[id_col])

    # Pad to reference features if provided
    if ref_cat_feats is not None and ref_cont_feats is not None:
        X_features, _, _ = _pad_to_reference_features(
            X_features, cat_feats, cont_feats, ref_cat_feats, ref_cont_feats,
        )

    if len(fold_models) == 0:
        return {pid: 0.0 for pid in pids}

    # Average predictions across fold models
    all_proba = np.zeros(len(X_features))

    for model, encoder, _, _ in fold_models:
        X_processed, _, _ = preprocess_features(
            X_features, cat_feats, cont_feats,
            encoder=encoder, fit=False,
            expected_cat_feats=expected_cat_feats,
            expected_cont_feats=expected_cont_feats
        )

        all_proba += model.predict_proba(X_processed)[:, 1]

    all_proba /= len(fold_models)

    return {pid: float(prob) for pid, prob in zip(pids, all_proba)}


def predict_holdout_with_deployment_model(
    holdout_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    final_model_dict: dict,
    ref_cat_feats: Optional[list] = None,
    ref_cont_feats: Optional[list] = None,
    concept_cache: Optional[dict] = None,
) -> Dict[int, float]:
    """
    Generate predictions for holdout patients using the deployment model.

    Uses the same model that will be loaded at inference time, ensuring
    holdout predictions in ebm_predictions.pkl match inference predictions
    for the same patient data.

    Args:
        holdout_df: Holdout patients base_df.
        cfg_dict: Configuration dictionary.
        masking_hours: Time point in hours.
        final_model_dict: Deployment model dict from train_final_ebm_at_timepoint().
        ref_cat_feats: Reference categorical features for consistent feature space.
        ref_cont_feats: Reference continuous features for consistent feature space.
        concept_cache: Pre-loaded concept data for holdout patients.

    Returns:
        {PID: predicted_probability}
    """
    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        holdout_df, cfg_dict, masking_hours,
        concept_cache=concept_cache,
    )

    id_col = cfg_dict["dataset"]["id_col"]
    pids = X_full[id_col].values
    X_features = X_full.drop(columns=[id_col])

    # Pad to reference features if provided
    if ref_cat_feats is not None and ref_cont_feats is not None:
        X_features, _, _ = _pad_to_reference_features(
            X_features, cat_feats, cont_feats, ref_cat_feats, ref_cont_feats,
        )

    X_processed, _, _ = preprocess_features(
        X_features,
        cat_feats=cat_feats,
        cont_feats=cont_feats,
        encoder=final_model_dict['encoder'],
        fit=False,
        expected_cat_feats=final_model_dict['expected_cat_feats'],
        expected_cont_feats=final_model_dict['expected_cont_feats'],
    )

    proba = final_model_dict['model'].predict_proba(X_processed)[:, 1]
    return {pid: float(p) for pid, p in zip(pids, proba)}


def train_final_ebm_at_timepoint(
    train_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    ebm_params: dict,
    ref_cat_feats: Optional[list] = None,
    ref_cont_feats: Optional[list] = None,
):
    """
    Train ONE final EBM on full trainval data for deployment.
    """
    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        train_df, cfg_dict, masking_hours
    )

    id_col = cfg_dict["dataset"]["id_col"]
    X_features = X_full.drop(columns=[id_col])

    # Pad to reference features if provided
    if ref_cat_feats is not None and ref_cont_feats is not None:
        X_features, cat_feats, cont_feats = _pad_to_reference_features(
            X_features, cat_feats, cont_feats, ref_cat_feats, ref_cont_feats,
        )

    X_processed, encoder, feature_names = preprocess_features(
        X_features, cat_feats, cont_feats, encoder=None, fit=True
    )

    ebm = ExplainableBoostingClassifier(
        feature_names=feature_names,
        **ebm_params,
    )
    ebm.fit(X_processed, y_full)

    return {
        "model": ebm,
        "encoder": encoder,
        "expected_cat_feats": cat_feats,
        "expected_cont_feats": cont_feats,
        "feature_names": feature_names,
    }


def _train_deployment_from_dataset(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    cat_feats: list,
    cont_feats: list,
    cfg_dict: dict,
    ebm_params: dict,
    ref_cat_feats: Optional[list] = None,
    ref_cont_feats: Optional[list] = None,
) -> dict:
    """
    Train deployment EBM from a pre-computed dataset (no disk I/O).

    Same logic as train_final_ebm_at_timepoint but accepts pre-built X/y
    so the caller can share dataset creation with K-fold training.
    """
    id_col = cfg_dict["dataset"]["id_col"]
    X_features = X_full.drop(columns=[id_col])

    if ref_cat_feats is not None and ref_cont_feats is not None:
        X_features, cat_feats, cont_feats = _pad_to_reference_features(
            X_features, cat_feats, cont_feats, ref_cat_feats, ref_cont_feats,
        )

    X_processed, encoder, feature_names = preprocess_features(
        X_features, cat_feats, cont_feats, encoder=None, fit=True
    )

    ebm = ExplainableBoostingClassifier(
        feature_names=feature_names,
        **ebm_params,
    )
    ebm.fit(X_processed, y_full)

    return {
        "model": ebm,
        "encoder": encoder,
        "expected_cat_feats": cat_feats,
        "expected_cont_feats": cont_feats,
        "feature_names": feature_names,
    }


def _train_kfold_from_dataset(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    cat_feats: list,
    cont_feats: list,
    cfg_dict: dict,
    masking_hours: float,
    n_folds: int = 5,
    ebm_params: Optional[dict] = None,
    ref_cat_feats: Optional[list] = None,
    ref_cont_feats: Optional[list] = None,
) -> Tuple[Dict[int, float], list, list, list]:
    """
    K-fold EBM training from a pre-computed dataset (no disk I/O).

    Same logic as train_ebm_kfold_at_timepoint but accepts pre-built X/y
    so the caller can share dataset creation with the deployment model.
    """
    if ebm_params is None:
        ebm_params = _get_default_ebm_params()

    id_col = cfg_dict["dataset"]["id_col"]
    pids = X_full[id_col].values
    X_features = X_full.drop(columns=[id_col])

    if ref_cat_feats is not None and ref_cont_feats is not None:
        X_features, cat_feats, cont_feats = _pad_to_reference_features(
            X_features, cat_feats, cont_feats, ref_cat_feats, ref_cont_feats,
        )

    X_processed, global_encoder, feature_names = preprocess_features(
        X_features, cat_feats, cont_feats, encoder=None, fit=True
    )

    oof_preds = {}
    fold_models = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_full)):
        X_train = X_processed.iloc[train_idx]
        y_train = y_full[train_idx]
        X_val = X_processed.iloc[val_idx]
        y_val = y_full[val_idx]
        val_pids = pids[val_idx]

        if len(set(y_train)) < 2:
            logger.warning(
                f"  Fold {fold_idx}: insufficient class diversity in train, skipping"
            )
            for pid in val_pids:
                oof_preds[pid] = 0.0
            continue

        if X_train.shape[1] != X_val.shape[1]:
            logger.error(
                f"  Fold {fold_idx}: Shape mismatch! Train={X_train.shape}, Val={X_val.shape}"
            )
            for pid in val_pids:
                oof_preds[pid] = 0.0
            continue

        ebm = ExplainableBoostingClassifier(
            feature_names=feature_names,
            **ebm_params,
        )
        ebm.fit(X_train, y_train)

        y_proba = ebm.predict_proba(X_val)[:, 1]

        for pid, prob in zip(val_pids, y_proba):
            oof_preds[pid] = float(prob)

        fold_models.append((ebm, global_encoder, cat_feats, cont_feats))

    return oof_preds, fold_models, cat_feats, cont_feats


def _model_filename(masking_hours: float) -> str:
    """Generate a stable filename for a given interval's deployment model."""
    label = _format_hours(masking_hours)
    return f"ebm_model_{label}.pkl"


def _completed_intervals(preds: dict) -> set:
    """Return the set of masking_hours that have predictions for at least one PID."""
    if not preds:
        return set()
    all_intervals: set = set()
    for pid_dict in preds.values():
        all_intervals.update(pid_dict.keys())
    return all_intervals


def generate_ebm_feature(
    cfg_dict: dict,
    save_dir: str = "data/interim/ebm_features",
    models_dir: str = "models/ebm",
    n_folds: Optional[int] = None,
    ebm_params: Optional[dict] = None,
) -> dict:

    if n_folds is None:
        n_folds = cfg_dict.get("ebm_feature", {}).get("n_folds", 5)
    if ebm_params is None:
        ebm_params = _get_default_ebm_params()

    logger.info("=" * 80)
    logger.info("GENERATING EBM FEATURE PREDICTIONS")
    logger.info("=" * 80)

    base_df_full = get_base_df()
    trainval_df, holdout_df = get_train_test_split(cfg_dict, base_df_full)

    logger.info(f"Trainval: {len(trainval_df)} patients")
    logger.info(f"Holdout:  {len(holdout_df)} patients")

    intervals = generate_ebm_intervals(cfg_dict)
    logger.info(f"EBM intervals: {len(intervals)} time points")
    logger.info(f"  Range: {intervals[0]:.2f}h to {intervals[-1]:.1f}h")

    # Pre-load concept data once to avoid re-reading pkls for every interval.
    # This is the main performance optimization: concept loading + filtering
    # (including prehospital data concat) happens once instead of ~3× per interval.
    logger.info("Pre-loading concept data for trainval and holdout...")
    trainval_concept_cache = preload_concept_cache(trainval_df, cfg_dict)
    holdout_concept_cache = preload_concept_cache(holdout_df, cfg_dict)
    logger.info(f"Cached {len(trainval_concept_cache)} trainval + {len(holdout_concept_cache)} holdout concepts")

    # Determine reference feature set from the latest interval (maximum masking time).
    # This ensures all EBM models share the same feature space regardless of how
    # sparse data is at early time points. Features that don't exist at a given
    # time point are zero-filled, so the EBM can still accept them at inference.
    logger.info("Determining reference feature set from latest interval...")
    _, _, ref_cat_feats, ref_cont_feats = _create_aggregated_dataset(
        trainval_df, cfg_dict, intervals[-1],
        concept_cache=trainval_concept_cache,
    )
    logger.info(
        f"Reference features: {len(ref_cont_feats)} cont + {len(ref_cat_feats)} cat "
        f"(from {_format_hours(intervals[-1])} masking point)"
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    predictions_path = os.path.join(save_dir, "ebm_predictions.pkl")

    # Load existing progress if available
    trainval_preds = {}
    holdout_preds = {}

    if os.path.exists(predictions_path):
        with open(predictions_path, "rb") as f:
            existing = pickle.load(f)
        trainval_preds = existing.get("trainval", {})
        holdout_preds = existing.get("holdout", {})
        logger.info(f"Loaded existing predictions with {len(_completed_intervals(trainval_preds))} intervals")

    # Determine which intervals are already done (predictions + model file both exist)
    pred_intervals = _completed_intervals(trainval_preds)
    model_intervals = {
        h for h in intervals
        if os.path.exists(os.path.join(models_dir, _model_filename(h)))
    }
    completed = pred_intervals & model_intervals
    if completed:
        logger.info(f"Found {len(completed)} completed intervals, will resume from where we left off")

    failed_intervals = []

    for i, masking_hours in enumerate(intervals):
        label = _format_hours(masking_hours)

        if masking_hours in completed:
            logger.info(f"[{i + 1}/{len(intervals)}] Skipping {label} (already done)")
            continue

        logger.info(f"\n[{i + 1}/{len(intervals)}] Training EBMs at {label}...")

        try:
            # Build trainval dataset ONCE for this interval (shared by
            # deployment model + K-fold training).
            X_trainval, y_trainval, cat_feats_tv, cont_feats_tv = \
                _create_aggregated_dataset(
                    trainval_df, cfg_dict, masking_hours,
                    concept_cache=trainval_concept_cache,
                )

            # Train deployment model FIRST so holdout predictions use the same
            # model that inference will load, eliminating model-identity divergence.
            final_model_dict = _train_deployment_from_dataset(
                X_trainval, y_trainval, cat_feats_tv, cont_feats_tv,
                cfg_dict, ebm_params,
                ref_cat_feats=ref_cat_feats, ref_cont_feats=ref_cont_feats,
            )
            # Save deployment model as individual file
            model_path = os.path.join(models_dir, _model_filename(masking_hours))
            with open(model_path, "wb") as f:
                pickle.dump(final_model_dict, f)

            # Holdout predictions using deployment model (matches inference)
            hold_preds = predict_holdout_with_deployment_model(
                holdout_df, cfg_dict, masking_hours, final_model_dict,
                ref_cat_feats=ref_cat_feats, ref_cont_feats=ref_cont_feats,
                concept_cache=holdout_concept_cache,
            )

            # K-fold training for trainval OOF (must stay K-fold to avoid leakage)
            oof_preds, fold_models, expected_cat_feats, expected_cont_feats = \
                _train_kfold_from_dataset(
                    X_trainval, y_trainval, cat_feats_tv, cont_feats_tv,
                    cfg_dict, masking_hours,
                    n_folds=n_folds, ebm_params=ebm_params,
                    ref_cat_feats=ref_cat_feats, ref_cont_feats=ref_cont_feats,
                )

            for pid, prob in oof_preds.items():
                trainval_preds.setdefault(pid, {})[masking_hours] = prob
            for pid, prob in hold_preds.items():
                holdout_preds.setdefault(pid, {})[masking_hours] = prob

            logger.info(
                f"  OOF mean={np.mean(list(oof_preds.values())):.3f}, "
                f"holdout mean={np.mean(list(hold_preds.values())):.3f}"
            )

            # Save predictions progress
            result = {
                "intervals_hours": intervals,
                "trainval": trainval_preds,
                "holdout": holdout_preds,
            }
            with open(predictions_path, "wb") as f:
                pickle.dump(result, f)

            n_done = len([h for h in intervals if os.path.exists(os.path.join(models_dir, _model_filename(h)))])
            logger.info(f"  Progress saved ({n_done}/{len(intervals)} intervals)")

        except Exception as e:
            logger.error(f"  Failed at {label}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_intervals.append(masking_hours)
            continue

    # Final save (ensures intervals_hours is up to date)
    result = {
        "intervals_hours": intervals,
        "trainval": trainval_preds,
        "holdout": holdout_preds,
    }
    with open(predictions_path, "wb") as f:
        pickle.dump(result, f)

    n_models = len([h for h in intervals if os.path.exists(os.path.join(models_dir, _model_filename(h)))])
    logger.info(f"\n{'=' * 80}")
    logger.info(f"EBM feature generation complete")
    logger.info(f"  Saved predictions to: {predictions_path}")
    logger.info(f"  Saved {n_models} deployment models to: {models_dir}/")
    logger.info("=" * 80)

    return result


def load_ebm_predictions(save_dir: str = "data/interim/ebm_features") -> dict:
    """Load cached EBM predictions from pickle."""
    load_path = os.path.join(save_dir, "ebm_predictions.pkl")
    with open(load_path, "rb") as f:
        return pickle.load(f)


def _format_hours(h: float) -> str:
    """Format hours to readable string."""
    if h < 1:
        return f"{h * 60:.0f}min"
    elif h < 24:
        return f"{h:.0f}h" if h == int(h) else f"{h:.1f}h"
    else:
        days = h / 24
        return f"{days:.0f}D" if days == int(days) else f"{days:.1f}D"


def main():
    parser = argparse.ArgumentParser(
        description="Generate EBM predictions for hybrid model feature"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/interim/ebm_features",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/ebm",
        help="Directory to save deployment models",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds for OOF predictions",
    )
    args = parser.parse_args()

    generate_ebm_feature(cfg, save_dir=args.save_dir, models_dir=args.models_dir, n_folds=args.n_folds)


if __name__ == "__main__":
    main()