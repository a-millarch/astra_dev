import os
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from interpret.glassbox import ExplainableBoostingClassifier

from astra.utils import get_base_df, get_train_test_split, cfg, logger
from astra.data.datasets import AggregatedDS


def generate_ebm_intervals(cfg_dict: dict) -> List[float]:
    """
    Generate EBM training intervals in hours.

    Schedule:
    - Early: 10min, 30min, 1h, 2h, 3h, 4h
    - 4h-72h: every 2 hours
    - Post-72h: match cfg bin_intervals boundaries
      (daily 72h-7D, 2-day 7D-14D, 3-day 14D-30D)

    Returns:
        Sorted list of masking times in hours.
    """
    intervals = [10 / 60, 30 / 60, 1, 2, 3, 4]
    intervals += list(range(6, 73, 2))  # 6h to 72h, step 2h

    # Post-72h: derive from cfg bin_intervals
    bin_intervals = cfg_dict.get("bin_intervals", {})
    bin_freq_include = set(cfg_dict.get("bin_freq_include", []))

    sorted_keys = sorted(
        [k for k in bin_intervals.keys() if k != "end"],
        key=lambda k: float(k[:-1]) * (24 if k.endswith("D") else 1),
    )

    def _key_to_hours(k):
        return float(k[:-1]) * (24 if k.endswith("D") else 1)

    for i, key in enumerate(sorted_keys):
        end_h = _key_to_hours(key)

        if i > 0:
            start_h = _key_to_hours(sorted_keys[i - 1])
        else:
            start_h = 0

        if end_h <= 72:
            continue

        start_h = max(start_h, 72)

        resolution_str = bin_intervals[key]
        if resolution_str not in bin_freq_include:
            continue

        if resolution_str.endswith("min"):
            res_h = float(resolution_str[:-3]) / 60
        elif resolution_str.endswith("h"):
            res_h = float(resolution_str[:-1])
        elif resolution_str.endswith("D"):
            res_h = float(resolution_str[:-1]) * 24
        else:
            continue

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
) -> Tuple[pd.DataFrame, np.ndarray, list, list]:
    """
    Create AggregatedDS at a masking point and extract X, y with PIDs.

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
    )

    X, y = agg_ds.get_X_y(include_id=True)
    return X, np.asarray(y), agg_ds.categorical_features, agg_ds.continuous_features


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
            # Create DataFrame with ALL expected categorical features
            X_cat_full = pd.DataFrame(index=X.index)
            for feat in expected_cat_feats:
                if feat in X.columns:
                    X_cat_full[feat] = X[feat]  # Use actual data
                else:
                    X_cat_full[feat] = np.nan   # Add missing feature as NaN
            
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
        # Create DataFrame with ALL expected continuous features
        X_cont = pd.DataFrame(index=X.index)
        for feat in expected_cont_feats:
            if feat in X.columns:
                X_cont[feat] = X[feat]  # Use actual data
            else:
                X_cont[feat] = 0.0      # Add missing feature as 0
        
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
) -> Tuple[Dict[int, float], list, list, list]:
    """
    Train K-fold EBMs at one masking point to generate OOF predictions.
    
    **FIX**: Uses explicit preprocessing to ensure consistent features across folds.

    Args:
        train_df: Training patients base_df (no holdout).
        cfg_dict: Configuration dictionary.
        masking_hours: Time point in hours.
        n_folds: Number of CV folds.
        ebm_params: EBM hyperparameters.

    Returns:
        oof_preds: {PID: predicted_probability} for all training patients.
        fold_models: List of (model, encoder, expected_cat_feats, expected_cont_feats) tuples.
        expected_cat_feats: List of categorical feature names from training.
        expected_cont_feats: List of continuous feature names from training.
    """
    if ebm_params is None:
        ebm_params = _get_default_ebm_params()

    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        train_df, cfg_dict, masking_hours
    )

    id_col = cfg_dict["dataset"]["id_col"]
    pids = X_full[id_col].values
    X_features = X_full.drop(columns=[id_col])

    # **FIX**: Fit encoder on FULL dataset to ensure all categories are known
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
) -> Dict[int, float]:
    """
    Generate averaged predictions for holdout patients across K fold models.
    
    **FIX**: Uses stored encoders from training to ensure consistent features,
    handling missing categorical AND continuous columns by adding zero-filled columns.

    Args:
        holdout_df: Holdout patients base_df.
        cfg_dict: Configuration dictionary.
        masking_hours: Time point in hours.
        fold_models: List of (model, encoder, expected_cat_feats, expected_cont_feats) tuples.
        expected_cat_feats: Categorical features expected from training.
        expected_cont_feats: Continuous features expected from training.

    Returns:
        {PID: averaged_probability}
    """
    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        holdout_df, cfg_dict, masking_hours
    )

    id_col = cfg_dict["dataset"]["id_col"]
    pids = X_full[id_col].values
    X_features = X_full.drop(columns=[id_col])

    if len(fold_models) == 0:
        return {pid: 0.0 for pid in pids}

    # Average predictions across fold models
    all_proba = np.zeros(len(X_features))
    
    for model, encoder, _, _ in fold_models:
        # **FIX**: Use the encoder from training to preprocess holdout data
        # Pass expected_cat_feats and expected_cont_feats so missing features can be handled
        X_processed, _, _ = preprocess_features(
            X_features, cat_feats, cont_feats, 
            encoder=encoder, fit=False, 
            expected_cat_feats=expected_cat_feats,
            expected_cont_feats=expected_cont_feats
        )
        
        all_proba += model.predict_proba(X_processed)[:, 1]
    
    all_proba /= len(fold_models)

    return {pid: float(prob) for pid, prob in zip(pids, all_proba)}

def train_final_ebm_at_timepoint(
    train_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    ebm_params: dict,
):
    """
    Train ONE final EBM on full trainval data for deployment.
    """
    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        train_df, cfg_dict, masking_hours
    )

    id_col = cfg_dict["dataset"]["id_col"]
    X_features = X_full.drop(columns=[id_col])

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
            oof_preds, fold_models, expected_cat_feats, expected_cont_feats = \
                train_ebm_kfold_at_timepoint(
                    trainval_df, cfg_dict, masking_hours,
                    n_folds=n_folds, ebm_params=ebm_params,
                )

            hold_preds = predict_holdout(
                holdout_df, cfg_dict, masking_hours, fold_models,
                expected_cat_feats, expected_cont_feats,
            )

            # Train final deployment model on full trainval
            final_model_dict = train_final_ebm_at_timepoint(
                trainval_df,
                cfg_dict,
                masking_hours,
                ebm_params,
            )
            # Save deployment model as individual file
            model_path = os.path.join(models_dir, _model_filename(masking_hours))
            with open(model_path, "wb") as f:
                pickle.dump(final_model_dict, f)

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