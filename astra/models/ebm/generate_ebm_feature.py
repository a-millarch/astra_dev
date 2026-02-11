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
) -> Tuple[pd.DataFrame, Optional[OneHotEncoder], list]:
    """
    Preprocess features with explicit one-hot encoding for categoricals.
    
    Args:
        X: Feature dataframe (no ID column)
        cat_feats: Categorical feature names
        cont_feats: Continuous feature names
        encoder: Fitted encoder (for transform-only mode)
        fit: Whether to fit encoder (True for train, False for val/test)
    
    Returns:
        X_processed: DataFrame with all continuous features
        encoder: Fitted encoder (if fit=True) or None
        feature_names: List of all feature names
    """
    # Separate categorical and continuous
    X_cat = X[cat_feats] if cat_feats else pd.DataFrame(index=X.index)
    X_cont = X[cont_feats] if cont_feats else pd.DataFrame(index=X.index)
    
    if len(cat_feats) > 0:
        if fit:
            # Fit encoder on full data
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',  # Critical: ignore unknown categories
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
        else:
            # Transform using fitted encoder
            if encoder is None:
                raise ValueError("Encoder must be provided when fit=False")
            
            X_cat_encoded = encoder.transform(X_cat)
            
            # Use same feature names from encoder
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
    
    return X_processed, encoder, feature_names


def train_ebm_kfold_at_timepoint(
    train_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    n_folds: int = 5,
    ebm_params: Optional[dict] = None,
) -> Tuple[Dict[int, float], list]:
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
        fold_models: List of (model, encoder) tuples for holdout prediction.
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

        # Store model with encoder for holdout prediction
        fold_models.append((ebm, global_encoder))

    return oof_preds, fold_models


def predict_holdout(
    holdout_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    fold_models: list,
) -> Dict[int, float]:
    """
    Generate averaged predictions for holdout patients across K fold models.
    
    **FIX**: Uses stored encoders from training to ensure consistent features.

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
    
    for model, encoder in fold_models:
        # **FIX**: Use the encoder from training to preprocess holdout data
        X_processed, _, _ = preprocess_features(
            X_features, cat_feats, cont_feats, encoder=encoder, fit=False
        )
        
        all_proba += model.predict_proba(X_processed)[:, 1]
    
    all_proba /= len(fold_models)

    return {pid: float(prob) for pid, prob in zip(pids, all_proba)}


def generate_ebm_feature(
    cfg_dict: dict,
    save_dir: str = "data/interim/ebm_features",
    n_folds: Optional[int] = None,
    ebm_params: Optional[dict] = None,
) -> dict:
    """
    Main orchestration: train EBMs at all intervals and generate predictions.

    Saves predictions as pickle with structure:
    {
        "intervals_hours": [0.167, 0.5, 1.0, ...],
        "trainval": {pid: {hours: pred, ...}, ...},
        "holdout": {pid: {hours: pred, ...}, ...},
    }

    Args:
        cfg_dict: Configuration dictionary.
        save_dir: Directory to save predictions.
        n_folds: Number of CV folds (default: from config or 5).
        ebm_params: EBM hyperparameters (default: standard params).

    Returns:
        The predictions dict.
    """
    if n_folds is None:
        n_folds = cfg_dict.get("ebm_feature", {}).get("n_folds", 5)
    if ebm_params is None:
        ebm_params = _get_default_ebm_params()

    logger.info("=" * 80)
    logger.info("GENERATING EBM FEATURE PREDICTIONS")
    logger.info("=" * 80)

    # Load and split data (same split as hybrid model)
    base_df_full = get_base_df()
    trainval_df, holdout_df = get_train_test_split(cfg_dict, base_df_full)

    logger.info(f"Trainval: {len(trainval_df)} patients")
    logger.info(f"Holdout:  {len(holdout_df)} patients")

    # Generate intervals
    intervals = generate_ebm_intervals(cfg_dict)
    logger.info(f"EBM intervals: {len(intervals)} time points")
    logger.info(f"  Range: {intervals[0]:.2f}h to {intervals[-1]:.1f}h")

    # Initialize prediction storage
    trainval_preds = {}  # {pid: {hours: pred}}
    holdout_preds = {}   # {pid: {hours: pred}}

    failed_intervals = []

    for i, masking_hours in enumerate(intervals):
        label = _format_hours(masking_hours)
        logger.info(f"\n[{i + 1}/{len(intervals)}] Training EBMs at {label}...")

        try:
            # K-fold OOF predictions for trainval
            oof_preds, fold_models = train_ebm_kfold_at_timepoint(
                trainval_df, cfg_dict, masking_hours,
                n_folds=n_folds, ebm_params=ebm_params,
            )

            # Averaged predictions for holdout
            hold_preds = predict_holdout(
                holdout_df, cfg_dict, masking_hours, fold_models,
            )

            # Store predictions
            for pid, prob in oof_preds.items():
                trainval_preds.setdefault(pid, {})[masking_hours] = prob
            for pid, prob in hold_preds.items():
                holdout_preds.setdefault(pid, {})[masking_hours] = prob

            logger.info(
                f"  OOF mean={np.mean(list(oof_preds.values())):.3f}, "
                f"holdout mean={np.mean(list(hold_preds.values())):.3f}"
            )

        except Exception as e:
            logger.error(f"  Failed at {label}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_intervals.append(masking_hours)
            continue

    # Save
    os.makedirs(save_dir, exist_ok=True)
    result = {
        "intervals_hours": intervals,
        "trainval": trainval_preds,
        "holdout": holdout_preds,
    }

    save_path = os.path.join(save_dir, "ebm_predictions.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"EBM feature generation complete")
    logger.info(f"  Successful: {len(intervals) - len(failed_intervals)}/{len(intervals)}")
    logger.info(f"  Saved to: {save_path}")
    if failed_intervals:
        logger.info(f"  Failed intervals: {[_format_hours(h) for h in failed_intervals]}")
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
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds for OOF predictions",
    )
    args = parser.parse_args()

    generate_ebm_feature(cfg, save_dir=args.save_dir, n_folds=args.n_folds)


if __name__ == "__main__":
    main()