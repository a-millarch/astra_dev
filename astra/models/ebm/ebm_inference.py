import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from generate_ebm_feature import (
    _create_aggregated_dataset,
    _model_filename,
    generate_ebm_intervals,
    preprocess_features,
)

logger = logging.getLogger(__name__)


def get_available_intervals(
    cfg_dict: dict,
    models_dir: str = "models/ebm",
) -> List[float]:
    """Return sorted list of intervals that have a saved deployment model."""
    if not os.path.isdir(models_dir):
        return []
    intervals = generate_ebm_intervals(cfg_dict)
    return [h for h in intervals if os.path.exists(os.path.join(models_dir, _model_filename(h)))]


def find_nearest_interval(
    elapsed_hours: float,
    cfg_dict: dict,
    models_dir: str = "models/ebm",
) -> Optional[float]:
    """Find the closest available interval <= elapsed_hours."""
    available = get_available_intervals(cfg_dict, models_dir)
    candidates = [h for h in available if h <= elapsed_hours]
    return max(candidates) if candidates else None


def load_deployment_model(
    masking_hours: float,
    models_dir: str = "models/ebm",
) -> dict:
    """Load a single deployment model for a given interval."""
    model_path = os.path.join(models_dir, _model_filename(masking_hours))
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _predict_at_interval(
    base_df,
    cfg_dict: dict,
    masking_hours: float,
    model_dict: dict,
) -> Dict[int, float]:
    """Run prediction for a single interval using a loaded model_dict."""
    X_full, _, _, _ = _create_aggregated_dataset(base_df, cfg_dict, masking_hours)

    id_col = cfg_dict["dataset"]["id_col"]
    pids = X_full[id_col].values
    X_features = X_full.drop(columns=[id_col])

    X_processed, _, _ = preprocess_features(
        X_features,
        model_dict["expected_cat_feats"],
        model_dict["expected_cont_feats"],
        encoder=model_dict["encoder"],
        fit=False,
        expected_cat_feats=model_dict["expected_cat_feats"],
        expected_cont_feats=model_dict["expected_cont_feats"],
    )

    probs = model_dict["model"].predict_proba(X_processed)[:, 1]
    return {pid: float(prob) for pid, prob in zip(pids, probs)}


def infer_at_interval(
    new_base_df,
    cfg_dict: dict,
    masking_hours: float,
    models_dir: str = "models/ebm",
) -> Dict[int, float]:
    """
    Generate EBM predictions for new patients at a single time interval.

    Loads only the model needed, keeping memory usage low.

    Returns:
        {PID: predicted_probability}
    """
    model_dict = load_deployment_model(masking_hours, models_dir)
    return _predict_at_interval(new_base_df, cfg_dict, masking_hours, model_dict)


def infer_new_patients(
    new_base_df,
    cfg_dict: dict,
    models_dir: str = "models/ebm",
    intervals: Optional[List[float]] = None,
) -> Dict[int, Dict[float, float]]:
    """
    Generate EBM interval predictions for new patients across all (or specified) intervals.

    Models are loaded one at a time to avoid holding all in memory.

    Args:
        new_base_df: Base DataFrame for new patients.
        cfg_dict: Configuration dictionary.
        models_dir: Directory containing saved models.
        intervals: Specific intervals to predict at. If None, uses all available.

    Returns:
        {PID: {masking_hours: probability}}
    """
    if intervals is None:
        intervals = get_available_intervals(cfg_dict, models_dir)

    results: Dict[int, Dict[float, float]] = {}

    for masking_hours in intervals:
        model_dict = load_deployment_model(masking_hours, models_dir)
        preds = _predict_at_interval(new_base_df, cfg_dict, masking_hours, model_dict)

        for pid, prob in preds.items():
            results.setdefault(pid, {})[masking_hours] = prob

    return results