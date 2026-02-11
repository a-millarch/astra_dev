"""
Align EBM predictions to hybrid model bin positions as a time series channel.

Creates a wide-format DataFrame with FEATURE="_ebm_pred" that can be injected
into the data pipeline alongside other continuous concepts.

Forward-fills the most recent EBM prediction at each bin position to ensure
no temporal data leakage.
"""

import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from astra.utils import get_bin_df, logger


def _compute_bin_elapsed_hours(
    base_df: pd.DataFrame,
    bin_freq_include: list,
) -> pd.DataFrame:
    """
    Compute elapsed_hours for each bin position for each patient.

    Reuses the same logic as _create_temporal_features_df in datasets.py.

    Returns:
        DataFrame with columns: ['PID', 'position', 'elapsed_hours']
    """
    bin_df = get_bin_df()

    base_pids = set(base_df["PID"].unique())
    bf = bin_df[
        (bin_df["PID"].isin(base_pids))
        & (bin_df["bin_freq"].isin(bin_freq_include))
    ].copy()

    bf = bf.merge(base_df[["PID", "start"]], on="PID", how="left")

    # Sort and assign sequential position per patient (0-indexed)
    bf = bf.sort_values(["PID", "bin_counter"])
    bf["position"] = bf.groupby("PID").cumcount()

    # Elapsed hours at bin midpoint (same formula as _create_temporal_features_df)
    bf["elapsed_hours"] = (
        (bf["bin_start"] - bf["start"]).dt.total_seconds() / 3600
        + (bf["bin_end"] - bf["bin_start"]).dt.total_seconds() / 7200
    )

    return bf[["PID", "position", "elapsed_hours"]]


def _forward_fill_predictions(
    patient_elapsed: np.ndarray,
    ebm_intervals_hours: list,
    patient_preds: Dict[float, float],
    default_value: float = 0.0,
) -> np.ndarray:
    """
    Forward-fill EBM predictions to bin positions for one patient.

    For each bin position with elapsed_hours `t`, assigns the prediction
    from the most recent EBM interval where masking_hours <= t.

    Args:
        patient_elapsed: Array of elapsed_hours per position.
        ebm_intervals_hours: Sorted list of EBM masking times in hours.
        patient_preds: {masking_hours: predicted_probability} for this patient.
        default_value: Value for bins before the first EBM interval.

    Returns:
        Array of prediction values per position.
    """
    intervals = np.array(ebm_intervals_hours)
    result = np.full(len(patient_elapsed), default_value)

    for i, elapsed_h in enumerate(patient_elapsed):
        # Find the largest interval <= elapsed_h
        valid_mask = intervals <= elapsed_h
        if valid_mask.any():
            best_interval = intervals[valid_mask].max()
            if best_interval in patient_preds:
                result[i] = patient_preds[best_interval]
            else:
                # Interval exists but no prediction (EBM training may have failed)
                # Fall back to the next most recent available prediction
                available = sorted(
                    [h for h in patient_preds.keys() if h <= elapsed_h],
                    reverse=True,
                )
                if available:
                    result[i] = patient_preds[available[0]]

    return result


def create_ebm_feature_df(
    cfg: dict,
    base_df: pd.DataFrame,
    split: str = "trainval",
    ebm_predictions: Optional[dict] = None,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create wide-format DataFrame for the _ebm_pred feature channel.

    Matches the schema produced by _create_temporal_features_df / _get_long_concept_df_single_label:
    Columns: [PID, FEATURE, target, '0', '1', '2', ...]

    Args:
        cfg: Configuration dictionary.
        base_df: Base DataFrame for this split (trainval or holdout patients).
        split: "trainval" or "holdout" — selects which predictions to use.
        ebm_predictions: Pre-loaded predictions dict (if None, loads from disk).
        save_dir: Directory containing ebm_predictions.pkl.

    Returns:
        Wide-format DataFrame with FEATURE="_ebm_pred".
    """
    if save_dir is None:
        save_dir = cfg.get("ebm_feature", {}).get(
            "save_dir", "data/interim/ebm_features"
        )

    # Load predictions
    if ebm_predictions is None:
        pred_path = os.path.join(save_dir, "ebm_predictions.pkl")
        with open(pred_path, "rb") as f:
            ebm_predictions = pickle.load(f)

    intervals_hours = ebm_predictions["intervals_hours"]
    preds_dict = ebm_predictions[split]  # {pid: {hours: pred}}
    default_value = cfg.get("ebm_feature", {}).get("default_value", 0.0)
    target = cfg["target"]
    bin_freq_include = cfg.get("bin_freq_include", [])

    # Compute elapsed hours per bin position per patient
    bin_elapsed = _compute_bin_elapsed_hours(base_df, bin_freq_include)
    max_pos = bin_elapsed["position"].max()
    ts_cols = [str(i) for i in range(max_pos + 1)]

    all_pids = base_df["PID"].unique()
    rows = []

    for pid in all_pids:
        patient_bins = bin_elapsed[bin_elapsed["PID"] == pid].sort_values("position")

        if len(patient_bins) == 0:
            # No bin data for this patient — fill with default
            values = [default_value] * (max_pos + 1)
        else:
            patient_elapsed = patient_bins["elapsed_hours"].values
            patient_positions = patient_bins["position"].values.astype(int)

            patient_preds = preds_dict.get(pid, {})

            # Forward-fill predictions for positions with data
            filled = _forward_fill_predictions(
                patient_elapsed, intervals_hours, patient_preds, default_value
            )

            # Create full-length array with 0.0 padding
            values = [0.0] * (max_pos + 1)
            for pos, val in zip(patient_positions, filled):
                values[pos] = val

        row = {"PID": pid, "FEATURE": "_ebm_pred"}
        row.update({ts_cols[i]: values[i] for i in range(max_pos + 1)})
        rows.append(row)

    result = pd.DataFrame(rows)

    # Merge target
    result = result.merge(base_df[["PID", target]], on="PID", how="left")
    result[target] = result[target].astype(int)
    result = result.sort_values(["PID", "FEATURE"]).reset_index(drop=True)

    logger.info(
        f"Created EBM feature channel ({split}): "
        f"{len(result)} rows, {max_pos + 1} timesteps, "
        f"value range [{result[ts_cols].min().min():.3f}, {result[ts_cols].max().max():.3f}]"
    )

    return result
