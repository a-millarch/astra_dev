"""
Align EBM predictions to hybrid model bin positions as a time series channel.

Creates a wide-format DataFrame with FEATURE="_ebm_pred" that can be injected
into the data pipeline alongside other continuous concepts.

Forward-fills the most recent EBM prediction at each bin position to ensure
no temporal data leakage.
"""

import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from astra.utils import get_bin_df

logger = logging.getLogger(__name__)


def _compute_bin_elapsed_hours(
    base_df: pd.DataFrame,
    bin_freq_include: list,
) -> pd.DataFrame:
    """
    Compute elapsed_hours at bin_start for each bin position for each patient.

    Uses bin_start (not midpoint) so that the forward-fill comparison
    ``masking_hours <= elapsed_hours`` is strictly conservative: an EBM
    trained at masking time M is only applied from the first bin that STARTS
    at or after M, guaranteeing the bin's entire data window is posterior to
    the EBM's training cutoff.

    Returns:
        DataFrame with columns: ['PID', 'position', 'elapsed_hours']
    """
    bin_df = get_bin_df()

    base_pids = set(base_df["PID"].unique())
    bf = bin_df[
        (bin_df["PID"].isin(base_pids))
        & (bin_df["bin_freq"].isin(bin_freq_include))
    ].copy()

    # 'start' is the universal earliest timestamp (incorporates prehospital when available)
    merge_cols = ["PID", "start"]
    bf = bf.merge(base_df[merge_cols], on="PID", how="left")

    # Sort and assign sequential position per patient (0-indexed)
    bf = bf.sort_values(["PID", "bin_counter"])
    bf["position"] = bf.groupby("PID").cumcount()

    # Elapsed hours at bin_start — used for causal EBM assignment.
    # Note: datasets.py uses midpoint for positional encoding; here we use
    # bin_start so masking_hours <= elapsed_hours means the EBM predates
    # the entire bin, not just its midpoint.
    ref_start = bf["start"]
    bf["elapsed_hours"] = (bf["bin_start"] - ref_start).dt.total_seconds() / 3600

    return bf[["PID", "position", "elapsed_hours"]]


def _forward_fill_predictions(
    patient_elapsed: np.ndarray,
    ebm_intervals_hours: list,
    patient_preds: Dict[float, float],
    default_value: float = np.nan,
) -> np.ndarray:
    """
    Forward-fill EBM predictions to bin positions for one patient.

    For each bin position with elapsed_hours `t` (= bin_start in hours),
    assigns the prediction from the most recent EBM interval where
    masking_hours <= t.  Because `t` is the bin_start, this guarantees
    the chosen EBM's training window does not overlap with the bin at all.

    Args:
        patient_elapsed: Array of elapsed_hours per position.
        ebm_intervals_hours: Sorted list of EBM masking times in hours.
        patient_preds: {masking_hours: predicted_probability} for this patient.
        default_value: Value for bins before the first EBM interval (default:
            NaN so that normalization treats them as missing rather than as a
            spurious "0% risk" measurement).

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
    target = cfg["target"]
    bin_freq_include = cfg.get("bin_freq_include", [])

    # Compute elapsed hours (at bin_start) per bin position per patient
    bin_elapsed = _compute_bin_elapsed_hours(base_df, bin_freq_include)
    max_pos = bin_elapsed["position"].max()
    ts_cols = [str(i) for i in range(max_pos + 1)]

    all_pids = base_df["PID"].unique()
    rows = []

    for pid in all_pids:
        patient_bins = bin_elapsed[bin_elapsed["PID"] == pid].sort_values("position")

        if len(patient_bins) == 0:
            # No bin data for this patient — all padding (0.0)
            values = [0.0] * (max_pos + 1)
        else:
            patient_elapsed = patient_bins["elapsed_hours"].values
            patient_positions = patient_bins["position"].values.astype(int)

            patient_preds = preds_dict.get(pid, {})

            # Forward-fill predictions for positions within the trajectory.
            # default_value=NaN marks pre-EBM steps as missing (not "0% risk"),
            # so normalize_with_padding_mask treats them correctly.
            filled = _forward_fill_predictions(
                patient_elapsed, intervals_hours, patient_preds, default_value=np.nan
            )

            # Trailing positions beyond trajectory end stay 0.0 (padding).
            # Within-trajectory positions use the forward-filled value (NaN for
            # pre-EBM steps, a probability otherwise).
            values = [0.0] * (max_pos + 1)
            for pos, val in zip(patient_positions, filled):
                values[pos] = val  # NaN propagates correctly here

        row = {"PID": pid, "FEATURE": "_ebm_pred"}
        row.update({ts_cols[i]: values[i] for i in range(max_pos + 1)})
        rows.append(row)

    result = pd.DataFrame(rows)

    # Merge target
    result = result.merge(base_df[["PID", target]], on="PID", how="left")
    result[target] = result[target].astype(int)
    result = result.sort_values(["PID", "FEATURE"]).reset_index(drop=True)

    # Logging: ignore NaN pre-EBM positions and 0.0 padding in range report
    ebm_vals = result[ts_cols].values.ravel()
    ebm_measured = ebm_vals[~np.isnan(ebm_vals) & (ebm_vals != 0.0)]
    val_range = (
        f"[{ebm_measured.min():.3f}, {ebm_measured.max():.3f}]"
        if len(ebm_measured) > 0 else "N/A"
    )
    logger.info(
        f"Created EBM feature channel ({split}): "
        f"{len(result)} rows, {max_pos + 1} timesteps, "
        f"EBM probability range {val_range}"
    )

    return result
