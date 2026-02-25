"""
Diagnostic script for EBM integration + general standardization analysis.

Checks:
  1. EBM signal preservation — does AUROC/AUPRC survive standardization?
  2. EBM distribution before/after standardization
  3. General standardization quality — all channels
  4. Early information density — what data is available at 1h, 6h?
  5. PID alignment (corrected — uses elapsed_hours matching)

Usage:
    python scripts/diagnose_ebm.py
"""

import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from astra.utils import get_bin_df, get_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_padding_mask(X_raw, traj_lengths):
    """Boolean mask [n_samples, seq_len] for non-padding positions."""
    n_samples, _, seq_len = X_raw.shape
    mask = np.zeros((n_samples, seq_len), dtype=bool)
    for i, length in enumerate(traj_lengths):
        mask[i, :length] = True
    return mask


def _section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def _get_bin_elapsed_hours(cfg, base_df):
    """Get elapsed hours per bin position per patient (same as ebm_features.py)."""
    bin_df = get_bin_df()
    bin_freq_include = cfg.get("bin_freq_include", [])
    base_pids = set(base_df["PID"].unique())
    bf = bin_df[
        (bin_df["PID"].isin(base_pids))
        & (bin_df["bin_freq"].isin(bin_freq_include))
    ].copy()
    bf = bf.merge(base_df[["PID", "start"]], on="PID", how="left")
    bf = bf.sort_values(["PID", "bin_counter"])
    bf["position"] = bf.groupby("PID").cumcount()
    bf["elapsed_hours"] = (
        (bf["bin_start"] - bf["start"]).dt.total_seconds() / 3600
        + (bf["bin_end"] - bf["bin_start"]).dt.total_seconds() / 7200
    )
    return bf[["PID", "position", "elapsed_hours"]]


def _find_bin_position_at_hours(bin_elapsed_df, pid, target_hours):
    """Find the bin position closest to (but not exceeding) target_hours."""
    patient = bin_elapsed_df[bin_elapsed_df["PID"] == pid]
    valid = patient[patient["elapsed_hours"] <= target_hours]
    if len(valid) == 0:
        return None, None
    row = valid.iloc[-1]
    return int(row["position"]), row["elapsed_hours"]


# ---------------------------------------------------------------------------
# Check 1: EBM Signal Preservation (AUROC/AUPRC from X tensor)
# ---------------------------------------------------------------------------

def check_signal_preservation(data, cfg):
    _section("CHECK 1: EBM Signal Preservation After Standardization")
    print("  Does AUROC/AUPRC survive from pkl → X_raw → X_normalized?")

    idx = data["ebm_channel_idx"]
    if idx is None:
        print("  SKIP: ebm_channel_idx is None")
        return

    ebm_save_dir = cfg.get("ebm_feature", {}).get("save_dir", "data/interim/ebm_features")
    pred_path = os.path.join(ebm_save_dir, "ebm_predictions.pkl")
    with open(pred_path, "rb") as f:
        ebm_preds = pickle.load(f)
    intervals = ebm_preds["intervals_hours"]

    for label, split_key, X_raw, X_norm, y, traj_lens, complete_df, base_df in [
        ("Trainval", "trainval", data["X_raw"], data["X"],
         data["y"], data["trajectory_lengths"],
         data["trainval"].complete, data["trainval"].base),
        ("Holdout", "holdout", data["tX_raw"], data["tX"],
         data["ty"], data["holdout_trajectory_lengths"],
         data["holdout"].complete, data["holdout"].base),
    ]:
        print(f"\n--- {label} ---")
        preds_dict = ebm_preds[split_key]
        sorted_pids = sorted(complete_df["PID"].unique())
        y_arr = np.array(y)
        prevalence = y_arr.mean()

        # Compute bin elapsed hours for position→time mapping
        bin_elapsed = _get_bin_elapsed_hours(cfg, base_df)

        print(f"  Prevalence: {prevalence:.3f} ({prevalence*100:.1f}%)")
        print()

        time_points = [1, 6, 24, 72]
        header = f"  {'Source':<25s} {'Time':>6s} {'AUROC':>8s} {'AUPRC':>8s}  {'n':>6s}"
        print(header)
        print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8}  {'-'*6}")

        for t_hours in time_points:
            # --- PKL source (ground truth) ---
            valid_intervals = [h for h in intervals if h <= t_hours]
            if not valid_intervals:
                continue
            closest_interval = max(valid_intervals)

            pkl_preds, pkl_valid = [], []
            for i, pid in enumerate(sorted_pids):
                pp = preds_dict.get(pid, {})
                if closest_interval in pp:
                    pkl_preds.append(pp[closest_interval])
                    pkl_valid.append(True)
                else:
                    pkl_preds.append(0.0)
                    pkl_valid.append(False)

            pkl_preds = np.array(pkl_preds)
            pkl_valid = np.array(pkl_valid)

            if pkl_valid.sum() > 10 and len(np.unique(y_arr[pkl_valid])) > 1:
                pkl_auroc = roc_auc_score(y_arr[pkl_valid], pkl_preds[pkl_valid])
                pkl_auprc = average_precision_score(y_arr[pkl_valid], pkl_preds[pkl_valid])
            else:
                pkl_auroc = pkl_auprc = float("nan")

            # --- X_raw source (before normalization) ---
            raw_preds, raw_valid = [], []
            for i, pid in enumerate(sorted_pids):
                pos, _ = _find_bin_position_at_hours(bin_elapsed, pid, t_hours)
                if pos is not None and pos < X_raw.shape[2]:
                    val = X_raw[i, idx, pos]
                    raw_preds.append(val)
                    raw_valid.append(val != 0.0)  # non-zero means has prediction
                else:
                    raw_preds.append(0.0)
                    raw_valid.append(False)

            raw_preds = np.array(raw_preds)
            raw_valid = np.array(raw_valid)

            if raw_valid.sum() > 10 and len(np.unique(y_arr[raw_valid])) > 1:
                raw_auroc = roc_auc_score(y_arr[raw_valid], raw_preds[raw_valid])
                raw_auprc = average_precision_score(y_arr[raw_valid], raw_preds[raw_valid])
            else:
                raw_auroc = raw_auprc = float("nan")

            # --- X_normalized source (after standardization) ---
            norm_preds, norm_valid = [], []
            for i, pid in enumerate(sorted_pids):
                pos, _ = _find_bin_position_at_hours(bin_elapsed, pid, t_hours)
                if pos is not None and pos < X_norm.shape[2]:
                    val = X_norm[i, idx, pos]
                    norm_preds.append(val)
                    norm_valid.append(val != 0.0)
                else:
                    norm_preds.append(0.0)
                    norm_valid.append(False)

            norm_preds = np.array(norm_preds)
            norm_valid = np.array(norm_valid)

            if norm_valid.sum() > 10 and len(np.unique(y_arr[norm_valid])) > 1:
                norm_auroc = roc_auc_score(y_arr[norm_valid], norm_preds[norm_valid])
                norm_auprc = average_precision_score(y_arr[norm_valid], norm_preds[norm_valid])
            else:
                norm_auroc = norm_auprc = float("nan")

            n_pkl = int(pkl_valid.sum())
            n_raw = int(raw_valid.sum())
            n_norm = int(norm_valid.sum())

            print(f"  {'pkl (ground truth)':<25s} {t_hours:>5d}h {pkl_auroc:>8.4f} {pkl_auprc:>8.4f}  {n_pkl:>6d}")
            print(f"  {'X_raw (pre-norm)':<25s} {t_hours:>5d}h {raw_auroc:>8.4f} {raw_auprc:>8.4f}  {n_raw:>6d}")
            print(f"  {'X_normalized (post-norm)':<25s} {t_hours:>5d}h {norm_auroc:>8.4f} {norm_auprc:>8.4f}  {n_norm:>6d}")
            print()

        # Also: latest-timestep for each patient
        latest_raw, latest_norm, latest_valid = [], [], []
        for i in range(len(sorted_pids)):
            tlen = traj_lens[i]
            if tlen > 0:
                r = X_raw[i, idx, tlen - 1]
                n = X_norm[i, idx, tlen - 1]
                latest_raw.append(r)
                latest_norm.append(n)
                latest_valid.append(r != 0.0)
            else:
                latest_raw.append(0.0)
                latest_norm.append(0.0)
                latest_valid.append(False)

        latest_raw = np.array(latest_raw)
        latest_norm = np.array(latest_norm)
        latest_valid = np.array(latest_valid)

        if latest_valid.sum() > 10:
            lr_auroc = roc_auc_score(y_arr[latest_valid], latest_raw[latest_valid])
            lr_auprc = average_precision_score(y_arr[latest_valid], latest_raw[latest_valid])
            ln_auroc = roc_auc_score(y_arr[latest_valid], latest_norm[latest_valid])
            ln_auprc = average_precision_score(y_arr[latest_valid], latest_norm[latest_valid])
            print(f"  {'X_raw (last timestep)':<25s} {'max':>6s} {lr_auroc:>8.4f} {lr_auprc:>8.4f}  {int(latest_valid.sum()):>6d}")
            print(f"  {'X_norm (last timestep)':<25s} {'max':>6s} {ln_auroc:>8.4f} {ln_auprc:>8.4f}  {int(latest_valid.sum()):>6d}")


# ---------------------------------------------------------------------------
# Check 2: EBM Distribution (pre/post standardization)
# ---------------------------------------------------------------------------

def check_ebm_distribution(data):
    _section("CHECK 2: EBM Channel Distribution (Raw vs Standardized)")

    idx = data["ebm_channel_idx"]
    if idx is None:
        print("  SKIP: ebm_channel_idx is None")
        return

    for label, X_raw, X_norm, traj_lens in [
        ("Trainval", data["X_raw"], data["X"], data["trajectory_lengths"]),
        ("Holdout", data["tX_raw"], data["tX"], data["holdout_trajectory_lengths"]),
    ]:
        print(f"\n--- {label} ---")
        pos_mask = _non_padding_mask(X_raw, traj_lens)

        raw_vals = X_raw[:, idx, :][pos_mask]
        norm_vals = X_norm[:, idx, :][pos_mask]

        raw_nz = raw_vals[raw_vals != 0]
        norm_nz = norm_vals[norm_vals != 0]

        print(f"  Non-padding positions: {len(raw_vals)}, non-zero: {len(raw_nz)} ({100*len(raw_nz)/len(raw_vals):.1f}%)")
        print()
        print(f"  {'Metric':<20s} {'Raw':>12s} {'Standardized':>12s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        print(f"  {'mean (non-zero)':<20s} {raw_nz.mean():>12.4f} {norm_nz.mean():>12.4f}")
        print(f"  {'std (non-zero)':<20s} {raw_nz.std():>12.4f} {norm_nz.std():>12.4f}")
        print(f"  {'p5':<20s} {np.percentile(raw_nz, 5):>12.4f} {np.percentile(norm_nz, 5):>12.4f}")
        print(f"  {'p25':<20s} {np.percentile(raw_nz, 25):>12.4f} {np.percentile(norm_nz, 25):>12.4f}")
        print(f"  {'p50 (median)':<20s} {np.percentile(raw_nz, 50):>12.4f} {np.percentile(norm_nz, 50):>12.4f}")
        print(f"  {'p75':<20s} {np.percentile(raw_nz, 75):>12.4f} {np.percentile(norm_nz, 75):>12.4f}")
        print(f"  {'p95':<20s} {np.percentile(raw_nz, 95):>12.4f} {np.percentile(norm_nz, 95):>12.4f}")
        print(f"  {'min':<20s} {raw_nz.min():>12.6f} {norm_nz.min():>12.6f}")
        print(f"  {'max':<20s} {raw_nz.max():>12.6f} {norm_nz.max():>12.6f}")

        # Skewness
        raw_skew = ((raw_nz - raw_nz.mean()) ** 3).mean() / (raw_nz.std() ** 3)
        norm_skew = ((norm_nz - norm_nz.mean()) ** 3).mean() / (norm_nz.std() ** 3)
        print(f"  {'skewness':<20s} {raw_skew:>12.2f} {norm_skew:>12.2f}")


# ---------------------------------------------------------------------------
# Check 3: General Standardization Quality
# ---------------------------------------------------------------------------

def check_general_standardization(data):
    _section("CHECK 3: General Standardization — All Channels")
    print("  Shows data density and distribution for each channel.")
    print("  Key concern: 0.0 = 'missing data' = 'padding' — model can't distinguish.")

    X_raw = data["X_raw"]
    X_norm = data["X"]
    traj_lens = data["trajectory_lengths"]
    pos_mask = _non_padding_mask(X_raw, traj_lens)
    n_positions = pos_mask.sum()

    feature_names = sorted(data["trainval"].complete["FEATURE"].unique())
    n_channels = X_raw.shape[1]
    ebm_idx = data.get("ebm_channel_idx")

    print(f"\n  Total non-padding positions per channel: {n_positions}")
    print()

    header = (f"  {'Channel':<26s} {'%data':>6s} {'raw_mean':>9s} {'raw_std':>9s} "
              f"{'norm_mean':>10s} {'norm_std':>9s} {'skew':>6s} {'min':>7s} {'max':>7s}")
    print(header)
    print(f"  {'-'*26} {'-'*6} {'-'*9} {'-'*9} {'-'*10} {'-'*9} {'-'*6} {'-'*7} {'-'*7}")

    channel_info = []
    for ch in range(n_channels):
        name = feature_names[ch] if ch < len(feature_names) else f"ch_{ch}"

        raw = X_raw[:, ch, :][pos_mask]
        norm = X_norm[:, ch, :][pos_mask]

        pct_nz = 100.0 * np.count_nonzero(raw) / len(raw)
        raw_mean = raw[raw != 0].mean() if np.count_nonzero(raw) > 0 else 0
        raw_std = raw[raw != 0].std() if np.count_nonzero(raw) > 1 else 0

        # Normalized stats (including zeros as-is, since that's what the model sees)
        norm_mean = norm.mean()  # model sees this distribution (zeros included)
        norm_std = norm.std()

        # Skewness of non-zero normalized values
        nz_norm = norm[norm != 0]
        if len(nz_norm) > 2:
            skew = ((nz_norm - nz_norm.mean()) ** 3).mean() / max(nz_norm.std() ** 3, 1e-10)
        else:
            skew = 0

        marker = " *" if ch == ebm_idx else ""
        print(f"  {name:<26s} {pct_nz:>5.1f}% {raw_mean:>9.3f} {raw_std:>9.3f} "
              f"{norm_mean:>10.4f} {norm_std:>9.4f} {skew:>6.1f} {norm.min():>7.2f} {norm.max():>7.2f}{marker}")

        channel_info.append({
            "name": name, "pct_data": pct_nz, "raw_mean": raw_mean,
            "raw_std": raw_std, "norm_mean": norm_mean, "norm_std": norm_std,
        })

    # Summary statistics
    print(f"\n  * = EBM channel")
    print()

    sparse = [c for c in channel_info if c["pct_data"] < 10]
    dense = [c for c in channel_info if c["pct_data"] >= 25]
    print(f"  Sparse channels (<10% data): {len(sparse)}")
    for c in sparse:
        print(f"    {c['name']:<26s} {c['pct_data']:.1f}% data")
    print(f"  Dense channels (>=25% data): {len(dense)}")
    for c in dense:
        print(f"    {c['name']:<26s} {c['pct_data']:.1f}% data")

    print()
    print("  Issue: Sparse channels have 90%+ zero values that look like padding")
    print("  to the model. The model cannot distinguish 'no measurement' from")
    print("  'end of trajectory'. This affects channels like TEG, LACTATE, etc.")


# ---------------------------------------------------------------------------
# Check 4: Early Information Density
# ---------------------------------------------------------------------------

def check_early_information(data, cfg):
    _section("CHECK 4: Early Information Density")
    print("  What data does the model actually have at 1h, 6h?")
    print("  This matters because EBM dominates AUPRC at early time points.")

    X_raw = data["X_raw"]
    traj_lens = data["trajectory_lengths"]
    feature_names = sorted(data["trainval"].complete["FEATURE"].unique())
    n_channels = X_raw.shape[1]
    n_samples = X_raw.shape[0]
    ebm_idx = data.get("ebm_channel_idx")

    # Get bin elapsed hours
    bin_elapsed = _get_bin_elapsed_hours(cfg, data["trainval"].base)

    # For each time cutoff, find the bin position range
    sorted_pids = sorted(data["trainval"].complete["PID"].unique())

    for t_hours in [1, 6, 24]:
        print(f"\n  --- At {t_hours}h ---")

        # Find the max bin position at this time for each patient
        positions = []
        for pid in sorted_pids:
            pos, _ = _find_bin_position_at_hours(bin_elapsed, pid, t_hours)
            positions.append(pos if pos is not None else 0)
        positions = np.array(positions)

        avg_pos = positions.mean()
        print(f"  Average bin positions available: {avg_pos:.1f}")

        # For each channel, check how many patients have ANY non-zero data up to this time
        print(f"\n  {'Channel':<26s} {'%patients':>10s} {'mean_nz':>10s}")
        print(f"  {'-'*26} {'-'*10} {'-'*10}")

        for ch in range(n_channels):
            name = feature_names[ch] if ch < len(feature_names) else f"ch_{ch}"
            n_with_data = 0
            nz_values = []
            for i in range(n_samples):
                max_pos = min(positions[i] + 1, X_raw.shape[2])
                window = X_raw[i, ch, :max_pos]
                nz = window[window != 0]
                if len(nz) > 0:
                    n_with_data += 1
                    nz_values.extend(nz.tolist())

            pct = 100.0 * n_with_data / n_samples
            mean_nz = np.mean(nz_values) if nz_values else 0

            marker = " <-- EBM" if ch == ebm_idx else ""
            if pct >= 5 or ch == ebm_idx:  # Only show channels with meaningful data
                print(f"  {name:<26s} {pct:>9.1f}% {mean_nz:>10.4f}{marker}")

        print(f"\n  (Channels with <5% patient coverage at {t_hours}h omitted)")


# ---------------------------------------------------------------------------
# Check 5: Corrected PID Alignment
# ---------------------------------------------------------------------------

def check_pid_alignment_corrected(data, cfg):
    _section("CHECK 5: PID Alignment (Corrected — elapsed_hours matching)")
    print("  Previous check compared wrong time points. This version matches")
    print("  X tensor values to pkl predictions at the SAME elapsed_hours.")

    idx = data["ebm_channel_idx"]
    if idx is None:
        print("  SKIP: ebm_channel_idx is None")
        return

    ebm_save_dir = cfg.get("ebm_feature", {}).get("save_dir", "data/interim/ebm_features")
    pred_path = os.path.join(ebm_save_dir, "ebm_predictions.pkl")
    if not os.path.exists(pred_path):
        print(f"  SKIP: {pred_path} not found")
        return

    with open(pred_path, "rb") as f:
        ebm_preds = pickle.load(f)
    intervals = sorted(ebm_preds["intervals_hours"])

    for label, split_key, X_raw, traj_lens, complete_df, base_df in [
        ("Trainval", "trainval", data["X_raw"], data["trajectory_lengths"],
         data["trainval"].complete, data["trainval"].base),
        ("Holdout", "holdout", data["tX_raw"], data["holdout_trajectory_lengths"],
         data["holdout"].complete, data["holdout"].base),
    ]:
        print(f"\n--- {label} ---")
        preds_dict = ebm_preds[split_key]
        sorted_pids = sorted(complete_df["PID"].unique())

        bin_elapsed = _get_bin_elapsed_hours(cfg, base_df)

        # Test at multiple known EBM intervals
        test_intervals = [0.5, 1.0, 6.0, 24.0]  # 30min, 1h, 6h, 24h
        test_intervals = [t for t in test_intervals if t in intervals]

        rng = np.random.RandomState(42)
        check_pids = rng.choice(sorted_pids, size=min(10, len(sorted_pids)), replace=False)

        n_match = 0
        n_checked = 0

        for interval_h in test_intervals[:2]:  # Check 2 intervals to keep output manageable
            print(f"\n  Interval: {interval_h}h")
            for pid in check_pids:
                if pid not in preds_dict or interval_h not in preds_dict[pid]:
                    continue
                n_checked += 1

                expected_val = preds_dict[pid][interval_h]
                sample_idx = sorted_pids.index(pid)

                # Find the bin position whose elapsed_hours corresponds to this interval
                # The forward-fill assigns the prediction at interval_h to bins where
                # elapsed_hours >= interval_h (until the next interval)
                patient_bins = bin_elapsed[bin_elapsed["PID"] == pid].sort_values("position")
                # Find first bin where elapsed_hours >= interval_h
                valid_bins = patient_bins[patient_bins["elapsed_hours"] >= interval_h]
                if len(valid_bins) == 0:
                    continue

                check_pos = int(valid_bins.iloc[0]["position"])
                if check_pos >= X_raw.shape[2]:
                    continue

                actual_val = X_raw[sample_idx, idx, check_pos]
                match = np.isclose(actual_val, expected_val, atol=1e-4)
                n_match += int(match)
                status = "OK" if match else "MISMATCH"

                if not match:
                    print(f"    PID {pid}: {status} — X[{check_pos}]={actual_val:.6f} "
                          f"vs pkl[{interval_h}h]={expected_val:.6f}")

        if n_checked > 0:
            pct = 100.0 * n_match / n_checked
            print(f"\n  Result: {n_match}/{n_checked} ({pct:.0f}%) match exactly")
            if pct >= 90:
                print("  PASS: PID alignment is correct")
            else:
                print("  FAIL: Significant misalignment detected!")
        else:
            print("  SKIP: Could not check any PID/interval combinations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = get_cfg()

    if not cfg.get("ebm_feature", {}).get("enabled", False):
        print("ERROR: ebm_feature.enabled is False in config.")
        sys.exit(1)

    print("Loading data with prepare_data_and_dls_cached(cfg)...")
    from astra.data.caching import prepare_data_and_dls_cached
    data = prepare_data_and_dls_cached(cfg)

    idx = data.get("ebm_channel_idx")
    print(f"EBM channel index: {idx}")
    print(f"X shape: {data['X'].shape} (trainval), {data['tX'].shape} (holdout)")

    check_signal_preservation(data, cfg)
    check_ebm_distribution(data)
    check_general_standardization(data)
    check_early_information(data, cfg)
    check_pid_alignment_corrected(data, cfg)

    _section("INTERPRETATION GUIDE")
    print("""
  CHECK 1 — Signal Preservation:
    If pkl AUROC ≈ X_raw AUROC ≈ X_norm AUROC: standardization preserves signal.
    If X_norm < X_raw: standardization is DISTORTING the signal.
    If X_raw < pkl: forward-fill mapping is LOSING information.
    AUPRC is critical at 5% prevalence — compare to hybrid model's AUPRC.

  CHECK 2 — Distribution:
    High skewness after standardization means heavy right tail (few high-risk).
    Model sees mostly values in a narrow band with rare extreme spikes.

  CHECK 3 — General Standardization:
    Sparse channels (<10% data) are 90%+ zeros that look like padding.
    The model CANNOT distinguish 'no lab drawn' from 'trajectory ended'.
    norm_mean far from 0: the zero-overloading shifts the apparent mean.

  CHECK 4 — Early Information:
    At 1h, which channels actually have data for most patients?
    If only EBM and HR/SBP have data, the model has very few features to work with.
    This tells us if the early-prediction gap is about data availability.

  CHECK 5 — PID Alignment:
    Tests whether the forward-fill correctly placed pkl predictions into X.
    If alignment fails: the EBM signal is corrupted at the data level.
    """)


if __name__ == "__main__":
    main()