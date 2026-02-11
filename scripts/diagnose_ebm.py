"""
Diagnostic script for EBM feature integration in the hybrid model.

Runs 5 checks to identify why the EBM prediction channel does not improve
model performance:
  1. EBM channel value distribution
  2. PID alignment spot-check
  3. Per-channel scale comparison
  4. EBM standalone predictive quality (AUROC)
  5. W_P gradient flow analysis

Usage:
    python scripts/diagnose_ebm.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from astra.utils import get_cfg, logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_padding_mask(X_raw, traj_lengths):
    """Create a boolean mask [n_samples, seq_len] for non-padding positions."""
    n_samples, _, seq_len = X_raw.shape
    mask = np.zeros((n_samples, seq_len), dtype=bool)
    for i, length in enumerate(traj_lengths):
        mask[i, :length] = True
    return mask


def _section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Check 1: EBM Channel Value Distribution
# ---------------------------------------------------------------------------

def check_ebm_distribution(data):
    _section("CHECK 1: EBM Channel Value Distribution")

    idx = data["ebm_channel_idx"]
    if idx is None:
        print("FAIL: ebm_channel_idx is None — EBM feature not enabled in config")
        return

    for label, X_raw, X_norm, traj_lens in [
        ("Trainval", data["X_raw"], data["X"], data["trajectory_lengths"]),
        ("Holdout", data["tX_raw"], data["tX"], data["holdout_trajectory_lengths"]),
    ]:
        print(f"\n--- {label} ---")
        ebm_raw = X_raw[:, idx, :]
        ebm_norm = X_norm[:, idx, :]
        pos_mask = _non_padding_mask(X_raw, traj_lens)

        raw_vals = ebm_raw[pos_mask]
        norm_vals = ebm_norm[pos_mask]

        n_total = raw_vals.size
        n_nonzero = np.count_nonzero(raw_vals)
        pct_nonzero = 100.0 * n_nonzero / n_total if n_total > 0 else 0

        print(f"  Raw values (within non-padding):")
        print(f"    n_positions   = {n_total}")
        print(f"    n_nonzero     = {n_nonzero} ({pct_nonzero:.1f}%)")
        print(f"    mean          = {raw_vals.mean():.6f}")
        print(f"    std           = {raw_vals.std():.6f}")
        print(f"    min / max     = {raw_vals.min():.6f} / {raw_vals.max():.6f}")

        # Histogram of non-zero values
        nz = raw_vals[raw_vals != 0]
        if len(nz) > 0:
            hist, edges = np.histogram(nz, bins=10)
            print(f"    histogram (non-zero):")
            for i in range(len(hist)):
                bar = "#" * min(40, int(40 * hist[i] / max(hist.max(), 1)))
                print(f"      [{edges[i]:.3f}, {edges[i+1]:.3f}) {hist[i]:>7d} {bar}")

        # Compare raw vs normalized
        raw_eq_norm = np.allclose(raw_vals, norm_vals, atol=1e-6)
        print(f"  Normalized == Raw?  {'YES (restore worked)' if raw_eq_norm else 'NO (normalized differently)'}")
        if not raw_eq_norm:
            print(f"    Normalized: mean={norm_vals.mean():.4f}, std={norm_vals.std():.4f}")

        if pct_nonzero < 20:
            print(f"  ** WARNING: Only {pct_nonzero:.1f}% non-zero — EBM channel is mostly empty")


# ---------------------------------------------------------------------------
# Check 2: PID Alignment Spot-Check
# ---------------------------------------------------------------------------

def check_pid_alignment(data, cfg):
    _section("CHECK 2: PID Alignment Spot-Check")

    idx = data["ebm_channel_idx"]
    if idx is None:
        print("SKIP: ebm_channel_idx is None")
        return

    ebm_save_dir = cfg.get("ebm_feature", {}).get("save_dir", "data/interim/ebm_features")
    pred_path = os.path.join(ebm_save_dir, "ebm_predictions.pkl")
    if not os.path.exists(pred_path):
        print(f"SKIP: {pred_path} not found")
        return

    with open(pred_path, "rb") as f:
        ebm_preds = pickle.load(f)

    intervals = ebm_preds["intervals_hours"]

    for label, split_key, X_raw, traj_lens, complete_df in [
        ("Trainval", "trainval", data["X_raw"], data["trajectory_lengths"],
         data["trainval"].complete),
        ("Holdout", "holdout", data["tX_raw"], data["holdout_trajectory_lengths"],
         data["holdout"].complete),
    ]:
        print(f"\n--- {label} ---")
        preds_dict = ebm_preds[split_key]

        # df2xy sorts by PID → sample order = sorted unique PIDs
        sorted_pids = sorted(complete_df["PID"].unique())
        n_samples = X_raw.shape[0]

        print(f"  PIDs in X tensor:     {n_samples}")
        print(f"  PIDs in predictions:  {len(preds_dict)}")

        # Check coverage
        pids_in_preds = set(preds_dict.keys())
        pids_in_data = set(sorted_pids)
        missing_from_preds = pids_in_data - pids_in_preds
        extra_in_preds = pids_in_preds - pids_in_data
        if missing_from_preds:
            print(f"  WARNING: {len(missing_from_preds)} PIDs in data but NOT in predictions")
        if extra_in_preds:
            print(f"  WARNING: {len(extra_in_preds)} PIDs in predictions but NOT in data")

        # Spot-check 10 random PIDs
        rng = np.random.RandomState(42)
        check_pids = rng.choice(sorted_pids, size=min(10, len(sorted_pids)), replace=False)
        n_match = 0
        n_checked = 0

        for pid in check_pids:
            if pid not in preds_dict or not preds_dict[pid]:
                continue
            n_checked += 1
            sample_idx = sorted_pids.index(pid)

            # Get last available interval prediction
            patient_intervals = sorted(preds_dict[pid].keys())
            last_interval = patient_intervals[-1]
            expected_val = preds_dict[pid][last_interval]

            # Find that value in the X tensor — search backward from end of trajectory
            ebm_row = X_raw[sample_idx, idx, :]
            traj_len = traj_lens[sample_idx]
            actual_vals = ebm_row[:traj_len]

            # The last non-zero value should correspond to the latest prediction
            nonzero_positions = np.where(actual_vals != 0)[0]
            if len(nonzero_positions) == 0:
                print(f"  PID {pid}: X tensor has ALL zeros (traj_len={traj_len}), "
                      f"expected {expected_val:.4f}")
                continue

            last_val = actual_vals[nonzero_positions[-1]]
            match = np.isclose(last_val, expected_val, atol=1e-4)
            n_match += int(match)
            status = "OK" if match else "MISMATCH"
            if not match:
                print(f"  PID {pid}: {status} — X[{nonzero_positions[-1]}]={last_val:.6f} "
                      f"vs pkl={expected_val:.6f}")

        if n_checked > 0:
            print(f"  Spot-check: {n_match}/{n_checked} PIDs match exactly")
            if n_match == n_checked:
                print(f"  PASS: All sampled PIDs aligned correctly")
            else:
                print(f"  FAIL: PID misalignment detected!")
        else:
            print(f"  SKIP: No PIDs with predictions available for spot-check")


# ---------------------------------------------------------------------------
# Check 3: Per-Channel Scale Comparison
# ---------------------------------------------------------------------------

def check_channel_scales(data):
    _section("CHECK 3: Per-Channel Scale Comparison")

    idx = data["ebm_channel_idx"]
    X = data["X"]
    X_raw = data["X_raw"]
    traj_lens = data["trajectory_lengths"]
    pos_mask = _non_padding_mask(X_raw, traj_lens)

    feature_names = sorted(data["trainval"].complete["FEATURE"].unique())
    n_channels = X.shape[1]

    print(f"\n  {'Channel':<30s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'range':>8s} {'%nz':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    other_stds = []

    for ch in range(n_channels):
        ch_data = X[:, ch, :][pos_mask]
        name = feature_names[ch] if ch < len(feature_names) else f"ch_{ch}"
        ch_mean = ch_data.mean()
        ch_std = ch_data.std()
        ch_min = ch_data.min()
        ch_max = ch_data.max()
        ch_range = ch_max - ch_min
        pct_nz = 100.0 * np.count_nonzero(ch_data) / max(ch_data.size, 1)

        marker = " <-- EBM" if ch == idx else ""
        print(f"  {name:<30s} {ch_mean:>8.4f} {ch_std:>8.4f} {ch_min:>8.4f} "
              f"{ch_max:>8.4f} {ch_range:>8.4f} {pct_nz:>5.1f}%{marker}")

        if ch != idx:
            other_stds.append(ch_std)

    if idx is not None and other_stds:
        ebm_data = X[:, idx, :][pos_mask]
        ebm_std = ebm_data.std()
        avg_other_std = np.mean(other_stds)
        ratio = avg_other_std / ebm_std if ebm_std > 1e-8 else float("inf")
        print(f"\n  Scale mismatch ratio: avg_other_std / ebm_std = {avg_other_std:.4f} / {ebm_std:.4f} = {ratio:.1f}x")
        if ratio > 5:
            print(f"  CONCERN: EBM channel has {ratio:.0f}x smaller dynamic range than average channel")
        else:
            print(f"  OK: Scale mismatch is moderate ({ratio:.1f}x)")


# ---------------------------------------------------------------------------
# Check 4: EBM Standalone Predictive Quality (AUROC)
# ---------------------------------------------------------------------------

def check_ebm_auroc(data, cfg):
    _section("CHECK 4: EBM Standalone Predictive Quality")

    ebm_save_dir = cfg.get("ebm_feature", {}).get("save_dir", "data/interim/ebm_features")
    pred_path = os.path.join(ebm_save_dir, "ebm_predictions.pkl")
    if not os.path.exists(pred_path):
        print(f"SKIP: {pred_path} not found")
        return

    with open(pred_path, "rb") as f:
        ebm_preds = pickle.load(f)

    intervals = ebm_preds["intervals_hours"]
    target = cfg["target"]

    for label, split_key, y, complete_df in [
        ("Trainval (OOF)", "trainval", data["y"], data["trainval"].complete),
        ("Holdout", "holdout", data["ty"], data["holdout"].complete),
    ]:
        print(f"\n--- {label} ---")
        preds_dict = ebm_preds[split_key]

        sorted_pids = sorted(complete_df["PID"].unique())
        y_array = np.array(y)

        # Latest-prediction AUROC
        latest_preds = []
        valid_mask = []
        for i, pid in enumerate(sorted_pids):
            patient_preds = preds_dict.get(pid, {})
            if patient_preds:
                latest_interval = max(patient_preds.keys())
                latest_preds.append(patient_preds[latest_interval])
                valid_mask.append(True)
            else:
                latest_preds.append(0.5)  # uninformative default
                valid_mask.append(False)

        latest_preds = np.array(latest_preds)
        valid_mask = np.array(valid_mask)

        n_valid = valid_mask.sum()
        print(f"  Patients with predictions: {n_valid}/{len(sorted_pids)}")

        if n_valid > 10 and len(np.unique(y_array[valid_mask])) > 1:
            auroc = roc_auc_score(y_array[valid_mask], latest_preds[valid_mask])
            print(f"  Latest-prediction AUROC: {auroc:.4f}")
        else:
            print(f"  Cannot compute AUROC (insufficient data or single class)")

        # Time-slice AUROCs
        time_points = [1, 6, 24, 72]
        print(f"\n  Time-slice AUROCs:")
        for t_hours in time_points:
            # Find the closest interval <= t_hours
            valid_intervals = [h for h in intervals if h <= t_hours]
            if not valid_intervals:
                print(f"    {t_hours:>4d}h: no EBM interval available")
                continue
            closest = max(valid_intervals)

            slice_preds = []
            slice_valid = []
            for i, pid in enumerate(sorted_pids):
                patient_preds = preds_dict.get(pid, {})
                if closest in patient_preds:
                    slice_preds.append(patient_preds[closest])
                    slice_valid.append(True)
                else:
                    slice_preds.append(0.5)
                    slice_valid.append(False)

            slice_preds = np.array(slice_preds)
            slice_valid = np.array(slice_valid)
            n_v = slice_valid.sum()

            if n_v > 10 and len(np.unique(y_array[slice_valid])) > 1:
                auroc = roc_auc_score(y_array[slice_valid], slice_preds[slice_valid])
                print(f"    {t_hours:>4d}h (interval={closest:.1f}h): AUROC={auroc:.4f}  (n={n_v})")
            else:
                print(f"    {t_hours:>4d}h: insufficient data (n_valid={n_v})")


# ---------------------------------------------------------------------------
# Check 5: W_P Gradient Flow
# ---------------------------------------------------------------------------

def check_gradient_flow(data, cfg):
    _section("CHECK 5: W_P Gradient Flow")

    idx = data["ebm_channel_idx"]
    if idx is None:
        print("SKIP: ebm_channel_idx is None")
        return

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("SKIP: torch not available")
        return

    try:
        from astra.training.finetune import load_pretrained_backbone
    except ImportError:
        print("SKIP: could not import load_pretrained_backbone")
        return

    checkpoint_dir = f'./pretrain_checkpoints/{cfg["model_name"]}'
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"SKIP: no pretrained checkpoint at {checkpoint_path}")
        return

    print("  Loading pretrained backbone with W_P expansion...")
    try:
        backbone = load_pretrained_backbone(data, cfg)
    except Exception as e:
        print(f"  SKIP: could not load backbone: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device)
    backbone.train()

    # Get one batch from the mixed dataloader
    mixed_dls = data["mixed_dls"]
    batch = next(iter(mixed_dls.train))
    inputs, targets = batch

    # Move to device
    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, (tuple, list)):
            return type(obj)(to_device(item) for item in obj)
        return obj

    inputs = to_device(inputs)
    targets = to_device(targets)

    # Forward + backward
    logits = backbone(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # Extract W_P gradients per channel
    w_p_grad = backbone.W_P.weight.grad  # [out_ch, in_ch, 1]
    if w_p_grad is None:
        print("  FAIL: W_P has no gradient (not in computation graph?)")
        return

    n_channels = w_p_grad.shape[1]
    feature_names = sorted(data["trainval"].complete["FEATURE"].unique())

    print(f"\n  {'Channel':<30s} {'grad_L2':>10s}")
    print(f"  {'-'*30} {'-'*10}")

    grad_norms = []
    for ch in range(n_channels):
        norm = w_p_grad[:, ch, :].norm().item()
        grad_norms.append(norm)
        name = feature_names[ch] if ch < len(feature_names) else f"ch_{ch}"
        marker = " <-- EBM" if ch == idx else ""
        print(f"  {name:<30s} {norm:>10.6f}{marker}")

    ebm_norm = grad_norms[idx]
    other_norms = [n for i, n in enumerate(grad_norms) if i != idx]
    avg_other = np.mean(other_norms) if other_norms else 0

    ratio = avg_other / ebm_norm if ebm_norm > 1e-10 else float("inf")
    print(f"\n  EBM grad norm:     {ebm_norm:.6f}")
    print(f"  Avg other norm:    {avg_other:.6f}")
    print(f"  Ratio (other/ebm): {ratio:.1f}x")

    if ratio > 5:
        print(f"  CONCERN: EBM receives {ratio:.0f}x weaker gradients — scale mismatch confirmed")
    else:
        print(f"  OK: Gradient ratio is moderate ({ratio:.1f}x)")

    backbone.zero_grad()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = get_cfg()

    if not cfg.get("ebm_feature", {}).get("enabled", False):
        print("ERROR: ebm_feature.enabled is False in config. "
              "Set it to true and ensure EBM predictions exist.")
        sys.exit(1)

    print("Loading data with prepare_data_and_dls(cfg)...")
    from astra.data.dataloader import prepare_data_and_dls
    data = prepare_data_and_dls(cfg)

    idx = data.get("ebm_channel_idx")
    print(f"EBM channel index: {idx}")
    print(f"X shape: {data['X'].shape} (trainval), {data['tX'].shape} (holdout)")

    check_ebm_distribution(data)
    check_pid_alignment(data, cfg)
    check_channel_scales(data)
    check_ebm_auroc(data, cfg)
    check_gradient_flow(data, cfg)

    _section("SUMMARY")
    print("  Review the output above to determine:")
    print("  - Check 1+3: Is there a scale mismatch? (most likely cause)")
    print("  - Check 2:   Are PIDs aligned correctly?")
    print("  - Check 4:   Is the EBM signal actually predictive?")
    print("  - Check 5:   Does the gradient reach the EBM channel?")
    print()


if __name__ == "__main__":
    main()
