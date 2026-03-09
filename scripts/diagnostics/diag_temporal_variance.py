#!/usr/bin/env python
"""
DIAGNOSTIC: Confirm temporal features are patient-invariant (zero discriminative signal).

Checks:
1. Cross-patient variance of each channel at each position
2. Temporal channels should have ~0 variance (same value for all patients)
3. Clinical channels should have high variance (different per patient)
4. Point-biserial correlation between temporal channels and outcome

Run on Azure:
    python -m scripts.diagnostics.diag_temporal_variance
"""
import logging
import numpy as np
from scipy.stats import pointbiserialr
from astra.utils import cfg
from astra.data.dataloader import prepare_data_and_dls

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 72)
    logger.info("DIAGNOSTIC: Temporal Feature Variance Analysis")
    logger.info("=" * 72)

    data = prepare_data_and_dls(cfg)

    X_raw = data["X_raw"]                      # pre-normalization
    X = data["X"]                              # post-normalization
    traj_lengths = data["trajectory_lengths"]
    ts_channel_names = data["ts_channel_names"]
    y = np.array(data["y"])

    n_samples, n_channels, seq_len = X_raw.shape

    tf_features = cfg.get("temporal_features", {}).get("features", [])
    temporal_idx = [i for i, n in enumerate(ts_channel_names) if n in tf_features]
    clinical_idx = [i for i, n in enumerate(ts_channel_names) if n not in tf_features]

    logger.info(f"Samples: {n_samples}, Channels: {n_channels}, Seq len: {seq_len}")
    logger.info(f"Temporal features: {[ts_channel_names[i] for i in temporal_idx]}")
    logger.info(f"Clinical channels: {len(clinical_idx)}")

    # ------------------------------------------------------------------ #
    # CHECK 1: Cross-patient variance at each position (RAW, pre-norm)
    # ------------------------------------------------------------------ #
    logger.info(f"\n[CHECK 1] Cross-patient variance per position (raw values):")
    logger.info(f"  Temporal features should have ~0 variance if patient-invariant.\n")

    positions_to_check = [0, 1, 5, 10, 20, 35, 50, 75, 100]
    positions_to_check = [p for p in positions_to_check if p < seq_len]

    # Temporal channels
    for ch_idx in temporal_idx:
        ch_name = ts_channel_names[ch_idx]
        logger.info(f"  [{ch_name}] (TEMPORAL):")
        for pos in positions_to_check:
            vals = []
            for i in range(n_samples):
                if pos < traj_lengths[i]:
                    v = X_raw[i, ch_idx, pos]
                    if not np.isnan(v):
                        vals.append(v)
            if len(vals) > 10:
                var = np.var(vals)
                mean = np.mean(vals)
                logger.info(f"    pos {pos:>4d}: mean={mean:>10.4f}, "
                            f"var={var:>12.8f}, n={len(vals):>5d}"
                            f"  {'<-- INVARIANT' if var < 1e-6 else ''}")
            else:
                logger.info(f"    pos {pos:>4d}: insufficient data (n={len(vals)})")

    # Sample 5 clinical channels for comparison
    sample_clinical = clinical_idx[:5]
    for ch_idx in sample_clinical:
        ch_name = ts_channel_names[ch_idx]
        logger.info(f"\n  [{ch_name}] (CLINICAL):")
        for pos in positions_to_check[:5]:
            vals = []
            for i in range(n_samples):
                if pos < traj_lengths[i]:
                    v = X_raw[i, ch_idx, pos]
                    if not np.isnan(v) and abs(v) > 1e-8:
                        vals.append(v)
            if len(vals) > 10:
                var = np.var(vals)
                mean = np.mean(vals)
                logger.info(f"    pos {pos:>4d}: mean={mean:>10.4f}, "
                            f"var={var:>12.4f}, n={len(vals):>5d}")
            else:
                logger.info(f"    pos {pos:>4d}: insufficient data (n={len(vals)})")

    # ------------------------------------------------------------------ #
    # CHECK 2: Summary statistics — variance ratio
    # ------------------------------------------------------------------ #
    logger.info(f"\n[CHECK 2] Average cross-patient variance (first 50 positions):")

    channel_avg_vars = {}
    for ch_idx in range(n_channels):
        ch_name = ts_channel_names[ch_idx]
        position_vars = []
        for pos in range(min(50, seq_len)):
            vals = []
            for i in range(n_samples):
                if pos < traj_lengths[i]:
                    v = X_raw[i, ch_idx, pos]
                    if not np.isnan(v) and abs(v) > 1e-8:
                        vals.append(v)
            if len(vals) > 10:
                position_vars.append(np.var(vals))
        if position_vars:
            channel_avg_vars[ch_name] = np.mean(position_vars)

    temporal_vars = [channel_avg_vars.get(ts_channel_names[i], 0) for i in temporal_idx]
    clinical_vars = [channel_avg_vars.get(ts_channel_names[i], 0) for i in clinical_idx
                     if ts_channel_names[i] in channel_avg_vars]

    if temporal_vars:
        logger.info(f"  Temporal channels avg variance: {np.mean(temporal_vars):.8f}")
    if clinical_vars:
        logger.info(f"  Clinical channels avg variance: {np.mean(clinical_vars):.4f}")
    if temporal_vars and clinical_vars:
        ratio = np.mean(clinical_vars) / max(np.mean(temporal_vars), 1e-12)
        logger.info(f"  Ratio (clinical/temporal): {ratio:.0f}x")
        if np.mean(temporal_vars) < 1e-6:
            logger.info(f"\n  ** CONFIRMED: Temporal features are patient-INVARIANT.")
            logger.info(f"     They carry ZERO patient-discriminative information.")
            logger.info(f"     They are a deterministic function of the position index.")

    # ------------------------------------------------------------------ #
    # CHECK 3: Correlation with outcome
    # ------------------------------------------------------------------ #
    logger.info(f"\n[CHECK 3] Point-biserial correlation (temporal channels vs outcome):")
    logger.info(f"  If temporal features are patient-invariant, correlation should be ~0")
    logger.info(f"  at each position (since all patients have the same value).\n")

    for ch_idx in temporal_idx:
        ch_name = ts_channel_names[ch_idx]
        logger.info(f"  [{ch_name}]:")
        for pos in [5, 20, 35, 50, 75]:
            if pos >= seq_len:
                continue
            vals = []
            labels = []
            for i in range(n_samples):
                if pos < traj_lengths[i]:
                    v = X[i, ch_idx, pos]  # normalized
                    if abs(v) > 1e-8:
                        vals.append(v)
                        labels.append(y[i])
            if len(set(labels)) > 1 and len(vals) > 50:
                corr, pval = pointbiserialr(labels, vals)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                logger.info(f"    pos {pos:>3d}: r={corr:>7.4f}, p={pval:.4e} {sig}")

                # If significant correlation exists for a patient-invariant channel,
                # it means the correlation is driven by trajectory LENGTH (survivorship bias)
                if pval < 0.05 and abs(corr) > 0.05:
                    logger.info(f"             ^ Likely driven by trajectory-length confound,")
                    logger.info(f"               not genuine clinical signal.")

    # ------------------------------------------------------------------ #
    # CHECK 4: Trajectory length as confound
    # ------------------------------------------------------------------ #
    logger.info(f"\n[CHECK 4] Trajectory length vs outcome:")
    corr, pval = pointbiserialr(y, traj_lengths)
    logger.info(f"  Correlation: r={corr:.4f}, p={pval:.4e}")
    logger.info(f"  Dead: mean_traj_len={traj_lengths[y == 1].mean():.1f}")
    logger.info(f"  Alive: mean_traj_len={traj_lengths[y == 0].mean():.1f}")

    logger.info("\n" + "=" * 72)
    logger.info("SUMMARY:")
    logger.info("  If temporal variance ≈ 0 at each position: patient-invariant (confirmed)")
    logger.info("  If correlation with outcome exists: confounded by trajectory length")
    logger.info("  These features encode bin structure, not patient state.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
