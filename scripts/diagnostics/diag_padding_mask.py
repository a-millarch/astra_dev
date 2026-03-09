#!/usr/bin/env python
"""
DIAGNOSTIC: Confirm padding mask is always None and quantify ghost positions.

Checks:
1. Whether _key_padding_mask returns None (expected: always)
2. How many "ghost" positions exist (within trajectory, clinical=0, temporal!=0)
3. What fraction of attention computation is wasted on unmasked padding

Run on Azure:
    python -m scripts.diagnostics.diag_padding_mask
"""
import logging
import numpy as np
import torch
from astra.utils import cfg
from astra.data.dataloader import (
    prepare_data_and_dls,
    get_trajectory_lengths,
)
from astra.models.hybrid.training import get_backbone

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 72)
    logger.info("DIAGNOSTIC: Padding Mask & Ghost Position Analysis")
    logger.info("=" * 72)

    data = prepare_data_and_dls(cfg)

    X = data["X"]                              # normalized [n, c, s]
    traj_lengths = data["trajectory_lengths"]
    ts_channel_names = data["ts_channel_names"]

    n_samples, n_channels, seq_len = X.shape

    # ------------------------------------------------------------------ #
    # CHECK 1: NaN count in normalized X (must be 0 for mask to be None)
    # ------------------------------------------------------------------ #
    n_nan = np.isnan(X).sum()
    logger.info(f"\n[CHECK 1] NaN count in normalized X: {n_nan}")
    logger.info(f"  → _key_padding_mask relies on NaN to build mask.")
    if n_nan == 0:
        logger.info(f"  → CONFIRMED: No NaN → mask is ALWAYS None.")
    else:
        logger.info(f"  → UNEXPECTED: {n_nan} NaN values found — mask may sometimes work.")

    # ------------------------------------------------------------------ #
    # CHECK 2: Model _key_padding_mask on a real batch
    # ------------------------------------------------------------------ #
    model = get_backbone(
        data, cfg,
        temporal_channel_idx=data.get("temporal_channel_idx"),
        exclude_channel_indices=data.get("exclude_channel_indices", []),
    )
    batch_size = min(64, n_samples)
    x_batch = torch.tensor(X[:batch_size], dtype=torch.float32)
    _, mask = model._key_padding_mask(x_batch.clone())
    logger.info(f"\n[CHECK 2] _key_padding_mask returned: {mask}")

    # ------------------------------------------------------------------ #
    # CHECK 3: Ideal padding statistics
    # ------------------------------------------------------------------ #
    pos_arr = np.arange(seq_len)[np.newaxis, :]
    is_padding = pos_arr >= traj_lengths[:, np.newaxis]
    n_padding = is_padding.sum()
    n_total = is_padding.size
    logger.info(f"\n[CHECK 3] Padding statistics (all samples):")
    logger.info(f"  Total positions: {n_total:,}")
    logger.info(f"  Padding positions: {n_padding:,} ({100*n_padding/n_total:.1f}%)")
    logger.info(f"  Real positions: {n_total - n_padding:,} ({100*(n_total-n_padding)/n_total:.1f}%)")
    logger.info(f"  Trajectory lengths — min: {traj_lengths.min()}, max: {traj_lengths.max()}, "
                f"mean: {traj_lengths.mean():.1f}, median: {np.median(traj_lengths):.0f}")

    # ------------------------------------------------------------------ #
    # CHECK 4: Ghost positions (temporal features analysis)
    # ------------------------------------------------------------------ #
    tf_cfg = cfg.get("temporal_features", {})
    tf_features = tf_cfg.get("features", [])
    tf_enabled = tf_cfg.get("enabled", False)

    temporal_idx = [i for i, n in enumerate(ts_channel_names) if n in tf_features]
    clinical_idx = [i for i, n in enumerate(ts_channel_names) if n not in tf_features]

    logger.info(f"\n[CHECK 4] Ghost position analysis:")
    logger.info(f"  Temporal features enabled: {tf_enabled}")
    logger.info(f"  Temporal channels ({len(temporal_idx)}): "
                f"{[ts_channel_names[i] for i in temporal_idx]}")
    logger.info(f"  Clinical channels ({len(clinical_idx)}): {len(clinical_idx)} channels")

    if temporal_idx:
        total_ghost = 0
        total_within_traj = 0
        ghost_per_sample = []

        for i in range(n_samples):
            tl = traj_lengths[i]
            if tl == 0:
                continue
            within = X[i, :, :tl]
            clinical_vals = within[clinical_idx, :]
            temporal_vals = within[temporal_idx, :]

            clinical_all_zero = np.all(np.abs(clinical_vals) < 1e-6, axis=0)
            temporal_any_nonzero = np.any(np.abs(temporal_vals) > 1e-6, axis=0)
            ghost = clinical_all_zero & temporal_any_nonzero

            n_ghost = ghost.sum()
            total_ghost += n_ghost
            total_within_traj += tl
            ghost_per_sample.append(n_ghost)

        ghost_arr = np.array(ghost_per_sample)
        logger.info(f"  Total ghost positions: {total_ghost:,} / {total_within_traj:,} "
                    f"within-trajectory ({100*total_ghost/max(total_within_traj,1):.1f}%)")
        logger.info(f"  Per-sample ghost count — mean: {ghost_arr.mean():.1f}, "
                    f"max: {ghost_arr.max()}, median: {np.median(ghost_arr):.0f}")
        logger.info(f"  Samples with >0 ghosts: {(ghost_arr > 0).sum()} / {n_samples}")

        if total_ghost > 0:
            logger.info(f"\n  ** These positions have no clinical data but NON-ZERO temporal values.")
            logger.info(f"     Without a padding mask, the model attends to them as if they")
            logger.info(f"     contain signal, creating noise in the attention computation.")
    else:
        logger.info(f"  No temporal channels present — ghost analysis not applicable.")

    # ------------------------------------------------------------------ #
    # CHECK 5: Values at padding positions
    # ------------------------------------------------------------------ #
    logger.info(f"\n[CHECK 5] Padding position values (first 10 samples):")
    for i in range(min(10, n_samples)):
        tl = traj_lengths[i]
        if tl >= seq_len:
            logger.info(f"  Sample {i}: no padding (traj_len={tl} == seq_len={seq_len})")
            continue
        padding_vals = X[i, :, tl:]
        nonzero = np.abs(padding_vals) > 1e-6
        logger.info(f"  Sample {i}: traj_len={tl}, padding_positions={seq_len-tl}, "
                    f"non-zero in padding={nonzero.sum()}")

    # ------------------------------------------------------------------ #
    # SUMMARY
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 72)
    logger.info("SUMMARY:")
    if n_nan == 0:
        logger.info("  1. _key_padding_mask ALWAYS returns None (confirmed)")
        logger.info(f"  2. {100*n_padding/n_total:.1f}% of positions are padding — all unmasked")
        if temporal_idx and total_ghost > 0:
            logger.info(f"  3. {100*total_ghost/max(total_within_traj,1):.1f}% of within-trajectory "
                        f"positions are 'ghosts' (temporal-only)")
            logger.info("  4. FIX NEEDED: Pass trajectory_lengths to model for proper masking")
        else:
            logger.info("  3. No ghost positions (temporal features disabled or no ghosts)")
            logger.info("  4. Padding mask fix still recommended for attention quality")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
