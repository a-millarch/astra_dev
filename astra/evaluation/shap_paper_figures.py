"""
SHAP analysis figures for JMIR paper submission.

Usage:
    python -m astra.evaluation.shap_paper_figures [--recompute] [--figures-only]

Flags:
    --recompute    Force recomputation of SHAP values (default: load cached if available)
    --figures-only  Skip SHAP computation, only regenerate figures from cached results
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from astra.data.caching import prepare_data_and_dls_cached
from astra.evaluation.behavior import (
    _get_clinical_only_channel_mask,
    _get_clinical_only_channel_order,
    calculate_shap_from_dataloaders,
    classify_channels,
    create_channel_mapping,
    get_category_names_from_encoding_info,
    get_holdout_pids,
    get_static_cat_names_from_classes,
)
from astra.evaluation.utils import prepare_model, time_to_step
from astra.utils import cfg, ensure_parent_dir

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

EVAL_TIMEPOINTS = [1, 6, 12, 72, 168, 720]  # hours
EVAL_LABELS = ['1H', '6H', '12H', '3D', '7D', '30D']
OUTPUT_DIR = 'reports/shap_paper'
CACHE_PATH = os.path.join(OUTPUT_DIR, 'shap_cache.pkl')
SAMPLES_PATH = os.path.join(OUTPUT_DIR, 'stratified_samples.pkl')
SEED = 42
MAX_BACKGROUND = 300
TOP_K_FEATURES = 15   # Figure A
TOP_K_HEATMAP = 20    # Figure B

# Matplotlib style
FIGSTYLE = 'seaborn-v0_8-whitegrid'
RC_PARAMS = {
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
}

# Colorblind-friendly palette
COLOR_POS = '#d62728'   # risk-increasing (red-ish)
COLOR_NEG = '#1f77b4'   # protective (blue)
COLOR_BAR = '#4c72b0'   # neutral bar color


# ============================================================================
# Stratified Sampling
# ============================================================================

def stratified_sample_at_timepoint(
    holdout_pids: np.ndarray,
    holdout_labels: np.ndarray,
    holdout_traj_lengths: np.ndarray,
    timepoint_hours: float,
    rng: np.random.Generator,
) -> Dict:
    """Draw a stratified 1:1 sample of active patients at a given timepoint.

    Returns dict with keys: pids, step, n_active_total, n_nonsurvivor,
    n_survivor_sampled, prevalence_active, active_mask.
    """
    step = time_to_step(timepoint_hours, 'h')
    active_mask = holdout_traj_lengths >= step

    active_pids = holdout_pids[active_mask]
    active_labels = holdout_labels[active_mask]
    n_active = int(active_mask.sum())

    nonsurvivor_mask = active_labels == 1
    survivor_mask = active_labels == 0
    nonsurvivor_pids = active_pids[nonsurvivor_mask]
    survivor_pids = active_pids[survivor_mask]

    n_nonsurvivor = len(nonsurvivor_pids)
    n_survivor_available = len(survivor_pids)

    if n_nonsurvivor == 0:
        logger.warning(
            f"  [{timepoint_hours}h] No non-survivors active at this timepoint. "
            f"Skipping."
        )
        return {
            'pids': [],
            'step': step,
            'timepoint_hours': timepoint_hours,
            'n_active_total': n_active,
            'n_active_nonsurvivor': 0,
            'n_active_survivor': n_survivor_available,
            'n_nonsurvivor': 0,
            'n_survivor_sampled': 0,
            'n_total_sampled': 0,
            'prevalence_active': 0.0,
        }

    if n_nonsurvivor < 10:
        logger.warning(
            f"  [{timepoint_hours}h] Only {n_nonsurvivor} non-survivors active "
            f"(< 10 threshold). Proceeding anyway."
        )

    # 1:1 matching: sample equal number of survivors
    n_sample_survivors = min(n_nonsurvivor, n_survivor_available)
    if n_sample_survivors < n_nonsurvivor:
        logger.warning(
            f"  [{timepoint_hours}h] Only {n_survivor_available} survivors available "
            f"for {n_nonsurvivor} non-survivors. Using {n_sample_survivors} of each."
        )
    sampled_survivor_pids = rng.choice(
        survivor_pids, size=n_sample_survivors, replace=False
    )

    sampled_pids = np.concatenate([nonsurvivor_pids, sampled_survivor_pids])
    prevalence = n_nonsurvivor / n_active * 100 if n_active > 0 else 0.0

    logger.info(
        f"  [{timepoint_hours}h → step {step}] "
        f"active={n_active}, deceased={n_nonsurvivor}, "
        f"sampled_survivors={n_sample_survivors}, "
        f"total_sampled={len(sampled_pids)}, "
        f"prevalence={prevalence:.1f}%"
    )

    return {
        'pids': sampled_pids.tolist(),
        'step': step,
        'timepoint_hours': timepoint_hours,
        'n_active_total': n_active,
        'n_active_nonsurvivor': n_nonsurvivor,
        'n_active_survivor': n_survivor_available,
        'n_nonsurvivor': n_nonsurvivor,
        'n_survivor_sampled': n_sample_survivors,
        'n_total_sampled': len(sampled_pids),
        'prevalence_active': prevalence,
    }


def run_stratified_sampling(data: Dict) -> Dict[str, Dict]:
    """Run stratified sampling at all evaluation timepoints.

    Returns dict keyed by EVAL_LABELS.
    """
    holdout_pids = np.array(data["holdout"].tab_df['PID'].tolist())
    holdout_labels = data["holdout"].tab_df[data["holdout"].target].values.astype(int)
    holdout_traj = np.array(data["holdout_trajectory_lengths"])

    logger.info(
        f"Holdout set: {len(holdout_pids)} patients, "
        f"{holdout_labels.sum()} non-survivors ({holdout_labels.mean()*100:.1f}%)"
    )

    rng = np.random.default_rng(SEED)
    sampling_results = {}

    for hours, label in zip(EVAL_TIMEPOINTS, EVAL_LABELS):
        result = stratified_sample_at_timepoint(
            holdout_pids, holdout_labels, holdout_traj, hours, rng
        )
        sampling_results[label] = result

    return sampling_results


# ============================================================================
# SHAP Computation
# ============================================================================

def compute_shap_per_timepoint(
    model,
    data: Dict,
    sampling_results: Dict[str, Dict],
    device: str,
) -> Dict[str, Dict]:
    """Compute SHAP values at each evaluation timepoint using stratified samples."""
    all_holdout_pids = data["holdout"].tab_df['PID'].tolist()
    encoding_info = data.get("encoding_info", {})
    all_results = {}

    for label, sample_info in sampling_results.items():
        hours = sample_info['timepoint_hours']
        step = sample_info['step']
        pids = sample_info['pids']
        n_deceased = sample_info['n_nonsurvivor']

        if len(pids) == 0:
            logger.info(f"Skipping {label} — no sampled patients.")
            continue

        logger.info(
            f"\n{'='*60}\n"
            f"Computing SHAP at {label} (step={step}, n={len(pids)}, "
            f"{n_deceased} deceased)\n"
            f"{'='*60}"
        )

        shap_results = calculate_shap_from_dataloaders(
            model=model,
            background_loader=data["mixed_dls"],
            test_loader=data["holdout_mixed_dls"],
            encoding_info=encoding_info,
            device=device,
            max_background_samples=MAX_BACKGROUND,
            max_test_samples=None,  # handled by specific_pids
            compute_per_category_shap=True,
            specific_pids=pids,
            all_pids=all_holdout_pids,
            eval_timestep=step,
        )

        # Squeeze trailing singleton dimension from GradientExplainer
        # (returns shape [..., 1] for single-output models)
        _squeeze_shap_results(shap_results)

        all_results[label] = shap_results
        logger.info(
            f"  ts_shap shape: {shap_results['ts_shap'].shape}, "
            f"eval_timestep: {shap_results['eval_timestep']}"
        )

    return all_results


def _squeeze_shap_results(shap_results: Dict) -> None:
    """Squeeze trailing singleton dimensions from SHAP arrays in-place."""
    for key in ('ts_shap', 'cat_ts_shap', 'cat_ts_shap_per_category',
                'cat_ts_shap_embedded', 'cat_shap', 'cat_shap_embedded',
                'cont_shap'):
        val = shap_results.get(key)
        if val is not None and isinstance(val, np.ndarray) and val.ndim > 1 and val.shape[-1] == 1:
            shap_results[key] = val.squeeze(-1)


def validate_shap_results(
    all_results: Dict[str, Dict],
    sampling_results: Dict[str, Dict],
) -> None:
    """Run validation checks on computed SHAP values."""
    logger.info("\n--- SHAP Validation ---")

    for label in EVAL_LABELS:
        if label not in all_results:
            continue
        res = all_results[label]
        sample_info = sampling_results[label]

        ts_shap = res['ts_shap']
        n_samples = ts_shap.shape[0]
        expected = sample_info['n_total_sampled']

        # Sample size check
        if n_samples != expected:
            logger.warning(
                f"  [{label}] Sample size mismatch: got {n_samples}, expected {expected}"
            )
        else:
            logger.info(f"  [{label}] Sample size OK: {n_samples}")

        # NaN check
        has_nan = np.isnan(ts_shap).any()
        if has_nan:
            logger.warning(f"  [{label}] NaN found in ts_shap!")
        else:
            logger.info(f"  [{label}] No NaN in ts_shap")

        # Mean absolute SHAP magnitude
        mean_abs = np.abs(ts_shap).mean()
        logger.info(f"  [{label}] Mean |SHAP| = {mean_abs:.6f}")

    logger.info("--- Validation complete ---\n")


# ============================================================================
# Figure A: Top-K Feature Importance Across Timepoints
# ============================================================================

def figure_a_topk_importance(
    all_results: Dict[str, Dict],
    sampling_results: Dict[str, Dict],
    channel2feature: Dict[int, str],
    n_channels: int,
    save_dir: str,
) -> None:
    """2x3 grid of horizontal bar charts showing top-15 clinical features."""
    plt.style.use(FIGSTYLE)
    plt.rcParams.update(RC_PARAMS)

    clinical_indices = _get_clinical_only_channel_mask(channel2feature, n_channels)
    clinical_names = [channel2feature[i] for i in clinical_indices]

    # Compute mean |SHAP| per channel per timepoint
    importance_per_tp = {}
    sem_per_tp = {}
    for label in EVAL_LABELS:
        if label not in all_results:
            continue
        ts_shap = all_results[label]['ts_shap']  # [n_samples, n_channels, seq_len]
        # Mean across seq_len, then across samples for clinical channels only
        per_sample_importance = np.abs(ts_shap[:, clinical_indices, :]).mean(axis=2)  # [n, n_clin]
        importance_per_tp[label] = per_sample_importance.mean(axis=0)  # [n_clin]
        sem_per_tp[label] = per_sample_importance.std(axis=0) / np.sqrt(per_sample_importance.shape[0])

    # Build union of top-K features across all timepoints
    top_features_union = set()
    for label, imp in importance_per_tp.items():
        top_idx = np.argsort(imp)[-TOP_K_FEATURES:]
        for i in top_idx:
            top_features_union.add(clinical_names[i])

    # Order by overall importance (mean across timepoints)
    overall_imp = {}
    for feat in top_features_union:
        feat_idx = clinical_names.index(feat)
        overall_imp[feat] = np.mean([
            importance_per_tp[l][feat_idx] for l in EVAL_LABELS if l in importance_per_tp
        ])
    ordered_features = sorted(overall_imp, key=lambda f: overall_imp[f])  # ascending for barh

    fig, axes = plt.subplots(2, 3, figsize=(7, 5.5), constrained_layout=True)
    axes = axes.flatten()

    for ax_idx, label in enumerate(EVAL_LABELS):
        ax = axes[ax_idx]
        if label not in importance_per_tp:
            ax.set_visible(False)
            continue

        imp = importance_per_tp[label]
        sem = sem_per_tp[label]
        sample_info = sampling_results[label]

        vals = []
        errs = []
        for feat in ordered_features:
            feat_idx = clinical_names.index(feat)
            vals.append(imp[feat_idx])
            errs.append(sem[feat_idx])

        y_pos = np.arange(len(ordered_features))
        ax.barh(y_pos, vals, xerr=errs, color=COLOR_BAR, edgecolor='white',
                height=0.7, capsize=2, error_kw={'linewidth': 0.8})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_features, fontsize=7)
        ax.set_xlabel('Mean |SHAP|', fontsize=8)
        n_total = sample_info['n_total_sampled']
        n_dec = sample_info['n_nonsurvivor']
        ax.set_title(f'{label} (n={n_total}, {n_dec} deceased)', fontsize=9)
        ax.tick_params(axis='x', labelsize=7)

    fig.suptitle('Top Clinical Feature Importance by Timepoint', fontsize=11, y=1.01)

    for fmt in ('png', 'pdf'):
        path = os.path.join(save_dir, f'figure_a_topk_importance.{fmt}')
        ensure_parent_dir(path)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure A saved to {save_dir}")


# ============================================================================
# Figure B: Feature Importance Heatmap Over Time
# ============================================================================

def figure_b_heatmap(
    all_results: Dict[str, Dict],
    channel2feature: Dict[int, str],
    n_channels: int,
    save_dir: str,
) -> None:
    """Heatmap: rows = top-20 clinical channels, columns = timepoints."""
    plt.style.use(FIGSTYLE)
    plt.rcParams.update(RC_PARAMS)

    clinical_indices = _get_clinical_only_channel_mask(channel2feature, n_channels)
    clinical_names = [channel2feature[i] for i in clinical_indices]

    # Check for EBM channel
    grouped = classify_channels(channel2feature)
    ebm_channels = grouped.get('EBM', [])
    has_ebm = len(ebm_channels) > 0

    # Build importance matrix: [n_clinical, n_timepoints]
    imp_matrix = np.zeros((len(clinical_indices), len(EVAL_LABELS)))
    for col_idx, label in enumerate(EVAL_LABELS):
        if label not in all_results:
            continue
        ts_shap = all_results[label]['ts_shap']
        per_channel = np.abs(ts_shap[:, clinical_indices, :]).mean(axis=2).mean(axis=0)
        imp_matrix[:, col_idx] = per_channel

    # Overall importance for ordering
    overall = imp_matrix.mean(axis=1)
    top_idx = np.argsort(overall)[-TOP_K_HEATMAP:][::-1]  # descending

    row_labels_clinical = [clinical_names[i] for i in top_idx]

    # Compute EBM row if present
    ebm_row = None
    if has_ebm:
        ebm_ch_idx = ebm_channels[0][0]  # (idx, name)
        ebm_row = np.zeros(len(EVAL_LABELS))
        for col_idx, label in enumerate(EVAL_LABELS):
            if label not in all_results:
                continue
            ts_shap = all_results[label]['ts_shap']
            ebm_row[col_idx] = np.abs(ts_shap[:, ebm_ch_idx, :]).mean(axis=1).mean(axis=0)

    available_labels = [l for l in EVAL_LABELS if l in all_results]
    df_heat = pd.DataFrame(
        imp_matrix[top_idx][:, [EVAL_LABELS.index(l) for l in available_labels]],
        index=row_labels_clinical,
        columns=available_labels,
    )
    if has_ebm:
        ebm_df = pd.DataFrame(
            ebm_row[[EVAL_LABELS.index(l) for l in available_labels]].reshape(1, -1),
            index=['EBM (_ebm_pred)'],
            columns=available_labels,
        )
        df_heat = pd.concat([df_heat, ebm_df])
    row_labels = list(df_heat.index)

    n_rows = len(row_labels)
    fig_height = max(4, n_rows * 0.28 + 1.0)
    fig, ax = plt.subplots(figsize=(5, fig_height), constrained_layout=True)

    sns.heatmap(
        df_heat, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
        linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8, 'label': 'Mean |SHAP|'},
        annot_kws={'fontsize': 7},
    )

    # Draw separator before EBM row
    if has_ebm:
        ax.axhline(y=len(row_labels) - 1, color='black', linewidth=2)

    ax.set_ylabel('')
    ax.set_xlabel('Evaluation Timepoint')
    ax.set_title('Feature Importance Across Patient Trajectory', fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=9)

    for fmt in ('png', 'pdf'):
        path = os.path.join(save_dir, f'figure_b_heatmap.{fmt}')
        ensure_parent_dir(path)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure B saved to {save_dir}")


# ============================================================================
# Figure C: Static Feature Importance Across Timepoints
# ============================================================================

def figure_c_static_features(
    all_results: Dict[str, Dict],
    sampling_results: Dict[str, Dict],
    static_cat_names: List[str],
    static_cont_names: List[str],
    save_dir: str,
) -> None:
    """Static feature SHAP (signed) across timepoints. Two sub-panels."""
    plt.style.use(FIGSTYLE)
    plt.rcParams.update(RC_PARAMS)

    has_cat = len(static_cat_names) > 0
    has_cont = len(static_cont_names) > 0
    n_panels = has_cat + has_cont
    if n_panels == 0:
        logger.warning("No static features to plot for Figure C.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, max(4, 0.3 * max(
        len(static_cat_names) if has_cat else 0,
        len(static_cont_names) if has_cont else 0,
    ) + 1.5)), constrained_layout=True)

    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # --- Static Categorical ---
    if has_cat:
        ax = axes[ax_idx]; ax_idx += 1
        # Build matrix: [n_cat_features, n_timepoints]
        cat_matrix = np.zeros((len(static_cat_names), len(EVAL_LABELS)))
        for col, label in enumerate(EVAL_LABELS):
            if label not in all_results:
                continue
            cat_shap = all_results[label].get('cat_shap')  # [n_samples, n_cat]
            if cat_shap is not None:
                cat_matrix[:cat_shap.shape[1], col] = cat_shap.mean(axis=0)

        # Plot grouped bars or dot plot — use dot plot for clarity
        for col, label in enumerate(EVAL_LABELS):
            vals = cat_matrix[:, col]
            y_pos = np.arange(len(static_cat_names))
            colors = [COLOR_POS if v > 0 else COLOR_NEG for v in vals]
            ax.scatter(vals, y_pos, c=colors, s=25, zorder=3, label=label if col == 0 else None,
                       marker='o', alpha=0.7 + col * 0.05)

        # Also show mean across timepoints as bars
        mean_vals = cat_matrix.mean(axis=1)
        colors_mean = [COLOR_POS if v > 0 else COLOR_NEG for v in mean_vals]
        y_pos = np.arange(len(static_cat_names))
        ax.barh(y_pos, mean_vals, color=colors_mean, alpha=0.3, height=0.6)
        ax.axvline(x=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(static_cat_names, fontsize=7)
        ax.set_xlabel('Mean SHAP (signed)')
        ax.set_title('Static Categorical', fontsize=10)

    # --- Static Continuous ---
    if has_cont:
        ax = axes[ax_idx]; ax_idx += 1
        cont_matrix = np.zeros((len(static_cont_names), len(EVAL_LABELS)))
        for col, label in enumerate(EVAL_LABELS):
            if label not in all_results:
                continue
            cont_shap = all_results[label].get('cont_shap')  # [n_samples, n_cont]
            if cont_shap is not None:
                cont_matrix[:cont_shap.shape[1], col] = cont_shap.mean(axis=0)

        mean_vals = cont_matrix.mean(axis=1)
        colors_mean = [COLOR_POS if v > 0 else COLOR_NEG for v in mean_vals]
        y_pos = np.arange(len(static_cont_names))
        ax.barh(y_pos, mean_vals, color=colors_mean, alpha=0.5, height=0.6)

        # Overlay per-timepoint dots
        for col, label in enumerate(EVAL_LABELS):
            vals = cont_matrix[:, col]
            ax.scatter(vals, y_pos, s=20, zorder=3, alpha=0.7, marker='D')

        ax.axvline(x=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(static_cont_names, fontsize=7)
        ax.set_xlabel('Mean SHAP (signed)')
        ax.set_title('Static Continuous', fontsize=10)

    fig.suptitle('Static Feature Importance', fontsize=11)

    for fmt in ('png', 'pdf'):
        path = os.path.join(save_dir, f'figure_c_static_features.{fmt}')
        ensure_parent_dir(path)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure C saved to {save_dir}")


# ============================================================================
# Figure D: Sampling Summary Table
# ============================================================================

def figure_d_sampling_table(
    sampling_results: Dict[str, Dict],
    save_dir: str,
) -> None:
    """Save CSV summary of stratified sampling at each timepoint."""
    rows = []
    for label in EVAL_LABELS:
        if label not in sampling_results:
            continue
        s = sampling_results[label]
        rows.append({
            'timepoint': label,
            'hours': s['timepoint_hours'],
            'n_active_total': s['n_active_total'],
            'n_active_nonsurvivor': s['n_active_nonsurvivor'],
            'n_active_survivor': s['n_active_survivor'],
            'prevalence_pct': round(s['prevalence_active'], 2),
            'n_sampled_nonsurvivor': s['n_nonsurvivor'],
            'n_sampled_survivor': s['n_survivor_sampled'],
            'n_total_sampled': s['n_total_sampled'],
        })

    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, 'sampling_summary.csv')
    ensure_parent_dir(path)
    df.to_csv(path, index=False)
    logger.info(f"Figure D (sampling table) saved to {path}")
    print(f"\nSampling Summary:\n{df.to_string(index=False)}\n")


# ============================================================================
# Figure E: Categorical TS Feature Importance
# ============================================================================

def figure_e_categorical_ts(
    all_results: Dict[str, Dict],
    encoding_info: Dict,
    save_dir: str,
    top_k: int = 20,
) -> None:
    """Heatmap of top categorical TS features across timepoints."""
    plt.style.use(FIGSTYLE)
    plt.rcParams.update(RC_PARAMS)

    # Check if per-category SHAP is available
    first_label = next((l for l in EVAL_LABELS if l in all_results), None)
    if first_label is None:
        logger.warning("No results for Figure E.")
        return

    cat_ts_shap = all_results[first_label].get('cat_ts_shap_per_category')
    if cat_ts_shap is None:
        logger.warning("No per-category SHAP values available for Figure E. Skipping.")
        return

    category_names = get_category_names_from_encoding_info(encoding_info)
    n_cats = len(category_names)

    # Build importance matrix: [n_categories, n_timepoints]
    imp_matrix = np.zeros((n_cats, len(EVAL_LABELS)))
    for col_idx, label in enumerate(EVAL_LABELS):
        if label not in all_results:
            continue
        cat_shap = all_results[label].get('cat_ts_shap_per_category')
        if cat_shap is None:
            continue
        # cat_shap: [n_samples, n_cats, seq_len] → mean |SHAP| across seq_len and samples
        per_cat = np.abs(cat_shap).mean(axis=2).mean(axis=0)  # [n_cats]
        imp_matrix[:min(n_cats, len(per_cat)), col_idx] = per_cat[:min(n_cats, len(per_cat))]

    # Top-K by overall importance
    overall = imp_matrix.mean(axis=1)
    top_idx = np.argsort(overall)[-top_k:][::-1]

    heatmap_data = imp_matrix[top_idx]
    row_labels = [category_names[i] if i < len(category_names) else f'Cat_{i}'
                  for i in top_idx]

    # Truncate long labels
    row_labels = [name[:40] + '...' if len(name) > 40 else name for name in row_labels]

    available_labels = [l for l in EVAL_LABELS if l in all_results]
    avail_col_idx = [EVAL_LABELS.index(l) for l in available_labels]
    df_heat = pd.DataFrame(
        heatmap_data[:, avail_col_idx], index=row_labels, columns=available_labels
    )

    n_rows = len(row_labels)
    fig_height = max(4, n_rows * 0.3 + 1.0)
    fig, ax = plt.subplots(figsize=(5, fig_height), constrained_layout=True)

    sns.heatmap(
        df_heat, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
        linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8, 'label': 'Mean |SHAP|'},
        annot_kws={'fontsize': 7},
    )

    ax.set_ylabel('')
    ax.set_xlabel('Evaluation Timepoint')
    ax.set_title('Categorical TS Feature Importance', fontsize=11)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=9)

    for fmt in ('png', 'pdf'):
        path = os.path.join(save_dir, f'figure_e_categorical_ts.{fmt}')
        ensure_parent_dir(path)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure E saved to {save_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SHAP analysis figures for JMIR paper'
    )
    parser.add_argument(
        '--recompute', action='store_true',
        help='Force recomputation of SHAP values'
    )
    parser.add_argument(
        '--figures-only', action='store_true',
        help='Skip SHAP computation, regenerate figures from cache'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable DEBUG logging'
    )
    args = parser.parse_args()

    # Logging setup
    from astra.utils import setup_logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Loading data...")
    data = prepare_data_and_dls_cached(cfg)

    logger.info("Loading model...")
    model, device = prepare_model(data, cfg)
    model.eval()

    # --- Channel & feature metadata ---
    channel2feature, feature2channel = create_channel_mapping(data)
    n_channels = data['c_in']
    static_cat_names = get_static_cat_names_from_classes(data['classes'])
    static_cont_names = cfg.get("dataset", {}).get("num_cols", data.get("num_cols", []))
    encoding_info = data.get("encoding_info", {})

    logger.info(
        f"Channels: {n_channels}, Static cat: {len(static_cat_names)}, "
        f"Static cont: {len(static_cont_names)}"
    )

    # --- Stratified sampling (always run — fast) ---
    logger.info("\n--- Stratified Sampling ---")
    sampling_results = run_stratified_sampling(data)

    # Save sampling info
    ensure_parent_dir(SAMPLES_PATH)
    with open(SAMPLES_PATH, 'wb') as f:
        pickle.dump(sampling_results, f)
    logger.info(f"Sampling results saved to {SAMPLES_PATH}")

    # --- SHAP computation or cache load ---
    if args.figures_only:
        if not os.path.exists(CACHE_PATH):
            logger.error(f"Cache not found at {CACHE_PATH}. Run without --figures-only first.")
            return
        logger.info(f"Loading cached SHAP results from {CACHE_PATH}...")
        with open(CACHE_PATH, 'rb') as f:
            all_results = pickle.load(f)
    elif args.recompute or not os.path.exists(CACHE_PATH):
        logger.info("\nComputing SHAP values per timepoint...")
        all_results = compute_shap_per_timepoint(model, data, sampling_results, device)

        ensure_parent_dir(CACHE_PATH)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"SHAP results cached to {CACHE_PATH}")
    else:
        logger.info(f"Loading cached SHAP results from {CACHE_PATH}...")
        with open(CACHE_PATH, 'rb') as f:
            all_results = pickle.load(f)

    # --- Squeeze cached results (handles trailing singleton from GradientExplainer) ---
    for label in list(all_results.keys()):
        _squeeze_shap_results(all_results[label])

    # --- Validation ---
    validate_shap_results(all_results, sampling_results)

    # --- Generate figures ---
    save_dir = OUTPUT_DIR
    logger.info("\n--- Generating Figures ---")

    logger.info("Figure A: Top-K Feature Importance...")
    figure_a_topk_importance(all_results, sampling_results, channel2feature, n_channels, save_dir)

    logger.info("Figure B: Feature Importance Heatmap...")
    figure_b_heatmap(all_results, channel2feature, n_channels, save_dir)

    logger.info("Figure C: Static Feature Importance...")
    figure_c_static_features(all_results, sampling_results, static_cat_names, static_cont_names, save_dir)

    logger.info("Figure D: Sampling Summary Table...")
    figure_d_sampling_table(sampling_results, save_dir)

    logger.info("Figure E: Categorical TS Feature Importance...")
    figure_e_categorical_ts(all_results, encoding_info, save_dir)

    logger.info(f"\nAll figures saved to {save_dir}/")
    logger.info(f"Random seed used: {SEED}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
