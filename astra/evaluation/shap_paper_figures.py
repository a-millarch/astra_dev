"""
SHAP analysis figures for JMIR paper submission.

Usage:
    python -m astra.evaluation.shap_paper_figures [--recompute] [--figures-only]
    python -m astra.evaluation.shap_paper_figures --renormalize-cat-ts --pickle-path <path>

Flags:
    --recompute           Force recomputation of SHAP values
    --figures-only        Skip SHAP computation, regenerate figures from cache
    --renormalize-cat-ts  Re-normalize categorical TS SHAP from existing pickle (no data/model needed)
    --pickle-path         Path to cohort_temporal_shap_results*.pkl (for --renormalize-cat-ts)
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
from astra.utils import cfg, ensure_parent_dir, get_cfg, save_base64, save_figure

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

EVAL_TIMEPOINTS = [1, 6, 12, 72, 168, 336, 720]  # hours
EVAL_LABELS = ['1H', '6H', '12H', '3D', '7D', '14D', '30D']
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

# Journal submission output constraints for PNG outputs (PDFs skip the pixel cap —
# they are vector-friendly). Mirrors predictive_performance._SUBMISSION_KW.
_SUBMISSION_KW = dict(
    fit_long_side_px=1200,
    max_long_side_px=1200,
    max_bytes=5_000_000,
)


def _save_shap_figure(fig, save_dir, stem):
    """Save *fig* as PDF (vector) + PNG + base64 sidecar, applying submission caps.

    PDF is written first (raw fig.savefig) so matplotlib still has the live figure;
    save_figure then handles PNG + base64 and closes the figure.
    """
    ensure_parent_dir(os.path.join(save_dir, stem))
    pdf_path = os.path.join(save_dir, f'{stem}.pdf')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    save_figure(fig, stem, save_dir=save_dir, **_SUBMISSION_KW)


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


def select_representative_sample(
    data: Dict,
    n_target: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[int], pd.DataFrame]:
    """Select holdout patients representative of the full cohort.

    Stratifies by outcome x trajectory_duration x sex x age_bin, then
    samples proportionally from each stratum.  Returns selected PIDs and
    a comparison DataFrame (cohort vs sample statistics).
    """
    holdout_base = data["holdout"].base.copy()
    holdout_traj = np.array(data["holdout_trajectory_lengths"])
    holdout_base["trajectory_length"] = holdout_traj

    target_col = data["holdout"].target  # e.g. 'deceased_30d'

    # --- build stratification bins ---
    holdout_base["traj_bin"] = pd.qcut(
        holdout_base["trajectory_length"], q=3, labels=["short", "mid", "long"],
        duplicates="drop",
    )
    holdout_base["age_bin"] = pd.qcut(
        holdout_base["AGE"], q=3, labels=["young", "mid", "old"],
        duplicates="drop",
    )
    holdout_base["stratum"] = (
        holdout_base[target_col].astype(str) + "_"
        + holdout_base["traj_bin"].astype(str) + "_"
        + holdout_base["SEX"].astype(str) + "_"
        + holdout_base["age_bin"].astype(str)
    )

    rng = np.random.default_rng(seed)
    n_cohort = len(holdout_base)
    selected_idx = []

    for _stratum, group in holdout_base.groupby("stratum", observed=True):
        n_stratum_target = max(1, round(n_target * len(group) / n_cohort))
        n_pick = min(n_stratum_target, len(group))
        chosen = rng.choice(group.index.values, size=n_pick, replace=False)
        selected_idx.extend(chosen.tolist())

    selected = holdout_base.loc[selected_idx]

    # --- comparison table ---
    comparison = _compare_sample_representativeness(
        holdout_base, selected, target_col
    )

    if verbose:
        logger.info(
            f"Representative sample: {len(selected)}/{n_cohort} patients "
            f"({len(holdout_base['stratum'].unique())} strata)"
        )
        logger.info(f"\n{comparison.to_string(index=False)}")

    return selected["PID"].tolist(), comparison


def _compare_sample_representativeness(
    cohort: pd.DataFrame,
    sample: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Compare cohort vs sample on key variables."""
    rows = []

    def _add(name, cohort_val, sample_val):
        rows.append({"variable": name, "cohort": cohort_val, "sample": sample_val})

    _add("N", len(cohort), len(sample))
    _add(f"{target_col} (%)", f"{cohort[target_col].mean()*100:.1f}", f"{sample[target_col].mean()*100:.1f}")
    _add("age (mean±std)", f"{cohort['AGE'].mean():.1f}±{cohort['AGE'].std():.1f}",
         f"{sample['AGE'].mean():.1f}±{sample['AGE'].std():.1f}")
    _add("sex=Male (%)", f"{(cohort['SEX']=='Male').mean()*100:.1f}",
         f"{(sample['SEX']=='Male').mean()*100:.1f}")
    _add("traj_length (mean±std)", f"{cohort['trajectory_length'].mean():.1f}±{cohort['trajectory_length'].std():.1f}",
         f"{sample['trajectory_length'].mean():.1f}±{sample['trajectory_length'].std():.1f}")

    for col in ["ISS", "ASMT_ELIX"]:
        if col in cohort.columns:
            c_valid = cohort[col].dropna()
            s_valid = sample[col].dropna()
            _add(f"{col} (mean±std)",
                 f"{c_valid.mean():.1f}±{c_valid.std():.1f}" if len(c_valid) else "N/A",
                 f"{s_valid.mean():.1f}±{s_valid.std():.1f}" if len(s_valid) else "N/A")

    if "LVL1TC" in cohort.columns:
        _add("LVL1TC (%)", f"{cohort['LVL1TC'].mean()*100:.1f}",
             f"{sample['LVL1TC'].mean()*100:.1f}")

    return pd.DataFrame(rows)


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
                'cat_shap', 'cont_shap'):
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

    _save_shap_figure(fig, save_dir, 'figure_a_topk_importance')
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

    _save_shap_figure(fig, save_dir, 'figure_b_heatmap')
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

    _save_shap_figure(fig, save_dir, 'figure_c_static_features')
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

    _save_shap_figure(fig, save_dir, 'figure_e_categorical_ts')
    logger.info(f"Figure E saved to {save_dir}")


# ============================================================================
# Figure F: Summary Panel (paper-quality version of visualize_shap_summary)
# ============================================================================

# Channels to exclude from clinical display (temporal/auxiliary)
_EXCLUDED_CHANNELS = {'elapsed_hours', 'bin_width_hours', '_data_present'}
_EBM_CHANNEL_NAMES = {'_ebm_pred'}

# Canonical timeframe order (excluding 'full')
_TF_ORDER = ['1H', '6H', '12H', '1D', '3D', '7D', '14D', '30D']

RC_PARAMS_SUMMARY = {
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
}


_AGG_SUFFIXES = ('_max', '_min', '_mean', '_std', '_count')


def _clean_feature_name(name: str) -> str:
    """Strip aggregation suffixes and shorten prefixes for display."""
    for suffix in _AGG_SUFFIXES:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    if name.startswith('medication:'):
        name = 'med:' + name[len('medication:'):]
    return name


def _clean_feature_names(names: list) -> list:
    """Clean a list of feature names, adding suffix back in parens if duplicates arise."""
    cleaned = [_clean_feature_name(n) for n in names]
    # Check for duplicates — disambiguate with suffix in parens
    from collections import Counter
    counts = Counter(cleaned)
    if any(c > 1 for c in counts.values()):
        result = []
        for orig, clean in zip(names, cleaned):
            if counts[clean] > 1:
                suffix = orig[len(clean):]  # e.g. '_max'
                result.append(f'{clean} ({suffix.strip("_")})')
            else:
                result.append(clean)
        return result
    return cleaned


def _add_subplot_label(ax, label, fontsize=18):
    """Add a bold subplot label (A, B, C, ...) to top-left of axes."""
    ax.text(-0.08, 1.05, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='right')


def _load_csv_data(csv_path: str) -> dict:
    """Load and parse cohort_shap_all_features CSV into structured dicts.

    Returns dict with keys:
        temporal: DataFrame of temporal channel rows
        static_cat: DataFrame of static categorical rows
        static_cont: DataFrame of static continuous rows
        cat_ts: DataFrame of categorical TS rows
        channel2feature: {channel_idx: feature_name}
        timeframes: ordered list of available timeframes (excluding 'full')
    """
    df = pd.read_csv(csv_path)
    # Ensure numeric columns are parsed correctly (guards against serialization edge cases)
    df['mean_abs_shap'] = pd.to_numeric(df['mean_abs_shap'], errors='coerce')

    # Split by feature prefix
    temporal = df[~df['feature'].str.startswith(('static_cat:', 'static_cont:', 'cat_ts:'))].copy()
    static_cat = df[df['feature'].str.startswith('static_cat:')].copy()
    static_cont = df[df['feature'].str.startswith('static_cont:')].copy()
    cat_ts = df[df['feature'].str.startswith('cat_ts:')].copy()

    # Strip prefixes for display names
    static_cat['display_name'] = static_cat['feature'].str.replace('static_cat:', '', n=1)
    static_cont['display_name'] = static_cont['feature'].str.replace('static_cont:', '', n=1)
    cat_ts['display_name'] = cat_ts['feature'].str.replace('cat_ts:', '', n=1)

    # Build channel2feature from temporal rows
    ch2feat = {}
    for _, row in temporal.drop_duplicates('channel_idx').iterrows():
        if pd.notna(row['channel_idx']):
            ch2feat[int(row['channel_idx'])] = row['feature']

    # Available timeframes in canonical order, excluding 'full'
    available = set(df['timeframe'].unique())
    timeframes = [tf for tf in _TF_ORDER if tf in available]

    return {
        'temporal': temporal,
        'static_cat': static_cat,
        'static_cont': static_cont,
        'cat_ts': cat_ts,
        'channel2feature': ch2feat,
        'timeframes': timeframes,
    }


def _load_cat_ts_from_pickle(pickle_path: str, timeframes: list) -> Optional[pd.DataFrame]:
    """Fallback: aggregate categorical TS per-category data from pickle."""
    import pickle as pkl
    with open(pickle_path, 'rb') as f:
        results = pkl.load(f)

    # Try the new aggregated field first
    if hasattr(results, 'cat_ts_per_category_importance') and results.cat_ts_per_category_importance:
        cat_names = getattr(results, 'cat_ts_category_names', [])
        if not cat_names and results.encoding_info:
            cat_names = get_category_names_from_encoding_info(results.encoding_info)

        rows = []
        for tf in timeframes:
            imp = results.cat_ts_per_category_importance.get(tf)
            if imp is not None:
                for j, name in enumerate(cat_names):
                    if j < len(imp):
                        rows.append({
                            'timeframe': tf, 'feature': f'cat_ts:{name}',
                            'display_name': name, 'mean_abs_shap': imp[j],
                        })
        if rows:
            return pd.DataFrame(rows)

    # Fallback: aggregate from patient_results
    if results.patient_results:
        cat_names = get_category_names_from_encoding_info(results.encoding_info) if results.encoding_info else []
        rows = []
        for tf in timeframes:
            cat_arrays = []
            for pr in results.patient_results:
                norm_tf = 'full' if tf.startswith('max(') else tf
                tfr = pr.timeframe_results.get(norm_tf) or pr.timeframe_results.get(tf)
                if tfr is None:
                    continue
                if tfr.cat_ts_category_importance is not None:
                    cat_arrays.append(tfr.cat_ts_category_importance)
                elif tfr.cat_ts_shap_per_category is not None and tfr.cat_ts_shap_per_category.size > 0:
                    arr = tfr.cat_ts_shap_per_category
                    if arr.ndim == 2:
                        cat_arrays.append(np.abs(arr).mean(axis=1))
                    elif arr.ndim == 1:
                        cat_arrays.append(np.abs(arr))
            if cat_arrays:
                mean_imp = np.mean(cat_arrays, axis=0)
                for j, name in enumerate(cat_names):
                    if j < len(mean_imp):
                        rows.append({
                            'timeframe': tf, 'feature': f'cat_ts:{name}',
                            'display_name': name, 'mean_abs_shap': mean_imp[j],
                        })
        if rows:
            return pd.DataFrame(rows)

    return None


def _load_all_from_pickle(pickle_path: str, timeframes: list) -> Optional[dict]:
    """Load all panel data from CohortTemporalSHAPResults pickle.

    Returns dict with keys matching _load_csv_data output, or None on failure.
    """
    import pickle as pkl
    try:
        with open(pickle_path, 'rb') as f:
            results = pkl.load(f)
    except Exception as e:
        logger.warning(f"Failed to load pickle: {e}")
        return None

    ch2feat = results.channel2feature
    density_normalize = getattr(results, 'density_normalize', False)

    # Build temporal DataFrame from channel_importance
    temporal_rows = []
    for tf in timeframes:
        ch_imp = results.channel_importance.get(tf)
        if ch_imp is None:
            continue
        for i in range(len(ch_imp)):
            temporal_rows.append({
                'timeframe': tf, 'channel_idx': i,
                'feature': ch2feat.get(int(i), f'Ch{i}'),
                'mean_abs_shap': float(ch_imp[i]),
            })
    temporal = pd.DataFrame(temporal_rows)

    # Build static categorical DataFrame
    static_cat_rows = []
    for tf in timeframes:
        cat_imp = results.static_cat_importance.get(tf)
        if cat_imp is None:
            continue
        cat_imp = np.squeeze(cat_imp)
        for j, name in enumerate(results.static_cat_names):
            if j < len(cat_imp):
                val = cat_imp[j]
                static_cat_rows.append({
                    'timeframe': tf, 'feature': f'static_cat:{name}',
                    'display_name': name,
                    'mean_abs_shap': float(val) if np.ndim(val) == 0 else float(np.mean(val)),
                })
    static_cat = pd.DataFrame(static_cat_rows)

    # Build static continuous DataFrame
    static_cont_rows = []
    for tf in timeframes:
        cont_imp = results.static_cont_importance.get(tf)
        if cont_imp is None:
            continue
        cont_imp = np.squeeze(cont_imp)
        for j, name in enumerate(results.static_cont_names):
            if j < len(cont_imp):
                val = cont_imp[j]
                static_cont_rows.append({
                    'timeframe': tf, 'feature': f'static_cont:{name}',
                    'display_name': name,
                    'mean_abs_shap': float(val) if np.ndim(val) == 0 else float(np.mean(val)),
                })
    static_cont = pd.DataFrame(static_cont_rows)

    # Build categorical TS DataFrame
    cat_ts = pd.DataFrame()
    cat_ts_imp = getattr(results, 'cat_ts_per_category_importance', {})
    cat_names = getattr(results, 'cat_ts_category_names', [])
    if not cat_names and results.encoding_info:
        cat_names = get_category_names_from_encoding_info(results.encoding_info)
    if cat_ts_imp:
        cat_ts_rows = []
        for tf in timeframes:
            imp = cat_ts_imp.get(tf)
            if imp is not None:
                for j, name in enumerate(cat_names):
                    if j < len(imp):
                        cat_ts_rows.append({
                            'timeframe': tf, 'feature': f'cat_ts:{name}',
                            'display_name': name,
                            'mean_abs_shap': float(np.mean(imp[j])),
                        })
        if cat_ts_rows:
            cat_ts = pd.DataFrame(cat_ts_rows)
    # Fallback: aggregate from patient_results
    if len(cat_ts) == 0 and results.patient_results and cat_names:
        fallback = _load_cat_ts_from_pickle(pickle_path, timeframes)
        if fallback is not None:
            cat_ts = fallback

    # Patient counts per timeframe
    patient_counts = {tf: results.patient_counts.get(tf, 0) for tf in timeframes}

    return {
        'temporal': temporal,
        'static_cat': static_cat,
        'static_cont': static_cont,
        'cat_ts': cat_ts,
        'channel2feature': ch2feat,
        'timeframes': timeframes,
        'density_normalize': density_normalize,
        'patient_counts': patient_counts,
        'cat_ts_gate_values': getattr(results, 'cat_ts_gate_values', None),
    }


def figure_shap_summary_panel(
    csv_path: str,
    save_dir: str,
    pickle_path: Optional[str] = None,
    max_display: int = 15,
    save_suffix: str = "",
) -> None:
    """Paper-quality 3x2 SHAP summary panel.

    Produces a figure similar to ``visualize_shap_summary`` but with:
    - Active-only temporal SHAP evaluation data
    - Density-normalized per-measurement importance
    - Larger text for paper readability
    - Subplot labels (A-F)
    - Symmetric 3x2 layout (all panels half-width)
    - Heatmaps for all panels with timeframe columns

    Args:
        csv_path: Path to cohort_shap_all_features*.csv
        save_dir: Output directory for PNG and PDF
        pickle_path: Optional path to cohort_temporal_shap_results*.pkl
            (preferred data source — avoids CSV serialization issues)
        max_display: Maximum features to show in heatmaps and bar charts
    """
    plt.style.use(FIGSTYLE)
    plt.rcParams.update(RC_PARAMS_SUMMARY)

    # Prefer pickle (clean numpy arrays) over CSV (potential serialization issues)
    data = None
    if pickle_path and os.path.exists(pickle_path):
        # Determine timeframes from CSV or defaults
        tf_order = _TF_ORDER
        if os.path.exists(csv_path):
            csv_df = pd.read_csv(csv_path)
            available = set(csv_df['timeframe'].unique())
            tf_order = [tf for tf in _TF_ORDER if tf in available]
        data = _load_all_from_pickle(pickle_path, tf_order)
        if data is not None:
            logger.info(f"Loaded all panel data from pickle: {pickle_path}")

    if data is None:
        logger.info(f"Loading from CSV: {csv_path}")
        data = _load_csv_data(csv_path)
        # Try pickle fallback for cat_ts only
        if len(data['cat_ts']) == 0 and pickle_path and os.path.exists(pickle_path):
            cat_ts_fallback = _load_cat_ts_from_pickle(pickle_path, data['timeframes'])
            if cat_ts_fallback is not None:
                data['cat_ts'] = cat_ts_fallback

    temporal = data['temporal']
    static_cat = data['static_cat']
    static_cont = data['static_cont']
    cat_ts = data['cat_ts']
    timeframes = data['timeframes']
    density_normalize = data.get('density_normalize', False)
    patient_counts = data.get('patient_counts', {})

    if not timeframes:
        logger.error("No valid timeframes found")
        return

    # Filter out excluded/auxiliary channels from temporal data
    clinical_temporal = temporal[
        ~temporal['feature'].isin(_EXCLUDED_CHANNELS | _EBM_CHANNEL_NAMES)
    ]

    _dn_suffix = ' (per-measurement)' if density_normalize else ''
    _shap_label = 'Mean |SHAP| / measured cell' if density_normalize else 'Mean |SHAP|'

    # ========================================================================
    # Figure setup: 3x2 grid
    # ========================================================================
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.45,
                          height_ratios=[1, 1.2, 1])

    # ========================================================================
    # Panel A: Feature Importance Over Time (line plot)
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    _add_subplot_label(ax_a, 'A')

    # Aggregate mean |SHAP| across all clinical channels per timeframe
    cont_ts_by_tf = []
    for tf in timeframes:
        tf_data = clinical_temporal[clinical_temporal['timeframe'] == tf]
        cont_ts_by_tf.append(tf_data['mean_abs_shap'].sum() if len(tf_data) > 0 else 0.0)

    ax_a.plot(range(len(timeframes)), cont_ts_by_tf, linewidth=2.5,
              color='#ff0051', marker='o', markersize=6, label='Continuous TS')
    ax_a.fill_between(range(len(timeframes)), cont_ts_by_tf, alpha=0.2, color='#ff0051')

    # Categorical TS line (if data available)
    if len(cat_ts) > 0:
        cat_ts_by_tf = []
        for tf in timeframes:
            tf_data = cat_ts[cat_ts['timeframe'] == tf]
            cat_ts_by_tf.append(tf_data['mean_abs_shap'].sum() if len(tf_data) > 0 else 0.0)
        ax_a.plot(range(len(timeframes)), cat_ts_by_tf, linewidth=2.5,
                  color='#00d4aa', marker='s', markersize=5, label='Categorical TS',
                  linestyle='--')
        ax_a.fill_between(range(len(timeframes)), cat_ts_by_tf, alpha=0.15, color='#00d4aa')

    ax_a.set_xticks(range(len(timeframes)))
    ax_a.set_xticklabels(timeframes, rotation=45, ha='right')
    ax_a.set_xlabel('Timeframe')
    _shap_label_sum = 'Sum |SHAP| / measured cell' if density_normalize else 'Sum |SHAP|'
    ax_a.set_ylabel(_shap_label_sum)
    ax_a.set_title(f'Total Feature Importance Over Time{_dn_suffix}', fontweight='bold')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    # ========================================================================
    # Panel B: Top N Features — combined continuous TS + categorical TS
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    _add_subplot_label(ax_b, 'B')

    # Continuous TS: average importance across timeframes per channel
    cont_avg = (clinical_temporal
                .groupby('feature')['mean_abs_shap']
                .mean())
    cont_avg_df = cont_avg.reset_index()
    cont_avg_df.columns = ['name', 'importance']
    cont_avg_df['source'] = 'continuous'

    # Categorical TS: average importance across timeframes per category
    if len(cat_ts) > 0:
        cat_avg_series = (cat_ts
                          .groupby('display_name')['mean_abs_shap']
                          .mean())
        cat_avg_df = cat_avg_series.reset_index()
        cat_avg_df.columns = ['name', 'importance']
        cat_avg_df['source'] = 'categorical'
        combined = pd.concat([cont_avg_df, cat_avg_df], ignore_index=True)
    else:
        combined = cont_avg_df

    combined = combined.sort_values('importance', ascending=False).head(max_display)

    bar_colors = []
    for _, row in combined.iterrows():
        if row['name'] in _EBM_CHANNEL_NAMES:
            bar_colors.append('#FF9800')
        elif row['source'] == 'categorical':
            bar_colors.append('#00d4aa')
        else:
            bar_colors.append('#ff0051')

    y_pos = range(len(combined))
    channel_display_names = _clean_feature_names(list(combined['name']))
    ax_b.barh(y_pos, combined['importance'].values, color=bar_colors, alpha=0.7)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(channel_display_names)
    ax_b.set_xlabel(_shap_label)
    ax_b.set_title(f'Top {len(combined)} Features{_dn_suffix}', fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='x')
    ax_b.invert_yaxis()

    # Legend for feature types
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor='#ff0051', alpha=0.7, label='Continuous TS')]
    if len(cat_ts) > 0 and (combined['source'] == 'categorical').any():
        legend_handles.append(Patch(facecolor='#00d4aa', alpha=0.7, label='Categorical TS'))
    ax_b.legend(handles=legend_handles, loc='lower right', fontsize=9)

    # ========================================================================
    # Panel C: Categorical TS |SHAP| Heatmap
    # ========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    _add_subplot_label(ax_c, 'C')

    if len(cat_ts) > 0:
        # Average importance across timeframes, pick top N
        cat_avg = (cat_ts
                   .groupby('display_name')['mean_abs_shap']
                   .mean()
                   .sort_values(ascending=False))
        top_cats = cat_avg.head(max_display).index.tolist()

        cat_matrix = np.zeros((len(top_cats), len(timeframes)))
        for col_idx, tf in enumerate(timeframes):
            tf_data = cat_ts[cat_ts['timeframe'] == tf]
            for row_idx, cat_name in enumerate(top_cats):
                match = tf_data[tf_data['display_name'] == cat_name]
                if len(match) > 0:
                    cat_matrix[row_idx, col_idx] = match['mean_abs_shap'].values[0]

        cat_display_names = _clean_feature_names(top_cats)
        df_heat_c = pd.DataFrame(cat_matrix, index=cat_display_names, columns=timeframes)
        sns.heatmap(df_heat_c, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax_c,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8, 'label': _shap_label},
                    annot_kws={'fontsize': 10})
        ax_c.set_ylabel('')
        ax_c.set_xlabel('Timeframe')
        _cat_dn_suffix = ' (per-event)' if density_normalize else ''
        ax_c.set_title(f'Categorical TS |SHAP|{_cat_dn_suffix}', fontweight='bold')
        ax_c.tick_params(axis='y', labelsize=12)
        ax_c.tick_params(axis='x', labelsize=12)
        cat_ts_gate_values = data.get('cat_ts_gate_values')
        if cat_ts_gate_values is not None:
            gate_mean = float(np.mean(cat_ts_gate_values))
            ax_c.text(0.02, 0.02, f'Gate factor: {gate_mean:.2f}',
                      transform=ax_c.transAxes, fontsize=8, style='italic', alpha=0.7)
    else:
        ax_c.text(0.5, 0.5, 'No categorical TS data available',
                  ha='center', va='center', transform=ax_c.transAxes, fontsize=12)
        ax_c.set_title('Categorical TS |SHAP|', fontweight='bold')

    # ========================================================================
    # Panel D: Continuous TS |SHAP| Heatmap
    # ========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    _add_subplot_label(ax_d, 'D')

    # Build matrix: channels x timeframes (top N by overall importance)
    top_channels = cont_avg.sort_values(ascending=False).head(max_display).index.tolist()

    cont_matrix = np.zeros((len(top_channels), len(timeframes)))
    for col_idx, tf in enumerate(timeframes):
        tf_data = clinical_temporal[clinical_temporal['timeframe'] == tf]
        for row_idx, ch_name in enumerate(top_channels):
            match = tf_data[tf_data['feature'] == ch_name]
            if len(match) > 0:
                cont_matrix[row_idx, col_idx] = match['mean_abs_shap'].values[0]

    cont_display_names = _clean_feature_names(top_channels)
    df_heat_d = pd.DataFrame(cont_matrix, index=cont_display_names, columns=timeframes)
    sns.heatmap(df_heat_d, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax_d,
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.8, 'label': _shap_label},
                annot_kws={'fontsize': 10})
    ax_d.set_ylabel('')
    ax_d.set_xlabel('Timeframe')
    ax_d.set_title(f'Continuous TS |SHAP|{_dn_suffix}', fontweight='bold')
    ax_d.tick_params(axis='y', labelsize=12)
    ax_d.tick_params(axis='x', labelsize=12)

    # ========================================================================
    # Panel E: Static Categorical Heatmap (features × timeframes)
    # ========================================================================
    ax_e = fig.add_subplot(gs[2, 0])
    _add_subplot_label(ax_e, 'E')

    if len(static_cat) > 0:
        # Sort by mean importance across timeframes
        scat_avg = (static_cat
                    .groupby('display_name')['mean_abs_shap']
                    .mean()
                    .sort_values(ascending=False))
        top_scat = scat_avg.head(max_display).index.tolist()

        scat_matrix = np.zeros((len(top_scat), len(timeframes)))
        for col_idx, tf in enumerate(timeframes):
            tf_data = static_cat[static_cat['timeframe'] == tf]
            for row_idx, feat_name in enumerate(top_scat):
                match = tf_data[tf_data['display_name'] == feat_name]
                if len(match) > 0:
                    scat_matrix[row_idx, col_idx] = match['mean_abs_shap'].values[0]

        scat_display_names = _clean_feature_names(top_scat)
        df_heat_e = pd.DataFrame(scat_matrix, index=scat_display_names, columns=timeframes)
        sns.heatmap(df_heat_e, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax_e,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8, 'label': _shap_label},
                    annot_kws={'fontsize': 10})
        ax_e.set_ylabel('')
        ax_e.set_xlabel('Timeframe')
    ax_e.set_title('Static Categorical', fontweight='bold')

    # ========================================================================
    # Panel F: Static Continuous Heatmap (features × timeframes)
    # ========================================================================
    ax_f = fig.add_subplot(gs[2, 1])
    _add_subplot_label(ax_f, 'F')

    if len(static_cont) > 0:
        scont_avg = (static_cont
                     .groupby('display_name')['mean_abs_shap']
                     .mean()
                     .sort_values(ascending=False))
        top_scont = scont_avg.head(max_display).index.tolist()

        scont_matrix = np.zeros((len(top_scont), len(timeframes)))
        for col_idx, tf in enumerate(timeframes):
            tf_data = static_cont[static_cont['timeframe'] == tf]
            for row_idx, feat_name in enumerate(top_scont):
                match = tf_data[tf_data['display_name'] == feat_name]
                if len(match) > 0:
                    scont_matrix[row_idx, col_idx] = match['mean_abs_shap'].values[0]

        scont_display_names = _clean_feature_names(top_scont)
        df_heat_f = pd.DataFrame(scont_matrix, index=scont_display_names, columns=timeframes)
        sns.heatmap(df_heat_f, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax_f,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8, 'label': _shap_label},
                    annot_kws={'fontsize': 10})
        ax_f.set_ylabel('')
        ax_f.set_xlabel('Timeframe')
    ax_f.set_title('Static Continuous', fontweight='bold')

    # ========================================================================
    # Save
    # ========================================================================
    plt.tight_layout()
    stem = f'figure_shap_summary_panel{save_suffix}'
    _save_shap_figure(fig, save_dir, stem)
    logger.info(f"Summary panel figure saved to {save_dir}/{stem}")


# ============================================================================
# CSV export from pickle
# ============================================================================

def export_csv_from_pickle(pickle_path: str, output_path: str) -> None:
    """Regenerate cohort_shap_all_features CSV from a pickle."""
    import pickle as pkl

    print(f"Loading pickle: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        results = pkl.load(f)

    rows = []
    for tf in results.get_available_timeframes():
        ch_imp = results.channel_importance[tf]
        ch_std = results.channel_importance_std[tf]
        for i in range(len(ch_imp)):
            rows.append({
                'timeframe': tf,
                'channel_idx': i,
                'feature': results.channel2feature.get(int(i), f'Ch{i}'),
                'mean_abs_shap': float(ch_imp[i]),
                'std_abs_shap': float(ch_std[i]),
                'n_patients': results.patient_counts[tf],
            })
        cat_imp = results.static_cat_importance.get(tf)
        if cat_imp is not None:
            for j, name in enumerate(results.static_cat_names):
                if j < len(cat_imp):
                    rows.append({
                        'timeframe': tf, 'channel_idx': None,
                        'feature': f'static_cat:{name}',
                        'mean_abs_shap': float(cat_imp[j]), 'std_abs_shap': None,
                        'n_patients': results.patient_counts[tf],
                    })
        cont_imp = results.static_cont_importance.get(tf)
        if cont_imp is not None:
            for j, name in enumerate(results.static_cont_names):
                if j < len(cont_imp):
                    rows.append({
                        'timeframe': tf, 'channel_idx': None,
                        'feature': f'static_cont:{name}',
                        'mean_abs_shap': float(cont_imp[j]), 'std_abs_shap': None,
                        'n_patients': results.patient_counts[tf],
                    })
        cat_ts_imp = results.cat_ts_per_category_importance.get(tf)
        if cat_ts_imp is not None:
            for j, name in enumerate(results.cat_ts_category_names):
                if j < len(cat_ts_imp):
                    rows.append({
                        'timeframe': tf, 'channel_idx': None,
                        'feature': f'cat_ts:{name}',
                        'mean_abs_shap': float(np.mean(cat_ts_imp[j])),
                        'std_abs_shap': None,
                        'n_patients': results.patient_counts[tf],
                    })

    df = pd.DataFrame(rows)
    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} rows to {output_path}")


# ============================================================================
# Post-hoc density renormalization
# ============================================================================

def recompute_cat_ts_shap(
    pickle_path: str,
    config_name: str,
    save_dir: str,
    data_cache_path: str | None = None,
) -> None:
    """Recompute cat TS SHAP for the same patients and patch the old pickle.

    Loads the old pickle to get the patient list, runs TemporalSHAPAnalyzer
    on the current branch (raw multi-hot SHAP + density normalization), then
    patches the old pickle with ONLY the new cat_ts values. All other panels
    (continuous TS, statics) remain bit-identical.
    """
    import pickle as pkl
    import torch
    from astra.evaluation.behavior import (
        TemporalSHAPAnalyzer,
        get_holdout_pids,
    )

    print(f"Loading old pickle: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        old_results = pkl.load(f)

    old_pids = old_results.pids
    print(f"  {len(old_pids)} patients, timeframes: "
          f"{old_results.get_available_timeframes()}")

    # Load data + model via config
    import astra.utils as _utils
    from astra.utils import setup_logging
    setup_logging(logging.INFO)
    _cfg = get_cfg(_utils.PROJECT_ROOT / "configs" / config_name)
    _utils.cfg.clear()
    _utils.cfg.update(_cfg)

    if data_cache_path:
        from astra.data.caching import load_data_cache_from_path
        print(f"Loading data from explicit cache: {data_cache_path}")
        data = load_data_cache_from_path(data_cache_path)
    else:
        print(f"Loading data via config ({config_name})...")
        data = prepare_data_and_dls_cached(cfg)

    print("Loading model...")
    model, device = prepare_model(data, cfg)
    model.eval()

    # Run TemporalSHAPAnalyzer for the same PIDs
    analyzer = TemporalSHAPAnalyzer(
        model, data, data["mixed_dls"].train,
        device, max_background_samples=200,
        active_only=True, density_normalize=True,
    )

    all_holdout_pids = get_holdout_pids(data)
    # Filter to only the PIDs from the old pickle, in dataloader order
    old_pid_set = set(old_pids)
    ordered_pids = [p for p in all_holdout_pids if p in old_pid_set]
    print(f"  Matched {len(ordered_pids)}/{len(old_pids)} PIDs in holdout set")

    if len(ordered_pids) == 0:
        print("ERROR: No matching PIDs found in holdout set.")
        return

    print(f"Recomputing SHAP for {len(ordered_pids)} patients...")
    new_results = analyzer.analyze_cohort(
        data["holdout_mixed_dls"].train, ordered_pids,
        max_patients=len(ordered_pids),
        verbose=True,
    )

    # Patch: take ONLY cat_ts fields from new results into old results
    old_results.cat_ts_per_category_importance = new_results.cat_ts_per_category_importance
    old_results.cat_ts_category_names = new_results.cat_ts_category_names
    old_results.cat_ts_gate_values = getattr(new_results, 'cat_ts_gate_values', None)
    old_results.density_normalize = True

    print(f"Patched cat_ts_per_category_importance: "
          f"{len(old_results.cat_ts_per_category_importance)} timeframes")
    for tf, imp in old_results.cat_ts_per_category_importance.items():
        n = len(imp) if imp is not None else 0
        print(f"  {tf}: {n} categories")

    # Save patched pickle
    base = Path(pickle_path)
    if base.stem.endswith('_dn'):
        new_stem = f"{base.stem}_v2"
    else:
        new_stem = f"{base.stem}_dn"
    new_path = str(base.parent / f"{new_stem}{base.suffix}")
    ensure_parent_dir(new_path)
    with open(new_path, 'wb') as f:
        pkl.dump(old_results, f)
    print(f"Saved patched pickle: {new_path}")

    # Regenerate summary panel
    csv_path = str(base.parent / base.name.replace(
        'cohort_temporal_shap_results', 'cohort_shap_all_features'
    ).replace('.pkl', '.csv'))
    figure_shap_summary_panel(
        csv_path=csv_path,
        save_dir=save_dir,
        pickle_path=new_path,
        save_suffix='_v2',
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SHAP analysis figures for JMIR paper'
    )
    parser.add_argument(
        '--config', type=str, default='defaults.yaml',
        help='Config YAML filename in configs/ dir (default: defaults.yaml). '
             'Must match the config used to produce the cached SHAP values, '
             'otherwise channel counts will mismatch.',
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
    parser.add_argument(
        '--summary-panel-only', action='store_true',
        help='Regenerate summary panel figure from an existing pickle. '
             'No SHAP computation or data loading. Requires --pickle-path.',
    )
    parser.add_argument(
        '--save-suffix', type=str, default='',
        help='Suffix for output filenames (e.g. "_v2")',
    )
    parser.add_argument(
        '--recompute-cat-ts', action='store_true',
        help='Recompute categorical TS SHAP with density normalization '
             'for the same patients in an existing pickle. Patches only '
             'cat TS fields; all other panels remain bit-identical. '
             'Requires --pickle-path and loads data/model via --config.',
    )
    parser.add_argument(
        '--pickle-path', type=str, default=None,
        help='Path to cohort_temporal_shap_results*.pkl '
             '(required for --recompute-cat-ts)',
    )
    parser.add_argument(
        '--data-cache', type=str, default=None,
        help='Explicit path to data_cache_*.pkl to bypass cache key '
             'lookup (use when config has changed since the cache was built)',
    )
    args = parser.parse_args()

    # Logging setup
    from astra.utils import setup_logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Fast path: regenerate summary panel + CSV from existing pickle (no data/model)
    if args.summary_panel_only:
        if not args.pickle_path:
            parser.error("--summary-panel-only requires --pickle-path")
        base = Path(args.pickle_path)
        csv_path = str(base.parent / base.name.replace(
            'cohort_temporal_shap_results', 'cohort_shap_all_features'
        ).replace('.pkl', '.csv'))
        export_csv_from_pickle(args.pickle_path, csv_path)
        figure_shap_summary_panel(
            csv_path=csv_path,
            save_dir=OUTPUT_DIR,
            pickle_path=args.pickle_path,
            save_suffix=args.save_suffix,
        )
        return

    # Fast path: recompute cat TS SHAP only (loads data/model, but only
    # recomputes categorical TS — all other panels stay bit-identical)
    if args.recompute_cat_ts:
        if not args.pickle_path:
            parser.error("--recompute-cat-ts requires --pickle-path")
        recompute_cat_ts_shap(
            pickle_path=args.pickle_path,
            config_name=args.config,
            save_dir=OUTPUT_DIR,
            data_cache_path=args.data_cache,
        )
        return

    # Load config from configs/ dir (mutate in place so imported references stay valid)
    import astra.utils as _utils
    _cfg = get_cfg(_utils.PROJECT_ROOT / "configs" / args.config)
    _utils.cfg.clear()
    _utils.cfg.update(_cfg)

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

    # Summary panel from cohort temporal SHAP CSV (if available)
    model_name = cfg.get("model_name", "")
    temporal_shap_dir = f'reports/eval/{model_name}/temporal_shap'
    # Try active+dn variant first, then plain
    for suffix in ('_active_dn', '_dn', '_active', ''):
        csv_candidate = os.path.join(temporal_shap_dir, f'cohort_shap_all_features{suffix}.csv')
        pkl_candidate = os.path.join(temporal_shap_dir, f'cohort_temporal_shap_results{suffix}.pkl')
        if os.path.exists(csv_candidate):
            logger.info(f"Figure F: Summary Panel from {csv_candidate}...")
            figure_shap_summary_panel(
                csv_path=csv_candidate, save_dir=save_dir,
                pickle_path=pkl_candidate if os.path.exists(pkl_candidate) else None,
            )
            break
    else:
        logger.warning("No cohort_shap_all_features CSV found — skipping summary panel")

    logger.info(f"\nAll figures saved to {save_dir}/")
    logger.info(f"Random seed used: {SEED}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
