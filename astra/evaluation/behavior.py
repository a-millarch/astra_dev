#behavior.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import seaborn as sns
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from scipy import stats
import shap
from collections import OrderedDict
import time
import os
import logging
import pickle
from pathlib import Path

from astra.utils import cfg, ensure_parent_dir, save_base64
from astra.models.hybrid.training import get_backbone
from astra.data.caching import prepare_data_and_dls_cached
from astra.evaluation.utils import prepare_model, step_to_time, time_to_step, time_to_hours, get_total_steps
from astra.training.finetune import _infer_trajectory_lengths_from_batch

logger = logging.getLogger(__name__)


def _seed_shap(seed):
    """Reset RNG state for reproducible GradientExplainer results."""
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_centered_norm(data, center=0.0):
    """
    Create a TwoSlopeNorm that centers the colormap at the specified value.
    This ensures 0 is always white/neutral in diverging colormaps.
    
    Args:
        data: Array of values to determine vmin/vmax
        center: Value to center at (default 0.0)
    
    Returns:
        TwoSlopeNorm instance or None if data is constant
    """
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    
    # Handle edge cases
    if np.isnan(vmin) or np.isnan(vmax):
        return None
    if vmin == vmax:
        return None
    
    # Ensure center is within range, adjust if needed
    if center <= vmin:
        center = vmin + 1e-10
    if center >= vmax:
        center = vmax - 1e-10
    
    # If all values are on one side of center, adjust
    if vmax <= center:
        vmax = center + abs(center - vmin)
    if vmin >= center:
        vmin = center - abs(vmax - center)
    
    return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

# ============================================================================
# Utility Functions
# ============================================================================

# step_to_time, time_to_step, time_to_hours imported from astra.evaluation.utils


def create_channel_mapping(data):
    feature_values = (
        data["trainval"].complete
        .sort_values(['PID', 'FEATURE'])
        ['FEATURE']
        .drop_duplicates()
        .tolist()
    )
    channel2feature = {i: feat for i, feat in enumerate(feature_values)}
    feature2channel = {feat: i for i, feat in enumerate(feature_values)}
    return channel2feature, feature2channel


# ============================================================================
# Channel Classification & Grouping
# ============================================================================

_TEMPORAL_CHANNELS = {'elapsed_hours', 'bin_width_hours'}
_EBM_CHANNELS = {'_ebm_pred'}
_AUXILIARY_CHANNELS = {'_data_present'}
_SHAP_EXCLUDED_CHANNELS = _TEMPORAL_CHANNELS | _AUXILIARY_CHANNELS
_SHAP_CLINICAL_DISPLAY_EXCLUDED = _SHAP_EXCLUDED_CHANNELS | _EBM_CHANNELS
_GROUP_COLORS = {
    'Clinical': '#008bfb',
    'EBM': '#FF9800',
}


def classify_channels(channel2feature: Dict[int, str]) -> OrderedDict:
    """
    Classify continuous TS channels into groups: Clinical, EBM.
    Temporal and auxiliary channels are excluded (not used as model features).
    """
    groups = OrderedDict([
        ('Clinical', []),
        ('EBM', []),
    ])

    for ch_idx in sorted(channel2feature.keys()):
        feat_name = channel2feature[ch_idx]
        if feat_name in _SHAP_EXCLUDED_CHANNELS:
            continue
        elif feat_name in _EBM_CHANNELS:
            groups['EBM'].append((ch_idx, feat_name))
        else:
            groups['Clinical'].append((ch_idx, feat_name))

    return OrderedDict((k, v) for k, v in groups.items() if v)


def _get_display_channel_mask(channel2feature: Dict[int, str], n_channels: int) -> list:
    """Return list of channel indices to include in SHAP displays (Clinical + EBM only)."""
    if not channel2feature:
        return list(range(n_channels))
    return [i for i in range(n_channels)
            if channel2feature.get(i, '') not in _SHAP_EXCLUDED_CHANNELS]


def _get_grouped_channel_order(channel2feature: Dict[int, str]):
    """
    Get channel indices reordered by group (Clinical, EBM).
    Temporal and auxiliary channels are excluded.

    Returns:
        ordered_indices, ordered_labels, group_boundaries
    """
    groups = classify_channels(channel2feature)
    ordered_indices = []
    ordered_labels = []
    group_boundaries = OrderedDict()

    row = 0
    for group_name, channels in groups.items():
        start = row
        for ch_idx, feat_name in channels:
            ordered_indices.append(ch_idx)
            ordered_labels.append(feat_name)
            row += 1
        if row > start:
            group_boundaries[group_name] = (start, row)

    return ordered_indices, ordered_labels, group_boundaries


def _draw_group_separators(ax, group_boundaries: OrderedDict):
    """Draw horizontal separator lines between channel groups on axes."""
    for group_name, (start, end) in group_boundaries.items():
        if start > 0:
            ax.axhline(y=start - 0.5, color='white', linewidth=3, zorder=5)
            ax.axhline(y=start - 0.5, color='black', linewidth=1.2,
                       linestyle='--', zorder=6)


def _get_clinical_only_channel_mask(channel2feature: Dict[int, str], n_channels: int) -> list:
    """Return list of channel indices for clinical-only displays (excludes EBM, temporal, auxiliary)."""
    if not channel2feature:
        return list(range(n_channels))
    return [i for i in range(n_channels)
            if channel2feature.get(i, '') not in _SHAP_CLINICAL_DISPLAY_EXCLUDED]


def _get_clinical_only_channel_order(channel2feature: Dict[int, str]):
    """Get channel indices for clinical-only heatmaps (no EBM, no temporal/auxiliary).

    Returns:
        ordered_indices, ordered_labels
    """
    groups = classify_channels(channel2feature)
    clinical = groups.get('Clinical', [])
    ordered_indices = [ch_idx for ch_idx, feat_name in clinical]
    ordered_labels = [feat_name for ch_idx, feat_name in clinical]
    return ordered_indices, ordered_labels


def _has_ebm_channels(channel2feature: Optional[Dict[int, str]]) -> bool:
    """Check whether any EBM channel exists in the mapping."""
    if not channel2feature:
        return False
    return any(name in _EBM_CHANNELS for name in channel2feature.values())


def compute_ebm_vs_clinical_budget(
    ts_shap: np.ndarray,
    channel2feature: Dict[int, str],
) -> Optional[Dict]:
    """Compute EBM vs Clinical SHAP budget breakdown.

    Args:
        ts_shap: SHAP values — single sample [n_ch, seq_len] or cohort [n_samples, n_ch, seq_len].
        channel2feature: channel index -> feature name mapping.

    Returns:
        Dict with 'ebm_pct', 'clinical_pct', 'ebm_temporal', 'clinical_temporal',
        'ebm_total', 'clinical_total', 'total'.  None if no EBM channel present.
    """
    if not channel2feature:
        return None

    ebm_indices = [i for i, name in channel2feature.items()
                   if name in _EBM_CHANNELS and name not in _SHAP_EXCLUDED_CHANNELS]
    clinical_indices = [i for i, name in channel2feature.items()
                        if name not in _SHAP_EXCLUDED_CHANNELS and name not in _EBM_CHANNELS]

    if not ebm_indices:
        return None

    is_cohort = ts_shap.ndim == 3

    if is_cohort:
        ebm_abs = np.abs(ts_shap[:, ebm_indices, :])
        clinical_abs = np.abs(ts_shap[:, clinical_indices, :])
        ebm_total = ebm_abs.sum(axis=(1, 2)).mean()
        clinical_total = clinical_abs.sum(axis=(1, 2)).mean()
        # Sum across channels per sample, then mean across samples
        ebm_temporal = ebm_abs.sum(axis=1).mean(axis=0)
        clinical_temporal = clinical_abs.sum(axis=1).mean(axis=0)
    else:
        ebm_abs = np.abs(ts_shap[ebm_indices, :])
        clinical_abs = np.abs(ts_shap[clinical_indices, :])
        ebm_total = ebm_abs.sum()
        clinical_total = clinical_abs.sum()
        # Sum across channels per timestep
        ebm_temporal = ebm_abs.sum(axis=0)
        clinical_temporal = clinical_abs.sum(axis=0)

    total = ebm_total + clinical_total
    ebm_pct = (ebm_total / total * 100) if total > 0 else 0
    clinical_pct = (clinical_total / total * 100) if total > 0 else 0

    return {
        'ebm_total': float(ebm_total),
        'clinical_total': float(clinical_total),
        'total': float(total),
        'ebm_pct': float(ebm_pct),
        'clinical_pct': float(clinical_pct),
        'ebm_temporal': ebm_temporal,
        'clinical_temporal': clinical_temporal,
    }


def _draw_inhospital_boundary(ax, inhospital_start_step, n_steps, label=True):
    """Draw a dotted vertical line marking the prehospital/inhospital boundary.

    Only draws if ``inhospital_start_step`` is not None and within the visible
    range (0, n_steps).  Skipped for patients without prehospital data.
    """
    if inhospital_start_step is None:
        return
    if not (0 < inhospital_start_step < n_steps):
        return
    ihs_min = step_to_time(inhospital_start_step)
    ihs_label = time_to_hours(ihs_min) if ihs_min is not None else str(inhospital_start_step)
    ax.axvline(x=inhospital_start_step, color='#2196F3', linewidth=1.5,
               linestyle=':', alpha=0.8,
               label=f'Hospital arrival ({ihs_label})' if label else None)


def _draw_ebm_budget_temporal(ax, budget: Dict, n_steps: int,
                              tick_idx, tick_labels,
                              inhospital_start_step=None,
                              title: str = 'SHAP Budget Over Time: EBM vs Clinical'):
    """Draw overlapping area chart of EBM vs Clinical SHAP budget over time."""
    clinical_t = budget['clinical_temporal'][:n_steps]
    ebm_t = budget['ebm_temporal'][:n_steps]
    x = np.arange(n_steps)

    ax.plot(x, clinical_t, linewidth=2, color=_GROUP_COLORS['Clinical'],
            label=f"Clinical: {budget['clinical_pct']:.1f}%")
    ax.fill_between(x, 0, clinical_t, alpha=0.25, color=_GROUP_COLORS['Clinical'])
    ax.plot(x, ebm_t, linewidth=2, color=_GROUP_COLORS['EBM'],
            label=f"EBM: {budget['ebm_pct']:.1f}%")
    ax.fill_between(x, 0, ebm_t, alpha=0.25, color=_GROUP_COLORS['EBM'])

    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(0)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=11)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sum |SHAP|', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    _draw_inhospital_boundary(ax, inhospital_start_step, n_steps)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)


def _parse_ebm_time_label(label: str) -> float:
    """Parse EBM model time label (e.g. '10min', '6h', '14D') to hours."""
    if label.endswith('min'):
        return float(label[:-3]) / 60
    elif label.endswith('h'):
        return float(label[:-1])
    elif label.endswith('D'):
        return float(label[:-1]) * 24
    raise ValueError(f"Cannot parse EBM time label: {label}")


def load_ebm_global_importances(
    models_dir: str = 'models/ebm',
    n_steps: int = None,
    top_n: int = 20,
) -> Optional[Dict]:
    """
    Load all EBM deployment models and extract global term importances.

    Returns a dict with importance matrix aligned to bin-step time axis
    (forward-filled to match how _ebm_pred is populated), or None if no
    models are found.
    """
    if n_steps is None:
        n_steps = get_total_steps()
    models_path = Path(models_dir)
    model_files = sorted(models_path.glob('ebm_model_*.pkl'))
    if not model_files:
        logger.info("No EBM models found in %s — skipping EBM importance panel", models_dir)
        return None

    # Load each model and extract importances
    records = []  # (step, label, {term: importance})
    for mf in model_files:
        # Parse time label from filename: ebm_model_{label}.pkl
        label = mf.stem.replace('ebm_model_', '')
        try:
            hours = _parse_ebm_time_label(label)
        except ValueError:
            logger.warning("Cannot parse EBM model filename: %s", mf.name)
            continue

        step = time_to_step(hours, 'h')
        if step is None or step >= n_steps:
            continue

        try:
            with open(mf, 'rb') as f:
                model_data = pickle.load(f)
            ebm = model_data['model']
            importances = ebm.term_importances()
            term_names = list(ebm.term_names_)
            records.append((step, label, dict(zip(term_names, importances))))
        except Exception as e:
            logger.warning("Failed to load EBM model %s: %s", mf.name, e)
            continue

    if not records:
        logger.info("No valid EBM models loaded — skipping EBM importance panel")
        return None

    records.sort(key=lambda r: r[0])
    logger.info("Loaded %d EBM models for importance visualization", len(records))

    # Build union of all term names
    all_terms: set = set()
    for _, _, imp_dict in records:
        all_terms.update(imp_dict.keys())
    all_terms_list = sorted(all_terms)

    # Build importance matrix at EBM model steps: [n_terms, n_model_steps]
    model_steps = [r[0] for r in records]
    model_labels = [r[1] for r in records]
    sparse_matrix = np.zeros((len(all_terms_list), len(records)))
    term_to_idx = {t: i for i, t in enumerate(all_terms_list)}
    for col, (_, _, imp_dict) in enumerate(records):
        for term, imp in imp_dict.items():
            sparse_matrix[term_to_idx[term], col] = imp

    # Select top_n terms by max importance across all time points
    max_imp = sparse_matrix.max(axis=1)
    top_indices = np.argsort(max_imp)[::-1][:top_n]
    top_terms = [all_terms_list[i] for i in top_indices]
    top_matrix = sparse_matrix[top_indices]  # [top_n, n_model_steps]

    # Forward-fill to full n_steps grid
    full_matrix = np.zeros((len(top_terms), n_steps))
    for step_col in range(len(model_steps)):
        start = model_steps[step_col]
        end = model_steps[step_col + 1] if step_col + 1 < len(model_steps) else n_steps
        full_matrix[:, start:end] = top_matrix[:, step_col:step_col + 1]

    return {
        'importance_matrix': full_matrix,
        'feature_names': top_terms,
        'model_steps': model_steps,
        'model_labels': model_labels,
        'n_models': len(records),
    }


def _draw_ebm_importance_heatmap(ax, ebm_imp: Dict, n_steps: int,
                                  tick_idx, tick_labels,
                                  title: str = 'EBM Feature Importance Over Time'):
    """Draw heatmap of EBM global feature importances across time."""
    matrix = ebm_imp['importance_matrix'][:, :n_steps]
    names = ebm_imp['feature_names']
    model_steps = [s for s in ebm_imp['model_steps'] if s < n_steps]
    n_feat = len(names)

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Time')
    ax.set_ylabel('EBM Feature')
    ax.set_title(title, fontweight='bold')

    # Y-axis labels
    fontsize = 7 if n_feat > 15 else (8 if n_feat > 10 else 9)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(names, fontsize=fontsize)

    # X-axis shared ticks
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45)

    # Vertical markers at EBM model change points
    for s in model_steps:
        ax.axvline(x=s, color='white', linewidth=0.5, alpha=0.6)

    plt.colorbar(im, ax=ax, label='Importance', shrink=0.8)


def _draw_ebm_importance_lines(ax, ebm_imp: Dict, n_steps: int,
                                tick_idx, tick_labels, top_k: int = 5,
                                title: str = 'Top EBM Features Over Time'):
    """Draw line plot of top EBM features' importance over time."""
    matrix = ebm_imp['importance_matrix'][:, :n_steps]
    names = ebm_imp['feature_names']
    cmap = plt.cm.tab10
    x = np.arange(n_steps)

    show_k = min(top_k, len(names))
    for i in range(show_k):
        ax.plot(x, matrix[i], linewidth=2, color=cmap(i), label=names[i])
        ax.fill_between(x, matrix[i], alpha=0.1, color=cmap(i))

    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(0)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Importance')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2 if show_k > 3 else 1)
    ax.grid(True, alpha=0.3)


def visualize_ebm_importances(ebm_importances: Dict, n_steps: int = None,
                               eval_timestep: Optional[int] = None,
                               save_path: Optional[str] = None,
                               title_suffix: str = ''):
    """
    Standalone plot of EBM glassbox feature importances over time.

    Shows two panels: top feature importance lines and full importance heatmap,
    using the same time axis as the SHAP plots.

    Args:
        ebm_importances: Dict from load_ebm_global_importances().
        n_steps: Total number of bin steps (derived from config if None).
        eval_timestep: Crop time axis to this step (optional).
        save_path: Path to save the figure (optional).
        title_suffix: Appended to plot titles (e.g. ' (PID: 123)').
    """
    if ebm_importances is None:
        return
    if n_steps is None:
        n_steps = get_total_steps()

    # Respect eval_timestep cropping
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1

    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps - 1, n_ticks, dtype=int)
    tick_labels = [time_fmt[i] for i in tick_idx]

    fig, (ax_lines, ax_hm) = plt.subplots(2, 1, figsize=(22, 12),
                                           gridspec_kw={'height_ratios': [0.7, 1],
                                                        'hspace': 0.35})

    _draw_ebm_importance_lines(ax_lines, ebm_importances, n_steps,
                                tick_idx, tick_labels,
                                title=f'Top EBM Features Over Time{title_suffix}')
    _draw_ebm_importance_heatmap(ax_hm, ebm_importances, n_steps,
                                  tick_idx, tick_labels,
                                  title=f'EBM Feature Importance Over Time{title_suffix}')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Per-patient EBM local explanation visualization
# ---------------------------------------------------------------------------

def _hours_to_label(hours: float) -> str:
    """Convert hours to a readable label (e.g. 0.167 -> '10min', 1.0 -> '1h', 24.0 -> '1D')."""
    if hours < 1:
        minutes = hours * 60
        return f"{minutes:.0f}min"
    elif hours < 24:
        if hours == int(hours):
            return f"{int(hours)}h"
        return f"{hours:.1f}h"
    else:
        days = hours / 24
        if days == int(days):
            return f"{int(days)}D"
        return f"{days:.1f}D"


def _build_contribution_matrix(
    local_explanations: Dict[float, Dict],
    top_n: int,
) -> tuple:
    """
    Build a unified feature × timeframe contribution matrix from local explanations.

    Returns:
        (matrix, feature_names, sorted_hours) where:
        - matrix: [top_n, n_timeframes] signed contributions
        - feature_names: list of top_n feature names (sorted by max |contribution|)
        - sorted_hours: list of masking_hours in ascending order
    """
    sorted_hours = sorted(local_explanations.keys())

    # Build union of all feature names
    all_features: set = set()
    for data in local_explanations.values():
        all_features.update(data['feature_names'])
    all_features_list = sorted(all_features)

    # Build full matrix: [n_features, n_timeframes]
    feat_to_idx = {f: i for i, f in enumerate(all_features_list)}
    full_matrix = np.zeros((len(all_features_list), len(sorted_hours)))

    for col, h in enumerate(sorted_hours):
        data = local_explanations[h]
        for name, contrib in zip(data['feature_names'], data['contributions']):
            full_matrix[feat_to_idx[name], col] = contrib

    # Select top_n by max absolute contribution across timeframes
    max_abs = np.abs(full_matrix).max(axis=1)
    top_indices = np.argsort(max_abs)[::-1][:top_n]
    top_names = [all_features_list[i] for i in top_indices]
    top_matrix = full_matrix[top_indices]

    return top_matrix, top_names, sorted_hours


def _draw_ebm_patient_single(
    local_explanations: Dict[float, Dict],
    top_n: int,
    title_suffix: str,
    save_path: Optional[str],
):
    """Case A: Single EBM model — horizontal bar chart of feature contributions."""
    hours = list(local_explanations.keys())[0]
    data = local_explanations[hours]

    names = np.array(data['feature_names'])
    contribs = np.array(data['contributions'])

    # Sort by absolute contribution, take top_n
    sorted_idx = np.argsort(np.abs(contribs))[::-1][:top_n]
    names = names[sorted_idx]
    contribs = contribs[sorted_idx]

    # Reverse for display (highest at top)
    names = names[::-1]
    contribs = contribs[::-1]

    n_feat = len(names)
    colors = ['#d32f2f' if c > 0 else '#1976d2' for c in contribs]

    fig, ax = plt.subplots(figsize=(12, max(6, n_feat * 0.4)))
    ax.barh(range(n_feat), contribs, color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=0.8)

    fontsize = 7 if n_feat > 15 else (8 if n_feat > 10 else 9)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(names, fontsize=fontsize)
    ax.set_xlabel('Contribution (log-odds)')
    ax.set_title(
        f'EBM Feature Contributions at {_hours_to_label(hours)}'
        f' (P={data["predicted_prob"]:.3f}){title_suffix}',
        fontweight='bold',
    )

    # Intercept annotation
    ax.annotate(
        f'Intercept: {data["intercept"]:.3f}',
        xy=(0.98, 0.02), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d32f2f', alpha=0.8, label='Risk-increasing'),
        Patch(facecolor='#1976d2', alpha=0.8, label='Protective'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _draw_ebm_patient_grouped(
    local_explanations: Dict[float, Dict],
    top_n: int,
    title_suffix: str,
    save_path: Optional[str],
):
    """Case B: 2-3 EBM models — grouped horizontal bars per timeframe."""
    matrix, feature_names, sorted_hours = _build_contribution_matrix(
        local_explanations, top_n
    )
    n_feat = len(feature_names)
    n_groups = len(sorted_hours)

    # Reverse for display (highest importance at top)
    feature_names = feature_names[::-1]
    matrix = matrix[::-1]

    # Timeframe colors from a sequential palette
    cmap = plt.cm.tab10
    group_colors = [cmap(i) for i in range(n_groups)]

    bar_height = 0.8 / n_groups
    y_positions = np.arange(n_feat)

    fig, ax = plt.subplots(figsize=(14, max(6, n_feat * 0.5)))

    for i, h in enumerate(sorted_hours):
        offset = (i - n_groups / 2 + 0.5) * bar_height
        label = f'{_hours_to_label(h)} (P={local_explanations[h]["predicted_prob"]:.3f})'
        ax.barh(
            y_positions + offset, matrix[:, i],
            height=bar_height, color=group_colors[i], alpha=0.8,
            label=label,
        )

    ax.axvline(x=0, color='black', linewidth=0.8)

    fontsize = 7 if n_feat > 15 else (8 if n_feat > 10 else 9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names, fontsize=fontsize)
    ax.set_xlabel('Contribution (log-odds)')
    ax.set_title(
        f'EBM Feature Contributions Across Timeframes{title_suffix}',
        fontweight='bold',
    )

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _draw_ebm_patient_temporal(
    local_explanations: Dict[float, Dict],
    top_n: int,
    top_k_lines: int,
    title_suffix: str,
    save_path: Optional[str],
):
    """Case C: 4+ EBM models — line plot + heatmap of contributions over time."""
    matrix, feature_names, sorted_hours = _build_contribution_matrix(
        local_explanations, top_n
    )
    n_feat = len(feature_names)
    n_timeframes = len(sorted_hours)
    x_labels = [_hours_to_label(h) for h in sorted_hours]

    fig, (ax_lines, ax_hm) = plt.subplots(
        2, 1, figsize=(max(14, n_timeframes * 1.2), 14),
        gridspec_kw={'height_ratios': [0.7, 1], 'hspace': 0.35},
    )

    # --- Top panel: line plot of top-K features ---
    cmap_lines = plt.cm.tab10
    show_k = min(top_k_lines, n_feat)
    x = np.arange(n_timeframes)

    for i in range(show_k):
        ax_lines.plot(
            x, matrix[i], linewidth=2, color=cmap_lines(i),
            label=feature_names[i], marker='o', markersize=4,
        )
        ax_lines.fill_between(x, matrix[i], alpha=0.1, color=cmap_lines(i))

    ax_lines.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax_lines.set_xlim(-0.5, n_timeframes - 0.5)
    ax_lines.set_xticks(x)
    ax_lines.set_xticklabels(x_labels, rotation=45)
    ax_lines.set_xlabel('EBM Evaluation Timeframe')
    ax_lines.set_ylabel('Contribution (log-odds)')
    ax_lines.set_title(
        f'Top EBM Feature Contributions Over Time{title_suffix}',
        fontweight='bold',
    )
    ax_lines.legend(loc='best', fontsize=8, ncol=2 if show_k > 3 else 1)
    ax_lines.grid(True, alpha=0.3)

    # Add probability trajectory as secondary annotation
    probs = [local_explanations[h]['predicted_prob'] for h in sorted_hours]
    ax_prob = ax_lines.twinx()
    ax_prob.plot(x, probs, color='gray', linewidth=1.5, linestyle=':', alpha=0.6,
                 label='P(deceased)')
    ax_prob.set_ylabel('P(deceased)', color='gray', alpha=0.7)
    ax_prob.tick_params(axis='y', labelcolor='gray')
    ax_prob.set_ylim(0, 1)

    # --- Bottom panel: heatmap ---
    # Use TwoSlopeNorm for diverging colormap centered at 0
    vmax = np.abs(matrix).max()
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax_hm.imshow(
        matrix, aspect='auto', cmap='RdBu_r', norm=norm,
        interpolation='nearest',
    )

    fontsize = 7 if n_feat > 15 else (8 if n_feat > 10 else 9)
    ax_hm.set_yticks(range(n_feat))
    ax_hm.set_yticklabels(feature_names, fontsize=fontsize)
    ax_hm.set_xticks(range(n_timeframes))
    ax_hm.set_xticklabels(x_labels, rotation=45)
    ax_hm.set_xlabel('EBM Evaluation Timeframe')
    ax_hm.set_ylabel('Feature')
    ax_hm.set_title(
        f'EBM Feature Contributions Heatmap{title_suffix}',
        fontweight='bold',
    )

    plt.colorbar(im, ax=ax_hm, label='Contribution (log-odds)', shrink=0.8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_ebm_patient_importance(
    local_explanations: Dict[float, Dict],
    top_n: int = 20,
    top_k_lines: int = 5,
    pid: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize per-patient EBM feature importance across available timeframes.

    Adapts layout based on number of available EBM models:
    - Case A (1 model): Single horizontal bar chart of feature contributions
    - Case B (2-3 models): Grouped horizontal bars per timeframe
    - Case C (4+ models): Line plot + heatmap

    Args:
        local_explanations: Dict from compute_ebm_local_explanations().
            {masking_hours: {'feature_names': [...], 'contributions': [...],
             'intercept': float, 'predicted_prob': float, ...}}
        top_n: Max features to display (default 20).
        top_k_lines: Top features to show as lines in Case C (default 5).
        pid: Optional patient ID for title.
        save_path: Optional path to save figure.
    """
    if not local_explanations:
        logger.info("No EBM local explanations to visualize.")
        return

    n_models = len(local_explanations)
    title_suffix = f' (PID: {pid})' if pid else ''

    if n_models == 1:
        _draw_ebm_patient_single(local_explanations, top_n, title_suffix, save_path)
    elif n_models <= 3:
        _draw_ebm_patient_grouped(local_explanations, top_n, title_suffix, save_path)
    else:
        _draw_ebm_patient_temporal(
            local_explanations, top_n, top_k_lines, title_suffix, save_path
        )


def _get_channel_color(channel2feature, ch_idx):
    """Get display color for a channel based on its group."""
    if channel2feature is None:
        return _GROUP_COLORS['Clinical']
    name = channel2feature.get(ch_idx, '')
    if name in _EBM_CHANNELS:
        return _GROUP_COLORS['EBM']
    return _GROUP_COLORS['Clinical']


def get_static_cat_names_from_classes(classes: Dict) -> List[str]:
    """
    Extract static categorical feature names from fastai classes dict.
    
    This handles the _na columns that fastai's Categorify adds for missing indicators.
    
    Args:
        classes: Dict from dataloader, e.g.:
                 {'SEX': ['#na#', 'Female', 'Male'],
                  'FIRST_HOSPITAL': ['#na#', 'AHH', ...],
                  'ASMT_ELIX_na': ['#na#', False, True],
                  'HEIGHT_na': ['#na#', False, True],
                  'WEIGHT_na': ['#na#', False, True]}
    
    Returns:
        List of feature names in order: ['SEX', 'FIRST_HOSPITAL', 'ASMT_ELIX_na', ...]
    """
    return list(classes.keys())


# ============================================================================
# Model Wrapper for SHAP
# ============================================================================

def _select_temporal_output(logits, eval_timestep, key_padding_mask, seq_len):
    """Select output from temporal head logits based on eval_timestep mode.

    Args:
        logits: [batch, seq_len] per-timestep logits from temporal_pred_head
        eval_timestep: int index, or 'mean' for padding-aware mean across
                       valid (non-padded) timesteps. Distributes SHAP gradients
                       evenly so aggregate importance isn't dominated by the
                       last position.
        key_padding_mask: [batch, seq_len + n_static] bool (True=padding)
        seq_len: number of temporal positions (to slice mask)
    """
    if eval_timestep == 'mean':
        ts_mask = ~key_padding_mask[:, :seq_len]  # [batch, seq_len] True=valid
        ts_mask_f = ts_mask.float()
        masked_logits = logits[:, :seq_len] * ts_mask_f
        mean_logit = masked_logits.sum(dim=1) / ts_mask_f.sum(dim=1).clamp(min=1)
        return mean_logit.unsqueeze(-1)  # [batch, 1]
    return logits[:, eval_timestep].unsqueeze(-1)  # [batch, 1]


def _build_padding_mask_for_shap(x_ts, traj_lengths, n_static_tokens=0):
    """Build key_padding_mask matching model._build_traj_padding_mask().

    Args:
        x_ts: [batch, c_in, seq_len]
        traj_lengths: [batch] int64
        n_static_tokens: number of appended static tokens (never masked)
    Returns:
        key_padding_mask: [batch, seq_len + n_static_tokens] bool, True=padding
    """
    bs, _, seq_len = x_ts.shape
    device = x_ts.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    tl = traj_lengths.to(device).unsqueeze(1)                      # [batch, 1]
    ts_mask = positions >= tl                                       # [batch, seq_len]
    if n_static_tokens > 0:
        static_mask = torch.zeros(bs, n_static_tokens, dtype=torch.bool, device=device)
        return torch.cat([ts_mask, static_mask], dim=1)
    return ts_mask


def _resolve_traj_lengths(x_ts, stored_traj_lengths):
    """Return traj_lengths for the current forward batch.

    If stored_traj_lengths matches the batch size, use it directly.
    Otherwise fall back to heuristic inference (handles GradientExplainer's
    internal batching where batch sizes may differ from stored lengths).
    """
    if stored_traj_lengths is not None and stored_traj_lengths.shape[0] == x_ts.shape[0]:
        return stored_traj_lengths.to(x_ts.device)
    return _infer_trajectory_lengths_from_batch(x_ts)


class SHAPModelWrapper(nn.Module):
    """
    Unified SHAP wrapper that accepts both raw multi-hot categorical TS and
    one-hot static categoricals as separate inputs. Gradients flow through
    both embedding paths, enabling per-category SHAP attribution for cat TS
    and meaningful static categorical SHAP simultaneously.

    Input ordering (positional, conditional on flags):
        [x_ts, x_ts_cat_raw?, x_cat_onehot?, x_cont?]
    """
    def __init__(self, model, has_cat_ts=False, has_static_cat=False, has_cont=False,
                 eval_timestep=-1, traj_lengths=None, survival_mode: bool = False):
        super().__init__()
        self.model = model
        self.has_cat_ts = has_cat_ts
        self.has_static_cat = has_static_cat
        self.has_cont = has_cont
        self.eval_timestep = eval_timestep
        self.traj_lengths = traj_lengths
        self.survival_mode = survival_mode

    def forward(self, *args):
        idx = 0
        x_ts = args[idx]; idx += 1

        x_ts_cat_raw = None
        if self.has_cat_ts:
            x_ts_cat_raw = args[idx]; idx += 1

        x_cat_onehot = None
        if self.has_static_cat:
            x_cat_onehot = args[idx]; idx += 1

        x_cont = None
        if self.has_cont:
            x_cont = args[idx]; idx += 1

        nan_mask = torch.isnan(x_ts)
        if nan_mask.any():
            x_ts = x_ts.clone()
            x_ts[nan_mask] = 0

        seq_len = x_ts.shape[2]
        traj_lengths = _resolve_traj_lengths(x_ts, self.traj_lengths)
        key_padding_mask = _build_padding_mask_for_shap(x_ts, traj_lengths)

        if self.model.temporal_channel_idx is not None:
            elapsed_hours = x_ts[:, self.model.temporal_channel_idx, :]
        else:
            elapsed_hours = None

        if self.model.bin_width_channel_idx is not None:
            bin_width_hours = x_ts[:, self.model.bin_width_channel_idx, :]
        else:
            bin_width_hours = None

        x_ts_signal = x_ts[:, self.model._signal_indices, :] if self.model.exclude_channel_indices else x_ts
        if self.model.local_temporal_conv is not None:
            x_ts_signal = self.model.local_temporal_conv(x_ts_signal)
        x = self.model.W_P(x_ts_signal).transpose(1, 2)
        if self.model.bin_width_mod is not None and bin_width_hours is not None:
            x = x * self.model.bin_width_mod(bin_width_hours.unsqueeze(-1))

        # Raw multi-hot cat TS → differentiable embedding (per-category SHAP)
        if self.has_cat_ts and x_ts_cat_raw is not None and self.model.n_ts_cat > 0:
            x_ts_cat = x_ts_cat_raw.float().transpose(1, 2)

            x_ts_cat_embedded_list = []
            dim_offset = 0
            for embed_layer, (feat_name, n_classes) in zip(
                self.model.ts_cat_embeds, self.model.ts_cat_dims.items()
            ):
                feat_multi_hot = x_ts_cat[:, :, dim_offset:dim_offset + n_classes]
                feat_embedded = embed_layer(feat_multi_hot)
                x_ts_cat_embedded_list.append(feat_embedded)
                dim_offset += n_classes

            if self.model.cat_ts_combine == 'add':
                stacked = torch.stack(x_ts_cat_embedded_list, dim=0)
                if self.model.cat_ts_gate_params is not None:
                    gates = torch.sigmoid(self.model.cat_ts_gate_params)
                    stacked = stacked * gates[:, None, None, None]
                x = x + stacked.sum(dim=0)
            else:
                x_ts_cat_embedded = torch.cat(x_ts_cat_embedded_list, dim=-1)
                x = torch.cat([x, x_ts_cat_embedded], dim=-1)

        # One-hot static categoricals → differentiable matmul embedding
        if x_cat_onehot is not None and x_cat_onehot.shape[1] > 0:
            x_cat_embedded_list = []
            for i, emb in enumerate(self.model.embeds):
                oh_i = x_cat_onehot[:, i, :emb.num_embeddings]
                emb_w = emb.weight
                x_cat_i = torch.matmul(oh_i, emb_w)
                x_cat_embedded_list.append(x_cat_i.unsqueeze(1))

            x_cat_embedded = torch.cat(x_cat_embedded_list, dim=1)
            x = torch.cat([x, x_cat_embedded], 1)

        if x_cont is not None and x_cont.shape[1] > 0:
            if self.model.cont_projections is not None:
                x_cont_emb = torch.stack([
                    proj(x_cont[:, i:i+1]) for i, proj in enumerate(self.model.cont_projections)
                ], dim=1)
            else:
                x_cont_emb = self.model.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_emb], 1)

        ts_padding_mask = key_padding_mask[:, :seq_len]
        x = self.model.pos_enc(x, elapsed_hours=elapsed_hours, ts_padding_mask=ts_padding_mask)
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)

        n_static = x.shape[1] - key_padding_mask.shape[1]
        if n_static > 0:
            static_mask = torch.zeros(
                key_padding_mask.shape[0], n_static,
                dtype=torch.bool, device=key_padding_mask.device,
            )
            key_padding_mask = torch.cat([key_padding_mask, static_mask], dim=1)

        attn_mask = self.model.causal_mask if self.model.causal else None
        x = self.model.transformer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = x * (~key_padding_mask).unsqueeze(-1).float()

        if self.model.temporal_head_enabled and self.model.temporal_pred_head is not None:
            logits = self.model.temporal_pred_head(x)
            if self.survival_mode:
                eval_t = self.eval_timestep if isinstance(self.eval_timestep, int) and self.eval_timestep >= 0 else logits.shape[1] - 1
                hazards = torch.sigmoid(logits[:, :eval_t + 1])
                log_surv = torch.sum(torch.log1p(-hazards + 1e-7), dim=1)
                surv = torch.exp(log_surv)
                return (1.0 - surv).unsqueeze(-1)
            return _select_temporal_output(logits, self.eval_timestep, key_padding_mask,
                                           self.model.seq_len)
        return self._apply_head(x, key_padding_mask)

    def _apply_head(self, x, key_padding_mask):
        if self.model.head_pool == 'mean_cat':
            x_temporal = x[:, :self.model.seq_len, :]
            x_static = x[:, self.model.seq_len:, :]
            if key_padding_mask is not None:
                ts_mask = ~key_padding_mask[:, :self.model.seq_len]
                ts_mask_f = ts_mask.unsqueeze(-1).float()
                x_pooled = (x_temporal * ts_mask_f).sum(dim=1) / ts_mask_f.sum(dim=1).clamp(min=1)
            else:
                x_pooled = x_temporal.mean(dim=1)
            x = torch.cat([x_pooled, x_static.reshape(x.shape[0], -1)], dim=1)
        return self.model.head(x)


def embed_categorical_features(model, x_cat):
    """
    Convert static categorical features to one-hot encodings for SHAP.

    Returns one-hot tensors [batch, n_cat, max_classes] with requires_grad=True.
    The model wrapper will perform matmul(one_hot, embedding.weight) for differentiable embedding.

    This allows GradientExplainer to compute meaningful gradients through the one-hot probability
    differences between foreground and background, rather than through the pre-computed embedding
    lookup (which gave near-zero SHAP for low-cardinality static categories).
    """
    if x_cat is None or x_cat.shape[1] == 0:
        return None

    # Get max embedding class count for padding
    max_classes = max(emb.num_embeddings for emb in model.embeds)

    # Convert to one-hot: [batch, n_cat, max_classes]
    onehot_list = []
    for i, emb in enumerate(model.embeds):
        # One-hot encode this categorical feature
        oh = F.one_hot(x_cat[:, i].long(), num_classes=emb.num_embeddings).float()  # [batch, num_classes]

        # Pad to max_classes if needed
        if emb.num_embeddings < max_classes:
            pad_size = max_classes - emb.num_embeddings
            oh = F.pad(oh, (0, pad_size), value=0.0)  # [batch, max_classes]

        onehot_list.append(oh.unsqueeze(1))  # [batch, 1, max_classes]

    x_cat_onehot = torch.cat(onehot_list, dim=1)  # [batch, n_cat, max_classes]
    x_cat_onehot.requires_grad = True
    return x_cat_onehot


# ============================================================================
# Data Extraction
# ============================================================================

def extract_data_from_dataloader(dataloader, max_samples=None, device='cpu'):
    # Handle DataLoaders (has .train/.valid) vs single DataLoader
    if hasattr(dataloader, 'train'):
        dataloader = dataloader.train

    all_ts, all_ts_cat, all_cat, all_cont, all_y, all_traj = [], [], [], [], [], []
    n_samples = 0

    for batch in dataloader:
        if max_samples is not None and n_samples >= max_samples:
            break
        inputs, targets = batch
        x_ts, x_tab, x_ts_cat = inputs[0], inputs[1], inputs[2]
        x_cat, x_cont = x_tab

        # Extract trajectory lengths (4-element tuple) with backward compat
        if len(inputs) >= 4:
            traj_lengths = inputs[3]
        else:
            traj_lengths = _infer_trajectory_lengths_from_batch(x_ts)

        all_ts.append(x_ts.cpu())
        all_ts_cat.append(x_ts_cat.cpu())
        all_cat.append(x_cat.cpu())
        all_cont.append(x_cont.cpu())
        all_y.append(targets.cpu())
        all_traj.append(traj_lengths.cpu())
        n_samples += x_ts.shape[0]

    x_ts_full = torch.cat(all_ts, dim=0)
    x_ts_cat_full = torch.cat(all_ts_cat, dim=0)
    x_cat_full = torch.cat(all_cat, dim=0)
    x_cont_full = torch.cat(all_cont, dim=0)
    y_full = torch.cat(all_y, dim=0)
    traj_full = torch.cat(all_traj, dim=0)

    if max_samples is not None and x_ts_full.shape[0] > max_samples:
        x_ts_full = x_ts_full[:max_samples]
        x_ts_cat_full = x_ts_cat_full[:max_samples]
        x_cat_full = x_cat_full[:max_samples]
        x_cont_full = x_cont_full[:max_samples]
        y_full = y_full[:max_samples]
        traj_full = traj_full[:max_samples]

    return (x_ts_full.to(device), x_ts_cat_full.to(device), x_cat_full.to(device),
            x_cont_full.to(device), y_full.to(device), traj_full.to(device))


def get_holdout_pids(data, max_samples=None, specific_pids: List = None):
    """
    Extract PIDs from holdout dataset in the order they appear in the dataloader.

    Args:
        data: Data dict containing 'holdout' TSDS object
        max_samples: Maximum number of samples (should match what was used in SHAP calculation)
        specific_pids: List of specific PIDs to include. If provided, returns only these PIDs
                       in the order they appear in the holdout set.

    Returns:
        List of PIDs in dataloader order
    """
    # Get PIDs from holdout tab_df (which is used by the dataloader)
    holdout_pids = data["holdout"].tab_df['PID'].tolist()

    if specific_pids is not None:
        # Return only specific PIDs, preserving their order in holdout set
        holdout_pids = [pid for pid in holdout_pids if pid in specific_pids]
    elif max_samples is not None and len(holdout_pids) > max_samples:
        holdout_pids = holdout_pids[:max_samples]

    return holdout_pids


def compute_inhospital_start_steps(data, pids):
    """Compute the step index where inhospital data starts for each PID.

    Returns an array of step indices (int), or None where the patient has no
    prehospital data (``prehospital_start`` is NaT → boundary at step 0, not
    meaningful to plot).

    Args:
        data: Data dict with ``data["holdout"].base`` containing timestamps.
        pids: List of PIDs in sample order.

    Returns:
        np.ndarray of shape ``[len(pids)]`` with dtype ``object`` — int step
        values or ``None`` per sample.
    """
    base = data["holdout"].base
    if "inhospital_start" not in base.columns or "start" not in base.columns:
        return None
    pid_col = base.set_index("PID")
    result = np.empty(len(pids), dtype=object)
    for i, pid in enumerate(pids):
        if pid not in pid_col.index:
            result[i] = None
            continue
        row = pid_col.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        ihs = row.get("inhospital_start")
        start = row.get("start")
        phs = row.get("prehospital_start")
        # Skip patients without prehospital data
        if pd.isna(phs):
            result[i] = None
            continue
        if pd.isna(ihs) or pd.isna(start):
            result[i] = None
            continue
        delta_min = (ihs - start).total_seconds() / 60
        if delta_min <= 0:
            result[i] = None
            continue
        step = time_to_step(delta_min, 'min')
        result[i] = step
    return result


def get_sample_idx_for_pid(pids: List, target_pid: Union[int, str]) -> Optional[int]:
    """
    Find the sample index for a given PID.
    
    Args:
        pids: List of PIDs in dataloader order
        target_pid: PID to find
    
    Returns:
        Sample index or None if not found
    """
    try:
        return pids.index(target_pid)
    except ValueError:
        return None


def get_pid_for_sample_idx(pids: List, sample_idx: int) -> Optional[Union[int, str]]:
    """
    Get the PID for a given sample index.
    
    Args:
        pids: List of PIDs in dataloader order
        sample_idx: Sample index
    
    Returns:
        PID or None if index out of range
    """
    if 0 <= sample_idx < len(pids):
        return pids[sample_idx]
    return None


# ============================================================================
# SHAP Calculation
# ============================================================================

def calculate_shap_from_dataloaders(model, background_loader, test_loader, encoding_info,
                                     device='cuda', max_background_samples=200, max_test_samples=100,
                                     compute_per_category_shap=True, specific_pids: List = None,
                                     all_pids: List = None, eval_timestep: int = -1,
                                     inhospital_start_steps: np.ndarray = None):
    """
    Calculate SHAP values for all model inputs.

    Args:
        compute_per_category_shap: If True, compute SHAP on raw multi-hot categorical TS
                                   to get per-category attributions. If False, compute on
                                   embedded representation (faster but less granular).
        specific_pids: List of specific PIDs to include. If provided, only these samples
                       will be used for SHAP calculation.
        all_pids: List of all PIDs in the test loader (in order). Required if specific_pids
                  is provided, to map PIDs to sample indices.
        eval_timestep: For temporal head models, which sequence position to evaluate.
                       Default -1 auto-switches to 'mean' when temporal head is detected.
                       Options:
                         - 'mean': padding-aware mean across valid timesteps (recommended
                           for aggregate SHAP — distributes gradients evenly)
                         - int >= 0: specific timestep index
                         - -1: last position (auto-converts to 'mean' for temporal head)
                       For per-timeframe analysis, use a fixed clinical timepoint:
                           from astra.evaluation.utils import time_to_step
                           eval_timestep=time_to_step(24, 'h')  # prediction at 24 h
        inhospital_start_steps: Per-sample step index where inhospital data starts
                                (from ``compute_inhospital_start_steps``). None entries
                                mean the patient has no prehospital data. Stored in
                                ``shap_results['test_data']`` for visualization.
    """
    logger.info("Extracting background data...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    bg_ts, bg_ts_cat, bg_cat, bg_cont, bg_y, bg_traj = extract_data_from_dataloader(
        background_loader, max_samples=max_background_samples, device=device)
    
    logger.info("  Background samples: %s", bg_ts.shape[0])
    logger.debug("    Continuous TS: %s", bg_ts.shape)
    logger.debug("    Categorical TS: %s", bg_ts_cat.shape)
    logger.debug("    Static categorical: %s", bg_cat.shape)
    logger.debug("    Static continuous: %s", bg_cont.shape)
    
    logger.info("Extracting test data...")
    # If specific_pids provided, extract enough samples to include them all
    extraction_max = None if specific_pids is not None else max_test_samples
    test_ts, test_ts_cat, test_cat, test_cont, test_y, test_traj = extract_data_from_dataloader(
        test_loader, max_samples=extraction_max, device=device)
    logger.info("  Test samples extracted: %s", test_ts.shape[0])

    # Filter to specific PIDs if provided
    if specific_pids is not None and all_pids is not None:
        # Find indices of specific PIDs in the all_pids list
        pid_indices = [i for i, pid in enumerate(all_pids) if pid in specific_pids]
        # Ensure indices are within extracted data range
        pid_indices = [i for i in pid_indices if i < test_ts.shape[0]]
        if not pid_indices:
            raise ValueError("No valid indices found for specific PIDs")
        pid_indices = torch.tensor(pid_indices, device=device)
        test_ts = test_ts[pid_indices]
        test_ts_cat = test_ts_cat[pid_indices]
        test_cat = test_cat[pid_indices]
        test_cont = test_cont[pid_indices]
        test_y = test_y[pid_indices]
        test_traj = test_traj[pid_indices]
        if inhospital_start_steps is not None:
            inhospital_start_steps = inhospital_start_steps[pid_indices.cpu().numpy()]
        logger.info("  Filtered to %s specific PIDs", len(pid_indices))

    logger.info("  Final test samples: %s", test_ts.shape[0])

    model.eval()
    model = model.to(device)
    
    logger.info("Model has %s static categorical embeddings", len(model.embeds))
    for i, emb in enumerate(model.embeds):
        logger.debug("  Embed %s: %s classes -> %s dim", i, emb.num_embeddings, emb.embedding_dim)
    
    has_cat_ts = model.n_ts_cat > 0 and bg_ts_cat is not None and bg_ts_cat.numel() > 0
    n_static_cat = bg_cat.shape[1]
    
    # Convert static categorical to one-hot (always)
    # Returns [batch, n_cat, max_classes] with requires_grad=True for SHAP
    bg_cat_onehot = embed_categorical_features(model, bg_cat) if n_static_cat > 0 else None
    test_cat_onehot = embed_categorical_features(model, test_cat) if n_static_cat > 0 else None
    
    if model.temporal_head_enabled:
        if eval_timestep == -1:
            eval_timestep = 'mean'
            logger.info("  Temporal head detected: auto-switching eval_timestep to 'mean' "
                        "(padding-aware average across valid timesteps)")
        else:
            logger.info("  Temporal head: eval_timestep=%s", eval_timestep)

    has_static_cat = bg_cat_onehot is not None
    has_cont = bg_cont.shape[1] > 0

    if has_static_cat:
        logger.debug("  Static categorical one-hot: %s", bg_cat_onehot.shape)

    logger.info("  Using SHAPModelWrapper (raw cat TS + one-hot static cats)")
    wrapped_model = SHAPModelWrapper(
        model, has_cat_ts=has_cat_ts,
        has_static_cat=has_static_cat, has_cont=has_cont,
        eval_timestep=eval_timestep, traj_lengths=test_traj,
    )

    bg_inputs = [bg_ts]
    test_inputs = [test_ts]
    if has_cat_ts:
        bg_inputs.append(bg_ts_cat.float().requires_grad_(True))
        test_inputs.append(test_ts_cat.float().requires_grad_(True))
    if has_static_cat:
        bg_inputs.append(bg_cat_onehot)
        test_inputs.append(test_cat_onehot)
    if has_cont:
        bg_inputs.append(bg_cont)
        test_inputs.append(test_cont)
    
    shap_seed = cfg.get("evaluation", {}).get("shap_seed", 42)
    shap_nsamples = cfg.get("evaluation", {}).get("shap_nsamples", 200)
    logger.info("Creating SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(wrapped_model, bg_inputs)

    logger.info("Calculating SHAP values...")
    _seed_shap(shap_seed)
    shap_values = explainer.shap_values(test_inputs, nsamples=shap_nsamples)
    logger.info("SHAP calculation complete!")

    # For multi-output models (e.g. 2-class), GradientExplainer returns
    # [[sv_per_input_class0], [sv_per_input_class1]].
    # Select class 1 (mortality) by default.
    if isinstance(shap_values, list) and len(shap_values) > 0:
        if isinstance(shap_values[0], list):
            n_classes = len(shap_values)
            selected_class = min(1, n_classes - 1)  # class 1 if available
            logger.info("  Multi-output model: %s classes, selecting class %s", n_classes, selected_class)
            shap_values = shap_values[selected_class]

    # Strip trailing singleton class dim (GradientExplainer format b)
    if isinstance(shap_values, list):
        shap_values = [
            sv.squeeze(-1) if isinstance(sv, np.ndarray) and sv.ndim > 1 and sv.shape[-1] == 1
            else sv for sv in shap_values
        ]

    logger.debug("SHAP value shapes:")
    for i, sv in enumerate(shap_values):
        logger.debug("  shap_values[%s]: %s", i, sv.shape)
    
    # Parse SHAP values
    idx = 0
    ts_shap = shap_values[idx]; idx += 1
    
    cat_ts_shap_per_category = None
    cat_ts_shap = None

    if has_cat_ts:
        cat_ts_shap_per_category = shap_values[idx]
        logger.debug("  cat_ts_shap_per_category: %s", cat_ts_shap_per_category.shape)
        cat_ts_shap = np.abs(cat_ts_shap_per_category).mean(axis=1)
        idx += 1

    cat_shap, cat_shap_onehot = None, None
    if has_static_cat:
        cat_shap_onehot = shap_values[idx]
        logger.debug("  cat_shap_onehot shape (one-hot SHAP): %s", cat_shap_onehot.shape)
        # cat_shap_onehot is [n_samples, n_cat, max_classes]
        # Aggregate over the one-hot class dimension: sum absolute value per feature
        # This gives overall feature importance (sum of all class probabilities' contribution)
        cat_shap = np.abs(cat_shap_onehot).sum(axis=2)  # [n_samples, n_cat]
        logger.debug("  cat_shap after sum(|one_hot|): %s", cat_shap.shape)
        idx += 1
    
    cont_shap = shap_values[idx] if has_cont else None

    # Zero SHAP values at padding positions (safety net — model already zeros
    # padding output, but explicit zeroing ensures clean SHAP values)
    test_traj_np = test_traj.cpu().numpy()
    for i in range(ts_shap.shape[0]):
        tl = int(test_traj_np[i])
        ts_shap[i, :, tl:] = 0.0
        if cat_ts_shap_per_category is not None:
            cat_ts_shap_per_category[i, :, tl:] = 0.0
        if cat_ts_shap is not None:
            cat_ts_shap[i, tl:] = 0.0

    return {
        'ts_shap': ts_shap,
        'cat_ts_shap': cat_ts_shap,
        'cat_ts_shap_per_category': cat_ts_shap_per_category,
        'cat_shap': cat_shap,
        'cat_shap_onehot': cat_shap_onehot,
        'cont_shap': cont_shap,
        'n_static_cat': n_static_cat,
        'eval_timestep': eval_timestep,  # stored for visualization cropping
        'test_data': {
            'ts': test_ts.cpu().numpy(),
            'ts_cat': test_ts_cat.cpu().numpy(),
            'cat': test_cat.cpu().numpy(),
            'cont': test_cont.cpu().numpy(),
            'y': test_y.cpu().numpy(),
            'traj_lengths': test_traj.cpu().numpy(),
            'inhospital_start_steps': inhospital_start_steps
        },
        'background_data': {
            'ts': bg_ts.cpu().numpy(),
            'ts_cat': bg_ts_cat.cpu().numpy(),
            'cat': bg_cat.cpu().numpy(),
            'cont': bg_cont.cpu().numpy(),
            'y': bg_y.cpu().numpy(),
            'traj_lengths': bg_traj.cpu().numpy()
        },
        'encoding_info': encoding_info
    }


# ============================================================================
# Debug function
# ============================================================================

def debug_shap_data(shap_results, feature_names_cat=None, feature_names_cont=None):
    """Debug function to identify data shape mismatches."""
    logger.debug("=" * 60)
    logger.debug("SHAP RESULTS DEBUG")
    logger.debug("=" * 60)
    
    logger.debug("1. Time Series SHAP:")
    logger.debug("   ts_shap shape: %s", shap_results['ts_shap'].shape)
    
    logger.debug("2. Categorical TS SHAP:")
    if shap_results['cat_ts_shap'] is not None:
        logger.debug("   cat_ts_shap shape: %s", shap_results['cat_ts_shap'].shape)
    else:
        logger.debug("   cat_ts_shap: None")
    
    logger.debug("3. Static Categorical SHAP:")
    if shap_results['cat_shap'] is not None:
        logger.debug("   cat_shap shape: %s", shap_results['cat_shap'].shape)
        n_from_shap = shap_results['cat_shap'].shape[1]
        logger.debug("   Number of static cat features (from SHAP): %s", n_from_shap)
    else:
        logger.debug("   cat_shap: None")
        n_from_shap = 0
    
    logger.debug("   feature_names_cat provided: %s", feature_names_cat)
    n_from_names = len(feature_names_cat) if feature_names_cat else 0
    logger.debug("   Number of names provided: %s", n_from_names)
    
    if n_from_shap != n_from_names:
        logger.warning("   ⚠️  MISMATCH! SHAP has %s features but %s names provided", n_from_shap, n_from_names)
        logger.warning("   TIP: Use get_static_cat_names_from_classes(data['classes']) to get all names")
    
    logger.debug("4. Static Continuous SHAP:")
    if shap_results['cont_shap'] is not None:
        logger.debug("   cont_shap shape: %s", shap_results['cont_shap'].shape)
    
    logger.debug("5. Test Data Shapes:")
    logger.debug("   cat: %s", shap_results['test_data']['cat'].shape)
    logger.debug("   cont: %s", shap_results['test_data']['cont'].shape)
    
    logger.debug("6. Encoding Info (Categorical TS):")
    enc = shap_results.get('encoding_info', {})
    logger.debug("   Keys: %s", list(enc.keys()))
    if 'feature_ranges' in enc:
        logger.debug("   feature_ranges:")
        for feat, (start, end) in enc['feature_ranges'].items():
            logger.debug("      %s: indices %s-%s (%s categories)", feat, start, end, end-start)
    if 'category_labels' in enc:
        logger.debug("   category_labels:")
        for feat, labels in enc['category_labels'].items():
            logger.debug(f"      {feat}: {len(labels)} labels - {labels[:3]}..." if len(labels) > 3 else f"      {feat}: {labels}")
    logger.debug("=" * 60)


def get_category_names_from_encoding_info(encoding_info: Dict) -> List[str]:
    """
    Extract actual category names from encoding_info.
    
    UPDATED: Now reads from 'category_labels' key which stores actual names.
    
    Returns list like ['medication:Aspirin', 'medication:Ibuprofen', ..., 'procedures:X-ray', ...]
    """
    category_names = []
    feature_ranges = encoding_info.get('feature_ranges', {})
    category_labels = encoding_info.get('category_labels', {})
    
    for feat_name, (start, end) in feature_ranges.items():
        n_cats = end - start
        
        if feat_name in category_labels:
            # Use actual labels from encoder
            labels = category_labels[feat_name]
            for i, label in enumerate(labels[:n_cats]):
                category_names.append(f"{feat_name}:{label}")
            # Pad if fewer labels than expected
            for i in range(len(labels), n_cats):
                category_names.append(f"{feat_name}:cat_{i}")
        else:
            # Fallback to indices
            for i in range(n_cats):
                category_names.append(f"{feat_name}:cat_{i}")
    
    return category_names


# ============================================================================
# Visualization
# ============================================================================

def visualize_shap_individual(shap_results: Dict, sample_idx: int = None,
                               pid: Union[int, str] = None,
                               holdout_pids: List = None,
                               channel2feature: Dict[int, str] = None,
                               feature_names_cat: List[str] = None,
                               feature_names_cont: List[str] = None,
                               class_idx: int = 1, save_path: str = None,
                               eval_timestep: Optional[int] = None):
    """
    Visualize SHAP values for individual sample.

    Args:
        sample_idx: Direct index into the test data (0-based)
        pid: Patient ID to visualize. If provided, will look up the sample_idx.
             Requires holdout_pids to be provided.
        holdout_pids: List of PIDs in dataloader order. Required if using pid parameter.
        feature_names_cat: List of static categorical feature names.
        class_idx: Which output class to show SHAP values for (default 1 for binary)
        save_path: Path to save the figure
        eval_timestep: Crop time axis to this step (default: read from shap_results).

    Note: Either sample_idx or (pid + holdout_pids) must be provided.
    """
    # Resolve sample_idx from PID if provided
    if pid is not None:
        if holdout_pids is None:
            raise ValueError("holdout_pids must be provided when using pid parameter. "
                           "Use get_holdout_pids(data, max_samples) to get the PID list.")
        sample_idx = get_sample_idx_for_pid(holdout_pids, pid)
        if sample_idx is None:
            available_pids = holdout_pids[:10]
            raise ValueError(f"PID {pid} not found in holdout data. "
                           f"First 10 available PIDs: {available_pids}...")
        logger.info("Found PID %s at sample index %s", pid, sample_idx)
    elif sample_idx is None:
        sample_idx = 0
        logger.info("No sample_idx or pid provided, using sample_idx=0")
    
    # Get PID for title if available
    display_pid = None
    if holdout_pids is not None and sample_idx < len(holdout_pids):
        display_pid = holdout_pids[sample_idx]
    
    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_channels, n_steps = ts_shap.shape

    # Crop time axis to eval_timestep — steps beyond have ~0 SHAP due to causal masking.
    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[..., :n_steps]

    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps-1, n_ticks, dtype=int)

    # Title with PID if available
    title_suffix = f" (PID: {display_pid})" if display_pid is not None else f" (Sample {sample_idx})"

    # EBM two-view: compute budget and adjust layout
    budget = compute_ebm_vs_clinical_budget(ts_shap, channel2feature)
    has_ebm = budget is not None

    # Compute inhospital boundary step (used across multiple plots)
    _ihs_steps = shap_results.get('test_data', {}).get('inhospital_start_steps')
    _ihs = int(_ihs_steps[sample_idx]) if (_ihs_steps is not None
               and sample_idx < len(_ihs_steps) and _ihs_steps[sample_idx] is not None) else None

    # Helper: move legend from data axes into the dedicated col-2 legend axes
    def _legend_to_col2(src_ax, row):
        leg_ax = fig.add_subplot(gs[row, 2])
        leg_ax.axis('off')
        handles, labels = src_ax.get_legend_handles_labels()
        old = src_ax.get_legend()
        if old is not None:
            old.remove()
        if handles:
            leg_ax.legend(handles, labels, loc='upper left',
                          fontsize=11, borderaxespad=0)

    if has_ebm:
        fig = plt.figure(figsize=(22, 23), constrained_layout=True)
        gs = fig.add_gridspec(6, 3, hspace=0.15, wspace=0.05,
                              height_ratios=[0.7, 1, 1, 1, 1, 1],
                              width_ratios=[1, 1, 0.04])
        row_offset = 1
        # Row 0: EBM budget over time
        ax_budget = fig.add_subplot(gs[0, 0:2])
        _draw_ebm_budget_temporal(ax_budget, budget, n_steps, tick_idx,
                                  [time_fmt[i] for i in tick_idx],
                                  inhospital_start_step=_ihs,
                                  title=f'SHAP Budget Over Time{title_suffix}')
        _legend_to_col2(ax_budget, 0)
    else:
        fig = plt.figure(figsize=(22, 20), constrained_layout=True)
        gs = fig.add_gridspec(5, 3, hspace=0.15, wspace=0.05,
                              height_ratios=[1, 1, 1, 1, 1],
                              width_ratios=[1, 1, 0.04])
        row_offset = 0

    # Plot 1: TS importance over time
    ax1 = fig.add_subplot(gs[0 + row_offset, 0:2])

    if has_ebm:
        # Two-view: separate Clinical and EBM lines
        clinical_ch = _get_clinical_only_channel_mask(channel2feature, n_channels)
        ebm_ch = [i for i, name in channel2feature.items() if name in _EBM_CHANNELS]

        clinical_ts_avg = np.abs(ts_shap[clinical_ch]).mean(axis=0)
        ax1.plot(clinical_ts_avg, linewidth=2, color=_GROUP_COLORS['Clinical'],
                 label='Clinical channels')
        ax1.fill_between(range(len(clinical_ts_avg)), clinical_ts_avg, alpha=0.2,
                         color=_GROUP_COLORS['Clinical'])

        if ebm_ch:
            ebm_ts_avg = np.abs(ts_shap[ebm_ch]).mean(axis=0)
            ax1.plot(ebm_ts_avg, linewidth=2, color=_GROUP_COLORS['EBM'],
                     label='EBM (_ebm_pred)', linestyle='--')
            ax1.fill_between(range(len(ebm_ts_avg)), ebm_ts_avg, alpha=0.2,
                             color=_GROUP_COLORS['EBM'])
    else:
        ts_shap_avg = np.abs(ts_shap).mean(axis=0)
        ax1.plot(ts_shap_avg, linewidth=2, color='#ff0051', label='Continuous TS')
        ax1.fill_between(range(len(ts_shap_avg)), ts_shap_avg, alpha=0.3, color='#ff0051')

    if shap_results['cat_ts_shap'] is not None:
        cat_ts = shap_results['cat_ts_shap'][sample_idx]
        if cat_ts.ndim == 2:
            cat_ts = cat_ts[..., min(class_idx, cat_ts.shape[-1] - 1)]
        cat_ts = cat_ts[:n_steps]  # crop to eval_timestep
        ax1.plot(cat_ts, linewidth=2, color='#00d4aa', label='Categorical TS', linestyle='--')
        ax1.fill_between(range(len(cat_ts)), cat_ts, alpha=0.2, color='#00d4aa')

    ax1.set_xlabel('Time', fontsize=12); ax1.set_ylabel('mean |SHAP Value|', fontsize=12)
    ax1.set_title(f'TS SHAP Over Time{title_suffix}, Class {class_idx}', fontweight='bold', fontsize=14)
    ax1.set_xticks(tick_idx); ax1.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45, fontsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    _draw_inhospital_boundary(ax1, _ihs, n_steps)
    ax1.legend()  # temporary; transferred to col 2 below
    _legend_to_col2(ax1, 0 + row_offset)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Continuous TS heatmap — clinical-only when EBM present
    ax2 = fig.add_subplot(gs[1 + row_offset, 0:2])
    if channel2feature and has_ebm:
        ordered_idx, ordered_labels = _get_clinical_only_channel_order(channel2feature)
        ts_shap_display = ts_shap[ordered_idx]
        n_display = len(ordered_idx)
    elif channel2feature:
        ordered_idx, ordered_labels, group_bounds = _get_grouped_channel_order(channel2feature)
        ts_shap_display = ts_shap[ordered_idx]
        n_display = len(ordered_idx)
    else:
        ts_shap_display = ts_shap
        ordered_labels = [f'Ch{i}' for i in range(n_channels)]
        n_display = n_channels
    norm2 = get_centered_norm(ts_shap_display, center=0.0)
    im = ax2.imshow(ts_shap_display, aspect='auto', cmap='RdBu_r', interpolation='nearest', norm=norm2)
    ax2.set_xlabel('Time', fontsize=12); ax2.set_ylabel('Channel', fontsize=12)
    heatmap_title = 'Clinical Continuous TS SHAP Heatmap' if has_ebm else 'Continuous TS SHAP Heatmap (grouped)'
    ax2.set_title(heatmap_title, fontweight='bold', fontsize=14)
    if n_display <= 40:
        ax2.set_yticks(range(n_display))
        ax2.set_yticklabels(ordered_labels, fontsize=8 if n_display > 25 else 10)
    else:
        step = max(1, n_display // 30)
        yticks = list(range(0, n_display, step))
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([ordered_labels[i] for i in yticks], fontsize=8)
    ax2.set_xticks(tick_idx); ax2.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45, fontsize=11)
    _draw_inhospital_boundary(ax2, _ihs, n_steps, label=False)
    cax2 = fig.add_subplot(gs[1 + row_offset, 2])
    fig.colorbar(im, cax=cax2, label='SHAP Value')
    if not has_ebm and channel2feature:
        _draw_group_separators(ax2, group_bounds)

    # Plot 3: Categorical TS heatmap - SHAP values with centered colormap
    if shap_results.get('encoding_info') is not None and shap_results.get('cat_ts_shap_per_category') is not None:
        # Use per-category SHAP values if available
        ax3 = fig.add_subplot(gs[2 + row_offset, 0:2])
        cat_ts_shap_data = shap_results['cat_ts_shap_per_category'][sample_idx]  # [n_cats, seq_len]
        if cat_ts_shap_data.ndim == 3:
            cat_ts_shap_data = cat_ts_shap_data[..., min(class_idx, cat_ts_shap_data.shape[-1] - 1)]
        cat_ts_shap_data = cat_ts_shap_data[..., :n_steps]  # crop to eval_timestep

        enc_info = shap_results['encoding_info']
        cat_names = get_category_names_from_encoding_info(enc_info)
        
        n_cats = cat_ts_shap_data.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f"cat_{len(cat_names)}")
        
        # Use centered colormap for SHAP values
        norm3 = get_centered_norm(cat_ts_shap_data, center=0.0)
        im3 = ax3.imshow(cat_ts_shap_data, aspect='auto', cmap='RdBu_r', 
                         interpolation='nearest', norm=norm3)
        ax3.set_xlabel('Time', fontsize=12); ax3.set_ylabel('Category', fontsize=12)
        ax3.set_title('Categorical TS SHAP Heatmap', fontweight='bold', fontsize=14)

        if n_cats <= 30:
            ax3.set_yticks(range(n_cats)); ax3.set_yticklabels(cat_names[:n_cats], fontsize=9)
        else:
            step = max(1, n_cats // 20)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks); ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=9)

        ax3.set_xticks(tick_idx); ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45, fontsize=11)
        _draw_inhospital_boundary(ax3, _ihs, n_steps, label=False)
        cax3 = fig.add_subplot(gs[2 + row_offset, 2])
        fig.colorbar(im3, cax=cax3, label='SHAP Value')

        # Feature boundaries
        for feat, (start, end) in enc_info.get('feature_ranges', {}).items():
            if start > 0:
                ax3.axhline(y=start - 0.5, color='black', linewidth=1.5, linestyle='--')

    elif shap_results.get('encoding_info') is not None:
        # Fallback: Show raw data with SHAP importance overlay
        ax3 = fig.add_subplot(gs[2 + row_offset, 0:2])
        cat_ts_data = shap_results['test_data']['ts_cat'][sample_idx, :, :n_steps]  # crop to eval_timestep
        enc_info = shap_results['encoding_info']
        
        cat_names = get_category_names_from_encoding_info(enc_info)
        
        n_cats = cat_ts_data.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f"cat_{len(cat_names)}")
        
        # Show activity data (0/1 so no centering needed)
        im3 = ax3.imshow(cat_ts_data, aspect='auto', cmap='YlGnBu', interpolation='nearest', vmin=0)
        ax3.set_xlabel('Time', fontsize=12); ax3.set_ylabel('Category', fontsize=12)
        ax3.set_title('Categorical TS Activity (use compute_per_category_shap=True for SHAP values)',
                      fontweight='bold', fontsize=12)

        if n_cats <= 30:
            ax3.set_yticks(range(n_cats)); ax3.set_yticklabels(cat_names[:n_cats], fontsize=9)
        else:
            step = max(1, n_cats // 20)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks); ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=9)

        ax3.set_xticks(tick_idx); ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45, fontsize=11)
        cax3 = fig.add_subplot(gs[2 + row_offset, 2])
        fig.colorbar(im3, cax=cax3, label='Active')

        # Feature boundaries
        for feat, (start, end) in enc_info.get('feature_ranges', {}).items():
            if start > 0:
                ax3.axhline(y=start - 0.5, color='white', linewidth=2)
    
    # Plot 4: Channel importance — clinical-only when EBM present
    ax4 = fig.add_subplot(gs[3 + row_offset, :])
    # Use traj_length for proper normalization (padding is zeroed, avoid dilution)
    _traj_np = shap_results.get('test_data', {}).get('traj_lengths')
    _tl = int(_traj_np[sample_idx]) if _traj_np is not None and sample_idx < len(_traj_np) else n_steps
    ch_imp = np.abs(ts_shap[:, :_tl]).mean(axis=1) if _tl > 0 else np.zeros(n_channels)
    if has_ebm:
        display_ch = _get_clinical_only_channel_mask(channel2feature, n_channels)
    else:
        display_ch = _get_display_channel_mask(channel2feature, n_channels)
    ch_imp_display = ch_imp[display_ch]
    sorted_display = np.argsort(ch_imp_display)[::-1]
    n_show = min(20, len(ch_imp_display))
    sorted_idx = [display_ch[i] for i in sorted_display[:n_show]]
    if channel2feature:
        names = [channel2feature.get(i, f'Ch{i}') for i in sorted_idx]
        if has_ebm:
            bar_colors = [_GROUP_COLORS['Clinical']] * n_show
        else:
            bar_colors = [_get_channel_color(channel2feature, int(i)) for i in sorted_idx]
    else:
        names = [f'Channel {i}' for i in sorted_idx]
        bar_colors = ['#008bfb'] * n_show
    ax4.barh(range(n_show), ch_imp[sorted_idx], color=bar_colors, alpha=0.7)
    ax4.set_yticks(range(n_show)); ax4.set_yticklabels(names, fontsize=10)
    bar_title = f'Top {n_show} Clinical Channels' if has_ebm else f'Top {n_show} Channels'
    ax4.set_xlabel('Mean |SHAP|', fontsize=12); ax4.set_title(bar_title, fontweight='bold', fontsize=14)
    ax4.tick_params(axis='x', labelsize=11)
    ax4.grid(True, alpha=0.3, axis='x'); ax4.invert_yaxis()
    if channel2feature and not has_ebm:
        used_groups = set()
        for i in sorted_idx:
            name = channel2feature.get(int(i), '')
            if name in _EBM_CHANNELS: used_groups.add('EBM')
            else: used_groups.add('Clinical')
        ax4.legend(handles=[Patch(facecolor=_GROUP_COLORS[g], label=g, alpha=0.7)
                            for g in ['Clinical', 'EBM'] if g in used_groups],
                   loc='lower right', fontsize=11)
    
    # Plot 5: Static categorical
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax5 = fig.add_subplot(gs[4 + row_offset, 0])
        cat_shap = shap_results['cat_shap'][sample_idx]
        cat_data = shap_results['test_data']['cat'][sample_idx]
        if cat_shap.ndim == 2:
            cat_shap = cat_shap[..., min(class_idx, cat_shap.shape[-1] - 1)]
        
        n_feats = len(cat_shap)
        names = list(feature_names_cat)[:n_feats] if feature_names_cat else []
        while len(names) < n_feats:
            names.append(f'StaticCat_{len(names)}')
        
        colors = ['#ff0051' if x > 0 else '#008bfb' for x in cat_shap]
        ax5.barh(range(n_feats), cat_shap, color=colors, alpha=0.7)
        ax5.set_yticks(range(n_feats))
        
        # Safe value formatting
        ylabels = []
        for i in range(n_feats):
            name = names[i]
            if i < len(cat_data):
                val = cat_data[i]
                try:
                    ylabels.append(f'{name}\n(val={int(val)})')
                except (ValueError, TypeError):
                    ylabels.append(f'{name}\n(val={val})')
            else:
                ylabels.append(f'{name}')
        ax5.set_yticklabels(ylabels, fontsize=10)

        ax5.set_xlabel('SHAP Value', fontsize=12); ax5.set_title('Static Categorical', fontweight='bold', fontsize=14)
        ax5.tick_params(axis='x', labelsize=11)
        ax5.axvline(x=0, color='black', linewidth=0.8)
        ax5.grid(True, alpha=0.3, axis='x'); ax5.invert_yaxis()
    
    # Plot 6: Static continuous
    if shap_results['cont_shap'] is not None and shap_results['cont_shap'].size > 0:
        ax6 = fig.add_subplot(gs[4 + row_offset, 1])
        cont_shap = shap_results['cont_shap'][sample_idx]
        cont_data = shap_results['test_data']['cont'][sample_idx]
        if cont_shap.ndim == 2:
            cont_shap = cont_shap[..., min(class_idx, cont_shap.shape[-1] - 1)]
        
        n_feats = len(cont_shap)
        names = list(feature_names_cont)[:n_feats] if feature_names_cont else []
        while len(names) < n_feats:
            names.append(f'StaticCont_{len(names)}')
        
        colors = ['#ff0051' if x > 0 else '#008bfb' for x in cont_shap]
        ax6.barh(range(n_feats), cont_shap, color=colors, alpha=0.7)
        ax6.set_yticks(range(n_feats))
        
        # Safe value formatting
        ylabels = []
        for i in range(n_feats):
            name = names[i]
            if i < len(cont_data):
                val = cont_data[i]
                try:
                    ylabels.append(f'{name}\n(val={float(val):.2f})')
                except (ValueError, TypeError):
                    ylabels.append(f'{name}\n(val={val})')
            else:
                ylabels.append(f'{name}')
        ax6.set_yticklabels(ylabels, fontsize=10)

        ax6.set_xlabel('SHAP Value', fontsize=12); ax6.set_title('Static Continuous', fontweight='bold', fontsize=14)
        ax6.tick_params(axis='x', labelsize=11)
        ax6.axvline(x=0, color='black', linewidth=0.8)
        ax6.grid(True, alpha=0.3, axis='x'); ax6.invert_yaxis()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return {'sample_idx': sample_idx, 'pid': display_pid}


def visualize_data_completeness(shap_results: Dict, sample_idx: int = None,
                                 pid: Union[int, str] = None,
                                 holdout_pids: List = None,
                                 channel2feature: Dict[int, str] = None,
                                 save_path: str = None):
    """
    Visualize data completeness/missingness for an individual patient.

    Shows a 3-state heatmap (present/missing/padding) for continuous TS channels,
    grouped by type (Clinical, EBM, Temporal). Also shows categorical TS activity
    if available.

    Args:
        shap_results: Dict from calculate_shap_from_dataloaders (contains test_data)
        sample_idx: Direct index into the test data
        pid: Patient ID to visualize (requires holdout_pids)
        holdout_pids: List of PIDs in dataloader order
        channel2feature: Mapping channel index -> feature name
        save_path: Path to save the figure
    """
    # --- Resolve sample index ---
    if pid is not None:
        if holdout_pids is None:
            raise ValueError("holdout_pids required when using pid parameter")
        sample_idx = get_sample_idx_for_pid(holdout_pids, pid)
        if sample_idx is None:
            raise ValueError(f"PID {pid} not found in holdout data")
    elif sample_idx is None:
        sample_idx = 0

    display_pid = None
    if holdout_pids is not None and sample_idx < len(holdout_pids):
        display_pid = holdout_pids[sample_idx]

    # --- Extract data ---
    ts_data = shap_results['test_data']['ts'][sample_idx]  # [n_channels, seq_len]
    n_channels, n_steps = ts_data.shape

    # Crop to eval_timestep — same as visualize_shap_individual.
    eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_data = ts_data[..., :n_steps]

    # Time axis
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps - 1, n_ticks, dtype=int)

    # --- Detect trajectory length ---
    # Priority: (0) explicit traj_lengths from test_data (most reliable, from dataset)
    #           (1) _data_present channel  → mask spans first..last measurement
    #           (2) elapsed_hours channel  → mask spans first..last non-zero elapsed
    #           (3) explicit trajectory_length (inference session) → mask spans 0..traj_len
    #               This fixes the zero-padded inference case where heuristics fail.
    #           (4) NaN-any fallback (unreliable for zero-padded data).
    trajectory_mask = None  # set by whichever branch succeeds first

    # (0) Explicit traj_lengths from test_data (most reliable)
    _traj_lengths_array = shap_results.get('test_data', {}).get('traj_lengths')
    if _traj_lengths_array is not None and sample_idx < len(_traj_lengths_array):
        effective_traj = min(int(_traj_lengths_array[sample_idx]), n_steps)
        trajectory_mask = np.zeros(n_steps, dtype=bool)
        trajectory_mask[:effective_traj] = True

    dp_idx = None
    eh_idx = None
    if channel2feature:
        for idx, name in channel2feature.items():
            if name == '_data_present':
                dp_idx = idx
            elif name == 'elapsed_hours':
                eh_idx = idx

    if trajectory_mask is None and dp_idx is not None:
        raw_mask = ts_data[dp_idx] > 0.5  # [seq_len]
        if np.any(raw_mask):
            # Contiguous fill: first..last measurement
            trajectory_mask = np.zeros(n_steps, dtype=bool)
            idxs = np.where(raw_mask)[0]
            if len(idxs) > 0:
                trajectory_mask[idxs[0]:idxs[-1] + 1] = True

    if trajectory_mask is None and eh_idx is not None:
        # elapsed_hours is >0 for in-trajectory steps, 0.0 for zero-padded
        eh = ts_data[eh_idx]
        raw_mask = ~np.isnan(eh) & (np.abs(eh) > 1e-8)
        trajectory_mask = np.zeros(n_steps, dtype=bool)
        idxs = np.where(raw_mask)[0]
        if len(idxs) > 0:
            trajectory_mask[idxs[0]:idxs[-1] + 1] = True

    if trajectory_mask is None:
        _traj_length_explicit = shap_results.get('trajectory_length')
        if _traj_length_explicit is not None:
            # Inference session path: use exact trajectory_length to avoid
            # treating zero-padded steps (0.0, non-NaN) as data present.
            effective_traj = min(int(_traj_length_explicit), n_steps)
            trajectory_mask = np.zeros(n_steps, dtype=bool)
            trajectory_mask[:effective_traj] = True
        else:
            # Last resort: NaN-any. Unreliable for zero-padded inference data.
            any_present = np.any(~np.isnan(ts_data), axis=0)
            trajectory_mask = np.zeros(n_steps, dtype=bool)
            idxs = np.where(any_present)[0]
            if len(idxs) > 0:
                trajectory_mask[idxs[0]:idxs[-1] + 1] = True

    traj_len = int(trajectory_mask.sum())

    # --- Group channels ---
    if channel2feature:
        ordered_indices, ordered_labels, group_boundaries = _get_grouped_channel_order(channel2feature)
    else:
        ordered_indices = list(range(n_channels))
        ordered_labels = [f'Ch{i}' for i in range(n_channels)]
        group_boundaries = OrderedDict([('All', (0, n_channels))])

    n_display = len(ordered_indices)

    # --- Build presence matrix (vectorized, NaN-aware) ---
    # 0 = padding (outside trajectory), 1 = missing (within trajectory, NaN), 2 = present
    ts_subset = ts_data[ordered_indices]  # [n_display, n_steps]
    is_present = ~np.isnan(ts_subset)  # NaN = missing; any real value (incl. zero) = present
    traj_broadcast = np.broadcast_to(trajectory_mask, (n_display, n_steps))
    presence = np.where(~traj_broadcast, 0, np.where(is_present, 2, 1)).astype(np.int8)

    # --- Compute completeness per channel ---
    if traj_len > 0:
        completeness = np.array([(presence[row][trajectory_mask] == 2).sum() / traj_len
                                 for row in range(n_display)])
    else:
        completeness = np.zeros(n_display)

    # --- Figure layout ---
    title_suffix = f" (PID: {display_pid})" if display_pid else f" (Sample {sample_idx})"
    has_cat = ('ts_cat' in shap_results.get('test_data', {}) and
               shap_results.get('encoding_info') is not None)

    if has_cat:
        fig = plt.figure(figsize=(22, 18))
        gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3,
                              height_ratios=[0.5, 1.5, 1.0, 1.0])
    else:
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3,
                              height_ratios=[0.5, 1.5, 1.0])

    # --- Plot 1: Data density timeline ---
    ax1 = fig.add_subplot(gs[0, :])
    clinical_rows = [row for row, ch_idx in enumerate(ordered_indices)
                     if not channel2feature or channel2feature.get(ch_idx, '')
                     not in (_SHAP_EXCLUDED_CHANNELS | _EBM_CHANNELS)]
    if not clinical_rows:
        clinical_rows = list(range(n_display))

    density = np.array([(presence[clinical_rows][:, t] == 2).sum() / len(clinical_rows)
                        for t in range(n_steps)]) * 100

    ax1.fill_between(range(n_steps), density, alpha=0.4, color='#2196F3')
    ax1.plot(range(n_steps), density, linewidth=2, color='#1565C0')

    # Mark trajectory end (first padding step after data begins)
    if traj_len > 0 and traj_len < n_steps:
        last_data_step = np.where(trajectory_mask)[0][-1]
        ax1.axvline(x=last_data_step + 0.5, color='red', linewidth=1.5, linestyle='--',
                    alpha=0.7, label='Trajectory end')
    # Mark prehospital/inhospital boundary
    _ihs_steps_c = shap_results.get('test_data', {}).get('inhospital_start_steps')
    _ihs_c = (int(_ihs_steps_c[sample_idx]) if (_ihs_steps_c is not None
               and sample_idx < len(_ihs_steps_c) and _ihs_steps_c[sample_idx] is not None) else None)
    _draw_inhospital_boundary(ax1, _ihs_c, n_steps)
    ax1.set_ylabel('% Clinical channels\nwith data')
    ax1.set_title(f'Data Completeness Over Time{title_suffix}', fontweight='bold', fontsize=14)
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Continuous TS presence heatmap (grouped) ---
    ax2 = fig.add_subplot(gs[1, :])
    cmap_presence = ListedColormap(['#E0E0E0', '#FF8A65', '#4CAF50'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm_presence = BoundaryNorm(bounds, cmap_presence.N)

    im2 = ax2.imshow(presence, aspect='auto', cmap=cmap_presence, norm=norm_presence,
                     interpolation='nearest')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Channel')
    ax2.set_title('Continuous TS Data Presence (grouped)', fontweight='bold')

    if n_display <= 40:
        ax2.set_yticks(range(n_display))
        ax2.set_yticklabels(ordered_labels, fontsize=7 if n_display > 25 else 8)
    else:
        step = max(1, n_display // 30)
        yticks = list(range(0, n_display, step))
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([ordered_labels[i] for i in yticks], fontsize=7)

    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    _draw_inhospital_boundary(ax2, _ihs_c, n_steps, label=False)
    _draw_group_separators(ax2, group_boundaries)

    ax2.legend(handles=[
        Patch(facecolor='#4CAF50', label='Present'),
        Patch(facecolor='#FF8A65', label='Missing'),
        Patch(facecolor='#E0E0E0', label='Padding'),
    ], loc='upper right', fontsize=9, framealpha=0.9, edgecolor='gray')

    # --- Plot 3 (optional): Categorical TS activity ---
    if has_cat:
        ax3 = fig.add_subplot(gs[2, :])
        cat_ts_data = shap_results['test_data']['ts_cat'][sample_idx]  # [n_cats, seq_len]
        enc_info = shap_results['encoding_info']
        cat_names = get_category_names_from_encoding_info(enc_info)

        n_cats = cat_ts_data.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f'Cat_{len(cat_names)}')

        # 3-state: -1=padding, 0=inactive, 1=active
        cat_presence = np.where(cat_ts_data > 0, 1.0, 0.0)
        for t in range(n_steps):
            if not trajectory_mask[t]:
                cat_presence[:, t] = -1

        cmap_cat = ListedColormap(['#E0E0E0', '#FFF9C4', '#66BB6A'])
        bounds_cat = [-1.5, -0.5, 0.5, 1.5]
        norm_cat = BoundaryNorm(bounds_cat, cmap_cat.N)

        im3 = ax3.imshow(cat_presence, aspect='auto', cmap=cmap_cat, norm=norm_cat,
                         interpolation='nearest')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Category')
        ax3.set_title('Categorical TS Activity', fontweight='bold')

        if n_cats <= 30:
            ax3.set_yticks(range(n_cats))
            ax3.set_yticklabels(cat_names[:n_cats], fontsize=8)
        else:
            step = max(1, n_cats // 20)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks)
            ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=8)

        ax3.set_xticks(tick_idx)
        ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)

        # Concept group separators from encoding_info
        for feat, (start, end) in enc_info.get('feature_ranges', {}).items():
            if start > 0:
                ax3.axhline(y=start - 0.5, color='white', linewidth=3, zorder=5)
                ax3.axhline(y=start - 0.5, color='black', linewidth=1.2,
                            linestyle='--', zorder=6)

        ax3.legend(handles=[
            Patch(facecolor='#66BB6A', label='Active'),
            Patch(facecolor='#FFF9C4', label='No activity'),
            Patch(facecolor='#E0E0E0', label='Padding'),
        ], loc='upper right', fontsize=9, framealpha=0.9, edgecolor='gray')

    # --- Last row: Completeness bar + Summary ---
    last_row = 3 if has_cat else 2

    # Left: Per-channel completeness bar chart (grouped)
    ax4 = fig.add_subplot(gs[last_row, 0])
    bar_colors = [_get_channel_color(channel2feature, ch_idx) if channel2feature
                  else '#2196F3' for ch_idx in ordered_indices]
    ax4.barh(range(n_display), completeness * 100, color=bar_colors, alpha=0.8)
    ax4.set_yticks(range(n_display))
    ax4.set_yticklabels(ordered_labels, fontsize=7 if n_display > 20 else 8)
    ax4.set_xlabel('% Completeness')
    ax4.set_title('Channel Completeness (within trajectory)', fontweight='bold')
    ax4.set_xlim(0, 105)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    _draw_group_separators(ax4, group_boundaries)

    if channel2feature:
        used_groups = set()
        for ch_idx in ordered_indices:
            name = channel2feature.get(ch_idx, '')
            if name in _EBM_CHANNELS:
                used_groups.add('EBM')
            else:
                used_groups.add('Clinical')
        ax4.legend(handles=[Patch(facecolor=_GROUP_COLORS[g], label=g, alpha=0.8)
                            for g in ['Clinical', 'EBM'] if g in used_groups],
                   loc='lower right', fontsize=8)

    # Right: Summary statistics
    ax5 = fig.add_subplot(gs[last_row, 1])
    ax5.axis('off')

    traj_hours = 0
    if traj_len > 0:
        last_step = np.where(trajectory_mask)[0][-1]
        t_min = step_to_time(last_step)
        traj_hours = t_min / 60 if t_min else 0

    overall_comp = completeness.mean() * 100

    group_stats = []
    for group_name, (start, end) in group_boundaries.items():
        g_comp = completeness[start:end].mean() * 100
        n_ch = end - start
        group_stats.append(f"  {group_name} ({n_ch} ch): {g_comp:.1f}%")

    sorted_comp = np.argsort(completeness)[::-1]
    top_3 = [f"  {ordered_labels[i]}: {completeness[i]*100:.0f}%"
             for i in sorted_comp[:3]]
    bottom_3 = [f"  {ordered_labels[i]}: {completeness[i]*100:.0f}%"
                for i in sorted_comp[-3:] if completeness[i] < 1.0]

    summary = (
        f"Summary\n{'=' * 30}\n\n"
        f"Patient: {display_pid or sample_idx}\n"
        f"Trajectory: {traj_len} steps ({traj_hours:.1f}h)\n"
        f"Channels: {n_display}\n"
        f"Overall completeness: {overall_comp:.1f}%\n\n"
        f"Per group:\n" + "\n".join(group_stats) + "\n\n"
        f"Most complete:\n" + "\n".join(top_3) + "\n\n"
        f"Least complete:\n" + ("\n".join(bottom_3) if bottom_3 else "  (all 100%)")
    )

    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#BDBDBD'))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return {'sample_idx': sample_idx, 'pid': display_pid,
            'completeness': dict(zip(ordered_labels, completeness))}


def visualize_shap_summary(shap_results: Dict, channel2feature: Dict[int, str] = None,
                           feature_names_cat: List[str] = None,
                           feature_names_cont: List[str] = None,
                           max_display: int = 20, class_idx: int = 1, save_path: str = None,
                           eval_timestep: Optional[int] = None,
                           density_normalize: bool = False):
    """Summary visualizations across cohort.

    Args:
        density_normalize: If True, normalize SHAP aggregation by measurement
            density per channel.  Channels with more non-zero (measured) cells
            naturally accumulate more |SHAP| under simple averaging; density
            normalization divides by the fraction of measured cells so the bar
            chart and heatmap reflect per-measurement importance.
    """

    ts_shap = shap_results['ts_shap']
    if ts_shap.ndim == 4:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_samples, n_channels, n_steps = ts_shap.shape

    # Crop time axis to eval_timestep — steps beyond have ~0 SHAP due to causal masking
    # and showing them extends the x-axis with meaningless zeros.
    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[..., :n_steps]

    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps-1, n_ticks, dtype=int)

    # EBM two-view: compute budget and adjust layout
    budget = compute_ebm_vs_clinical_budget(ts_shap, channel2feature)
    has_ebm = budget is not None

    if has_ebm:
        fig = plt.figure(figsize=(22, 21))
        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3,
                              height_ratios=[0.7, 1, 1, 1.2, 1])
        row_offset = 1
        # Row 0: EBM budget over time
        ax_budget = fig.add_subplot(gs[0, :])
        _draw_ebm_budget_temporal(ax_budget, budget, n_steps, tick_idx,
                                  [time_fmt[i] for i in tick_idx],
                                  title=f'SHAP Budget Over Time: EBM vs Clinical (Class {class_idx})')
    else:
        fig = plt.figure(figsize=(22, 18))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1.2, 1])
        row_offset = 0

    # Channels to display (exclude temporal/auxiliary — not model features)
    display_ch = _get_display_channel_mask(channel2feature, n_channels)

    # Plot 1: TS importance over time
    ax1 = fig.add_subplot(gs[0 + row_offset, :])

    # Build per-sample validity mask for masked averaging over time
    traj_np = shap_results.get('test_data', {}).get('traj_lengths')
    if traj_np is not None:
        # valid_time: [n_samples, n_steps] — True where step < traj_length
        valid_time = np.arange(n_steps)[None, :] < traj_np[:n_samples, None]
    else:
        valid_time = None  # fallback: treat all positions as valid

    # Build measurement-density mask from input data (non-zero = measured)
    # Used when density_normalize=True to avoid rewarding channels that are
    # simply measured more often.
    _measured_mask = None  # [n_samples, n_channels, n_steps] bool
    _cat_measured_mask = None  # [n_samples, n_categories, n_steps] bool
    if density_normalize:
        test_ts = shap_results.get('test_data', {}).get('ts')
        if test_ts is not None:
            _measured_mask = test_ts[:n_samples, :, :n_steps] != 0.0
            logger.info("Density normalization enabled for SHAP summary")
        else:
            logger.warning("density_normalize=True but test_data['ts'] not available; "
                           "falling back to standard aggregation")
        test_ts_cat = shap_results.get('test_data', {}).get('ts_cat')
        if test_ts_cat is not None:
            _cat_measured_mask = test_ts_cat[:n_samples, :, :n_steps] != 0.0
            logger.info("Density normalization enabled for categorical TS SHAP summary")

    def _masked_temporal_mean(arr_3d, ch_indices):
        """Mean |SHAP| over samples & channels -> [n_steps], padding-aware."""
        subset = np.abs(arr_3d[:, ch_indices, :])  # [n_samples, n_ch, n_steps]
        if valid_time is not None:
            mask = valid_time[:, None, :]  # [n_samples, 1, n_steps]
            mask = np.broadcast_to(mask, subset.shape)
            denom = mask.sum(axis=(0, 1)).clip(1)
            return (subset * mask).sum(axis=(0, 1)) / denom
        return subset.mean(axis=(0, 1))

    def _masked_channel_mean(arr_3d):
        """Mean |SHAP| over samples & time -> [n_channels], padding-aware.

        When ``_measured_mask`` is set (density_normalize=True), the denominator
        counts only positions where the channel had an actual measurement
        (non-zero input), so channels are not rewarded for being measured more
        frequently.
        """
        subset = np.abs(arr_3d)  # [n_samples, n_channels, n_steps]
        if _measured_mask is not None:
            # Density-normalized: only count measured positions
            mask = _measured_mask.copy()
            if valid_time is not None:
                # Also exclude padding
                mask = mask & valid_time[:, None, :]
            denom = mask.sum(axis=(0, 2)).clip(1)  # [n_channels]
            return (subset * mask).sum(axis=(0, 2)) / denom
        if valid_time is not None:
            mask = valid_time[:, None, :]
            mask = np.broadcast_to(mask, subset.shape)
            denom = mask.sum(axis=(0, 2)).clip(1)
            return (subset * mask).sum(axis=(0, 2)) / denom
        return subset.mean(axis=(0, 2))

    if has_ebm:
        # Two-view: separate Clinical and EBM lines
        clinical_ch = _get_clinical_only_channel_mask(channel2feature, n_channels)
        ebm_ch = [i for i, name in channel2feature.items() if name in _EBM_CHANNELS]

        clinical_imp = _masked_temporal_mean(ts_shap, clinical_ch)
        ax1.plot(clinical_imp, linewidth=2, color=_GROUP_COLORS['Clinical'],
                 label='Clinical channels')
        ax1.fill_between(range(len(clinical_imp)), clinical_imp, alpha=0.2,
                         color=_GROUP_COLORS['Clinical'])

        if ebm_ch:
            ebm_imp = _masked_temporal_mean(ts_shap, ebm_ch)
            ax1.plot(ebm_imp, linewidth=2, color=_GROUP_COLORS['EBM'],
                     label='EBM (_ebm_pred)', linestyle='--')
            ax1.fill_between(range(len(ebm_imp)), ebm_imp, alpha=0.2,
                             color=_GROUP_COLORS['EBM'])
    else:
        ts_imp = _masked_temporal_mean(ts_shap, display_ch)
        ax1.plot(ts_imp, linewidth=2, color='#ff0051', label='Continuous TS')
        ax1.fill_between(range(len(ts_imp)), ts_imp, alpha=0.3, color='#ff0051')

    if shap_results['cat_ts_shap'] is not None:
        cat_ts = shap_results['cat_ts_shap']
        if cat_ts.ndim == 3:
            cat_ts = cat_ts[..., min(class_idx, cat_ts.shape[-1] - 1)]
        cat_ts = cat_ts[..., :n_steps]  # crop to eval_timestep
        if _cat_measured_mask is not None:
            # Density-normalized: any category active at this timestep per sample
            _cat_any_active = _cat_measured_mask[:cat_ts.shape[0]].any(axis=1)[:, :cat_ts.shape[1]]  # [n_samples, n_steps]
            if valid_time is not None:
                _cat_any_active = _cat_any_active & valid_time[:_cat_any_active.shape[0], :_cat_any_active.shape[1]]
            _cat_denom_t = _cat_any_active.sum(axis=0).clip(1)  # [n_steps]
            cat_imp = (np.abs(cat_ts) * _cat_any_active).sum(axis=0) / _cat_denom_t
        else:
            cat_imp = np.abs(cat_ts).mean(axis=0)
        ax1.plot(cat_imp, linewidth=2, color='#00d4aa', label='Categorical TS', linestyle='--')
        ax1.fill_between(range(len(cat_imp)), cat_imp, alpha=0.2, color='#00d4aa')

    ax1.set_xlabel('Time'); ax1.set_ylabel('Mean |SHAP|')
    ax1.set_title(f'Feature Importance Over Time (Class {class_idx})', fontweight='bold')
    ax1.set_xticks(tick_idx); ax1.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top channels — clinical-only when EBM present
    ax2 = fig.add_subplot(gs[1 + row_offset, 0])
    ch_imp = _masked_channel_mean(ts_shap)
    if has_ebm:
        bar_display_ch = _get_clinical_only_channel_mask(channel2feature, n_channels)
    else:
        bar_display_ch = display_ch
    ch_imp_display = ch_imp[bar_display_ch]
    sorted_display = np.argsort(ch_imp_display)[::-1][:max_display]
    sorted_idx = [bar_display_ch[i] for i in sorted_display]
    if channel2feature:
        names = [channel2feature.get(int(i), f'Ch{i}') for i in sorted_idx]
        if has_ebm:
            bar_colors = [_GROUP_COLORS['Clinical']] * len(sorted_idx)
        else:
            bar_colors = [_get_channel_color(channel2feature, int(i)) for i in sorted_idx]
    else:
        names = [f'Channel {i}' for i in sorted_idx]
        bar_colors = ['#008bfb'] * len(sorted_idx)
    ax2.barh(range(len(sorted_idx)), ch_imp[sorted_idx], color=bar_colors, alpha=0.7)
    ax2.set_yticks(range(len(sorted_idx))); ax2.set_yticklabels(names, fontsize=10)
    _dn_suffix = ' (per-measurement)' if _measured_mask is not None else ''
    bar_title = f'Top {len(sorted_idx)} Clinical Channels{_dn_suffix}' if has_ebm else f'Top {len(sorted_idx)} Channels{_dn_suffix}'
    _shap_label = 'Mean |SHAP| / measured cell' if _measured_mask is not None else 'Mean |SHAP|'
    ax2.set_xlabel(_shap_label); ax2.set_title(bar_title, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x'); ax2.invert_yaxis()
    if channel2feature and not has_ebm:
        used_groups = set()
        for i in sorted_idx:
            name = channel2feature.get(int(i), '')
            if name in _EBM_CHANNELS: used_groups.add('EBM')
            else: used_groups.add('Clinical')
        ax2.legend(handles=[Patch(facecolor=_GROUP_COLORS[g], label=g, alpha=0.7)
                            for g in ['Clinical', 'EBM'] if g in used_groups],
                   loc='lower right', fontsize=8)
    
    # Plot 3: Categorical TS SHAP heatmap (mean across cohort)
    if shap_results.get('encoding_info') is not None and shap_results.get('cat_ts_shap_per_category') is not None:
        ax3 = fig.add_subplot(gs[1 + row_offset, 1])
        cat_ts_shap = shap_results['cat_ts_shap_per_category']  # [n_samples, n_cats, seq_len]
        if cat_ts_shap.ndim == 4:
            cat_ts_shap = cat_ts_shap[..., min(class_idx, cat_ts_shap.shape[-1] - 1)]
        cat_ts_shap = cat_ts_shap[..., :n_steps]  # crop to eval_timestep
        if _cat_measured_mask is not None:
            _cat_hm_mask = _cat_measured_mask[:cat_ts_shap.shape[0], :cat_ts_shap.shape[1], :cat_ts_shap.shape[2]]
            if valid_time is not None:
                _cat_hm_mask = _cat_hm_mask & valid_time[:_cat_hm_mask.shape[0], None, :_cat_hm_mask.shape[2]]
            _cat_hm_denom = _cat_hm_mask.sum(axis=0).clip(1)  # [n_cats, n_steps]
            cat_ts_mean = (np.abs(cat_ts_shap) * _cat_hm_mask).sum(axis=0) / _cat_hm_denom
        else:
            cat_ts_mean = np.abs(cat_ts_shap).mean(axis=0)  # [n_cats, seq_len]

        enc_info = shap_results['encoding_info']
        cat_names = get_category_names_from_encoding_info(enc_info)
        n_cats = cat_ts_mean.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f"cat_{len(cat_names)}")

        _cat_dn_suffix = ' (per-event)' if _cat_measured_mask is not None else ' (Mean)'
        im3 = ax3.imshow(cat_ts_mean, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_xlabel('Time'); ax3.set_ylabel('Category')
        ax3.set_title(f'Categorical TS |SHAP|{_cat_dn_suffix}', fontweight='bold')
        
        if n_cats <= 20:
            ax3.set_yticks(range(n_cats)); ax3.set_yticklabels(cat_names, fontsize=8)
        else:
            step = max(1, n_cats // 15)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks); ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=8)
        ax3.set_xticks(tick_idx); ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
        _cat_cbar_label = 'Mean |SHAP| / active event' if _cat_measured_mask is not None else 'Mean |SHAP|'
        plt.colorbar(im3, ax=ax3, label=_cat_cbar_label)

    elif shap_results.get('encoding_info') is not None:
        # Fallback: show activity data
        ax3 = fig.add_subplot(gs[1 + row_offset, 1])
        cat_ts_data = shap_results['test_data']['ts_cat'][..., :n_steps]  # crop to eval_timestep
        cat_ts_mean = cat_ts_data.mean(axis=0)
        enc_info = shap_results['encoding_info']
        cat_names = get_category_names_from_encoding_info(enc_info)
        n_cats = cat_ts_mean.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f"cat_{len(cat_names)}")
        
        im3 = ax3.imshow(cat_ts_mean, aspect='auto', cmap='YlGnBu', interpolation='nearest')
        ax3.set_xlabel('Time'); ax3.set_ylabel('Category')
        ax3.set_title('Categorical TS Activity (Mean)', fontweight='bold')
        
        if n_cats <= 20:
            ax3.set_yticks(range(n_cats)); ax3.set_yticklabels(cat_names, fontsize=8)
        else:
            step = max(1, n_cats // 15)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks); ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=8)
        ax3.set_xticks(tick_idx); ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
        plt.colorbar(im3, ax=ax3, label='Mean Activity')
    
    # Plot 4: Continuous TS heatmap — clinical-only when EBM present
    ax4 = fig.add_subplot(gs[2 + row_offset, :])
    if _measured_mask is not None:
        # Density-normalized heatmap: per-channel, per-timestep mean over
        # measured positions only.
        _hm_mask = _measured_mask.copy()
        if valid_time is not None:
            _hm_mask = _hm_mask & valid_time[:, None, :]
        _hm_denom = _hm_mask.sum(axis=0).clip(1)  # [n_channels, n_steps]
        ts_mean = (np.abs(ts_shap) * _hm_mask).sum(axis=0) / _hm_denom
    else:
        ts_mean = np.abs(ts_shap).mean(axis=0)
    if channel2feature and has_ebm:
        ordered_idx, ordered_labels_4 = _get_clinical_only_channel_order(channel2feature)
        ts_mean_display = ts_mean[ordered_idx]
        n_display_4 = len(ordered_idx)
    elif channel2feature:
        ordered_idx, ordered_labels_4, group_bounds_4 = _get_grouped_channel_order(channel2feature)
        ts_mean_display = ts_mean[ordered_idx]
        n_display_4 = len(ordered_idx)
    else:
        ts_mean_display = ts_mean
        ordered_labels_4 = [f'Ch{i}' for i in range(n_channels)]
        n_display_4 = n_channels
    im = ax4.imshow(ts_mean_display, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0)
    ax4.set_xlabel('Time'); ax4.set_ylabel('Channel')
    if _measured_mask is not None:
        heatmap_title_4 = 'Clinical |SHAP| Heatmap (per-measurement)' if has_ebm else '|SHAP| Heatmap (per-measurement, grouped)'
    else:
        heatmap_title_4 = 'Clinical Continuous TS |SHAP| Heatmap (Mean)' if has_ebm else 'Continuous TS |SHAP| Heatmap (Mean, grouped)'
    ax4.set_title(heatmap_title_4, fontweight='bold')
    if n_display_4 <= 40:
        ax4.set_yticks(range(n_display_4))
        ax4.set_yticklabels(ordered_labels_4, fontsize=7 if n_display_4 > 25 else 9)
    else:
        step = max(1, n_display_4 // 30)
        yticks = list(range(0, n_display_4, step))
        ax4.set_yticks(yticks)
        ax4.set_yticklabels([ordered_labels_4[i] for i in yticks], fontsize=7)
    ax4.set_xticks(tick_idx); ax4.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    plt.colorbar(im, ax=ax4, label='Mean |SHAP|')
    if not has_ebm and channel2feature:
        _draw_group_separators(ax4, group_bounds_4)
    
    # Plot 5: Static categorical
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax5 = fig.add_subplot(gs[3 + row_offset, 0])
        cat_shap = shap_results['cat_shap']
        if cat_shap.ndim == 3:
            cat_shap = cat_shap[..., min(class_idx, cat_shap.shape[-1] - 1)]
        cat_imp = np.abs(cat_shap).mean(axis=0) if cat_shap.ndim > 1 else np.abs(cat_shap)
        n_feats = len(cat_imp)
        
        names = list(feature_names_cat)[:n_feats] if feature_names_cat else []
        while len(names) < n_feats:
            names.append(f'StaticCat_{len(names)}')
        
        sorted_idx = np.argsort(cat_imp)[::-1][:min(max_display, n_feats)]
        ax5.barh(range(len(sorted_idx)), cat_imp[sorted_idx], color='#ff0051', alpha=0.7)
        ax5.set_yticks(range(len(sorted_idx)))
        ax5.set_yticklabels([names[int(i)] for i in sorted_idx], fontsize=10)
        ax5.set_xlabel('Mean |SHAP|'); ax5.set_title('Static Categorical', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x'); ax5.invert_yaxis()
    
    # Plot 6: Static continuous
    if shap_results['cont_shap'] is not None and shap_results['cont_shap'].size > 0:
        ax6 = fig.add_subplot(gs[3 + row_offset, 1])
        cont_shap = shap_results['cont_shap']
        if cont_shap.ndim == 3:
            cont_shap = cont_shap[..., min(class_idx, cont_shap.shape[-1] - 1)]
        cont_imp = np.abs(cont_shap).mean(axis=0) if cont_shap.ndim > 1 else np.abs(cont_shap)
        n_feats = len(cont_imp)
        
        names = list(feature_names_cont)[:n_feats] if feature_names_cont else []
        while len(names) < n_feats:
            names.append(f'StaticCont_{len(names)}')
        
        sorted_idx = np.argsort(cont_imp)[::-1][:min(max_display, n_feats)]
        ax6.barh(range(len(sorted_idx)), cont_imp[sorted_idx], color='#008bfb', alpha=0.7)
        ax6.set_yticks(range(len(sorted_idx)))
        ax6.set_yticklabels([names[int(i)] for i in sorted_idx], fontsize=10)
        ax6.set_xlabel('Mean |SHAP|'); ax6.set_title('Static Continuous', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x'); ax6.invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        save_base64(fig, save_path, dpi=150)
    plt.show()




def shap_analysis(data=None, model=None, model_name='13012025', compute_per_category_shap=True, max_background_samples=600,
                  max_test_samples=90, visualize=True, specific_pids: List = None,
                  density_normalize: bool = False) -> Dict:
    """
    Run full SHAP analysis.

    Args:
        data: Prepared data dict
        model: Trained nn.Module (on device). If None, loaded via prepare_model.
        model_name: Model checkpoint name
        compute_per_category_shap: If True, compute SHAP on raw multi-hot categorical TS
                                   to get per-category attributions (shows which specific
                                   medications/procedures matter). If False, faster but
                                   only shows aggregate categorical TS importance.
        max_test_samples: Maximum number of test samples for SHAP calculation
        specific_pids: List of specific PIDs to include in analysis. If provided,
                       only these PIDs will be analyzed (must exist in holdout set).
                       This ensures specific patients are available for individual plotting.

    Returns:
        dict with 'shap_results', 'holdout_pids', 'channel2feature', 'static_cat_names'
    """
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if data is None:
        data = prepare_data_and_dls_cached(cfg)
    if model is None:
        model, device = prepare_model(data, cfg)

    # Get all holdout PIDs first (needed for filtering by specific_pids)
    all_holdout_pids = data["holdout"].tab_df['PID'].tolist()

    # Validate specific_pids if provided
    if specific_pids is not None:
        missing_pids = [pid for pid in specific_pids if pid not in all_holdout_pids]
        if missing_pids:
            logger.warning("PIDs not found in holdout set: %s", missing_pids)
        specific_pids = [pid for pid in specific_pids if pid in all_holdout_pids]
        if not specific_pids:
            raise ValueError("None of the specified PIDs were found in holdout set")
        logger.info("Analyzing %s specific PIDs: %s", len(specific_pids), specific_pids)

    shap_results = calculate_shap_from_dataloaders(
        model=model,
        background_loader=data["mixed_dls"].train,
        test_loader=data["holdout_mixed_dls"].train,
        device=device,
        max_background_samples=max_background_samples,
        max_test_samples=max_test_samples,
        encoding_info=data["encoding_info"],
        compute_per_category_shap=compute_per_category_shap,
        specific_pids=specific_pids,
        all_pids=all_holdout_pids
    )

    channel2feature, _ = create_channel_mapping(data)

    # Get holdout PIDs for individual plots (filtered if specific_pids provided)
    holdout_pids = get_holdout_pids(data, max_samples=max_test_samples, specific_pids=specific_pids)
    logger.info("Extracted %s holdout PIDs", len(holdout_pids))
    logger.debug("First 5 PIDs: %s", holdout_pids[:5])
    
    # Get static categorical names from classes (includes _na columns)
    static_cat_names = get_static_cat_names_from_classes(data["classes"])
    logger.info("Static categorical features: %s", static_cat_names)
    
    # Debug output
    debug_shap_data(shap_results, 
                    feature_names_cat=static_cat_names,
                    feature_names_cont=cfg["dataset"]["num_cols"])
    
    # Load EBM glassbox importances if EBM channels are present
    ebm_importances = None
    if _has_ebm_channels(channel2feature):
        ebm_importances = load_ebm_global_importances(
            models_dir='models/ebm',
            n_steps=data['seq_len'],
            top_n=20,
        )

    if visualize is True:
        visualize_shap_summary(
            shap_results, channel2feature=channel2feature,
            feature_names_cat=static_cat_names,
            feature_names_cont=cfg["dataset"]["num_cols"],
            class_idx=1, max_display=20,
            save_path='reports/shap/shap_summary_cohort.png',
            density_normalize=density_normalize,
        )

        # Use first PID for individual plot
        first_pid = holdout_pids[0] if holdout_pids else None
        visualize_shap_individual(
            shap_results,
            pid=first_pid,
            holdout_pids=holdout_pids,
            channel2feature=channel2feature,
            feature_names_cat=static_cat_names,
            feature_names_cont=cfg["dataset"]["num_cols"],
            class_idx=1,
            save_path='reports/shap/shap_individual_sample_0.png',
        )

        # Standalone EBM feature importance plot
        if ebm_importances is not None:
            visualize_ebm_importances(
                ebm_importances,
                n_steps=data['seq_len'],
                save_path='reports/shap/ebm_feature_importance.png',
            )

        visualize_data_completeness(
            shap_results,
            pid=first_pid,
            holdout_pids=holdout_pids,
            channel2feature=channel2feature,
            save_path='reports/shap/data_completeness_sample_0.png'
        )

    # Return comprehensive results for further analysis
    return {
        'shap_results': shap_results,
        'holdout_pids': holdout_pids,
        'channel2feature': channel2feature,
        'static_cat_names': static_cat_names,
        'data': data,
        'ebm_importances': ebm_importances,
    }


# ============================================================================
# TIMEFRAMES
# ============================================================================

# Default timeframes (hours)
DEFAULT_TIMEFRAMES = OrderedDict([
    ('1H', 1), ('6H', 6), ('12H', 12), ('1D', 24), ('3D', 72),
    ('7D', 168), ('14D', 336), ('30D', 720), ('full', None)
])

# Time utilities (step_to_time, time_to_step, time_to_hours) imported from
# astra.evaluation.utils — reads bin intervals from config instead of hardcoding.
# Alias for backward compat:
time_to_hours_str = time_to_hours


def get_actual_data_length(ts_data: np.ndarray, threshold: float = 1e-6) -> int:
    """Find last non-zero timestep."""
    if ts_data.ndim == 1:
        ts_data = ts_data.reshape(1, -1)
    has_data = np.abs(ts_data).max(axis=0) > threshold
    if not has_data.any():
        return 0
    return np.where(has_data)[0][-1] + 1


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TimeframeSHAPResult:
    """SHAP values for a single timeframe."""
    timeframe_name: str
    timeframe_hours: Optional[float]
    censor_step: Optional[int]
    actual_data_steps: int
    ts_shap: np.ndarray
    cat_ts_shap: Optional[np.ndarray]
    cat_ts_shap_per_category: Optional[np.ndarray]
    cat_shap: Optional[np.ndarray]
    cont_shap: Optional[np.ndarray]
    ts_data: np.ndarray
    cat_ts_data: Optional[np.ndarray]
    cat_data: Optional[np.ndarray]
    cont_data: Optional[np.ndarray]
    ts_channel_importance: np.ndarray
    ts_temporal_importance: np.ndarray
    cat_ts_category_importance: Optional[np.ndarray] = None
    n_active_background: Optional[int] = None

    @property
    def effective_steps(self) -> int:
        if self.censor_step is None:
            return self.actual_data_steps
        return min(self.censor_step, self.actual_data_steps)


@dataclass 
class TemporalSHAPResults:
    """Complete temporal SHAP analysis results."""
    pid: Union[int, str]
    sample_idx: int
    actual_data_length_steps: int
    actual_data_length_hours: float
    timeframe_results: Dict[str, TimeframeSHAPResult]
    channel2feature: Dict[int, str]
    static_cat_names: List[str]
    static_cont_names: List[str]
    encoding_info: Dict
    stability_metrics: Optional[Dict] = None
    inhospital_start_step: Optional[int] = None
    active_only: bool = False
    density_normalize: bool = False

    def get_available_timeframes(self) -> List[str]:
        return list(self.timeframe_results.keys())

    def get_result(self, timeframe: str) -> Optional[TimeframeSHAPResult]:
        return self.timeframe_results.get(timeframe)


@dataclass
class CohortTemporalSHAPResults:
    """Aggregated temporal SHAP results across a cohort of patients."""
    n_patients: int
    pids: List
    # Per-timeframe aggregated arrays
    channel_importance: Dict[str, np.ndarray]       # tf -> [n_channels]
    channel_importance_std: Dict[str, np.ndarray]   # tf -> [n_channels]
    temporal_importance: Dict[str, np.ndarray]       # tf -> [seq_len]
    temporal_importance_std: Dict[str, np.ndarray]   # tf -> [seq_len]
    ts_shap_mean: Dict[str, np.ndarray]             # tf -> [n_channels, seq_len]
    static_cat_importance: Dict[str, Optional[np.ndarray]]
    static_cont_importance: Dict[str, Optional[np.ndarray]]
    patient_counts: Dict[str, int]                  # tf -> n patients with this tf
    # Metadata
    channel2feature: Dict[int, str]
    static_cat_names: List[str]
    static_cont_names: List[str]
    encoding_info: Dict
    active_only: bool = False
    density_normalize: bool = False
    # Categorical TS per-category importance (optional)
    cat_ts_per_category_importance: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)  # tf -> [n_categories]
    cat_ts_category_names: List[str] = field(default_factory=list)
    cat_ts_gate_values: Optional[np.ndarray] = None
    # Individual patient results (optional, for deep-dive)
    patient_results: Optional[List[TemporalSHAPResults]] = None

    def get_available_timeframes(self) -> List[str]:
        return list(self.channel_importance.keys())


# ============================================================================
# MODEL WRAPPER
# ============================================================================

# NOTE: SHAPModelWrapper is defined above (used by both
# calculate_shap_from_dataloaders and TemporalSHAPAnalyzer).
# NOTE: embed_categorical_features is defined above (converts to one-hot for SHAP).

# ============================================================================
# MAIN ANALYZER
# ============================================================================

class TemporalSHAPAnalyzer:
    """Analyzes SHAP values across timeframes using Option A (re-compute per timeframe)."""
    
    def __init__(self, model: nn.Module, data: Dict, background_loader,
                 device: str = 'cuda', max_background_samples: int = 200,
                 class_idx: int = 1, active_only: bool = False,
                 density_normalize: bool = False):
        self.model = model
        self.data = data
        self.background_loader = background_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_background_samples = max_background_samples
        self.class_idx = class_idx
        self.active_only = active_only
        self.density_normalize = density_normalize

        self.encoding_info = data.get("encoding_info", {})
        self.channel2feature, self.feature2channel = self._create_channel_mapping()
        self.static_cat_names = list(data.get("classes", {}).keys())
        self.static_cont_names = data.get("num_cols", [])

        self._bg_data = None
        self.model.eval()
        self.model = self.model.to(device)
        self.has_cat_ts = model.n_ts_cat > 0

        self.cat_ts_gate_values = None
        if hasattr(model, 'cat_ts_gate_params') and model.cat_ts_gate_params is not None:
            with torch.no_grad():
                self.cat_ts_gate_values = torch.sigmoid(model.cat_ts_gate_params).cpu().numpy()

        mode_parts = []
        if active_only: mode_parts.append("active-only")
        if density_normalize: mode_parts.append("density-norm")
        mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
        logger.info(f"TemporalSHAPAnalyzer{mode_str}: {len(self.channel2feature)} channels, "
                    f"cat_ts={self.has_cat_ts}, bg_samples={max_background_samples}")
        if self.cat_ts_gate_values is not None:
            logger.info(f"Categorical TS gate values (sigmoid): {self.cat_ts_gate_values}")
            logger.info(f"Gate suppression factor: mean={self.cat_ts_gate_values.mean():.3f} "
                        f"(range {self.cat_ts_gate_values.min():.3f}-{self.cat_ts_gate_values.max():.3f})")
    
    def _create_channel_mapping(self):
        features = self.data["trainval"].complete.sort_values(['PID', 'FEATURE'])['FEATURE'].drop_duplicates().tolist()
        return {i: f for i, f in enumerate(features)}, {f: i for i, f in enumerate(features)}
    
    def _extract_background_data(self):
        if self._bg_data is not None:
            return self._bg_data

        logger.info("Extracting background data...")
        all_ts, all_ts_cat, all_cat, all_cont, all_traj = [], [], [], [], []
        n = 0
        for batch in self.background_loader:
            if n >= self.max_background_samples:
                break
            inputs, _ = batch
            x_ts, x_tab, x_ts_cat = inputs[0], inputs[1], inputs[2]
            # Extract trajectory lengths with backward compat
            if len(inputs) >= 4:
                traj_lengths = inputs[3]
            else:
                traj_lengths = _infer_trajectory_lengths_from_batch(x_ts)
            all_ts.append(x_ts.cpu())
            all_ts_cat.append(x_ts_cat.cpu())
            all_cat.append(x_tab[0].cpu())
            all_cont.append(x_tab[1].cpu())
            all_traj.append(traj_lengths.cpu())
            n += x_ts.shape[0]

        self._bg_data = {
            'ts': torch.cat(all_ts)[:self.max_background_samples].to(self.device),
            'ts_cat': torch.cat(all_ts_cat)[:self.max_background_samples].to(self.device),
            'cat': torch.cat(all_cat)[:self.max_background_samples].to(self.device),
            'cont': torch.cat(all_cont)[:self.max_background_samples].to(self.device),
            'traj_lengths': torch.cat(all_traj)[:self.max_background_samples].to(self.device)
        }
        return self._bg_data

    def _get_active_background(self, censor_step):
        """Filter background to patients with trajectory_length > censor_step.

        Returns (filtered_bg_dict, n_active).  When active_only is False or
        censor_step is None, returns the full background unchanged.
        """
        bg = self._extract_background_data()
        if not self.active_only or censor_step is None:
            return bg, bg['ts'].shape[0]

        mask = bg['traj_lengths'] > censor_step
        n_active = int(mask.sum())
        if n_active < 2:
            # Fall back to full background to avoid degenerate SHAP
            logger.warning("  Only %s active bg samples at step %s, using all", n_active, censor_step)
            return bg, bg['ts'].shape[0]

        return {k: v[mask] for k, v in bg.items()}, n_active

    def _censor_data(self, ts, ts_cat, censor_step):
        if censor_step is None:
            return ts, ts_cat
        ts_c, ts_cat_c = ts.clone(), ts_cat.clone()
        if censor_step < ts.shape[2] - 1:
            ts_c[:, :, censor_step+1:] = 0.0
            ts_cat_c[:, :, censor_step+1:] = 0
        return ts_c, ts_cat_c
    
    def _compute_shap_for_sample(self, sample_ts, sample_ts_cat, sample_cat, sample_cont,
                                censor_step=None, traj_length=None):
        bg, n_active = self._get_active_background(censor_step)
        bg_ts_c, bg_ts_cat_c = self._censor_data(bg['ts'], bg['ts_cat'], censor_step)

        if sample_ts.dim() == 2:
            sample_ts = sample_ts.unsqueeze(0)
            sample_ts_cat = sample_ts_cat.unsqueeze(0)
            sample_cat = sample_cat.unsqueeze(0)
            sample_cont = sample_cont.unsqueeze(0)

        sample_ts_c, sample_ts_cat_c = self._censor_data(sample_ts, sample_ts_cat, censor_step)

        # Convert static categorical to one-hot encoding for SHAP
        bg_cat_onehot = embed_categorical_features(self.model, bg['cat']) if bg['cat'].shape[1] > 0 else None
        sample_cat_onehot = embed_categorical_features(self.model, sample_cat) if sample_cat.shape[1] > 0 else None

        # Build traj_lengths tensor for wrapper (single sample -> [1])
        # Clamp to censor_step+1 so the padding mask excludes censored positions
        # from attention and mean pooling (matches training behavior with variable-length sequences)
        if traj_length is not None:
            tl = traj_length
            if censor_step is not None:
                effective = min(int(tl), censor_step + 1)
                tl = torch.tensor(effective, dtype=traj_length.dtype, device=traj_length.device)
            wrapper_traj = tl.unsqueeze(0) if tl.dim() == 0 else tl
        else:
            wrapper_traj = None

        if censor_step is not None:
            eval_ts = censor_step
        elif self.model.temporal_head_enabled:
            # No censoring (full timeframe): use mean across valid timesteps
            # to distribute SHAP gradients evenly instead of only explaining
            # the last position (where gradients decay through causal chain)
            eval_ts = 'mean'
        else:
            eval_ts = -1

        has_static_cat = bg_cat_onehot is not None
        has_cont = bg['cont'].shape[1] > 0

        wrapped = SHAPModelWrapper(
            self.model, has_cat_ts=self.has_cat_ts,
            has_static_cat=has_static_cat, has_cont=has_cont,
            eval_timestep=eval_ts, traj_lengths=wrapper_traj,
        )

        bg_inputs = [bg_ts_c]
        sample_inputs = [sample_ts_c]
        if self.has_cat_ts:
            bg_inputs.append(bg_ts_cat_c.float().requires_grad_(True))
            sample_inputs.append(sample_ts_cat_c.float().requires_grad_(True))
        if has_static_cat:
            bg_inputs.append(bg_cat_onehot)
            sample_inputs.append(sample_cat_onehot)
        if has_cont:
            bg_inputs.append(bg['cont'])
            sample_inputs.append(sample_cont)

        shap_seed = cfg.get("evaluation", {}).get("shap_seed", 42)
        shap_nsamples = cfg.get("evaluation", {}).get("shap_nsamples", 200)
        explainer = shap.GradientExplainer(wrapped, bg_inputs)
        _seed_shap(shap_seed)
        shap_values = explainer.shap_values(sample_inputs, nsamples=shap_nsamples)

        # For multi-output models (e.g. 2-class): select class 1 (mortality)
        if isinstance(shap_values, list) and shap_values and isinstance(shap_values[0], list):
            n_classes = len(shap_values)
            selected_class = min(1, n_classes - 1)  # class 1 if available
            shap_values = shap_values[selected_class]

        # Strip trailing singleton class dim (GradientExplainer format b)
        if isinstance(shap_values, list):
            shap_values = [
                sv.squeeze(-1) if isinstance(sv, np.ndarray) and sv.ndim > 1 and sv.shape[-1] == 1
                else sv for sv in shap_values
            ]

        idx = 0
        ts_shap = shap_values[idx][0]
        idx += 1

        cat_ts_shap_per_cat, cat_ts_shap = None, None
        if self.has_cat_ts:
            cat_ts_shap_per_cat = shap_values[idx][0]
            cat_ts_shap = np.abs(cat_ts_shap_per_cat).mean(axis=0)
            idx += 1

        # Static categorical SHAP: one-hot encoding -> [n_cat, max_classes]
        # Aggregate over class dimension (sum absolute values per feature)
        cat_shap = np.abs(shap_values[idx][0]).sum(axis=1) if has_static_cat else None
        if has_static_cat:
            idx += 1

        cont_shap = shap_values[idx][0] if has_cont else None

        # Zero SHAP values beyond effective trajectory (padding + censored positions)
        if traj_length is not None:
            tl = int(traj_length)
            if censor_step is not None:
                tl = min(tl, censor_step + 1)
            ts_shap[:, tl:] = 0.0
            if cat_ts_shap_per_cat is not None:
                cat_ts_shap_per_cat[:, tl:] = 0.0
            if cat_ts_shap is not None:
                cat_ts_shap[tl:] = 0.0

        return {'ts_shap': ts_shap, 'cat_ts_shap': cat_ts_shap,
                'cat_ts_shap_per_category': cat_ts_shap_per_cat,
                'cat_shap': cat_shap, 'cont_shap': cont_shap,
                'n_active_background': n_active}
    
    def get_holdout_pids(self, max_samples=None):
        pids = self.data["holdout"].tab_df['PID'].tolist()
        return pids[:max_samples] if max_samples else pids
    
    def get_sample_data(self, test_loader, sample_idx):
        curr = 0
        for batch in test_loader:
            inputs, targets = batch
            x_ts, x_tab, x_ts_cat = inputs[0], inputs[1], inputs[2]
            # Extract trajectory lengths with backward compat
            if len(inputs) >= 4:
                traj_lengths = inputs[3]
            else:
                traj_lengths = _infer_trajectory_lengths_from_batch(x_ts)
            bs = x_ts.shape[0]
            if curr <= sample_idx < curr + bs:
                i = sample_idx - curr
                return (x_ts[i].to(self.device), x_ts_cat[i].to(self.device),
                        x_tab[0][i].to(self.device), x_tab[1][i].to(self.device),
                        targets[i], traj_lengths[i].to(self.device))
            curr += bs
        raise IndexError(f"Sample {sample_idx} out of range")
    
    def analyze_patient(self, test_loader, pid=None, sample_idx=None, holdout_pids=None,
                       timeframes=None, verbose=True) -> TemporalSHAPResults:
        """Run temporal SHAP analysis for a patient."""
        if pid is not None:
            if holdout_pids is None:
                raise ValueError("holdout_pids required with pid")
            sample_idx = holdout_pids.index(pid)
            if verbose: logger.debug("PID %s -> sample %s", pid, sample_idx)
        elif sample_idx is None:
            sample_idx = 0
        
        display_pid = pid or (holdout_pids[sample_idx] if holdout_pids else sample_idx)
        sample_ts, sample_ts_cat, sample_cat, sample_cont, _, traj_length = self.get_sample_data(test_loader, sample_idx)

        ts_np = sample_ts.cpu().numpy()
        # Prefer explicit traj_length over heuristic
        actual_steps = int(traj_length) if traj_length is not None else get_actual_data_length(ts_np)
        actual_min = step_to_time(actual_steps - 1) if actual_steps > 0 else 0
        actual_hours = (actual_min or 0) / 60

        # Compute inhospital boundary step for this patient
        ihs_step = None
        try:
            ihs_steps = compute_inhospital_start_steps(self.data, [display_pid])
            if ihs_steps is not None and ihs_steps[0] is not None:
                ihs_step = int(ihs_steps[0])
        except Exception:
            # base_df may not have the columns
            logger.debug("Could not compute inhospital start step for patient %s; leaving unset", display_pid)

        if verbose:
            ihs_info = f", inhospital at step {ihs_step}" if ihs_step is not None else ""
            logger.info(f"Patient {display_pid}: {actual_steps} steps ({actual_hours:.1f}h){ihs_info}")
        
        timeframes = timeframes or list(DEFAULT_TIMEFRAMES.keys())
        
        # Filter timeframes based on actual data length
        # Track the largest skipped timeframe to potentially add 'max' instead
        valid_tfs = []
        skipped_any = False
        for tf in timeframes:
            tf_h = DEFAULT_TIMEFRAMES.get(tf)
            if tf_h is None or (actual_min and tf_h * 60 <= actual_min):
                valid_tfs.append(tf)
            else:
                skipped_any = True
                if verbose:
                    logger.warning("  Skip %s (need %sh, have %.1fh)", tf, tf_h, actual_hours)
        
        # If we skipped some timeframes, add 'max' which uses actual data length
        # This replaces 'full' behavior with a named timeframe showing actual hours
        # Use a local copy to avoid mutating the global DEFAULT_TIMEFRAMES
        local_timeframes = OrderedDict(DEFAULT_TIMEFRAMES)
        if skipped_any and 'full' in valid_tfs:
            # Replace 'full' with 'max' to make it clearer this is the max available
            max_label = f'max({actual_hours:.1f}h)'
            valid_tfs = [tf if tf != 'full' else max_label for tf in valid_tfs]
            local_timeframes[max_label] = None  # None means full/no censoring
        
        if verbose: logger.info("Analyzing: %s", valid_tfs)
        
        results = {}
        t0 = time.time()
        
        for i, tf in enumerate(valid_tfs):
            tf_h = local_timeframes.get(tf)
            censor = None if tf_h is None else time_to_step(tf_h, 'h')
            if verbose: logger.info("  [%s/%s] %s...", i+1, len(valid_tfs), tf)
            
            t1 = time.time()
            shap_res = self._compute_shap_for_sample(
                sample_ts, sample_ts_cat, sample_cat, sample_cont, censor,
                traj_length=traj_length)
            n_active_bg = shap_res['n_active_background']
            if verbose:
                active_str = f", bg={n_active_bg}" if self.active_only else ""
                logger.info(f"done ({time.time()-t1:.1f}s{active_str})")

            ts_shap = shap_res['ts_shap']
            if ts_shap.ndim == 3:
                ts_shap = ts_shap[..., min(self.class_idx, ts_shap.shape[-1] - 1)]

            # Compute importance using effective steps (respects both trajectory
            # length AND censoring boundary — avoids dilution from zeroed positions)
            eff = min(actual_steps, (censor + 1) if censor is not None else actual_steps)
            if eff > 0 and self.density_normalize:
                # Density-normalized: mean |SHAP| over measured cells only
                shap_eff = np.abs(ts_shap[:, :eff])  # [n_channels, eff_steps]
                measured = ts_np[:, :eff] != 0.0       # [n_channels, eff_steps]
                denom = measured.sum(axis=1).clip(1)   # [n_channels]
                ts_channel_importance = (shap_eff * measured).sum(axis=1) / denom
                # Debug: log a sample of denominator values for verification
                if verbose:
                    logger.debug(f"    Density-normalized denominators (sample): {denom[:5]}")
            elif eff > 0:
                ts_channel_importance = np.abs(ts_shap[:, :eff]).mean(axis=1)
            else:
                ts_channel_importance = np.zeros(ts_shap.shape[0])
            # Temporal importance: mean |SHAP| across channels at each timestep
            full_steps = ts_shap.shape[1]
            if eff > 0 and self.density_normalize:
                # Density-normalized: per-timestep mean over measured channels only
                shap_eff = np.abs(ts_shap[:, :eff])      # [n_channels, eff]
                measured = ts_np[:, :eff] != 0.0           # [n_channels, eff]
                denom_t = measured.sum(axis=0).clip(1)     # [eff]
                ts_temporal_importance = np.zeros(full_steps)
                ts_temporal_importance[:eff] = (shap_eff * measured).sum(axis=0) / denom_t
            elif eff > 0:
                ts_temporal_importance = np.zeros(full_steps)
                ts_temporal_importance[:eff] = np.abs(ts_shap[:, :eff]).mean(axis=0)
            else:
                ts_temporal_importance = np.zeros(full_steps)

            # Categorical TS per-category importance (density-normalized when enabled)
            cat_ts_category_importance = None
            if shap_res['cat_ts_shap_per_category'] is not None:
                cat_ts_raw = shap_res['cat_ts_shap_per_category']
                cat_ts_data_np = sample_ts_cat.cpu().numpy()
                if eff > 0 and self.density_normalize:
                    shap_cat_eff = np.abs(cat_ts_raw[:, :eff])
                    cat_measured = cat_ts_data_np[:, :eff] != 0.0
                    cat_denom = cat_measured.sum(axis=1).clip(1)
                    cat_ts_category_importance = (shap_cat_eff * cat_measured).sum(axis=1) / cat_denom
                elif eff > 0:
                    cat_ts_category_importance = np.abs(cat_ts_raw[:, :eff]).mean(axis=1)
                else:
                    cat_ts_category_importance = np.zeros(cat_ts_raw.shape[0])

            results[tf] = TimeframeSHAPResult(
                timeframe_name=tf, timeframe_hours=tf_h, censor_step=censor,
                actual_data_steps=actual_steps, ts_shap=ts_shap,
                cat_ts_shap=shap_res['cat_ts_shap'],
                cat_ts_shap_per_category=shap_res['cat_ts_shap_per_category'],
                cat_shap=shap_res['cat_shap'], cont_shap=shap_res['cont_shap'],
                ts_data=ts_np, cat_ts_data=sample_ts_cat.cpu().numpy(),
                cat_data=sample_cat.cpu().numpy(), cont_data=sample_cont.cpu().numpy(),
                ts_channel_importance=ts_channel_importance,
                ts_temporal_importance=ts_temporal_importance,
                cat_ts_category_importance=cat_ts_category_importance,
                n_active_background=n_active_bg,
            )
        
        if verbose: logger.info("Total: %.1fs", time.time()-t0)
        
        out = TemporalSHAPResults(
            pid=display_pid, sample_idx=sample_idx,
            actual_data_length_steps=actual_steps, actual_data_length_hours=actual_hours,
            timeframe_results=results, channel2feature=self.channel2feature,
            static_cat_names=self.static_cat_names, static_cont_names=self.static_cont_names,
            encoding_info=self.encoding_info, inhospital_start_step=ihs_step,
            active_only=self.active_only,
            density_normalize=self.density_normalize,
        )
        out.stability_metrics = self._compute_stability_metrics(out)
        return out
    
    def _compute_stability_metrics(self, results: TemporalSHAPResults) -> Optional[Dict]:
        """
        Compute comprehensive stability metrics across ALL feature types.
        
        Returns dict with:
        - ts_channels: Time-series channel stability metrics
        - cat_ts: Categorical time-series (medications/procedures) stability
        - static_cat: Static categorical features stability
        - static_cont: Static continuous features stability
        - overall: Combined stability score across all feature types
        """
        tfs = list(results.timeframe_results.keys())
        if len(tfs) < 2:
            return None
        
        n = len(tfs)
        metrics = {'timeframes': tfs}
        
        # Helper function to compute correlations safely
        def safe_spearman(a, b):
            a = np.asarray(a).flatten()
            b = np.asarray(b).flatten()
            if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                return float('nan')
            if np.std(a) == 0 or np.std(b) == 0:
                return float('nan')
            result = stats.spearmanr(a, b)[0]
            return float(result) if not np.isnan(result) else float('nan')
        
        def safe_pearson(a, b):
            a = np.asarray(a).flatten()
            b = np.asarray(b).flatten()
            if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                return float('nan')
            if np.std(a) == 0 or np.std(b) == 0:
                return float('nan')
            result = stats.pearsonr(a, b)[0]
            return float(result) if not np.isnan(result) else float('nan')
        
        def compute_top_k_jaccard(imp_dict, k):
            """Compute top-k Jaccard overlap matrix."""
            m = np.zeros((n, n))
            for i, t1 in enumerate(tfs):
                for j, t2 in enumerate(tfs):
                    if len(imp_dict[t1]) >= k:
                        s1 = set(np.argsort(imp_dict[t1])[-k:])
                        s2 = set(np.argsort(imp_dict[t2])[-k:])
                        m[i, j] = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 1.0
                    else:
                        # Use all features if fewer than k
                        s1 = set(range(len(imp_dict[t1])))
                        s2 = set(range(len(imp_dict[t2])))
                        m[i, j] = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 1.0
            return m
        
        # ====================================================================
        # 1. TIME-SERIES CHANNELS
        # ====================================================================
        ts_imp = {tf: results.timeframe_results[tf].ts_channel_importance for tf in tfs}
        
        ts_rank_corr = np.zeros((n, n))
        ts_val_corr = np.zeros((n, n))
        for i, t1 in enumerate(tfs):
            for j, t2 in enumerate(tfs):
                ts_rank_corr[i, j] = safe_spearman(ts_imp[t1], ts_imp[t2])
                ts_val_corr[i, j] = safe_pearson(ts_imp[t1], ts_imp[t2])
        
        ts_top_k = {k: compute_top_k_jaccard(ts_imp, k) for k in [5, 10, 20]}
        
        metrics['ts_channels'] = {
            'rank_correlation': ts_rank_corr,
            'value_correlation': ts_val_corr,
            'top_k_overlap': ts_top_k,
            'n_features': len(ts_imp[tfs[0]])
        }
        
        # Legacy keys for backward compatibility
        metrics['channel_rank_correlation'] = ts_rank_corr
        metrics['channel_value_correlation'] = ts_val_corr
        metrics['top_k_overlap'] = ts_top_k
        
        # ====================================================================
        # 2. CATEGORICAL TIME-SERIES (medications, procedures, etc.)
        # ====================================================================
        first_result = results.timeframe_results[tfs[0]]
        
        if first_result.cat_ts_shap_per_category is not None:
            # Per-category importance: mean |SHAP| across time for each category
            cat_ts_imp = {}
            for tf in tfs:
                cat_ts_shap = results.timeframe_results[tf].cat_ts_shap_per_category
                if cat_ts_shap is not None and cat_ts_shap.size > 0:
                    # Handle different possible shapes
                    # Expected: [n_categories, seq_len] -> mean over time axis
                    if cat_ts_shap.ndim == 1:
                        cat_ts_imp[tf] = np.abs(cat_ts_shap).flatten()
                    elif cat_ts_shap.ndim == 2:
                        cat_ts_imp[tf] = np.abs(cat_ts_shap).mean(axis=1).flatten()
                    elif cat_ts_shap.ndim == 3:
                        # [n_categories, seq_len, n_classes] - take mean over time and classes
                        cat_ts_imp[tf] = np.abs(cat_ts_shap).mean(axis=(1, 2)).flatten()
                    else:
                        cat_ts_imp[tf] = np.abs(cat_ts_shap).flatten()
                else:
                    cat_ts_imp[tf] = np.array([])
            
            # Check we have valid data
            first_imp = cat_ts_imp[tfs[0]]
            if len(first_imp) > 1:
                cat_ts_rank_corr = np.zeros((n, n))
                cat_ts_val_corr = np.zeros((n, n))
                for i, t1 in enumerate(tfs):
                    for j, t2 in enumerate(tfs):
                        imp1 = cat_ts_imp[t1]
                        imp2 = cat_ts_imp[t2]
                        # Ensure same length
                        if len(imp1) == len(imp2) and len(imp1) > 1:
                            cat_ts_rank_corr[i, j] = safe_spearman(imp1, imp2)
                            cat_ts_val_corr[i, j] = safe_pearson(imp1, imp2)
                        else:
                            cat_ts_rank_corr[i, j] = np.nan
                            cat_ts_val_corr[i, j] = np.nan
                
                cat_ts_top_k = {k: compute_top_k_jaccard(cat_ts_imp, k) for k in [5, 10, 20]}
                
                metrics['cat_ts'] = {
                    'rank_correlation': cat_ts_rank_corr,
                    'value_correlation': cat_ts_val_corr,
                    'top_k_overlap': cat_ts_top_k,
                    'n_features': len(first_imp)
                }
            else:
                metrics['cat_ts'] = None
        else:
            metrics['cat_ts'] = None
        
        # ====================================================================
        # 3. STATIC CATEGORICAL FEATURES
        # ====================================================================
        if first_result.cat_shap is not None and len(first_result.cat_shap) > 0:
            static_cat_imp = {}
            for tf in tfs:
                cat_shap = results.timeframe_results[tf].cat_shap
                if cat_shap is not None:
                    # Ensure we have scalar values
                    static_cat_imp[tf] = np.array([
                        float(x) if np.isscalar(x) or x.ndim == 0 else float(np.abs(x).mean())
                        for x in cat_shap
                    ])
                else:
                    static_cat_imp[tf] = np.array([])
            
            if len(static_cat_imp[tfs[0]]) > 1:
                static_cat_rank_corr = np.zeros((n, n))
                static_cat_val_corr = np.zeros((n, n))
                for i, t1 in enumerate(tfs):
                    for j, t2 in enumerate(tfs):
                        static_cat_rank_corr[i, j] = safe_spearman(
                            np.abs(static_cat_imp[t1]), np.abs(static_cat_imp[t2])
                        )
                        static_cat_val_corr[i, j] = safe_pearson(
                            static_cat_imp[t1], static_cat_imp[t2]
                        )
                
                # Top-k for static features (use smaller k if few features)
                n_static_cat = len(static_cat_imp[tfs[0]])
                static_cat_top_k = {}
                for k in [3, 5, 10]:
                    if n_static_cat >= 2:
                        static_cat_top_k[k] = compute_top_k_jaccard(
                            {tf: np.abs(v) for tf, v in static_cat_imp.items()}, 
                            min(k, n_static_cat)
                        )
                
                metrics['static_cat'] = {
                    'rank_correlation': static_cat_rank_corr,
                    'value_correlation': static_cat_val_corr,
                    'top_k_overlap': static_cat_top_k,
                    'n_features': n_static_cat
                }
            else:
                metrics['static_cat'] = None
        else:
            metrics['static_cat'] = None
        
        # ====================================================================
        # 4. STATIC CONTINUOUS FEATURES
        # ====================================================================
        if first_result.cont_shap is not None and len(first_result.cont_shap) > 0:
            static_cont_imp = {}
            for tf in tfs:
                cont_shap = results.timeframe_results[tf].cont_shap
                if cont_shap is not None:
                    static_cont_imp[tf] = np.array([
                        float(x) if np.isscalar(x) or x.ndim == 0 else float(np.abs(x).mean())
                        for x in cont_shap
                    ])
                else:
                    static_cont_imp[tf] = np.array([])
            
            if len(static_cont_imp[tfs[0]]) > 1:
                static_cont_rank_corr = np.zeros((n, n))
                static_cont_val_corr = np.zeros((n, n))
                for i, t1 in enumerate(tfs):
                    for j, t2 in enumerate(tfs):
                        static_cont_rank_corr[i, j] = safe_spearman(
                            np.abs(static_cont_imp[t1]), np.abs(static_cont_imp[t2])
                        )
                        static_cont_val_corr[i, j] = safe_pearson(
                            static_cont_imp[t1], static_cont_imp[t2]
                        )
                
                n_static_cont = len(static_cont_imp[tfs[0]])
                static_cont_top_k = {}
                for k in [3, 5, 10]:
                    if n_static_cont >= 2:
                        static_cont_top_k[k] = compute_top_k_jaccard(
                            {tf: np.abs(v) for tf, v in static_cont_imp.items()},
                            min(k, n_static_cont)
                        )
                
                metrics['static_cont'] = {
                    'rank_correlation': static_cont_rank_corr,
                    'value_correlation': static_cont_val_corr,
                    'top_k_overlap': static_cont_top_k,
                    'n_features': n_static_cont
                }
            else:
                metrics['static_cont'] = None
        else:
            metrics['static_cont'] = None
        
        # ====================================================================
        # 5. OVERALL COMBINED STABILITY SCORE
        # ====================================================================
        # Weighted average of rank correlations across feature types
        # Weight by number of features in each type
        overall_rank_corr = np.zeros((n, n))
        total_weight = 0
        
        feature_type_weights = []
        feature_type_corrs = []
        
        if metrics['ts_channels'] is not None:
            w = metrics['ts_channels']['n_features']
            feature_type_weights.append(w)
            feature_type_corrs.append(metrics['ts_channels']['rank_correlation'])
            total_weight += w
        
        if metrics.get('cat_ts') is not None:
            w = metrics['cat_ts']['n_features']
            feature_type_weights.append(w)
            feature_type_corrs.append(metrics['cat_ts']['rank_correlation'])
            total_weight += w
        
        if metrics.get('static_cat') is not None:
            w = metrics['static_cat']['n_features']
            feature_type_weights.append(w)
            feature_type_corrs.append(metrics['static_cat']['rank_correlation'])
            total_weight += w
        
        if metrics.get('static_cont') is not None:
            w = metrics['static_cont']['n_features']
            feature_type_weights.append(w)
            feature_type_corrs.append(metrics['static_cont']['rank_correlation'])
            total_weight += w
        
        if total_weight > 0:
            for w, corr in zip(feature_type_weights, feature_type_corrs):
                # Handle NaN values
                corr_clean = np.nan_to_num(corr, nan=0.0)
                overall_rank_corr += (w / total_weight) * corr_clean
        
        metrics['overall'] = {
            'weighted_rank_correlation': overall_rank_corr,
            'feature_type_weights': {
                'ts_channels': metrics['ts_channels']['n_features'] if metrics['ts_channels'] else 0,
                'cat_ts': metrics['cat_ts']['n_features'] if metrics.get('cat_ts') else 0,
                'static_cat': metrics['static_cat']['n_features'] if metrics.get('static_cat') else 0,
                'static_cont': metrics['static_cont']['n_features'] if metrics.get('static_cont') else 0,
            },
            'total_features': total_weight
        }
        
        return metrics
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_temporal_comparison(self, results: TemporalSHAPResults, max_channels=15,
                                 figsize=(24, 20), save_path=None):
        """Side-by-side comparison panels for each timeframe."""
        tfs = results.get_available_timeframes()
        n_tf = len(tfs)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, n_tf, hspace=0.35, wspace=0.25, height_ratios=[1, 1.2, 1.5, 1])
        
        seq_len = results.timeframe_results[tfs[0]].ts_shap.shape[1]

        all_temp = [results.timeframe_results[t].ts_temporal_importance for t in tfs]
        all_chan = [results.timeframe_results[t].ts_channel_importance for t in tfs]
        temp_max = max(np.max(x) for x in all_temp)

        # Clinical-only channel selection when EBM present
        has_ebm = _has_ebm_channels(results.channel2feature)
        if has_ebm:
            clinical_mask = _get_clinical_only_channel_mask(
                results.channel2feature, len(all_chan[0]))
            clinical_chan = [ch[clinical_mask] for ch in all_chan]
            chan_max = max(np.max(x) for x in clinical_chan)
            top_clinical = np.argsort(np.mean(clinical_chan, axis=0))[-max_channels:][::-1]
            top_idx = np.array([clinical_mask[i] for i in top_clinical])
        else:
            chan_max = max(np.max(x) for x in all_chan)
            top_idx = np.argsort(np.mean(all_chan, axis=0))[-max_channels:][::-1]

        for col, tf in enumerate(tfs):
            r = results.timeframe_results[tf]
            # Handle both 'full' and 'max(X.Xh)' style timeframe names
            if r.timeframe_hours is None:
                if 'max(' in tf:
                    suffix = f"(max available)"
                else:
                    suffix = f"(full: {results.actual_data_length_hours:.1f}h)"
            else:
                suffix = f"({r.timeframe_hours}h)"

            # Compute per-timeframe EBM budget for annotation
            ebm_annotation = ''
            if has_ebm:
                tf_budget = compute_ebm_vs_clinical_budget(r.ts_shap, results.channel2feature)
                if tf_budget is not None:
                    ebm_annotation = f"  [EBM: {tf_budget['ebm_pct']:.0f}%]"

            # Active background count annotation
            active_annotation = ''
            if results.active_only and r.n_active_background is not None:
                active_annotation = f"\nn_bg={r.n_active_background}"

            # Per-column display range: zoom to effective timeframe
            eff = r.effective_steps
            margin = max(3, int(eff * 0.08))
            display_limit = min(eff + margin, seq_len)
            col_ticks = np.linspace(0, display_limit - 1, min(8, display_limit), dtype=int)
            col_tick_labels = [time_to_hours_str(step_to_time(i)) for i in col_ticks]

            # Row 1: Temporal importance
            ax1 = fig.add_subplot(gs[0, col])
            ax1.plot(r.ts_temporal_importance, lw=2, color='#ff0051')
            ax1.fill_between(range(len(r.ts_temporal_importance)), r.ts_temporal_importance, alpha=0.3, color='#ff0051')
            if r.censor_step: ax1.axvline(r.censor_step, color='black', ls='--', lw=2)
            ax1.axvline(results.actual_data_length_steps, color='gray', ls=':', lw=1.5, alpha=0.7)
            _draw_inhospital_boundary(ax1, results.inhospital_start_step, display_limit, label=(col == 0))
            ax1.set_xlim(0, display_limit); ax1.set_ylim(0, temp_max*1.1)
            ax1.set_xticks(col_ticks); ax1.set_xticklabels(col_tick_labels, rotation=45, fontsize=8)
            ax1.set_title(f'{tf} {suffix}{ebm_annotation}{active_annotation}', fontweight='bold'); ax1.grid(True, alpha=0.3)

            # Row 2: Channel bars (clinical-only when EBM present)
            ax2 = fig.add_subplot(gs[1, col])
            names = [results.channel2feature.get(int(i), f'Ch{i}') for i in top_idx]
            ax2.barh(range(len(top_idx)), r.ts_channel_importance[top_idx], color=plt.cm.Blues(np.linspace(0.4,0.9,len(top_idx))))
            ax2.set_yticks(range(len(top_idx))); ax2.set_yticklabels(names, fontsize=9)
            ax2.set_xlim(0, chan_max*1.1); ax2.invert_yaxis(); ax2.grid(True, alpha=0.3, axis='x')

            # Row 3: Heatmap
            ax3 = fig.add_subplot(gs[2, col])
            ts_top = r.ts_shap[top_idx]
            vmax = np.abs(ts_top).max()
            # Handle edge case where all values are zero or very small
            if vmax < 1e-10:
                vmax = 1e-10
            im = ax3.imshow(ts_top, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                           norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
            if r.censor_step: ax3.axvline(r.censor_step, color='black', ls='--', lw=2)
            ax3.axvline(results.actual_data_length_steps, color='gray', ls=':', lw=1.5)
            _draw_inhospital_boundary(ax3, results.inhospital_start_step, display_limit, label=False)
            ax3.set_xlim(-0.5, display_limit - 0.5)
            ax3.set_yticks(range(len(top_idx))); ax3.set_yticklabels(names, fontsize=8)
            ax3.set_xticks(col_ticks); ax3.set_xticklabels(col_tick_labels, rotation=45, fontsize=8)
            plt.colorbar(im, ax=ax3, shrink=0.8)
            
            # Row 4: Static features
            ax4 = fig.add_subplot(gs[3, col])
            static_names, static_shap = [], []
            if r.cat_shap is not None:
                for i, nm in enumerate(results.static_cat_names[:len(r.cat_shap)]):
                    static_names.append(nm)
                    # Handle both scalar and array values
                    val = r.cat_shap[i]
                    static_shap.append(float(val) if np.isscalar(val) or val.ndim == 0 else float(val.mean()))
            if r.cont_shap is not None:
                for i, nm in enumerate(results.static_cont_names[:len(r.cont_shap)]):
                    static_names.append(nm)
                    val = r.cont_shap[i]
                    static_shap.append(float(val) if np.isscalar(val) or val.ndim == 0 else float(val.mean()))
            if static_shap:
                static_shap = np.array(static_shap)
                colors = ['#ff0051' if float(x) > 0 else '#008bfb' for x in static_shap]
                ax4.barh(range(len(static_shap)), static_shap, color=colors, alpha=0.7)
                ax4.set_yticks(range(len(static_shap))); ax4.set_yticklabels(static_names, fontsize=8)
                ax4.axvline(0, color='black', lw=0.8); ax4.invert_yaxis(); ax4.grid(True, alpha=0.3, axis='x')
            else:
                ax4.text(0.5, 0.5, 'No static features', ha='center', va='center', transform=ax4.transAxes)
        
        active_label = " [active-only]" if results.active_only else ""
        dn_label = " [density-norm]" if results.density_normalize else ""
        fig.suptitle(f'Temporal SHAP - PID: {results.pid} ({results.actual_data_length_hours:.1f}h data){active_label}{dn_label}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path: ensure_parent_dir(save_path); plt.savefig(save_path, dpi=150, bbox_inches='tight'); logger.info("Saved: %s", save_path)
        return fig

    def plot_stability_heatmap(self, results: TemporalSHAPResults, figsize=(24, 20), save_path=None):
        """
        Comprehensive heatmaps of stability metrics for ALL feature types.
        
        Shows:
        - Row 1: Time-series channels (rank corr, value corr, top-5 overlap)
        - Row 2: Categorical TS (if available)
        - Row 3: Static categorical (if available)
        - Row 4: Static continuous (if available)
        - Row 5: Overall combined stability
        """
        if not results.stability_metrics:
            logger.warning("No stability metrics"); return None
        
        m = results.stability_metrics
        tfs = m['timeframes']
        
        # Count how many feature types we have
        feature_types = ['ts_channels']  # Always have this
        if m.get('cat_ts') is not None:
            feature_types.append('cat_ts')
        if m.get('static_cat') is not None:
            feature_types.append('static_cat')
        if m.get('static_cont') is not None:
            feature_types.append('static_cont')
        feature_types.append('overall')  # Always show overall
        
        n_rows = len(feature_types)
        fig, axes = plt.subplots(n_rows, 3, figsize=(figsize[0], 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        row_idx = 0
        
        # Helper to plot a row
        def plot_row(ax_row, metrics_dict, title_prefix, n_feat):
            if metrics_dict is None:
                for ax in ax_row:
                    ax.text(0.5, 0.5, 'Not available', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{title_prefix}', fontweight='bold')
                    ax.axis('off')
                return
            
            # Rank correlation
            sns.heatmap(metrics_dict['rank_correlation'], 
                       xticklabels=tfs, yticklabels=tfs,
                       annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                       ax=ax_row[0], cbar_kws={'shrink': 0.8})
            ax_row[0].set_title(f'{title_prefix}\nRank Corr (Spearman, n={n_feat})', fontweight='bold')
            
            # Value correlation
            sns.heatmap(metrics_dict['value_correlation'],
                       xticklabels=tfs, yticklabels=tfs,
                       annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                       ax=ax_row[1], cbar_kws={'shrink': 0.8})
            ax_row[1].set_title(f'{title_prefix}\nValue Corr (Pearson)', fontweight='bold')
            
            # Top-K overlap (use best available k)
            top_k = metrics_dict.get('top_k_overlap', {})
            if top_k:
                # Prefer k=5, fallback to smaller
                k = 5 if 5 in top_k else (3 if 3 in top_k else list(top_k.keys())[0])
                sns.heatmap(top_k[k],
                           xticklabels=tfs, yticklabels=tfs,
                           annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                           ax=ax_row[2], cbar_kws={'shrink': 0.8})
                ax_row[2].set_title(f'{title_prefix}\nTop-{k} Overlap (Jaccard)', fontweight='bold')
            else:
                ax_row[2].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              transform=ax_row[2].transAxes)
                ax_row[2].axis('off')
        
        # Row 1: Time-series channels
        ts_metrics = m['ts_channels']
        plot_row(axes[row_idx], ts_metrics, 'TS Channels', ts_metrics['n_features'])
        row_idx += 1
        
        # Row 2: Categorical TS (if available)
        if 'cat_ts' in feature_types:
            cat_ts_metrics = m.get('cat_ts')
            n_feat = cat_ts_metrics['n_features'] if cat_ts_metrics else 0
            plot_row(axes[row_idx], cat_ts_metrics, 'Categorical TS', n_feat)
            row_idx += 1
        
        # Row 3: Static categorical (if available)
        if 'static_cat' in feature_types:
            static_cat_metrics = m.get('static_cat')
            n_feat = static_cat_metrics['n_features'] if static_cat_metrics else 0
            plot_row(axes[row_idx], static_cat_metrics, 'Static Categorical', n_feat)
            row_idx += 1
        
        # Row 4: Static continuous (if available)
        if 'static_cont' in feature_types:
            static_cont_metrics = m.get('static_cont')
            n_feat = static_cont_metrics['n_features'] if static_cont_metrics else 0
            plot_row(axes[row_idx], static_cont_metrics, 'Static Continuous', n_feat)
            row_idx += 1
        
        # Final row: Overall combined
        overall = m.get('overall', {})
        if overall:
            # Overall weighted rank correlation
            sns.heatmap(overall['weighted_rank_correlation'],
                       xticklabels=tfs, yticklabels=tfs,
                       annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                       ax=axes[row_idx, 0], cbar_kws={'shrink': 0.8})
            axes[row_idx, 0].set_title('OVERALL\nWeighted Rank Corr', fontweight='bold')
            
            # Feature type weights summary
            weights = overall['feature_type_weights']
            weight_text = (
                f"Feature Weights:\n"
                f"  TS Channels: {weights['ts_channels']}\n"
                f"  Cat TS: {weights['cat_ts']}\n"
                f"  Static Cat: {weights['static_cat']}\n"
                f"  Static Cont: {weights['static_cont']}\n"
                f"  ─────────────\n"
                f"  Total: {overall['total_features']}"
            )
            axes[row_idx, 1].text(0.5, 0.5, weight_text, ha='center', va='center',
                                 transform=axes[row_idx, 1].transAxes, fontsize=11,
                                 family='monospace',
                                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[row_idx, 1].set_title('Feature Type Weights', fontweight='bold')
            axes[row_idx, 1].axis('off')
            
            # Summary interpretation
            # Calculate mean off-diagonal correlation as stability score
            corr_matrix = overall['weighted_rank_correlation']
            n_tf = len(tfs)
            if n_tf > 1:
                off_diag_mask = ~np.eye(n_tf, dtype=bool)
                mean_stability = np.nanmean(corr_matrix[off_diag_mask])
                
                if mean_stability >= 0.8:
                    stability_text = f"HIGH STABILITY\n(mean ρ = {mean_stability:.2f})\n\nFeature importance\nis consistent across\ntimeframes"
                    color = 'green'
                elif mean_stability >= 0.5:
                    stability_text = f"MODERATE STABILITY\n(mean ρ = {mean_stability:.2f})\n\nSome features shift\nin importance"
                    color = 'orange'
                else:
                    stability_text = f"LOW STABILITY\n(mean ρ = {mean_stability:.2f})\n\nModel uses different\nfeatures at different\ntimeframes"
                    color = 'red'
                
                axes[row_idx, 2].text(0.5, 0.5, stability_text, ha='center', va='center',
                                     transform=axes[row_idx, 2].transAxes, fontsize=12,
                                     color=color, fontweight='bold',
                                     bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
            axes[row_idx, 2].set_title('Stability Summary', fontweight='bold')
            axes[row_idx, 2].axis('off')
        
        active_label = " [active-only]" if results.active_only else ""
        fig.suptitle(f'Comprehensive SHAP Stability Analysis - PID: {results.pid}{active_label}\n'
                    f'Data: {results.actual_data_length_hours:.1f}h | Timeframes: {", ".join(tfs)}',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        if save_path:
            ensure_parent_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved: %s", save_path)
        return fig

    def plot_correlation_analysis(self, results: TemporalSHAPResults, reference='full',
                                  max_features=20, figsize=(20, 16), save_path=None):
        """
        Scatter plots comparing feature importance vs reference timeframe.
        
        Now includes ALL feature types:
        - TS Channels (top row)
        - Categorical TS (second row, if available)
        - Static features combined (third row, if available)
        """
        tfs = results.get_available_timeframes()
        
        # Find reference - prefer 'full', then 'max(...)', then last available
        if reference not in tfs:
            max_tfs = [t for t in tfs if 'max(' in t]
            reference = max_tfs[0] if max_tfs else tfs[-1]
        
        others = [t for t in tfs if t != reference]
        if not others:
            logger.warning("Need 2+ timeframes"); return None
        
        ref_result = results.timeframe_results[reference]
        
        # Determine which feature types are available
        has_cat_ts = ref_result.cat_ts_shap_per_category is not None
        has_static = (ref_result.cat_shap is not None or ref_result.cont_shap is not None)
        
        n_feature_types = 1 + int(has_cat_ts) + int(has_static)
        n_cols = len(others)
        
        fig, axes = plt.subplots(n_feature_types, n_cols, figsize=(figsize[0], 5 * n_feature_types), 
                                 squeeze=False)
        
        # Color scheme for different feature types
        colors = {'ts': '#1f77b4', 'cat_ts': '#2ca02c', 'static': '#ff7f0e'}
        
        # ====================================================================
        # ROW 1: TS CHANNELS (clinical-only when EBM present)
        # ====================================================================
        ref_ts_imp = ref_result.ts_channel_importance
        if _has_ebm_channels(results.channel2feature):
            clinical_mask = _get_clinical_only_channel_mask(
                results.channel2feature, len(ref_ts_imp))
            clinical_imp = ref_ts_imp[clinical_mask]
            top_clinical = np.argsort(clinical_imp)[-max_features:]
            top_ts_idx = np.array([clinical_mask[i] for i in top_clinical])
        else:
            top_ts_idx = np.argsort(ref_ts_imp)[-max_features:]
        ts_names = [results.channel2feature.get(int(i), f'Ch{i}') for i in top_ts_idx]
        
        for col, tf in enumerate(others):
            ax = axes[0, col]
            tf_result = results.timeframe_results[tf]
            tf_ts_imp = tf_result.ts_channel_importance
            
            x, y = ref_ts_imp[top_ts_idx], tf_ts_imp[top_ts_idx]
            ax.scatter(x, y, alpha=0.7, s=50, c=colors['ts'], label='TS Channels')
            
            # Regression line
            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                slope, intercept, r, _, _ = stats.linregress(x, y)
                x_line = np.array([x.min(), x.max()])
                ax.plot(x_line, slope * x_line + intercept, 'r--', lw=2, label=f'r={r:.3f}')
                
                # Annotate outliers
                residuals = np.abs(y - (slope * x + intercept))
                for idx in np.argsort(residuals)[-3:]:
                    ax.annotate(ts_names[idx], (x[idx], y[idx]), fontsize=8, alpha=0.8)
            
            # Identity line
            max_val = max(x.max(), y.max()) if len(x) > 0 else 1
            ax.plot([0, max_val], [0, max_val], 'k:', alpha=0.5, label='y=x')
            
            ax.set_xlabel(f'{reference} |SHAP|')
            ax.set_ylabel(f'{tf} |SHAP|')
            ax.set_title(f'TS Channels: {tf} vs {reference}', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        row_idx = 1
        
        # ====================================================================
        # ROW 2: CATEGORICAL TS (if available)
        # ====================================================================
        if has_cat_ts:
            # Get category names from encoding info
            enc_info = results.encoding_info
            cat_names = []
            if enc_info and 'feature_ranges' in enc_info:
                for feat_name, (start, end) in enc_info.get('feature_ranges', {}).items():
                    labels = enc_info.get('category_labels', {}).get(feat_name, [])
                    for i in range(end - start):
                        if i < len(labels):
                            cat_names.append(f"{feat_name}:{labels[i]}")
                        else:
                            cat_names.append(f"{feat_name}:cat_{i}")
            
            ref_cat_ts = ref_result.cat_ts_shap_per_category
            if ref_cat_ts is not None:
                # Get importance per category (mean over time)
                if ref_cat_ts.ndim == 2:
                    ref_cat_ts_imp = np.abs(ref_cat_ts).mean(axis=1)
                elif ref_cat_ts.ndim == 3:
                    ref_cat_ts_imp = np.abs(ref_cat_ts).mean(axis=(1, 2))
                else:
                    ref_cat_ts_imp = np.abs(ref_cat_ts).flatten()
                
                n_cats = len(ref_cat_ts_imp)
                top_cat_idx = np.argsort(ref_cat_ts_imp)[-min(max_features, n_cats):]
                
                # Ensure we have names for all categories
                while len(cat_names) < n_cats:
                    cat_names.append(f"cat_{len(cat_names)}")
                
                for col, tf in enumerate(others):
                    ax = axes[row_idx, col]
                    tf_result = results.timeframe_results[tf]
                    tf_cat_ts = tf_result.cat_ts_shap_per_category
                    
                    if tf_cat_ts is not None:
                        if tf_cat_ts.ndim == 2:
                            tf_cat_ts_imp = np.abs(tf_cat_ts).mean(axis=1)
                        elif tf_cat_ts.ndim == 3:
                            tf_cat_ts_imp = np.abs(tf_cat_ts).mean(axis=(1, 2))
                        else:
                            tf_cat_ts_imp = np.abs(tf_cat_ts).flatten()
                        
                        x, y = ref_cat_ts_imp[top_cat_idx], tf_cat_ts_imp[top_cat_idx]
                        ax.scatter(x, y, alpha=0.7, s=50, c=colors['cat_ts'], label='Cat TS')
                        
                        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                            slope, intercept, r, _, _ = stats.linregress(x, y)
                            x_line = np.array([x.min(), x.max()])
                            ax.plot(x_line, slope * x_line + intercept, 'r--', lw=2, label=f'r={r:.3f}')
                            
                            residuals = np.abs(y - (slope * x + intercept))
                            for idx in np.argsort(residuals)[-3:]:
                                orig_idx = top_cat_idx[idx]
                                ax.annotate(cat_names[orig_idx][:20], (x[idx], y[idx]), 
                                           fontsize=7, alpha=0.8)
                        
                        max_val = max(x.max(), y.max()) if len(x) > 0 else 1
                        ax.plot([0, max_val], [0, max_val], 'k:', alpha=0.5)
                    
                    ax.set_xlabel(f'{reference} |SHAP|')
                    ax.set_ylabel(f'{tf} |SHAP|')
                    ax.set_title(f'Categorical TS: {tf} vs {reference}', fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                
                row_idx += 1
        
        # ====================================================================
        # ROW 3: STATIC FEATURES (categorical + continuous combined)
        # ====================================================================
        if has_static:
            # Combine static categorical and continuous
            def get_static_importance(result):
                names = []
                values = []
                
                if result.cat_shap is not None:
                    for i, name in enumerate(results.static_cat_names[:len(result.cat_shap)]):
                        val = result.cat_shap[i]
                        val = float(val) if np.isscalar(val) or getattr(val, 'ndim', 1) == 0 else float(np.abs(val).mean())
                        names.append(f"[C] {name}")
                        values.append(abs(val))
                
                if result.cont_shap is not None:
                    for i, name in enumerate(results.static_cont_names[:len(result.cont_shap)]):
                        val = result.cont_shap[i]
                        val = float(val) if np.isscalar(val) or getattr(val, 'ndim', 1) == 0 else float(np.abs(val).mean())
                        names.append(f"[N] {name}")
                        values.append(abs(val))
                
                return names, np.array(values)
            
            ref_static_names, ref_static_imp = get_static_importance(ref_result)
                
            if len(ref_static_imp) > 1:
                top_static_idx = np.argsort(ref_static_imp)[-min(max_features, len(ref_static_imp)):]
                
                for col, tf in enumerate(others):
                    ax = axes[row_idx, col]
                    tf_result = results.timeframe_results[tf]
                    _, tf_static_imp = get_static_importance(tf_result)
                    
                    if len(tf_static_imp) == len(ref_static_imp):
                        x, y = ref_static_imp[top_static_idx], tf_static_imp[top_static_idx]
                        ax.scatter(x, y, alpha=0.7, s=50, c=colors['static'], label='Static')
                        
                        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                            slope, intercept, r, _, _ = stats.linregress(x, y)
                            x_line = np.array([x.min(), x.max()])
                            ax.plot(x_line, slope * x_line + intercept, 'r--', lw=2, label=f'r={r:.3f}')
                            
                            residuals = np.abs(y - (slope * x + intercept))
                            for idx in np.argsort(residuals)[-3:]:
                                orig_idx = top_static_idx[idx]
                                ax.annotate(ref_static_names[orig_idx], (x[idx], y[idx]), 
                                           fontsize=8, alpha=0.8)
                        
                        max_val = max(x.max(), y.max()) if len(x) > 0 else 1
                        ax.plot([0, max_val], [0, max_val], 'k:', alpha=0.5)
                    
                    ax.set_xlabel(f'{reference} |SHAP|')
                    ax.set_ylabel(f'{tf} |SHAP|')
                    ax.set_title(f'Static Features: {tf} vs {reference}', fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
        
        active_label = " [active-only]" if results.active_only else ""
        fig.suptitle(f'Feature Importance Correlation Analysis - PID: {results.pid}{active_label}\n'
                    f'Reference: {reference}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            ensure_parent_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved: %s", save_path)
        return fig

    def plot_feature_trajectory(self, results: TemporalSHAPResults, feature_names=None,
                               top_k=10, figsize=(14, 8), save_path=None):
        """Track feature importance across timeframes."""
        tfs = results.get_available_timeframes()
        
        if feature_names is None:
            # Find reference - prefer 'full', then 'max(...)', then last
            ref = 'full' if 'full' in tfs else next((t for t in tfs if 'max(' in t), tfs[-1])
            ref_imp = results.timeframe_results[ref].ts_channel_importance
            # Exclude EBM from auto-selection (clinical-only)
            if _has_ebm_channels(results.channel2feature):
                clinical_mask = _get_clinical_only_channel_mask(
                    results.channel2feature, len(ref_imp))
                clinical_imp = ref_imp[clinical_mask]
                top_clinical = np.argsort(clinical_imp)[-top_k:][::-1]
                top_idx = np.array([clinical_mask[i] for i in top_clinical])
            else:
                top_idx = np.argsort(ref_imp)[-top_k:][::-1]
            feature_names = [results.channel2feature.get(int(i), f'Ch{i}') for i in top_idx]
            feat_idx = top_idx
        else:
            feat_idx = [next((i for i,f in results.channel2feature.items() if f==n), None) for n in feature_names]
            feat_idx = [i for i in feat_idx if i is not None]
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))
        
        for i, (name, idx) in enumerate(zip(feature_names, feat_idx)):
            y = [results.timeframe_results[tf].ts_channel_importance[idx] for tf in tfs]
            ax.plot(range(len(tfs)), y, marker='o', lw=2, ms=8, label=name, color=colors[i])
        
        ax.set_xticks(range(len(tfs))); ax.set_xticklabels(tfs, rotation=45)
        ax.set_xlabel('Timeframe'); ax.set_ylabel('Mean |SHAP|')
        active_label = " [active-only]" if results.active_only else ""
        dn_label = " [density-norm]" if results.density_normalize else ""
        ax.set_title(f'Feature Trajectory - PID: {results.pid}{active_label}{dn_label}', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left'); ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path: ensure_parent_dir(save_path); plt.savefig(save_path, dpi=150, bbox_inches='tight'); logger.info("Saved: %s", save_path)
        return fig

    def generate_summary_report(self, results: TemporalSHAPResults) -> pd.DataFrame:
        """
        Comprehensive summary DataFrame with metrics per timeframe for ALL feature types.
        """
        tfs = results.get_available_timeframes()
        # Find reference for correlation - prefer 'full', then 'max(...)', then last
        ref_tf = 'full' if 'full' in tfs else next((t for t in tfs if 'max(' in t), tfs[-1])
        ref_result = results.timeframe_results[ref_tf]
        
        records = []
        for tf in tfs:
            r = results.timeframe_results[tf]
            
            # TS Channels
            ts_top5 = [results.channel2feature.get(int(i), f'Ch{i}') 
                      for i in np.argsort(r.ts_channel_importance)[-5:][::-1]]
            ts_corr = (stats.spearmanr(r.ts_channel_importance, 
                                       ref_result.ts_channel_importance)[0] 
                      if tf != ref_tf else 1.0)
            
            # EBM budget for this timeframe
            tf_budget = compute_ebm_vs_clinical_budget(r.ts_shap, results.channel2feature)

            record = {
                'timeframe': tf,
                'hours': r.timeframe_hours,
                'censor_step': r.censor_step,
                'effective_steps': r.effective_steps,
                'n_active_bg': r.n_active_background,

                # EBM vs Clinical budget
                'ebm_pct': tf_budget['ebm_pct'] if tf_budget else np.nan,
                'clinical_pct': tf_budget['clinical_pct'] if tf_budget else np.nan,

                # TS Channels
                'ts_top_5': ', '.join(ts_top5),
                'ts_mean_shap': np.abs(r.ts_shap).mean(),
                'ts_max_shap': np.abs(r.ts_shap).max(),
                'ts_corr_with_ref': ts_corr,
            }
            
            # Categorical TS
            if r.cat_ts_shap_per_category is not None:
                cat_ts_imp = np.abs(r.cat_ts_shap_per_category).mean(axis=1)
                record['cat_ts_mean_shap'] = cat_ts_imp.mean()
                record['cat_ts_max_shap'] = cat_ts_imp.max()
                if tf != ref_tf and ref_result.cat_ts_shap_per_category is not None:
                    ref_cat_ts_imp = np.abs(ref_result.cat_ts_shap_per_category).mean(axis=1)
                    record['cat_ts_corr_with_ref'] = stats.spearmanr(cat_ts_imp, ref_cat_ts_imp)[0]
                else:
                    record['cat_ts_corr_with_ref'] = 1.0 if tf == ref_tf else np.nan
            else:
                record['cat_ts_mean_shap'] = np.nan
                record['cat_ts_max_shap'] = np.nan
                record['cat_ts_corr_with_ref'] = np.nan
            
            # Static Categorical
            if r.cat_shap is not None and len(r.cat_shap) > 0:
                cat_vals = np.array([float(x) if np.isscalar(x) or getattr(x, 'ndim', 1) == 0 
                                    else float(np.abs(x).mean()) for x in r.cat_shap])
                record['static_cat_mean_shap'] = np.abs(cat_vals).mean()
                record['static_cat_max_shap'] = np.abs(cat_vals).max()
                
                if tf != ref_tf and ref_result.cat_shap is not None:
                    ref_cat_vals = np.array([float(x) if np.isscalar(x) or getattr(x, 'ndim', 1) == 0 
                                            else float(np.abs(x).mean()) for x in ref_result.cat_shap])
                    if len(cat_vals) > 1:
                        record['static_cat_corr_with_ref'] = stats.spearmanr(np.abs(cat_vals), 
                                                                             np.abs(ref_cat_vals))[0]
                    else:
                        record['static_cat_corr_with_ref'] = np.nan
                else:
                    record['static_cat_corr_with_ref'] = 1.0 if tf == ref_tf else np.nan
            else:
                record['static_cat_mean_shap'] = np.nan
                record['static_cat_max_shap'] = np.nan
                record['static_cat_corr_with_ref'] = np.nan
            
            # Static Continuous
            if r.cont_shap is not None and len(r.cont_shap) > 0:
                cont_vals = np.array([float(x) if np.isscalar(x) or getattr(x, 'ndim', 1) == 0 
                                     else float(np.abs(x).mean()) for x in r.cont_shap])
                record['static_cont_mean_shap'] = np.abs(cont_vals).mean()
                record['static_cont_max_shap'] = np.abs(cont_vals).max()
                
                if tf != ref_tf and ref_result.cont_shap is not None:
                    ref_cont_vals = np.array([float(x) if np.isscalar(x) or getattr(x, 'ndim', 1) == 0 
                                             else float(np.abs(x).mean()) for x in ref_result.cont_shap])
                    if len(cont_vals) > 1:
                        record['static_cont_corr_with_ref'] = stats.spearmanr(np.abs(cont_vals), 
                                                                              np.abs(ref_cont_vals))[0]
                    else:
                        record['static_cont_corr_with_ref'] = np.nan
                else:
                    record['static_cont_corr_with_ref'] = 1.0 if tf == ref_tf else np.nan
            else:
                record['static_cont_mean_shap'] = np.nan
                record['static_cont_max_shap'] = np.nan
                record['static_cont_corr_with_ref'] = np.nan
            
            records.append(record)

        return pd.DataFrame(records)

    # ====================================================================
    # COHORT-LEVEL ANALYSIS
    # ====================================================================

    def analyze_cohort(self, test_loader, holdout_pids, max_patients=20,
                       timeframes=None, verbose=True) -> CohortTemporalSHAPResults:
        """Run temporal SHAP analysis for multiple patients and aggregate.

        ``max_patients`` is the target count **per timeframe**, not a global
        cap.  Patients with short trajectories only contribute to early
        timeframes, so we keep analyzing until every timeframe has reached
        the target (or we run out of holdout patients).

        Trajectory lengths are read upfront from the dataset to decide
        which timeframes each patient can contribute to, skipping patients
        and timeframes that are already saturated — avoiding unnecessary
        SHAP computation.

        Args:
            test_loader: DataLoader for holdout set
            holdout_pids: List of all holdout PIDs (in dataloader order)
            max_patients: Target number of patients per timeframe
            timeframes: List of timeframe names (default: all standard)
            verbose: Print progress

        Returns:
            CohortTemporalSHAPResults with per-timeframe aggregated SHAP
        """
        timeframes = timeframes or list(DEFAULT_TIMEFRAMES.keys())

        # Pre-read trajectory lengths from dataset (cheap — no SHAP yet)
        holdout_ds = self.data["holdout_mixed_dls"]._train_ds
        if hasattr(holdout_ds, 'dataset'):
            holdout_ds = holdout_ds.dataset  # unwrap Subset
        all_traj = holdout_ds.traj_lengths.numpy()  # [n_holdout]

        # Track per-timeframe saturation
        tf_counts = {tf: 0 for tf in timeframes}
        patient_results = []

        for i, pid in enumerate(holdout_pids):
            if all(c >= max_patients for c in tf_counts.values()):
                break

            # Determine which timeframes this patient can contribute to
            # and that still need more patients
            traj_steps = int(all_traj[i])
            traj_hours = (step_to_time(traj_steps - 1) or 0) / 60 if traj_steps > 0 else 0
            needed_tfs = []
            for tf in timeframes:
                if tf_counts[tf] >= max_patients:
                    continue
                tf_h = DEFAULT_TIMEFRAMES.get(tf)
                if tf_h is None or tf_h <= traj_hours:
                    needed_tfs.append(tf)

            if not needed_tfs:
                continue  # this patient can't help any unsaturated timeframe

            if verbose:
                logger.info(f"\n[Patient {len(patient_results)+1}, PID {pid}] "
                            f"traj={traj_hours:.1f}h, computing {len(needed_tfs)} timeframes")
            try:
                result = self.analyze_patient(
                    test_loader, pid=pid, holdout_pids=holdout_pids,
                    timeframes=needed_tfs, verbose=verbose
                )
                patient_results.append(result)
                for tf in result.timeframe_results:
                    norm_tf = 'full' if tf.startswith('max(') else tf
                    if norm_tf in tf_counts:
                        tf_counts[norm_tf] += 1
            except Exception as e:
                logger.warning(f"  FAILED for PID {pid}: {e}")

        if not patient_results:
            raise RuntimeError("No patients were successfully analyzed")

        if verbose:
            logger.info(f"\nAnalyzed {len(patient_results)} patients. "
                        f"Per-timeframe counts: {tf_counts}")

        return self._aggregate_patient_results(patient_results)

    def _aggregate_patient_results(self, patient_results: List[TemporalSHAPResults]
                                    ) -> CohortTemporalSHAPResults:
        """Aggregate per-patient temporal SHAP into cohort-level results."""

        def _normalize_tf(tf):
            """Map patient-specific 'max(...)' back to 'full' for aggregation."""
            return 'full' if tf.startswith('max(') else tf

        # Collect per-timeframe results across patients
        tf_collections = OrderedDict()  # normalized_tf -> list of TimeframeSHAPResult
        for pr in patient_results:
            for tf, tfr in pr.timeframe_results.items():
                norm_tf = _normalize_tf(tf)
                tf_collections.setdefault(norm_tf, []).append(tfr)

        # Order timeframes by DEFAULT_TIMEFRAMES order
        ordered_tfs = [tf for tf in DEFAULT_TIMEFRAMES if tf in tf_collections]
        # Add any extra timeframes not in defaults
        for tf in tf_collections:
            if tf not in ordered_tfs:
                ordered_tfs.append(tf)

        channel_importance = OrderedDict()
        channel_importance_std = OrderedDict()
        temporal_importance = OrderedDict()
        temporal_importance_std = OrderedDict()
        ts_shap_mean = OrderedDict()
        static_cat_importance = OrderedDict()
        static_cont_importance = OrderedDict()
        patient_counts = OrderedDict()

        for tf in ordered_tfs:
            tf_results = tf_collections[tf]
            patient_counts[tf] = len(tf_results)

            # Channel importance: [n_patients, n_channels] -> mean/std
            ch_imp = np.stack([r.ts_channel_importance for r in tf_results])
            channel_importance[tf] = ch_imp.mean(axis=0)
            channel_importance_std[tf] = ch_imp.std(axis=0)

            # Temporal importance: [n_patients, seq_len] -> mean/std
            temp_imp = np.stack([r.ts_temporal_importance for r in tf_results])
            temporal_importance[tf] = temp_imp.mean(axis=0)
            temporal_importance_std[tf] = temp_imp.std(axis=0)

            # SHAP heatmap: [n_patients, n_channels, seq_len] -> mean |SHAP|
            ts_shap_mean[tf] = np.mean(
                [np.abs(r.ts_shap) for r in tf_results], axis=0)

            # Static features (squeeze trailing singleton class dim if present)
            cat_shaps = [np.abs(r.cat_shap).squeeze() for r in tf_results
                         if r.cat_shap is not None]
            cont_shaps = [np.abs(r.cont_shap).squeeze() for r in tf_results
                          if r.cont_shap is not None]
            static_cat_importance[tf] = (
                np.mean(cat_shaps, axis=0) if cat_shaps else None)
            static_cont_importance[tf] = (
                np.mean(cont_shaps, axis=0) if cont_shaps else None)

        # Categorical TS per-category importance (density-normalized when available)
        cat_ts_per_category_importance = OrderedDict()
        cat_ts_category_names = get_category_names_from_encoding_info(self.encoding_info) if self.encoding_info else []
        for tf in ordered_tfs:
            tf_results = tf_collections[tf]
            cat_ts_arrays = []
            for r in tf_results:
                if r.cat_ts_category_importance is not None:
                    cat_ts_arrays.append(r.cat_ts_category_importance)
                elif r.cat_ts_shap_per_category is not None and r.cat_ts_shap_per_category.size > 0:
                    arr = r.cat_ts_shap_per_category
                    if arr.ndim == 2:
                        cat_ts_arrays.append(np.abs(arr).mean(axis=1))
                    elif arr.ndim == 1:
                        cat_ts_arrays.append(np.abs(arr))
                    elif arr.ndim == 3:
                        cat_ts_arrays.append(np.abs(arr).mean(axis=(1, 2)))
            cat_ts_per_category_importance[tf] = (
                np.mean(cat_ts_arrays, axis=0) if cat_ts_arrays else None)

        return CohortTemporalSHAPResults(
            n_patients=len(patient_results),
            pids=[pr.pid for pr in patient_results],
            channel_importance=channel_importance,
            channel_importance_std=channel_importance_std,
            temporal_importance=temporal_importance,
            temporal_importance_std=temporal_importance_std,
            ts_shap_mean=ts_shap_mean,
            static_cat_importance=static_cat_importance,
            static_cont_importance=static_cont_importance,
            patient_counts=patient_counts,
            channel2feature=self.channel2feature,
            static_cat_names=self.static_cat_names,
            static_cont_names=self.static_cont_names,
            encoding_info=self.encoding_info,
            active_only=self.active_only,
            density_normalize=self.density_normalize,
            cat_ts_per_category_importance=cat_ts_per_category_importance,
            cat_ts_category_names=cat_ts_category_names,
            cat_ts_gate_values=self.cat_ts_gate_values,
            patient_results=patient_results,
        )

    def plot_cohort_temporal_comparison(self, results: CohortTemporalSHAPResults,
                                        max_channels=15, figsize=(18, 10),
                                        save_path=None):
        """Cohort-averaged temporal SHAP comparison across timeframes.

        2-panel layout:
          A (left):  Channel × Timeframe importance heatmap (excludes 'full')
          B (right): Static Feature × Timeframe importance heatmap
        """
        tfs = [tf for tf in results.get_available_timeframes() if tf != 'full']
        if not tfs:
            tfs = results.get_available_timeframes()
        n_tf = len(tfs)

        # ── Top channels by mean importance across timeframes ──
        has_ebm = _has_ebm_channels(results.channel2feature)
        all_chan = [results.channel_importance[t] for t in tfs]
        if has_ebm:
            clinical_mask = _get_clinical_only_channel_mask(
                results.channel2feature, len(all_chan[0]))
            clinical_chan = [ch[clinical_mask] for ch in all_chan]
            top_clinical = np.argsort(
                np.mean(clinical_chan, axis=0))[-max_channels:][::-1]
            top_idx = np.array([clinical_mask[i] for i in top_clinical])
        else:
            top_idx = np.argsort(
                np.mean(all_chan, axis=0))[-max_channels:][::-1]

        ch_names = [results.channel2feature.get(int(i), f'Ch{i}')
                    for i in top_idx]

        _dn = results.density_normalize
        _shap_label = 'Mean |SHAP| / measured cell' if _dn else 'Mean |SHAP|'

        # ── Figure layout ──
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

        # ── Panel A: Channel × Timeframe importance heatmap ──
        ax_a = fig.add_subplot(gs[0, 0])
        chan_matrix = np.column_stack(
            [results.channel_importance[tf][top_idx] for tf in tfs])  # [n_ch, n_tf]
        vmax_a = max(np.abs(chan_matrix).max(), 1e-10)
        im_a = ax_a.imshow(chan_matrix, aspect='auto', cmap='YlOrRd',
                           interpolation='nearest', vmin=0, vmax=vmax_a)
        ax_a.set_yticks(range(len(top_idx)))
        ax_a.set_yticklabels(ch_names, fontsize=10)
        tf_labels = [f'{tf}\nn={results.patient_counts[tf]}' for tf in tfs]
        ax_a.set_xticks(range(n_tf))
        ax_a.set_xticklabels(tf_labels, fontsize=9, ha='center')
        ax_a.set_xlabel('Timeframe', fontsize=10)
        ax_a.set_title('Channel Importance Across Timeframes', fontsize=12,
                       fontweight='bold')
        plt.colorbar(im_a, ax=ax_a, shrink=0.8, label=_shap_label)
        # Annotate cells with values when few enough to read
        if max_channels <= 20:
            for r in range(chan_matrix.shape[0]):
                for c in range(chan_matrix.shape[1]):
                    v = chan_matrix[r, c]
                    color = 'white' if v > vmax_a * 0.6 else 'black'
                    ax_a.text(c, r, f'{v:.4f}', ha='center', va='center',
                              fontsize=7, color=color)

        # ── Panel B: Static Feature × Timeframe heatmap ──
        ax_b = fig.add_subplot(gs[0, 1])
        static_names_all, static_matrix = [], []
        for tf in tfs:
            col_vals = []
            if tf == tfs[0]:  # build names once
                if results.static_cat_importance[tf] is not None:
                    for i, nm in enumerate(
                            results.static_cat_names[:len(results.static_cat_importance[tf])]):
                        static_names_all.append(nm)
                if results.static_cont_importance[tf] is not None:
                    for i, nm in enumerate(
                            results.static_cont_names[:len(results.static_cont_importance[tf])]):
                        static_names_all.append(nm)
            # Collect values for this timeframe
            if results.static_cat_importance[tf] is not None:
                for i in range(len(results.static_cat_importance[tf])):
                    val = results.static_cat_importance[tf][i]
                    col_vals.append(float(val) if np.isscalar(val)
                                   or getattr(val, 'ndim', 1) == 0
                                   else float(val.mean()))
            if results.static_cont_importance[tf] is not None:
                for i in range(len(results.static_cont_importance[tf])):
                    val = results.static_cont_importance[tf][i]
                    col_vals.append(float(val) if np.isscalar(val)
                                   or getattr(val, 'ndim', 1) == 0
                                   else float(val.mean()))
            static_matrix.append(col_vals)

        if static_names_all and static_matrix and len(static_matrix[0]) > 0:
            static_matrix = np.array(static_matrix).T  # [n_features, n_tf]
            # Sort by mean importance, take top 15
            mean_imp = static_matrix.mean(axis=1)
            sorted_s = np.argsort(mean_imp)[::-1][:15]
            static_matrix = static_matrix[sorted_s]
            static_names_sorted = [static_names_all[i] for i in sorted_s]

            vmax_b = max(np.abs(static_matrix).max(), 1e-10)
            im_b = ax_b.imshow(static_matrix, aspect='auto', cmap='YlOrRd',
                               interpolation='nearest', vmin=0, vmax=vmax_b)
            ax_b.set_yticks(range(len(static_names_sorted)))
            ax_b.set_yticklabels(static_names_sorted, fontsize=9)
            ax_b.set_xticks(range(n_tf))
            ax_b.set_xticklabels(tfs, fontsize=8, rotation=45, ha='right')
            ax_b.set_title('Static Features', fontsize=12, fontweight='bold')
            plt.colorbar(im_b, ax=ax_b, shrink=0.8, label=_shap_label)
        else:
            ax_b.text(0.5, 0.5, 'No static features', ha='center',
                      va='center', transform=ax_b.transAxes, fontsize=12)
            ax_b.set_title('Static Features', fontsize=12, fontweight='bold')

        # ── Suptitle & save ──
        dn_label = " [density-norm]" if results.density_normalize else ""
        active_label = " [active-only]" if results.active_only else ""
        fig.suptitle(
            f'Cohort Temporal SHAP (n={results.n_patients}){active_label}{dn_label}',
            fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            ensure_parent_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            save_base64(fig, save_path, dpi=150)
            logger.info(f"Saved: {save_path}")
        return fig

    def plot_cohort_feature_trajectory(self, results: CohortTemporalSHAPResults,
                                        top_k=10, figsize=(14, 8),
                                        save_path=None):
        """Track top feature importance across timeframes (cohort average ± std)."""
        tfs = results.get_available_timeframes()

        # Select top channels from the last (broadest) timeframe
        ref_imp = results.channel_importance[tfs[-1]]
        has_ebm = _has_ebm_channels(results.channel2feature)
        if has_ebm:
            clinical_mask = _get_clinical_only_channel_mask(
                results.channel2feature, len(ref_imp))
            clinical_imp = ref_imp[clinical_mask]
            top_clinical = np.argsort(clinical_imp)[-top_k:][::-1]
            feat_idx = np.array([clinical_mask[i] for i in top_clinical])
        else:
            feat_idx = np.argsort(ref_imp)[-top_k:][::-1]

        feature_names = [results.channel2feature.get(int(i), f'Ch{i}')
                         for i in feat_idx]

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))

        for i, (name, idx) in enumerate(zip(feature_names, feat_idx)):
            means = [results.channel_importance[tf][idx] for tf in tfs]
            stds = [results.channel_importance_std[tf][idx] for tf in tfs]
            x = range(len(tfs))
            ax.plot(x, means, marker='o', lw=2, ms=8, label=name,
                    color=colors[i])
            ax.fill_between(x,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.1, color=colors[i])

        # Annotate patient counts per timeframe
        for j, tf in enumerate(tfs):
            n = results.patient_counts[tf]
            ax.annotate(f'n={n}', (j, 0), fontsize=7, ha='center',
                        alpha=0.5, xytext=(0, -18),
                        textcoords='offset points')

        ax.set_xticks(range(len(tfs)))
        ax.set_xticklabels(tfs, rotation=45)
        ax.set_xlabel('Timeframe')
        _dn = results.density_normalize
        ax.set_ylabel('Mean |SHAP| / measured cell' if _dn else 'Mean |SHAP|')
        active_label = " [active-only]" if results.active_only else ""
        dn_label = " [density-norm]" if _dn else ""
        ax.set_title(
            f'Feature Trajectory — Cohort (n={results.n_patients}){active_label}{dn_label}',
            fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            ensure_parent_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            save_base64(fig, save_path, dpi=150)
            logger.info(f"Saved: {save_path}")
        return fig

    def generate_cohort_summary_report(self, results: CohortTemporalSHAPResults
                                        ) -> pd.DataFrame:
        """Summary DataFrame with per-timeframe cohort-averaged metrics."""
        tfs = results.get_available_timeframes()
        has_ebm = _has_ebm_channels(results.channel2feature)

        records = []
        for tf in tfs:
            ch_imp = results.channel_importance[tf]
            if has_ebm:
                clinical_mask = _get_clinical_only_channel_mask(
                    results.channel2feature, len(ch_imp))
                display_imp = ch_imp[clinical_mask]
                display_names = [results.channel2feature.get(
                    int(clinical_mask[i]), f'Ch{i}')
                    for i in range(len(clinical_mask))]
            else:
                display_imp = ch_imp
                display_names = [results.channel2feature.get(int(i), f'Ch{i}')
                                 for i in range(len(ch_imp))]

            top5_idx = np.argsort(display_imp)[-5:][::-1]
            top5 = [display_names[i] for i in top5_idx]

            tf_hours = DEFAULT_TIMEFRAMES.get(tf)
            record = {
                'timeframe': tf,
                'hours': tf_hours,
                'n_patients': results.patient_counts[tf],
                'top_5_channels': ', '.join(top5),
                'mean_channel_shap': ch_imp.mean(),
                'max_channel_shap': ch_imp.max(),
            }

            # Mean active background count when active_only is enabled
            if results.active_only and results.patient_results:
                bg_counts = []
                for pr in results.patient_results:
                    # Map patient-specific max(...) back to 'full'
                    for ptf, tfr in pr.timeframe_results.items():
                        norm = 'full' if ptf.startswith('max(') else ptf
                        if norm == tf and tfr.n_active_background is not None:
                            bg_counts.append(tfr.n_active_background)
                if bg_counts:
                    record['mean_n_active_bg'] = np.mean(bg_counts)

            # EBM budget (from mean SHAP heatmap)
            if has_ebm:
                budget = compute_ebm_vs_clinical_budget(
                    results.ts_shap_mean[tf], results.channel2feature)
                if budget:
                    record['ebm_pct'] = budget['ebm_pct']
                    record['clinical_pct'] = budget['clinical_pct']

            # Static features
            if results.static_cat_importance[tf] is not None:
                record['static_cat_mean'] = results.static_cat_importance[tf].mean()
            if results.static_cont_importance[tf] is not None:
                record['static_cont_mean'] = results.static_cont_importance[tf].mean()

            records.append(record)

        return pd.DataFrame(records)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_temporal_shap_analysis(data, model, pid=None, sample_idx=None, timeframes=None,
                               max_background_samples=200, save_dir='reports/shap',
                               verbose=False, active_only=False,
                               density_normalize: bool = False):
    """Run complete temporal SHAP analysis with all visualizations.

    Args:
        data: Data dict from prepare_data_and_dls()
        model: Trained nn.Module
        pid: Patient ID to analyze
        sample_idx: Alternative to pid
        timeframes: List of timeframe names (default: all)
        save_dir: Output directory
        verbose: Print progress
        active_only: If True, only use background patients with active
                     trajectories at each evaluation timeframe
        density_normalize: If True, normalize channel importance by measurement
                          density (per-measured-cell mean instead of per-timestep).

    Returns:
        TemporalSHAPResults
    """
    os.makedirs(save_dir, exist_ok=True)

    analyzer = TemporalSHAPAnalyzer(
        model, data, data["mixed_dls"].train, 'cuda' if torch.cuda.is_available() else 'cpu',
        max_background_samples, active_only=active_only,
        density_normalize=density_normalize,
    )
    
    holdout_pids = analyzer.get_holdout_pids()
    if pid is None and sample_idx is None:
        pid = holdout_pids[0]
        logger.info(f"Using first PID: {pid}")
    
    results = analyzer.analyze_patient(
        data["holdout_mixed_dls"].train, pid, sample_idx, holdout_pids, timeframes, verbose
    )
    
    p = results.pid
    logger.info("\nGenerating visualizations...")
    
    for name, method in [
        ('temporal_comparison', analyzer.plot_temporal_comparison),
        ('stability_heatmap', analyzer.plot_stability_heatmap),
        ('correlation_analysis', analyzer.plot_correlation_analysis),
        ('feature_trajectory', analyzer.plot_feature_trajectory),
    ]:
        fig = method(results, save_path=f'{save_dir}/{name}_pid_{p}.png')
        if fig: plt.close(fig)
    
    summary = analyzer.generate_summary_report(results)
    summary.to_csv(f'{save_dir}/temporal_shap_summary_pid_{p}.csv', index=False)
    
    if verbose:
        logger.info(f"\n{'='*60}\nTEMPORAL SHAP SUMMARY - PID: {p}\n{'='*60}")
        logger.info(f"Data: {results.actual_data_length_hours:.1f}h, Timeframes: {results.get_available_timeframes()}")
        logger.info(summary.to_string(index=False))
    
    return results


def run_cohort_temporal_shap_analysis(data, model, max_patients=20,
                                      max_background_samples=200,
                                      timeframes=None,
                                      save_dir='reports/shap',
                                      verbose=True, active_only=False,
                                      density_normalize: bool = False,
                                      representative: bool = False,
                                      representative_seed: int = 42):
    """Run temporal SHAP analysis across a cohort of holdout patients.

    This is the cohort-level counterpart of ``run_temporal_shap_analysis``.
    For each patient, SHAP values are re-computed per timeframe (with future
    data censored), then results are aggregated across the cohort.

    Args:
        data: Data dict from prepare_data_and_dls()
        model: Trained nn.Module
        max_patients: Maximum number of holdout patients to analyze
        max_background_samples: Background samples for SHAP explainer
        timeframes: List of timeframe names (default: all from DEFAULT_TIMEFRAMES)
        save_dir: Output directory for plots and CSV
        verbose: Print progress
        active_only: Only use background patients active at each timeframe
        density_normalize: Normalize channel importance by measurement density
        representative: Select patients via stratified representative sampling
            (matching cohort on outcome, trajectory, age, sex)
        representative_seed: Random seed for representative sampling

    Returns:
        CohortTemporalSHAPResults
    """
    os.makedirs(save_dir, exist_ok=True)

    analyzer = TemporalSHAPAnalyzer(
        model, data, data["mixed_dls"].train,
        'cuda' if torch.cuda.is_available() else 'cpu',
        max_background_samples, active_only=active_only,
        density_normalize=density_normalize,
    )

    holdout_pids = analyzer.get_holdout_pids()

    if representative:
        from astra.evaluation.shap_paper_figures import select_representative_sample
        selected_pids, comparison_df = select_representative_sample(
            data, n_target=max_patients, seed=representative_seed, verbose=verbose,
        )
        comparison_df.to_csv(f'{save_dir}/sample_representativeness.csv', index=False)
        # Filter holdout_pids to selected, preserving dataloader order
        selected_set = set(selected_pids)
        holdout_pids = [p for p in holdout_pids if p in selected_set]
        logger.info(f"Representative sampling: {len(holdout_pids)} patients selected")

    logger.info(f"Cohort temporal SHAP: analyzing up to {max_patients} of "
                f"{len(holdout_pids)} holdout patients")

    results = analyzer.analyze_cohort(
        data["holdout_mixed_dls"].train, holdout_pids,
        max_patients=max_patients, timeframes=timeframes, verbose=verbose,
    )

    logger.info("\nGenerating cohort visualizations...")

    # Build suffix for file names to distinguish different mode runs
    suffix_parts = []
    if active_only:
        suffix_parts.append('active')
    if density_normalize:
        suffix_parts.append('dn')
    suffix = '_' + '_'.join(suffix_parts) if suffix_parts else ''

    for name, method in [
        ('cohort_temporal_comparison',
         analyzer.plot_cohort_temporal_comparison),
        ('cohort_feature_trajectory',
         analyzer.plot_cohort_feature_trajectory),
    ]:
        fig = method(results, save_path=f'{save_dir}/{name}{suffix}.png')
        if fig:
            plt.close(fig)

    summary = analyzer.generate_cohort_summary_report(results)
    summary.to_csv(f'{save_dir}/cohort_temporal_shap_summary{suffix}.csv',
                   index=False)

    # Save full results: pickle + all-features CSV
    import pickle
    pickle_path = f'{save_dir}/cohort_temporal_shap_results{suffix}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Full results saved to {pickle_path}")

    # Detailed CSV: all features x timeframes (mean |SHAP| ± std)
    # Use float() casts to ensure numpy scalars serialize correctly in CSV.
    detail_rows = []
    for tf in results.get_available_timeframes():
        ch_imp = results.channel_importance[tf]
        ch_std = results.channel_importance_std[tf]
        for i in range(len(ch_imp)):
            detail_rows.append({
                'timeframe': tf,
                'channel_idx': i,
                'feature': results.channel2feature.get(int(i), f'Ch{i}'),
                'mean_abs_shap': float(ch_imp[i]),
                'std_abs_shap': float(ch_std[i]),
                'n_patients': results.patient_counts[tf],
            })
        # Static categorical features
        cat_imp = results.static_cat_importance.get(tf)
        if cat_imp is not None:
            for j, name in enumerate(results.static_cat_names):
                if j < len(cat_imp):
                    detail_rows.append({
                        'timeframe': tf, 'channel_idx': None,
                        'feature': f'static_cat:{name}',
                        'mean_abs_shap': float(cat_imp[j]), 'std_abs_shap': None,
                        'n_patients': results.patient_counts[tf],
                    })
        # Static continuous features
        cont_imp = results.static_cont_importance.get(tf)
        if cont_imp is not None:
            for j, name in enumerate(results.static_cont_names):
                if j < len(cont_imp):
                    detail_rows.append({
                        'timeframe': tf, 'channel_idx': None,
                        'feature': f'static_cont:{name}',
                        'mean_abs_shap': float(cont_imp[j]), 'std_abs_shap': None,
                        'n_patients': results.patient_counts[tf],
                    })
        # Categorical TS per-category features
        cat_ts_imp = results.cat_ts_per_category_importance.get(tf)
        if cat_ts_imp is not None:
            for j, name in enumerate(results.cat_ts_category_names):
                if j < len(cat_ts_imp):
                    detail_rows.append({
                        'timeframe': tf, 'channel_idx': None,
                        'feature': f'cat_ts:{name}',
                        'mean_abs_shap': float(np.mean(cat_ts_imp[j])), 'std_abs_shap': None,
                        'n_patients': results.patient_counts[tf],
                    })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = f'{save_dir}/cohort_shap_all_features{suffix}.csv'
    detail_df.to_csv(detail_path, index=False)
    logger.info(f"All-features SHAP saved to {detail_path}")

    if verbose:
        logger.info(f"\n{'='*60}\nCOHORT TEMPORAL SHAP SUMMARY (n={results.n_patients})"
                    f"\n{'='*60}")
        logger.info(summary.to_string(index=False))

    return results


# ============================================================================
# INTERACTIVE PLOTLY HEATMAPS
# ============================================================================
# Paste this entire block at the bottom of astra/evaluation/behavior.py
#
# Then in astra/inference/run_inference.py, replace default_session_plot()
# with the version in the companion snippet.
# ============================================================================

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _build_time_axis_plotly(n_steps, start_step=0):
    """Build time axis with sparse tick labels for readability."""
    all_labels = []
    for i in range(n_steps):
        t = step_to_time(start_step + i)
        all_labels.append(time_to_hours(t))
    n_ticks = min(15, n_steps)
    tick_vals = np.linspace(0, n_steps - 1, n_ticks, dtype=int).tolist()
    tick_labels = [all_labels[i] for i in tick_vals]
    return all_labels, tick_vals, tick_labels


def _hours_to_label_plotly(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f}min"
    elif hours < 24:
        return f"{int(hours)}h" if hours == int(hours) else f"{hours:.1f}h"
    else:
        d = hours / 24
        return f"{int(d)}D" if d == int(d) else f"{d:.1f}D"


def plot_continuous_ts_shap_plotly(
    shap_results: Dict,
    sample_idx: int = 0,
    channel2feature: Optional[Dict[int, str]] = None,
    eval_timestep: Optional[int] = None,
    class_idx: int = 1,
    height: int = 700,
    width: int = 1100,
    title: str = "Continuous TS SHAP Heatmap (interactive)",
    channel_subset: Optional[list] = None,
):
    """Interactive Plotly heatmap for continuous time-series SHAP values."""
    if not HAS_PLOTLY:
        logger.warning("plotly not installed — falling back to matplotlib")
        return None

    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_ch, n_steps = ts_shap.shape

    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[:, :n_steps]

    time_labels, tick_vals, tick_text = _build_time_axis_plotly(n_steps)

    if channel_subset is not None:
        ordered_idx = [idx for idx, _ in channel_subset]
        ordered_labels = [label for _, label in channel_subset]
    else:
        has_ebm = channel2feature and _has_ebm_channels(channel2feature)
        if has_ebm:
            ordered_idx, ordered_labels = _get_clinical_only_channel_order(channel2feature)
        elif channel2feature:
            ordered_idx, ordered_labels, _ = _get_grouped_channel_order(channel2feature)
        else:
            ordered_idx = list(range(n_ch))
            ordered_labels = [f'Ch{i}' for i in range(n_ch)]

    valid_idx = [i for i in ordered_idx if i < n_ch]
    valid_labels = [ordered_labels[j] for j, i in enumerate(ordered_idx) if i < n_ch]
    ordered_idx, ordered_labels = valid_idx, valid_labels

    ts_display = ts_shap[ordered_idx]
    vmax = max(abs(float(np.nanmin(ts_display))),
               abs(float(np.nanmax(ts_display))), 1e-10)

    hover = np.empty(ts_display.shape, dtype=object)
    for r in range(ts_display.shape[0]):
        for c in range(ts_display.shape[1]):
            hover[r, c] = (
                f"<b>{ordered_labels[r]}</b><br>"
                f"Time: {time_labels[c]}<br>"
                f"SHAP: {ts_display[r, c]:.5f}"
            )

    fig = go.Figure(data=go.Heatmap(
        z=ts_display,
        x=list(range(n_steps)),
        y=ordered_labels,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale='RdBu_r',
        zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="SHAP"),
    ))
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="Channel",
        height=height, width=width,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
        margin=dict(l=180),
    )
    return fig


_CONCEPT_TO_DISPLAY_GROUP = {
    'VitaleVaerdier': 'Vitals',
    'InvasiveMonitoring': 'Vitals',
    'Labsvar': 'Labs',
    'ITAOversigtsrapport': 'ICU',
    'EWS': 'Scores',
    'ISS_notes': 'Scores',
    'ISS_computed': 'Scores',
    'Medicin': 'Medicine',
    'ADTHaendelser': 'ADT',
    'Procedurer': 'Procedures',
    'Events': 'Events',
    '_ebm': 'EBM',
}

_FEATURE_GROUP_OVERRIDES = {
    'GCS': 'Scores',
    'ISS': 'Scores',
}

_DISPLAY_GROUP_ORDER = [
    'Vitals', 'Scores', 'Labs', 'ICU',
    'Medicine', 'ADT', 'Procedures', 'Events',
    'EBM', 'Other',
]


def _resolve_display_group(
    feat_name: str,
    channel_map: Optional[Dict[str, Dict]] = None,
) -> str:
    """Resolve a channel/feature name to its display group for the heatmap."""
    if feat_name in _EBM_CHANNELS:
        return 'EBM'

    if channel_map and feat_name in channel_map:
        info = channel_map[feat_name]
        raw_feature = info.get('feature', '')
        concept = info.get('concept', '')
        for prefix, group in _FEATURE_GROUP_OVERRIDES.items():
            if raw_feature.startswith(prefix):
                return group
        return _CONCEPT_TO_DISPLAY_GROUP.get(concept, 'Other')

    for concept, group in _CONCEPT_TO_DISPLAY_GROUP.items():
        if f'_{concept}_' in feat_name or feat_name.startswith(f'{concept}_'):
            return group

    return 'Other'


def _strip_concept_from_label(feat_name: str, display_group: str) -> str:
    """Create a compact display label from a feature name."""
    if ':' in feat_name:
        return feat_name.split(':', 1)[1]
    return feat_name


def plot_unified_shap_heatmap_plotly(
    shap_results: Dict,
    sample_idx: int = 0,
    channel2feature: Optional[Dict[int, str]] = None,
    channel_map: Optional[Dict[str, Dict]] = None,
    eval_timestep: Optional[int] = None,
    start_timestep: Optional[int] = None,
    class_idx: int = 1,
    height: Optional[int] = None,
    width: int = 1100,
    title: str = "SHAP Heatmap (all channels)",
):
    """Unified SHAP heatmap: continuous + categorical channels grouped by concept."""
    if not HAS_PLOTLY:
        return None

    # --- Continuous TS SHAP ---
    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_ch, n_steps = ts_shap.shape

    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[:, :n_steps]

    start_step = 0
    if isinstance(start_timestep, int) and 0 < start_timestep < n_steps:
        start_step = start_timestep
        ts_shap = ts_shap[:, start_step:]
        n_steps = ts_shap.shape[1]

    # Collect continuous channels with concept tags
    rows = []  # list of (concept, label, shap_row)
    if channel2feature:
        for ch_idx in sorted(channel2feature.keys()):
            feat = channel2feature[ch_idx]
            if feat in _SHAP_EXCLUDED_CHANNELS:
                continue
            if ch_idx >= ts_shap.shape[0]:
                continue
            concept = _resolve_display_group(feat, channel_map)
            rows.append((concept, feat, ts_shap[ch_idx]))
    else:
        for i in range(ts_shap.shape[0]):
            rows.append(('Other', f'Ch{i}', ts_shap[i]))

    # --- Categorical TS SHAP ---
    cat_shap = shap_results.get('cat_ts_shap_per_category')
    enc_info = shap_results.get('encoding_info')
    if cat_shap is not None and enc_info is not None:
        cat_data = cat_shap[sample_idx]
        if cat_data.ndim == 3:
            cat_data = cat_data[..., min(class_idx, cat_data.shape[-1] - 1)]
        cat_data = cat_data[:, start_step:start_step + n_steps]
        cat_names = get_category_names_from_encoding_info(enc_info)
        feature_ranges = enc_info.get('feature_ranges', {})
        idx_to_concept = {}
        for feat_name, (start, end) in feature_ranges.items():
            for i in range(start, end):
                idx_to_concept[i] = feat_name
        for cat_idx in range(cat_data.shape[0]):
            raw_concept = idx_to_concept.get(cat_idx, 'Other')
            concept = _CONCEPT_TO_DISPLAY_GROUP.get(raw_concept, raw_concept)
            label = cat_names[cat_idx] if cat_idx < len(cat_names) else f'cat_{cat_idx}'
            rows.append((concept, label, cat_data[cat_idx]))

    if not rows:
        return None

    # --- Group by concept, preserving order ---
    concept_order = _DISPLAY_GROUP_ORDER
    seen = set()
    ordered_concepts = []
    for c in concept_order:
        if any(r[0] == c for r in rows) and c not in seen:
            ordered_concepts.append(c)
            seen.add(c)
    for r in rows:
        if r[0] not in seen:
            ordered_concepts.append(r[0])
            seen.add(r[0])

    ordered_labels = []  # short display labels
    full_labels = []     # original names for hover
    ordered_shap = []
    group_boundaries = []
    for concept in ordered_concepts:
        group_rows = [(label, shap_row) for c, label, shap_row in rows if c == concept]
        group_rows.sort(key=lambda x: x[0])
        start = len(ordered_labels)
        for label, shap_row in group_rows:
            short = _strip_concept_from_label(label, concept)
            ordered_labels.append(short)
            full_labels.append(label)
            ordered_shap.append(shap_row)
        group_boundaries.append((concept, start, len(ordered_labels)))

    z = np.array(ordered_shap)  # [n_rows, n_steps]

    time_labels, tick_vals, tick_text = _build_time_axis_plotly(n_steps, start_step)

    vmax = max(abs(float(np.nanmin(z))), abs(float(np.nanmax(z))), 1e-10)

    hover = np.empty(z.shape, dtype=object)
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            hover[r, c] = (
                f"<b>{full_labels[r]}</b><br>"
                f"Time: {time_labels[c]}<br>"
                f"SHAP: {z[r, c]:.5f}"
            )

    n_rows = len(ordered_labels)
    if height is None:
        height = max(500, n_rows * 18)

    y_indices = list(range(n_rows))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=list(range(n_steps)),
        y=y_indices,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale='RdBu_r',
        zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="SHAP"),
    ))

    # Add concept group separators with label on the right
    for concept, start, end in group_boundaries:
        if start > 0:
            fig.add_shape(
                type="line",
                x0=0, x1=1, xref="paper",
                y0=start - 0.5, y1=start - 0.5, yref="y",
                line=dict(color="rgba(0,0,0,0.5)", width=2),
            )
        mid_row = (start + end - 1) / 2.0
        fig.add_annotation(
            x=1.01, y=mid_row,
            xref="paper", yref="y",
            text=f"<b>{concept}</b>",
            showarrow=False,
            font=dict(size=9, color="rgba(80,80,80,1)"),
            xanchor="left",
        )

    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="",
        height=height, width=width,
        yaxis=dict(
            tickvals=y_indices,
            ticktext=ordered_labels,
            tickfont=dict(size=10),
        ),
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
        margin=dict(l=160, r=80),
    )
    return fig


def plot_delta_shap_temporal_plotly(
    shap_results: Dict,
    sample_idx: int = 0,
    channel2feature: Optional[Dict[int, str]] = None,
    class_idx: int = 1,
    height: int = 350,
    width: int = 1100,
    title: str = "ΔSHAP Over Time",
):
    """ΔSHAP over time: continuous + categorical mean delta per timestep."""
    if not HAS_PLOTLY:
        return None

    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_ch, n_steps = ts_shap.shape

    eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[:, :n_steps]

    time_labels, tick_vals, tick_text = _build_time_axis_plotly(n_steps)
    x = list(range(n_steps))

    has_ebm = channel2feature and _has_ebm_channels(channel2feature)
    fig = go.Figure()

    if has_ebm:
        clinical_ch = _get_clinical_only_channel_mask(channel2feature, n_ch)
        ebm_ch = [i for i, name in channel2feature.items() if name in _EBM_CHANNELS]
        clinical_avg = np.mean(ts_shap[clinical_ch], axis=0)
        fig.add_trace(go.Scatter(
            x=x, y=clinical_avg, fill='tozeroy',
            fillcolor='rgba(0,139,251,0.15)',
            line=dict(color='#008bfb', width=2),
            name='Clinical channels',
            hovertemplate='<b>%{customdata}</b><br>Clinical ΔSHAP: %{y:.5f}<extra></extra>',
            customdata=[time_labels[i] for i in range(n_steps)],
        ))
        if ebm_ch:
            ebm_avg = np.mean(ts_shap[ebm_ch], axis=0)
            fig.add_trace(go.Scatter(
                x=x, y=ebm_avg,
                line=dict(color='#FF9800', width=2, dash='dash'),
                name='EBM',
                hovertemplate='<b>%{customdata}</b><br>EBM ΔSHAP: %{y:.5f}<extra></extra>',
                customdata=[time_labels[i] for i in range(n_steps)],
            ))
    else:
        ts_avg = np.mean(ts_shap, axis=0)
        fig.add_trace(go.Scatter(
            x=x, y=ts_avg, fill='tozeroy',
            fillcolor='rgba(255,0,81,0.2)',
            line=dict(color='#ff0051', width=2),
            name='Continuous TS',
            hovertemplate='<b>%{customdata}</b><br>ΔSHAP: %{y:.5f}<extra></extra>',
            customdata=[time_labels[i] for i in range(n_steps)],
        ))

    cat_ts = shap_results.get('cat_ts_shap')
    if cat_ts is not None:
        cat_data = cat_ts[sample_idx]
        if cat_data.ndim == 2:
            cat_data = cat_data[..., min(class_idx, cat_data.shape[-1] - 1)]
        cat_data = cat_data[:n_steps]
        fig.add_trace(go.Scatter(
            x=x, y=cat_data,
            line=dict(color='#00d4aa', width=2, dash='dash'),
            name='Categorical TS',
            hovertemplate='<b>%{customdata}</b><br>Cat ΔSHAP: %{y:.5f}<extra></extra>',
            customdata=[time_labels[i] for i in range(n_steps)],
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

    fig.update_layout(
        title=title,
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
        xaxis_title="Time", yaxis_title="Mean ΔSHAP",
        height=height, width=width,
    )
    return fig


def plot_categorical_ts_shap_plotly(
    shap_results: Dict,
    sample_idx: int = 0,
    eval_timestep: Optional[int] = None,
    class_idx: int = 1,
    height: int = 600,
    width: int = 1100,
    title: str = "Categorical TS SHAP Heatmap (interactive)",
):
    """Interactive Plotly heatmap for per-category SHAP values."""
    if not HAS_PLOTLY:
        logger.warning("plotly not installed — falling back to matplotlib")
        return None

    cat_shap = shap_results.get('cat_ts_shap_per_category')
    enc_info = shap_results.get('encoding_info')
    if cat_shap is None or enc_info is None:
        return None

    data = cat_shap[sample_idx]
    if data.ndim == 3:
        data = data[..., min(class_idx, data.shape[-1] - 1)]
    n_cats, n_steps = data.shape

    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        data = data[:, :n_steps]

    time_labels, tick_vals, tick_text = _build_time_axis_plotly(n_steps)
    cat_names = get_category_names_from_encoding_info(enc_info)
    while len(cat_names) < n_cats:
        cat_names.append(f"cat_{len(cat_names)}")
    cat_names = cat_names[:n_cats]

    vmax = max(abs(float(np.nanmin(data))),
               abs(float(np.nanmax(data))), 1e-10)

    hover = np.empty(data.shape, dtype=object)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            hover[r, c] = (
                f"<b>{cat_names[r]}</b><br>"
                f"Time: {time_labels[c]}<br>"
                f"SHAP: {data[r, c]:.5f}"
            )

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=list(range(n_steps)),
        y=cat_names,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale='RdBu_r',
        zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="SHAP"),
    ))

    for feat, (start, _) in enc_info.get('feature_ranges', {}).items():
        if start > 0:
            fig.add_hline(y=start - 0.5, line_dash="dash",
                          line_color="black", line_width=1, opacity=0.5)

    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="Category",
        height=height, width=width,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
        margin=dict(l=220),
    )
    return fig


def plot_ebm_contributions_plotly(
    ebm_explanations: Dict[float, Dict],
    top_n: int = 25,
    height: int = 650,
    width: int = 1100,
    title: str = "EBM Feature Contributions (interactive)",
):
    """Interactive Plotly heatmap of EBM local contributions across timeframes."""
    if not HAS_PLOTLY:
        logger.warning("plotly not installed — falling back to matplotlib")
        return None
    if not ebm_explanations:
        return None

    sorted_hours = sorted(ebm_explanations.keys())
    x_labels = [_hours_to_label_plotly(h) for h in sorted_hours]
    x_labels_prob = [
        f"{_hours_to_label_plotly(h)} (P={ebm_explanations[h]['predicted_prob']:.3f})"
        for h in sorted_hours
    ]

    all_features = set()
    for d in ebm_explanations.values():
        all_features.update(d['feature_names'])
    all_features_list = sorted(all_features)

    feat_to_idx = {f: i for i, f in enumerate(all_features_list)}
    matrix = np.zeros((len(all_features_list), len(sorted_hours)))
    for col, h in enumerate(sorted_hours):
        d = ebm_explanations[h]
        for name, contrib in zip(d['feature_names'], d['contributions']):
            matrix[feat_to_idx[name], col] = contrib

    max_abs = np.abs(matrix).max(axis=1)
    top_idx = np.argsort(max_abs)[::-1][:top_n]
    top_names = [all_features_list[i] for i in top_idx]
    top_matrix = matrix[top_idx]

    vmax = max(abs(float(np.nanmin(top_matrix))),
               abs(float(np.nanmax(top_matrix))), 1e-10)

    hover = np.empty(top_matrix.shape, dtype=object)
    for r in range(top_matrix.shape[0]):
        feat_name = top_names[r]
        for c in range(top_matrix.shape[1]):
            h = sorted_hours[c]
            d = ebm_explanations[h]
            val_str = ""
            if feat_name in d['feature_names']:
                fi = list(d['feature_names']).index(feat_name)
                fv = d.get('feature_values')
                if fv is not None and fi < len(fv):
                    val_str = f"<br>Value: {fv[fi]}"
            hover[r, c] = (
                f"<b>{feat_name}</b><br>"
                f"Time: {x_labels[c]}<br>"
                f"Contribution: {top_matrix[r, c]:.4f}"
                f"{val_str}"
            )

    fig = go.Figure(data=go.Heatmap(
        z=top_matrix,
        x=x_labels_prob,
        y=top_names,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale='RdBu_r',
        zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="Contrib<br>(log-odds)"),
    ))
    fig.update_layout(
        title=title, xaxis_title="EBM Timeframe", yaxis_title="Feature",
        height=height, width=width,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=250),
    )
    return fig




# ═════════════════════════════════════════════════════════════════════════
# PREDICTION TRAJECTORY (Plotly)
# ═════════════════════════════════════════════════════════════════════════


def plot_shap_budget_plotly(shap_results, sample_idx=0, channel2feature=None,
                            eval_timestep=None, class_idx=1, height=300, width=1100):
    """Interactive SHAP budget over time: EBM vs Clinical."""
    if not HAS_PLOTLY or not channel2feature or not _has_ebm_channels(channel2feature):
        return None
    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_ch, n_steps = ts_shap.shape
    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[:, :n_steps]
    budget = compute_ebm_vs_clinical_budget(ts_shap, channel2feature)
    if budget is None:
        return None
    time_labels, tick_vals, tick_text = _build_time_axis_plotly(n_steps)
    clinical_t = budget['clinical_temporal'][:n_steps]
    ebm_t = budget['ebm_temporal'][:n_steps]
    x = list(range(n_steps))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=clinical_t, fill='tozeroy',
        fillcolor='rgba(0,139,251,0.2)', line=dict(color='#008bfb', width=2),
        name=f"Clinical: {budget['clinical_pct']:.1f}%",
        hovertemplate='<b>%{customdata}</b><br>Clinical: %{y:.5f}<extra></extra>',
        customdata=[time_labels[i] for i in range(n_steps)]))
    fig.add_trace(go.Scatter(x=x, y=ebm_t, fill='tozeroy',
        fillcolor='rgba(255,152,0,0.2)', line=dict(color='#FF9800', width=2),
        name=f"EBM: {budget['ebm_pct']:.1f}%",
        hovertemplate='<b>%{customdata}</b><br>EBM: %{y:.5f}<extra></extra>',
        customdata=[time_labels[i] for i in range(n_steps)]))
    fig.update_layout(title="SHAP Budget Over Time: EBM vs Clinical",
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
        xaxis_title="Time", yaxis_title="Sum |SHAP|", height=height, width=width)
    return fig


def plot_shap_temporal_plotly(shap_results, sample_idx=0, channel2feature=None,
                              eval_timestep=None, class_idx=1, height=350, width=1100):
    """Interactive TS SHAP over time (Clinical + EBM + Categorical)."""
    if not HAS_PLOTLY:
        return None
    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_ch, n_steps = ts_shap.shape
    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[:, :n_steps]
    time_labels, tick_vals, tick_text = _build_time_axis_plotly(n_steps)
    x = list(range(n_steps))
    has_ebm = channel2feature and _has_ebm_channels(channel2feature)
    fig = go.Figure()
    if has_ebm:
        clinical_ch = _get_clinical_only_channel_mask(channel2feature, n_ch)
        ebm_ch = [i for i, name in channel2feature.items() if name in _EBM_CHANNELS]
        clinical_avg = np.abs(ts_shap[clinical_ch]).mean(axis=0)
        fig.add_trace(go.Scatter(x=x, y=clinical_avg, fill='tozeroy',
            fillcolor='rgba(0,139,251,0.15)', line=dict(color='#008bfb', width=2),
            name='Clinical channels',
            hovertemplate='<b>%{customdata}</b><br>Clinical: %{y:.5f}<extra></extra>',
            customdata=[time_labels[i] for i in range(n_steps)]))
        if ebm_ch:
            ebm_avg = np.abs(ts_shap[ebm_ch]).mean(axis=0)
            fig.add_trace(go.Scatter(x=x, y=ebm_avg,
                line=dict(color='#FF9800', width=2, dash='dash'), name='EBM (_ebm_pred)',
                hovertemplate='<b>%{customdata}</b><br>EBM: %{y:.5f}<extra></extra>',
                customdata=[time_labels[i] for i in range(n_steps)]))
    else:
        ts_avg = np.abs(ts_shap).mean(axis=0)
        fig.add_trace(go.Scatter(x=x, y=ts_avg, fill='tozeroy',
            fillcolor='rgba(255,0,81,0.2)', line=dict(color='#ff0051', width=2),
            name='Continuous TS',
            hovertemplate='<b>%{customdata}</b><br>|SHAP|: %{y:.5f}<extra></extra>',
            customdata=[time_labels[i] for i in range(n_steps)]))
    if shap_results['cat_ts_shap'] is not None:
        cat_ts = shap_results['cat_ts_shap'][sample_idx]
        if cat_ts.ndim == 2:
            cat_ts = cat_ts[..., min(class_idx, cat_ts.shape[-1] - 1)]
        cat_ts = cat_ts[:n_steps]
        fig.add_trace(go.Scatter(x=x, y=cat_ts,
            line=dict(color='#00d4aa', width=2, dash='dash'), name='Categorical TS',
            hovertemplate='<b>%{customdata}</b><br>Cat TS: %{y:.5f}<extra></extra>',
            customdata=[time_labels[i] for i in range(n_steps)]))
    fig.update_layout(title="TS SHAP Over Time",
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
        xaxis_title="Time", yaxis_title="Mean |SHAP|", height=height, width=width)
    return fig


def plot_top_channels_plotly(shap_results, sample_idx=0, channel2feature=None,
                             eval_timestep=None, class_idx=1, top_n=20, height=450, width=1100):
    """Interactive top-N channel importance."""
    if not HAS_PLOTLY:
        return None
    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_ch, n_steps = ts_shap.shape
    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps:
        ts_shap = ts_shap[:, :eval_timestep + 1]
    has_ebm = channel2feature and _has_ebm_channels(channel2feature)
    ch_imp = np.abs(ts_shap).mean(axis=1)
    if has_ebm:
        display_ch = _get_clinical_only_channel_mask(channel2feature, n_ch)
    else:
        display_ch = _get_display_channel_mask(channel2feature, n_ch) if channel2feature else list(range(n_ch))
    ch_imp_display = ch_imp[display_ch]
    sorted_display = np.argsort(ch_imp_display)[::-1]
    n_show = min(top_n, len(ch_imp_display))
    sorted_idx = [display_ch[i] for i in sorted_display[:n_show]]
    names = [channel2feature.get(i, f'Ch{i}') for i in sorted_idx] if channel2feature else [f'Ch {i}' for i in sorted_idx]
    values = ch_imp[sorted_idx]
    # Reverse so largest is at top in the horizontal bar chart
    names, values = names[::-1], values[::-1]
    hover = [f"<b>{names[i]}</b><br>Mean |SHAP|: {values[i]:.5f}" for i in range(n_show)]
    fig = go.Figure(data=go.Bar(x=values, y=names, orientation='h',
        marker_color='#008bfb', customdata=hover,
        hovertemplate="%{customdata}<extra></extra>"))
    title = f"Top {n_show} Clinical Channels" if has_ebm else f"Top {n_show} Channels"
    fig.update_layout(title=title, xaxis_title="Mean |SHAP|",
        yaxis=dict(dtick=1),
        height=max(height, n_show * 22), width=width, margin=dict(l=180))
    return fig


def plot_static_features_plotly(shap_results, sample_idx=0, feature_names_cat=None,
                                feature_names_cont=None, class_idx=1, height=350, width=1100):
    """Interactive static features (categorical + continuous)."""
    if not HAS_PLOTLY:
        return None
    from plotly.subplots import make_subplots
    has_cat = shap_results.get('cat_shap') is not None and shap_results['cat_shap'].size > 0
    has_cont = shap_results.get('cont_shap') is not None and shap_results['cont_shap'].size > 0
    if not has_cat and not has_cont:
        return None
    cols = int(has_cat) + int(has_cont)
    subtitles = []
    if has_cat: subtitles.append("Static Categorical")
    if has_cont: subtitles.append("Static Continuous")
    fig = make_subplots(rows=1, cols=cols, subplot_titles=subtitles)
    col_idx = 1
    if has_cat:
        cat_shap = shap_results['cat_shap'][sample_idx]
        cat_data = shap_results['test_data']['cat'][sample_idx]
        if cat_shap.ndim == 2:
            cat_shap = cat_shap[..., min(class_idx, cat_shap.shape[-1] - 1)]
        n = len(cat_shap)
        nms = list(feature_names_cat)[:n] if feature_names_cat else [f'Cat_{i}' for i in range(n)]
        while len(nms) < n: nms.append(f'Cat_{len(nms)}')
        labels = []
        for i in range(n):
            try: labels.append(f"{nms[i]} (val={int(cat_data[i])})" if i < len(cat_data) else nms[i])
            except Exception:
                logger.debug("Label formatting failed for static categorical feature %s; using name only", nms[i])
                labels.append(nms[i])
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in cat_shap]
        hover = [f"<b>{labels[i]}</b><br>SHAP: {cat_shap[i]:.5f}" for i in range(n)]
        fig.add_trace(go.Bar(x=cat_shap, y=labels, orientation='h', marker_color=colors,
            customdata=hover, hovertemplate="%{customdata}<extra></extra>", showlegend=False), row=1, col=col_idx)
        col_idx += 1
    if has_cont:
        cont_shap = shap_results['cont_shap'][sample_idx]
        cont_data = shap_results['test_data']['cont'][sample_idx]
        if cont_shap.ndim == 2:
            cont_shap = cont_shap[..., min(class_idx, cont_shap.shape[-1] - 1)]
        n = len(cont_shap)
        nms = list(feature_names_cont)[:n] if feature_names_cont else [f'Cont_{i}' for i in range(n)]
        while len(nms) < n: nms.append(f'Cont_{len(nms)}')
        labels = []
        for i in range(n):
            try: labels.append(f"{nms[i]} (val={float(cont_data[i]):.2f})" if i < len(cont_data) else nms[i])
            except Exception:
                logger.debug("Label formatting failed for static continuous feature %s; using name only", nms[i])
                labels.append(nms[i])
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in cont_shap]
        hover = [f"<b>{labels[i]}</b><br>SHAP: {cont_shap[i]:.5f}" for i in range(n)]
        fig.add_trace(go.Bar(x=cont_shap, y=labels, orientation='h', marker_color=colors,
            customdata=hover, hovertemplate="%{customdata}<extra></extra>", showlegend=False), row=1, col=col_idx)
    fig.update_layout(height=height, width=width)
    return fig


def plot_prediction_trajectory_plotly(
    result, ctx, model_name=None, height=350, width=1100, title=None,
):
    """Interactive Plotly prediction trajectory (temporal + non-temporal)."""
    if not HAS_PLOTLY:
        return None

    hours, probs = None, None

    if result.predictions_over_time is not None:
        traj_len = result.trajectory_length
        bin_df = ctx.bin_df
        admission = ctx.admission_time
        hours = [
            (row.bin_start - admission).total_seconds() / 3600
            + (row.bin_end - row.bin_start).total_seconds() / 7200
            for _, row in bin_df.iterrows()
        ]
        hours = hours[:traj_len]
        probs = result.predictions_over_time[:traj_len]
    else:
        if model_name is None:
            return None
        csv_path = f"reports/predictions/preds_df_{model_name}.csv"
        if not os.path.exists(csv_path):
            return None
        preds_df = pd.read_csv(csv_path)
        cohort_pid = None
        try:
            from astra.utils import get_base_df, make_inference_pid
            base = get_base_df()
            pid_map = {
                make_inference_pid(row.CPR_hash, row.ServiceDate): row.PID
                for _, row in base[["PID", "CPR_hash", "ServiceDate"]].iterrows()
            }
            cohort_pid = pid_map.get(str(ctx.pid))
        except Exception:
            logger.debug("Could not map inference PID %s to cohort PID; falling back to ctx.pid", ctx.pid)
        if cohort_pid is not None:
            patient_df = preds_df[preds_df["PID"] == cohort_pid].sort_values("time_hours")
        else:
            patient_df = preds_df[preds_df["PID"] == ctx.pid].sort_values("time_hours")
            if patient_df.empty:
                patient_df = preds_df[preds_df["PID"].astype(str) == str(ctx.pid)].sort_values("time_hours")
        if patient_df.empty:
            return None
        admission = ctx.admission_time
        bin_df = ctx.bin_df
        max_hours = (
            (bin_df.iloc[min(ctx.trajectory_length, len(bin_df)) - 1].bin_end - admission)
            .total_seconds() / 3600
        )
        patient_df = patient_df[patient_df["time_hours"] <= max_hours]
        hours = patient_df["time_hours"].values.tolist()
        probs = patient_df["pred"].values.tolist()

    if not hours or not probs:
        return None
    if title is None:
        title = f"Prediction trajectory — Patient {ctx.pid}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=probs,
        fill="tozeroy", fillcolor="rgba(70,130,180,0.15)",
        line=dict(color="steelblue", width=2),
        mode="lines+markers", marker=dict(size=4),
        name="P(deceased 30d)",
        hovertemplate="<b>%{x:.1f}h</b><br>P(deceased): %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="0.5 threshold", annotation_position="top right")
    fig.update_layout(
        title=title, xaxis_title="Hours since admission",
        yaxis_title="P(deceased 30d)", yaxis=dict(range=[-0.02, 1.02]),
        height=height, width=width, showlegend=False,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════
# DATA COMPLETENESS (Plotly)
# ═════════════════════════════════════════════════════════════════════════

def plot_data_completeness_plotly(
    shap_results, sample_idx=0, channel2feature=None,
    height_density=250, height_heatmap=500, height_bars=400, width=1100,
):
    """Interactive Plotly data completeness. Returns dict of figures."""
    if not HAS_PLOTLY:
        return None

    ts_data = shap_results["test_data"]["ts"][sample_idx]
    n_channels, n_steps_full = ts_data.shape
    eval_timestep = shap_results.get("eval_timestep")
    if isinstance(eval_timestep, int) and 0 <= eval_timestep < n_steps_full:
        n_steps = eval_timestep + 1
        ts_data = ts_data[:, :n_steps]
    else:
        n_steps = n_steps_full

    time_labels, tick_vals, tick_text = _build_time_axis_plotly(n_steps)

    _EXCLUDED = {"elapsed_hours", "bin_width_hours", "_data_present"}
    _EBM_SET = {"_ebm_pred"}
    dp_idx, eh_idx = None, None
    if channel2feature:
        for idx_ch, name in channel2feature.items():
            if name == "_data_present": dp_idx = idx_ch
            elif name == "elapsed_hours": eh_idx = idx_ch

    trajectory_mask = None
    if dp_idx is not None:
        raw_mask = ts_data[dp_idx] > 0.5
        if np.any(raw_mask):
            trajectory_mask = np.zeros(n_steps, dtype=bool)
            idxs = np.where(raw_mask)[0]
            trajectory_mask[idxs[0]:idxs[-1]+1] = True
    if trajectory_mask is None and eh_idx is not None:
        eh = ts_data[eh_idx]
        raw_mask = ~np.isnan(eh) & (np.abs(eh) > 1e-8)
        trajectory_mask = np.zeros(n_steps, dtype=bool)
        idxs = np.where(raw_mask)[0]
        if len(idxs) > 0:
            trajectory_mask[idxs[0]:idxs[-1]+1] = True
    if trajectory_mask is None:
        traj_explicit = shap_results.get("trajectory_length")
        if traj_explicit is not None:
            trajectory_mask = np.zeros(n_steps, dtype=bool)
            trajectory_mask[:min(int(traj_explicit), n_steps)] = True
        else:
            any_present = np.any(~np.isnan(ts_data), axis=0)
            trajectory_mask = np.zeros(n_steps, dtype=bool)
            idxs = np.where(any_present)[0]
            if len(idxs) > 0:
                trajectory_mask[idxs[0]:idxs[-1]+1] = True

    traj_len = int(trajectory_mask.sum())

    if channel2feature:
        groups = classify_channels(channel2feature)
        ordered_indices, ordered_labels = [], []
        group_boundaries = OrderedDict()
        row = 0
        for gname, channels in groups.items():
            start = row
            for ch_idx, feat_name in channels:
                ordered_indices.append(ch_idx)
                ordered_labels.append(feat_name)
                row += 1
            if row > start:
                group_boundaries[gname] = (start, row)
    else:
        ordered_indices = list(range(n_channels))
        ordered_labels = [f"Ch{i}" for i in range(n_channels)]
        group_boundaries = OrderedDict([("All", (0, n_channels))])

    n_display = len(ordered_indices)
    ts_subset = ts_data[ordered_indices]
    is_present = ~np.isnan(ts_subset)
    traj_broadcast = np.broadcast_to(trajectory_mask, (n_display, n_steps))
    presence = np.where(~traj_broadcast, 0, np.where(is_present, 2, 1)).astype(int)

    if traj_len > 0:
        completeness = np.array([(presence[r][trajectory_mask] == 2).sum() / traj_len
                                 for r in range(n_display)])
    else:
        completeness = np.zeros(n_display)

    clinical_rows = [r for r, ch in enumerate(ordered_indices)
                     if not channel2feature or channel2feature.get(ch, "") not in (_EXCLUDED | _EBM_SET)]
    if not clinical_rows:
        clinical_rows = list(range(n_display))

    figs = {}

    # 1. Density
    density = np.array([(presence[clinical_rows][:, t] == 2).sum() / len(clinical_rows)
                        for t in range(n_steps)]) * 100
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=list(range(n_steps)), y=density,
        fill="tozeroy", fillcolor="rgba(33,150,243,0.3)",
        line=dict(color="#1565C0", width=2), mode="lines",
        hovertemplate="<b>%{customdata}</b><br>%{y:.1f}%<extra></extra>",
        customdata=[time_labels[i] for i in range(n_steps)],
    ))
    if traj_len > 0 and traj_len < n_steps:
        fig_d.add_vline(x=np.where(trajectory_mask)[0][-1]+0.5,
                        line_dash="dash", line_color="red", opacity=0.7,
                        annotation_text="Trajectory end")
    fig_d.update_layout(title="Data density over time",
                        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
                        yaxis=dict(range=[0, 105]),
                        xaxis_title="Time", yaxis_title="% channels with data",
                        height=height_density, width=width)
    figs["density"] = fig_d

    # 2. Presence heatmap
    colorscale = [[0, "#E0E0E0"], [0.33, "#E0E0E0"],
                  [0.33, "#FF8A65"], [0.66, "#FF8A65"],
                  [0.66, "#4CAF50"], [1.0, "#4CAF50"]]
    state_names = {0: "Padding", 1: "Missing", 2: "Present"}
    hover_p = np.empty(presence.shape, dtype=object)
    for r in range(presence.shape[0]):
        for c in range(presence.shape[1]):
            hover_p[r, c] = f"<b>{ordered_labels[r]}</b><br>Time: {time_labels[c]}<br>{state_names[presence[r,c]]}"
    fig_p = go.Figure(data=go.Heatmap(
        z=presence, x=list(range(n_steps)), y=ordered_labels,
        customdata=hover_p, hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale, zmin=0, zmax=2, showscale=False))
    for gname, (s, e) in group_boundaries.items():
        if s > 0:
            fig_p.add_hline(y=s-0.5, line_dash="dash", line_color="black", line_width=1, opacity=0.5)
    fig_p.update_layout(title="Data presence (green=present, orange=missing, gray=padding)",
                        xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="Time", yaxis_title="Channel",
                        height=height_heatmap, width=width, margin=dict(l=180))
    figs["presence"] = fig_p

    # 3. Categorical
    enc_info = shap_results.get("encoding_info")
    has_cat = "ts_cat" in shap_results.get("test_data", {}) and enc_info is not None
    if has_cat:
        cat_ts = shap_results["test_data"]["ts_cat"][sample_idx][:, :n_steps]
        cat_names = get_category_names_from_encoding_info(enc_info)
        n_cats = cat_ts.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f"Cat_{len(cat_names)}")
        cat_names = cat_names[:n_cats]
        cat_pres = np.where(cat_ts > 0, 2, 1).astype(int)
        for t in range(n_steps):
            if not trajectory_mask[t]:
                cat_pres[:, t] = 0
        cat_cs = [[0,"#E0E0E0"],[0.33,"#E0E0E0"],[0.33,"#FFF9C4"],[0.66,"#FFF9C4"],[0.66,"#66BB6A"],[1.0,"#66BB6A"]]
        cat_sn = {0: "Padding", 1: "No activity", 2: "Active"}
        hc = np.empty(cat_pres.shape, dtype=object)
        for r in range(cat_pres.shape[0]):
            for c in range(cat_pres.shape[1]):
                hc[r,c] = f"<b>{cat_names[r]}</b><br>Time: {time_labels[c]}<br>{cat_sn[cat_pres[r,c]]}"
        fig_c = go.Figure(data=go.Heatmap(
            z=cat_pres, x=list(range(n_steps)), y=cat_names,
            customdata=hc, hovertemplate="%{customdata}<extra></extra>",
            colorscale=cat_cs, zmin=0, zmax=2, showscale=False))
        for feat, (s, _) in enc_info.get("feature_ranges", {}).items():
            if s > 0:
                fig_c.add_hline(y=s-0.5, line_dash="dash", line_color="black", line_width=1, opacity=0.5)
        fig_c.update_layout(title="Categorical TS (green=active, yellow=inactive, gray=padding)",
                            xaxis=dict(tickvals=tick_vals, ticktext=tick_text, tickangle=45),
                            yaxis=dict(autorange="reversed"),
                            xaxis_title="Time", yaxis_title="Category",
                            height=height_heatmap, width=width, margin=dict(l=220))
        figs["categorical"] = fig_c
    else:
        figs["categorical"] = None

    # 4. Bars
    si = np.argsort(completeness)
    sl = [ordered_labels[i] for i in si]
    sc = completeness[si] * 100
    colors = ["#4CAF50" if c >= 80 else "#FF9800" if c >= 40 else "#f44336" for c in sc]
    hb = [f"<b>{sl[i]}</b><br>{sc[i]:.1f}%" for i in range(len(sl))]
    fig_b = go.Figure(data=go.Bar(
        x=sc, y=sl, orientation="h", marker_color=colors,
        customdata=hb, hovertemplate="%{customdata}<extra></extra>"))
    fig_b.update_layout(title="Channel completeness within trajectory",
                        xaxis=dict(range=[0, 105]), xaxis_title="% Completeness",
                        yaxis=dict(autorange="reversed"),
                        height=max(height_bars, n_display*18), width=width, margin=dict(l=180))
    figs["bars"] = fig_b

    # Summary
    traj_hours = 0
    if traj_len > 0:
        t_min = step_to_time(np.where(trajectory_mask)[0][-1])
        traj_hours = t_min / 60 if t_min else 0
    figs["summary"] = {
        "trajectory_steps": traj_len, "trajectory_hours": round(traj_hours, 1),
        "n_channels": n_display, "overall_completeness": round(float(completeness.mean()*100), 1),
        "completeness": dict(zip(ordered_labels, completeness.tolist())),
    }
    return figs


def visualize_shap_individual_interactive(
    shap_results: Dict,
    sample_idx: int = 0,
    channel2feature: Optional[Dict[int, str]] = None,
    feature_names_cat: Optional[List[str]] = None,
    feature_names_cont: Optional[List[str]] = None,
    ebm_explanations: Optional[Dict[float, Dict]] = None,
    eval_timestep: Optional[int] = None,
    class_idx: int = 1,
    width: int = 1100,
):
    """
    Interactive replacement for visualize_shap_individual's heatmaps.

    Produces the same three heatmaps as the matplotlib version but with
    Plotly hover tooltips. Non-heatmap plots (bar charts, line plots) are
    still rendered via the original matplotlib function.

    Called automatically by default_session_plot when plotly is available.
    """
    if not HAS_PLOTLY:
        logger.warning("plotly not installed — call visualize_shap_individual() instead")
        return

    # 1. Continuous TS SHAP
    plot_continuous_ts_shap_plotly(
        shap_results, sample_idx=sample_idx,
        channel2feature=channel2feature,
        eval_timestep=eval_timestep,
        class_idx=class_idx, width=width,
    )

    # 2. Categorical TS SHAP
    plot_categorical_ts_shap_plotly(
        shap_results, sample_idx=sample_idx,
        eval_timestep=eval_timestep,
        class_idx=class_idx, width=width,
    )

    # 3. EBM contributions
    if ebm_explanations:
        plot_ebm_contributions_plotly(
            ebm_explanations, width=width,
        )

    # 4. Keep the original matplotlib plots for bar charts / line plots
    visualize_shap_individual(
        shap_results, sample_idx=sample_idx,
        channel2feature=channel2feature,
        feature_names_cat=feature_names_cat,
        feature_names_cont=feature_names_cont,
        class_idx=class_idx,
        eval_timestep=eval_timestep,
    )