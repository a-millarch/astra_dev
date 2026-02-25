#behavior.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import seaborn as sns
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from scipy import stats
import shap
from collections import OrderedDict
import time
import os
import logging
import pickle
from pathlib import Path

from astra.utils import cfg
from astra.models.hybrid.training import get_backbone
from astra.data.caching import prepare_data_and_dls_cached
from astra.evaluation.utils import prepare_model, step_to_time, time_to_step, time_to_hours

logger = logging.getLogger(__name__)

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


def _draw_ebm_budget_temporal(ax, budget: Dict, n_steps: int,
                              tick_idx, tick_labels,
                              title: str = 'SHAP Budget Over Time: EBM vs Clinical'):
    """Draw a stacked area chart of EBM vs Clinical SHAP budget over time."""
    clinical_t = budget['clinical_temporal'][:n_steps]
    ebm_t = budget['ebm_temporal'][:n_steps]
    x = np.arange(n_steps)

    ax.fill_between(x, 0, clinical_t, alpha=0.7, color=_GROUP_COLORS['Clinical'],
                    label=f"Clinical: {budget['clinical_pct']:.1f}%")
    ax.fill_between(x, clinical_t, clinical_t + ebm_t, alpha=0.7,
                    color=_GROUP_COLORS['EBM'],
                    label=f"EBM: {budget['ebm_pct']:.1f}%")

    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(0)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Sum |SHAP|')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
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
    n_steps: int = 114,
    top_n: int = 20,
) -> Optional[Dict]:
    """
    Load all EBM deployment models and extract global term importances.

    Returns a dict with importance matrix aligned to bin-step time axis
    (forward-filled to match how _ebm_pred is populated), or None if no
    models are found.
    """
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

class ModelWrapperWithEmbeddings(nn.Module):
    """Wrapper that takes pre-embedded categorical features."""
    def __init__(self, model, has_cat_ts=False, eval_timestep=-1):
        super().__init__()
        self.model = model
        self.has_cat_ts = has_cat_ts
        self.eval_timestep = eval_timestep

    def forward(self, x_ts, x_ts_cat_embedded=None, x_cat_embedded=None, x_cont=None):
        nan_mask = torch.isnan(x_ts)
        if nan_mask.any():
            x_ts = x_ts.clone()
            x_ts[nan_mask] = 0

        # Match model's _key_padding_mask: only mask when NaN values were present.
        # During training on clean data _key_padding_mask returns None; using a
        # zero-based mask here would compute gradients through a different attention
        # pattern, invalidating the SHAP explanation.
        key_padding_mask = None

        # Extract elapsed_hours for positional encoding (before stripping aux channels)
        if self.model.temporal_channel_idx is not None:
            elapsed_hours = x_ts[:, self.model.temporal_channel_idx, :]
        else:
            elapsed_hours = None

        # Strip auxiliary channels before W_P (same as model forward)
        x_ts_signal = x_ts[:, self.model._signal_indices, :] if self.model.exclude_channel_indices else x_ts
        x = self.model.W_P(x_ts_signal).transpose(1, 2)

        if self.has_cat_ts and x_ts_cat_embedded is not None:
            if self.model.cat_ts_combine == 'add':
                x = x + x_ts_cat_embedded
            else:
                x = torch.cat([x, x_ts_cat_embedded], dim=-1)

        if x_cat_embedded is not None and x_cat_embedded.shape[1] > 0:
            x = torch.cat([x, x_cat_embedded], 1)

        if x_cont is not None and x_cont.shape[1] > 0:
            x_cont_emb = self.model.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_emb], 1)

        x = self.model.pos_enc(x, elapsed_hours=elapsed_hours)
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)

        # Extend key_padding_mask for static tokens (never masked)
        if key_padding_mask is not None:
            n_static = x.shape[1] - key_padding_mask.shape[1]
            if n_static > 0:
                static_mask = torch.zeros(
                    key_padding_mask.shape[0], n_static,
                    dtype=torch.bool, device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat([key_padding_mask, static_mask], dim=1)

        attn_mask = self.model.causal_mask if self.model.causal else None
        x = self.model.transformer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        if self.model.temporal_head_enabled and self.model.temporal_pred_head is not None:
            logits = self.model.temporal_pred_head(x)  # [batch, seq_len]
            return logits[:, self.eval_timestep].unsqueeze(-1)  # [batch, 1]
        return self.model.head(x)


class ModelWrapperWithRawCatTS(nn.Module):
    """
    Wrapper that takes RAW multi-hot categorical TS (not pre-embedded).
    This allows SHAP to compute per-category attributions.
    """
    def __init__(self, model, has_cat_ts=False, eval_timestep=-1):
        super().__init__()
        self.model = model
        self.has_cat_ts = has_cat_ts
        self.eval_timestep = eval_timestep

    def forward(self, x_ts, x_ts_cat_raw=None, x_cat_embedded=None, x_cont=None):
        """
        Args:
            x_ts: [bs, c_in, seq_len] - continuous time series
            x_ts_cat_raw: [bs, n_categories, seq_len] - raw multi-hot categorical TS
            x_cat_embedded: [bs, n_cat, d_model] - pre-embedded static categorical
            x_cont: [bs, n_cont] - static continuous
        """
        nan_mask = torch.isnan(x_ts)
        if nan_mask.any():
            x_ts = x_ts.clone()
            x_ts[nan_mask] = 0

        # Match model's _key_padding_mask: only mask when NaN values were present.
        # During training on clean data _key_padding_mask returns None; using a
        # zero-based mask here would compute gradients through a different attention
        # pattern, invalidating the SHAP explanation.
        key_padding_mask = None

        # Extract elapsed_hours for positional encoding (before stripping aux channels)
        if self.model.temporal_channel_idx is not None:
            elapsed_hours = x_ts[:, self.model.temporal_channel_idx, :]
        else:
            elapsed_hours = None

        # Strip auxiliary channels before W_P (same as model forward)
        x_ts_signal = x_ts[:, self.model._signal_indices, :] if self.model.exclude_channel_indices else x_ts
        x = self.model.W_P(x_ts_signal).transpose(1, 2)  # [bs, seq_len, d_model]

        # Embed categorical TS from raw multi-hot (this is differentiable!)
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
                x_ts_cat_embedded = torch.stack(x_ts_cat_embedded_list, dim=0).sum(dim=0)
                x = x + x_ts_cat_embedded
            else:
                x_ts_cat_embedded = torch.cat(x_ts_cat_embedded_list, dim=-1)
                x = torch.cat([x, x_ts_cat_embedded], dim=-1)

        # Static categorical (pre-embedded)
        if x_cat_embedded is not None and x_cat_embedded.shape[1] > 0:
            x = torch.cat([x, x_cat_embedded], 1)

        # Static continuous
        if x_cont is not None and x_cont.shape[1] > 0:
            x_cont_emb = self.model.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_emb], 1)

        x = self.model.pos_enc(x, elapsed_hours=elapsed_hours)
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)

        # Extend key_padding_mask for static tokens (never masked)
        if key_padding_mask is not None:
            n_static = x.shape[1] - key_padding_mask.shape[1]
            if n_static > 0:
                static_mask = torch.zeros(
                    key_padding_mask.shape[0], n_static,
                    dtype=torch.bool, device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat([key_padding_mask, static_mask], dim=1)

        attn_mask = self.model.causal_mask if self.model.causal else None
        x = self.model.transformer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        if self.model.temporal_head_enabled and self.model.temporal_pred_head is not None:
            logits = self.model.temporal_pred_head(x)  # [batch, seq_len]
            return logits[:, self.eval_timestep].unsqueeze(-1)  # [batch, 1]
        return self.model.head(x)


def embed_categorical_ts(model, x_ts_cat, encoding_info):
    if x_ts_cat is None or model.n_ts_cat == 0:
        return None
    
    if hasattr(x_ts_cat, 'data'):
        x_ts_cat = x_ts_cat.data
    x_ts_cat = x_ts_cat.float().transpose(1, 2)
    
    with torch.no_grad():
        x_ts_cat_embedded_list = []
        dim_offset = 0
        for embed_layer, (feat_name, n_classes) in zip(model.ts_cat_embeds, model.ts_cat_dims.items()):
            feat_multi_hot = x_ts_cat[:, :, dim_offset:dim_offset + n_classes]
            feat_embedded = embed_layer(feat_multi_hot)
            x_ts_cat_embedded_list.append(feat_embedded)
            dim_offset += n_classes
        
        if model.cat_ts_combine == 'add':
            x_ts_cat_embedded = torch.stack(x_ts_cat_embedded_list, dim=0).sum(dim=0)
        else:
            x_ts_cat_embedded = torch.cat(x_ts_cat_embedded_list, dim=-1)
    
    x_ts_cat_embedded.requires_grad = True
    return x_ts_cat_embedded


def embed_categorical_features(model, x_cat):
    if x_cat is None or x_cat.shape[1] == 0:
        return None
    with torch.no_grad():
        x_cat_emb = [model.embeds[i](x_cat[:, i]).unsqueeze(1) for i in range(x_cat.shape[1])]
        x_cat_emb = torch.cat(x_cat_emb, 1)
    x_cat_emb.requires_grad = True
    return x_cat_emb


# ============================================================================
# Data Extraction
# ============================================================================

def extract_data_from_dataloader(dataloader, max_samples=None, device='cpu'):
    # Handle DataLoaders (has .train/.valid) vs single DataLoader
    if hasattr(dataloader, 'train'):
        dataloader = dataloader.train

    all_ts, all_ts_cat, all_cat, all_cont, all_y = [], [], [], [], []
    n_samples = 0

    for batch in dataloader:
        if max_samples is not None and n_samples >= max_samples:
            break
        inputs, targets = batch
        x_ts, x_tab, x_ts_cat = inputs[0], inputs[1], inputs[2]
        x_cat, x_cont = x_tab
        
        all_ts.append(x_ts.cpu())
        all_ts_cat.append(x_ts_cat.cpu())
        all_cat.append(x_cat.cpu())
        all_cont.append(x_cont.cpu())
        all_y.append(targets.cpu())
        n_samples += x_ts.shape[0]
    
    x_ts_full = torch.cat(all_ts, dim=0)
    x_ts_cat_full = torch.cat(all_ts_cat, dim=0)
    x_cat_full = torch.cat(all_cat, dim=0)
    x_cont_full = torch.cat(all_cont, dim=0)
    y_full = torch.cat(all_y, dim=0)
    
    if max_samples is not None and x_ts_full.shape[0] > max_samples:
        x_ts_full = x_ts_full[:max_samples]
        x_ts_cat_full = x_ts_cat_full[:max_samples]
        x_cat_full = x_cat_full[:max_samples]
        x_cont_full = x_cont_full[:max_samples]
        y_full = y_full[:max_samples]
    
    return (x_ts_full.to(device), x_ts_cat_full.to(device), x_cat_full.to(device), 
            x_cont_full.to(device), y_full.to(device))


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
                                     all_pids: List = None, eval_timestep: int = -1):
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
                       Default -1 (last position) is wrong for causal models — SHAP gradients
                       decay to near-zero for early steps through the long attention chain.
                       Use a fixed clinical timepoint instead, e.g.:
                           from astra.evaluation.utils import time_to_step
                           eval_timestep=time_to_step(24, 'h')  # prediction at 24 h
    """
    print("Extracting background data...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    bg_ts, bg_ts_cat, bg_cat, bg_cont, bg_y = extract_data_from_dataloader(
        background_loader, max_samples=max_background_samples, device=device)
    
    print(f"  Background samples: {bg_ts.shape[0]}")
    print(f"    Continuous TS: {bg_ts.shape}")
    print(f"    Categorical TS: {bg_ts_cat.shape}")
    print(f"    Static categorical: {bg_cat.shape}")
    print(f"    Static continuous: {bg_cont.shape}")
    
    print("Extracting test data...")
    # If specific_pids provided, extract enough samples to include them all
    extraction_max = None if specific_pids is not None else max_test_samples
    test_ts, test_ts_cat, test_cat, test_cont, test_y = extract_data_from_dataloader(
        test_loader, max_samples=extraction_max, device=device)
    print(f"  Test samples extracted: {test_ts.shape[0]}")

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
        print(f"  Filtered to {len(pid_indices)} specific PIDs")

    print(f"  Final test samples: {test_ts.shape[0]}")

    model.eval()
    model = model.to(device)
    
    print(f"\nModel has {len(model.embeds)} static categorical embeddings")
    for i, emb in enumerate(model.embeds):
        print(f"  Embed {i}: {emb.num_embeddings} classes -> {emb.embedding_dim} dim")
    
    has_cat_ts = model.n_ts_cat > 0 and bg_ts_cat is not None and bg_ts_cat.numel() > 0
    n_static_cat = bg_cat.shape[1]
    
    # Pre-embed static categorical (always)
    bg_cat_emb = embed_categorical_features(model, bg_cat) if n_static_cat > 0 else None
    test_cat_emb = embed_categorical_features(model, test_cat) if n_static_cat > 0 else None
    
    if model.temporal_head_enabled:
        print(f"  Temporal head: eval_timestep={eval_timestep} "
              f"({'last position — consider a clinical timepoint' if eval_timestep == -1 else 'OK'})")

    if bg_cat_emb is not None:
        print(f"  Static categorical embedded: {bg_cat_emb.shape}")
    
    print(f"\ncompute_per_category_shap: {compute_per_category_shap}")
    
    if compute_per_category_shap and has_cat_ts:
        # Use wrapper that takes RAW categorical TS for per-category SHAP
        print("  Using ModelWrapperWithRawCatTS for per-category SHAP values")
        wrapped_model = ModelWrapperWithRawCatTS(model, has_cat_ts=has_cat_ts,
                                                 eval_timestep=eval_timestep)

        # Ensure categorical TS is float and requires grad
        bg_ts_cat_input = bg_ts_cat.float()
        test_ts_cat_input = test_ts_cat.float()
        bg_ts_cat_input.requires_grad = True
        test_ts_cat_input.requires_grad = True

        bg_inputs = [bg_ts, bg_ts_cat_input]
        test_inputs = [test_ts, test_ts_cat_input]
    else:
        # Use wrapper with pre-embedded categorical TS (faster, less granular)
        print("  Using ModelWrapperWithEmbeddings (embedded categorical TS)")
        wrapped_model = ModelWrapperWithEmbeddings(model, has_cat_ts=has_cat_ts,
                                                   eval_timestep=eval_timestep)
        
        if has_cat_ts:
            bg_ts_cat_emb = embed_categorical_ts(model, bg_ts_cat, encoding_info)
            test_ts_cat_emb = embed_categorical_ts(model, test_ts_cat, encoding_info)
            bg_inputs = [bg_ts, bg_ts_cat_emb]
            test_inputs = [test_ts, test_ts_cat_emb]
        else:
            bg_inputs = [bg_ts]
            test_inputs = [test_ts]
    
    if bg_cat_emb is not None:
        bg_inputs.append(bg_cat_emb)
        test_inputs.append(test_cat_emb)
    
    if bg_cont.shape[1] > 0:
        bg_inputs.append(bg_cont)
        test_inputs.append(test_cont)
    
    print("\nCreating SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(wrapped_model, bg_inputs)
    
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(test_inputs)
    print("SHAP calculation complete!")

    # For multi-output models (e.g. 2-class), GradientExplainer returns
    # [[sv_per_input_class0], [sv_per_input_class1]].
    # Select class 1 (mortality) by default.
    if isinstance(shap_values, list) and len(shap_values) > 0:
        if isinstance(shap_values[0], list):
            n_classes = len(shap_values)
            selected_class = min(1, n_classes - 1)  # class 1 if available
            print(f"  Multi-output model: {n_classes} classes, selecting class {selected_class}")
            shap_values = shap_values[selected_class]
    
    print("\nSHAP value shapes:")
    for i, sv in enumerate(shap_values):
        print(f"  shap_values[{i}]: {sv.shape}")
    
    # Parse SHAP values
    idx = 0
    ts_shap = shap_values[idx]; idx += 1
    
    cat_ts_shap_per_category = None
    cat_ts_shap = None
    cat_ts_shap_embedded = None
    
    if has_cat_ts:
        if compute_per_category_shap:
            # Raw multi-hot SHAP: [n_samples, n_categories, seq_len, n_classes]
            cat_ts_shap_per_category = shap_values[idx]
            print(f"  cat_ts_shap_per_category: {cat_ts_shap_per_category.shape}")
            # Also compute per-timestep importance (mean across categories)
            cat_ts_shap = np.abs(cat_ts_shap_per_category).mean(axis=1)
        else:
            # Embedded SHAP: [n_samples, seq_len, d_model, n_classes]
            cat_ts_shap_embedded = shap_values[idx]
            cat_ts_shap = np.abs(cat_ts_shap_embedded).mean(axis=2)
        idx += 1
    
    cat_shap, cat_shap_embedded = None, None
    if bg_cat_emb is not None:
        cat_shap_embedded = shap_values[idx]
        print(f"  cat_shap_embedded shape: {cat_shap_embedded.shape}")
        cat_shap = cat_shap_embedded.mean(axis=2)
        print(f"  cat_shap after mean: {cat_shap.shape}")
        idx += 1
    
    cont_shap = shap_values[idx] if bg_cont.shape[1] > 0 else None
    
    return {
        'ts_shap': ts_shap,
        'cat_ts_shap': cat_ts_shap,
        'cat_ts_shap_per_category': cat_ts_shap_per_category,  # NEW: per-category SHAP
        'cat_ts_shap_embedded': cat_ts_shap_embedded,
        'cat_shap': cat_shap,
        'cat_shap_embedded': cat_shap_embedded,
        'cont_shap': cont_shap,
        'n_static_cat': n_static_cat,
        'eval_timestep': eval_timestep,  # stored for visualization cropping
        'test_data': {
            'ts': test_ts.cpu().numpy(),
            'ts_cat': test_ts_cat.cpu().numpy(),
            'cat': test_cat.cpu().numpy(),
            'cont': test_cont.cpu().numpy(),
            'y': test_y.cpu().numpy()
        },
        'background_data': {
            'ts': bg_ts.cpu().numpy(),
            'ts_cat': bg_ts_cat.cpu().numpy(),
            'cat': bg_cat.cpu().numpy(),
            'cont': bg_cont.cpu().numpy(),
            'y': bg_y.cpu().numpy()
        },
        'encoding_info': encoding_info
    }


# ============================================================================
# Debug function
# ============================================================================

def debug_shap_data(shap_results, feature_names_cat=None, feature_names_cont=None):
    """Debug function to identify data shape mismatches."""
    print("=" * 60)
    print("SHAP RESULTS DEBUG")
    print("=" * 60)
    
    print("\n1. Time Series SHAP:")
    print(f"   ts_shap shape: {shap_results['ts_shap'].shape}")
    
    print("\n2. Categorical TS SHAP:")
    if shap_results['cat_ts_shap'] is not None:
        print(f"   cat_ts_shap shape: {shap_results['cat_ts_shap'].shape}")
    else:
        print("   cat_ts_shap: None")
    
    print("\n3. Static Categorical SHAP:")
    if shap_results['cat_shap'] is not None:
        print(f"   cat_shap shape: {shap_results['cat_shap'].shape}")
        n_from_shap = shap_results['cat_shap'].shape[1]
        print(f"   Number of static cat features (from SHAP): {n_from_shap}")
    else:
        print("   cat_shap: None")
        n_from_shap = 0
    
    print(f"\n   feature_names_cat provided: {feature_names_cat}")
    n_from_names = len(feature_names_cat) if feature_names_cat else 0
    print(f"   Number of names provided: {n_from_names}")
    
    if n_from_shap != n_from_names:
        print(f"\n   ⚠️  MISMATCH! SHAP has {n_from_shap} features but {n_from_names} names provided")
        print(f"   TIP: Use get_static_cat_names_from_classes(data['classes']) to get all names")
    
    print("\n4. Static Continuous SHAP:")
    if shap_results['cont_shap'] is not None:
        print(f"   cont_shap shape: {shap_results['cont_shap'].shape}")
    
    print("\n5. Test Data Shapes:")
    print(f"   cat: {shap_results['test_data']['cat'].shape}")
    print(f"   cont: {shap_results['test_data']['cont'].shape}")
    
    print("\n6. Encoding Info (Categorical TS):")
    enc = shap_results.get('encoding_info', {})
    print(f"   Keys: {list(enc.keys())}")
    if 'feature_ranges' in enc:
        print(f"   feature_ranges:")
        for feat, (start, end) in enc['feature_ranges'].items():
            print(f"      {feat}: indices {start}-{end} ({end-start} categories)")
    if 'category_labels' in enc:
        print(f"   category_labels:")
        for feat, labels in enc['category_labels'].items():
            print(f"      {feat}: {len(labels)} labels - {labels[:3]}..." if len(labels) > 3 else f"      {feat}: {labels}")
    print("=" * 60)


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
                               eval_timestep: Optional[int] = None,
                               ebm_importances: Optional[Dict] = None):
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
        ebm_importances: Dict from load_ebm_global_importances() with EBM feature
            importance data, or None to skip EBM importance panels.

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
        print(f"Found PID {pid} at sample index {sample_idx}")
    elif sample_idx is None:
        sample_idx = 0
        print(f"No sample_idx or pid provided, using sample_idx=0")
    
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
    if eval_timestep is not None and 0 <= eval_timestep < n_steps:
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
    has_ebm_imp = has_ebm and ebm_importances is not None

    if has_ebm and has_ebm_imp:
        fig = plt.figure(figsize=(22, 28))
        gs = fig.add_gridspec(8, 2, hspace=0.4, wspace=0.3,
                              height_ratios=[0.7, 0.7, 1, 1, 1, 1, 1, 1])
        row_offset = 3
        # Row 0: EBM budget over time
        ax_budget = fig.add_subplot(gs[0, :])
        _draw_ebm_budget_temporal(ax_budget, budget, n_steps, tick_idx,
                                  [time_fmt[i] for i in tick_idx],
                                  title=f'SHAP Budget Over Time{title_suffix}')
        # Row 1: EBM importance lines
        ax_ebm_lines = fig.add_subplot(gs[1, :])
        _draw_ebm_importance_lines(ax_ebm_lines, ebm_importances, n_steps,
                                    tick_idx, [time_fmt[i] for i in tick_idx],
                                    title=f'Top EBM Features Over Time{title_suffix}')
        # Row 2: EBM importance heatmap
        ax_ebm_hm = fig.add_subplot(gs[2, :])
        _draw_ebm_importance_heatmap(ax_ebm_hm, ebm_importances, n_steps,
                                      tick_idx, [time_fmt[i] for i in tick_idx],
                                      title=f'EBM Feature Importance Over Time{title_suffix}')
    elif has_ebm:
        fig = plt.figure(figsize=(22, 23))
        gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3,
                              height_ratios=[0.7, 1, 1, 1, 1, 1])
        row_offset = 1
        # Row 0: EBM budget over time
        ax_budget = fig.add_subplot(gs[0, :])
        _draw_ebm_budget_temporal(ax_budget, budget, n_steps, tick_idx,
                                  [time_fmt[i] for i in tick_idx],
                                  title=f'SHAP Budget Over Time{title_suffix}')
    else:
        fig = plt.figure(figsize=(22, 20))
        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1, 1])
        row_offset = 0

    # Plot 1: TS importance over time
    ax1 = fig.add_subplot(gs[0 + row_offset, :])

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

    ax1.set_xlabel('Time'); ax1.set_ylabel('|SHAP Value|')
    ax1.set_title(f'TS SHAP Over Time{title_suffix}, Class {class_idx}', fontweight='bold')
    ax1.set_xticks(tick_idx); ax1.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Plot 2: Continuous TS heatmap — clinical-only when EBM present
    ax2 = fig.add_subplot(gs[1 + row_offset, :])
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
    ax2.set_xlabel('Time'); ax2.set_ylabel('Channel')
    heatmap_title = 'Clinical Continuous TS SHAP Heatmap' if has_ebm else 'Continuous TS SHAP Heatmap (grouped)'
    ax2.set_title(heatmap_title, fontweight='bold')
    if n_display <= 40:
        ax2.set_yticks(range(n_display))
        ax2.set_yticklabels(ordered_labels, fontsize=7 if n_display > 25 else 9)
    else:
        step = max(1, n_display // 30)
        yticks = list(range(0, n_display, step))
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([ordered_labels[i] for i in yticks], fontsize=7)
    ax2.set_xticks(tick_idx); ax2.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    plt.colorbar(im, ax=ax2, label='SHAP Value')
    if not has_ebm and channel2feature:
        _draw_group_separators(ax2, group_bounds)
    
    # Plot 3: Categorical TS heatmap - SHAP values with centered colormap
    if shap_results.get('encoding_info') is not None and shap_results.get('cat_ts_shap_per_category') is not None:
        # Use per-category SHAP values if available
        ax3 = fig.add_subplot(gs[2 + row_offset, :])
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
        ax3.set_xlabel('Time'); ax3.set_ylabel('Category')
        ax3.set_title('Categorical TS SHAP Heatmap', fontweight='bold')
        
        if n_cats <= 30:
            ax3.set_yticks(range(n_cats)); ax3.set_yticklabels(cat_names[:n_cats], fontsize=8)
        else:
            step = max(1, n_cats // 20)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks); ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=8)
        
        ax3.set_xticks(tick_idx); ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
        plt.colorbar(im3, ax=ax3, label='SHAP Value')
        
        # Feature boundaries
        for feat, (start, end) in enc_info.get('feature_ranges', {}).items():
            if start > 0:
                ax3.axhline(y=start - 0.5, color='black', linewidth=1.5, linestyle='--')
    
    elif shap_results.get('encoding_info') is not None:
        # Fallback: Show raw data with SHAP importance overlay
        ax3 = fig.add_subplot(gs[2 + row_offset, :])
        cat_ts_data = shap_results['test_data']['ts_cat'][sample_idx, :, :n_steps]  # crop to eval_timestep
        enc_info = shap_results['encoding_info']
        
        cat_names = get_category_names_from_encoding_info(enc_info)
        
        n_cats = cat_ts_data.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f"cat_{len(cat_names)}")
        
        # Show activity data (0/1 so no centering needed)
        im3 = ax3.imshow(cat_ts_data, aspect='auto', cmap='YlGnBu', interpolation='nearest', vmin=0)
        ax3.set_xlabel('Time'); ax3.set_ylabel('Category')
        ax3.set_title('Categorical TS Activity (use compute_per_category_shap=True for SHAP values)', 
                      fontweight='bold', fontsize=11)
        
        if n_cats <= 30:
            ax3.set_yticks(range(n_cats)); ax3.set_yticklabels(cat_names[:n_cats], fontsize=8)
        else:
            step = max(1, n_cats // 20)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks); ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=8)
        
        ax3.set_xticks(tick_idx); ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
        plt.colorbar(im3, ax=ax3, label='Active')
        
        # Feature boundaries
        for feat, (start, end) in enc_info.get('feature_ranges', {}).items():
            if start > 0:
                ax3.axhline(y=start - 0.5, color='white', linewidth=2)
    
    # Plot 4: Channel importance — clinical-only when EBM present
    ax4 = fig.add_subplot(gs[3 + row_offset, :])
    ch_imp = np.abs(ts_shap).mean(axis=1)
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
    ax4.set_yticks(range(n_show)); ax4.set_yticklabels(names, fontsize=9)
    bar_title = f'Top {n_show} Clinical Channels' if has_ebm else f'Top {n_show} Channels'
    ax4.set_xlabel('Mean |SHAP|'); ax4.set_title(bar_title, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x'); ax4.invert_yaxis()
    if channel2feature and not has_ebm:
        used_groups = set()
        for i in sorted_idx:
            name = channel2feature.get(int(i), '')
            if name in _EBM_CHANNELS: used_groups.add('EBM')
            else: used_groups.add('Clinical')
        ax4.legend(handles=[Patch(facecolor=_GROUP_COLORS[g], label=g, alpha=0.7)
                            for g in ['Clinical', 'EBM'] if g in used_groups],
                   loc='lower right', fontsize=8)
    
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
        ax5.set_yticklabels(ylabels, fontsize=9)
        
        ax5.set_xlabel('SHAP Value'); ax5.set_title('Static Categorical', fontweight='bold')
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
        ax6.set_yticklabels(ylabels, fontsize=9)
        
        ax6.set_xlabel('SHAP Value'); ax6.set_title('Static Continuous', fontweight='bold')
        ax6.axvline(x=0, color='black', linewidth=0.8)
        ax6.grid(True, alpha=0.3, axis='x'); ax6.invert_yaxis()
    
    plt.tight_layout()
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
    if eval_timestep is not None and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_data = ts_data[..., :n_steps]

    # Time axis
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps - 1, n_ticks, dtype=int)

    # --- Detect trajectory length ---
    # Priority: (1) _data_present channel  → mask spans first..last measurement
    #           (2) elapsed_hours channel  → mask spans first..last non-zero elapsed
    #           (3) explicit trajectory_length (inference session) → mask spans 0..traj_len
    #               This fixes the zero-padded inference case where heuristics fail.
    #           (4) NaN-any fallback (unreliable for zero-padded data).
    trajectory_mask = None  # set by whichever branch succeeds first

    dp_idx = None
    eh_idx = None
    if channel2feature:
        for idx, name in channel2feature.items():
            if name == '_data_present':
                dp_idx = idx
            elif name == 'elapsed_hours':
                eh_idx = idx

    if dp_idx is not None:
        raw_mask = ts_data[dp_idx] > 0.5  # [seq_len]
        if not np.any(raw_mask):
            dp_idx = None  # all zero → fall through

    if dp_idx is not None:
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
                           ebm_importances: Optional[Dict] = None):
    """Summary visualizations across cohort."""

    ts_shap = shap_results['ts_shap']
    if ts_shap.ndim == 4:
        ts_shap = ts_shap[..., min(class_idx, ts_shap.shape[-1] - 1)]
    n_samples, n_channels, n_steps = ts_shap.shape

    # Crop time axis to eval_timestep — steps beyond have ~0 SHAP due to causal masking
    # and showing them extends the x-axis with meaningless zeros.
    if eval_timestep is None:
        eval_timestep = shap_results.get('eval_timestep')
    if eval_timestep is not None and 0 <= eval_timestep < n_steps:
        n_steps = eval_timestep + 1
        ts_shap = ts_shap[..., :n_steps]

    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps-1, n_ticks, dtype=int)

    # EBM two-view: compute budget and adjust layout
    budget = compute_ebm_vs_clinical_budget(ts_shap, channel2feature)
    has_ebm = budget is not None
    has_ebm_imp = has_ebm and ebm_importances is not None

    if has_ebm and has_ebm_imp:
        fig = plt.figure(figsize=(22, 25))
        gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3,
                              height_ratios=[0.7, 0.7, 1, 1, 1.2, 1])
        row_offset = 3
        # Row 0: EBM budget over time
        ax_budget = fig.add_subplot(gs[0, :])
        _draw_ebm_budget_temporal(ax_budget, budget, n_steps, tick_idx,
                                  [time_fmt[i] for i in tick_idx],
                                  title=f'SHAP Budget Over Time: EBM vs Clinical (Class {class_idx})')
        # Row 1: EBM importance lines
        ax_ebm_lines = fig.add_subplot(gs[1, :])
        _draw_ebm_importance_lines(ax_ebm_lines, ebm_importances, n_steps,
                                    tick_idx, [time_fmt[i] for i in tick_idx],
                                    title=f'Top EBM Features Over Time (Class {class_idx})')
        # Row 2: EBM importance heatmap
        ax_ebm_hm = fig.add_subplot(gs[2, :])
        _draw_ebm_importance_heatmap(ax_ebm_hm, ebm_importances, n_steps,
                                      tick_idx, [time_fmt[i] for i in tick_idx],
                                      title=f'EBM Feature Importance Over Time (Class {class_idx})')
    elif has_ebm:
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

    if has_ebm:
        # Two-view: separate Clinical and EBM lines
        clinical_ch = _get_clinical_only_channel_mask(channel2feature, n_channels)
        ebm_ch = [i for i, name in channel2feature.items() if name in _EBM_CHANNELS]

        clinical_imp = np.abs(ts_shap[:, clinical_ch, :]).mean(axis=(0, 1))
        ax1.plot(clinical_imp, linewidth=2, color=_GROUP_COLORS['Clinical'],
                 label='Clinical channels')
        ax1.fill_between(range(len(clinical_imp)), clinical_imp, alpha=0.2,
                         color=_GROUP_COLORS['Clinical'])

        if ebm_ch:
            ebm_imp = np.abs(ts_shap[:, ebm_ch, :]).mean(axis=(0, 1))
            ax1.plot(ebm_imp, linewidth=2, color=_GROUP_COLORS['EBM'],
                     label='EBM (_ebm_pred)', linestyle='--')
            ax1.fill_between(range(len(ebm_imp)), ebm_imp, alpha=0.2,
                             color=_GROUP_COLORS['EBM'])
    else:
        ts_imp = np.abs(ts_shap[:, display_ch, :]).mean(axis=(0, 1))
        ax1.plot(ts_imp, linewidth=2, color='#ff0051', label='Continuous TS')
        ax1.fill_between(range(len(ts_imp)), ts_imp, alpha=0.3, color='#ff0051')

    if shap_results['cat_ts_shap'] is not None:
        cat_ts = shap_results['cat_ts_shap']
        if cat_ts.ndim == 3:
            cat_ts = cat_ts[..., min(class_idx, cat_ts.shape[-1] - 1)]
        cat_ts = cat_ts[..., :n_steps]  # crop to eval_timestep
        cat_imp = np.abs(cat_ts).mean(axis=0)
        ax1.plot(cat_imp, linewidth=2, color='#00d4aa', label='Categorical TS', linestyle='--')
        ax1.fill_between(range(len(cat_imp)), cat_imp, alpha=0.2, color='#00d4aa')

    ax1.set_xlabel('Time'); ax1.set_ylabel('Mean |SHAP|')
    ax1.set_title(f'Feature Importance Over Time (Class {class_idx})', fontweight='bold')
    ax1.set_xticks(tick_idx); ax1.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top channels — clinical-only when EBM present
    ax2 = fig.add_subplot(gs[1 + row_offset, 0])
    ch_imp = np.abs(ts_shap).mean(axis=(0, 2))
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
    bar_title = f'Top {len(sorted_idx)} Clinical Channels' if has_ebm else f'Top {len(sorted_idx)} Channels'
    ax2.set_xlabel('Mean |SHAP|'); ax2.set_title(bar_title, fontweight='bold')
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
        cat_ts_mean = np.abs(cat_ts_shap).mean(axis=0)  # [n_cats, seq_len]
        
        enc_info = shap_results['encoding_info']
        cat_names = get_category_names_from_encoding_info(enc_info)
        n_cats = cat_ts_mean.shape[0]
        while len(cat_names) < n_cats:
            cat_names.append(f"cat_{len(cat_names)}")
        
        im3 = ax3.imshow(cat_ts_mean, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_xlabel('Time'); ax3.set_ylabel('Category')
        ax3.set_title('Categorical TS |SHAP| (Mean)', fontweight='bold')
        
        if n_cats <= 20:
            ax3.set_yticks(range(n_cats)); ax3.set_yticklabels(cat_names, fontsize=8)
        else:
            step = max(1, n_cats // 15)
            yticks = list(range(0, n_cats, step))
            ax3.set_yticks(yticks); ax3.set_yticklabels([cat_names[i] for i in yticks], fontsize=8)
        ax3.set_xticks(tick_idx); ax3.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
        plt.colorbar(im3, ax=ax3, label='Mean |SHAP|')
    
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()




def shap_analysis(data=None, model=None, model_name='13012025', compute_per_category_shap=True,
                  max_test_samples=90, visualize=True, specific_pids: List = None) -> Dict:
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
            print(f"Warning: PIDs not found in holdout set: {missing_pids}")
        specific_pids = [pid for pid in specific_pids if pid in all_holdout_pids]
        if not specific_pids:
            raise ValueError("None of the specified PIDs were found in holdout set")
        print(f"Analyzing {len(specific_pids)} specific PIDs: {specific_pids}")

    shap_results = calculate_shap_from_dataloaders(
        model=model,
        background_loader=data["mixed_dls"].train,
        test_loader=data["holdout_mixed_dls"].train,
        device=device,
        max_background_samples=600,
        max_test_samples=max_test_samples,
        encoding_info=data["encoding_info"],
        compute_per_category_shap=compute_per_category_shap,
        specific_pids=specific_pids,
        all_pids=all_holdout_pids
    )

    channel2feature, _ = create_channel_mapping(data)

    # Get holdout PIDs for individual plots (filtered if specific_pids provided)
    holdout_pids = get_holdout_pids(data, max_samples=max_test_samples, specific_pids=specific_pids)
    print(f"\nExtracted {len(holdout_pids)} holdout PIDs")
    print(f"First 5 PIDs: {holdout_pids[:5]}")
    
    # Get static categorical names from classes (includes _na columns)
    static_cat_names = get_static_cat_names_from_classes(data["classes"])
    print(f"Static categorical features: {static_cat_names}")
    
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
            ebm_importances=ebm_importances,
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
            ebm_importances=ebm_importances,
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
    
    def get_available_timeframes(self) -> List[str]:
        return list(self.timeframe_results.keys())
    
    def get_result(self, timeframe: str) -> Optional[TimeframeSHAPResult]:
        return self.timeframe_results.get(timeframe)


# ============================================================================
# MODEL WRAPPER
# ============================================================================

# NOTE: ModelWrapperWithRawCatTS is defined above (used by both
# calculate_shap_from_dataloaders and TemporalSHAPAnalyzer).


def embed_categorical_features(model, x_cat):
    """Pre-embed static categorical features."""
    if x_cat is None or x_cat.shape[1] == 0:
        return None
    with torch.no_grad():
        x_cat_emb = [model.embeds[i](x_cat[:, i]).unsqueeze(1) for i in range(x_cat.shape[1])]
        x_cat_emb = torch.cat(x_cat_emb, 1)
    x_cat_emb.requires_grad = True
    return x_cat_emb


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class TemporalSHAPAnalyzer:
    """Analyzes SHAP values across timeframes using Option A (re-compute per timeframe)."""
    
    def __init__(self, model: nn.Module, data: Dict, background_loader,
                 device: str = 'cuda', max_background_samples: int = 200, class_idx: int = 1):
        self.model = model
        self.data = data
        self.background_loader = background_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_background_samples = max_background_samples
        self.class_idx = class_idx
        
        self.encoding_info = data.get("encoding_info", {})
        self.channel2feature, self.feature2channel = self._create_channel_mapping()
        self.static_cat_names = list(data.get("classes", {}).keys())
        self.static_cont_names = data.get("num_cols", [])
        
        self._bg_data = None
        self.model.eval()
        self.model = self.model.to(device)
        self.has_cat_ts = model.n_ts_cat > 0
        
        print(f"TemporalSHAPAnalyzer: {len(self.channel2feature)} channels, "
              f"cat_ts={self.has_cat_ts}, bg_samples={max_background_samples}")
    
    def _create_channel_mapping(self):
        features = self.data["trainval"].complete.sort_values(['PID', 'FEATURE'])['FEATURE'].drop_duplicates().tolist()
        return {i: f for i, f in enumerate(features)}, {f: i for i, f in enumerate(features)}
    
    def _extract_background_data(self):
        if self._bg_data is not None:
            return self._bg_data
        
        print("Extracting background data...")
        all_ts, all_ts_cat, all_cat, all_cont = [], [], [], []
        n = 0
        for batch in self.background_loader:
            if n >= self.max_background_samples:
                break
            inputs, _ = batch
            x_ts, x_tab, x_ts_cat = inputs[0], inputs[1], inputs[2]
            all_ts.append(x_ts.cpu())
            all_ts_cat.append(x_ts_cat.cpu())
            all_cat.append(x_tab[0].cpu())
            all_cont.append(x_tab[1].cpu())
            n += x_ts.shape[0]
        
        self._bg_data = {
            'ts': torch.cat(all_ts)[:self.max_background_samples].to(self.device),
            'ts_cat': torch.cat(all_ts_cat)[:self.max_background_samples].to(self.device),
            'cat': torch.cat(all_cat)[:self.max_background_samples].to(self.device),
            'cont': torch.cat(all_cont)[:self.max_background_samples].to(self.device)
        }
        return self._bg_data
    
    def _censor_data(self, ts, ts_cat, censor_step):
        if censor_step is None:
            return ts, ts_cat
        ts_c, ts_cat_c = ts.clone(), ts_cat.clone()
        if censor_step < ts.shape[2] - 1:
            ts_c[:, :, censor_step+1:] = 0.0
            ts_cat_c[:, :, censor_step+1:] = 0
        return ts_c, ts_cat_c
    
    def _compute_shap_for_sample(self, sample_ts, sample_ts_cat, sample_cat, sample_cont, censor_step=None):
        bg = self._extract_background_data()
        bg_ts_c, bg_ts_cat_c = self._censor_data(bg['ts'], bg['ts_cat'], censor_step)
        
        if sample_ts.dim() == 2:
            sample_ts = sample_ts.unsqueeze(0)
            sample_ts_cat = sample_ts_cat.unsqueeze(0)
            sample_cat = sample_cat.unsqueeze(0)
            sample_cont = sample_cont.unsqueeze(0)
        
        sample_ts_c, sample_ts_cat_c = self._censor_data(sample_ts, sample_ts_cat, censor_step)
        
        bg_cat_emb = embed_categorical_features(self.model, bg['cat']) if bg['cat'].shape[1] > 0 else None
        sample_cat_emb = embed_categorical_features(self.model, sample_cat) if sample_cat.shape[1] > 0 else None
        
        eval_ts = censor_step if censor_step is not None else -1
        wrapped = ModelWrapperWithRawCatTS(self.model, self.has_cat_ts, eval_timestep=eval_ts)
        
        bg_inputs = [bg_ts_c, bg_ts_cat_c.float().requires_grad_(True)]
        sample_inputs = [sample_ts_c, sample_ts_cat_c.float().requires_grad_(True)]
        
        if bg_cat_emb is not None:
            bg_inputs.append(bg_cat_emb)
            sample_inputs.append(sample_cat_emb)
        if bg['cont'].shape[1] > 0:
            bg_inputs.append(bg['cont'])
            sample_inputs.append(sample_cont)
        
        explainer = shap.GradientExplainer(wrapped, bg_inputs)
        shap_values = explainer.shap_values(sample_inputs)
        
        if isinstance(shap_values, list) and shap_values and isinstance(shap_values[0], list):
            shap_values = shap_values[0]
        
        idx = 0
        ts_shap = shap_values[idx][0]
        idx += 1
        
        cat_ts_shap_per_cat, cat_ts_shap = None, None
        if self.has_cat_ts:
            cat_ts_shap_per_cat = shap_values[idx][0]
            cat_ts_shap = np.abs(cat_ts_shap_per_cat).mean(axis=0)
            idx += 1
        
        cat_shap = shap_values[idx][0].mean(axis=1) if bg_cat_emb is not None else None
        if bg_cat_emb is not None:
            idx += 1
        
        cont_shap = shap_values[idx][0] if bg['cont'].shape[1] > 0 else None
        
        return {'ts_shap': ts_shap, 'cat_ts_shap': cat_ts_shap, 
                'cat_ts_shap_per_category': cat_ts_shap_per_cat, 
                'cat_shap': cat_shap, 'cont_shap': cont_shap}
    
    def get_holdout_pids(self, max_samples=None):
        pids = self.data["holdout"].tab_df['PID'].tolist()
        return pids[:max_samples] if max_samples else pids
    
    def get_sample_data(self, test_loader, sample_idx):
        curr = 0
        for batch in test_loader:
            inputs, targets = batch
            x_ts, x_tab, x_ts_cat = inputs[0], inputs[1], inputs[2]
            bs = x_ts.shape[0]
            if curr <= sample_idx < curr + bs:
                i = sample_idx - curr
                return (x_ts[i].to(self.device), x_ts_cat[i].to(self.device),
                        x_tab[0][i].to(self.device), x_tab[1][i].to(self.device), targets[i])
            curr += bs
        raise IndexError(f"Sample {sample_idx} out of range")
    
    def analyze_patient(self, test_loader, pid=None, sample_idx=None, holdout_pids=None,
                       timeframes=None, verbose=True) -> TemporalSHAPResults:
        """Run temporal SHAP analysis for a patient."""
        if pid is not None:
            if holdout_pids is None:
                raise ValueError("holdout_pids required with pid")
            sample_idx = holdout_pids.index(pid)
            if verbose: print(f"PID {pid} -> sample {sample_idx}")
        elif sample_idx is None:
            sample_idx = 0
        
        display_pid = pid or (holdout_pids[sample_idx] if holdout_pids else sample_idx)
        sample_ts, sample_ts_cat, sample_cat, sample_cont, _ = self.get_sample_data(test_loader, sample_idx)
        
        ts_np = sample_ts.cpu().numpy()
        actual_steps = get_actual_data_length(ts_np)
        actual_min = step_to_time(actual_steps - 1) if actual_steps > 0 else 0
        actual_hours = (actual_min or 0) / 60
        
        if verbose:
            print(f"Patient {display_pid}: {actual_steps} steps ({actual_hours:.1f}h)")
        
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
                    print(f"  Skip {tf} (need {tf_h}h, have {actual_hours:.1f}h)")
        
        # If we skipped some timeframes, add 'max' which uses actual data length
        # This replaces 'full' behavior with a named timeframe showing actual hours
        if skipped_any and 'full' in valid_tfs:
            # Replace 'full' with 'max' to make it clearer this is the max available
            valid_tfs = [tf if tf != 'full' else f'max({actual_hours:.1f}h)' for tf in valid_tfs]
            # Add max to DEFAULT_TIMEFRAMES temporarily for this analysis
            DEFAULT_TIMEFRAMES[f'max({actual_hours:.1f}h)'] = None  # None means full/no censoring
        
        if verbose: print(f"Analyzing: {valid_tfs}")
        
        results = {}
        t0 = time.time()
        
        for i, tf in enumerate(valid_tfs):
            tf_h = DEFAULT_TIMEFRAMES.get(tf)
            censor = None if tf_h is None else time_to_step(tf_h, 'h')
            if verbose: print(f"  [{i+1}/{len(valid_tfs)}] {tf}...", end=" ", flush=True)
            
            t1 = time.time()
            shap_res = self._compute_shap_for_sample(sample_ts, sample_ts_cat, sample_cat, sample_cont, censor)
            if verbose: print(f"done ({time.time()-t1:.1f}s)")
            
            ts_shap = shap_res['ts_shap']
            if ts_shap.ndim == 3:
                ts_shap = ts_shap[..., min(self.class_idx, ts_shap.shape[-1] - 1)]
            
            results[tf] = TimeframeSHAPResult(
                timeframe_name=tf, timeframe_hours=tf_h, censor_step=censor,
                actual_data_steps=actual_steps, ts_shap=ts_shap,
                cat_ts_shap=shap_res['cat_ts_shap'],
                cat_ts_shap_per_category=shap_res['cat_ts_shap_per_category'],
                cat_shap=shap_res['cat_shap'], cont_shap=shap_res['cont_shap'],
                ts_data=ts_np, cat_ts_data=sample_ts_cat.cpu().numpy(),
                cat_data=sample_cat.cpu().numpy(), cont_data=sample_cont.cpu().numpy(),
                ts_channel_importance=np.abs(ts_shap).mean(axis=1),
                ts_temporal_importance=np.abs(ts_shap).mean(axis=0)
            )
        
        if verbose: print(f"Total: {time.time()-t0:.1f}s")
        
        out = TemporalSHAPResults(
            pid=display_pid, sample_idx=sample_idx,
            actual_data_length_steps=actual_steps, actual_data_length_hours=actual_hours,
            timeframe_results=results, channel2feature=self.channel2feature,
            static_cat_names=self.static_cat_names, static_cont_names=self.static_cont_names,
            encoding_info=self.encoding_info
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
        tick_idx = np.linspace(0, seq_len-1, min(8, seq_len), dtype=int)
        tick_labels = [time_to_hours_str(step_to_time(i)) for i in tick_idx]
        
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

            # Row 1: Temporal importance
            ax1 = fig.add_subplot(gs[0, col])
            ax1.plot(r.ts_temporal_importance, lw=2, color='#ff0051')
            ax1.fill_between(range(len(r.ts_temporal_importance)), r.ts_temporal_importance, alpha=0.3, color='#ff0051')
            if r.censor_step: ax1.axvline(r.censor_step, color='black', ls='--', lw=2)
            ax1.axvline(results.actual_data_length_steps, color='gray', ls=':', lw=1.5, alpha=0.7)
            ax1.set_xlim(0, seq_len); ax1.set_ylim(0, temp_max*1.1)
            ax1.set_xticks(tick_idx); ax1.set_xticklabels(tick_labels, rotation=45, fontsize=8)
            ax1.set_title(f'{tf} {suffix}{ebm_annotation}', fontweight='bold'); ax1.grid(True, alpha=0.3)

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
            ax3.set_yticks(range(len(top_idx))); ax3.set_yticklabels(names, fontsize=8)
            ax3.set_xticks(tick_idx); ax3.set_xticklabels(tick_labels, rotation=45, fontsize=8)
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
        
        fig.suptitle(f'Temporal SHAP - PID: {results.pid} ({results.actual_data_length_hours:.1f}h data)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"Saved: {save_path}")
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
            print("No stability metrics"); return None
        
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
        
        fig.suptitle(f'Comprehensive SHAP Stability Analysis - PID: {results.pid}\n'
                    f'Data: {results.actual_data_length_hours:.1f}h | Timeframes: {", ".join(tfs)}',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
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
            print("Need 2+ timeframes"); return None
        
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
        
        fig.suptitle(f'Feature Importance Correlation Analysis - PID: {results.pid}\n'
                    f'Reference: {reference}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
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
        ax.set_title(f'Feature Trajectory - PID: {results.pid}', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left'); ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"Saved: {save_path}")
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


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_temporal_shap_analysis(data, model, pid=None, sample_idx=None, timeframes=None,
                               max_background_samples=200, save_dir='reports/shap', verbose=False):
    """
    Run complete temporal SHAP analysis with all visualizations.

    Args:
        data: Data dict from prepare_data_and_dls()
        model: Trained nn.Module
        pid: Patient ID to analyze
        sample_idx: Alternative to pid
        timeframes: List of timeframe names (default: all)
        save_dir: Output directory
        verbose: Print progress

    Returns:
        TemporalSHAPResults
    """
    os.makedirs(save_dir, exist_ok=True)

    analyzer = TemporalSHAPAnalyzer(
        model, data, data["mixed_dls"].train, 'cuda' if torch.cuda.is_available() else 'cpu', max_background_samples
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