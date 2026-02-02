#behavior.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from scipy import stats
import shap
from collections import OrderedDict
import time
import os

from astra.utils import logger, cfg
from astra.models.hybrid.training import get_backbone, Learner, patch_learner_get_preds
from astra.data.dataloader import prepare_data_and_dls
from astra.evaluation.utils import prepare_learner

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

def step_to_time(step):
    """Convert time step to actual time in minutes."""
    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]
    bins_cum = [0]
    for interval in intervals[:-1]:
        duration_min = (interval['end_h'] - interval['start_h']) * 60
        bins = duration_min // interval['bin_min']
        bins_cum.append(bins_cum[-1] + bins)
    
    for i in range(len(bins_cum) - 1):
        if bins_cum[i] <= step < bins_cum[i+1]:
            interval = intervals[i]
            step_offset = step - bins_cum[i]
            start_min = interval['start_h'] * 60
            return start_min + (step_offset + 1) * interval['bin_min']
    return None


def time_to_hours(minutes):
    if minutes is None:
        return "N/A"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    else:
        return f"{hours/24:.1f}d"


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
    def __init__(self, model, has_cat_ts=False):
        super().__init__()
        self.model = model
        self.has_cat_ts = has_cat_ts
        
    def forward(self, x_ts, x_ts_cat_embedded=None, x_cat_embedded=None, x_cont=None):
        device = x_ts.device
        mask = torch.isnan(x_ts)
        if mask.any():
            x_ts = x_ts.clone()
            x_ts[mask] = 0
        
        x = self.model.W_P(x_ts).transpose(1, 2)
        
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
        
        x += self.model.pos_enc
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)
        x = self.model.transformer(x, key_padding_mask=None)
        x = self.model.head(x)
        return x


class ModelWrapperWithRawCatTS(nn.Module):
    """
    Wrapper that takes RAW multi-hot categorical TS (not pre-embedded).
    This allows SHAP to compute per-category attributions.
    """
    def __init__(self, model, has_cat_ts=False):
        super().__init__()
        self.model = model
        self.has_cat_ts = has_cat_ts
        
    def forward(self, x_ts, x_ts_cat_raw=None, x_cat_embedded=None, x_cont=None):
        """
        Args:
            x_ts: [bs, c_in, seq_len] - continuous time series
            x_ts_cat_raw: [bs, n_categories, seq_len] - raw multi-hot categorical TS
            x_cat_embedded: [bs, n_cat, d_model] - pre-embedded static categorical
            x_cont: [bs, n_cont] - static continuous
        """
        device = x_ts.device
        mask = torch.isnan(x_ts)
        if mask.any():
            x_ts = x_ts.clone()
            x_ts[mask] = 0
        
        # Continuous TS encoding
        x = self.model.W_P(x_ts).transpose(1, 2)  # [bs, seq_len, d_model]
        
        # Embed categorical TS from raw multi-hot (this is differentiable!)
        if self.has_cat_ts and x_ts_cat_raw is not None and self.model.n_ts_cat > 0:
            # x_ts_cat_raw: [bs, n_categories, seq_len]
            # Transpose to [bs, seq_len, n_categories]
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
        
        x += self.model.pos_enc
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)
        x = self.model.transformer(x, key_padding_mask=None)
        x = self.model.head(x)
        return x


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


def get_holdout_pids(data, max_samples=None):
    """
    Extract PIDs from holdout dataset in the order they appear in the dataloader.
    
    Args:
        data: Data dict containing 'holdout' TSDS object
        max_samples: Maximum number of samples (should match what was used in SHAP calculation)
    
    Returns:
        List of PIDs in dataloader order
    """
    # Get PIDs from holdout tab_df (which is used by the dataloader)
    holdout_pids = data["holdout"].tab_df['PID'].tolist()
    
    if max_samples is not None and len(holdout_pids) > max_samples:
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
                                     compute_per_category_shap=True):
    """
    Calculate SHAP values for all model inputs.
    
    Args:
        compute_per_category_shap: If True, compute SHAP on raw multi-hot categorical TS
                                   to get per-category attributions. If False, compute on
                                   embedded representation (faster but less granular).
    """
    print("Extracting background data...")
    bg_ts, bg_ts_cat, bg_cat, bg_cont, bg_y = extract_data_from_dataloader(
        background_loader, max_samples=max_background_samples, device=device)
    
    print(f"  Background samples: {bg_ts.shape[0]}")
    print(f"    Continuous TS: {bg_ts.shape}")
    print(f"    Categorical TS: {bg_ts_cat.shape}")
    print(f"    Static categorical: {bg_cat.shape}")
    print(f"    Static continuous: {bg_cont.shape}")
    
    print("Extracting test data...")
    test_ts, test_ts_cat, test_cat, test_cont, test_y = extract_data_from_dataloader(
        test_loader, max_samples=max_test_samples, device=device)
    print(f"  Test samples: {test_ts.shape[0]}")
    
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
    
    if bg_cat_emb is not None:
        print(f"  Static categorical embedded: {bg_cat_emb.shape}")
    
    print(f"\ncompute_per_category_shap: {compute_per_category_shap}")
    
    if compute_per_category_shap and has_cat_ts:
        # Use wrapper that takes RAW categorical TS for per-category SHAP
        print("  Using ModelWrapperWithRawCatTS for per-category SHAP values")
        wrapped_model = ModelWrapperWithRawCatTS(model, has_cat_ts=has_cat_ts)
        
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
        wrapped_model = ModelWrapperWithEmbeddings(model, has_cat_ts=has_cat_ts)
        
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
    
    if isinstance(shap_values, list) and len(shap_values) > 0:
        if isinstance(shap_values[0], list):
            shap_values = shap_values[0]
    
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
                               class_idx: int = 1, save_path: str = None):
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
    
    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1, 1])
    
    ts_shap = shap_results['ts_shap'][sample_idx]
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[:, :, class_idx]
    n_channels, n_steps = ts_shap.shape
    
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps-1, n_ticks, dtype=int)
    
    # Title with PID if available
    title_suffix = f" (PID: {display_pid})" if display_pid is not None else f" (Sample {sample_idx})"
    
    # Plot 1: TS importance over time
    ax1 = fig.add_subplot(gs[0, :])
    ts_shap_avg = np.abs(ts_shap).mean(axis=0)
    ax1.plot(ts_shap_avg, linewidth=2, color='#ff0051', label='Continuous TS')
    ax1.fill_between(range(len(ts_shap_avg)), ts_shap_avg, alpha=0.3, color='#ff0051')
    
    if shap_results['cat_ts_shap'] is not None:
        cat_ts = shap_results['cat_ts_shap'][sample_idx]
        if cat_ts.ndim == 2:
            cat_ts = cat_ts[:, class_idx]
        ax1.plot(cat_ts, linewidth=2, color='#00d4aa', label='Categorical TS', linestyle='--')
        ax1.fill_between(range(len(cat_ts)), cat_ts, alpha=0.2, color='#00d4aa')
    
    ax1.set_xlabel('Time'); ax1.set_ylabel('|SHAP Value|')
    ax1.set_title(f'TS SHAP Over Time{title_suffix}, Class {class_idx}', fontweight='bold')
    ax1.set_xticks(tick_idx); ax1.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Plot 2: Continuous TS heatmap - with centered colormap
    ax2 = fig.add_subplot(gs[1, :])
    norm2 = get_centered_norm(ts_shap, center=0.0)
    im = ax2.imshow(ts_shap, aspect='auto', cmap='RdBu_r', interpolation='nearest', norm=norm2)
    ax2.set_xlabel('Time'); ax2.set_ylabel('Channel')
    ax2.set_title('Continuous TS SHAP Heatmap', fontweight='bold')
    if channel2feature:
        labels = [channel2feature.get(i, f'Ch{i}') for i in range(n_channels)]
        if n_channels > 20:
            step = n_channels // 20
            yticks = list(range(0, n_channels, step))
            ax2.set_yticks(yticks); ax2.set_yticklabels([labels[i] for i in yticks], fontsize=8)
        else:
            ax2.set_yticks(range(n_channels)); ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xticks(tick_idx); ax2.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    plt.colorbar(im, ax=ax2, label='SHAP Value')
    
    # Plot 3: Categorical TS heatmap - SHAP values with centered colormap
    if shap_results.get('encoding_info') is not None and shap_results.get('cat_ts_shap_per_category') is not None:
        # Use per-category SHAP values if available
        ax3 = fig.add_subplot(gs[2, :])
        cat_ts_shap_data = shap_results['cat_ts_shap_per_category'][sample_idx]  # [n_cats, seq_len]
        if cat_ts_shap_data.ndim == 3:
            cat_ts_shap_data = cat_ts_shap_data[:, :, class_idx]
        
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
        ax3 = fig.add_subplot(gs[2, :])
        cat_ts_data = shap_results['test_data']['ts_cat'][sample_idx]  # [n_cats, seq_len]
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
    
    # Plot 4: Channel importance
    ax4 = fig.add_subplot(gs[3, :])
    ch_imp = np.abs(ts_shap).mean(axis=1)
    sorted_idx = np.argsort(ch_imp)[::-1]
    n_show = min(20, len(ch_imp))
    if channel2feature:
        names = [channel2feature.get(i, f'Ch{i}') for i in sorted_idx[:n_show]]
    else:
        names = [f'Channel {i}' for i in sorted_idx[:n_show]]
    ax4.barh(range(n_show), ch_imp[sorted_idx[:n_show]], color='#008bfb', alpha=0.7)
    ax4.set_yticks(range(n_show)); ax4.set_yticklabels(names, fontsize=9)
    ax4.set_xlabel('Mean |SHAP|'); ax4.set_title(f'Top {n_show} Channels', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x'); ax4.invert_yaxis()
    
    # Plot 5: Static categorical
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax5 = fig.add_subplot(gs[4, 0])
        cat_shap = shap_results['cat_shap'][sample_idx]
        cat_data = shap_results['test_data']['cat'][sample_idx]
        if cat_shap.ndim == 2:
            cat_shap = cat_shap[:, class_idx]
        
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
        ax6 = fig.add_subplot(gs[4, 1])
        cont_shap = shap_results['cont_shap'][sample_idx]
        cont_data = shap_results['test_data']['cont'][sample_idx]
        if cont_shap.ndim == 2:
            cont_shap = cont_shap[:, class_idx]
        
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'sample_idx': sample_idx, 'pid': display_pid}


def visualize_shap_summary(shap_results: Dict, channel2feature: Dict[int, str] = None,
                           feature_names_cat: List[str] = None,
                           feature_names_cont: List[str] = None,
                           max_display: int = 20, class_idx: int = 1, save_path: str = None):
    """Summary visualizations across cohort."""
    
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1.2, 1])
    
    ts_shap = shap_results['ts_shap']
    if ts_shap.ndim == 4:
        ts_shap = ts_shap[:, :, :, class_idx]
    n_samples, n_channels, n_steps = ts_shap.shape
    
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_fmt = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps-1, n_ticks, dtype=int)
    
    # Plot 1: TS importance over time
    ax1 = fig.add_subplot(gs[0, :])
    ts_imp = np.abs(ts_shap).mean(axis=(0, 1))
    ax1.plot(ts_imp, linewidth=2, color='#ff0051', label='Continuous TS')
    ax1.fill_between(range(len(ts_imp)), ts_imp, alpha=0.3, color='#ff0051')
    
    if shap_results['cat_ts_shap'] is not None:
        cat_ts = shap_results['cat_ts_shap']
        if cat_ts.ndim == 3:
            cat_ts = cat_ts[:, :, class_idx]
        cat_imp = np.abs(cat_ts).mean(axis=0)
        ax1.plot(cat_imp, linewidth=2, color='#00d4aa', label='Categorical TS', linestyle='--')
        ax1.fill_between(range(len(cat_imp)), cat_imp, alpha=0.2, color='#00d4aa')
    
    ax1.set_xlabel('Time'); ax1.set_ylabel('Mean |SHAP|')
    ax1.set_title(f'Feature Importance Over Time (Class {class_idx})', fontweight='bold')
    ax1.set_xticks(tick_idx); ax1.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top channels
    ax2 = fig.add_subplot(gs[1, 0])
    ch_imp = np.abs(ts_shap).mean(axis=(0, 2))
    sorted_idx = np.argsort(ch_imp)[::-1][:max_display]
    if channel2feature:
        names = [channel2feature.get(int(i), f'Ch{i}') for i in sorted_idx]
    else:
        names = [f'Channel {i}' for i in sorted_idx]
    ax2.barh(range(len(sorted_idx)), ch_imp[sorted_idx], color='#008bfb', alpha=0.7)
    ax2.set_yticks(range(len(sorted_idx))); ax2.set_yticklabels(names, fontsize=10)
    ax2.set_xlabel('Mean |SHAP|'); ax2.set_title(f'Top {len(sorted_idx)} Channels', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x'); ax2.invert_yaxis()
    
    # Plot 3: Categorical TS SHAP heatmap (mean across cohort)
    if shap_results.get('encoding_info') is not None and shap_results.get('cat_ts_shap_per_category') is not None:
        ax3 = fig.add_subplot(gs[1, 1])
        cat_ts_shap = shap_results['cat_ts_shap_per_category']  # [n_samples, n_cats, seq_len]
        if cat_ts_shap.ndim == 4:
            cat_ts_shap = cat_ts_shap[:, :, :, class_idx]
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
        ax3 = fig.add_subplot(gs[1, 1])
        cat_ts_data = shap_results['test_data']['ts_cat']
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
    
    # Plot 4: Continuous TS heatmap - with centered colormap
    ax4 = fig.add_subplot(gs[2, :])
    ts_mean = np.abs(ts_shap).mean(axis=0)
    # For absolute values, use sequential colormap (no centering needed)
    im = ax4.imshow(ts_mean, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0)
    ax4.set_xlabel('Time'); ax4.set_ylabel('Channel')
    ax4.set_title('Continuous TS |SHAP| Heatmap (Mean)', fontweight='bold')
    if channel2feature:
        labels = [channel2feature.get(i, f'Ch{i}') for i in range(n_channels)]
        if n_channels > 25:
            step = max(1, n_channels // 25)
            yticks = list(range(0, n_channels, step))
            ax4.set_yticks(yticks); ax4.set_yticklabels([labels[i] for i in yticks], fontsize=8)
        else:
            ax4.set_yticks(range(n_channels)); ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xticks(tick_idx); ax4.set_xticklabels([time_fmt[i] for i in tick_idx], rotation=45)
    plt.colorbar(im, ax=ax4, label='Mean |SHAP|')
    
    # Plot 5: Static categorical
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax5 = fig.add_subplot(gs[3, 0])
        cat_shap = shap_results['cat_shap']
        if cat_shap.ndim == 3:
            cat_shap = cat_shap[:, :, class_idx]
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
        ax6 = fig.add_subplot(gs[3, 1])
        cont_shap = shap_results['cont_shap']
        if cont_shap.ndim == 3:
            cont_shap = cont_shap[:, :, class_idx]
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




def shap_analysis(data=None, learn=None, model_name='13012025', compute_per_category_shap=True,
                  max_test_samples=90, visualize=True) -> Dict:
    """
    Run full SHAP analysis.
    
    Args:
        data: Prepared data dict
        learn: Trained learner
        model_name: Model checkpoint name
        compute_per_category_shap: If True, compute SHAP on raw multi-hot categorical TS
                                   to get per-category attributions (shows which specific
                                   medications/procedures matter). If False, faster but
                                   only shows aggregate categorical TS importance.
        max_test_samples: Maximum number of test samples for SHAP calculation
    
    Returns:
        dict with 'shap_results', 'holdout_pids', 'channel2feature', 'static_cat_names'
    """
    if data is None:
        data = prepare_data_and_dls()
    if learn is None:
        learn = prepare_learner(data, model_name)
    
    shap_results = calculate_shap_from_dataloaders(
        model=learn.model,
        background_loader=data["mixed_dls"].train,
        test_loader=data["holdout_mixed_dls"].train,
        device='cuda',
        max_background_samples=600,
        max_test_samples=max_test_samples,
        encoding_info=data["encoding_info"],
        compute_per_category_shap=compute_per_category_shap
    )
    
    channel2feature, _ = create_channel_mapping(data)
    
    # Get holdout PIDs for individual plots
    holdout_pids = get_holdout_pids(data, max_samples=max_test_samples)
    print(f"\nExtracted {len(holdout_pids)} holdout PIDs")
    print(f"First 5 PIDs: {holdout_pids[:5]}")
    
    # Get static categorical names from classes (includes _na columns)
    static_cat_names = get_static_cat_names_from_classes(data["classes"])
    print(f"Static categorical features: {static_cat_names}")
    
    # Debug output
    debug_shap_data(shap_results, 
                    feature_names_cat=static_cat_names,
                    feature_names_cont=cfg["dataset"]["num_cols"])
    
    if visualize is True:
        visualize_shap_summary(
            shap_results, channel2feature=channel2feature,
            feature_names_cat=static_cat_names,
            feature_names_cont=cfg["dataset"]["num_cols"],
            class_idx=1, max_display=20,
            save_path='reports/shap_summary_cohort.png'
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
            save_path='reports/shap_individual_sample_0.png'
        )
    
    # Return comprehensive results for further analysis
    return {
        'shap_results': shap_results,
        'holdout_pids': holdout_pids,
        'channel2feature': channel2feature,
        'static_cat_names': static_cat_names,
        'data': data
    }


# ============================================================================
# TIMEFRAMES
# ============================================================================

# Default timeframes (hours)
DEFAULT_TIMEFRAMES = OrderedDict([
    ('1H', 1), ('6H', 6), ('12H', 12), ('1D', 24), ('3D', 72),
    ('7D', 168), ('14D', 336), ('30D', 720), ('full', None)
])

# ============================================================================
# TIME UTILITIES
# ============================================================================

def step_to_time(step: int) -> Optional[float]:
    """Convert time step to minutes."""
    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]
    bins_cum = [0]
    for interval in intervals[:-1]:
        duration_min = (interval['end_h'] - interval['start_h']) * 60
        bins_cum.append(bins_cum[-1] + duration_min // interval['bin_min'])
    
    for i in range(len(bins_cum) - 1):
        if bins_cum[i] <= step < bins_cum[i+1]:
            step_offset = step - bins_cum[i]
            start_min = intervals[i]['start_h'] * 60
            return start_min + (step_offset + 1) * intervals[i]['bin_min']
    
    if step >= bins_cum[-1]:
        step_offset = step - bins_cum[-1]
        return intervals[-1]['start_h'] * 60 + (step_offset + 1) * intervals[-1]['bin_min']
    return None


def time_to_step(time_hours: float) -> int:
    """Convert hours to step index."""
    time_min = time_hours * 60
    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]
    bins_cum = [0]
    for interval in intervals[:-1]:
        duration_min = (interval['end_h'] - interval['start_h']) * 60
        bins_cum.append(bins_cum[-1] + duration_min // interval['bin_min'])
    
    for i, interval in enumerate(intervals):
        start_min = interval['start_h'] * 60
        end_min = interval['end_h'] * 60 if interval['end_h'] else float('inf')
        if start_min <= time_min < end_min:
            return bins_cum[i] + int((time_min - start_min) / interval['bin_min'])
    return bins_cum[-1]


def time_to_hours_str(minutes: Optional[float]) -> str:
    if minutes is None: return "N/A"
    hours = minutes / 60
    return f"{hours:.1f}h" if hours < 24 else f"{hours/24:.1f}d"


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

class ModelWrapperWithRawCatTS(nn.Module):
    """Wrapper for SHAP with raw multi-hot categorical TS."""
    def __init__(self, model, has_cat_ts=False):
        super().__init__()
        self.model = model
        self.has_cat_ts = has_cat_ts
        
    def forward(self, x_ts, x_ts_cat_raw=None, x_cat_embedded=None, x_cont=None):
        mask = torch.isnan(x_ts)
        if mask.any():
            x_ts = x_ts.clone()
            x_ts[mask] = 0
        
        x = self.model.W_P(x_ts).transpose(1, 2)
        
        if self.has_cat_ts and x_ts_cat_raw is not None and self.model.n_ts_cat > 0:
            x_ts_cat = x_ts_cat_raw.float().transpose(1, 2)
            x_ts_cat_embedded_list = []
            dim_offset = 0
            for embed_layer, (feat_name, n_classes) in zip(
                self.model.ts_cat_embeds, self.model.ts_cat_dims.items()
            ):
                feat_multi_hot = x_ts_cat[:, :, dim_offset:dim_offset + n_classes]
                x_ts_cat_embedded_list.append(embed_layer(feat_multi_hot))
                dim_offset += n_classes
            
            if self.model.cat_ts_combine == 'add':
                x = x + torch.stack(x_ts_cat_embedded_list, dim=0).sum(dim=0)
            else:
                x = torch.cat([x, torch.cat(x_ts_cat_embedded_list, dim=-1)], dim=-1)
        
        if x_cat_embedded is not None and x_cat_embedded.shape[1] > 0:
            x = torch.cat([x, x_cat_embedded], 1)
        
        if x_cont is not None and x_cont.shape[1] > 0:
            x_cont_emb = self.model.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_emb], 1)
        
        x += self.model.pos_enc
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)
        x = self.model.transformer(x, key_padding_mask=None)
        return self.model.head(x)


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
        self.device = device
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
        
        wrapped = ModelWrapperWithRawCatTS(self.model, self.has_cat_ts)
        
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
            censor = None if tf_h is None else time_to_step(tf_h)
            if verbose: print(f"  [{i+1}/{len(valid_tfs)}] {tf}...", end=" ", flush=True)
            
            t1 = time.time()
            shap_res = self._compute_shap_for_sample(sample_ts, sample_ts_cat, sample_cat, sample_cont, censor)
            if verbose: print(f"done ({time.time()-t1:.1f}s)")
            
            ts_shap = shap_res['ts_shap']
            if ts_shap.ndim == 3:
                ts_shap = ts_shap[:, :, self.class_idx]
            
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
            
            # Row 1: Temporal importance
            ax1 = fig.add_subplot(gs[0, col])
            ax1.plot(r.ts_temporal_importance, lw=2, color='#ff0051')
            ax1.fill_between(range(len(r.ts_temporal_importance)), r.ts_temporal_importance, alpha=0.3, color='#ff0051')
            if r.censor_step: ax1.axvline(r.censor_step, color='black', ls='--', lw=2)
            ax1.axvline(results.actual_data_length_steps, color='gray', ls=':', lw=1.5, alpha=0.7)
            ax1.set_xlim(0, seq_len); ax1.set_ylim(0, temp_max*1.1)
            ax1.set_xticks(tick_idx); ax1.set_xticklabels(tick_labels, rotation=45, fontsize=8)
            ax1.set_title(f'{tf} {suffix}', fontweight='bold'); ax1.grid(True, alpha=0.3)
            
            # Row 2: Channel bars
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
        # ROW 1: TS CHANNELS
        # ====================================================================
        ref_ts_imp = ref_result.ts_channel_importance
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
            
            record = {
                'timeframe': tf, 
                'hours': r.timeframe_hours, 
                'censor_step': r.censor_step,
                'effective_steps': r.effective_steps,
                
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

def run_temporal_shap_analysis(data, learn, pid=None, sample_idx=None, timeframes=None,
                               max_background_samples=200, save_dir='reports/', verbose=False):
    """
    Run complete temporal SHAP analysis with all visualizations.
    
    Args:
        data: Data dict from prepare_data_and_dls()
        learn: Trained Learner
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
        learn.model, data, data["mixed_dls"].train, 'cuda', max_background_samples
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