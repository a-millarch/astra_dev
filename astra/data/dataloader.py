# dataloader.py


import pandas as pd
import numpy as np

from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.preparation import df2xy

from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle

from astra.utils import get_base_df, logger, align_dataframes
from astra.data.preprocessing import MultiHotCategoricalEncoder
from astra.data.datasets import TSDS 


# ============================================================================
# FIXED: Masked Normalization Functions
# ============================================================================

def normalize_with_padding_mask(X, scaler, padding_value=0.0, fit=True):
    """
    Normalize time series data while preserving padding zeros.
    
    The standard approach (StandardScaler) transforms ALL values including padding:
        0 → (0 - mean) / std = -mean/std ≠ 0
    
    This creates false signal in padding regions, causing incorrect SHAP attributions.
    
    This function:
    1. Identifies padding positions (where value == padding_value)
    2. Fits scaler only on non-padding values
    3. Transforms only non-padding values
    4. Restores padding positions to padding_value
    
    Args:
        X: Array of shape [n_samples, n_channels, seq_len]
        scaler: sklearn scaler (StandardScaler, RobustScaler, etc.)
        padding_value: Value used for padding (typically 0.0)
        fit: If True, fit the scaler. If False, only transform.
    
    Returns:
        X_normalized: Normalized array with padding preserved as padding_value
    """
    n_samples, n_channels, seq_len = X.shape
    
    # Create mask for non-padding values
    non_padding_mask = ~np.isclose(X, padding_value, atol=1e-8)
    
    # Reshape for scaler: [samples*seq_len, n_channels]
    X_reshaped = X.reshape(-1, n_channels)
    mask_reshaped = non_padding_mask.reshape(-1, n_channels)
    
    if fit:
        # Compute statistics only on non-padding values (per feature)
        means = np.zeros(n_channels)
        stds = np.zeros(n_channels)
        
        for ch in range(n_channels):
            ch_data = X_reshaped[:, ch]
            ch_mask = mask_reshaped[:, ch]
            valid_data = ch_data[ch_mask]
            
            if len(valid_data) > 0:
                means[ch] = valid_data.mean()
                stds[ch] = valid_data.std()
                if stds[ch] == 0 or np.isnan(stds[ch]):
                    stds[ch] = 1.0  # Avoid division by zero
            else:
                means[ch] = 0.0
                stds[ch] = 1.0
        
        # Store in scaler-compatible format
        scaler.mean_ = means
        scaler.scale_ = stds
        scaler.var_ = stds ** 2
        scaler.n_features_in_ = n_channels
        
        logger.info(f"Fitted scaler on non-padding data:")
        logger.info(f"  Mean range: [{means.min():.4f}, {means.max():.4f}]")
        logger.info(f"  Std range: [{stds.min():.4f}, {stds.max():.4f}]")
    
    # Transform: (x - mean) / std, but only for non-padding
    X_normalized = np.zeros_like(X_reshaped)
    
    for ch in range(n_channels):
        ch_data = X_reshaped[:, ch]
        ch_mask = mask_reshaped[:, ch]
        
        # Normalize non-padding values
        X_normalized[ch_mask, ch] = (ch_data[ch_mask] - scaler.mean_[ch]) / scaler.scale_[ch]
        
        # Keep padding as padding_value (already 0 from np.zeros_like)
        X_normalized[~ch_mask, ch] = padding_value
    
    # Reshape back
    X_normalized = X_normalized.reshape(n_samples, n_channels, seq_len)
    
    return X_normalized


def get_trajectory_lengths(X, padding_value=0.0):
    """
    Get the actual trajectory length for each sample (last timestep with data).
    
    Args:
        X: Array of shape [n_samples, n_channels, seq_len]
        padding_value: Value used for padding
    
    Returns:
        trajectory_lengths: Array [n_samples] with length of each trajectory
    """
    n_samples, n_channels, seq_len = X.shape
    
    # A timestep has data if ANY channel has non-padding value
    has_data = ~np.isclose(X, padding_value, atol=1e-8).all(axis=1)  # [n_samples, seq_len]
    
    trajectory_lengths = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        nonzero_idx = np.where(has_data[i])[0]
        if len(nonzero_idx) > 0:
            trajectory_lengths[i] = nonzero_idx[-1] + 1
    
    return trajectory_lengths


# ============================================================================
# Utility functions (unchanged)
# ============================================================================

def tscatdfwide2x(df_wide:pd.DataFrame, sample_col:str='PID', cat_col='FEATURE'):
    encoder = MultiHotCategoricalEncoder()
    X_multi_hot, encoding_info = encoder.fit_transform(
        df_wide,
        sample_col=sample_col,
        timestep_cols=df_wide.timestep_cols,
        cat_col=cat_col,
        feature_names=df_wide.FEATURE.dropna().unique()
    )
    return X_multi_hot, encoding_info 


def dfwide2ts_dls(df_wide, y, cfg, encoder=None):
    """Create categorical TS dataloader with optional pre-fitted encoder."""
    if encoder is None:
        encoder = MultiHotCategoricalEncoder()
        X_multi_hot, encoding_info = encoder.fit_transform(
            df_wide,
            sample_col='PID',
            timestep_cols=df_wide.timestep_cols,
            cat_col='FEATURE',
            feature_names=df_wide.FEATURE.dropna().unique()
        )
    else:
        X_multi_hot, encoding_info = encoder.transform(
            df_wide,
            sample_col='PID',
            timestep_cols=df_wide.timestep_cols,
            cat_col='FEATURE'
        )
    
    logger.debug(f"X_multi_hot shape: {X_multi_hot.shape}")
    
    ts_cat_dls = get_ts_dls(
        X_multi_hot.astype(np.int64), 
        y, 
        splits=None, 
        bs=cfg["training"]["bs"],
        shuffle=False
    )
    
    ts_cat_dims = {
        feat_name: end - start 
        for feat_name, (start, end) in encoding_info['feature_ranges'].items()
    }
    ts_cat_dls.ts_cat_dims = ts_cat_dims
    ts_cat_dls.X_multi_hot = X_multi_hot
    return ts_cat_dls, encoding_info, encoder


# ============================================================================
# FIXED: Main data preparation function
# ============================================================================

def prepare_data_and_dls(cfg):
    """
    Prepare data and dataloaders with FIXED normalization that preserves padding.
    
    KEY FIX: Uses normalize_with_padding_mask() to ensure:
    - Scaler is fit only on non-padding (real) data
    - Padding zeros remain as zeros after normalization
    - Model correctly distinguishes signal from padding
    """
    # Load dataframes
    base = get_base_df()
    if cfg["dataset"]["exclusion"] == "lvl1tc":
        base = base[base.LVL1TC == 1]

    concepts = cfg["concepts"]

    # Use temporal split from config
    split_date = cfg.get("holdout_split_date", "2023-06-01")
    logger.info(f"Using temporal split date: {split_date}")
    logger.info(f"  Training set: ServiceDate <= {split_date}")
    logger.info(f"  Holdout set:  ServiceDate > {split_date}")

    holdout = TSDS(cfg, base[base.ServiceDate > split_date].copy(deep=True))
    trainval = TSDS(cfg, base[base.ServiceDate <= split_date].copy(deep=True))

    # Split concepts into categorical and continuous
    for tsds in [holdout, trainval]:
        tsds.cat_concepts = {
            k: tsds.concepts[k] 
            for k in cfg["dataset"]["ts_cat_names"] 
            if k in tsds.concepts
        }
        tsds.cont_concepts = {
            k: v 
            for k, v in tsds.concepts.items() 
            if k not in cfg["dataset"]["ts_cat_names"]
        }
        tsds.complete = pd.concat(tsds.cont_concepts).fillna(0.0)
        tsds.complete_cat = pd.concat(tsds.cat_concepts)
        tsds.complete_cat.timestep_cols = tsds.timestep_cols
    
    # Align dataframes
    trainval.complete, holdout.complete = align_dataframes(
        trainval.complete, 
        holdout.complete
    )
    
    cat_cols = cfg["dataset"]["cat_cols"]
    num_cols = cfg["dataset"]["num_cols"]
    logger.info(f'Categoricals: {cat_cols}\nNumericals: {num_cols}')

    # Common transforms (NO batch transforms!)
    tfms = [None, [Categorize()]]
    batch_tfms = None
    procs = [Categorify, FillMissing]
    
    # Get classes from combined data
    complete_tab_dls = get_tabular_dls(
        pd.concat([trainval.tab_df, holdout.tab_df]),
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        drop_last=False,
        shuffle=False
    )
    classes = complete_tab_dls.classes
    
    # ============================================================================
    # TRAINVAL DATA EXTRACTION
    # ============================================================================
    logger.info("Setting up X,y for training and validation")
    X, y = df2xy(
        trainval.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=trainval.complete.columns[3:-1],
        target_col=cfg["target"]
    )
    y = list(y[:, 0].flatten())
    logger.info(f'Train/val X shape (before normalization): {X.shape}')
    
    # Store raw X for debugging
    X_raw = X.copy()

    # ============================================================================
    # FIXED: FIT SCALERS ON TRAINVAL ONLY, PRESERVING PADDING
    # ============================================================================
    logger.info("Fitting normalization scalers on trainval data (excluding padding)...")
    
    # 1. CONTINUOUS TIME SERIES SCALER - FIXED
    ts_scaler = StandardScaler()
    
    # Get trajectory lengths to understand padding
    traj_lengths = get_trajectory_lengths(X, padding_value=0.0)
    logger.info(f'Trajectory lengths - min: {traj_lengths.min()}, max: {traj_lengths.max()}, '
               f'mean: {traj_lengths.mean():.1f}')
    
    # FIXED: Normalize while preserving padding
    X_normalized = normalize_with_padding_mask(X, ts_scaler, padding_value=0.0, fit=True)
    
    logger.info(f'Train/val X shape (after normalization): {X_normalized.shape}')
    
    # Verify padding is preserved
    padding_mask = np.isclose(X_raw, 0.0, atol=1e-8)
    padding_after = X_normalized[padding_mask]
    logger.info(f'Padding verification:')
    logger.info(f'  Padding positions: {padding_mask.sum()}')
    logger.info(f'  Padding values after norm - mean: {padding_after.mean():.6f}, std: {padding_after.std():.6f}')
    
    if np.abs(padding_after.mean()) > 0.001:
        logger.warning(f'⚠️ Padding was not preserved! Mean should be ~0, got {padding_after.mean():.6f}')
    else:
        logger.info(f'✓ Padding preserved correctly (mean ≈ 0)')
    
    # Stats on non-padding data
    non_padding_data = X_normalized[~padding_mask]
    logger.info(f'Non-padding data stats: mean={non_padding_data.mean():.4f}, std={non_padding_data.std():.4f}')
    
    # 2. TABULAR DATA SCALER (unchanged - no padding issue)
    tab_scaler = StandardScaler()
    
    if num_cols:
        logger.info(f'Fitting tabular scaler on {len(num_cols)} continuous features')
        tab_scaler.fit(trainval.tab_df[num_cols])
        trainval_tab_normalized = trainval.tab_df.copy()
        trainval_tab_normalized[num_cols] = tab_scaler.transform(trainval.tab_df[num_cols])
    else:
        trainval_tab_normalized = trainval.tab_df

    # ============================================================================
    # TRAINVAL DATALOADERS
    # ============================================================================
    logger.info("Creating trainval dataloaders...")
    
    ts_dls = get_ts_dls(
        X_normalized,
        y,
        splits=None,
        tfms=tfms,
        batch_tfms=None,
        bs=cfg["training"]["bs"],
        drop_last=False,
        shuffle=False
    )
   
    tab_dls = get_tabular_dls(
        trainval_tab_normalized,
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        bs=cfg["training"]["bs"],
        drop_last=False,
        shuffle=False
    )

    ts_cat_dls, encoding_info, cat_encoder = dfwide2ts_dls(
        trainval.complete_cat, 
        y, 
        cfg,
        encoder=None
    )
    
    mixed_dls = get_mixed_dls(
        ts_dls,
        tab_dls,
        ts_cat_dls,
        bs=cfg["training"]["bs"]
    )

    # ============================================================================
    # HOLDOUT DATA EXTRACTION
    # ============================================================================
    logger.info('Preparing holdout data...')
    tX, ty = df2xy(
        holdout.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=holdout.complete.columns[3:-1],
        target_col=holdout.target
    )
    ty = list(ty[:, 0].flatten())
    logger.info(f'Holdout X shape (before normalization): {tX.shape}')
    
    # Store raw for debugging
    tX_raw = tX.copy()

    # ============================================================================
    # FIXED: TRANSFORM HOLDOUT WITH FITTED SCALERS, PRESERVING PADDING
    # ============================================================================
    logger.info("Applying normalization to holdout (preserving padding)...")
    
    # FIXED: Use masked normalization for holdout too
    tX_normalized = normalize_with_padding_mask(tX, ts_scaler, padding_value=0.0, fit=False)
    
    # Verify
    holdout_traj_lengths = get_trajectory_lengths(tX, padding_value=0.0)
    logger.info(f'Holdout trajectory lengths - min: {holdout_traj_lengths.min()}, '
               f'max: {holdout_traj_lengths.max()}, mean: {holdout_traj_lengths.mean():.1f}')
    
    holdout_padding_mask = np.isclose(tX_raw, 0.0, atol=1e-8)
    holdout_padding_after = tX_normalized[holdout_padding_mask]
    logger.info(f'Holdout padding verification:')
    logger.info(f'  Padding values after norm - mean: {holdout_padding_after.mean():.6f}')
    
    # Tabular
    if num_cols:
        holdout_tab_normalized = holdout.tab_df.copy()
        holdout_tab_normalized[num_cols] = tab_scaler.transform(holdout.tab_df[num_cols])
    else:
        holdout_tab_normalized = holdout.tab_df

    # ============================================================================
    # HOLDOUT DATALOADERS
    # ============================================================================
    logger.info("Creating holdout dataloaders...")
    
    test_ts_dls = get_ts_dls(
        tX_normalized,
        ty,
        splits=None,
        tfms=tfms,
        batch_tfms=None,
        bs=cfg["training"]["bs"],
        drop_last=False,
        shuffle=False
    )
    
    test_tab_dls = get_tabular_dls(
        holdout_tab_normalized,
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        drop_last=False,
        shuffle=False
    )

    test_ts_cat_dls, holdout_encoding_info, _ = dfwide2ts_dls(
        holdout.complete_cat, 
        ty, 
        cfg,
        encoder=cat_encoder
    )

    holdout_mixed_dls = get_mixed_dls(
        test_ts_dls,
        test_tab_dls,
        test_ts_cat_dls,
        bs=cfg["training"]["bs"]
    )

    # ============================================================================
    # VALIDATION
    # ============================================================================
    # Check non-padding data is normalized
    trainval_non_padding = X_normalized[~padding_mask]
    assert abs(trainval_non_padding.mean()) < 0.5, f"Non-padding data not centered! mean={trainval_non_padding.mean():.4f}"
    
    # Check padding is preserved
    assert abs(X_normalized[padding_mask].mean()) < 0.001, "Padding not preserved!"
    
    logger.info("✓ Normalization validation passed!")
    logger.info("✓ Padding preserved correctly!")
    
    # ============================================================================
    # RETURN
    # ============================================================================
    return {
        "base": base,
        "trainval": trainval,
        "holdout": holdout,
        "X": X_normalized,
        "X_raw": X_raw,  # NEW: Include raw data for debugging
        "X_multi_hot": ts_cat_dls.X_multi_hot, 
        "y": y,
        "tX": tX_normalized,
        "tX_raw": tX_raw,  # NEW: Include raw data for debugging
        "tX_multi_hot": test_ts_cat_dls.X_multi_hot,
        "ty": ty,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "tfms": tfms,
        "batch_tfms": None,
        "procs": procs,
        "classes": classes,
        "mixed_dls": mixed_dls,
        "holdout_mixed_dls": holdout_mixed_dls,
        "ts_dls": ts_dls,
        "holdout_ts_dls": test_ts_dls,
        "ts_cat_dls": ts_cat_dls,
        "holdout_ts_cat_dls": test_ts_cat_dls,
        "encoding_info": encoding_info,
        "cat_encoder": cat_encoder,
        "ts_scaler": ts_scaler,
        "tab_scaler": tab_scaler,
        "ts_feature_names": trainval.complete.columns[3:-1].tolist(),
        "trajectory_lengths": traj_lengths,  # NEW: trainval trajectory lengths
        "holdout_trajectory_lengths": holdout_traj_lengths,  # NEW: holdout trajectory lengths
    }


# ============================================================================
# UTILITY: Save/Load (unchanged)
# ============================================================================

def save_normalization_artifacts(data, model_name, save_dir='models/scalers'):
    """Save normalization scalers and metadata for deployment."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    artifacts = {
        'ts_scaler': data['ts_scaler'],
        'tab_scaler': data['tab_scaler'],
        'ts_feature_names': data.get('ts_feature_names'),
        'tab_feature_names': data['num_cols'],
        'cat_feature_names': data['cat_cols'],
        'encoding_info': data['encoding_info'],
        'cat_encoder': data['cat_encoder'],
        'model_name': model_name,
        'scaler_type': type(data['ts_scaler']).__name__,
    }
    
    save_path = f'{save_dir}/normalization_{model_name}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    logger.info(f"Saved normalization artifacts to {save_path}")
    return save_path


def load_normalization_artifacts(model_name, load_dir='models/scalers'):
    """Load saved normalization artifacts."""
    load_path = f'{load_dir}/normalization_{model_name}.pkl'
    with open(load_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    logger.info(f"Loaded normalization artifacts from {load_path}")
    return artifacts


def normalize_new_patient(patient_ts_data, patient_tab_data, artifacts):
    """
    Apply saved normalization to new patient data, preserving padding.
    """
    if patient_ts_data.ndim == 2:
        patient_ts_data = patient_ts_data[np.newaxis, ...]
    
    # Use masked normalization to preserve padding
    ts_normalized = normalize_with_padding_mask(
        patient_ts_data, 
        artifacts['ts_scaler'], 
        padding_value=0.0, 
        fit=False
    )
    
    # Tabular (no padding issue)
    num_cols = artifacts['tab_feature_names']
    if num_cols:
        if isinstance(patient_tab_data, dict):
            patient_tab_data = pd.DataFrame([patient_tab_data])
        tab_normalized = patient_tab_data.copy()
        tab_normalized[num_cols] = artifacts['tab_scaler'].transform(patient_tab_data[num_cols])
    else:
        tab_normalized = patient_tab_data
    
    return ts_normalized, tab_normalized


# ============================================================================
# CACHING: Save/Load prepared data to avoid recomputation
# ============================================================================

import hashlib
import json
import os

def _get_cache_key(cfg):
    """Generate a unique cache key based on config parameters that affect data preparation."""
    # Extract config values that affect data preparation
    key_params = {
        "dataset": cfg.get("dataset", {}),
        "concepts": cfg.get("concepts", []),
        "target": cfg.get("target"),
        "holdout_split_date": cfg.get("holdout_split_date", "2023-06-01"),
        "training_bs": cfg.get("training", {}).get("bs"),
    }
    # Create deterministic hash
    config_str = json.dumps(key_params, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def save_data_cache(data, cfg, cache_dir='cache/data'):
    """
    Save prepared data to disk for faster subsequent loads.

    Saves all arrays, scalers, encoders, and metadata needed to recreate
    the full data dictionary (including dataloaders) on load.

    Args:
        data: Dictionary returned by prepare_data_and_dls()
        cfg: Config dictionary used to generate cache key
        cache_dir: Directory to save cache files

    Returns:
        cache_path: Path to saved cache file
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _get_cache_key(cfg)
    cache_path = os.path.join(cache_dir, f'data_cache_{cache_key}.pkl')

    # Store everything needed to recreate dataloaders
    cache_data = {
        # Raw and normalized arrays
        'X': data['X'],
        'X_raw': data['X_raw'],
        'X_multi_hot': data['X_multi_hot'],
        'y': data['y'],
        'tX': data['tX'],
        'tX_raw': data['tX_raw'],
        'tX_multi_hot': data['tX_multi_hot'],
        'ty': data['ty'],

        # Trajectory lengths
        'trajectory_lengths': data['trajectory_lengths'],
        'holdout_trajectory_lengths': data['holdout_trajectory_lengths'],

        # Scalers and encoders
        'ts_scaler': data['ts_scaler'],
        'tab_scaler': data['tab_scaler'],
        'cat_encoder': data['cat_encoder'],
        'encoding_info': data['encoding_info'],

        # Feature metadata
        'cat_cols': data['cat_cols'],
        'num_cols': data['num_cols'],
        'ts_feature_names': data['ts_feature_names'],
        'classes': data['classes'],
        'tfms': data['tfms'],
        'batch_tfms': data['batch_tfms'],
        'procs': data['procs'],

        # Tabular dataframes (needed to recreate tab_dls)
        'trainval_tab_df': data['trainval'].tab_df,
        'holdout_tab_df': data['holdout'].tab_df,

        # Categorical TS data (needed to recreate ts_cat_dls)
        'trainval_complete_cat': data['trainval'].complete_cat,
        'holdout_complete_cat': data['holdout'].complete_cat,

        # Base dataframe and TSDS objects (for downstream use)
        'base': data['base'],
        'trainval': data['trainval'],
        'holdout': data['holdout'],

        # Config snapshot for validation
        '_cache_key': cache_key,
        '_cfg_snapshot': {
            'target': cfg.get('target'),
            'holdout_split_date': cfg.get('holdout_split_date'),
        }
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved data cache to {cache_path}")
    logger.info(f"  Cache key: {cache_key}")
    logger.info(f"  File size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB")

    return cache_path


def load_data_cache(cfg, cache_dir='cache/data'):
    """
    Load cached data and recreate dataloaders.

    Args:
        cfg: Config dictionary (used to find correct cache and recreate dataloaders)
        cache_dir: Directory containing cache files

    Returns:
        data: Dictionary matching prepare_data_and_dls() output, or None if cache not found
    """
    cache_key = _get_cache_key(cfg)
    cache_path = os.path.join(cache_dir, f'data_cache_{cache_key}.pkl')

    if not os.path.exists(cache_path):
        logger.info(f"No cache found for key {cache_key}")
        return None

    logger.info(f"Loading data cache from {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    # Validate cache key matches
    if cache_data.get('_cache_key') != cache_key:
        logger.warning("Cache key mismatch - regenerating data")
        return None

    logger.info("Recreating dataloaders from cached data...")

    # Common parameters
    tfms = cache_data['tfms']
    procs = cache_data['procs']
    cat_cols = cache_data['cat_cols']
    num_cols = cache_data['num_cols']
    bs = cfg["training"]["bs"]

    # ========== TRAINVAL DATALOADERS ==========
    ts_dls = get_ts_dls(
        cache_data['X'],
        cache_data['y'],
        splits=None,
        tfms=tfms,
        batch_tfms=None,
        bs=bs,
        drop_last=False,
        shuffle=False
    )

    tab_dls = get_tabular_dls(
        cache_data['trainval_tab_df'],
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        bs=bs,
        drop_last=False,
        shuffle=False
    )

    # Recreate ts_cat_dls using cached encoder
    ts_cat_dls, _, _ = dfwide2ts_dls(
        cache_data['trainval_complete_cat'],
        cache_data['y'],
        cfg,
        encoder=cache_data['cat_encoder']
    )

    mixed_dls = get_mixed_dls(ts_dls, tab_dls, ts_cat_dls, bs=bs)

    # ========== HOLDOUT DATALOADERS ==========
    test_ts_dls = get_ts_dls(
        cache_data['tX'],
        cache_data['ty'],
        splits=None,
        tfms=tfms,
        batch_tfms=None,
        bs=bs,
        drop_last=False,
        shuffle=False
    )

    test_tab_dls = get_tabular_dls(
        cache_data['holdout_tab_df'],
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        drop_last=False,
        shuffle=False
    )

    test_ts_cat_dls, _, _ = dfwide2ts_dls(
        cache_data['holdout_complete_cat'],
        cache_data['ty'],
        cfg,
        encoder=cache_data['cat_encoder']
    )

    holdout_mixed_dls = get_mixed_dls(test_ts_dls, test_tab_dls, test_ts_cat_dls, bs=bs)

    # ========== ASSEMBLE OUTPUT ==========
    data = {
        "base": cache_data['base'],
        "trainval": cache_data['trainval'],
        "holdout": cache_data['holdout'],
        "X": cache_data['X'],
        "X_raw": cache_data['X_raw'],
        "X_multi_hot": cache_data['X_multi_hot'],
        "y": cache_data['y'],
        "tX": cache_data['tX'],
        "tX_raw": cache_data['tX_raw'],
        "tX_multi_hot": cache_data['tX_multi_hot'],
        "ty": cache_data['ty'],
        "cat_cols": cache_data['cat_cols'],
        "num_cols": cache_data['num_cols'],
        "tfms": cache_data['tfms'],
        "batch_tfms": cache_data['batch_tfms'],
        "procs": cache_data['procs'],
        "classes": cache_data['classes'],
        "mixed_dls": mixed_dls,
        "holdout_mixed_dls": holdout_mixed_dls,
        "ts_dls": ts_dls,
        "holdout_ts_dls": test_ts_dls,
        "ts_cat_dls": ts_cat_dls,
        "holdout_ts_cat_dls": test_ts_cat_dls,
        "encoding_info": cache_data['encoding_info'],
        "cat_encoder": cache_data['cat_encoder'],
        "ts_scaler": cache_data['ts_scaler'],
        "tab_scaler": cache_data['tab_scaler'],
        "ts_feature_names": cache_data['ts_feature_names'],
        "trajectory_lengths": cache_data['trajectory_lengths'],
        "holdout_trajectory_lengths": cache_data['holdout_trajectory_lengths'],
    }

    logger.info("✓ Data loaded from cache successfully")
    return data


def prepare_data_and_dls_cached(cfg, use_cache=True, cache_dir='cache/data', force_refresh=False):
    """
    Wrapper for prepare_data_and_dls with caching support.

    First attempts to load from cache. If cache miss or force_refresh=True,
    runs full data preparation and saves to cache.

    Args:
        cfg: Config dictionary
        use_cache: If False, always run full preparation (but still save cache)
        cache_dir: Directory for cache files
        force_refresh: If True, ignore existing cache and regenerate

    Returns:
        data: Dictionary matching prepare_data_and_dls() output
    """
    # Try loading from cache
    if use_cache and not force_refresh:
        data = load_data_cache(cfg, cache_dir=cache_dir)
        if data is not None:
            return data

    # Cache miss or refresh requested - run full preparation
    logger.info("Running full data preparation...")
    data = prepare_data_and_dls(cfg)

    # Save to cache for next time
    if use_cache:
        try:
            save_data_cache(data, cfg, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    return data


def clear_data_cache(cache_dir='cache/data', cfg=None):
    """
    Clear cached data files.

    Args:
        cache_dir: Directory containing cache files
        cfg: If provided, only clear cache for this specific config.
             If None, clear all cache files.
    """
    if not os.path.exists(cache_dir):
        logger.info("No cache directory found")
        return

    if cfg is not None:
        # Clear specific cache
        cache_key = _get_cache_key(cfg)
        cache_path = os.path.join(cache_dir, f'data_cache_{cache_key}.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Removed cache: {cache_path}")
        else:
            logger.info(f"No cache found for key {cache_key}")
    else:
        # Clear all caches
        import glob
        cache_files = glob.glob(os.path.join(cache_dir, 'data_cache_*.pkl'))
        for f in cache_files:
            os.remove(f)
        logger.info(f"Cleared {len(cache_files)} cache file(s)")