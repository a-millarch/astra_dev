# ============================================================================
# CACHING: Save/Load prepared data to avoid recomputation
# ============================================================================
import hashlib
import json
import logging
import os
import pickle

import numpy as np

from astra.data.dataloader import prepare_data_and_dls
from astra.data.mixed_dataloader import (
    AstraMixedDataset,
    AstraMixedDataLoader,
)
from astra.utils import cfg

logger = logging.getLogger(__name__)

# Bump this when cached data format changes to auto-invalidate old caches
_CACHE_VERSION = 5


def _get_cache_key(cfg):
    """Generate a unique cache key based on config parameters that affect data preparation."""
    # Resolve the active exclusion profile contents (not just the name)
    # so that changing criteria within a profile invalidates the cache.
    from astra.data.datasets import resolve_exclusion_criteria
    resolved_exclusion = resolve_exclusion_criteria(cfg)

    key_params = {
        "_cache_version": _CACHE_VERSION,
        "target": cfg.get("target"),
        "dataset": cfg.get("dataset", {}),
        "exclusion_criteria_resolved": resolved_exclusion,
        "prehospital": cfg.get("prehospital", False),
        "prehospital_only": cfg.get("prehospital_only", False),
        "concepts": cfg.get("concepts", []),
        "target": cfg.get("target"),
        "holdout_split_date": cfg.get("holdout_split_date", "2023-06-01"),
        "training_bs": cfg.get("training", {}).get("bs"),
        "bin_intervals": cfg.get("bin_intervals", {}),
        "bin_freq_include": cfg.get("bin_freq_include", []),
        "ebm_enabled": cfg.get("ebm_feature", {}).get("enabled", False),
        "temporal_features": cfg.get("temporal_features", {}).get("enabled", False),
        "categorical_profiles": cfg.get("categorical_profiles", {}),
        "agg_func": cfg.get("agg_func", {}),
        "normalization": cfg.get("normalization", {}),
        "drop_features": cfg.get("drop_features", {}),
    }
    config_str = json.dumps(key_params, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def save_data_cache(data, cfg, cache_dir='data/cache'):
    """
    Save prepared data to disk for faster subsequent loads.

    Saves all arrays, scalers, encoders, and metadata needed to recreate
    the full data dictionary (including dataloaders) on load.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _get_cache_key(cfg)
    cache_path = os.path.join(cache_dir, f'data_cache_{cache_key}.pkl')

    # Extract x_cat / x_cont from trainval and holdout datasets
    trainval_ds = data['mixed_dls']._train_ds
    holdout_ds = data['holdout_mixed_dls']._train_ds
    # AstraMixedDataset stores tensors; convert to numpy for pickling
    if hasattr(trainval_ds, 'dataset'):
        # It's a Subset — get underlying dataset
        trainval_ds = trainval_ds.dataset
    if hasattr(holdout_ds, 'dataset'):
        holdout_ds = holdout_ds.dataset

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

        # Encoded tabular arrays (avoid re-encoding on load)
        'trainval_x_cat': trainval_ds.x_cat.numpy(),
        'trainval_x_cont': trainval_ds.x_cont.numpy(),
        'holdout_x_cat': holdout_ds.x_cat.numpy(),
        'holdout_x_cont': holdout_ds.x_cont.numpy(),

        # Trajectory lengths
        'trajectory_lengths': data['trajectory_lengths'],
        'holdout_trajectory_lengths': data['holdout_trajectory_lengths'],

        # Scalers and encoders
        'ts_scaler': data['ts_scaler'],
        'tab_scaler': data['tab_scaler'],
        'cat_encoder': data['cat_encoder'],
        'tab_encoder': data['tab_encoder'],
        'encoding_info': data['encoding_info'],

        # Feature metadata
        'cat_cols': data['cat_cols'],
        'num_cols': data['num_cols'],
        'ts_feature_names': data['ts_feature_names'],
        'ts_channel_names': data['ts_channel_names'],
        'classes': data['classes'],

        # Channel indices
        'ebm_channel_idx': data.get('ebm_channel_idx'),
        'temporal_channel_idx': data.get('temporal_channel_idx'),
        'exclude_channel_indices': data.get('exclude_channel_indices'),

        # Base dataframe and TSDS objects (for downstream use)
        'base': data['base'],
        'trainval': data['trainval'],
        'holdout': data['holdout'],

        # Dimensions
        'c_in': data['c_in'],
        'seq_len': data['seq_len'],
        'ts_cat_dims': data['ts_cat_dims'],

        # Config
        'cfg': dict(cfg),
        '_cache_key': cache_key,
        '_cache_version': _CACHE_VERSION,
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved data cache to {cache_path}")
    logger.info(f"  Cache key: {cache_key}")
    logger.info(f"  File size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB")

    return cache_path


def load_data_cache(cfg, cache_dir='data/cache'):
    """
    Load cached data and recreate dataloaders.

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

    # Validate cache key and version
    if cache_data.get('_cache_key') != cache_key:
        logger.warning("Cache key mismatch - regenerating data")
        return None

    if cache_data.get('_cache_version', 1) != _CACHE_VERSION:
        logger.warning(f"Cache version mismatch (got {cache_data.get('_cache_version', 1)}, "
                       f"expected {_CACHE_VERSION}) - regenerating data")
        return None

    logger.info("Recreating dataloaders from cached data...")

    bs = cfg["training"]["bs"]

    # ========== TRAINVAL DATALOADERS ==========
    trainval_dataset = AstraMixedDataset(
        X_ts=cache_data['X'],
        x_cat=cache_data['trainval_x_cat'],
        x_cont=cache_data['trainval_x_cont'],
        X_ts_cat=cache_data['X_multi_hot'],
        y=cache_data['y'],
        trajectory_lengths=cache_data['trajectory_lengths'],
    )
    mixed_dls = AstraMixedDataLoader(
        trainval_dataset,
        splits=None,
        bs=bs,
        shuffle_train=False,
    )

    # ========== HOLDOUT DATALOADERS ==========
    holdout_dataset = AstraMixedDataset(
        X_ts=cache_data['tX'],
        x_cat=cache_data['holdout_x_cat'],
        x_cont=cache_data['holdout_x_cont'],
        X_ts_cat=cache_data['tX_multi_hot'],
        y=cache_data['ty'],
        trajectory_lengths=cache_data['holdout_trajectory_lengths'],
    )
    holdout_mixed_dls = AstraMixedDataLoader(
        holdout_dataset,
        splits=None,
        bs=bs,
        shuffle_train=False,
    )

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
        "classes": cache_data['classes'],
        "mixed_dls": mixed_dls,
        "holdout_mixed_dls": holdout_mixed_dls,
        "encoding_info": cache_data['encoding_info'],
        "cat_encoder": cache_data['cat_encoder'],
        "tab_encoder": cache_data['tab_encoder'],
        "ts_scaler": cache_data['ts_scaler'],
        "tab_scaler": cache_data['tab_scaler'],
        "ts_feature_names": cache_data['ts_feature_names'],
        "ts_channel_names": cache_data['ts_channel_names'],
        "trajectory_lengths": cache_data['trajectory_lengths'],
        "holdout_trajectory_lengths": cache_data['holdout_trajectory_lengths'],
        "ebm_channel_idx": cache_data.get('ebm_channel_idx'),
        "temporal_channel_idx": cache_data.get('temporal_channel_idx'),
        "exclude_channel_indices": cache_data.get('exclude_channel_indices'),
        "c_in": cache_data['c_in'],
        "seq_len": cache_data['seq_len'],
        "ts_cat_dims": cache_data['ts_cat_dims'],
        "cfg": cache_data.get('cfg'),
    }

    logger.info("Data loaded from cache successfully")
    return data


def load_data_cache_from_path(cache_path: str, bs: int = 64):
    """Load a data cache from an explicit file path, bypassing cache key validation."""
    logger.info(f"Loading data cache from explicit path: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    logger.info("Recreating dataloaders from cached data...")

    trainval_dataset = AstraMixedDataset(
        X_ts=cache_data['X'],
        x_cat=cache_data['trainval_x_cat'],
        x_cont=cache_data['trainval_x_cont'],
        X_ts_cat=cache_data['X_multi_hot'],
        y=cache_data['y'],
        trajectory_lengths=cache_data['trajectory_lengths'],
    )
    mixed_dls = AstraMixedDataLoader(
        trainval_dataset, splits=None, bs=bs, shuffle_train=False,
    )

    holdout_dataset = AstraMixedDataset(
        X_ts=cache_data['tX'],
        x_cat=cache_data['holdout_x_cat'],
        x_cont=cache_data['holdout_x_cont'],
        X_ts_cat=cache_data['tX_multi_hot'],
        y=cache_data['ty'],
        trajectory_lengths=cache_data['holdout_trajectory_lengths'],
    )
    holdout_mixed_dls = AstraMixedDataLoader(
        holdout_dataset, splits=None, bs=bs, shuffle_train=False,
    )

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
        "classes": cache_data['classes'],
        "mixed_dls": mixed_dls,
        "holdout_mixed_dls": holdout_mixed_dls,
        "encoding_info": cache_data['encoding_info'],
        "cat_encoder": cache_data['cat_encoder'],
        "tab_encoder": cache_data['tab_encoder'],
        "ts_scaler": cache_data['ts_scaler'],
        "tab_scaler": cache_data['tab_scaler'],
        "ts_feature_names": cache_data['ts_feature_names'],
        "ts_channel_names": cache_data['ts_channel_names'],
        "trajectory_lengths": cache_data['trajectory_lengths'],
        "holdout_trajectory_lengths": cache_data['holdout_trajectory_lengths'],
        "ebm_channel_idx": cache_data.get('ebm_channel_idx'),
        "temporal_channel_idx": cache_data.get('temporal_channel_idx'),
        "exclude_channel_indices": cache_data.get('exclude_channel_indices'),
        "c_in": cache_data['c_in'],
        "seq_len": cache_data['seq_len'],
        "ts_cat_dims": cache_data['ts_cat_dims'],
        "cfg": cache_data.get('cfg'),
    }

    logger.info("Data loaded from explicit cache path successfully")
    return data


def prepare_data_and_dls_cached(cfg, use_cache=True, cache_dir='data/cache', force_refresh=False):
    """
    Wrapper for prepare_data_and_dls with caching support.

    First attempts to load from cache. If cache miss or force_refresh=True,
    runs full data preparation and saves to cache.
    """
    if use_cache and not force_refresh:
        data = load_data_cache(cfg, cache_dir=cache_dir)
        if data is not None:
            return data

    logger.info("Running full data preparation...")
    data = prepare_data_and_dls(cfg)

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
        cache_key = _get_cache_key(cfg)
        cache_path = os.path.join(cache_dir, f'data_cache_{cache_key}.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Removed cache: {cache_path}")
        else:
            logger.info(f"No cache found for key {cache_key}")
    else:
        import glob
        cache_files = glob.glob(os.path.join(cache_dir, 'data_cache_*.pkl'))
        for f in cache_files:
            os.remove(f)
        logger.info(f"Cleared {len(cache_files)} cache file(s)")
