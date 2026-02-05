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