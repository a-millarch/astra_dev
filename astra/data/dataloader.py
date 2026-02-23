# dataloader.py


import os
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

def normalize_with_padding_mask(X, scaler, trajectory_lengths, fit=True):
    """
    Normalize time series data per-channel, using trajectory_lengths for padding
    and NaN for missing measurements.

    After normalization:
      - Measured values  → standardized per-channel (≈zero mean, unit variance)
      - Missing measurements within trajectory → 0.0
      - Padding beyond trajectory end         → 0.0

    Args:
        X: Array [n_samples, n_channels, seq_len]. May contain NaN for
           positions where no clinical measurement was recorded.
        scaler: sklearn StandardScaler (stores mean_/scale_ per channel).
        trajectory_lengths: Array [n_samples] — number of real timesteps
           per sample.  Positions >= trajectory_lengths[i] are padding.
        fit: If True, compute and store per-channel statistics.

    Returns:
        X_normalized: Array of same shape.  0.0 at missing/padding positions.
    """
    n_samples, n_channels, seq_len = X.shape

    # --- build padding mask from trajectory_lengths [n_samples, seq_len] ---
    pos = np.arange(seq_len)[np.newaxis, :]                   # [1, seq_len]
    tl  = trajectory_lengths[:, np.newaxis]                    # [n_samples, 1]
    padding_2d = pos >= tl                                     # True = padding

    # expand to [n_samples, n_channels, seq_len]
    padding_3d = np.broadcast_to(
        padding_2d[:, np.newaxis, :], (n_samples, n_channels, seq_len)
    )

    # measured = has a real value (not NaN) AND within trajectory
    measured_mask = ~np.isnan(X) & ~padding_3d

    # ------------------------------------------------------------------
    if fit:
        means = np.zeros(n_channels)
        stds  = np.zeros(n_channels)

        for ch in range(n_channels):
            vals = X[:, ch, :][measured_mask[:, ch, :]]
            if len(vals) > 0:
                means[ch] = vals.mean()
                stds[ch]  = vals.std()
                if stds[ch] == 0 or np.isnan(stds[ch]):
                    stds[ch] = 1.0
            else:
                means[ch] = 0.0
                stds[ch]  = 1.0

        scaler.mean_          = means
        scaler.scale_         = stds
        scaler.var_           = stds ** 2
        scaler.n_features_in_ = n_channels

        logger.info("Fitted per-channel scaler on measured data:")
        logger.info(f"  Mean range: [{means.min():.4f}, {means.max():.4f}]")
        logger.info(f"  Std range:  [{stds.min():.4f}, {stds.max():.4f}]")

    # ------------------------------------------------------------------
    # normalise: only measured positions get values; rest stays 0.0
    X_normalized = np.zeros((n_samples, n_channels, seq_len), dtype=np.float64)

    for ch in range(n_channels):
        m = measured_mask[:, ch, :]
        if m.any():
            X_normalized[:, ch, :][m] = (
                (X[:, ch, :][m] - scaler.mean_[ch]) / scaler.scale_[ch]
            )

    return X_normalized


def get_trajectory_lengths(X, padding_value=0.0):
    """
    Get the actual trajectory length for each sample (last timestep with data).

    A timestep is padding if ALL channels are either NaN or equal to
    padding_value.  This handles both legacy (all-zero padding) and the
    NaN-for-missing convention.

    Args:
        X: Array of shape [n_samples, n_channels, seq_len]
        padding_value: Value used for padding (typically 0.0)

    Returns:
        trajectory_lengths: Array [n_samples] with length of each trajectory
    """
    n_samples, n_channels, seq_len = X.shape

    # A value is "absent" if NaN or equal to padding_value
    is_absent = np.isnan(X) | np.isclose(X, padding_value, atol=1e-8)

    # A timestep has data if at least one channel is present
    has_data = ~is_absent.all(axis=1)  # [n_samples, seq_len]

    trajectory_lengths = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        data_idx = np.where(has_data[i])[0]
        if len(data_idx) > 0:
            trajectory_lengths[i] = data_idx[-1] + 1

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
        tsds.complete = pd.concat(tsds.cont_concepts)  # NaN = missing measurement
        tsds.complete_cat = pd.concat(tsds.cat_concepts)
        tsds.complete_cat.timestep_cols = tsds.timestep_cols

    # Inject EBM prediction channel if enabled
    ebm_channel_idx = None
    if cfg.get('ebm_feature', {}).get('enabled', False):
        from astra.data.ebm_features import create_ebm_feature_df
        from astra.models.ebm.generate_ebm_feature import load_ebm_predictions
        logger.info("Injecting EBM prediction channel...")
        ebm_save_dir = cfg['ebm_feature'].get('save_dir', 'data/interim/ebm_features')
        ebm_predictions = load_ebm_predictions(ebm_save_dir)
        for tsds, split_name in [(trainval, 'trainval'), (holdout, 'holdout')]:
            ebm_df = create_ebm_feature_df(
                cfg, tsds.base, split=split_name,
                ebm_predictions=ebm_predictions, save_dir=ebm_save_dir,
            )
            tsds.cont_concepts['_ebm'] = ebm_df
            tsds.complete = pd.concat(tsds.cont_concepts)  # NaN = missing measurement
    
    # Align continuous dataframes (string column names)
    trainval.complete, holdout.complete = align_dataframes(
        trainval.complete,
        holdout.complete
    )
    # Align categorical TS to match continuous TS timestep columns.
    # align_dataframes expects string columns but categorical DFs use integer columns,
    # so we pad missing timestep columns directly.
    cont_ts_ints = sorted(
        int(str(c)) for c in trainval.complete.columns if str(c).isdigit()
    )
    for tsds_obj in [trainval, holdout]:
        df = tsds_obj.complete_cat
        cat_ts_ints = set(c for c in df.columns if isinstance(c, int))
        missing = set(cont_ts_ints) - cat_ts_ints
        if missing:
            for col in missing:
                df[col] = np.nan
        # Reorder: non-timestep columns first, then sorted timestep columns
        non_ts = [c for c in df.columns if not isinstance(c, int)]
        ts = sorted(c for c in df.columns if isinstance(c, int))
        tsds_obj.complete_cat = df[non_ts + ts]
        tsds_obj.complete_cat.timestep_cols = ts
    
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

    # Ensure classes includes _na columns added by FillMissing.
    # FillMissing creates {col}_na boolean indicators for numeric columns
    # with NaN values. These become categorical features in the model, but
    # depending on FastAI/TSAI version, .classes may not include them.
    df_combined = pd.concat([trainval.tab_df, holdout.tab_df])
    for col in num_cols:
        na_name = f'{col}_na'
        if df_combined[col].isna().any() and na_name not in classes:
            classes[na_name] = ['#na#', False, True]
            logger.info(f'  Added missing indicator to classes: {na_name}')

    # ============================================================================
    # TRAINVAL DATA EXTRACTION
    # ============================================================================
    logger.info("Setting up X,y for training and validation")
    X, y = df2xy(
        trainval.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=trainval.complete.columns[3:],
        target_col=cfg["target"]
    )
    y = list(y[:, 0].flatten())
    logger.info(f'Train/val X shape (before normalization): {X.shape}')

    # Channel names — df2xy sorts by FEATURE ascending, so this IS the channel order.
    # Computed once here and reused for EBM, temporal features, and save_deployment_bundle.
    ts_channel_names = sorted(trainval.complete['FEATURE'].unique())

    # Compute EBM channel index
    if cfg.get('ebm_feature', {}).get('enabled', False):
        ebm_channel_idx = ts_channel_names.index("_ebm_pred")
        logger.info(f'EBM channel "_ebm_pred" at index {ebm_channel_idx}/{len(ts_channel_names)}')

    # Store raw X for debugging
    X_raw = X.copy()

    # ============================================================================
    # FIXED: FIT SCALERS ON TRAINVAL ONLY, PRESERVING PADDING
    # ============================================================================
    logger.info("Fitting normalization scalers on trainval data (excluding padding)...")
    
    # 1. CONTINUOUS TIME SERIES SCALER
    ts_scaler = StandardScaler()

    # Get trajectory lengths (works with NaN for missing measurements)
    traj_lengths = get_trajectory_lengths(X, padding_value=0.0)
    logger.info(f'Trajectory lengths - min: {traj_lengths.min()}, max: {traj_lengths.max()}, '
               f'mean: {traj_lengths.mean():.1f}')

    # Per-channel normalization using trajectory_lengths + NaN awareness
    X_normalized = normalize_with_padding_mask(X, ts_scaler, traj_lengths, fit=True)

    # === TEMPORAL FEATURES: mode-aware index computation + elapsed_hours restoration ===
    # 'off'        (enabled: false)  → no temporal channels, no-op
    # 'channel'    (enabled: true, mode: channel)   → temporal features go through W_P
    #              as normalized inputs — no special treatment (Option A)
    # 'sinusoidal' (enabled: true, mode: sinusoidal) → elapsed_hours excluded from W_P,
    #              restored to raw hours for sinusoidal positional encoding (Option B)
    tf_cfg = cfg.get('temporal_features', {})
    tf_enabled = tf_cfg.get('enabled', False)
    tf_mode = tf_cfg.get('mode', 'channel')  # default to 'channel' when omitted

    temporal_channel_idx = None
    exclude_channel_indices = []

    if tf_enabled and tf_mode == 'sinusoidal':
        _aux_names = set(tf_cfg.get('features', []))   # e.g. {'elapsed_hours', 'bin_width_hours'}
        if 'elapsed_hours' in ts_channel_names:
            eh_idx = ts_channel_names.index('elapsed_hours')
            # Restore raw values: sinusoidal PE requires actual hours (0–720), not ~N(0,1)
            X_normalized[:, eh_idx, :] = X_raw[:, eh_idx, :]
            temporal_channel_idx = eh_idx
            logger.info(
                f'Temporal PE (sinusoidal): restored raw elapsed_hours at channel {eh_idx}'
            )
        else:
            logger.warning(
                "temporal_features.mode=sinusoidal but 'elapsed_hours' not found in channels; "
                "falling back to learned positional encoding."
            )
        exclude_channel_indices = [i for i, n in enumerate(ts_channel_names) if n in _aux_names]
        if exclude_channel_indices:
            excluded_names = [ts_channel_names[i] for i in exclude_channel_indices]
            logger.info(
                f'Temporal PE: excluding {excluded_names} (indices {exclude_channel_indices}) from W_P'
            )
    elif tf_enabled and tf_mode == 'channel':
        logger.info('Temporal features mode=channel: elapsed_hours/bin_width_hours go through W_P normally')

    if cfg.get('ebm_feature', {}).get('enabled', False):
        ebm_norm = X_normalized[:, ebm_channel_idx, :]
        ebm_nonzero = ebm_norm[ebm_norm != 0]
        if len(ebm_nonzero) > 0:
            logger.info(f'EBM channel after standardization: '
                        f'mean={ebm_nonzero.mean():.3f}, std={ebm_nonzero.std():.3f}, '
                        f'range=[{ebm_nonzero.min():.3f}, {ebm_nonzero.max():.3f}]')

    logger.info(f'Train/val X shape (after normalization): {X_normalized.shape}')
    
    # Verify padding is preserved (use trajectory_lengths, not zero-detection)
    s_len = X_normalized.shape[2]
    pos_arr = np.arange(s_len)[np.newaxis, :]
    is_padding = pos_arr >= traj_lengths[:, np.newaxis]  # [n_samples, seq_len]
    is_padding_3d = np.broadcast_to(is_padding[:, np.newaxis, :], X_normalized.shape)
    padding_vals = X_normalized[is_padding_3d]
    non_padding_vals = X_normalized[~is_padding_3d]
    logger.info(f'Padding verification:')
    logger.info(f'  Padding positions: {is_padding_3d.sum()}, non-padding: {(~is_padding_3d).sum()}')
    logger.info(f'  Padding values after norm - mean: {padding_vals.mean():.6f}, std: {padding_vals.std():.6f}')
    if np.abs(padding_vals.mean()) > 0.001:
        logger.warning(f'Padding was not preserved! Mean should be ~0, got {padding_vals.mean():.6f}')
    else:
        logger.info(f'Padding preserved correctly (mean = 0)')
    logger.info(f'Non-padding data stats: mean={non_padding_vals.mean():.4f}, std={non_padding_vals.std():.4f}')
    # Count measured vs missing within non-padding
    n_measured = np.sum(non_padding_vals != 0)
    n_missing = np.sum(non_padding_vals == 0)
    logger.info(f'  Within trajectory: {n_measured} measured ({100*n_measured/(n_measured+n_missing):.1f}%), '
               f'{n_missing} missing ({100*n_missing/(n_measured+n_missing):.1f}%)')
    
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
        data_cols=holdout.complete.columns[3:],
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
    
    # Holdout trajectory lengths (computed before normalization)
    holdout_traj_lengths = get_trajectory_lengths(tX, padding_value=0.0)
    logger.info(f'Holdout trajectory lengths - min: {holdout_traj_lengths.min()}, '
               f'max: {holdout_traj_lengths.max()}, mean: {holdout_traj_lengths.mean():.1f}')

    # Per-channel normalization using trainval-fitted scaler
    tX_normalized = normalize_with_padding_mask(tX, ts_scaler, holdout_traj_lengths, fit=False)

    # Restore raw elapsed_hours in holdout (same logic as trainval above)
    if tf_enabled and tf_mode == 'sinusoidal' and temporal_channel_idx is not None:
        tX_normalized[:, temporal_channel_idx, :] = tX_raw[:, temporal_channel_idx, :]

    if cfg.get('ebm_feature', {}).get('enabled', False):
        ebm_norm_h = tX_normalized[:, ebm_channel_idx, :]
        ebm_nz_h = ebm_norm_h[ebm_norm_h != 0]
        if len(ebm_nz_h) > 0:
            logger.info(f'Holdout EBM after standardization: '
                        f'mean={ebm_nz_h.mean():.3f}, std={ebm_nz_h.std():.3f}, '
                        f'range=[{ebm_nz_h.min():.3f}, {ebm_nz_h.max():.3f}]')
    
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
    # Padding must be zero
    pad_check = X_normalized[is_padding_3d]
    assert abs(pad_check.mean()) < 0.001, f"Padding not preserved! mean={pad_check.mean():.6f}"
    logger.info("Normalization validation passed — padding preserved correctly")
    
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
        "ts_feature_names": trainval.complete.columns[3:].tolist(),
        "ts_channel_names": ts_channel_names,        # sorted FEATURE names = channel order in X
        "trajectory_lengths": traj_lengths,
        "holdout_trajectory_lengths": holdout_traj_lengths,
        "ebm_channel_idx": ebm_channel_idx,
        "temporal_channel_idx": temporal_channel_idx,      # index of elapsed_hours, or None
        "exclude_channel_indices": exclude_channel_indices, # aux channel indices to skip in W_P
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

    traj_lens = get_trajectory_lengths(patient_ts_data, padding_value=0.0)
    ts_normalized = normalize_with_padding_mask(
        patient_ts_data,
        artifacts['ts_scaler'],
        traj_lens,
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
# DEPLOYMENT BUNDLE: Save/Load everything needed for standalone inference
# ============================================================================

def extract_shap_background(data, max_samples=200):
    """
    Extract background data tensors from training dataloader for SHAP.

    Returns dict with numpy arrays {ts, ts_cat, cat, cont} ready for
    later conversion to tensors.
    """
    all_ts, all_ts_cat, all_cat, all_cont = [], [], [], []
    n = 0
    for batch in data["mixed_dls"].train:
        if n >= max_samples:
            break
        inputs, _ = batch
        x_ts, x_tab, x_ts_cat = inputs[0], inputs[1], inputs[2]
        all_ts.append(x_ts.cpu().numpy())
        all_ts_cat.append(x_ts_cat.cpu().numpy())
        all_cat.append(x_tab[0].cpu().numpy())
        all_cont.append(x_tab[1].cpu().numpy())
        n += x_ts.shape[0]

    return {
        'ts': np.concatenate(all_ts)[:max_samples],
        'ts_cat': np.concatenate(all_ts_cat)[:max_samples],
        'cat': np.concatenate(all_cat)[:max_samples],
        'cont': np.concatenate(all_cont)[:max_samples],
    }


def _build_channel_map(ts_channel_names, cfg):
    """
    Build a definitive mapping from each channel name to its source.

    Resolves ambiguous names like BASE_EXCESS_max at save time when all
    config information is available.

    Returns:
        dict: {channel_name: {'concept': str, 'feature': str, 'agg_func': str|None, 'type': str}}
    """
    channel_map = {}

    # 1. Temporal features (elapsed_hours, bin_width_hours)
    temporal_features = cfg.get('temporal_features', {}).get('features', [])
    for ch in ts_channel_names:
        if ch in temporal_features:
            channel_map[ch] = {
                'concept': '_temporal', 'feature': ch,
                'agg_func': None, 'type': 'temporal',
            }

    # 2. EBM feature
    for ch in ts_channel_names:
        if ch == '_ebm_pred':
            channel_map[ch] = {
                'concept': '_ebm', 'feature': '_ebm_pred',
                'agg_func': None, 'type': 'ebm',
            }

    # 3. Continuous features: {FEATURE}_{agg_func}
    # For each non-categorical concept, try to match channel names
    ts_cat_names = cfg['dataset'].get('ts_cat_names', [])
    for concept in cfg['concepts']:
        if concept in ts_cat_names:
            continue
        for agg_func in cfg['agg_func'].get(concept, []):
            suffix = f'_{agg_func}'
            for ch in ts_channel_names:
                if ch in channel_map:
                    continue
                if ch.endswith(suffix):
                    raw_feature = ch[:-len(suffix)]
                    channel_map[ch] = {
                        'concept': concept,
                        'feature': raw_feature,
                        'agg_func': agg_func,
                        'type': 'continuous',
                    }

    # Warn about unmapped channels
    unmapped = [ch for ch in ts_channel_names if ch not in channel_map]
    if unmapped:
        logger.warning(f"Channel map: {len(unmapped)} unmapped channels: {unmapped}")

    return channel_map


def save_deployment_bundle(data, cfg, model_name, save_dir='models/deployment',
                           max_bg_samples=200):
    """
    Save all artifacts needed for standalone single-patient inference.

    Includes normalization scalers, model construction params, channel ordering,
    and pre-extracted SHAP background data.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Channel names computed in prepare_data_and_dls; reuse to stay consistent
    ts_channel_names = data.get('ts_channel_names') or sorted(
        data["trainval"].complete['FEATURE'].unique()
    )

    bundle = {
        # --- Normalization artifacts ---
        'ts_scaler': data['ts_scaler'],
        'tab_scaler': data['tab_scaler'],
        'encoding_info': data['encoding_info'],
        'cat_encoder': data['cat_encoder'],
        'tab_feature_names': data['num_cols'],
        'cat_feature_names': data['cat_cols'],
        'ts_channel_names': ts_channel_names,

        # --- Model construction params (replaces get_backbone + data dict) ---
        'model_params': {
            'c_in': data["ts_dls"].vars,
            'seq_len': data["mixed_dls"].len,
            'classes': {k: list(v) for k, v in data["classes"].items()},
            'cont_names': list(data["num_cols"]),
            'ts_cat_dims': dict(data["ts_cat_dls"].ts_cat_dims),
            'd_model': cfg["model"]["d_model"],
            'n_layers': cfg["model"]["n_layers"],
            'n_heads': cfg["model"]["n_heads"],
            'fc_dropout': cfg["model"]["fc_dropout"],
            'res_dropout': cfg["model"]["res_dropout"],
            'fc_mults': (cfg["model"]["fc_mults_1"], cfg["model"]["fc_mults_2"]),
            'temporal_head': cfg.get("model", {}).get("temporal_head", False),
            'causal': cfg.get("model", {}).get("causal", False),
            'temporal_head_dropout': cfg.get("model", {}).get("temporal_head_dropout", 0.3),
            'temporal_channel_idx': data.get('temporal_channel_idx', None),
            'exclude_channel_indices': data.get('exclude_channel_indices', []),
        },

        # --- SHAP background data ---
        'shap_background': extract_shap_background(data, max_bg_samples),

        # --- Data processing config (for inference data_prep) ---
        'data_config': {
            'bin_intervals': dict(cfg['bin_intervals']),
            'bin_freq_include': list(cfg['bin_freq_include']),
            'channel_map': _build_channel_map(ts_channel_names, cfg),
            'ts_cat_names': list(cfg['dataset'].get('ts_cat_names', [])),
            'temporal_features': cfg.get('temporal_features', {}),
        },

        # --- Metadata ---
        'model_name': model_name,
    }

    save_path = os.path.join(save_dir, f'deployment_{model_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)

    logger.info(f"Saved deployment bundle to {save_path}")
    return save_path


def load_deployment_bundle(model_name, load_dir='models/deployment'):
    """Load a saved deployment bundle."""
    load_path = os.path.join(load_dir, f'deployment_{model_name}.pkl')
    with open(load_path, 'rb') as f:
        bundle = pickle.load(f)
    logger.info(f"Loaded deployment bundle from {load_path}")
    return bundle