# dataloader.py

import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from astra.utils import get_base_df, align_dataframes
from astra.data.preprocessing import MultiHotCategoricalEncoder
from astra.data.datasets import TSDS, get_effective_cat_cols

from astra.data.mixed_dataloader import (
    df2xy_pure,
    TabularEncoder,
    AstraMixedDataset,
    AstraMixedDataLoader,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Adaptive Per-Channel Scaler
# ============================================================================

class AstraScaler:
    """Per-channel scaler supporting multiple normalization methods.

    Wraps sklearn transformers per-channel for 3D time series data
    [n_samples, n_channels, seq_len].  Pickle-serializable.

    Methods
    -------
    standard  – z-score (mean / std), original behaviour.
    quantile  – QuantileTransformer → normal or uniform output.
    power     – PowerTransformer (Yeo-Johnson).
    robust    – median / IQR scaling.
    adaptive  – auto-select per channel from skewness & kurtosis.
    """

    # Adaptive thresholds (matching diagnose_distributions.py logic)
    _SKEW_NORMAL = 0.5
    _SKEW_MODERATE = 2.0
    _KURT_NORMAL = 3.0
    _KURT_MODERATE = 7.0

    # Boundary concentration thresholds: override quantile → robust when
    # >= _BOUNDARY_MASS_THRESH of values cluster within _BOUNDARY_RANGE_FRAC
    # of the observed range at either boundary (e.g. SPO2 ceiling at 100).
    _BOUNDARY_RANGE_FRAC = 0.05
    _BOUNDARY_MASS_THRESH = 0.40

    def __init__(self, method='adaptive', n_quantiles=1000,
                 quantile_output='normal', clip_range=None,
                 boundary_range_frac=None, boundary_mass_thresh=None):
        self.method = method
        self.n_quantiles = n_quantiles
        self.quantile_output = quantile_output
        self.clip_range = clip_range  # e.g. (-3.0, 3.0) to clip normalized output
        if boundary_range_frac is not None:
            self._BOUNDARY_RANGE_FRAC = boundary_range_frac
        if boundary_mass_thresh is not None:
            self._BOUNDARY_MASS_THRESH = boundary_mass_thresh

        # Populated during fit
        self.channel_scalers_ = {}   # ch_idx → fitted object / dict
        self.channel_methods_ = {}   # ch_idx → method name used
        self.n_channels_ = None

        # Backward-compat attributes expected by inference pipeline
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_features_in_ = None

    # ------------------------------------------------------------------
    # Adaptive method selection
    # ------------------------------------------------------------------
    @staticmethod
    def _is_boundary_concentrated(values, range_frac=None, mass_thresh=None):
        """Detect ceiling/floor-bounded distributions.

        Returns True if >= *mass_thresh* of values cluster within
        *range_frac* of the observed range at either boundary.  This
        guards against quantile normalisation amplifying tiny raw
        changes (e.g. SPO2 95-100 → full N(0,1) spread).
        """
        if range_frac is None:
            range_frac = AstraScaler._BOUNDARY_RANGE_FRAC
        if mass_thresh is None:
            mass_thresh = AstraScaler._BOUNDARY_MASS_THRESH

        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        span = vmax - vmin
        if span == 0:
            return False
        zone = span * range_frac
        at_ceiling = np.nansum(values >= vmax - zone) / len(values)
        at_floor = np.nansum(values <= vmin + zone) / len(values)
        return float(max(at_ceiling, at_floor)) >= mass_thresh

    @staticmethod
    def _select_method(values, range_frac=None, mass_thresh=None):
        """Pick normalisation method from distribution shape."""
        s = abs(float(_skew(values, nan_policy='omit')))
        k = float(_kurtosis(values, nan_policy='omit'))  # excess
        if s < AstraScaler._SKEW_NORMAL and k < AstraScaler._KURT_NORMAL:
            return 'standard'
        elif s < AstraScaler._SKEW_MODERATE and k < AstraScaler._KURT_MODERATE:
            return 'power'
        # Check for boundary-concentrated distributions before defaulting
        # to quantile — quantile amplifies small changes in dense boundary
        # regions (e.g. SPO2 ceiling, GCS ceiling).
        if AstraScaler._is_boundary_concentrated(values, range_frac, mass_thresh):
            return 'robust'
        return 'quantile'

    # ------------------------------------------------------------------
    # Per-channel fit / transform
    # ------------------------------------------------------------------
    def fit_channel(self, ch_idx, values):
        """Fit one channel on its measured (non-NaN, non-padding) values."""
        if len(values) < 3:
            self.channel_methods_[ch_idx] = 'standard'
            self.channel_scalers_[ch_idx] = {'mean': 0.0, 'std': 1.0}
            return

        method = (self._select_method(
                      values, self._BOUNDARY_RANGE_FRAC, self._BOUNDARY_MASS_THRESH)
                  if self.method == 'adaptive' else self.method)
        self.channel_methods_[ch_idx] = method

        if method == 'standard':
            m, s = float(values.mean()), float(values.std())
            if s == 0 or np.isnan(s):
                s = 1.0
            self.channel_scalers_[ch_idx] = {'mean': m, 'std': s}

        elif method == 'quantile':
            nq = min(self.n_quantiles, len(values))
            qt = QuantileTransformer(
                n_quantiles=nq,
                output_distribution=self.quantile_output,
            )
            qt.fit(values.reshape(-1, 1))
            self.channel_scalers_[ch_idx] = qt

        elif method == 'power':
            pt = PowerTransformer(method='yeo-johnson')
            try:
                pt.fit(values.reshape(-1, 1))
                self.channel_scalers_[ch_idx] = pt
            except Exception:
                # Fallback: constant or pathological channel
                m, s = float(values.mean()), float(values.std())
                if s == 0 or np.isnan(s):
                    s = 1.0
                self.channel_scalers_[ch_idx] = {'mean': m, 'std': s}
                self.channel_methods_[ch_idx] = 'standard'

        elif method == 'robust':
            med = float(np.median(values))
            q75, q25 = float(np.percentile(values, 75)), float(np.percentile(values, 25))
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            self.channel_scalers_[ch_idx] = {'median': med, 'iqr': iqr}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def transform_channel(self, ch_idx, values):
        """Transform measured values for one channel.  Returns 1-D array."""
        sc = self.channel_scalers_[ch_idx]
        method = self.channel_methods_[ch_idx]

        if method == 'standard':
            out = (values - sc['mean']) / sc['std']
        elif method in ('quantile', 'power'):
            out = sc.transform(values.reshape(-1, 1)).ravel()
        elif method == 'robust':
            out = (values - sc['median']) / sc['iqr']
        else:
            raise ValueError(f"Unknown method for channel {ch_idx}: {method}")

        if self.clip_range is not None:
            out = np.clip(out, self.clip_range[0], self.clip_range[1])
        return out

    # ------------------------------------------------------------------
    # Populate backward-compat attributes after all channels are fitted
    # ------------------------------------------------------------------
    def _populate_compat_attrs(self, n_channels):
        """Set .mean_ / .scale_ arrays for inference pipeline compat."""
        self.n_channels_ = n_channels
        self.n_features_in_ = n_channels
        means = np.zeros(n_channels)
        scales = np.ones(n_channels)
        for ch in range(n_channels):
            method = self.channel_methods_.get(ch, 'standard')
            sc = self.channel_scalers_.get(ch)
            if sc is None:
                continue
            if method == 'standard':
                means[ch] = sc['mean']
                scales[ch] = sc['std']
            elif method == 'robust':
                means[ch] = sc['median']
                scales[ch] = sc['iqr']
            else:
                # quantile / power: output is ~N(0,1)
                means[ch] = 0.0
                scales[ch] = 1.0
        self.mean_ = means
        self.scale_ = scales
        self.var_ = scales ** 2

    def summary(self):
        """Return dict of method → count for logging."""
        from collections import Counter
        return dict(Counter(self.channel_methods_.values()))


class AdaptiveTabScaler:
    """Per-column adaptive tabular scaler.  Pickle-safe.

    Selects normalization method per column based on skewness/kurtosis
    (same thresholds as ``AstraScaler``).  Exposes ``.mean_`` for the
    inference pipeline's missing-value fill logic.
    """

    def __init__(self, num_cols, n_quantiles=1000, quantile_output='normal'):
        self.num_cols = list(num_cols)
        self.n_quantiles = n_quantiles
        self.quantile_output = quantile_output
        self.col_scalers_ = {}   # col → fitted sklearn scaler
        self.col_methods_ = {}   # col → method name
        self.mean_ = None        # backward-compat: column means for fill

    def fit(self, X, y=None):
        """Fit on a DataFrame (or ndarray) of numeric columns."""
        for i, col in enumerate(self.num_cols):
            vals = X[col].dropna().values.astype(float) if hasattr(X, '__getitem__') and isinstance(col, str) else X[:, i]
            vals = vals[~np.isnan(vals)]

            if len(vals) < 3:
                method = 'standard'
            else:
                method = AstraScaler._select_method(vals)
            self.col_methods_[col] = method

            if method == 'quantile':
                sc = QuantileTransformer(
                    n_quantiles=min(self.n_quantiles, len(vals)),
                    output_distribution=self.quantile_output,
                )
            elif method == 'power':
                sc = PowerTransformer(method='yeo-johnson')
            elif method == 'robust':
                sc = RobustScaler()
            else:
                sc = StandardScaler()

            sc.fit(vals.reshape(-1, 1))
            self.col_scalers_[col] = sc

        # Populate .mean_ for inference fill-value compat
        means = []
        for col in self.num_cols:
            if hasattr(X, '__getitem__') and isinstance(col, str):
                vals = X[col].dropna().values
            else:
                vals = X[:, self.num_cols.index(col)]
                vals = vals[~np.isnan(vals)]
            means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        self.mean_ = np.array(means)
        return self

    def transform(self, X):
        """Transform numeric columns.  Accepts DataFrame or ndarray."""
        if hasattr(X, 'values'):
            out = X.copy()
            for col in self.num_cols:
                vals = out[col].values.astype(np.float64).reshape(-1, 1)
                out[col] = self.col_scalers_[col].transform(vals).ravel()
            return out.values if hasattr(out, 'values') else out
        else:
            out = X.copy().astype(np.float64)
            for i, col in enumerate(self.num_cols):
                out[:, i] = self.col_scalers_[col].transform(
                    out[:, i].reshape(-1, 1)
                ).ravel()
            return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def _create_adaptive_tab_scaler(tab_df, num_cols, norm_cfg):
    """Create a per-column adaptive tabular scaler."""
    if not num_cols:
        return StandardScaler()

    nq = norm_cfg.get('n_quantiles', 1000)
    qout = norm_cfg.get('quantile_output', 'normal')
    scaler = AdaptiveTabScaler(num_cols, n_quantiles=nq, quantile_output=qout)

    for col in num_cols:
        vals = tab_df[col].dropna().values.astype(float)
        method = AstraScaler._select_method(vals) if len(vals) >= 3 else 'standard'
        logger.info(f"  Tabular {col}: {method} (skew={_skew(vals):.2f})")

    return scaler


# ============================================================================
# Masked Normalization Functions
# ============================================================================

def normalize_with_padding_mask(X, scaler, trajectory_lengths, fit=True):
    """
    Normalize time series data per-channel, using trajectory_lengths for padding
    and NaN for missing measurements.

    After normalization:
      - Measured values  → standardized per-channel (≈zero mean, unit variance)
      - Missing measurements within trajectory → 0.0
      - Padding beyond trajectory end         → 0.0

    Supports both legacy ``StandardScaler`` and the new ``AstraScaler``
    (per-channel adaptive normalization).

    Args:
        X: Array [n_samples, n_channels, seq_len]. May contain NaN for
           positions where no clinical measurement was recorded.
        scaler: ``AstraScaler`` or sklearn ``StandardScaler``.
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
    # AstraScaler path (per-channel adaptive)
    # ------------------------------------------------------------------
    if isinstance(scaler, AstraScaler):
        if fit:
            for ch in range(n_channels):
                vals = X[:, ch, :][measured_mask[:, ch, :]]
                scaler.fit_channel(ch, vals)
            scaler._populate_compat_attrs(n_channels)

            summary = scaler.summary()
            logger.info(f"Fitted AstraScaler ({scaler.method}): {summary}")
            for ch in range(n_channels):
                logger.debug(f"  ch {ch}: {scaler.channel_methods_.get(ch, '?')}")
            logger.info(f"  Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
            logger.info(f"  Scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

        X_normalized = np.zeros((n_samples, n_channels, seq_len), dtype=np.float64)
        for ch in range(n_channels):
            m = measured_mask[:, ch, :]
            if m.any():
                vals = X[:, ch, :][m]
                X_normalized[:, ch, :][m] = scaler.transform_channel(ch, vals)
        return X_normalized

    # ------------------------------------------------------------------
    # Legacy StandardScaler path (backward compat with old caches)
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

        logger.info("Fitted per-channel StandardScaler on measured data:")
        logger.info(f"  Mean range: [{means.min():.4f}, {means.max():.4f}]")
        logger.info(f"  Std range:  [{stds.min():.4f}, {stds.max():.4f}]")

    # normalise: only measured positions get values; rest stays 0.0
    X_normalized = np.zeros((n_samples, n_channels, seq_len), dtype=np.float64)

    for ch in range(n_channels):
        m = measured_mask[:, ch, :]
        if m.any():
            X_normalized[:, ch, :][m] = (
                (X[:, ch, :][m] - scaler.mean_[ch]) / scaler.scale_[ch]
            )

    return X_normalized


def get_trajectory_lengths(X, padding_value=0.0, exclude_channels=None):
    """
    Get the actual trajectory length for each sample (last timestep with data).

    A timestep is padding if ALL channels are either NaN or equal to
    padding_value.  This handles both legacy (all-zero padding) and the
    NaN-for-missing convention.

    Args:
        X: Array of shape [n_samples, n_channels, seq_len]
        padding_value: Value used for padding (typically 0.0)
        exclude_channels: Optional list of channel indices to ignore when
            determining trajectory length.  Used to prevent derived channels
            (e.g. EBM predictions) from artificially extending the trajectory
            beyond where clinical measurements exist.

    Returns:
        trajectory_lengths: Array [n_samples] with length of each trajectory
    """
    n_samples, n_channels, seq_len = X.shape

    # Exclude specified channels (e.g. EBM) so they cannot inflate trajectory length
    if exclude_channels:
        keep = [c for c in range(n_channels) if c not in exclude_channels]
        X = X[:, keep, :]

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
# Utility functions
# ============================================================================

def tscatdfwide2x(df_wide:pd.DataFrame, sample_col:str='PID', cat_col='FEATURE'):
    encoder = MultiHotCategoricalEncoder()
    X_multi_hot, encoding_info = encoder.fit_transform(
        df_wide,
        sample_col=sample_col,
        timestep_cols=df_wide.attrs["timestep_cols"],
        cat_col=cat_col,
        feature_names=df_wide.FEATURE.dropna().unique()
    )
    return X_multi_hot, encoding_info


def encode_categorical_ts(df_wide, y, cfg, encoder=None):
    """Encode categorical TS to multi-hot arrays with optional pre-fitted encoder.

    Returns:
        X_multi_hot, encoding_info, ts_cat_dims, encoder
    """
    if encoder is None:
        encoder = MultiHotCategoricalEncoder()
        X_multi_hot, encoding_info = encoder.fit_transform(
            df_wide,
            sample_col='PID',
            timestep_cols=df_wide.attrs["timestep_cols"],
            cat_col='FEATURE',
            feature_names=df_wide.FEATURE.dropna().unique()
        )
    else:
        X_multi_hot, encoding_info = encoder.transform(
            df_wide,
            sample_col='PID',
            timestep_cols=df_wide.attrs["timestep_cols"],
            cat_col='FEATURE'
        )

    logger.debug(f"X_multi_hot shape: {X_multi_hot.shape}")

    ts_cat_dims = {
        feat_name: end - start
        for feat_name, (start, end) in encoding_info['feature_ranges'].items()
    }

    zero_feats = [k for k, v in ts_cat_dims.items() if v == 0]
    if zero_feats:
        logger.warning(
            f"Categorical TS features with 0 classes (no observed codes): {zero_feats}"
        )

    return X_multi_hot, encoding_info, ts_cat_dims, encoder


def _extract_profile_tensors(tsds, final_pids, cfg):
    """Extract and re-index profile tensors from a TSDS to match final PID order.

    Args:
        tsds: TSDS instance with ``_profile_data`` populated by ``collect_concepts()``.
        final_pids: Sorted list of PIDs matching the final X/y ordering.
        cfg: Config dict.

    Returns:
        profiles: np.ndarray [n_samples, total_profiled_categories, seq_len] (int8)
                  or None if profiles are disabled / no profiled categories.
        profile_dims: Dict {category_name: n_levels} across all concepts, or None.
        category_order: List of profiled category names (matches tensor dim 1 order), or None.
    """
    from astra.data.profiles import profiles_enabled

    if not profiles_enabled(cfg) or not hasattr(tsds, '_profile_data') or not tsds._profile_data:
        return None, None, None

    all_profile_arrays = []
    all_profile_dims = {}
    all_category_order = []

    for concept, (profile_array, profile_dims, category_order, original_pids) in tsds._profile_data.items():
        # Re-index profile_array rows to match final_pids
        pid_to_orig_idx = {pid: idx for idx, pid in enumerate(original_pids)}
        reindexed = np.zeros(
            (len(final_pids), profile_array.shape[1], profile_array.shape[2]),
            dtype=np.int8,
        )
        for new_idx, pid in enumerate(final_pids):
            if pid in pid_to_orig_idx:
                reindexed[new_idx] = profile_array[pid_to_orig_idx[pid]]

        all_profile_arrays.append(reindexed)
        all_profile_dims.update(profile_dims)
        all_category_order.extend(category_order)

    if not all_profile_arrays:
        return None, None, None

    # Concatenate along category dimension (dim 1) across concepts
    profiles = np.concatenate(all_profile_arrays, axis=1)
    logger.info(f"Profile tensor shape: {profiles.shape} (dims: {all_profile_dims})")

    return profiles, all_profile_dims, all_category_order


# ============================================================================
# Main data preparation function
# ============================================================================

def prepare_data_and_dls(cfg):
    """
    Prepare data and dataloaders (pure PyTorch, no TSAI/FastAI).

    Uses normalize_with_padding_mask() to ensure:
    - Scaler is fit only on non-padding (real) data
    - Padding zeros remain as zeros after normalization
    - Model correctly distinguishes signal from padding
    """
    # Load dataframes — exclusion criteria are applied inside TSDS.__init__
    base = get_base_df()

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
        tsds.complete_cat.attrs["timestep_cols"] = tsds.timestep_cols

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

    # Inject tier mapping features from categorical profiles
    from astra.data.profiles import profiles_enabled, load_profiles_config, CategoricalProfileEncoder
    if profiles_enabled(cfg):
        profiles_cfg = load_profiles_config(cfg)
        for concept_name, concept_profile_cfg in profiles_cfg.items():
            if concept_name in ('version',) or not isinstance(concept_profile_cfg, dict):
                continue
            encoder = CategoricalProfileEncoder(concept_profile_cfg)
            if encoder.tier_categories:
                concept_pkl = f"data/interim/concepts/{concept_name}.pkl"
                logger.info(f"Injecting tier mapping features for {concept_name}...")
                for tsds in [trainval, holdout]:
                    tier_df = encoder.compute_tier_features(
                        concept_pkl, tsds.base, cfg
                    )
                    if tier_df is not None:
                        key = f'_tier_{concept_name.lower()}'
                        tsds.cont_concepts[key] = tier_df
                        tsds.complete = pd.concat(tsds.cont_concepts)
                logger.info(
                    f"Injected tier features for {concept_name}: "
                    f"{list(encoder.tier_categories.keys())}"
                )

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
        tsds_obj.complete_cat.attrs["timestep_cols"] = ts

    cat_cols = get_effective_cat_cols(cfg)
    num_cols = cfg["dataset"]["num_cols"]
    logger.info(f'Categoricals: {cat_cols}\nNumericals: {num_cols}')

    # ============================================================================
    # TABULAR ENCODING (replaces FastAI Categorify + FillMissing)
    # ============================================================================
    tab_encoder = TabularEncoder()
    tab_encoder.fit(
        pd.concat([trainval.tab_df, holdout.tab_df]),
        cat_cols=cat_cols,
        num_cols=num_cols,
    )
    classes = tab_encoder.classes

    # ============================================================================
    # TRAINVAL DATA EXTRACTION
    # ============================================================================
    logger.info("Setting up X,y for training and validation")
    X, y = df2xy_pure(
        trainval.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=trainval.complete.columns[3:],
        target_col=cfg["target"]
    )
    y = list(y[:, 0].flatten())
    logger.info(f'Train/val X shape (before normalization): {X.shape}')

    # Extract survival labels (event_time_steps, event_indicator) aligned with PID order
    # df2xy_pure sorts by [PID, FEATURE], so samples = sorted unique PIDs
    survival_mode = cfg.get('model', {}).get('survival_mode', False)
    trainval_event_times = None
    trainval_event_indicators = None
    if survival_mode:
        _tv_sorted_pids = sorted(trainval.complete['PID'].unique())
        _tv_surv = trainval.base.set_index('PID').loc[_tv_sorted_pids]
        trainval_event_times = _tv_surv['event_time_steps'].values.astype(int)
        trainval_event_indicators = _tv_surv['event_indicator'].values.astype(int)
        logger.info(
            f"Survival labels (trainval): {trainval_event_indicators.sum()} events, "
            f"{len(trainval_event_indicators) - trainval_event_indicators.sum()} censored"
        )

    # Channel names — df2xy_pure sorts by FEATURE ascending, so this IS the channel order.
    ts_channel_names = sorted(trainval.complete['FEATURE'].unique())

    # Compute EBM channel index
    if cfg.get('ebm_feature', {}).get('enabled', False):
        ebm_channel_idx = ts_channel_names.index("_ebm_pred")
        logger.info(f'EBM channel "_ebm_pred" at index {ebm_channel_idx}/{len(ts_channel_names)}')

    # Store raw X for debugging
    X_raw = X.copy()

    # ============================================================================
    # FIT SCALERS ON TRAINVAL ONLY, PRESERVING PADDING
    # ============================================================================
    logger.info("Fitting normalization scalers on trainval data (excluding padding)...")

    # 1. CONTINUOUS TIME SERIES SCALER
    norm_cfg = cfg.get('normalization', {})
    ts_method = norm_cfg.get('ts_method', 'standard')
    clip_range_cfg = norm_cfg.get('clip_range', None)
    clip_range = tuple(clip_range_cfg) if clip_range_cfg else None
    ts_scaler = AstraScaler(
        method=ts_method,
        n_quantiles=norm_cfg.get('n_quantiles', 1000),
        quantile_output=norm_cfg.get('quantile_output', 'normal'),
        clip_range=clip_range,
        boundary_range_frac=norm_cfg.get('boundary_range_frac'),
        boundary_mass_thresh=norm_cfg.get('boundary_mass_thresh'),
    )
    logger.info(f"Using {ts_method} normalization for time series"
                + (f" with clip_range={clip_range}" if clip_range else ""))

    # Get trajectory lengths (works with NaN for missing measurements).
    # Exclude non-clinical channels (EBM, temporal features) whose values
    # are non-zero everywhere and would inflate trajectory length detection.
    ebm_enabled = cfg.get('ebm_feature', {}).get('enabled', False)
    traj_exclude_chs = [ebm_channel_idx] if ebm_enabled else []
    tf_cfg = cfg.get('temporal_features', {})
    if tf_cfg.get('enabled', False):
        _tf_names = set(tf_cfg.get('features', []))
        for i, name in enumerate(ts_channel_names):
            if name in _tf_names:
                traj_exclude_chs.append(i)
    # Exclude tier mapping features from trajectory length detection
    from astra.data.profiles import get_tier_feature_names
    _tier_names = get_tier_feature_names(cfg)
    if _tier_names:
        for i, name in enumerate(ts_channel_names):
            if name in _tier_names:
                traj_exclude_chs.append(i)
    traj_exclude_chs = traj_exclude_chs or None
    traj_lengths = get_trajectory_lengths(X, padding_value=0.0, exclude_channels=traj_exclude_chs)
    logger.info(f'Trajectory lengths - min: {traj_lengths.min()}, max: {traj_lengths.max()}, '
               f'mean: {traj_lengths.mean():.1f}')

    # --- Filter samples with trajectory shorter than min_bin_seq_len ----------
    from astra.data.datasets import resolve_exclusion_criteria
    _excl = resolve_exclusion_criteria(cfg) or {}
    min_seq_len = _excl.get('min_bin_seq_len',
                            cfg.get('dataset', {}).get('min_bin_seq_len', 0))
    if min_seq_len > 0:
        keep_mask = traj_lengths >= min_seq_len
        n_short = (~keep_mask).sum()
        if n_short:
            sorted_pids = sorted(trainval.complete['PID'].unique())
            drop_pids = {sorted_pids[i] for i in range(len(sorted_pids))
                         if not keep_mask[i]}
            logger.info(f'Dropping {n_short} trainval samples with trajectory '
                        f'< {min_seq_len} steps (PIDs: {len(drop_pids)})')
            X = X[keep_mask]
            X_raw = X_raw[keep_mask]
            y = [y[i] for i in range(len(y)) if keep_mask[i]]
            traj_lengths = traj_lengths[keep_mask]
            if trainval_event_times is not None:
                trainval_event_times = trainval_event_times[keep_mask]
                trainval_event_indicators = trainval_event_indicators[keep_mask]
            id_col = cfg['dataset']['id_col']
            trainval.base = trainval.base[
                ~trainval.base[id_col].isin(drop_pids)
            ].reset_index(drop=True)
            trainval._base_pids -= drop_pids
            trainval.tab_df = trainval.tab_df[
                ~trainval.tab_df[id_col].isin(drop_pids)
            ].reset_index(drop=True)
            trainval.complete = trainval.complete[
                ~trainval.complete[id_col].isin(drop_pids)
            ].reset_index(drop=True)
            trainval.complete_cat = trainval.complete_cat[
                ~trainval.complete_cat[id_col].isin(drop_pids)
            ].reset_index(drop=True)

    # Per-channel normalization using trajectory_lengths + NaN awareness
    X_normalized = normalize_with_padding_mask(X, ts_scaler, traj_lengths, fit=True)

    # === TEMPORAL FEATURES: mode-aware index computation + elapsed_hours restoration ===
    # tf_cfg already assigned above for trajectory length exclusion
    tf_enabled = tf_cfg.get('enabled', False)
    tf_mode = tf_cfg.get('mode', 'channel')

    temporal_channel_idx = None
    bin_width_channel_idx = None
    exclude_channel_indices = []

    if tf_enabled and tf_mode == 'sinusoidal':
        _aux_names = set(tf_cfg.get('features', []))
        if 'elapsed_hours' in ts_channel_names:
            eh_idx = ts_channel_names.index('elapsed_hours')
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
        if 'bin_width_hours' in ts_channel_names:
            bw_idx = ts_channel_names.index('bin_width_hours')
            X_normalized[:, bw_idx, :] = X_raw[:, bw_idx, :]
            bin_width_channel_idx = bw_idx
            logger.info(
                f'Temporal PE (sinusoidal): restored raw bin_width_hours at channel {bw_idx}'
            )
        exclude_channel_indices = [i for i, n in enumerate(ts_channel_names) if n in _aux_names]
        if exclude_channel_indices:
            excluded_names = [ts_channel_names[i] for i in exclude_channel_indices]
            logger.info(
                f'Temporal PE: excluding {excluded_names} (indices {exclude_channel_indices}) from W_P'
            )
        # Re-zero padding positions in restored temporal channels.
        # normalize_with_padding_mask already zeroed them, but restoring raw
        # values above re-introduced non-zero time data in padding positions.
        _s_len = X_normalized.shape[2]
        _pos = np.arange(_s_len)[np.newaxis, :]
        _beyond = _pos >= traj_lengths[:, np.newaxis]  # [n_samples, seq_len]
        for ch_idx in exclude_channel_indices:
            X_normalized[:, ch_idx, :][_beyond] = 0.0
        logger.info(f'Zeroed temporal features in {_beyond.sum()} padding positions')
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

    # Verify padding is preserved
    s_len = X_normalized.shape[2]
    pos_arr = np.arange(s_len)[np.newaxis, :]
    is_padding = pos_arr >= traj_lengths[:, np.newaxis]
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
    n_measured = np.sum(non_padding_vals != 0)
    n_missing = np.sum(non_padding_vals == 0)
    logger.info(f'  Within trajectory: {n_measured} measured ({100*n_measured/(n_measured+n_missing):.1f}%), '
               f'{n_missing} missing ({100*n_missing/(n_measured+n_missing):.1f}%)')

    # 2. TABULAR DATA SCALER
    tab_method = norm_cfg.get('tab_method', 'standard')
    if tab_method == 'quantile':
        nq = norm_cfg.get('n_quantiles', 1000)
        qout = norm_cfg.get('quantile_output', 'normal')
        tab_scaler = QuantileTransformer(n_quantiles=nq, output_distribution=qout)
    elif tab_method == 'power':
        tab_scaler = PowerTransformer(method='yeo-johnson')
    elif tab_method == 'robust':
        tab_scaler = RobustScaler()
    elif tab_method == 'adaptive':
        tab_scaler = _create_adaptive_tab_scaler(trainval.tab_df, num_cols, norm_cfg)
    else:
        tab_scaler = StandardScaler()
    logger.info(f"Using {tab_method} normalization for tabular features")

    if num_cols:
        logger.info(f'Fitting tabular scaler on {len(num_cols)} continuous features')
        tab_scaler.fit(trainval.tab_df[num_cols])
        trainval_tab_normalized = trainval.tab_df.copy()
        trainval_tab_normalized[num_cols] = tab_scaler.transform(trainval.tab_df[num_cols])
    else:
        trainval_tab_normalized = trainval.tab_df

    # ============================================================================
    # TRAINVAL: Encode tabular + categorical TS → build dataloaders
    # ============================================================================
    logger.info("Creating trainval dataloaders...")

    trainval_tab_encoded = tab_encoder.transform(trainval_tab_normalized, cat_cols, num_cols)
    trainval_x_cat, trainval_x_cont = tab_encoder.get_cat_cont_arrays(
        trainval_tab_encoded, cat_cols, num_cols,
    )

    X_multi_hot, encoding_info, ts_cat_dims, cat_encoder = encode_categorical_ts(
        trainval.complete_cat, y, cfg, encoder=None,
    )

    # Extract and align profile tensors from TSDS (if profiles enabled)
    trainval_profiles, ts_cat_profile_dims, profile_category_order = _extract_profile_tensors(
        trainval, sorted(trainval.complete_cat['PID'].unique()), cfg
    )

    trainval_dataset = AstraMixedDataset(
        X_ts=X_normalized,
        x_cat=trainval_x_cat,
        x_cont=trainval_x_cont,
        X_ts_cat=X_multi_hot,
        y=y,
        trajectory_lengths=traj_lengths,
        event_times=trainval_event_times,
        event_indicators=trainval_event_indicators,
        X_ts_cat_profiles=trainval_profiles,
    )
    mixed_dls = AstraMixedDataLoader(
        trainval_dataset,
        splits=None,
        bs=cfg["training"]["bs"],
        shuffle_train=False,
    )

    # ============================================================================
    # HOLDOUT DATA EXTRACTION
    # ============================================================================
    logger.info('Preparing holdout data...')
    tX, ty = df2xy_pure(
        holdout.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=holdout.complete.columns[3:],
        target_col=holdout.target
    )
    ty = list(ty[:, 0].flatten())
    logger.info(f'Holdout X shape (before normalization): {tX.shape}')

    # Holdout survival labels
    holdout_event_times = None
    holdout_event_indicators = None
    if survival_mode:
        _ho_sorted_pids = sorted(holdout.complete['PID'].unique())
        _ho_surv = holdout.base.set_index('PID').loc[_ho_sorted_pids]
        holdout_event_times = _ho_surv['event_time_steps'].values.astype(int)
        holdout_event_indicators = _ho_surv['event_indicator'].values.astype(int)
        logger.info(
            f"Survival labels (holdout): {holdout_event_indicators.sum()} events, "
            f"{len(holdout_event_indicators) - holdout_event_indicators.sum()} censored"
        )

    tX_raw = tX.copy()

    # ============================================================================
    # TRANSFORM HOLDOUT WITH FITTED SCALERS, PRESERVING PADDING
    # ============================================================================
    logger.info("Applying normalization to holdout (preserving padding)...")

    holdout_traj_lengths = get_trajectory_lengths(tX, padding_value=0.0, exclude_channels=traj_exclude_chs)
    logger.info(f'Holdout trajectory lengths - min: {holdout_traj_lengths.min()}, '
               f'max: {holdout_traj_lengths.max()}, mean: {holdout_traj_lengths.mean():.1f}')

    # --- Filter holdout samples with trajectory shorter than min_bin_seq_len --
    if min_seq_len > 0:
        keep_mask_h = holdout_traj_lengths >= min_seq_len
        n_short_h = (~keep_mask_h).sum()
        if n_short_h:
            sorted_pids_h = sorted(holdout.complete['PID'].unique())
            drop_pids_h = {sorted_pids_h[i] for i in range(len(sorted_pids_h))
                           if not keep_mask_h[i]}
            logger.info(f'Dropping {n_short_h} holdout samples with trajectory '
                        f'< {min_seq_len} steps (PIDs: {len(drop_pids_h)})')
            tX = tX[keep_mask_h]
            tX_raw = tX_raw[keep_mask_h]
            ty = [ty[i] for i in range(len(ty)) if keep_mask_h[i]]
            holdout_traj_lengths = holdout_traj_lengths[keep_mask_h]
            if holdout_event_times is not None:
                holdout_event_times = holdout_event_times[keep_mask_h]
                holdout_event_indicators = holdout_event_indicators[keep_mask_h]
            id_col = cfg['dataset']['id_col']
            holdout.base = holdout.base[
                ~holdout.base[id_col].isin(drop_pids_h)
            ].reset_index(drop=True)
            holdout._base_pids -= drop_pids_h
            holdout.tab_df = holdout.tab_df[
                ~holdout.tab_df[id_col].isin(drop_pids_h)
            ].reset_index(drop=True)
            holdout.complete = holdout.complete[
                ~holdout.complete[id_col].isin(drop_pids_h)
            ].reset_index(drop=True)
            holdout.complete_cat = holdout.complete_cat[
                ~holdout.complete_cat[id_col].isin(drop_pids_h)
            ].reset_index(drop=True)

    tX_normalized = normalize_with_padding_mask(tX, ts_scaler, holdout_traj_lengths, fit=False)

    if tf_enabled and tf_mode == 'sinusoidal':
        if temporal_channel_idx is not None:
            tX_normalized[:, temporal_channel_idx, :] = tX_raw[:, temporal_channel_idx, :]
        if bin_width_channel_idx is not None:
            tX_normalized[:, bin_width_channel_idx, :] = tX_raw[:, bin_width_channel_idx, :]
        # Re-zero padding in restored temporal channels
        _h_slen = tX_normalized.shape[2]
        _h_pos = np.arange(_h_slen)[np.newaxis, :]
        _h_beyond = _h_pos >= holdout_traj_lengths[:, np.newaxis]
        for ch_idx in exclude_channel_indices:
            tX_normalized[:, ch_idx, :][_h_beyond] = 0.0

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
    # HOLDOUT: Encode tabular + categorical TS → build dataloaders
    # ============================================================================
    logger.info("Creating holdout dataloaders...")

    holdout_tab_encoded = tab_encoder.transform(holdout_tab_normalized, cat_cols, num_cols)
    holdout_x_cat, holdout_x_cont = tab_encoder.get_cat_cont_arrays(
        holdout_tab_encoded, cat_cols, num_cols,
    )

    tX_multi_hot, holdout_encoding_info, _, _ = encode_categorical_ts(
        holdout.complete_cat, ty, cfg, encoder=cat_encoder,
    )

    holdout_profiles, _, _ = _extract_profile_tensors(
        holdout, sorted(holdout.complete_cat['PID'].unique()), cfg
    )

    holdout_dataset = AstraMixedDataset(
        X_ts=tX_normalized,
        x_cat=holdout_x_cat,
        x_cont=holdout_x_cont,
        X_ts_cat=tX_multi_hot,
        y=ty,
        trajectory_lengths=holdout_traj_lengths,
        event_times=holdout_event_times,
        event_indicators=holdout_event_indicators,
        X_ts_cat_profiles=holdout_profiles,
    )
    holdout_mixed_dls = AstraMixedDataLoader(
        holdout_dataset,
        splits=None,
        bs=cfg["training"]["bs"],
        shuffle_train=False,
    )

    # ============================================================================
    # VALIDATION
    # ============================================================================
    pad_check = X_normalized[is_padding_3d]
    assert abs(pad_check.mean()) < 0.001, f"Padding not preserved! mean={pad_check.mean():.6f}"
    logger.info("Normalization validation passed — padding preserved correctly")

    # ============================================================================
    # RETURN
    # ============================================================================
    c_in = X_normalized.shape[1]
    seq_len = X_normalized.shape[2]

    # Validate seq_len matches config-derived total
    from astra.evaluation.utils import get_total_steps
    expected_steps = get_total_steps()
    if seq_len != expected_steps:
        logger.warning(
            f"seq_len from data ({seq_len}) != config-derived total ({expected_steps}). "
            f"Re-run 'make data' to regenerate bin_df."
        )

    return {
        "base": base,
        "trainval": trainval,
        "holdout": holdout,
        "X": X_normalized,
        "X_raw": X_raw,
        "X_multi_hot": X_multi_hot,
        "y": y,
        "tX": tX_normalized,
        "tX_raw": tX_raw,
        "tX_multi_hot": tX_multi_hot,
        "ty": ty,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "classes": classes,
        "mixed_dls": mixed_dls,
        "holdout_mixed_dls": holdout_mixed_dls,
        "encoding_info": encoding_info,
        "cat_encoder": cat_encoder,
        "tab_encoder": tab_encoder,
        "ts_scaler": ts_scaler,
        "tab_scaler": tab_scaler,
        "ts_feature_names": trainval.complete.columns[3:].tolist(),
        "ts_channel_names": ts_channel_names,
        "trajectory_lengths": traj_lengths,
        "holdout_trajectory_lengths": holdout_traj_lengths,
        "ebm_channel_idx": ebm_channel_idx,
        "temporal_channel_idx": temporal_channel_idx,
        "bin_width_channel_idx": bin_width_channel_idx,
        "exclude_channel_indices": exclude_channel_indices,
        # Survival labels (None when survival_mode is disabled)
        "event_times": trainval_event_times,
        "event_indicators": trainval_event_indicators,
        "holdout_event_times": holdout_event_times,
        "holdout_event_indicators": holdout_event_indicators,
        "survival_mode": survival_mode,
        # Explicit scalars (replace TSAI DL attributes)
        "c_in": c_in,
        "seq_len": seq_len,
        "ts_cat_dims": ts_cat_dims,
        # Profile-based categorical TS (None when profiles disabled)
        "ts_cat_profile_dims": ts_cat_profile_dims,
        "profile_category_order": profile_category_order,
        "X_ts_cat_profiles": trainval_profiles,
        "tX_ts_cat_profiles": holdout_profiles,
    }


# ============================================================================
# UTILITY: Save/Load (unchanged)
# ============================================================================

def save_normalization_artifacts(data, model_name, save_dir='models/scalers'):
    """Save normalization scalers and metadata for deployment."""
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


def normalize_new_patient(patient_ts_data, patient_tab_data, artifacts, exclude_channels=None):
    """
    Apply saved normalization to new patient data, preserving padding.
    """
    if patient_ts_data.ndim == 2:
        patient_ts_data = patient_ts_data[np.newaxis, ...]

    traj_lens = get_trajectory_lengths(patient_ts_data, padding_value=0.0, exclude_channels=exclude_channels)
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
    """
    channel_map = {}

    temporal_features = cfg.get('temporal_features', {}).get('features', [])
    for ch in ts_channel_names:
        if ch in temporal_features:
            channel_map[ch] = {
                'concept': '_temporal', 'feature': ch,
                'agg_func': None, 'type': 'temporal',
            }

    for ch in ts_channel_names:
        if ch == '_ebm_pred':
            channel_map[ch] = {
                'concept': '_ebm', 'feature': '_ebm_pred',
                'agg_func': None, 'type': 'ebm',
            }

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

    unmapped = [ch for ch in ts_channel_names if ch not in channel_map]
    if unmapped:
        logger.warning(f"Channel map: {len(unmapped)} unmapped channels: {unmapped}")

    return channel_map


def save_deployment_bundle(data, cfg, model_name, save_dir='models/deployment',
                           max_bg_samples=200):
    """
    Save all artifacts needed for standalone single-patient inference.
    """
    os.makedirs(save_dir, exist_ok=True)

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

        # --- Model construction params ---
        'model_params': {
            'c_in': data["c_in"],
            'seq_len': data["seq_len"],
            'classes': {k: list(v) for k, v in data["classes"].items()},
            'cont_names': list(data["num_cols"]),
            'ts_cat_dims': dict(data["ts_cat_dims"]),
            'd_model': cfg["model"]["d_model"],
            'n_layers': cfg["model"]["n_layers"],
            'n_heads': cfg["model"]["n_heads"],
            'fc_dropout': cfg["model"]["fc_dropout"],
            'res_dropout': cfg["model"]["res_dropout"],
            'fc_mults': (cfg["model"]["fc_mults_1"], cfg["model"]["fc_mults_2"]),
            'temporal_head': cfg.get("model", {}).get("temporal_head", False),
            'causal': cfg.get("model", {}).get("causal", False),
            'temporal_head_dropout': cfg.get("model", {}).get("temporal_head_dropout", 0.3),
            'temporal_head_mult': cfg.get("model", {}).get("temporal_head_mult", 0.5),
            'temporal_channel_idx': data.get('temporal_channel_idx', None),
            'bin_width_channel_idx': data.get('bin_width_channel_idx', None),
            'exclude_channel_indices': data.get('exclude_channel_indices', []),
            'head_pool': cfg.get("model", {}).get("head_pool", "flatten"),
            'per_feature_cont_proj': cfg.get("model", {}).get("per_feature_cont_proj", False),
            'cat_ts_gate': cfg.get("model", {}).get("cat_ts_gate", False),
            'local_temporal_kernel': cfg.get("model", {}).get("local_temporal_kernel", 1),
            'bin_width_modulation': cfg.get("model", {}).get("bin_width_modulation", False),
            'survival_mode': cfg.get("model", {}).get("survival_mode", False),
            'ts_cat_profile_dims': data.get('ts_cat_profile_dims'),
        },

        # --- SHAP background data ---
        'shap_background': extract_shap_background(data, max_bg_samples),

        # --- Data processing config (for inference data_prep) ---
        'data_config': {
            'bin_intervals': dict(cfg['bin_intervals']),
            'bin_freq_include': list(cfg['bin_freq_include']),
            'channel_map': _build_channel_map(ts_channel_names, cfg),
            'ts_cat_names': list(cfg['dataset'].get('ts_cat_names', [])),
            'cat_encoder_names': dict(cfg['dataset'].get('cat_encoder_names', {})),
            'concepts': list(cfg.get('concepts', [])),
            'temporal_features': cfg.get('temporal_features', {}),
            'ebm_channel_idx': data.get('ebm_channel_idx'),
            'categorical_profiles': cfg.get('categorical_profiles', {}),
        },

        # --- Profile-based categorical TS ---
        'ts_cat_profile_dims': data.get('ts_cat_profile_dims'),
        'profile_category_order': data.get('profile_category_order'),

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
