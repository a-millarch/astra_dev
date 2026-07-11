"""
Single-patient data preparation for ASTRA inference.

Two entry points:

1. prepare_single_patient(raw_data, bundle)
   Expects already-standardized feature names (HR, SBP, LACTATE, ...).

2. prepare_from_raw_ehr(raw_ehr, bundle)
   Accepts raw hospital EHR data (Danish names, ATC codes, procedure codes)
   and applies all mapping/filtering logic from filters.py + build_patient_info.py.

Usage:
    from astra.inference.data_prep import prepare_from_raw_ehr

    result = prepare_from_raw_ehr(raw_ehr, bundle)
    session.predict(**result)
"""

import os
import time
from contextlib import contextmanager, nullcontext as _nullcontext

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


# ============================================================================
# Timing instrumentation
# ============================================================================

@contextmanager
def timed_stage(timing_dict: dict, stage_name: str):
    """Record wall-clock seconds for a code block into *timing_dict*.

    Usage::

        timing = {}
        with timed_stage(timing, 'continuous_build'):
            x_ts = _build_continuous_ts(...)
        # timing == {'continuous_build': [0.0123]}
    """
    start = time.perf_counter()
    yield
    timing_dict.setdefault(stage_name, []).append(time.perf_counter() - start)

from astra.data.mappings import (
    VITALS_MAP, VITALS_BOUNDS, BP_TYPES, HEIGHT_WEIGHT_MAP, LABS_REVERSE_MAP, ICU_MAP, EWS_MAP,
    ATC_LVL3_REVERSE, ATC_LVL4_REVERSE,
    INVASIVE_BP_TYPES, INVASIVE_VITALS_MAP,
    PROCEDURE_MAP, PROCEDURE_PREFIXES, SEX_MAP,
    classify_department, classify_atc, derive_first_hospital, parse_numeric,
)


# Keys in raw_data that are metadata, not concept event lists.
_RAW_DATA_META_KEYS = frozenset({
    'pid', 'admission_time', 'current_time', 'demographics',
})


# ============================================================================
# Time binning
# ============================================================================

def _max_prediction_window(data_config: dict) -> pd.Timedelta:
    """Derive the maximum prediction window from bin_intervals config.

    Returns the largest named interval boundary (excluding the open-ended
    'end' key).  E.g. for ``{6h: 10min, ..., 90D: 7D, end: 30D}`` → 90 days.
    """
    max_td = pd.Timedelta(0)
    for key in data_config.get('bin_intervals', {}):
        if key == 'end':
            continue
        td = pd.Timedelta(key)
        if td > max_td:
            max_td = td
    if max_td == pd.Timedelta(0):
        max_td = pd.Timedelta(days=30)  # safe fallback
    return max_td


def _create_patient_bins(
    admission_time: pd.Timestamp,
    data_config: dict,
) -> pd.DataFrame:
    """
    Create the full fixed-duration bin grid for a single patient trajectory.

    Creates bins spanning [admission_time, admission_time + max_window] where
    max_window is derived from the largest named interval in bin_intervals.
    This makes the bin grid stable across re-inferences: position N always
    maps to the same time window regardless of when inference is called.

    Use ``time_to_step()`` from ``astra.evaluation.utils`` to convert
    elapsed time to a step index on this grid.

    Returns:
        DataFrame with columns [bin_start, bin_end, bin_counter, bin_freq, position]
        where position is the 0-based contiguous index after frequency filtering.
    """
    bin_intervals = data_config['bin_intervals']
    bin_freq_include = data_config['bin_freq_include']

    start_time = admission_time
    end_time = admission_time + _max_prediction_window(data_config)

    current = start_time
    bin_counter = 1
    bin_list = []

    for interval, freq in bin_intervals.items():
        if current >= end_time:
            break

        if interval == "end":
            interval_end = end_time
        else:
            interval_end = start_time + pd.Timedelta(interval)

        bins = pd.date_range(
            start=current,
            end=min(interval_end, end_time),
            freq=freq,
        )

        bin_list.extend(
            (bin_start, bin_end, bin_counter + i, freq)
            for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:]))
        )

        current = bins[-1] if len(bins) > 0 else current
        bin_counter += max(len(bins) - 1, 0)

    if not bin_list:
        # Very short trajectory: create at least one bin
        bin_list.append((start_time, end_time, 1, list(bin_intervals.values())[0]))

    bin_df = pd.DataFrame(
        bin_list, columns=["bin_start", "bin_end", "bin_counter", "bin_freq"]
    )

    # Filter by included frequencies
    bin_df = bin_df[bin_df['bin_freq'].isin(bin_freq_include)].copy()

    # 0-based contiguous position after filtering
    bin_df = bin_df.sort_values('bin_counter').reset_index(drop=True)
    bin_df['position'] = range(len(bin_df))

    return bin_df


def _count_visible_bins(bin_df: pd.DataFrame, current_time: pd.Timestamp) -> int:
    """Count how many bins have started by ``current_time``.

    A bin is "visible" (i.e. could contain data) if its start is at or before
    ``current_time``.  This determines the effective trajectory length for a
    given point in time on the fixed 30-day bin grid.
    """
    return int((bin_df['bin_start'] <= current_time).sum())


# ============================================================================
# Bin assignment
# ============================================================================

def _assign_to_bins(
    measurements: pd.DataFrame,
    bin_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign point-event timestamps to time bins using searchsorted.

    Args:
        measurements: DataFrame with columns [timestamp, feature, value].
        bin_df: Output of _create_patient_bins().

    Returns:
        DataFrame with original columns plus 'position'.
    """
    if measurements.empty or bin_df.empty:
        return measurements.assign(position=pd.Series(dtype=int)).iloc[0:0]

    bin_starts = bin_df['bin_start'].values
    bin_ends = bin_df['bin_end'].values
    positions = bin_df['position'].values

    timestamps = pd.to_datetime(measurements['timestamp']).values

    # searchsorted: find which bin each timestamp falls into
    indices = np.searchsorted(bin_starts, timestamps, side='right') - 1

    # Validate: index in range and timestamp < bin_end (strict, matching batch pipeline)
    valid_mask = (indices >= 0) & (indices < len(bin_df))
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return measurements.assign(position=pd.Series(dtype=int)).iloc[0:0]

    within_bin = timestamps[valid_idx] < bin_ends[indices[valid_idx]]
    final_idx = valid_idx[within_bin]

    if len(final_idx) == 0:
        return measurements.assign(position=pd.Series(dtype=int)).iloc[0:0]

    result = measurements.iloc[final_idx].copy()
    result['position'] = positions[indices[final_idx]]

    return result


def _expand_intervals_to_bins(
    intervals: pd.DataFrame,
    bin_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand interval events to one row per overlapping bin.

    For each event with (start, end, value), creates a row for every bin
    where the event is active: event_start < bin_end AND event_end > bin_start.

    Args:
        intervals: DataFrame with columns [start, end, value].
        bin_df: Output of _create_patient_bins().

    Returns:
        DataFrame with columns [timestamp, value, position].
    """
    if intervals.empty or bin_df.empty:
        return pd.DataFrame(columns=['timestamp', 'value', 'position'])

    starts = pd.to_datetime(intervals['start']).values
    ends = pd.to_datetime(intervals['end']).values
    values = intervals['value'].values

    bin_starts = bin_df['bin_start'].values
    bin_ends = bin_df['bin_end'].values
    positions = bin_df['position'].values

    rows = []
    for i in range(len(intervals)):
        # Find overlapping bins: event_start < bin_end AND event_end > bin_start
        overlap = (starts[i] < bin_ends) & (ends[i] > bin_starts)
        for j in np.where(overlap)[0]:
            rows.append({
                'timestamp': bin_starts[j],
                'value': values[i],
                'position': positions[j],
            })

    if not rows:
        return pd.DataFrame(columns=['timestamp', 'value', 'position'])

    return pd.DataFrame(rows)


# ============================================================================
# Aggregation
# ============================================================================

def _aggregate_measurements(
    assigned_df: pd.DataFrame,
    channel_map: dict,
) -> Dict[str, Dict[int, float]]:
    """
    Aggregate assigned measurements per (feature, agg_func) pair.

    Returns:
        Dict mapping channel_name -> {position: aggregated_value}.
    """
    if assigned_df.empty:
        return {}

    # Build reverse lookup: (raw_feature, agg_func) -> channel_name
    feature_agg_to_channel = {}
    for ch_name, info in channel_map.items():
        if info['type'] == 'continuous':
            key = (info['feature'], info['agg_func'])
            feature_agg_to_channel[key] = ch_name

    # Find which features in data have corresponding channels
    result = {}
    for feature in assigned_df['feature'].unique():
        feature_data = assigned_df[assigned_df['feature'] == feature]
        feature_data = feature_data.copy()
        feature_data['value'] = pd.to_numeric(feature_data['value'], errors='coerce')
        feature_data = feature_data.dropna(subset=['value'])

        if feature_data.empty:
            continue

        # Find all agg_funcs for this feature
        matching_keys = [
            (feat, agg) for (feat, agg) in feature_agg_to_channel
            if feat == feature
        ]

        for feat, agg_func in matching_keys:
            ch_name = feature_agg_to_channel[(feat, agg_func)]
            grouped = feature_data.groupby('position')['value']

            if agg_func == 'mean':
                agg_values = grouped.mean()
            elif agg_func == 'min':
                agg_values = grouped.min()
            elif agg_func == 'max':
                agg_values = grouped.max()
            elif agg_func == 'std':
                agg_values = grouped.std()
            elif agg_func == 'sum':
                agg_values = grouped.sum()
            elif agg_func == 'count':
                agg_values = grouped.count().astype(float)
            elif agg_func == 'first':
                agg_values = grouped.first()
            elif agg_func == 'last':
                agg_values = grouped.last()
            else:
                agg_values = grouped.mean()

            result[ch_name] = agg_values.to_dict()

    return result


# ============================================================================
# Temporal features
# ============================================================================

def _compute_temporal_features(
    bin_df: pd.DataFrame,
    admission_time: pd.Timestamp,
    channel_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute elapsed_hours and bin_width_hours from bin_df.

    Matches _create_temporal_features_df() in datasets.py.
    """
    result = {}
    n_positions = len(bin_df)

    if 'elapsed_hours' in channel_names:
        elapsed = (
            (bin_df['bin_start'] - admission_time).dt.total_seconds() / 3600
            + (bin_df['bin_end'] - bin_df['bin_start']).dt.total_seconds() / 7200
        )
        result['elapsed_hours'] = elapsed.values

    if 'bin_width_hours' in channel_names:
        widths = (
            (bin_df['bin_end'] - bin_df['bin_start']).dt.total_seconds() / 3600
        )
        result['bin_width_hours'] = widths.values

    return result


# ============================================================================
# Tier mapping features (antibiotic escalation tiers, etc.)
# ============================================================================

def _compute_tier_features_for_patient(
    raw_data: dict,
    bin_df: pd.DataFrame,
    ts_channel_names: List[str],
    bundle: dict,
) -> Dict[str, np.ndarray]:
    """Compute per-bin tier mapping features from raw medication events.

    Checks whether any tier feature channels exist in the model's channel
    list.  If so, classifies medication ATC codes into tiers and aggregates
    per bin.

    Returns:
        Dict mapping feature name (e.g. ``'abx_max_level'``) to a 1-D array
        of length ``len(bin_df)`` with per-bin values.  Empty dict if the
        model has no tier feature channels.
    """
    data_config = bundle.get('data_config', {})
    profile_cfg = data_config.get('categorical_profiles', {})
    if not profile_cfg.get('enabled'):
        return {}

    # Load profiles config to find tier mapping categories
    from astra.data.profiles import load_profiles_config
    profiles = load_profiles_config({'categorical_profiles': profile_cfg})

    # Check for composite mode in any concept
    for concept_name, concept_cfg in profiles.items():
        if not isinstance(concept_cfg, dict):
            continue
        if concept_cfg.get('composite_mode'):
            from astra.data.composite_features import compute_composite_features_for_patient
            med_events = raw_data.get(concept_name, [])
            return compute_composite_features_for_patient(
                med_events, bin_df, ts_channel_names,
            )

    # Collect tier mapping configs and expected channel names
    tier_configs = []  # (cat_name, tm_cfg, concept_name)
    for concept_name, concept_cfg in profiles.items():
        if not isinstance(concept_cfg, dict):
            continue
        for cat_name, cat_cfg in concept_cfg.get('categories', {}).items():
            tm = cat_cfg.get('tier_mapping')
            if tm:
                tier_configs.append((cat_name, tm, concept_name))

    if not tier_configs:
        return {}

    # Check if any tier channels are in the model
    channel_set = set(ts_channel_names)
    has_any = False
    for _cat, tm_cfg, _concept in tier_configs:
        short = tm_cfg.get('short_name', _cat)
        for feat in tm_cfg.get('features', []):
            if f"{short}_{feat}" in channel_set:
                has_any = True
                break
    if not has_any:
        return {}

    from astra.data.tier_mappings import get_mapping
    from astra.data.mappings import ATC_LVL3_MAP, ATC_LVL4_MAP

    n_positions = len(bin_df)
    bin_starts = bin_df['bin_start'].values
    bin_ends = bin_df['bin_end'].values
    positions = bin_df['position'].values if 'position' in bin_df.columns else np.arange(n_positions)
    result = {}

    for cat_name, tm_cfg, concept_name in tier_configs:
        mapping_name = tm_cfg.get('mapping')
        short_name = tm_cfg.get('short_name', cat_name)
        features = tm_cfg.get('features', ['max_level'])

        # Resolve ATC prefixes: config > atc_codes > ATC map fallback
        cat_prefixes = tm_cfg.get('atc_prefixes', [])
        if not cat_prefixes:
            cat_prefixes = tm_cfg.get('atc_codes', [])
        if not cat_prefixes:
            for source_map in [ATC_LVL3_MAP, ATC_LVL4_MAP]:
                cat_prefixes.extend(source_map.get(cat_name, []))

        # Resolve mapping (registered or auto-binary)
        mapping = get_mapping(mapping_name) if mapping_name else None

        # Collect medication events with ATC codes belonging to this category
        med_events = raw_data.get(concept_name, [])
        matched = []
        for ev in med_events:
            atc = ev.get('atc_code', '')
            if not atc:
                continue
            if any(atc.startswith(pfx) for pfx in cat_prefixes):
                if mapping is not None:
                    tier = mapping.classify(atc)
                    if tier is None:
                        continue
                else:
                    tier = 1  # auto-binary
                matched.append({
                    'timestamp': pd.Timestamp(ev['timestamp']),
                    'atc': atc,
                    'tier': tier,
                })

        if not matched:
            for feat in features:
                feat_col = f"{short_name}_{feat}"
                if feat_col in channel_set:
                    result[feat_col] = np.zeros(n_positions)
            continue

        # Assign to bins
        matched_df = pd.DataFrame(matched)

        assigned_rows = []
        for _, row in matched_df.iterrows():
            ts = np.datetime64(row['timestamp'])
            idx = np.searchsorted(bin_starts, ts, side='right') - 1
            if 0 <= idx < n_positions and ts < bin_ends[idx]:
                assigned_rows.append({
                    'position': int(positions[idx]),
                    'atc': row['atc'],
                    'tier': row['tier'],
                })

        if not assigned_rows:
            for feat in features:
                feat_col = f"{short_name}_{feat}"
                if feat_col in channel_set:
                    result[feat_col] = np.zeros(n_positions)
            continue

        assigned_df = pd.DataFrame(assigned_rows)

        for feat in features:
            feat_col = f"{short_name}_{feat}"
            if feat_col not in channel_set:
                continue

            values = np.zeros(n_positions)

            if feat == 'max_level':
                agg = assigned_df.groupby('position')['tier'].max()
            elif feat == 'n_distinct':
                agg = assigned_df.groupby('position')['atc'].nunique()
            else:
                continue

            for pos, val in agg.items():
                if 0 <= pos < n_positions:
                    values[pos] = float(val)

            result[feat_col] = values

    return result


# ============================================================================
# Continuous time series
# ============================================================================

def _build_continuous_ts(
    raw_data: dict,
    bin_df: pd.DataFrame,
    bundle: dict,
    trajectory_length: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Build raw (unnormalized) continuous time series tensor.

    Args:
        trajectory_length: If provided, use this as the effective trajectory
            length (from visibility masking). If None, defaults to len(bin_df).

    Returns:
        Tuple of (x_ts [n_channels, seq_len], trajectory_length).
    """
    ts_channel_names = bundle['ts_channel_names']
    seq_len = bundle['model_params']['seq_len']
    data_config = bundle['data_config']
    channel_map = data_config['channel_map']

    n_channels = len(ts_channel_names)
    channel_to_idx = {name: i for i, name in enumerate(ts_channel_names)}

    # Initialize with NaN (missing measurement)
    x_ts = np.full((n_channels, seq_len), np.nan, dtype=np.float64)

    # Combine all continuous concept measurements into a single DataFrame.
    # Continuous concepts have records with 'feature' key (vs categorical
    # which only have 'value').  Classification is config-driven via
    # ts_cat_names stored in bundle['data_config'].
    ts_cat_names = set(data_config.get('ts_cat_names', []))
    records = []
    for key, items in raw_data.items():
        if key in _RAW_DATA_META_KEYS or key in ts_cat_names:
            continue
        if not isinstance(items, list):
            continue
        for m in items:
            if 'feature' in m:
                records.append({
                    'timestamp': m['timestamp'],
                    'feature': m['feature'],
                    'value': m['value'],
                })

    if records:
        measurements_df = pd.DataFrame(records)
        measurements_df['timestamp'] = pd.to_datetime(measurements_df['timestamp'])

        # Assign to bins
        assigned = _assign_to_bins(measurements_df, bin_df)

        # Aggregate per channel
        aggregated = _aggregate_measurements(assigned, channel_map)

        # Fill x_ts from aggregated values
        for ch_name, pos_values in aggregated.items():
            if ch_name not in channel_to_idx:
                continue
            ch_idx = channel_to_idx[ch_name]
            for pos, val in pos_values.items():
                if pos < seq_len:
                    x_ts[ch_idx, pos] = val

    # Compute temporal features
    temporal_features = _compute_temporal_features(
        bin_df, raw_data['admission_time'], ts_channel_names
    )
    for feat_name, values in temporal_features.items():
        if feat_name not in channel_to_idx:
            continue
        ch_idx = channel_to_idx[feat_name]
        n = min(len(values), seq_len)
        x_ts[ch_idx, :n] = values[:n]

    # Compute _data_present indicator: 1.0 where any clinical channel has a measurement.
    # Must happen BEFORE padding zeros are applied so the padding bins stay 0.0.
    # Auxiliary channels (elapsed_hours, bin_width_hours, _data_present, _ebm_pred) are
    # excluded from the presence check — only actual clinical measurements count.
    # Compute tier mapping features (antibiotic escalation tiers, etc.)
    tier_features = _compute_tier_features_for_patient(
        raw_data, bin_df, ts_channel_names, bundle
    )
    for feat_name, values in tier_features.items():
        if feat_name not in channel_to_idx:
            continue
        ch_idx = channel_to_idx[feat_name]
        n = min(len(values), seq_len)
        x_ts[ch_idx, :n] = values[:n]

    # Compute _data_present indicator: 1.0 where any clinical channel has a measurement.
    # Must happen BEFORE padding zeros are applied so the padding bins stay 0.0.
    # Auxiliary channels (elapsed_hours, bin_width_hours, _data_present, _ebm_pred,
    # and tier mapping features) are excluded — only actual clinical measurements count.
    _AUXILIARY = {'elapsed_hours', 'bin_width_hours', '_data_present', '_ebm_pred'}
    _AUXILIARY |= set(tier_features.keys())

    if '_data_present' in channel_to_idx:
        dp_ch = channel_to_idx['_data_present']
        clinical_indices = [
            channel_to_idx[name]
            for name in ts_channel_names
            if name not in _AUXILIARY and name in channel_to_idx
        ]
        if clinical_indices:
            # any non-NaN value in clinical channels → measurement present
            has_data = ~np.all(np.isnan(x_ts[clinical_indices, :]), axis=0)  # [seq_len]
            x_ts[dp_ch, :] = has_data.astype(np.float64)
        else:
            x_ts[dp_ch, :] = 0.0

    # Trajectory length: use explicit value (from visibility masking) or bin count
    if trajectory_length is None:
        trajectory_length = len(bin_df)
    trajectory_length = min(trajectory_length, seq_len)

    # Set padding beyond trajectory to 0.0
    if trajectory_length < seq_len:
        x_ts[:, trajectory_length:] = 0.0

    logger.debug("_build_continuous_ts: shape=%s channels=%d traj_len=%d",
                 x_ts.shape, len(ts_channel_names), trajectory_length)
    return x_ts, trajectory_length


# ============================================================================
# Categorical time series
# ============================================================================

def _build_categorical_ts(
    raw_data: dict,
    bin_df: pd.DataFrame,
    bundle: dict,
) -> np.ndarray:
    """
    Build multi-hot encoded categorical time series tensor.

    Constructs the multi-hot array directly using encoder internals
    rather than building a wide-format DataFrame.  Categorical concepts
    and their encoder feature names are read from the bundle's
    ``data_config['cat_encoder_names']`` (config-driven).

    Returns:
        np.ndarray of shape [n_cat_dims, seq_len].
    """
    seq_len = bundle['model_params']['seq_len']
    encoding_info = bundle['encoding_info']
    cat_encoder = bundle['cat_encoder']
    cat_encoder_names = bundle['data_config'].get('cat_encoder_names', {})

    if not cat_encoder_names:
        logger.warning(
            "Bundle has NO cat_encoder_names — categorical TS will be all zeros! "
            "Regenerate bundle with updated save_deployment_bundle()."
        )

    logger.info(
        f"_build_categorical_ts: cat_encoder_names={cat_encoder_names}, "
        f"raw_data keys={[k for k in raw_data if k not in _RAW_DATA_META_KEYS]}"
    )

    # Total categorical dimensions
    total_dim = sum(
        end - start for start, end in encoding_info['feature_ranges'].values()
    )
    x_ts_cat = np.zeros((total_dim, seq_len), dtype=np.float32)

    for concept_name, encoder_feat_name in cat_encoder_names.items():
        events = raw_data.get(concept_name, [])
        if not events:
            logger.info(f"  {concept_name}: no events in raw_data")
            continue

        # Check encoder has this feature
        if encoder_feat_name not in cat_encoder.encoders_:
            logger.warning(
                f"Encoder has no feature '{encoder_feat_name}' — skipping {concept_name}"
            )
            continue

        encoder_info = cat_encoder.encoders_[encoder_feat_name]
        value_to_idx = encoder_info['value_to_idx']
        dim_start, dim_end = encoding_info['feature_ranges'][encoder_feat_name]

        # Detect interval vs point events from data structure
        is_interval = 'start' in events[0]

        # Assign events to bins
        if is_interval:
            # Interval-based events (e.g. ADTHaendelser)
            intervals_df = pd.DataFrame(events)
            if intervals_df.empty:
                continue
            intervals_df['start'] = pd.to_datetime(intervals_df['start'])
            intervals_df['end'] = pd.to_datetime(intervals_df['end'])
            assigned = _expand_intervals_to_bins(intervals_df, bin_df)
        else:
            # Point events (e.g. Medicin, Procedurer)
            events_df = pd.DataFrame(events)
            if events_df.empty:
                continue
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            # These don't have a 'feature' column — add a dummy for _assign_to_bins
            events_df['feature'] = encoder_feat_name
            assigned = _assign_to_bins(events_df, bin_df)

        if assigned.empty:
            logger.info(f"  {concept_name}: {len(events)} events but none assigned to bins")
            continue

        # Set multi-hot values
        n_matched = 0
        n_unknown = 0
        for _, row in assigned.iterrows():
            pos = int(row['position'])
            if pos >= seq_len:
                continue
            val = row['value']
            if val in value_to_idx:
                idx = value_to_idx[val]
                x_ts_cat[dim_start + idx, pos] = 1.0
                n_matched += 1
            else:
                n_unknown += 1
                logger.debug(f"Unknown category '{val}' for {encoder_feat_name} — skipping")

        logger.info(
            f"  {concept_name}: {len(events)} events → {len(assigned)} assigned → "
            f"{n_matched} encoded, {n_unknown} unknown"
        )

    nonzero = int(np.count_nonzero(x_ts_cat))
    logger.info(f"_build_categorical_ts: shape={x_ts_cat.shape} nonzero={nonzero}")
    return x_ts_cat


def _build_profile_ts(
    raw_data: dict,
    bin_df: pd.DataFrame,
    bundle: dict,
) -> Optional[np.ndarray]:
    """Build profile-encoded categorical TS tensor for inference.

    Groups medication/procedure events by bin and category, counts distinct
    sub-codes per group, and applies profile rules to determine ordinal levels.

    Returns:
        np.ndarray of shape ``[n_profiled_categories, seq_len]`` (int8),
        or None if profiles are disabled.
    """
    profile_dims = bundle.get('ts_cat_profile_dims')
    category_order = bundle.get('profile_category_order')
    profile_cfg = bundle.get('data_config', {}).get('categorical_profiles', {})

    if not profile_dims or not category_order or not profile_cfg.get('enabled'):
        return None

    from astra.data.profiles import load_profiles_config, evaluate_profile_rules

    # Build a temporary cfg dict for load_profiles_config
    profiles_config = load_profiles_config({'categorical_profiles': profile_cfg})

    seq_len = bundle['model_params']['seq_len']
    n_profiled = len(category_order)
    x_profiles = np.zeros((n_profiled, seq_len), dtype=np.int8)

    cat_encoder_names = bundle['data_config'].get('cat_encoder_names', {})

    # For each concept that has profiles defined, collect sub-codes per bin per category
    for concept_name in cat_encoder_names:
        concept_profiles = profiles_config.get(concept_name, {})
        profiled_cats = concept_profiles.get('categories', {})
        sub_code_level = concept_profiles.get('sub_code_level', 0)

        if not profiled_cats or sub_code_level == 0:
            continue

        events = raw_data.get(concept_name, [])
        if not events:
            continue

        # Assign events to bins (reuse existing logic)
        is_interval = 'start' in events[0]
        events_df = pd.DataFrame(events)
        if events_df.empty:
            continue

        if is_interval:
            events_df['start'] = pd.to_datetime(events_df['start'])
            events_df['end'] = pd.to_datetime(events_df['end'])
            assigned = _expand_intervals_to_bins(events_df, bin_df)
        else:
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            events_df['feature'] = 'tmp'
            assigned = _assign_to_bins(events_df, bin_df)

        if assigned.empty:
            continue

        # Group by bin position and category, collect sub-codes
        # Events should have 'value' (category name) and 'sub_code' (ATC detail)
        bin_cat_subcodes: dict = {}  # {(pos, category): set(sub_codes)}
        for _, row in assigned.iterrows():
            pos = int(row['position'])
            if pos >= seq_len:
                continue
            category = row.get('value', '')
            sub_code = row.get('sub_code', '')
            if not category or not sub_code or category not in profiled_cats:
                continue
            key = (pos, category)
            if key not in bin_cat_subcodes:
                bin_cat_subcodes[key] = set()
            bin_cat_subcodes[key].add(sub_code)

        # Evaluate profile rules and fill tensor
        for (pos, category), sub_codes in bin_cat_subcodes.items():
            if category not in profiled_cats:
                continue
            rules = profiled_cats[category].get('rules', [])
            level = evaluate_profile_rules(rules, sub_codes)
            if category in category_order:
                cat_idx = category_order.index(category)
                x_profiles[cat_idx, pos] = level

    nonzero = int(np.count_nonzero(x_profiles))
    logger.info(f"_build_profile_ts: shape={x_profiles.shape} nonzero={nonzero}")
    return x_profiles


# ============================================================================
# BinCache & incremental tensor builders
# ============================================================================

@dataclass
class BinCache:
    """Caches raw measurements per bin for incremental tensor updates.

    On each refresh, only newly arriving measurements are assigned to bins
    and only the affected ("dirty") positions are re-aggregated.
    """

    # {position: [(feature, value, timestamp)]}
    continuous_bins: Dict[int, list] = field(default_factory=dict)
    # {(concept_name, position): [value]}  — concept_name from cfg['concepts']
    categorical_bins: Dict[tuple, list] = field(default_factory=dict)
    # Positions modified since the last tensor write
    dirty_continuous: set = field(default_factory=set)
    dirty_categorical: set = field(default_factory=set)


def _populate_cache_from_raw_data(
    raw_data: dict,
    bin_df: pd.DataFrame,
    bundle: dict,
) -> BinCache:
    """Build a :class:`BinCache` from *raw_data* (used on first creation)."""
    cache = BinCache()
    data_config = bundle['data_config']
    ts_cat_names = set(data_config.get('ts_cat_names', []))
    cat_encoder_names = data_config.get('cat_encoder_names', {})

    # --- continuous: iterate all non-meta, non-categorical keys ---
    records = []
    for key, items in raw_data.items():
        if key in _RAW_DATA_META_KEYS or key in ts_cat_names:
            continue
        if not isinstance(items, list):
            continue
        for m in items:
            if 'feature' in m:
                records.append({
                    'timestamp': m['timestamp'],
                    'feature': m['feature'],
                    'value': m['value'],
                })
    if records:
        mdf = pd.DataFrame(records)
        mdf['timestamp'] = pd.to_datetime(mdf['timestamp'])
        assigned = _assign_to_bins(mdf, bin_df)
        for _, row in assigned.iterrows():
            pos = int(row['position'])
            cache.continuous_bins.setdefault(pos, []).append(
                (row['feature'], row['value'], row['timestamp'])
            )

    # --- categorical: iterate concepts from cat_encoder_names ---
    for concept_name, _enc_name in cat_encoder_names.items():
        events = raw_data.get(concept_name, [])
        if not events:
            continue
        is_interval = 'start' in events[0]
        if is_interval:
            idf = pd.DataFrame(events)
            if idf.empty:
                continue
            idf['start'] = pd.to_datetime(idf['start'])
            idf['end'] = pd.to_datetime(idf['end'])
            assigned = _expand_intervals_to_bins(idf, bin_df)
        else:
            edf = pd.DataFrame(events)
            if edf.empty:
                continue
            edf['timestamp'] = pd.to_datetime(edf['timestamp'])
            edf['feature'] = _enc_name
            assigned = _assign_to_bins(edf, bin_df)
        for _, row in assigned.iterrows():
            pos = int(row['position'])
            cache.categorical_bins.setdefault((concept_name, pos), []).append(row['value'])

    # All positions are "dirty" on first build (they will be written to tensors)
    cache.dirty_continuous = set(cache.continuous_bins.keys())
    cache.dirty_categorical = {k for k in cache.categorical_bins.keys()}

    return cache


def _assign_and_cache_continuous(
    new_records: List[dict],
    bin_df: pd.DataFrame,
    cache: BinCache,
) -> set:
    """Assign new continuous measurements to bins and update *cache*.

    Returns the set of dirty bin positions.
    """
    if not new_records:
        return set()

    mdf = pd.DataFrame(new_records)
    mdf['timestamp'] = pd.to_datetime(mdf['timestamp'])
    assigned = _assign_to_bins(mdf, bin_df)

    dirty = set()
    for _, row in assigned.iterrows():
        pos = int(row['position'])
        cache.continuous_bins.setdefault(pos, []).append(
            (row['feature'], row['value'], row['timestamp'])
        )
        dirty.add(pos)

    cache.dirty_continuous |= dirty
    return dirty


def _assign_and_cache_categorical(
    new_events: List[dict],
    concept_name: str,
    bin_df: pd.DataFrame,
    cache: BinCache,
    bundle: dict,
) -> set:
    """Assign new categorical events to bins and update *cache*.

    Returns the set of dirty (concept_name, position) tuples.
    """
    cat_encoder_names = bundle['data_config'].get('cat_encoder_names', {})
    if not new_events:
        return set()

    # Detect interval vs point events from data structure
    is_interval = 'start' in new_events[0]

    if is_interval:
        idf = pd.DataFrame(new_events)
        if idf.empty:
            return set()
        idf['start'] = pd.to_datetime(idf['start'])
        idf['end'] = pd.to_datetime(idf['end'])
        assigned = _expand_intervals_to_bins(idf, bin_df)
    else:
        edf = pd.DataFrame(new_events)
        if edf.empty:
            return set()
        edf['timestamp'] = pd.to_datetime(edf['timestamp'])
        edf['feature'] = cat_encoder_names.get(concept_name, concept_name)
        assigned = _assign_to_bins(edf, bin_df)

    dirty = set()
    for _, row in assigned.iterrows():
        pos = int(row['position'])
        key = (concept_name, pos)
        cache.categorical_bins.setdefault(key, []).append(row['value'])
        dirty.add(key)

    cache.dirty_categorical |= dirty
    return dirty


def _reaggregate_dirty_bins(
    cache: BinCache,
    channel_map: dict,
    channel_to_idx: Dict[str, int],
    x_ts: np.ndarray,
    dirty_positions: set,
) -> None:
    """Re-aggregate dirty continuous bins from cached raw values into *x_ts*.

    Modifies *x_ts* in place.
    """
    # Build reverse lookup: (raw_feature, agg_func) → channel_name
    feature_agg_to_channel = {}
    for ch_name, info in channel_map.items():
        if info['type'] == 'continuous':
            key = (info['feature'], info['agg_func'])
            feature_agg_to_channel[key] = ch_name

    seq_len = x_ts.shape[1]

    for pos in dirty_positions:
        if pos >= seq_len:
            continue
        entries = cache.continuous_bins.get(pos, [])
        if not entries:
            continue

        # Group entries by feature
        by_feature: Dict[str, list] = {}
        for feat, val, _ts in entries:
            by_feature.setdefault(feat, []).append(val)

        for feat, raw_vals in by_feature.items():
            nums = pd.to_numeric(pd.Series(raw_vals), errors='coerce').dropna()
            if nums.empty:
                continue
            for (f, agg_func), ch_name in feature_agg_to_channel.items():
                if f != feat:
                    continue
                if ch_name not in channel_to_idx:
                    continue
                ch_idx = channel_to_idx[ch_name]

                if agg_func == 'mean':
                    x_ts[ch_idx, pos] = nums.mean()
                elif agg_func == 'min':
                    x_ts[ch_idx, pos] = nums.min()
                elif agg_func == 'max':
                    x_ts[ch_idx, pos] = nums.max()
                elif agg_func == 'std':
                    x_ts[ch_idx, pos] = nums.std() if len(nums) > 1 else np.nan
                elif agg_func == 'sum':
                    x_ts[ch_idx, pos] = nums.sum()
                elif agg_func == 'count':
                    x_ts[ch_idx, pos] = float(len(nums))
                elif agg_func == 'first':
                    x_ts[ch_idx, pos] = float(nums.iloc[0])
                elif agg_func == 'last':
                    x_ts[ch_idx, pos] = float(nums.iloc[-1])
                else:
                    x_ts[ch_idx, pos] = nums.mean()


def _build_continuous_ts_incremental(
    new_records: List[dict],
    bin_df: pd.DataFrame,
    bundle: dict,
    cache: BinCache,
    x_ts_existing: Optional[np.ndarray],
    old_trajectory_length: int,
    trajectory_length: int,
    admission_time: pd.Timestamp = None,
    profiling: Optional[dict] = None,
    raw_data: Optional[dict] = None,
) -> Tuple[np.ndarray, int]:
    """Incrementally update the continuous time series tensor.

    If *x_ts_existing* is ``None``, falls back to a full build via
    :func:`_build_continuous_ts` and populates the cache.

    Otherwise, only assigns *new_records* to bins, re-aggregates dirty
    positions, and extends the visible window.

    Args:
        profiling: Optional dict to collect sub-stage timing (for perf analysis).

    Returns ``(x_ts, trajectory_length)``.
    """
    ts_channel_names = bundle['ts_channel_names']
    seq_len = bundle['model_params']['seq_len']
    data_config = bundle['data_config']
    channel_map = data_config['channel_map']

    n_channels = len(ts_channel_names)
    channel_to_idx = {name: i for i, name in enumerate(ts_channel_names)}

    from astra.data.profiles import get_tier_feature_names
    _AUXILIARY = (
        {'elapsed_hours', 'bin_width_hours', '_data_present', '_ebm_pred'}
        | get_tier_feature_names(data_config)
    )

    if x_ts_existing is None:
        # First build — fall back to full
        # (cache should have been populated already by _populate_cache_from_raw_data)
        x_ts = np.full((n_channels, seq_len), np.nan, dtype=np.float64)
        # Re-aggregate everything in the cache
        _reaggregate_dirty_bins(cache, channel_map, channel_to_idx, x_ts, cache.dirty_continuous)
        cache.dirty_continuous.clear()
    else:
        x_ts = x_ts_existing
        # Reveal newly visible bins: reset padding → NaN so aggregation can fill them
        if trajectory_length > old_trajectory_length:
            for ch_name in ts_channel_names:
                if ch_name in _AUXILIARY:
                    continue
                ch_idx = channel_to_idx[ch_name]
                x_ts[ch_idx, old_trajectory_length:trajectory_length] = np.nan

        # Assign new records and re-aggregate dirty bins
        if new_records:
            with timed_stage(profiling, 'cts_assign') if profiling is not None else _nullcontext():
                _assign_and_cache_continuous(new_records, bin_df, cache)

        if cache.dirty_continuous:
            with timed_stage(profiling, 'cts_reaggregate') if profiling is not None else _nullcontext():
                _reaggregate_dirty_bins(
                    cache, channel_map, channel_to_idx, x_ts,
                    cache.dirty_continuous,
                )
            cache.dirty_continuous.clear()

    # Temporal features (always recomputed for full grid)
    with timed_stage(profiling, 'cts_temporal_features') if profiling is not None else _nullcontext():
        if admission_time is None:
            admission_time = pd.Timestamp(bin_df['bin_start'].iloc[0])
        temporal_features = _compute_temporal_features(
            bin_df, admission_time, ts_channel_names,
        )
        for feat_name, values in temporal_features.items():
            if feat_name not in channel_to_idx:
                continue
            ch_idx = channel_to_idx[feat_name]
            n = min(len(values), seq_len)
            x_ts[ch_idx, :n] = values[:n]

    # Recompute tier features from accumulated raw_data
    if raw_data is not None:
        with timed_stage(profiling, 'cts_tier_features') if profiling is not None else _nullcontext():
            tier_features = _compute_tier_features_for_patient(
                raw_data, bin_df, ts_channel_names, bundle,
            )
            for feat_name, values in tier_features.items():
                if feat_name not in channel_to_idx:
                    continue
                ch_idx = channel_to_idx[feat_name]
                n = min(len(values), seq_len)
                x_ts[ch_idx, :n] = values[:n]

    # Update _data_present
    with timed_stage(profiling, 'cts_data_present') if profiling is not None else _nullcontext():
        if '_data_present' in channel_to_idx:
            dp_ch = channel_to_idx['_data_present']
            clinical_indices = [
                channel_to_idx[name]
                for name in ts_channel_names
                if name not in _AUXILIARY and name in channel_to_idx
            ]
            if clinical_indices:
                has_data = ~np.all(np.isnan(x_ts[clinical_indices, :]), axis=0)
                x_ts[dp_ch, :] = has_data.astype(np.float64)
            else:
                x_ts[dp_ch, :] = 0.0

    # Clamp trajectory length and apply padding
    trajectory_length = min(trajectory_length, seq_len)
    if trajectory_length < seq_len:
        x_ts[:, trajectory_length:] = 0.0

    return x_ts, trajectory_length


def _build_categorical_ts_incremental(
    new_events: Dict[str, List[dict]],
    bin_df: pd.DataFrame,
    bundle: dict,
    cache: BinCache,
    x_ts_cat_existing: Optional[np.ndarray],
    trajectory_length: int,
) -> np.ndarray:
    """Incrementally update the categorical time series tensor.

    Multi-hot encoding is additive — new events just set additional bits.

    Args:
        new_events: ``{concept_name: [event_dicts]}`` keyed by concept name
            (e.g. ``'Medicin'``, ``'Procedurer'``, ``'ADTHaendelser'``).
    """
    seq_len = bundle['model_params']['seq_len']
    encoding_info = bundle['encoding_info']
    cat_encoder = bundle['cat_encoder']
    cat_encoder_names = bundle['data_config'].get('cat_encoder_names', {})

    total_dim = sum(
        end - start for start, end in encoding_info['feature_ranges'].values()
    )

    if x_ts_cat_existing is None:
        x_ts_cat = np.zeros((total_dim, seq_len), dtype=np.float32)
    else:
        x_ts_cat = x_ts_cat_existing

    # Assign new events to cache
    for concept_name, events in new_events.items():
        if events:
            _assign_and_cache_categorical(events, concept_name, bin_df, cache, bundle)

    # Write dirty positions to tensor
    for (concept_name, pos) in cache.dirty_categorical:
        if pos >= seq_len:
            continue
        encoder_feat_name = cat_encoder_names.get(concept_name)
        if encoder_feat_name is None or encoder_feat_name not in cat_encoder.encoders_:
            continue

        encoder_info = cat_encoder.encoders_[encoder_feat_name]
        value_to_idx = encoder_info['value_to_idx']
        dim_start, _ = encoding_info['feature_ranges'][encoder_feat_name]

        for val in cache.categorical_bins.get((concept_name, pos), []):
            if val in value_to_idx:
                idx = value_to_idx[val]
                x_ts_cat[dim_start + idx, pos] = 1.0

    cache.dirty_categorical.clear()

    # Zero out beyond visibility
    trajectory_length = min(trajectory_length, seq_len)
    if trajectory_length < seq_len:
        x_ts_cat[:, trajectory_length:] = 0.0

    return x_ts_cat


# ============================================================================
# Static tabular features
# ============================================================================

def _build_tab_df(
    raw_data: dict,
    bundle: dict,
) -> pd.DataFrame:
    """
    Build static tabular DataFrame for a single patient.

    Returns:
        pd.DataFrame with one row.
    """
    demographics = raw_data.get('demographics', {})

    row = {
        'PID': raw_data.get('pid', 0),
        'deceased_30d': 0,  # Placeholder for inference
    }

    # Numeric columns
    for col in bundle['tab_feature_names']:
        row[col] = demographics.get(col, np.nan)

    # Categorical columns
    for col in bundle['cat_feature_names']:
        row[col] = demographics.get(col, np.nan)

    tab_df = pd.DataFrame([row])

    # Ensure numeric columns are float
    for col in bundle['tab_feature_names']:
        tab_df[col] = pd.to_numeric(tab_df[col], errors='coerce')

    return tab_df


# ============================================================================
# Public API
# ============================================================================

def prepare_single_patient(
    raw_data: dict,
    bundle: dict,
) -> dict:
    """
    Convert raw EHR data for a single patient into model-ready tensors.

    Args:
        raw_data: Dict with patient data. Expected keys:
            - pid: Patient identifier
            - admission_time: Trajectory start (str or Timestamp)
            - current_time: Time of inference (str or Timestamp)
            - demographics: Dict with AGE, SEX, FIRST_HOSPITAL, HEIGHT, WEIGHT, ASMT_ELIX
            - vitals: List of {timestamp, feature, value} dicts
            - labs: List of {timestamp, feature, value} dicts
            - icu: List of {timestamp, feature, value} dicts
            - medications: List of {timestamp, value} dicts
            - procedures: List of {timestamp, value} dicts
            - adt: List of {start, end, value} dicts
        bundle: Deployment bundle from load_deployment_bundle().

    Returns:
        Dict with keys:
            x_ts: np.ndarray [n_channels, seq_len]
            x_ts_cat: np.ndarray [n_cat_dims, seq_len]
            tab_df: pd.DataFrame (1 row)
            trajectory_length: int
            bin_df: pd.DataFrame (for debugging)
    """
    if 'data_config' not in bundle:
        raise ValueError(
            "Bundle missing 'data_config'. Re-save the deployment bundle "
            "using the updated save_deployment_bundle() to include data "
            "processing configuration."
        )

    data_config = bundle['data_config']

    # Parse timestamps
    raw_data = dict(raw_data)  # shallow copy to avoid mutating caller's dict
    raw_data['admission_time'] = pd.Timestamp(raw_data['admission_time'])
    raw_data['current_time'] = pd.Timestamp(raw_data['current_time'])

    # 1. Create fixed 30-day bin grid (stable across re-inferences)
    bin_df = _create_patient_bins(
        raw_data['admission_time'],
        data_config,
    )

    # Determine how many bins are "visible" at current_time using cfg bin intervals
    from astra.evaluation.utils import time_to_step
    delta_minutes = (raw_data['current_time'] - raw_data['admission_time']).total_seconds() / 60
    step = time_to_step(delta_minutes, 'min', data_config=data_config)
    visible_bins = (step + 1) if step is not None else len(bin_df)
    logger.info(
        f"Created {len(bin_df)} bins (30-day grid), "
        f"{visible_bins} visible at {raw_data['current_time']}"
    )

    # 2. Build continuous TS (trajectory_length clamped by visibility)
    x_ts, trajectory_length = _build_continuous_ts(
        raw_data, bin_df, bundle, trajectory_length=visible_bins,
    )

    # 3. Build categorical TS
    x_ts_cat = _build_categorical_ts(raw_data, bin_df, bundle)

    # 3b. Build profile TS (if profiles enabled)
    x_ts_cat_profiles = _build_profile_ts(raw_data, bin_df, bundle)

    # Zero out bins beyond the visible horizon (guards against future data
    # leaking in when simulating with historic patients).
    seq_len = bundle['model_params']['seq_len']
    if trajectory_length < seq_len:
        x_ts_cat[:, trajectory_length:] = 0.0
        if x_ts_cat_profiles is not None:
            x_ts_cat_profiles[:, trajectory_length:] = 0

    # 4. Build tabular features
    tab_df = _build_tab_df(raw_data, bundle)

    logger.info(
        f"Prepared patient {raw_data.get('pid', '?')}: "
        f"x_ts={x_ts.shape}, x_ts_cat={x_ts_cat.shape}, "
        f"trajectory_length={trajectory_length}"
    )

    result = {
        'x_ts': x_ts,
        'x_ts_cat': x_ts_cat,
        'tab_df': tab_df,
        'trajectory_length': trajectory_length,
        'bin_df': bin_df,
    }
    if x_ts_cat_profiles is not None:
        result['x_ts_cat_profiles'] = x_ts_cat_profiles
    return result


# ============================================================================
# Raw EHR standardization (uses shared mappings from astra.data.mappings)
# ============================================================================

def _standardize_vitals(raw_vitals: List[dict]) -> List[dict]:
    """Convert raw vital signs to standardized format.

    Input:  [{'timestamp': ..., 'parameter': 'Puls', 'value': '92'}, ...]
    Output: [{'timestamp': ..., 'feature': 'HR', 'value': 92.0}, ...]
    """
    result = []
    for v in raw_vitals:
        ts = v['timestamp']
        param = v.get('parameter', v.get('feature', ''))
        value_str = str(v['value'])

        if param in BP_TYPES:
            parts = value_str.split('/', 1)
            if len(parts) == 2:
                sbp_val = parse_numeric(parts[0])
                dbp_val = parse_numeric(parts[1])
                if sbp_val is not None:
                    result.append({'timestamp': ts, 'feature': 'SBP', 'value': sbp_val})
                if dbp_val is not None:
                    result.append({'timestamp': ts, 'feature': 'DBP', 'value': dbp_val})
            continue

        feature = VITALS_MAP.get(param, param)
        val = parse_numeric(value_str)
        if val is not None:
            result.append({'timestamp': ts, 'feature': feature, 'value': val})

    # Apply physiological bounds (same as batch pipeline in filters.py)
    n_before = len(result)
    result = [
        r for r in result
        if r['feature'] not in VITALS_BOUNDS
        or VITALS_BOUNDS[r['feature']][0] <= r['value'] <= VITALS_BOUNDS[r['feature']][1]
    ]
    if len(result) < n_before:
        logger.debug(f"Vitals bounds: removed {n_before - len(result)} out-of-range values")

    return result


def _extract_invasive_events(raw_vitals: List[dict]) -> List[dict]:
    """Extract invasive monitoring events from raw vitals for categorical TS.

    Returns list of {'timestamp': ..., 'value': 'arterial_bp'|'arterial_hr'|'invasive_temp'}.
    """
    result = []
    for v in raw_vitals:
        param = v.get('parameter', v.get('feature', ''))
        ts = v['timestamp']
        if param in INVASIVE_VITALS_MAP:
            result.append({'timestamp': ts, 'value': INVASIVE_VITALS_MAP[param]})
        elif param in INVASIVE_BP_TYPES:
            result.append({'timestamp': ts, 'value': 'arterial_bp'})
    return result


def _standardize_labs(raw_labs: List[dict]) -> List[dict]:
    """Convert raw lab results to standardized format.

    Input:  [{'timestamp': ..., 'test_name': 'LAKTAT(POC);P(AB)', 'value': '2,1'}, ...]
    Output: [{'timestamp': ..., 'feature': 'LACTATE', 'value': 2.1}, ...]
    """
    result = []
    for lab in raw_labs:
        ts = lab['timestamp']
        test = lab.get('test_name', lab.get('feature', ''))
        value_str = str(lab['value']).replace(',', '.').replace('*', '')
        feature = LABS_REVERSE_MAP.get(test, test)
        val = parse_numeric(value_str)
        if val is not None:
            result.append({'timestamp': ts, 'feature': feature, 'value': val})
    return result


def _standardize_icu(raw_icu: List[dict]) -> List[dict]:
    """Convert raw ICU scores to standardized format.

    Input:  [{'timestamp': ..., 'measurement': 'GLASGOW COMA SCORE', 'value': 15}, ...]
    Output: [{'timestamp': ..., 'feature': 'GCS', 'value': 15.0}, ...]
    """
    result = []
    for s in raw_icu:
        ts = s['timestamp']
        measurement = s.get('measurement', s.get('feature', ''))
        feature = ICU_MAP.get(measurement, measurement)
        val = parse_numeric(str(s['value']))
        if val is not None:
            result.append({'timestamp': ts, 'feature': feature, 'value': val})
    return result


def _standardize_ews(raw_ews: List[dict]) -> List[dict]:
    """Convert raw EWS measurements to standardized format.

    Input:  [{'timestamp': ..., 'measurement': 'EWS korr. total score', 'value': 3}, ...]
    Output: [{'timestamp': ..., 'feature': 'EWS_SCORE', 'value': 3.0}, ...]
    """
    result = []
    for e in raw_ews:
        ts = e['timestamp']
        measurement = e.get('measurement', e.get('feature', ''))
        feature = EWS_MAP.get(measurement, measurement)
        val = parse_numeric(str(e['value']))
        if val is not None:
            result.append({'timestamp': ts, 'feature': feature, 'value': val})
    return result


def _get_inference_sub_code_level(bundle: Optional[dict], concept: str) -> int:
    """Get sub_code_level for a concept from the deployment bundle."""
    if bundle is None:
        return 0
    profile_cfg = bundle.get('data_config', {}).get('categorical_profiles', {})
    if not profile_cfg.get('enabled'):
        return 0
    from astra.data.profiles import load_profiles_config
    profiles = load_profiles_config({'categorical_profiles': profile_cfg})
    return profiles.get(concept, {}).get('sub_code_level', 0)


def _standardize_medications(raw_meds: List[dict], sub_code_level: int = 0) -> List[dict]:
    """Convert raw ATC codes to medication category names.

    Input:  [{'timestamp': ..., 'atc_code': 'N02AB02'}, ...]
    Output: [{'timestamp': ..., 'value': 'opiods', 'atc_code': 'N02AB02'}, ...]

    Always preserves the full ``atc_code`` for tier mapping features.
    When *sub_code_level* > 0 (profiles enabled), also preserves a truncated
    ATC sub-code in each event dict for profile determination.
    """
    result = []
    for med in raw_meds:
        ts = med['timestamp']
        atc = str(med.get('atc_code', med.get('value', '')))
        category = classify_atc(atc)
        if category is not None:
            entry = {'timestamp': ts, 'value': category, 'atc_code': atc}
            if sub_code_level > 0:
                entry['sub_code'] = atc[:sub_code_level]
            result.append(entry)
    return result


def _standardize_procedures(raw_procs: List[dict], sub_code_level: int = 0) -> List[dict]:
    """Convert raw procedure codes to category names via prefix matching.

    Input:  [{'timestamp': ..., 'code': 'KNGJ22'}, ...]
    Output: [{'timestamp': ..., 'value': 'orto'}, ...]

    When *sub_code_level* > 0, also preserves a truncated procedure code.
    """
    result = []
    for p in raw_procs:
        ts = p['timestamp']
        code = str(p.get('code', p.get('value', '')))
        for prefix in PROCEDURE_PREFIXES:
            if code.startswith(prefix):
                entry = {'timestamp': ts, 'value': PROCEDURE_MAP[prefix]}
                if sub_code_level > 0:
                    entry['sub_code'] = code[:sub_code_level]
                result.append(entry)
                break
    return result


def _standardize_adt(raw_adt: List[dict]) -> List[dict]:
    """Convert raw ADT events with department names to location types.

    Input:  [{'start': ..., 'end': ..., 'department': 'RH TRAUMECENTER'}, ...]
    Output: [{'start': ..., 'end': ..., 'value': 'TC'}, ...]
    """
    result = []
    for evt in raw_adt:
        dept = evt.get('department', evt.get('value', ''))
        location = classify_department(dept)
        if location is not None:
            result.append({
                'start': evt['start'],
                'end': evt['end'],
                'value': location,
            })
    return result


# ============================================================================
# Raw EHR → standardized conversion
# ============================================================================

def prepare_from_raw_ehr(
    raw_ehr: dict,
    bundle: dict,
) -> dict:
    """
    Full pipeline from raw hospital EHR data to model-ready tensors.

    Accepts raw Danish clinical data (parameter names, ATC codes, procedure
    codes, department names) and applies all mapping/filtering from the
    training pipeline before producing tensors.

    Args:
        raw_ehr: Dict with raw patient data:
            - pid: Patient identifier
            - admission_time: Trajectory start
            - current_time: Time of inference

            Demographics (provide what's available):
            - sex: 'Mand'/'Kvinde' or 'Male'/'Female' or 'M'/'F'
            - dob: Date of birth (str or Timestamp) — for AGE computation
            - age: Age in years (alternative to dob)
            - first_department: First department name (e.g., 'RH TRAUMECENTER')
            - first_hospital: Already-standardized hospital (alternative)
            - height_cm: Height in cm (or None)
            - weight_kg: Weight in kg (or None)
            - elixhauser_score: Pre-computed Elixhauser score (or None)

            Clinical data (Danish raw format accepted):
            - vitals: [{timestamp, parameter, value}, ...]
            - labs: [{timestamp, test_name, value}, ...]
            - icu_scores: [{timestamp, measurement, value}, ...]
            - medications: [{timestamp, atc_code}, ...]
            - procedures: [{timestamp, code}, ...]
            - adt: [{start, end, department}, ...]

        bundle: Deployment bundle from load_deployment_bundle().

    Returns:
        Same as prepare_single_patient(): dict with x_ts, x_ts_cat, tab_df,
        trajectory_length, bin_df.
    """
    logger.info("prepare_from_raw_ehr: pid=%s admission=%s current=%s",
                raw_ehr.get('pid', '?'), raw_ehr.get('admission_time'),
                raw_ehr.get('current_time'))
    admission_time = pd.Timestamp(raw_ehr['admission_time'])

    # AGE
    age = raw_ehr.get('age')
    if age is None and raw_ehr.get('dob') is not None:
        dob = pd.Timestamp(raw_ehr['dob'])
        age = int((admission_time - dob).days / 365.25)

    # SEX
    sex_raw = raw_ehr.get('sex', np.nan)
    sex = SEX_MAP.get(str(sex_raw), sex_raw)

    # FIRST_HOSPITAL
    hosp = raw_ehr.get('first_hospital')
    if hosp is None and raw_ehr.get('first_department'):
        hosp = derive_first_hospital(raw_ehr['first_department'])

    # Map API short keys → concept names used throughout the pipeline.
    raw_data = {
        'pid': raw_ehr.get('pid'),
        'admission_time': raw_ehr['admission_time'],
        'current_time': raw_ehr['current_time'],
        'demographics': {
            'AGE': age,
            'SEX': sex,
            'FIRST_HOSPITAL': hosp,
            'HEIGHT': raw_ehr.get('height_cm'),
            'WEIGHT': raw_ehr.get('weight_kg'),
            'ASMT_ELIX': raw_ehr.get('elixhauser_score'),
        },
        'VitaleVaerdier': _standardize_vitals(raw_ehr.get('vitals', [])),
        'InvasiveMonitoring': _extract_invasive_events(raw_ehr.get('vitals', [])),
        'Labsvar': _standardize_labs(raw_ehr.get('labs', [])),
        'ITAOversigtsrapport': _standardize_icu(raw_ehr.get('icu_scores', [])),
        'Medicin': _standardize_medications(
            raw_ehr.get('medications', []),
            sub_code_level=_get_inference_sub_code_level(bundle, 'Medicin'),
        ),
        'Procedurer': _standardize_procedures(
            raw_ehr.get('procedures', []),
            sub_code_level=_get_inference_sub_code_level(bundle, 'Procedurer'),
        ),
        'ADTHaendelser': _standardize_adt(raw_ehr.get('adt', [])),
        'EWS': _standardize_ews(raw_ehr.get('ews', [])),
    }

    return prepare_single_patient(raw_data, bundle)


# ============================================================================
# CSV-based single-patient pipeline (deployment)
# ============================================================================

def prepare_patient_from_csv(
    cpr_hash: str,
    service_date,
    current_time,
    bundle: dict,
    cfg: dict = None,
    data_dir: str = 'data/raw',
    ebm_models_dir: str = 'models/ebm',
    patient_dir: str = 'data/patients',
) -> dict:
    """
    Full pipeline from raw CSV files to model-ready tensors for a single patient.

    Reads EHR CSVs from data_dir, applies the same filtering and processing
    as the batch training pipeline (build_patient_info → filters → binning →
    aggregation), then produces tensors via prepare_single_patient().

    If the deployed model expects EBM predictions (_ebm_pred channel),
    computes them on-the-fly at relevant time intervals and injects into x_ts.

    Args:
        cpr_hash: Patient identifier (hashed CPR number).
        service_date: Trauma call date/time (str or pd.Timestamp).
        current_time: Time of inference — controls trajectory end and how
            many time bins are created (str or pd.Timestamp).
        bundle: Deployment bundle from load_deployment_bundle().
        cfg: Config dict. Defaults to loading configs/defaults.yaml.
        data_dir: Directory containing raw CSV files.
        ebm_models_dir: Directory containing trained EBM deployment models.
        patient_dir: Directory with pre-split per-patient CSVs. Falls back
            to data_dir if per-patient files are not found.

    Returns:
        Same as prepare_single_patient(): dict with x_ts, x_ts_cat, tab_df,
        trajectory_length, bin_df.
    """
    if cfg is None:
        from astra.utils import get_cfg
        cfg = get_cfg()

    # Phase 1: Build base_df
    base_df = _build_single_patient_base_df(cpr_hash, service_date, cfg, data_dir,
                                            patient_dir=patient_dir)
    logger.info(
        f"Built base_df for patient {cpr_hash[:8]}...: "
        f"trajectory {base_df['start'].iloc[0]} → {base_df['end'].iloc[0]}"
    )

    # Clamp current_time to patient's actual trajectory end so that
    # visible bins match the batch path (which is bounded by data extent).
    patient_end = base_df['end'].iloc[0]
    if pd.notna(patient_end):
        clamped = min(pd.Timestamp(current_time), pd.Timestamp(patient_end))
        if clamped < pd.Timestamp(current_time):
            logger.info(f"Clamped current_time from {current_time} to {clamped} (patient end)")
        current_time = clamped

    # Phase 2: Filter concepts
    filtered_concepts = _filter_concepts_for_patient(base_df, cfg, data_dir,
                                                     patient_dir=patient_dir)
    logger.info(
        f"Filtered {len(filtered_concepts)} concepts: "
        f"{list(filtered_concepts.keys())}"
    )

    # Phase 3: Convert to raw_data dict
    raw_data = _filtered_dfs_to_raw_data(base_df, filtered_concepts, current_time, cfg)

    # Phase 4: Build tensors via existing prepare_single_patient
    result = prepare_single_patient(raw_data, bundle)

    # Phase 5: Inject EBM predictions if model expects them
    if '_ebm_pred' in bundle.get('ts_channel_names', []):
        from astra.inference.ebm import compute_ebm_predictions, inject_ebm_into_x_ts

        ebm_preds = compute_ebm_predictions(
            raw_data, filtered_concepts, base_df, cfg, ebm_models_dir
        )
        result['x_ts'] = inject_ebm_into_x_ts(
            result['x_ts'], ebm_preds, result['bin_df'],
            raw_data['admission_time'], bundle,
            trajectory_length=result['trajectory_length'],
        )

    return result


# ---- Phase 1: Build base_df ------------------------------------------------

def _build_single_patient_base_df(
    cpr_hash: str,
    service_date,
    cfg: dict,
    data_dir: str,
    patient_dir: str = 'data/patients',
) -> pd.DataFrame:
    """
    Build a 1-row base_df for a single patient, reusing batch pipeline functions.

    Mirrors create_base_df() from build_patient_info.py but avoids Azure
    dependencies and file I/O for intermediate results.
    """
    import astra.data.build_patient_info as bpi
    from astra.utils import ensure_datetime, inches_to_cm, ounces_to_kg
    from astra.inference.patient_store import load_patient_csv

    service_date = pd.Timestamp(service_date)

    # 1. Create single-patient population
    population = pd.DataFrame({
        'CPR_hash': [cpr_hash],
        'ServiceDate': [service_date],
    })

    # 2. Load ADT events, filter to this patient
    df_ad = load_patient_csv(
        cpr_hash, 'ADTHaendelser', data_dir, patient_dir,
        dtype={"CPR_hash": str}, index_col=0,
    )
    df_ad = df_ad[df_ad['CPR_hash'] == cpr_hash].copy()
    df_ad[["Flyt_ind", "Flyt_ud"]] = df_ad[["Flyt_ind", "Flyt_ud"]].apply(
        pd.to_datetime, format="mixed", errors="coerce"
    )
    df_ad.loc[
        df_ad.ADT_haendelse == "Flyt Ind", "Flyt_ind"
    ] += pd.Timedelta(seconds=1)
    df_ad = df_ad.sort_values(["CPR_hash", "Flyt_ind"]).reset_index(drop=True)

    # 3. Build trajectories (reuse existing)
    of = bpi.build_trajectories(df_ad)

    # 4. Match ServiceDate to trajectory (reuse existing)
    population = ensure_datetime(population, "ServiceDate")
    matched = bpi.match_population_to_trajectories(of, population)

    # 5. First contacts and hospital (reuse existing)
    merged_df = bpi.add_first_contacts(matched, df_ad)
    merged_df = bpi.add_first_hospital(merged_df)

    # 6. Load patient info directly (skip Azure parquet filter)
    pi = load_patient_csv(cpr_hash, 'PatientInfo', data_dir, patient_dir, index_col=0)
    pi = pi.rename(columns={"Fødselsdato": "DOB", "Dødsdato": "DOD", "Køn": "SEX"})
    pi["SEX"] = pi["SEX"].replace({"Mand": "Male", "Kvinde": "Female"})
    result = merged_df.merge(
        pi[["CPR_hash", "DOB", "DOD", "SEX"]], on="CPR_hash", how="left"
    )

    # 7. Assign PID (deterministic inference PID from CPR_hash + ServiceDate)
    from astra.utils import make_inference_pid
    result["PID"] = make_inference_pid(cpr_hash, service_date)

    # 8. Cleanup (reuse existing)
    result = bpi.final_cleanup(result)

    # 9. Compute statics inline (avoids add_to_base's Height_Weight pickle I/O)
    result["start"] = pd.to_datetime(result["start"])
    result["end"] = pd.to_datetime(result["end"])
    result["DOB"] = pd.to_datetime(result["DOB"], errors='coerce')
    result["DOD"] = pd.to_datetime(result["DOD"], errors='coerce')

    result["DURATION"] = (
        (result["end"] - result["start"]) / np.timedelta64(1, "D")
    )
    result["AGE"] = np.floor(
        (result["start"] - result["DOB"]).dt.days / 365.25
    ).astype(int)

    # HEIGHT / WEIGHT from VitaleVaerdier
    result = _extract_height_weight(result, data_dir, patient_dir=patient_dir)

    # Mortality labels (inference: patient is alive)
    result["deceased_30d"] = 0
    result["deceased_90d"] = 0

    # LVL1TC (Level 1 Trauma Center)
    result["LVL1TC"] = 0
    if "first_RH" in result.columns:
        result.loc[result["first_RH"].notnull(), "LVL1TC"] = 1

    # 10. Elixhauser comorbidity score (pure Python, no file I/O)
    result = _try_add_elixhauser(result, data_dir=data_dir, patient_dir=patient_dir)

    # 11. Prehospital start — aligns bin grid with batch pipeline
    if cfg.get("prehospital"):
        result = _apply_prehospital_start(result, cpr_hash, cfg)

    return result


def _apply_prehospital_start(
    result: pd.DataFrame,
    cpr_hash: str,
    cfg: dict,
) -> pd.DataFrame:
    """Look up prehospital_start from the batch base_df and align bin grid.

    When ``cfg["prehospital"]`` is enabled, the batch pipeline sets
    ``start = min(prehospital_start, inhospital_start)`` so that the bin
    grid begins at the earliest prehospital encounter.  The inference
    pipeline must do the same to keep bin positions aligned.

    Reads the pre-built batch base_df (already processed by
    :func:`~astra.data.prehospital.run_prehospital_pipeline` on Azure)
    and copies ``prehospital_start`` + ABCD columns into the inference
    base_df.

    External deployments have no cohort base_df; a registered
    :class:`~astra.inference.datasource.PatientDataSource` may instead expose
    ``fetch_prehospital(cpr_hash)`` returning ``{'prehospital_start': ts,
    'A': …, 'B': …, 'C': …, 'D': …}`` (or ``None``), which takes precedence
    here.
    """
    import os
    from astra.inference.patient_store import get_data_source

    source = get_data_source()
    fetch_ph = getattr(source, 'fetch_prehospital', None) if source is not None else None
    if fetch_ph is not None:
        rec = fetch_ph(cpr_hash)
        result["inhospital_start"] = result["start"].copy()
        ph_start = rec.get("prehospital_start") if rec else None
        result["prehospital_start"] = ph_start
        if ph_start is not None and pd.notna(ph_start):
            ph_start = pd.Timestamp(ph_start)
            result["start"] = result["start"].apply(
                lambda s: min(ph_start, pd.Timestamp(s))
            )
            logger.info(
                f"Prehospital start from data source: {ph_start} "
                f"(shifted bin grid by "
                f"{(result['inhospital_start'].iloc[0] - ph_start).total_seconds() / 60:.0f} min)"
            )
        else:
            logger.info("Data source has no prehospital record — start unchanged")
        if rec:
            for col in ['A', 'B', 'C', 'D']:
                if col in rec:
                    result[col] = rec[col]
        return result

    batch_base_path = cfg.get("base_df_path", "data/interim/base_df.pkl")
    if not os.path.isfile(batch_base_path):
        logger.warning(
            f"Prehospital enabled but batch base_df not found at "
            f"{batch_base_path} — bin grid may be misaligned"
        )
        return result

    batch_base = pd.read_pickle(batch_base_path)

    if "prehospital_start" not in batch_base.columns:
        logger.info("Batch base_df has no prehospital_start column — skipping")
        return result

    # Match by CPR_hash (and trajectory overlap if multiple rows)
    cpr_match = batch_base[batch_base["CPR_hash"] == cpr_hash]
    if cpr_match.empty:
        logger.info(f"Patient {cpr_hash[:8]}... not found in batch base_df")
        return result

    # If multiple trajectories for same CPR_hash, pick the one overlapping
    # with the inference trajectory's start time
    if len(cpr_match) > 1:
        inf_start = result["start"].iloc[0]
        cpr_match = cpr_match[
            (cpr_match["start"] <= inf_start + pd.Timedelta(hours=1))
            & (cpr_match["end"] >= inf_start - pd.Timedelta(hours=1))
        ]
        if cpr_match.empty:
            logger.warning("No matching trajectory in batch base_df for this patient")
            return result

    row = cpr_match.iloc[0]
    ph_start = row.get("prehospital_start")

    result["inhospital_start"] = result["start"].copy()
    result["prehospital_start"] = ph_start

    if pd.notna(ph_start):
        ph_start = pd.Timestamp(ph_start)
        result["start"] = result["start"].apply(
            lambda s: min(ph_start, pd.Timestamp(s))
        )
        logger.info(
            f"Prehospital start: {ph_start} "
            f"(shifted bin grid by "
            f"{(result['inhospital_start'].iloc[0] - ph_start).total_seconds() / 60:.0f} min)"
        )
    else:
        logger.info("Patient has no prehospital data — start unchanged")

    # Copy ABCD categorical columns if present in batch base_df
    for col in ['A', 'B', 'C', 'D']:
        if col in row.index:
            result[col] = row[col]

    # Store batch PID so prehospital pkl data can be matched later.
    # Batch PIDs are sequential integers; inference PIDs are string hashes.
    if "PID" in row.index:
        result["_batch_pid"] = row["PID"]

    return result


def _extract_height_weight(base_df: pd.DataFrame, data_dir: str,
                           patient_dir: str = 'data/patients') -> pd.DataFrame:
    """
    Extract HEIGHT/WEIGHT from VitaleVaerdier.csv for this patient.

    Mirrors prepare_height_weight() from build_patient_info.py but operates
    in-memory without writing to data/interim/Height_Weight.pkl.
    """
    from astra.utils import inches_to_cm, ounces_to_kg
    from astra.inference.patient_store import load_patient_csv

    try:
        cpr_hash = base_df["CPR_hash"].iloc[0]
        vit_raw = load_patient_csv(cpr_hash, 'VitaleVaerdier', data_dir,
                                   patient_dir, index_col=0)
        vit_raw = vit_raw[vit_raw["CPR_hash"].isin(base_df["CPR_hash"].unique())]
        if len(vit_raw) == 0:
            base_df["HEIGHT"] = np.nan
            base_df["WEIGHT"] = np.nan
            return base_df  
        else:
            hw_map = {"Højde": "HEIGHT", "Vægt": "WEIGHT"}
            vit_raw.rename(
                columns={
                    "Værdi": "VALUE",
                    "Vital_parametre": "FEATURE",
                    "Registreringstidspunkt": "TIMESTAMP",
                },
                inplace=True,
            )
            vit_raw["FEATURE"] = vit_raw["FEATURE"].replace(to_replace=hw_map)
            vit_raw["VALUE"] = pd.to_numeric(vit_raw["VALUE"], errors="coerce")
            vit_raw = vit_raw.dropna(subset=["VALUE"])
            vit_raw.loc[vit_raw.FEATURE == "HEIGHT", "VALUE"] = inches_to_cm(
                vit_raw[vit_raw.FEATURE == "HEIGHT"].VALUE.astype(float)
            )
            vit_raw.loc[vit_raw.FEATURE == "WEIGHT", "VALUE"] = ounces_to_kg(
                vit_raw[vit_raw.FEATURE == "WEIGHT"].VALUE.astype(float)
            )

            hw = vit_raw[vit_raw.FEATURE.isin(["HEIGHT", "WEIGHT"])].copy()
            hw = hw.merge(
                base_df[["PID", "CPR_hash", "start", "end"]],
                on="CPR_hash",
                how="inner",
            )
            hw["TIMESTAMP"] = pd.to_datetime(hw["TIMESTAMP"])
            hw = hw[hw.TIMESTAMP <= hw.end]
            hw = hw.sort_values(
                ["CPR_hash", "TIMESTAMP"], ascending=False
            ).drop_duplicates(subset=["CPR_hash", "FEATURE"], keep="first")

            if len(hw) > 0:
                pivot = hw.pivot(
                    index="PID", columns="FEATURE", values="VALUE"
                ).reset_index()
                base_df = base_df.merge(pivot, how="left", on="PID")

            if "HEIGHT" not in base_df.columns:
                base_df["HEIGHT"] = np.nan
            if "WEIGHT" not in base_df.columns:
                base_df["WEIGHT"] = np.nan

    except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Could not extract HEIGHT/WEIGHT: {e}")
            base_df["HEIGHT"] = np.nan
            base_df["WEIGHT"] = np.nan

    return base_df


def _try_add_elixhauser(
    base_df: pd.DataFrame,
    data_dir: str = 'data/raw',
    patient_dir: str = 'data/patients',
) -> pd.DataFrame:
    """
    Compute Elixhauser score using pure-Python implementation.

    Replaces the previous R subprocess chain (prepare_elix_df → R script →
    computed_elix_df.csv) with an in-memory computation that does not write
    to any shared files.
    """
    from astra.inference.comorbidity import compute_elixhauser_for_patient

    try:
        return compute_elixhauser_for_patient(base_df, data_dir,
                                              patient_dir=patient_dir)
    except Exception as e:
        logger.warning(
            f"Elixhauser computation failed ({e}). Setting ASMT_ELIX=0.0."
        )
        base_df["ASMT_ELIX"] = 0.0
        return base_df


# ---- Phase 2: Filter concepts ----------------------------------------------

def _match_computed_iss(iss_df, cpr_hash, inference_pid, inference_start):
    """Find this patient's row(s) in the R-computed ISS frame.

    ``computed_iss_df.csv`` is keyed by *cohort* PIDs (sequential integers
    from the batch base_df), while the inference base_df carries the
    deterministic ``make_inference_pid`` string — a direct PID match can
    never succeed. Match by ``CPR_hash`` instead (the R output retains it),
    picking the encounter whose trajectory ``start`` is nearest when the
    patient has several. The direct-PID match is kept as a fallback for
    frames regenerated with inference PIDs.
    """
    if 'CPR_hash' in iss_df.columns:
        rows = iss_df[iss_df['CPR_hash'] == cpr_hash]
        if len(rows) > 1 and 'start' in rows.columns:
            starts = pd.to_datetime(rows['start'], errors='coerce')
            deltas = (starts - pd.Timestamp(inference_start)).abs()
            if deltas.notna().any():
                rows = rows.loc[[deltas.idxmin()]]
                logger.info(
                    "ISS_computed: %d encounters for patient — picked the one "
                    "with trajectory start nearest %s", len(deltas), inference_start,
                )
        if not rows.empty:
            return rows
    return iss_df[iss_df['PID'] == inference_pid]


def _filter_concepts_for_patient(
    base_df: pd.DataFrame,
    cfg: dict,
    data_dir: str,
    patient_dir: str = 'data/patients',
) -> Dict[str, pd.DataFrame]:
    """
    Filter raw concept CSVs for a single patient.

    For each concept in cfg['concepts']:
      1. Loads the raw CSV from data_dir (or per-patient dir if available)
      2. Calls filter_inhospital() to keep data within the patient's trajectory
      3. Calls the concept-specific filter (filter_vitals, filter_labs, etc.)

    Returns:
        Dict mapping concept name → filtered DataFrame with standardized
        columns (TIMESTAMP, PID, FEATURE, VALUE, and END_TIMESTAMP for ADT).
    """
    from astra.data.filters import (
        filter_inhospital, collect_filter, filter_adt as _filter_adt,
    )
    from astra.inference.patient_store import load_patient_csv

    metadata = pd.read_csv("data/external/metadata.csv")
    filtered = {}
    patient_cpr = base_df['CPR_hash'].iloc[0]
    concept_timing = {}  # per-concept load+filter timing

    # Concepts that are derived from Notater (clinical notes), not raw CSVs
    _NOTES_DERIVED_CONCEPTS = {'ISS_notes', 'ISS_computed', 'Events'}
    _NOTES_AUGMENTED_CONCEPTS = {'ITAOversigtsrapport'}
    _ALL_NOTES_CONCEPTS = _NOTES_DERIVED_CONCEPTS | _NOTES_AUGMENTED_CONCEPTS

    # Pre-load and filter Notater once if any notes-dependent concepts are needed
    # (mirrors mapper.py:886-888 loading Notater.pkl once)
    notater_inhospital = None
    if _ALL_NOTES_CONCEPTS & set(cfg['concepts']):
        notater_meta = metadata[metadata['filename'] == 'Notater']
        if not notater_meta.empty:
            try:
                with timed_stage(concept_timing, 'load_Notater'):
                    notater_raw = load_patient_csv(
                        patient_cpr, 'Notater', data_dir, patient_dir,
                        low_memory=False, index_col=0,
                    )
                # Pre-filter by CPR_hash (no-op for per-patient files)
                if 'CPR_hash' in notater_raw.columns:
                    notater_raw = notater_raw[notater_raw['CPR_hash'] == patient_cpr]
                if not notater_raw.empty:
                    n_dt = str(notater_meta['dt_colname'].iat[0])
                    n_offset = int(notater_meta['ts_offset'].iat[0])
                    notater_inhospital = filter_inhospital(
                        base_df, notater_raw, cfg, n_dt, offset=n_offset
                    )
                    if notater_inhospital.empty:
                        notater_inhospital = None
                    else:
                        logger.info(
                            f"Loaded Notater: {len(notater_inhospital)} rows for patient"
                        )
            except FileNotFoundError:
                logger.warning(
                    "Notater CSV not found — notes-based features will be unavailable"
                )

    for concept in cfg['concepts']:
        # --- Notes-derived concepts: built entirely from Notater, no CSV ---
        if concept == 'ISS_notes':
            from astra.data.notes_features import build_iss_from_notes
            filter_fn = collect_filter(concept)
            if notater_inhospital is not None and not notater_inhospital.empty:
                iss_notes = build_iss_from_notes(notater_inhospital)
                if not iss_notes.empty:
                    concept_filtered = filter_fn(iss_notes)
                    if not concept_filtered.empty:
                        filtered[concept] = concept_filtered
                        logger.info(f"ISS_notes: {len(concept_filtered)} row(s)")
            continue

        if concept == 'ISS_computed':
            filter_fn = collect_filter(concept)

            # A registered data source is authoritative for ISS_computed
            # (external deployments have no data/interim/ R output). Contract:
            # fetch(cpr_hash, 'ISS_computed') returns a frame with a numeric
            # 'VALUE' column (first valid value wins), or None when absent.
            from astra.inference.patient_store import get_data_source
            source = get_data_source()
            if source is not None:
                try:
                    iss_src = source.fetch(patient_cpr, 'ISS_computed')
                except Exception as e:
                    iss_src = None
                    logger.warning(f"Data source ISS_computed fetch failed: {e}")
                if iss_src is not None and not iss_src.empty and 'VALUE' in iss_src.columns:
                    iss_vals = pd.to_numeric(iss_src['VALUE'], errors='coerce').dropna()
                    if not iss_vals.empty:
                        iss_r = pd.DataFrame({
                            'PID': [base_df['PID'].iloc[0]],
                            'TIMESTAMP': [base_df['start'].iloc[0]],
                            'FEATURE': ['ISS_computed'],
                            'VALUE': [float(iss_vals.iloc[0])],
                        })
                        concept_filtered = filter_fn(iss_r)
                        if not concept_filtered.empty:
                            filtered[concept] = concept_filtered
                            logger.info(f"ISS_computed (data source): {iss_vals.iloc[0]}")
                continue

            iss_r_csv = "data/interim/computed_iss_df.csv"
            if os.path.exists(iss_r_csv):
                try:
                    iss_r_full = pd.read_csv(iss_r_csv, low_memory=False)
                    patient_pid = base_df['PID'].iloc[0]
                    iss_r_patient = _match_computed_iss(
                        iss_r_full, patient_cpr, patient_pid,
                        base_df['start'].iloc[0],
                    )
                    if not iss_r_patient.empty:
                        riss = pd.to_numeric(iss_r_patient.get('riss'), errors='coerce').iloc[0]
                        niss = pd.to_numeric(iss_r_patient.get('niss'), errors='coerce').iloc[0]
                        iss_val = riss if pd.notna(riss) else niss
                        if pd.notna(iss_val):
                            # Use admission time as timestamp for inference
                            # (batch pipeline uses latest diagnosis date)
                            iss_r = pd.DataFrame({
                                'PID': [patient_pid],
                                'TIMESTAMP': [base_df['start'].iloc[0]],
                                'FEATURE': ['ISS_computed'],
                                'VALUE': [float(iss_val)],
                            })
                            concept_filtered = filter_fn(iss_r)
                            if not concept_filtered.empty:
                                filtered[concept] = concept_filtered
                                logger.info(f"ISS_computed: {iss_val}")
                except Exception as e:
                    logger.warning(f"Failed to load R-computed ISS: {e}")
            continue

        if concept == 'InvasiveMonitoring':
            # Derived from VitaleVaerdier — saved by filter_vitals() above
            inv_pkl = "data/interim/concepts/InvasiveMonitoring.pkl"
            if os.path.exists(inv_pkl):
                concept_filtered = pd.read_pickle(inv_pkl)
                patient_pid = base_df['PID'].iloc[0]
                if 'PID' in concept_filtered.columns:
                    concept_filtered = concept_filtered[
                        concept_filtered['PID'] == patient_pid
                    ]
                if not concept_filtered.empty:
                    filtered[concept] = concept_filtered
                    logger.info(
                        f"InvasiveMonitoring: {len(concept_filtered)} events"
                    )
            else:
                logger.debug("InvasiveMonitoring pkl not found — skipping")
            continue

        if concept == 'Events':
            # Cardiac arrest + intubation from notes (mirrors mapper.py)
            if notater_inhospital is not None and not notater_inhospital.empty:
                from astra.data.cardiac_arrest import build_cardiac_arrest_from_notes
                from astra.data.notes_features import build_intubation_from_notes
                filter_fn = collect_filter(concept)
                ca_df = build_cardiac_arrest_from_notes(notater_inhospital)
                intub_df = build_intubation_from_notes(notater_inhospital)
                # Restrict intubation to within 24h of admission
                admission_start = base_df['start'].iloc[0]
                if not intub_df.empty:
                    intub_df = intub_df[
                        pd.to_datetime(intub_df['TIMESTAMP'])
                        <= admission_start + pd.Timedelta(hours=24)
                    ]
                concept_filtered = pd.concat(
                    [ca_df, intub_df], ignore_index=True
                ).reset_index(drop=True)
                # Reformat for categorical: FEATURE=constant, VALUE=event_type
                concept_filtered["VALUE"] = concept_filtered["FEATURE"]
                concept_filtered["FEATURE"] = "event"
                concept_filtered = filter_fn(concept_filtered)
                if not concept_filtered.empty:
                    filtered[concept] = concept_filtered
                    logger.info(f"Events from notes: {len(concept_filtered)} rows")
            else:
                logger.info("No Notater data — skipping Events")
            continue

        # --- Standard concepts: loaded from raw CSV ---
        meta_row = metadata[metadata['filename'] == concept]
        if meta_row.empty:
            logger.warning(f"No metadata for concept '{concept}', skipping")
            continue

        dt_name = str(meta_row['dt_colname'].iat[0])
        offset = int(meta_row['ts_offset'].iat[0])

        try:
            with timed_stage(concept_timing, f'load_{concept}'):
                raw_df = load_patient_csv(
                    patient_cpr, concept, data_dir, patient_dir,
                    low_memory=False, index_col=0,
                )
        except FileNotFoundError:
            logger.warning(f"CSV not found for '{concept}', skipping")
            continue

        # Pre-filter by CPR_hash (no-op for per-patient files, safety net for fallback)
        if 'CPR_hash' in raw_df.columns:
            raw_df = raw_df[raw_df['CPR_hash'] == patient_cpr]
            if raw_df.empty:
                logger.info(f"No {concept} data for this patient (CPR pre-filter)")
                continue

        # Time-filter to this patient's trajectory
        inhospital = filter_inhospital(base_df, raw_df, cfg, dt_name, offset=offset)

        if inhospital.empty:
            logger.info(f"No {concept} data for this patient after time filter")
            continue

        # Apply concept-specific filter
        # ADTHaendelser needs explicit base_df to avoid get_base_df() disk I/O.
        # All other concepts use the batch filter functions via collect_filter().
        patient_pids = set(inhospital['PID'].unique()) if 'PID' in inhospital.columns else None
        # Batch filters concat prehospital pkl data keyed by batch PIDs
        # (sequential integers), which differ from inference PIDs (string hashes).
        # Include the batch PID so the patient's prehospital rows survive filtering.
        if patient_pids is not None and '_batch_pid' in base_df.columns:
            batch_pid = base_df['_batch_pid'].iloc[0]
            if pd.notna(batch_pid):
                patient_pids.add(batch_pid)

        filter_fn = collect_filter(concept)

        if concept == 'ADTHaendelser':
            concept_filtered = _filter_adt(inhospital, base_df=base_df)
        elif concept == 'VitaleVaerdier' and 'EWS' in cfg.get('concepts', []):
            # Cross-concept augmentation: merge EWS vitals into VitaleVaerdier
            # (mirrors batch pipeline in mapper.py:891-894)
            ews_meta = metadata[metadata['filename'] == 'EWS']
            if not ews_meta.empty:
                ews_dt = str(ews_meta['dt_colname'].iat[0])
                ews_offset = int(ews_meta['ts_offset'].iat[0])
                try:
                    with timed_stage(concept_timing, 'load_EWS_augment'):
                        ews_raw = load_patient_csv(
                            patient_cpr, 'EWS', data_dir, patient_dir,
                            low_memory=False, index_col=0,
                        )
                    # Pre-filter by CPR_hash (no-op for per-patient files)
                    if 'CPR_hash' in ews_raw.columns:
                        ews_raw = ews_raw[ews_raw['CPR_hash'] == patient_cpr]
                    ews_inhospital = filter_inhospital(
                        base_df, ews_raw, cfg, ews_dt, offset=ews_offset
                    )
                    concept_filtered = filter_fn(inhospital, ews=ews_inhospital)
                except FileNotFoundError:
                    logger.warning(
                        "EWS CSV not found — processing VitaleVaerdier without EWS"
                    )
                    concept_filtered = filter_fn(inhospital)
            else:
                concept_filtered = filter_fn(inhospital)
        elif concept == 'ITAOversigtsrapport':
            # Normal filter first, then augment with GCS from notes
            # (mirrors mapper.py:895-901)
            concept_filtered = filter_fn(inhospital)
            if notater_inhospital is not None and not notater_inhospital.empty:
                from astra.data.notes_features import build_gcs_from_notes
                gcs_df = build_gcs_from_notes(notater_inhospital)
                if not gcs_df.empty:
                    concept_filtered = pd.concat(
                        [concept_filtered, gcs_df], ignore_index=True
                    )
                    logger.info(
                        f"Augmented ITAOversigtsrapport with "
                        f"{len(gcs_df)} GCS values from notes"
                    )
        else:
            concept_filtered = filter_fn(inhospital)

        # Batch filters may concat population-level prehospital data;
        # keep only the current patient's rows.
        if patient_pids is not None and 'PID' in concept_filtered.columns:
            concept_filtered = concept_filtered[concept_filtered['PID'].isin(patient_pids)]

        # Normalize PIDs and re-deduplicate (prehospital cross-source dedup).
        # Only needed when prehospital is enabled, since hospital CSV and
        # prehospital PKL may contribute overlapping measurements.
        # NOTE: We do NOT normalize VALUE types (string vs float) because
        # the batch pipeline preserves type distinction during dedup —
        # normalizing would over-deduplicate and lose measurements.
        if (
            cfg.get('prehospital', False)
            and patient_pids is not None
            and 'PID' in concept_filtered.columns
        ):
            inference_pid = base_df['PID'].iloc[0]
            concept_filtered = concept_filtered.copy()
            concept_filtered['PID'] = inference_pid

            dedup_cols = ['PID', 'TIMESTAMP', 'FEATURE', 'VALUE']
            if all(c in concept_filtered.columns for c in dedup_cols):
                n_before = len(concept_filtered)
                concept_filtered = concept_filtered.drop_duplicates(
                    subset=dedup_cols, keep='first'
                ).reset_index(drop=True)
                n_removed = n_before - len(concept_filtered)
                if n_removed:
                    logger.debug(
                        f"{concept}: PID-normalized dedup removed {n_removed} rows"
                    )

        if concept_filtered.empty:
            logger.info(f"No {concept} data after concept filter")
            continue

        filtered[concept] = concept_filtered

    # Log per-concept timing breakdown
    if concept_timing:
        parts = []
        for stage, durations in sorted(concept_timing.items()):
            total_ms = sum(durations) * 1000
            parts.append(f"{stage}={total_ms:.0f}ms")
        logger.info("Concept load timing: %s", ', '.join(parts))

    return filtered


# ---- Phase 3: Convert to raw_data dict -------------------------------------

def _filtered_dfs_to_raw_data(
    base_df: pd.DataFrame,
    filtered_concepts: Dict[str, pd.DataFrame],
    current_time,
    cfg: dict,
    filter_by_time: bool = False,
) -> dict:
    """
    Convert filtered concept DataFrames + base_df into the raw_data dict
    format expected by prepare_single_patient().

    Uses concept names as dict keys (config-driven). Continuous vs categorical
    classification is derived from ``cfg['dataset']['ts_cat_names']``.
    Interval vs point events are detected from DataFrame columns.

    Args:
        cfg: Configuration dictionary (needs ``dataset.ts_cat_names``).
        filter_by_time: When True, only include records with timestamps
            ``<= current_time``.  ADT intervals that started before
            *current_time* are kept but their end is clamped.
    """
    cutoff = pd.Timestamp(current_time) if filter_by_time else None
    row = base_df.iloc[0]
    ts_cat_names = set(cfg.get('dataset', {}).get('ts_cat_names', []))

    raw_data = {
        'pid': row.get('PID', 1),
        'admission_time': row['start'],
        'current_time': pd.Timestamp(current_time),
        # Include all base_df columns as demographics so _build_tab_df()
        # can find any tabular feature the model needs (e.g., prehospital ABCD).
        'demographics': {col: row[col] for col in row.index},
    }

    drop_features = cfg.get('drop_features', {})

    for concept, df in filtered_concepts.items():
        if df.empty:
            continue

        # Apply drop_features filter (mirrors batch pipeline in datasets.py).
        # For continuous concepts, drop_features matches on FEATURE column.
        # For categorical concepts (e.g. ADTHaendelser), FEATURE is constant
        # (e.g. "ADT") so we filter on VALUE instead.
        concept_drops = drop_features.get(concept, [])
        if concept_drops:
            is_cat = concept in ts_cat_names
            col = 'VALUE' if is_cat else 'FEATURE'
            if col in df.columns:
                df = df[~df[col].isin(concept_drops)]
        if df.empty:
            continue

        is_categorical = concept in ts_cat_names
        has_intervals = 'END_TIMESTAMP' in df.columns

        if cutoff is not None:
            df = df[pd.to_datetime(df['TIMESTAMP']) <= cutoff]
            if has_intervals:
                df = df.copy()
                ends = pd.to_datetime(df['END_TIMESTAMP'])
                df['END_TIMESTAMP'] = ends.clip(upper=cutoff)

        if df.empty:
            continue

        if is_categorical and has_intervals:
            # Interval events (e.g. ADTHaendelser)
            raw_data[concept] = [
                {'start': r['TIMESTAMP'], 'end': r['END_TIMESTAMP'], 'value': r['VALUE']}
                for _, r in df.iterrows()
            ]
        elif is_categorical:
            # Categorical point events (e.g. Medicin, Procedurer)
            # Preserve ATC/dose/unit for Medicin (needed by composite features)
            has_atc = 'ATC' in df.columns
            has_dose = 'Administrationsdosis' in df.columns
            has_unit = 'Dosisenhed' in df.columns
            events = []
            for _, r in df.iterrows():
                ev = {'timestamp': r['TIMESTAMP'], 'value': r['VALUE']}
                if has_atc:
                    ev['atc_code'] = r.get('ATC', '')
                if has_dose:
                    ev['dose'] = r.get('Administrationsdosis')
                if has_unit:
                    ev['unit'] = r.get('Dosisenhed')
                events.append(ev)
            raw_data[concept] = events
        else:
            # Continuous concepts (e.g. VitaleVaerdier, Labsvar, EWS, etc.)
            raw_data[concept] = [
                {'timestamp': r['TIMESTAMP'], 'feature': r['FEATURE'], 'value': r['VALUE']}
                for _, r in df.iterrows()
            ]

    return raw_data


# ---- Phase 4: Time-filter raw_data dict ------------------------------------

def _filter_raw_data_by_time(raw_data: dict, cutoff_time) -> dict:
    """
    Return a copy of *raw_data* containing only events up to *cutoff_time*.

    - Point events (vitals, labs, icu, medications, procedures): keep
      records whose ``timestamp <= cutoff_time``.
    - Interval events (adt): keep if ``start <= cutoff_time``; clamp
      ``end`` to ``min(end, cutoff_time)`` so ongoing stays are truncated.
    - Identity / demographics / admission_time are preserved as-is.
    - ``current_time`` is set to *cutoff_time*.

    Does **not** mutate *raw_data*.
    """
    cutoff = pd.Timestamp(cutoff_time)

    filtered = {
        'pid': raw_data.get('pid'),
        'admission_time': raw_data['admission_time'],
        'current_time': cutoff,
        'demographics': raw_data.get('demographics', {}),
    }

    # Generic iteration: filter all concept event lists by time.
    # Event type is detected from data structure:
    #   - 'start' key → interval event (keep if started, clamp end)
    #   - 'timestamp' key → point event (keep if timestamp <= cutoff)
    for key, items in raw_data.items():
        if key in _RAW_DATA_META_KEYS:
            continue
        if not isinstance(items, list):
            continue

        if not items:
            filtered[key] = []
            continue

        sample = items[0]
        if 'start' in sample:
            # Interval events: keep if started, clamp end
            filtered[key] = []
            for evt in items:
                start = pd.Timestamp(evt['start'])
                if start > cutoff:
                    continue
                end = pd.Timestamp(evt['end'])
                filtered[key].append({
                    'start': evt['start'],
                    'end': min(end, cutoff),
                    'value': evt['value'],
                })
        elif 'timestamp' in sample:
            # Point events (continuous or categorical)
            filtered[key] = [
                m for m in items
                if pd.Timestamp(m['timestamp']) <= cutoff
            ]
        else:
            # Unknown event format — copy as-is
            filtered[key] = list(items)

    return filtered
