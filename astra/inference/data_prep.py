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

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

from astra.data.mappings import (
    VITALS_MAP, BP_TYPES, HEIGHT_WEIGHT_MAP, LABS_REVERSE_MAP, ICU_MAP,
    ATC_LVL3_REVERSE, ATC_LVL4_REVERSE,
    PROCEDURE_REVERSE_MAP, SEX_MAP,
    classify_department, classify_atc, derive_first_hospital, parse_numeric,
)


# ============================================================================
# Time binning
# ============================================================================

MAX_PREDICTION_WINDOW = pd.Timedelta(days=30)


def _create_patient_bins(
    admission_time: pd.Timestamp,
    data_config: dict,
) -> pd.DataFrame:
    """
    Create the full fixed-duration bin grid for a single patient trajectory.

    Always creates bins spanning [admission_time, admission_time + 30 days].
    This makes the bin grid stable across re-inferences: position N always
    maps to the same time window regardless of when inference is called.

    Use ``_count_visible_bins()`` to determine how many bins are "live"
    at a given ``current_time``.

    Returns:
        DataFrame with columns [bin_start, bin_end, bin_counter, bin_freq, position]
        where position is the 0-based contiguous index after frequency filtering.
    """
    bin_intervals = data_config['bin_intervals']
    bin_freq_include = data_config['bin_freq_include']

    start_time = admission_time
    end_time = admission_time + MAX_PREDICTION_WINDOW

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
            inclusive="left",
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

    # Validate: index in range and timestamp < bin_end
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

    # Combine all point measurements into a single DataFrame
    records = []
    for source_key in ('vitals', 'labs', 'icu'):
        for m in raw_data.get(source_key, []):
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
    _AUXILIARY = {'elapsed_hours', 'bin_width_hours', '_data_present', '_ebm_pred'}
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

    return x_ts, trajectory_length


# ============================================================================
# Categorical time series
# ============================================================================

# Mapping from raw_data keys to encoder feature names (as used in training)
_CAT_KEY_TO_FEATURE = {
    'medications': 'medication',
    'procedures': 'procedures',
    'adt': 'ADT',
}


def _build_categorical_ts(
    raw_data: dict,
    bin_df: pd.DataFrame,
    bundle: dict,
) -> np.ndarray:
    """
    Build multi-hot encoded categorical time series tensor.

    Constructs the multi-hot array directly using encoder internals
    rather than building a wide-format DataFrame.

    Returns:
        np.ndarray of shape [n_cat_dims, seq_len].
    """
    seq_len = bundle['model_params']['seq_len']
    encoding_info = bundle['encoding_info']
    cat_encoder = bundle['cat_encoder']

    # Total categorical dimensions
    total_dim = sum(
        end - start for start, end in encoding_info['feature_ranges'].values()
    )
    x_ts_cat = np.zeros((total_dim, seq_len), dtype=np.float32)

    for raw_key, encoder_feat_name in _CAT_KEY_TO_FEATURE.items():
        events = raw_data.get(raw_key, [])
        if not events:
            continue

        # Check encoder has this feature
        if encoder_feat_name not in cat_encoder.encoders_:
            logger.warning(
                f"Encoder has no feature '{encoder_feat_name}' — skipping {raw_key}"
            )
            continue

        encoder_info = cat_encoder.encoders_[encoder_feat_name]
        value_to_idx = encoder_info['value_to_idx']
        dim_start, dim_end = encoding_info['feature_ranges'][encoder_feat_name]

        # Assign events to bins
        if raw_key == 'adt':
            # Interval-based events
            intervals_df = pd.DataFrame(events)
            if intervals_df.empty:
                continue
            intervals_df['start'] = pd.to_datetime(intervals_df['start'])
            intervals_df['end'] = pd.to_datetime(intervals_df['end'])
            assigned = _expand_intervals_to_bins(intervals_df, bin_df)
        else:
            # Point events
            events_df = pd.DataFrame(events)
            if events_df.empty:
                continue
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            # These don't have a 'feature' column — add a dummy for _assign_to_bins
            events_df['feature'] = encoder_feat_name
            assigned = _assign_to_bins(events_df, bin_df)

        if assigned.empty:
            continue

        # Set multi-hot values
        for _, row in assigned.iterrows():
            pos = int(row['position'])
            if pos >= seq_len:
                continue
            val = row['value']
            if val in value_to_idx:
                idx = value_to_idx[val]
                x_ts_cat[dim_start + idx, pos] = 1.0
            else:
                logger.debug(f"Unknown category '{val}' for {encoder_feat_name} — skipping")

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

    # Determine how many bins are "visible" at current_time
    visible_bins = _count_visible_bins(bin_df, raw_data['current_time'])
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

    # Zero out bins beyond the visible horizon (guards against future data
    # leaking in when simulating with historic patients).
    seq_len = bundle['model_params']['seq_len']
    if trajectory_length < seq_len:
        x_ts_cat[:, trajectory_length:] = 0.0

    # 4. Build tabular features
    tab_df = _build_tab_df(raw_data, bundle)

    logger.info(
        f"Prepared patient {raw_data.get('pid', '?')}: "
        f"x_ts={x_ts.shape}, x_ts_cat={x_ts_cat.shape}, "
        f"trajectory_length={trajectory_length}"
    )

    return {
        'x_ts': x_ts,
        'x_ts_cat': x_ts_cat,
        'tab_df': tab_df,
        'trajectory_length': trajectory_length,
        'bin_df': bin_df,
    }


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


def _standardize_medications(raw_meds: List[dict]) -> List[dict]:
    """Convert raw ATC codes to medication category names.

    Input:  [{'timestamp': ..., 'atc_code': 'N02AB02'}, ...]
    Output: [{'timestamp': ..., 'value': 'opiods'}, ...]
    """
    result = []
    for med in raw_meds:
        ts = med['timestamp']
        atc = str(med.get('atc_code', med.get('value', '')))
        category = classify_atc(atc)
        if category is not None:
            result.append({'timestamp': ts, 'value': category})
    return result


def _standardize_procedures(raw_procs: List[dict]) -> List[dict]:
    """Convert raw procedure codes to category names.

    Input:  [{'timestamp': ..., 'code': 'KNGJ22'}, ...]
    Output: [{'timestamp': ..., 'value': 'orto_major'}, ...]
    """
    result = []
    for p in raw_procs:
        ts = p['timestamp']
        code = str(p.get('code', p.get('value', '')))
        category = PROCEDURE_REVERSE_MAP.get(code)
        if category is not None:
            result.append({'timestamp': ts, 'value': category})
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
        'vitals': _standardize_vitals(raw_ehr.get('vitals', [])),
        'labs': _standardize_labs(raw_ehr.get('labs', [])),
        'icu': _standardize_icu(raw_ehr.get('icu_scores', [])),
        'medications': _standardize_medications(raw_ehr.get('medications', [])),
        'procedures': _standardize_procedures(raw_ehr.get('procedures', [])),
        'adt': _standardize_adt(raw_ehr.get('adt', [])),
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

    Returns:
        Same as prepare_single_patient(): dict with x_ts, x_ts_cat, tab_df,
        trajectory_length, bin_df.
    """
    if cfg is None:
        from astra.utils import get_cfg
        cfg = get_cfg()

    # Phase 1: Build base_df
    base_df = _build_single_patient_base_df(cpr_hash, service_date, cfg, data_dir)
    logger.info(
        f"Built base_df for patient {cpr_hash[:8]}...: "
        f"trajectory {base_df['start'].iloc[0]} → {base_df['end'].iloc[0]}"
    )

    # Phase 2: Filter concepts
    filtered_concepts = _filter_concepts_for_patient(base_df, cfg, data_dir)
    logger.info(
        f"Filtered {len(filtered_concepts)} concepts: "
        f"{list(filtered_concepts.keys())}"
    )

    # Phase 3: Convert to raw_data dict
    raw_data = _filtered_dfs_to_raw_data(base_df, filtered_concepts, current_time)

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
) -> pd.DataFrame:
    """
    Build a 1-row base_df for a single patient, reusing batch pipeline functions.

    Mirrors create_base_df() from build_patient_info.py but avoids Azure
    dependencies and file I/O for intermediate results.
    """
    import astra.data.build_patient_info as bpi
    from astra.utils import ensure_datetime, inches_to_cm, ounces_to_kg

    service_date = pd.Timestamp(service_date)

    # 1. Create single-patient population
    population = pd.DataFrame({
        'CPR_hash': [cpr_hash],
        'ServiceDate': [service_date],
    })

    # 2. Load ADT events, filter to this patient
    df_ad = pd.read_csv(
        f"{data_dir}/ADTHaendelser.csv", dtype={"CPR_hash": str}, index_col=0
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
    pi = pd.read_csv(f"{data_dir}/PatientInfo.csv", index_col=0)
    pi = pi.rename(columns={"Fødselsdato": "DOB", "Dødsdato": "DOD", "Køn": "SEX"})
    pi["SEX"] = pi["SEX"].replace({"Mand": "Male", "Kvinde": "Female"})
    result = merged_df.merge(
        pi[["CPR_hash", "DOB", "DOD", "SEX"]], on="CPR_hash", how="left"
    )

    # 7. Assign PID (single patient)
    result["PID"] = 1

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
    result = _extract_height_weight(result, data_dir)

    # Mortality labels (inference: patient is alive)
    result["deceased_30d"] = 0
    result["deceased_90d"] = 0

    # LVL1TC (Level 1 Trauma Center)
    result["LVL1TC"] = 0
    if "first_RH" in result.columns:
        result.loc[result["first_RH"].notnull(), "LVL1TC"] = 1

    # 10. Elixhauser comorbidity score (pure Python, no file I/O)
    result = _try_add_elixhauser(result, data_dir=data_dir)

    return result


def _extract_height_weight(base_df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Extract HEIGHT/WEIGHT from VitaleVaerdier.csv for this patient.

    Mirrors prepare_height_weight() from build_patient_info.py but operates
    in-memory without writing to data/interim/Height_Weight.pkl.
    """
    from astra.utils import inches_to_cm, ounces_to_kg

    try:
        vit_raw = pd.read_csv(f"{data_dir}/VitaleVaerdier.csv", index_col=0)
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
) -> pd.DataFrame:
    """
    Compute Elixhauser score using pure-Python implementation.

    Replaces the previous R subprocess chain (prepare_elix_df → R script →
    computed_elix_df.csv) with an in-memory computation that does not write
    to any shared files.
    """
    from astra.inference.comorbidity import compute_elixhauser_for_patient

    try:
        return compute_elixhauser_for_patient(base_df, data_dir)
    except Exception as e:
        logger.warning(
            f"Elixhauser computation failed ({e}). Setting ASMT_ELIX=NaN."
        )
        base_df["ASMT_ELIX"] = np.nan
        return base_df


# ---- Phase 2: Filter concepts ----------------------------------------------

def _filter_vitals_stateless(vit: pd.DataFrame) -> pd.DataFrame:
    """
    Stateless version of filters.filter_vitals().

    Replicates the exact same logic (temp conversion, BP splitting, feature
    mapping, numeric filtering) but does NOT write Height_Weight.pkl to disk.
    """
    from astra.utils import inches_to_cm, ounces_to_kg

    vit = vit.copy()

    # Fix temperature in fahrenheit
    vit.loc[vit.Vital_parametre == 'Temp.', 'Værdi'] = vit["Værdi_Omregnet"]

    # Rename to standard columns
    vit.rename(
        columns={
            "Værdi": "VALUE",
            "Vital_parametre": "FEATURE",
            "Registreringstidspunkt": "TIMESTAMP",
        },
        inplace=True,
    )
    vit = vit[["TIMESTAMP", "PID", "FEATURE", "VALUE"]]

    # Split blood pressure into SBP/DBP
    for bt in BP_TYPES:
        mask = vit['FEATURE'] == bt
        if len(vit.loc[mask]) > 0:
            split_values = vit.loc[mask, 'VALUE'].str.split('/', n=1, expand=True)
            vit.loc[mask, 'FEATURE'] = 'SBP'
            vit.loc[mask, 'VALUE'] = split_values[0]
            diastolic_rows = vit[mask].copy()
            diastolic_rows['FEATURE'] = 'DBP'
            diastolic_rows['VALUE'] = split_values[1]
            vit = pd.concat([vit, diastolic_rows], ignore_index=True)
            vit.loc[vit['FEATURE'].isin(['SBP', 'DBP']), 'VALUE'] = pd.to_numeric(
                vit.loc[vit['FEATURE'].isin(['SBP', 'DBP']), 'VALUE'],
                errors='coerce',
            )
            vit['VALUE'] = vit['VALUE'].astype(str)

    # Map feature names (Danish → standard)
    vit["FEATURE"] = vit["FEATURE"].replace(to_replace=VITALS_MAP)
    vit["FEATURE"] = vit["FEATURE"].replace(to_replace=HEIGHT_WEIGHT_MAP)
    vit.loc[vit.FEATURE == 'HEIGHT', 'VALUE'] = inches_to_cm(
        vit[vit.FEATURE == 'HEIGHT'].VALUE.astype(float)
    )
    vit.loc[vit.FEATURE == 'WEIGHT', 'VALUE'] = ounces_to_kg(
        vit[vit.FEATURE == 'WEIGHT'].VALUE.astype(float)
    )

    # NOTE: Original filter_vitals writes Height_Weight.pkl here — we skip that.

    # Keep only vitals (not HEIGHT/WEIGHT) with valid numeric values
    pattern = r'([<>]\s*)?[-+]?\d*\.\d+|\d+\.?\d*'
    vit = vit[
        (vit.FEATURE.isin(list(set(VITALS_MAP.values()))))
        & (vit.VALUE.notnull())
        & (
            (vit['VALUE'].str.contains(pattern, regex=True))
            | (vit['VALUE'].dtype == float)
        )
    ].copy(deep=True)

    return vit


def _filter_concepts_for_patient(
    base_df: pd.DataFrame,
    cfg: dict,
    data_dir: str,
) -> Dict[str, pd.DataFrame]:
    """
    Filter raw concept CSVs for a single patient.

    For each concept in cfg['concepts']:
      1. Loads the raw CSV from data_dir
      2. Calls filter_inhospital() to keep data within the patient's trajectory
      3. Calls the concept-specific filter (filter_vitals, filter_labs, etc.)

    Returns:
        Dict mapping concept name → filtered DataFrame with standardized
        columns (TIMESTAMP, PID, FEATURE, VALUE, and END_TIMESTAMP for ADT).
    """
    from astra.data.filters import (
        filter_inhospital, collect_filter, filter_adt as _filter_adt,
    )

    metadata = pd.read_csv("data/external/metadata.csv")
    filtered = {}

    for concept in cfg['concepts']:
        # Get metadata for this concept
        meta_row = metadata[metadata['filename'] == concept]
        if meta_row.empty:
            logger.warning(f"No metadata for concept '{concept}', skipping")
            continue

        dt_name = str(meta_row['dt_colname'].iat[0])
        offset = int(meta_row['ts_offset'].iat[0])

        # Load raw CSV
        csv_path = f"{data_dir}/{concept}.csv"
        try:
            raw_df = pd.read_csv(csv_path, low_memory=False, index_col=0)
        except FileNotFoundError:
            logger.warning(f"Raw CSV not found: {csv_path}, skipping '{concept}'")
            continue

        # Time-filter to this patient's trajectory
        inhospital = filter_inhospital(base_df, raw_df, cfg, dt_name, offset=offset)

        if inhospital.empty:
            logger.info(f"No {concept} data for this patient after time filter")
            continue

        # Apply concept-specific filter
        # VitaleVaerdier uses a stateless variant to avoid writing
        # Height_Weight.pkl to data/interim/ (shared with cohort pipeline).
        if concept == 'ADTHaendelser':
            concept_filtered = _filter_adt(inhospital, base_df=base_df)
        elif concept == 'VitaleVaerdier':
            concept_filtered = _filter_vitals_stateless(inhospital)
        else:
            filter_fn = collect_filter(concept)
            concept_filtered = filter_fn(inhospital)

        if concept_filtered.empty:
            logger.info(f"No {concept} data after concept filter")
            continue

        filtered[concept] = concept_filtered

    return filtered


# ---- Phase 3: Convert to raw_data dict -------------------------------------

def _filtered_dfs_to_raw_data(
    base_df: pd.DataFrame,
    filtered_concepts: Dict[str, pd.DataFrame],
    current_time,
) -> dict:
    """
    Convert filtered concept DataFrames + base_df into the raw_data dict
    format expected by prepare_single_patient().

    The concept-specific filters have already standardized feature names
    (VITALS_MAP, LABS_REVERSE_MAP, etc.), so no further name mapping is needed.
    """
    row = base_df.iloc[0]

    raw_data = {
        'pid': row.get('PID', 1),
        'admission_time': row['start'],
        'current_time': pd.Timestamp(current_time),
        'demographics': {
            'AGE': row.get('AGE', np.nan),
            'SEX': row.get('SEX', np.nan),
            'FIRST_HOSPITAL': row.get('FIRST_HOSPITAL', np.nan),
            'HEIGHT': row.get('HEIGHT', np.nan),
            'WEIGHT': row.get('WEIGHT', np.nan),
            'ASMT_ELIX': row.get('ASMT_ELIX', np.nan),
        },
        'vitals': [],
        'labs': [],
        'icu': [],
        'medications': [],
        'procedures': [],
        'adt': [],
    }

    # Continuous concepts: TIMESTAMP, FEATURE, VALUE → {timestamp, feature, value}
    _CONTINUOUS_MAP = {
        'VitaleVaerdier': 'vitals',
        'Labsvar': 'labs',
        'ITAOversigtsrapport': 'icu',
    }
    for concept, key in _CONTINUOUS_MAP.items():
        if concept not in filtered_concepts:
            continue
        df = filtered_concepts[concept]
        raw_data[key] = [
            {'timestamp': r['TIMESTAMP'], 'feature': r['FEATURE'], 'value': r['VALUE']}
            for _, r in df.iterrows()
        ]

    # Categorical point events: TIMESTAMP, VALUE → {timestamp, value}
    if 'Medicin' in filtered_concepts:
        df = filtered_concepts['Medicin']
        raw_data['medications'] = [
            {'timestamp': r['TIMESTAMP'], 'value': r['VALUE']}
            for _, r in df.iterrows()
        ]

    if 'Procedurer' in filtered_concepts:
        df = filtered_concepts['Procedurer']
        raw_data['procedures'] = [
            {'timestamp': r['TIMESTAMP'], 'value': r['VALUE']}
            for _, r in df.iterrows()
        ]

    # Interval events: TIMESTAMP, END_TIMESTAMP, VALUE → {start, end, value}
    if 'ADTHaendelser' in filtered_concepts:
        df = filtered_concepts['ADTHaendelser']
        raw_data['adt'] = [
            {'start': r['TIMESTAMP'], 'end': r['END_TIMESTAMP'], 'value': r['VALUE']}
            for _, r in df.iterrows()
        ]

    return raw_data
