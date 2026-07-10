import logging
import operator
import os
import warnings
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

from astra.utils import get_bin_df, ensure_parent_dir, PROJECT_ROOT
from astra.data.filters import collect_filter

logger = logging.getLogger(__name__)


def get_effective_cat_cols(cfg: dict) -> list:
    """Return the full list of categorical columns, including PPJ ABCD when prehospital is enabled."""
    cat_cols = list(cfg["dataset"]["cat_cols"])
    if cfg.get("prehospital"):
        ppj_cat_cols = cfg["dataset"].get("ppj_cat_cols", [])
        cat_cols.extend(c for c in ppj_cat_cols if c not in cat_cols)
    return cat_cols


# ============================================================================
# Exclusion criteria
# ============================================================================

_CUSTOM_OPS = {
    "==": operator.eq, "!=": operator.ne,
    "<": operator.lt,  "<=": operator.le,
    ">": operator.gt,  ">=": operator.ge,
}


def resolve_exclusion_criteria(cfg: dict) -> Optional[dict]:
    """Look up the active exclusion profile from config.

    Returns the criteria dict for the selected profile, or None if exclusion
    is disabled (null / false / 0).
    """
    profile = cfg.get("dataset", {}).get("exclusion")
    if not profile:
        return None
    profiles = cfg.get("exclusion_criteria", {})
    if profile not in profiles:
        raise ValueError(
            f"Exclusion profile '{profile}' not found in exclusion_criteria. "
            f"Available: {list(profiles.keys())}"
        )
    return profiles[profile]


def _sanitize_yaml(val):
    """Normalize YAML quirks: turn the string ``"None"`` into Python ``None``.

    YAML only recognizes ``null`` / ``~`` as null — unquoted ``None`` is parsed
    as a string.  This helper is applied recursively so that any value in a
    criteria dict written as ``None`` in YAML behaves like ``null``.
    """
    if isinstance(val, str) and val.lower() == "none":
        return None
    if isinstance(val, dict):
        return {k: _sanitize_yaml(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_yaml(v) for v in val]
    return val


def apply_exclusion_criteria(
    base_df: pd.DataFrame,
    criteria: dict,
) -> pd.DataFrame:
    """Filter *base_df* according to a criteria dict.

    Each key is optional — only applied when present and has a truthy /
    non-empty value (``null`` / ``None`` → skip).  Returns a filtered **copy**.
    """
    criteria = _sanitize_yaml(criteria)
    n_before = len(base_df)
    mask = pd.Series(True, index=base_df.index)

    # --- named criteria ------------------------------------------------
    age_min = criteria.get("age_min")
    if isinstance(age_min, (int, float)):
        m = base_df["AGE"] >= age_min
        excluded = (~m & mask).sum()
        if excluded:
            logger.info(f"  exclusion  age_min >= {age_min}: -{excluded}")
        mask &= m

    age_max = criteria.get("age_max")
    if isinstance(age_max, (int, float)):
        m = base_df["AGE"] <= age_max
        excluded = (~m & mask).sum()
        if excluded:
            logger.info(f"  exclusion  age_max <= {age_max}: -{excluded}")
        mask &= m

    start_year = criteria.get("start_year")
    if isinstance(start_year, (int, float)):
        m = base_df["ServiceDate"].dt.year >= start_year
        excluded = (~m & mask).sum()
        if excluded:
            logger.info(f"  exclusion  start_year >= {start_year}: -{excluded}")
        mask &= m

    end_year = criteria.get("end_year")
    if isinstance(end_year, (int, float)):
        m = base_df["ServiceDate"].dt.year <= end_year
        excluded = (~m & mask).sum()
        if excluded:
            logger.info(f"  exclusion  end_year <= {end_year}: -{excluded}")
        mask &= m

    if criteria.get("lvl1tc"):
        m = base_df["LVL1TC"] == 1
        excluded = (~m & mask).sum()
        if excluded:
            logger.info(f"  exclusion  lvl1tc only: -{excluded}")
        mask &= m

    if criteria.get("prehospital_only"):
        # prehospital_start is NaT for patients without PPJ data
        col = "prehospital_start" if "prehospital_start" in base_df.columns else "prehospital_end"
        if col in base_df.columns:
            m = base_df[col].notna()
            excluded = (~m & mask).sum()
            if excluded:
                logger.info(f"  exclusion  prehospital_only ({col} notna): -{excluded}")
            mask &= m
        else:
            logger.warning("  exclusion  prehospital_only requested but no prehospital column found")

    if criteria.get("traumatext"):
        if "TRAUMATEXT" in base_df.columns:
            m = base_df["TRAUMATEXT"] == True
            excluded = (~m & mask).sum()
            if excluded:
                logger.info(f"  exclusion  traumatext (any time): -{excluded}")
            mask &= m
        else:
            logger.warning(
                "  exclusion  traumatext requested but TRAUMATEXT column "
                "not found in base_df. Run mark_traumatext() first."
            )

    if criteria.get("traumatext_12h"):
        if "TRAUMATEXT_12H" in base_df.columns:
            m = base_df["TRAUMATEXT_12H"] == True
            excluded = (~m & mask).sum()
            if excluded:
                logger.info(f"  exclusion  traumatext_12h: -{excluded}")
            mask &= m
        else:
            logger.warning(
                "  exclusion  traumatext_12h requested but TRAUMATEXT_12H column "
                "not found in base_df. Run mark_traumatext() first."
            )

    max_dur = criteria.get("max_duration_days")
    if isinstance(max_dur, (int, float)):
        if "DURATION" in base_df.columns:
            m = base_df["DURATION"] <= max_dur
            excluded = (~m & mask).sum()
            if excluded:
                logger.info(f"  exclusion  max_duration_days <= {max_dur}: -{excluded}")
            mask &= m
        else:
            logger.warning(
                "  exclusion  max_duration_days requested but DURATION column "
                "not found in base_df."
            )

    min_bins = criteria.get("min_bin_seq_len")
    if isinstance(min_bins, (int, float)) and min_bins > 0:
        from astra.evaluation.utils import time_to_step, get_total_steps
        if "start" in base_df.columns and "end" in base_df.columns:
            total_steps = get_total_steps()
            start = pd.to_datetime(base_df["start"])
            end = pd.to_datetime(base_df["end"])
            duration_hours = (end - start).dt.total_seconds() / 3600

            def _duration_to_bins(h):
                if h <= 0:
                    return 0
                step = time_to_step(h, 'h')
                if step is None:
                    return total_steps  # exceeds max range → full trajectory
                return step + 1

            bin_counts = duration_hours.apply(_duration_to_bins)
            m = bin_counts >= min_bins
            excluded = (~m & mask).sum()
            if excluded:
                logger.info(f"  exclusion  min_bin_seq_len >= {min_bins}: -{excluded}")
            mask &= m
        else:
            logger.warning(
                "  exclusion  min_bin_seq_len requested but start/end columns "
                "not found in base_df."
            )

    first_hospital = [v for v in (criteria.get("first_hospital") or []) if v is not None]
    if first_hospital:
        m = base_df["FIRST_HOSPITAL"].isin(first_hospital)
        excluded = (~m & mask).sum()
        if excluded:
            logger.info(f"  exclusion  first_hospital in {first_hospital}: -{excluded}")
        mask &= m

    # --- PID allowlist from external file --------------------------------
    pid_file = criteria.get("pid_file")
    if pid_file:
        pid_path = PROJECT_ROOT / pid_file
        if pid_path.exists():
            pid_df = pd.read_csv(pid_path)
            if "PID" in pid_df.columns:
                allowed_pids = pid_df["PID"]
            else:
                allowed_pids = pid_df.iloc[:, 0]
            m = base_df["PID"].isin(allowed_pids)
            excluded = (~m & mask).sum()
            if excluded:
                logger.info(f"  exclusion  pid_file ({pid_file}): -{excluded}")
            mask &= m
        else:
            logger.warning(f"  exclusion  pid_file requested but {pid_path} not found")

    # --- generic custom_filters ----------------------------------------
    for filt in criteria.get("custom_filters", []) or []:
        col = filt["column"]
        op_str = filt["op"]
        val = filt["value"]

        if op_str == "in":
            m = base_df[col].isin(val)
        elif op_str == "not_in":
            m = ~base_df[col].isin(val)
        elif op_str in _CUSTOM_OPS:
            m = _CUSTOM_OPS[op_str](base_df[col], val)
        else:
            raise ValueError(f"Unknown operator '{op_str}' in custom_filter for column '{col}'")

        excluded = (~m & mask).sum()
        if excluded:
            logger.info(f"  exclusion  {col} {op_str} {val}: -{excluded}")
        mask &= m

    filtered = base_df.loc[mask].copy()
    n_after = len(filtered)
    logger.info(
        f"Exclusion criteria applied: {n_before} → {n_after} patients "
        f"(-{n_before - n_after})"
    )
    return filtered


class AggregatedDS:
    """
    High-performance dataset class that aggregates time series data into tabular format.
    
    Optimizations:
    - Vectorized operations (no patient loops)
    - GPU acceleration with cuDF/CuPy (if available)
    - Efficient memory management
    - Parallel aggregations
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing dataset parameters
    base_df : pd.DataFrame
        Base dataframe containing patient IDs, target, and baseline features
    masking_point : str or pd.Timedelta, optional
        Time offset from patient start time to mask data
    agg_funcs : list of str, optional
        Aggregation functions to apply
    concepts : list of str, optional
        Concept names to aggregate
    use_gpu : bool, optional
        Whether to use GPU acceleration if available. Default: True
    default_mode : bool, optional
        If True, automatically loads and aggregates concepts. Default: True
    """
    
    def __init__(
        self,
        cfg: dict,
        base_df: pd.DataFrame,
        masking_point: Optional[Union[str, pd.Timedelta]] = None,
        agg_funcs: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        use_gpu: bool = True,
        default_mode: bool = True,
        concept_cache: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.cfg = cfg
        self.target = cfg["target"]
        self.masking_point = masking_point

        # Store unfiltered base for reset_filters()
        self._unfiltered_base = base_df.copy()

        # Apply config-driven exclusion criteria BEFORE any aggregation
        criteria = resolve_exclusion_criteria(cfg)
        if criteria:
            base_df = apply_exclusion_criteria(base_df, criteria)

        # Reorder by date for temporal split
        self.base = base_df.sort_values('start').reset_index(drop=True).copy(deep=True)

        # Try to import GPU libraries
        try:
            import cudf
            import cupy as cp
            GPU_AVAILABLE = True
            logger.info("GPU support available (cuDF/CuPy)")
        except ImportError:
            GPU_AVAILABLE = False
            logger.info("GPU support not available, using CPU-only optimizations")
            
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            logger.info("Using GPU acceleration")
        
        # Default aggregation functions
        if agg_funcs is None:
            self.agg_funcs = ['first', 'last', 'min', 'max', 'mean', 'std']
        else:
            self.agg_funcs = agg_funcs
            
        # Default concepts
        if concepts is None:
            self.concepts = self.cfg["concepts"]
        else:
            self.concepts = concepts
        
        # Track feature types
        self.continuous_features = []
        self.categorical_features = []
        
        # Get categorical configuration
        self.cat_concepts = self.cfg["dataset"]["ts_cat_names"]
        self.multi_label_concepts = self.cfg["dataset"]["ts_categorical_multi_label"] #no discrimination for now, for future implementation
         
        # Cache PID set once for fast lookups
        self._base_pids = set(self.base['PID'].unique())

        if default_mode:
            self.set_tab_df()
            self.collect_and_aggregate_concepts(concept_cache=concept_cache)
            self.create_final_dataset()

    def set_tab_df(self):
        """Initialize the base tabular dataframe."""
        id_col = self.cfg["dataset"]["id_col"]
        num_cols = self.cfg["dataset"]["num_cols"]
        cat_cols = get_effective_cat_cols(self.cfg)

        self.tab_df = self.base[[id_col, self.target] + num_cols + cat_cols].copy()
        self.tab_df[num_cols] = self.tab_df[num_cols].astype(float)

        self.continuous_features.extend(num_cols)
        self.categorical_features.extend(cat_cols)

        logger.debug(f"Base tabular columns: {self.tab_df.columns.tolist()}")

    def _parse_masking_point(self) -> Optional[pd.Timedelta]:
        """Convert masking point to pd.Timedelta."""
        if self.masking_point is None:
            return None
        if isinstance(self.masking_point, pd.Timedelta):
            return self.masking_point
        if isinstance(self.masking_point, str):
            return pd.Timedelta(self.masking_point)
        raise ValueError(f"Invalid masking_point type: {type(self.masking_point)}")

    def collect_and_aggregate_concepts(self, concept_cache=None):
        """Collect, filter, mask, and aggregate all concepts.

        Args:
            concept_cache: Optional dict mapping concept name to pre-loaded
                DataFrames (output of _load_and_filter_concept). When provided,
                skips disk I/O and concept-specific filtering, only applying
                masking and aggregation. Used by generate_ebm_feature to avoid
                reloading concept pkls for every masking interval.
        """
        self.aggregated_concepts = {}
        masking_delta = self._parse_masking_point()

        for concept in self.concepts:
            logger.info(f"Processing concept: {concept}")

            try:
                # Use cache if available, otherwise load from disk
                if concept_cache is not None and concept in concept_cache:
                    concept_data = concept_cache[concept]
                else:
                    concept_data = self._load_and_filter_concept(concept)
                
                if len(concept_data) == 0:
                    logger.warning(f"No data for {concept}")
                    continue
                
                # Apply masking (vectorized)
                if masking_delta is not None:
                    concept_data = self._apply_masking_vectorized(concept_data, masking_delta)
                    logger.debug(f"After masking: {len(concept_data)} rows")
                
                if len(concept_data) == 0:
                    logger.warning(f"No data after masking for {concept}")
                    continue
                
                # Determine if categorical
                is_categorical = concept in self.cat_concepts
                is_multi_label = concept in self.multi_label_concepts
                
                # Aggregate
                aggregated_df = self._aggregate_concept_optimized(
                    concept_data, concept, is_categorical, is_multi_label
                )
                
                self.aggregated_concepts[concept] = aggregated_df
                logger.info(f"Aggregated {concept}: {aggregated_df.shape}")
                
            except Exception as e:
                logger.error(f"Failed to process {concept}: {e}")
                continue

    def _load_and_filter_concept(self, concept: str) -> pd.DataFrame:
        """
        Load concept and apply filter function.

        Optimized: Filter to relevant PIDs BEFORE applying expensive
        concept-specific filters (regex, string ops, etc.).

        ISS and Events are derived from Notater.pkl (clinical notes) rather
        than having their own raw CSVs, so they are built on-the-fly here.
        """
        # Notater-derived concepts: build from clinical notes
        _NOTATER_DERIVED = {"ISS_notes", "ISS_computed", "Events"}
        if concept in _NOTATER_DERIVED:
            return self._build_notater_derived_concept(concept)

        # Standard concepts: load from interim pickle
        concept_path = f"data/interim/concepts/{concept}.pkl"
        try:
            df = pd.read_pickle(concept_path)
        except FileNotFoundError:
            concept_path = f"data/interim/concepts/{concept}.csv"
            df = pd.read_csv(concept_path, low_memory=False)

        # Filter to PIDs in base FIRST (cheap operation, reduces rows
        # before expensive concept-specific filters run)
        if 'PID' in df.columns:
            df = df[df['PID'].isin(self._base_pids)]

        # Apply concept-specific filter (now on reduced dataset)
        filter_function = collect_filter(concept)
        if concept == "VitaleVaerdier":
            ews = pd.read_pickle("data/interim/concepts/EWS.pkl")
            if 'PID' in ews.columns:
                ews = ews[ews['PID'].isin(self._base_pids)]
            filtered_df = filter_function(df, ews=ews)
        elif concept == "ITAOversigtsrapport":
            filtered_df = filter_function(df)
            # Augment with GCS extracted from clinical notes
            from astra.data.notes_features import build_gcs_from_notes
            notater_df = pd.read_pickle("data/interim/concepts/Notater.pkl")
            if 'PID' in notater_df.columns:
                notater_df = notater_df[notater_df['PID'].isin(self._base_pids)]
            gcs_df = build_gcs_from_notes(notater_df)
            filtered_df = pd.concat([filtered_df, gcs_df], ignore_index=True)
        else:
            filtered_df = filter_function(df)

        # Ensure TIMESTAMP is datetime
        if 'TIMESTAMP' in filtered_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['TIMESTAMP']):
                filtered_df['TIMESTAMP'] = pd.to_datetime(filtered_df['TIMESTAMP'])

        logger.debug(f"Loaded & filtered {concept}: {len(filtered_df)} rows")
        return filtered_df[['PID', 'FEATURE', 'VALUE', 'TIMESTAMP']]

    def _build_notater_derived_concept(self, concept: str) -> pd.DataFrame:
        """Build ISS or Events concept from Notater.pkl (clinical notes).

        For ISS: try pre-built pickle (which may include R-computed sources),
        fall back to rebuilding from Notater if not available.
        """
        if concept in ("ISS_notes", "ISS_computed"):
            iss_pkl = f"data/interim/concepts/{concept}.pkl"
            filtered_df = pd.read_pickle(iss_pkl)
            if 'PID' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['PID'].isin(self._base_pids)]
            logger.debug(f"Loaded {concept}: {len(filtered_df)} rows")
        elif concept == "Events":
            # For Events, rebuild from Notater
            notater_df = pd.read_pickle("data/interim/concepts/Notater.pkl")
            if 'PID' in notater_df.columns:
                notater_df = notater_df[notater_df['PID'].isin(self._base_pids)]
            from astra.data.cardiac_arrest import build_cardiac_arrest_from_notes
            from astra.data.notes_features import build_intubation_from_notes
            ca_df = build_cardiac_arrest_from_notes(notater_df)
            intub_df = build_intubation_from_notes(notater_df)
            # 24h admission cutoff for intubation
            bin_df = get_bin_df()
            start_times = bin_df.groupby("PID")["bin_start"].min()
            intub_df = intub_df.merge(start_times, on="PID", how="left")
            intub_df = intub_df[intub_df["TIMESTAMP"] <= intub_df["bin_start"] + pd.Timedelta(hours=24)]
            intub_df = intub_df.drop(columns=["bin_start"])
            filtered_df = pd.concat([ca_df, intub_df], ignore_index=True)
            # Reformat for categorical: FEATURE=constant, VALUE=event_type
            filtered_df["VALUE"] = filtered_df["FEATURE"]
            filtered_df["FEATURE"] = "event"
        else:
            raise ValueError(f"Unknown Notater-derived concept: {concept}")

        if 'TIMESTAMP' in filtered_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['TIMESTAMP']):
                filtered_df['TIMESTAMP'] = pd.to_datetime(filtered_df['TIMESTAMP'])

        logger.debug(f"Built {concept} from Notater: {len(filtered_df)} rows")
        return filtered_df[['PID', 'FEATURE', 'VALUE', 'TIMESTAMP']]

    def _apply_masking_vectorized(
        self, 
        concept_data: pd.DataFrame, 
        masking_delta: pd.Timedelta
    ) -> pd.DataFrame:
        """
        Apply masking using vectorized operations (NO LOOPS).
        
        This is the key optimization: instead of looping over patients,
        we merge start times and filter in one vectorized operation.
        """
        if 'start' not in self.base.columns:
            logger.warning("No 'start' column, using min timestamp per patient")
            # Vectorized: compute min timestamp per patient
            patient_starts = concept_data.groupby('PID')['TIMESTAMP'].min().reset_index()
            patient_starts.columns = ['PID', 'start']
        else:
            patient_starts = self.base[['PID', 'start']].copy()
            if not pd.api.types.is_datetime64_any_dtype(patient_starts['start']):
                patient_starts['start'] = pd.to_datetime(patient_starts['start'])
        
        # Merge start times onto concept data (vectorized)
        with_starts = concept_data.merge(patient_starts, on='PID', how='left')
        
        # Calculate cutoff time (vectorized)
        with_starts['cutoff'] = with_starts['start'] + masking_delta
        
        # Filter in one vectorized operation
        masked = with_starts[with_starts['TIMESTAMP'] <= with_starts['cutoff']].copy()
        
        # Drop helper columns
        masked = masked[['PID', 'FEATURE', 'VALUE', 'TIMESTAMP']]
        
        return masked

    def _aggregate_concept_optimized(
        self,
        concept_data: pd.DataFrame,
        concept_name: str,
        is_categorical: bool = False,
        is_multi_label: bool = False
    ) -> pd.DataFrame:
        """
        Optimized aggregation using GPU or efficient pandas operations.
        """
        # Apply drop_features filter if specified.
        # For categorical concepts, FEATURE is constant (e.g. "ADT") so
        # we filter on VALUE instead.
        if "drop_features" in self.cfg and concept_name in self.cfg["drop_features"]:
            drop_list = self.cfg["drop_features"][concept_name]
            col = 'VALUE' if is_categorical else 'FEATURE'
            concept_data = concept_data[~concept_data[col].isin(drop_list)]
        
        if len(concept_data) == 0:
            logger.warning(f"No data to aggregate for {concept_name} ")
            return pd.DataFrame()
        
        if is_categorical:
            logger.info(f"Aggregating categorical concept: {concept_name}")
            return self._aggregate_categorical_optimized(concept_data, concept_name)
        else:
            logger.info(f"Aggregating numeric concept: {concept_name}")
            return self._aggregate_numeric_optimized(concept_data, concept_name)

    def _aggregate_categorical_optimized(
        self,
        concept_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """
        Optimized categorical aggregation using pivot_table (no per-value loop).
        """
        # Drop NaN values upfront
        concept_data = concept_data[concept_data['VALUE'].notna()]
        unique_values = concept_data['VALUE'].unique()

        if len(unique_values) == 0:
            logger.warning(f"No unique values for categorical concept {concept_name}")
            return pd.DataFrame()

        # Count occurrences per patient-value pair
        value_counts = (
            concept_data.groupby(['PID', 'VALUE'])
            .size()
            .reset_index(name='count')
        )

        # Pivot counts: one column per VALUE, one row per PID
        count_pivot = value_counts.pivot_table(
            index='PID', columns='VALUE', values='count', fill_value=0
        )

        # Build given (binary) from counts
        given_pivot = (count_pivot > 0).astype(int)

        # Clean value names and rename columns
        def _clean(v):
            return str(v).replace(' ', '_').replace('/', '_').replace('-', '_')

        count_cols = {v: f'{_clean(v)}_{concept_name}_count' for v in count_pivot.columns}
        given_cols = {v: f'{_clean(v)}_{concept_name}_given' for v in given_pivot.columns}

        count_pivot = count_pivot.rename(columns=count_cols)
        given_pivot = given_pivot.rename(columns=given_cols)

        # Track features
        self.categorical_features.extend(given_cols.values())
        self.continuous_features.extend(count_cols.values())

        # Combine into single df
        result = pd.concat([given_pivot, count_pivot], axis=1).reset_index()
        result = result.fillna(0)
        return result

    def _aggregate_numeric_optimized(
        self,
        concept_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """
        Highly optimized numeric aggregation.
        
        Key optimizations:
        1. Single groupby with multiple aggregations
        2. GPU acceleration if available
        3. Efficient pivoting
        """
        # Convert to numeric
        concept_data = concept_data.copy()
        concept_data['VALUE_numeric'] = pd.to_numeric(concept_data['VALUE'], errors='coerce')
        
        # Remove non-numeric
        valid_data = concept_data[concept_data['VALUE_numeric'].notna()].copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Try GPU acceleration
        if self.use_gpu:
            try:
                return self._aggregate_numeric_gpu(valid_data, concept_name)
            except Exception as e:
                logger.warning(f"GPU aggregation failed, falling back to CPU: {e}")
        
        # CPU optimized version
        return self._aggregate_numeric_cpu(valid_data, concept_name)

    def _aggregate_numeric_gpu(
        self,
        valid_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """
        GPU-accelerated numeric aggregation using cuDF.
        """
        # Convert to cuDF
        gdf = cudf.from_pandas(valid_data)
        
        # Build aggregation dict
        agg_dict = {}
        for agg_func in self.agg_funcs:
            if agg_func in ['first', 'last']:
                # Sort once for first/last
                if agg_func not in agg_dict:
                    gdf_sorted = gdf.sort_values('TIMESTAMP')
                    if agg_func == 'first':
                        agg_result = gdf_sorted.groupby(['PID', 'FEATURE'])['VALUE_numeric'].first()
                    else:
                        agg_result = gdf_sorted.groupby(['PID', 'FEATURE'])['VALUE_numeric'].last()
                    agg_dict[agg_func] = agg_result
            else:
                # Standard aggregations
                agg_dict[agg_func] = (
                    gdf.groupby(['PID', 'FEATURE'])['VALUE_numeric']
                    .agg(agg_func)
                )
        
        # Convert results to pandas and pivot
        result_dfs = []
        for agg_func, agg_data in agg_dict.items():
            # Convert to pandas
            agg_df = agg_data.to_pandas().reset_index()
            
            # Pivot
            pivoted = agg_df.pivot(
                index='PID',
                columns='FEATURE',
                values='VALUE_numeric'
            )
            
            # Rename columns
            pivoted.columns = [f"{col}_{concept_name}_{agg_func}" for col in pivoted.columns]
            
            # Track features
            self.continuous_features.extend(pivoted.columns.tolist())
            
            result_dfs.append(pivoted)
        
        # Concatenate all aggregations
        result = pd.concat(result_dfs, axis=1).reset_index()
        result = result.fillna(0.0)
        
        return result

    def _aggregate_numeric_cpu(
        self,
        valid_data: pd.DataFrame,
        concept_name: str
    ) -> pd.DataFrame:
        """
        CPU-optimized numeric aggregation.
        
        Key optimization: Single groupby with multiple aggregations at once.
        """
        # Sort once for first/last
        valid_data_sorted = valid_data.sort_values('TIMESTAMP')
        
        # Build aggregation dictionary
        agg_operations = {}
        for agg_func in self.agg_funcs:
            if agg_func == 'first':
                agg_operations['first'] = 'first'
            elif agg_func == 'last':
                agg_operations['last'] = 'last'
            elif agg_func in ['min', 'max', 'mean', 'std', 'count', 'median']:
                agg_operations[agg_func] = agg_func
        
        # Single groupby with multiple aggregations (MUCH FASTER)
        grouped = valid_data_sorted.groupby(['PID', 'FEATURE'])['VALUE_numeric'].agg(
            list(agg_operations.values())
        ).reset_index()
        
        # Rename aggregation columns
        grouped.columns = ['PID', 'FEATURE'] + list(agg_operations.keys())
        
        # Pivot each aggregation
        result_dfs = []
        for agg_func in agg_operations.keys():
            if agg_func not in grouped.columns:
                continue
            
            # Pivot this aggregation
            pivoted = grouped[['PID', 'FEATURE', agg_func]].pivot(
                index='PID',
                columns='FEATURE',
                values=agg_func
            )
            
            # Rename columns
            pivoted.columns = [f"{col}_{concept_name}_{agg_func}" for col in pivoted.columns]
            
            # Track features
            self.continuous_features.extend(pivoted.columns.tolist())
            
            result_dfs.append(pivoted)
        
        # Concatenate all aggregations
        result = pd.concat(result_dfs, axis=1).reset_index()
        result = result.fillna(0.0)
        
        return result

    def create_final_dataset(self):
        """Merge all aggregated concepts with base tabular data."""
        final_df = self.tab_df.copy()
        id_col = self.cfg["dataset"]["id_col"]
        
        # Merge all concepts at once (more efficient)
        for concept_name, agg_df in self.aggregated_concepts.items():
            if len(agg_df) > 0:
                pre_merge_len = len(final_df)
                final_df = final_df.merge(agg_df, left_on=id_col, right_on='PID', how='left')
                
                if 'PID' in final_df.columns and id_col != 'PID':
                    final_df = final_df.drop(columns=['PID'])
                
                assert len(final_df) == pre_merge_len, f"Merge changed row count for {concept_name}"
                logger.info(f"Merged {concept_name}: {len(agg_df.columns)-1} features")
        
        # Fill NaN
        feature_cols = [col for col in final_df.columns if col not in [id_col, self.target]]
        final_df[feature_cols] = final_df[feature_cols].fillna(0.0)
        
        self.final_df = final_df
        
        # Remove duplicates
        self.continuous_features = list(dict.fromkeys(self.continuous_features))
        self.categorical_features = list(dict.fromkeys(self.categorical_features))
        
        logger.info(f"Final dataset: {final_df.shape}")
        logger.info(f"Features: {len(self.continuous_features)} cont + {len(self.categorical_features)} cat")

    def get_features_by_type(self) -> Dict[str, List[str]]:
        """Get feature names by type."""
        return {
            'continuous': self.continuous_features,
            'categorical': self.categorical_features
        }

    def get_X_y(self, include_id: bool = False):
        """Get feature matrix and target."""
        id_col = self.cfg["dataset"]["id_col"]
        
        if include_id:
            X = self.final_df.drop(columns=[self.target])
        else:
            X = self.final_df.drop(columns=[id_col, self.target])
        
        y = self.final_df[self.target]
        
        return X, y

    def to_csv(self, filepath: str):
        """Save to CSV."""
        ensure_parent_dir(filepath)
        self.final_df.to_csv(filepath, index=False)
        logger.info(f"Saved to {filepath}")
    
    def to_pickle(self, filepath: str):
        """Save to pickle."""
        ensure_parent_dir(filepath)
        self.final_df.to_pickle(filepath)
        logger.info(f"Saved to {filepath}")

    # --- post-hoc exclusion for experimentation -------------------------

    def filter(self, criteria: dict):
        """Apply exclusion criteria to the already-built dataset.

        Filters all internal DataFrames by PID set.  Use ``reset_filters()``
        to restore the original unfiltered state.  Returns *self* for chaining.
        """
        filtered_base = apply_exclusion_criteria(self.base, criteria)
        keep_pids = set(filtered_base["PID"].unique())

        self.base = filtered_base.reset_index(drop=True)
        self._base_pids = keep_pids
        self.tab_df = self.tab_df[self.tab_df["PID"].isin(keep_pids)].reset_index(drop=True)

        if hasattr(self, "final_df") and self.final_df is not None:
            id_col = self.cfg["dataset"]["id_col"]
            self.final_df = self.final_df[self.final_df[id_col].isin(keep_pids)].reset_index(drop=True)

        if hasattr(self, "aggregated_concepts"):
            for name, df in self.aggregated_concepts.items():
                self.aggregated_concepts[name] = df[df["PID"].isin(keep_pids)].reset_index(drop=True)

        return self

    def reset_filters(self):
        """Restore original unfiltered base_df and rebuild internal DataFrames."""
        base_df = self._unfiltered_base.copy()

        # Re-apply config exclusion if set
        criteria = resolve_exclusion_criteria(self.cfg)
        if criteria:
            base_df = apply_exclusion_criteria(base_df, criteria)

        self.base = base_df.sort_values("start").reset_index(drop=True)
        self._base_pids = set(self.base["PID"].unique())

        self.continuous_features = []
        self.categorical_features = []
        self.set_tab_df()
        self.collect_and_aggregate_concepts()
        self.create_final_dataset()

        return self


class TSDS:
    def __init__(
        self,
        cfg,
        base_df,
        default_mode=True,
        concepts=None
    ):
        self.cfg = cfg
        self.target = cfg["target"]

        # Store unfiltered base for reset_filters()
        self._unfiltered_base = base_df.copy()

        # Apply config-driven exclusion criteria BEFORE concept collection
        criteria = resolve_exclusion_criteria(cfg)
        if criteria:
            base_df = apply_exclusion_criteria(base_df, criteria)

        self.base = base_df
        self._base_pids = set(base_df['PID'].unique())

        if concepts is None:
            self.concepts=self.cfg["concepts"]
        else:
            self.concepts = concepts

        if default_mode:
            self.set_tab_df()
            self.collect_concepts()
            

    def set_tab_df(self):
        cat_cols = get_effective_cat_cols(self.cfg)
        self.tab_df = self.base[[self.cfg["dataset"]["id_col"],self.cfg["target"]]+self.cfg["dataset"]["num_cols"]+cat_cols].copy(deep=True)
        self.tab_df[self.cfg["dataset"]["num_cols"]] = self.tab_df[self.cfg["dataset"]["num_cols"]].astype(float)
        logger.debug(self.base.columns)


    
    def collect_concepts(self):
        concepts = {}
        concepts_raw = {}
        self.timestep_cols = []
        self._profile_data = {}  # {concept: (profile_array, profile_dims, category_order)}
        for concept in self.concepts:
            logger.debug(f"getting {concept}")
            concepts_raw[concept] = get_concept(concept, self.cfg, self._base_pids)

            logger.debug(f"getting long version of {concept}")
            if concept in self.cfg["dataset"]["ts_cat_names"]:
                agg_func_name = self.cfg["agg_func"][concept]
                concept_long_df = concepts_raw[concept][self.cfg["agg_func"][concept][0]].copy(deep=True)
                concepts[concept] = _get_long_concept_df_multi_label(concept_long_df, self.base, self.cfg, self._base_pids)

                # Compute categorical profiles if enabled for this concept
                from astra.data.profiles import profiles_enabled, get_profiled_categories, CategoricalProfileEncoder, load_profiles_config
                if profiles_enabled(self.cfg):
                    profiles_cfg = load_profiles_config(self.cfg)
                    concept_profiles = profiles_cfg.get(concept, {})
                    profiled_cats = concept_profiles.get("categories", {})
                    sub_code_long = concepts[concept].attrs.get("sub_code_long")

                    if profiled_cats and sub_code_long is not None:
                        encoder = CategoricalProfileEncoder(concept_profiles)
                        sub_code_index = encoder.build_sub_code_index_fast(sub_code_long)
                        pids = sorted(concepts[concept]['PID'].unique())
                        timestep_cols = concepts[concept].attrs["timestep_cols"]
                        profile_array, profile_dims, category_order = encoder.compute_profiles(
                            sub_code_index, pids, timestep_cols
                        )
                        self._profile_data[concept] = (profile_array, profile_dims, category_order, pids)
                        # Remove profiled category values from wide df (they'll use the profile tensor)
                        concepts[concept] = encoder.strip_profiled_from_wide(
                            concepts[concept], timestep_cols
                        )
                        logger.info(f"Profiles for {concept}: {profile_dims}")

                # specifcy max ts dims
                if len(concepts[concept].attrs["timestep_cols"]) > len(self.timestep_cols):
                    self.timestep_cols = concepts[concept].attrs["timestep_cols"]
            else:
                concepts[concept] = _get_long_concept_df_single_label(
                    self.cfg,
                    self.base,
                    concepts_raw[concept],
                    concept,
                    self.cfg["target"],
                    self.cfg["bin_freq_include"],
                    self._base_pids,
                )

        # Add temporal features if enabled
        if self.cfg.get('temporal_features', {}).get('enabled', False):
            temporal_df = _create_temporal_features_df(
                self.cfg, self.base, self.cfg["target"], self.cfg["bin_freq_include"]
            )
            if temporal_df is not None:
                concepts['_temporal'] = temporal_df

        self.concepts = concepts
        self.concepts_raw = concepts_raw

    def change_na_fill(self, mode="forward"): #OBSOLETE?
        if mode == "forward":
            logger.info("Forward filling vitals")
            self.vitals = self.vitals.replace({0.0: np.nan})
            # if first row missing, fill with 0, forward fill the rest
            self.vitals["0"] = self.vitals["0"].fillna(0.0)
            self.vitals.iloc[:, :-1] = self.vitals.iloc[:, :-1].ffill(axis=1)
            # for target and if ffill not available
            self.vitals = self.vitals.fillna(0.0)

    # --- post-hoc exclusion for experimentation -------------------------

    def filter(self, criteria: dict):
        """Apply exclusion criteria to the already-built dataset.

        Filters all internal DataFrames by PID set.  Use ``reset_filters()``
        to restore the original unfiltered state.  Returns *self* for chaining.
        """
        filtered_base = apply_exclusion_criteria(self.base, criteria)
        keep_pids = set(filtered_base["PID"].unique())

        self.base = filtered_base.reset_index(drop=True)
        self._base_pids = keep_pids
        self.tab_df = self.tab_df[self.tab_df["PID"].isin(keep_pids)].reset_index(drop=True)

        if hasattr(self, "concepts") and isinstance(self.concepts, dict):
            for name, df in self.concepts.items():
                self.concepts[name] = df[df["PID"].isin(keep_pids)].reset_index(drop=True)

        return self

    def reset_filters(self):
        """Restore original unfiltered base_df and rebuild internal DataFrames."""
        base_df = self._unfiltered_base.copy()

        # Re-apply config exclusion if set
        criteria = resolve_exclusion_criteria(self.cfg)
        if criteria:
            base_df = apply_exclusion_criteria(base_df, criteria)

        self.base = base_df
        self._base_pids = set(self.base["PID"].unique())
        self.set_tab_df()
        self.collect_concepts()

        return self


def _create_temporal_features_df(
    cfg: Dict,
    base: pd.DataFrame,
    target: str,
    bin_freq_include: List[str],
) -> pd.DataFrame:
    """
    Create temporal features (elapsed_hours, bin_width_hours) as a wide-format
    DataFrame matching the schema of _get_long_concept_df_single_label output.

    Each patient gets temporal values at bin positions within their trajectory,
    with 0.0 padding beyond their trajectory length.
    """
    bin_df = get_bin_df()

    # Filter to base PIDs and included frequencies
    base_pids = set(base['PID'].unique())
    bf = bin_df[
        (bin_df['PID'].isin(base_pids)) &
        (bin_df['bin_freq'].isin(bin_freq_include))
    ].copy()

    # 'start' is the universal earliest timestamp (incorporates prehospital when available)
    merge_cols = ['PID', 'start']
    bf = bf.merge(base[merge_cols], on='PID', how='left')

    # Sort and assign sequential position per patient (0-indexed)
    bf = bf.sort_values(['PID', 'bin_counter'])
    bf['position'] = bf.groupby('PID').cumcount()

    # Compute temporal features — reference from universal trajectory start
    ref_start = bf['start']
    bf['elapsed_hours'] = (
        (bf['bin_start'] - ref_start).dt.total_seconds() / 3600
        + (bf['bin_end'] - bf['bin_start']).dt.total_seconds() / 7200  # midpoint
    )
    bf['bin_width_hours'] = (
        (bf['bin_end'] - bf['bin_start']).dt.total_seconds() / 3600
    )

    # Select which features to include from config
    enabled_features = cfg.get('temporal_features', {}).get('features', [])
    available = {'elapsed_hours', 'bin_width_hours'}
    features_to_add = [f for f in enabled_features if f in available]

    if not features_to_add:
        return None

    max_pos = bf['position'].max()

    # Build wide-format rows for each feature
    rows = []
    for feat_name in features_to_add:
        # Pivot: PID × position → value
        feat_wide = bf.pivot(index='PID', columns='position', values=feat_name)
        feat_wide = feat_wide.reindex(columns=range(max_pos + 1), fill_value=0.0)
        feat_wide = feat_wide.fillna(0.0)
        feat_wide = feat_wide.reset_index()
        feat_wide.insert(1, 'FEATURE', feat_name)
        # Rename position columns to string indices
        feat_wide.columns = ['PID', 'FEATURE'] + [
            str(i) for i in range(max_pos + 1)
        ]
        rows.append(feat_wide)

    result = pd.concat(rows, ignore_index=True)

    # Ensure all base PIDs are present (fill missing with 0)
    all_pids = base['PID'].unique()
    existing_pids = set(result['PID'].unique())
    missing_pids = [p for p in all_pids if p not in existing_pids]

    if missing_pids:
        ts_cols = [str(i) for i in range(max_pos + 1)]
        missing_rows = []
        for feat_name in features_to_add:
            for pid in missing_pids:
                row = {'PID': pid, 'FEATURE': feat_name}
                row.update({col: 0.0 for col in ts_cols})
                missing_rows.append(row)
        result = pd.concat([result, pd.DataFrame(missing_rows)], ignore_index=True)

    # Merge target
    result = result.merge(base[['PID', target]], on='PID', how='left')
    result[target] = result[target].astype(int)
    result = result.sort_values(['PID', 'FEATURE']).reset_index(drop=True)

    logger.info(
        f"Created temporal features: {features_to_add} "
        f"({len(result)} rows, {max_pos + 1} timesteps)"
    )

    return result


def _get_long_concept_df_single_label(
    cfg: Dict,
    base: pd.DataFrame,
    concepts: Dict,
    concept: str,
    target: str,
    bin_freq_include: Optional[List],
    base_pids: set = None,
) -> pd.DataFrame:
    """
    Process single-label data: PIVOT to wide format.
    This is the original behavior.
    """
    if base_pids is None:
        base_pids = set(base['PID'].unique())

    pivoted = []

    for agg_func in cfg["agg_func"][concept]:
        logger.debug(f"Single-label: {concept} with {agg_func}")
        df = concepts[agg_func].copy()

        if bin_freq_include is not None:
            df = df[df.bin_freq.isin(bin_freq_include)]

        try:
            df = df[
                (~df.FEATURE.isin(cfg["drop_features"].get(concept, []))) &
                (df.PID.isin(base_pids))
            ][["PID", "bin_counter", "FEATURE", "VALUE"]]
        except Exception:
            logger.debug(f"drop_features filter failed for {concept}; filtering on PID only")
            df = df[
                (df.PID.isin(base_pids))
            ][["PID", "bin_counter", "FEATURE", "VALUE"]]
        
        # Convert to numeric
        try:
            df['VALUE'] = pd.to_numeric(df['VALUE'])
        except (ValueError, TypeError):
            pass  # Keep as string if conversion fails
        
        # Pivot the dataframe (safe because single-label = no duplicates)
        pivoted_df = df.pivot(
            index=["PID", "FEATURE"],
            columns="bin_counter",
            values="VALUE"
        )

        pivoted_df = pivoted_df.reset_index()
        pivoted_df.columns.name = None
        # Rename counter columns to 0-based positions (counter - 1).
        # Using sequential range(len) is WRONG for sparse concepts where
        # not all bin_counters appear across the population — missing
        # columns collapse the grid and shift positions.
        counter_cols = pivoted_df.columns[2:]  # bin_counter values (ints)
        pivoted_df.columns = list(pivoted_df.columns[:2]) + [
            f"{int(c) - 1}" for c in counter_cols
        ]
        # Ensure all positions from 0..max exist (fill missing with NaN)
        max_pos = max(int(c) - 1 for c in counter_cols)
        all_pos_cols = [f"{i}" for i in range(max_pos + 1)]
        missing_cols = [c for c in all_pos_cols if c not in pivoted_df.columns]
        if missing_cols:
            for c in missing_cols:
                pivoted_df[c] = np.nan
            pivoted_df = pivoted_df[
                list(pivoted_df.columns[:2]) + all_pos_cols
            ]
        
        pivoted_df = pivoted_df.sort_values(["PID", "FEATURE"]).reset_index(drop=True)
        
        # Create complete set of PID-FEATURE combinations
        unique_pids = base["PID"].unique()
        unique_features = pivoted_df["FEATURE"].unique()
        
        complete_set = pd.MultiIndex.from_product(
            [unique_pids, unique_features], 
            names=["PID", "FEATURE"]
        )
        complete_df = pd.DataFrame(index=complete_set).reset_index()
        
        merged_df = complete_df.merge(pivoted_df, on=["PID", "FEATURE"], how="left")
        
        # NaN intentionally preserved for missing measurements.
        # Normalization in dataloader.py uses trajectory_lengths + NaN detection
        # to distinguish "no measurement" from "padding beyond trajectory".
        numeric_cols = [col for col in merged_df.columns if col.isdigit()]
        
        merged_df = merged_df.sort_values(["PID", "FEATURE"])
        
        # Rename features
        col_mapper = {
            feat: f"{feat}_{agg_func}" 
            for feat in df.FEATURE.unique()
        }
        merged_df["FEATURE"] = merged_df["FEATURE"].replace(col_mapper)
        
        pivoted_df = merged_df.reset_index(drop=True)
        pivoted_df = pivoted_df[pivoted_df.FEATURE.notnull()]
        
        pivoted.append(pivoted_df)
    
    # Concat all aggregation functions
    complete = pd.concat(pivoted, ignore_index=True)
    
    # NaN intentionally preserved for missing measurements.
    # Normalization in dataloader.py uses trajectory_lengths + NaN detection
    # to distinguish "no measurement" from "padding beyond trajectory".
    numeric_cols = [col for col in complete.columns if col.isdigit()]
    
    # Merge target
    prelen = len(complete)
    complete = complete.merge(base[["PID", target]].copy(deep=True), on="PID", how="left")
    complete[target] = complete[target].astype(int)
    assert prelen == len(complete), f"Length mismatch: {prelen} vs {len(complete)}"
    
    complete = complete[complete.FEATURE.notnull()].reset_index(drop=True)
    
    return complete

def _get_long_concept_df_multi_label(df_long:pd.DataFrame, base:pd.DataFrame, cfg, base_pids: set = None):
    if base_pids is None:
        base_pids = set(base['PID'].unique())

    logger.debug(f"Initial PID count {df_long.PID.nunique()}")
    df_long = df_long[df_long.bin_freq.isin(cfg["bin_freq_include"])].copy(deep=True)
    logger.debug(f">>minus bin freq: {df_long.PID.nunique()}")

    # Preserve SUB_CODE for categorical profile encoding if present
    keep_cols = ['PID', 'bin_counter', 'FEATURE', 'VALUE']
    has_sub_code = 'SUB_CODE' in df_long.columns
    if has_sub_code:
        keep_cols.append('SUB_CODE')

    df_long = df_long[keep_cols].rename(columns={'bin_counter':'TIMESTEP'})
    df_long["TIMESTEP"] = df_long["TIMESTEP"]-1 # matching df2xy function index 0

    # Re-index to contiguous 0-based positions (matching single-label behavior)
    # After bin_freq_include filtering, TIMESTEP values may have gaps
    unique_ts = sorted(df_long["TIMESTEP"].unique())
    ts_remap = {old: new for new, old in enumerate(unique_ts)}
    df_long["TIMESTEP"] = df_long["TIMESTEP"].map(ts_remap)

    # Filter to relevant PIDs BEFORE expensive pivot
    df_long = df_long[df_long.PID.isin(base_pids)]
    logger.debug(f">>after PID filter: {df_long.PID.nunique()}")

    # Store sub-code long-format data before pivot (for profile encoding)
    # The pivot aggregates values into lists, losing per-row sub-code detail.
    sub_code_long = None
    if has_sub_code:
        sub_code_long = df_long[['PID', 'TIMESTEP', 'VALUE', 'SUB_CODE']].copy()

    # Pivot to wide format (your format)
    df_wide = df_long.pivot_table(
      index=['PID', 'FEATURE'],
      columns='TIMESTEP',
      values='VALUE',
      aggfunc=lambda x: list(x) if len(x) > 1 else x.iloc[0],
      dropna=False
    ).reset_index()
    # keep all timesteps, fill in feature name
    feat_name = df_wide.FEATURE.dropna().unique()
    assert len(feat_name) == 1
    df_wide['FEATURE'] = feat_name[0]
    timestep_cols = list(range(df_long.TIMESTEP.max() + 1))

    # Ensure ALL base PIDs are present (matching _get_long_concept_df_single_label)
    missing_pids = base_pids - set(df_wide['PID'].unique())
    if missing_pids:
        logger.debug(f"{len(missing_pids)} PIDs have no {feat_name[0]} data — adding empty rows")
        placeholder = pd.DataFrame({
            'PID': list(missing_pids),
            'FEATURE': feat_name[0],
            **{col: np.nan for col in timestep_cols}
        })
        df_wide = pd.concat([df_wide, placeholder], ignore_index=True)

    df_wide.attrs["timestep_cols"] = timestep_cols
    if sub_code_long is not None:
        df_wide.attrs["sub_code_long"] = sub_code_long
    logger.debug(f">>> after wide: {df_wide.PID.nunique()}")
    return df_wide

def get_concept(concept: str, cfg: Dict, base_pids: set = None) -> Dict:
    """Get concept from mapped files."""
    drop_cols = cfg["drop_features"].get(concept, [])
    concept_dict = {}

    for agg_func in cfg["agg_func"][concept]:
        df = pd.read_csv(f"data/interim/mapped/{concept}_{agg_func}.csv")

        # Filter to relevant PIDs early (before any other processing)
        if base_pids is not None and 'PID' in df.columns:
            df = df[df['PID'].isin(base_pids)]

        if concept in cfg["dataset"]["ts_cat_names"]:
            # Categorical: FEATURE is constant (e.g. "ADT"), filter on VALUE
            if drop_cols:
                df = df[~df.VALUE.isin(drop_cols)]
        else:
            try:
                df = df[~df.FEATURE.isin(drop_cols + [np.nan])]
            except Exception:
                logger.debug(f"drop_features filter failed for {concept}; dropping NaN FEATUREs only")
                df = df[~df.FEATURE.isin([np.nan])]

        concept_dict[agg_func] = df

    return concept_dict


# NOTE: Legacy ppjDataset class removed — replaced by astra.data.prehospital module.
