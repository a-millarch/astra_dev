import logging
import os
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

from astra.utils import get_bin_df, ensure_parent_dir
from astra.data.filters import collect_filter

logger = logging.getLogger(__name__)

def expand_interval_to_bins(
    events_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Expand interval-based events to one row per overlapping bin.

    Events must have TIMESTAMP (start) and END_TIMESTAMP (end).
    For each event, creates a row for every bin where the event is active:
        event_start < bin_end AND event_end > bin_start

    Returns a DataFrame with standard columns (PID, FEATURE, VALUE, TIMESTAMP)
    where TIMESTAMP is set to the bin_start of each overlapping bin.
    """
    events_df = events_df.copy()
    bin_df = bin_df.copy()
    events_df["PID"] = events_df["PID"].astype("int32")
    bin_df["PID"] = bin_df["PID"].astype("int32")

    # Pre-filter to matching PIDs
    bin_pids = set(bin_df["PID"].unique())
    events_df = events_df[events_df["PID"].isin(bin_pids)]

    if len(events_df) == 0:
        logger.warning("No interval events match any bin PIDs")
        return events_df.drop(columns=["END_TIMESTAMP"], errors="ignore")

    unique_pids = events_df["PID"].unique()
    results = []

    for i in range(0, len(unique_pids), chunk_size):
        chunk_pids = set(unique_pids[i : i + chunk_size])
        chunk_events = events_df[events_df["PID"].isin(chunk_pids)]
        chunk_bins = bin_df[bin_df["PID"].isin(chunk_pids)]

        # Merge on PID (creates event x bin pairs within each patient)
        merged = chunk_events.merge(
            chunk_bins[["PID", "bin_start", "bin_end"]],
            on="PID",
        )

        # Keep only overlapping bins
        overlap = merged[
            (merged["TIMESTAMP"] < merged["bin_end"])
            & (merged["END_TIMESTAMP"] > merged["bin_start"])
        ].copy()

        results.append(overlap)

    if len(results) == 0:
        logger.warning("No interval events overlap any bins")
        return events_df.drop(columns=["END_TIMESTAMP"], errors="ignore").iloc[0:0]

    result = pd.concat(results, ignore_index=True)

    # Set TIMESTAMP to bin_start so downstream searchsorted assigns the correct bin
    result["TIMESTAMP"] = result["bin_start"]
    result = result.drop(columns=["END_TIMESTAMP", "bin_start", "bin_end"])

    logger.info(
        f"Interval expansion: {len(events_df)} events -> {len(result)} bin-aligned rows"
    )
    return result


def merge_and_aggregate(
    bin_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    agg_func: str = "mean",
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> pd.DataFrame:
    """
    Enhanced merge and aggregate that handles both continuous and categorical data.
    
    Args:
        bin_df: DataFrame with time bins
        subset_df: DataFrame with values to aggregate
        agg_func: Aggregation function for continuous data
        is_categorical: Whether this feature is categorical
        is_multi_label: Whether this categorical feature can have multiple values per bin
    
    Returns:
        Aggregated DataFrame
    """
    # Ensure datatypes for memory efficiency
    bin_df["PID"] = bin_df["PID"].astype("int32")
    subset_df["PID"] = subset_df["PID"].astype("int32")
    
    # For continuous: ensure numeric
    # For categorical: keep as object/string
    if not is_categorical:
        subset_df["VALUE"] = subset_df["VALUE"].astype("float")
    
    # Merge on PID
    merged_df = pd.merge(bin_df, subset_df, on="PID", how="left")
    
    # Filter based on timestamp conditions
    filtered_df = merged_df[
        (merged_df["TIMESTAMP"] >= merged_df["bin_start"])
        & (merged_df["TIMESTAMP"] <= merged_df["bin_end"])
    ]
    
    # === CATEGORICAL HANDLING (NEW) ===
    if is_categorical:
        if is_multi_label:
            # Keep all values as separate rows
            keep_cols = ["PID", "bin_counter", "bin_start", "bin_end", "FEATURE", "VALUE"]
            if "SUB_CODE" in filtered_df.columns:
                keep_cols.append("SUB_CODE")
            aggregated_df = filtered_df[keep_cols].drop_duplicates()

        else:
            # Single-label: Take mode (most common) or last value
            logger.debug(f"Single-label categorical: using mode/last")
            
            if agg_func == "mode":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan})
                    .reset_index()
                )
            elif agg_func == "last":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "last"})
                    .reset_index()
                )
            elif agg_func == "first":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "first"})
                    .reset_index()
                )
            else:
                # Default to last
                logger.warning(f"Unknown agg_func '{agg_func}' for categorical, using 'last'")
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "last"})
                    .reset_index()
                )
    
    # === CONTINUOUS HANDLING (ORIGINAL) ===
    else:
        aggregation = {
            "first": "first",
            "mean": "mean",
            "max": "max",
            "min": "min",
            "std": "std",
            "sum": "sum",
            "count": "count",
            "last": "last",
        }
        agg_function = aggregation.get(agg_func, "mean")
        
        aggregated_df = (
            filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
            .agg({"VALUE": agg_function})
            .reset_index()
        )
    
    # Merge the result back to bin_df to maintain all rows
    result_df = pd.merge(
        bin_df,
        aggregated_df,
        on=["PID", "bin_counter", "bin_start", "bin_end"],
        how="left",
    )
    
    return result_df


def map_concept(
    cfg: Dict,
    concept: str,
    agg_func: str,
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> None:
    """
    Enhanced map_concept that handles categorical features.
    
    Args:
        cfg: Configuration dictionary
        concept: Name of concept (e.g., 'Medicin', 'VitaleVaerdier')
        agg_func: Aggregation function
        is_categorical: Whether this concept contains categorical features
        is_multi_label: Whether categorical features can have multiple values per bin
    """

    
    # Notater-derived concepts must use map_concept_optimized
    _NOTATER_DERIVED = {"ISS_notes", "ISS_computed", "Events"}
    if concept in _NOTATER_DERIVED:
        raise ValueError(
            f"'{concept}' is derived from Notater.pkl. Use map_concept_optimized() instead."
        )

    output_path = f"data/interim/mapped/{concept}"

    # Load binning DataFrame
    bin_df = get_bin_df()
    logger.info(f"Prepared bin df for {concept} (categorical={is_categorical}, multi_label={is_multi_label})")

    # Load and filter concept
    concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
    filter_function = collect_filter(concept)
    concept_df = filter_function(concept_df)
    
    # Process each feature
    dfs = []
    logger.info(f"Processing {len(concept_df.FEATURE.unique())} features")
    
    for feat in concept_df.FEATURE.unique():
        logger.info(f"Processing feature: {feat}")
        subset = concept_df[concept_df.FEATURE == feat]
        
        # Merge and aggregate with categorical handling
        result_df = merge_and_aggregate(
            bin_df, 
            subset, 
            agg_func=agg_func,
            is_categorical=is_categorical,
            is_multi_label=is_multi_label
        )
        
        dfs.append(result_df)
    
    logger.info("Concatenating feature dataframes")
    
    # Concatenate and save
    if len(dfs) < 1:
        logger.warning(f"Concept {concept} failed - no features processed")
        bin_df["FEATURE"] = np.nan
        bin_df["VALUE"] = np.nan
        ensure_parent_dir(f"{output_path}_{agg_func}.pkl")
        bin_df.to_pickle(f"{output_path}_{agg_func}.pkl", protocol=4)
        bin_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
    else:
        result_df = (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values(["PID", "bin_counter"])
            .reset_index(drop=True)
        )
        
        # === HANDLE MULTI-LABEL EXPANSION (NEW) ===
        if is_categorical and is_multi_label:
            logger.info("Expanding multi-label values")
            
            # Expand lists into separate rows
            # This creates multiple rows per PID/bin/feature for multi-label
            expanded_rows = []
            
            for idx, row in result_df.iterrows():
                if pd.notna(row['VALUE']):
                    # Check if VALUE is a list
                    if isinstance(row['VALUE'], list):
                        # Create separate row for each value in list
                        for val in row['VALUE']:
                            new_row = row.copy()
                            new_row['VALUE'] = val
                            expanded_rows.append(new_row)
                    else:
                        # Single value, keep as is
                        expanded_rows.append(row)
                else:
                    # NaN value, keep as is
                    expanded_rows.append(row)
            
            result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
            logger.info(f"Expanded to {len(result_df)} rows (from multi-label)")
        
        # Remove bin placeholder/nan rows if data rows present
        grouped = result_df.groupby(["PID", "bin_counter"])
        
        def filter_rows(group):
            """Keep NaN rows only if no data exists for that group"""
            if group["FEATURE"].isna().all() and group["VALUE"].isna().all():
                return group
            else:
                return group.dropna(subset=["FEATURE", "VALUE"])
        
        logger.info(f"Cleaning binned {concept} dataframe")
        filtered_df = grouped.apply(filter_rows).reset_index(drop=True)
        
        # Log statistics
        logger.info(f"Final shape: {filtered_df.shape}")
        if is_categorical:
            n_unique_values = filtered_df['VALUE'].nunique()
            logger.info(f"Unique categorical values: {n_unique_values}")
            if is_multi_label:
                avg_values_per_bin = filtered_df.groupby(['PID', 'bin_counter']).size().mean()
                logger.info(f"Average values per bin: {avg_values_per_bin:.2f}")
        
        logger.info(f"Saving file to {output_path}")
        ensure_parent_dir(f"{output_path}_{agg_func}.pkl")
        filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl", protocol=4)
        filtered_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)


def map_all_concepts(cfg: Dict, force: bool = False) -> None:
    """
    Map all concepts defined in config with proper categorical handling.
    
    Args:
        cfg: Configuration dictionary with:
            - concepts: List of concept names
            - agg_func: Dict mapping concept to list of agg functions
            - cat_time_series: Dict with categorical configuration
        force: Whether to reprocess existing files
    """
    import os
    from pathlib import Path
    
    # Get categorical configuration
    cat_config = cfg.get("cat_time_series", {})
    cat_concepts = cat_config.get("concepts", {})
    multi_label_concepts = cat_config.get("multi_label", [])
    
    logger.info("="*80)
    logger.info("MAPPING ALL CONCEPTS")
    logger.info("="*80)
    
    concepts = cfg.get("concepts", [])
    
    for concept in concepts:
        # Determine if concept is categorical
        is_categorical = concept in cat_concepts
        is_multi_label = concept in multi_label_concepts
        
        if is_categorical:
            logger.info(f"\n{concept}: CATEGORICAL" + 
                       (" + MULTI-LABEL" if is_multi_label else ""))
        else:
            logger.info(f"\n{concept}: CONTINUOUS")
        
        # Get aggregation functions for this concept
        agg_funcs = cfg["agg_func"].get(concept, ["mean"])
        
        for agg_func in agg_funcs:
            output_file = f"data/interim/mapped/{concept}_{agg_func}.csv"
            
            # Check if file exists
            if not force and os.path.exists(output_file):
                logger.info(f"  {agg_func}: Already exists, skipping")
                continue
            
            logger.info(f"  Processing with agg_func: {agg_func}")
            
            try:
                map_concept(
                    cfg=cfg,
                    concept=concept,
                    agg_func=agg_func,
                    is_categorical=is_categorical,
                    is_multi_label=is_multi_label
                )
                logger.info(f"  ✓ {concept}_{agg_func} completed")
            except Exception as e:
                logger.error(f"  ✗ {concept}_{agg_func} failed: {str(e)}")
                raise
    
    logger.info("\n" + "="*80)
    logger.info("ALL CONCEPTS MAPPED SUCCESSFULLY")
    logger.info("="*80)


# ============================================================================
# Utility Functions
# ============================================================================

def check_concept_type(concept_path: str) -> Dict[str, bool]:
    """
    Analyze a concept to determine if it's categorical.
    
    Args:
        concept_path: Path to concept pickle file
    
    Returns:
        Dict with analysis results
    """
    df = pd.read_pickle(concept_path)
    
    results = {}
    
    for feature in df['FEATURE'].unique():
        feature_data = df[df['FEATURE'] == feature]['VALUE']
        
        # Try to convert to numeric
        numeric_data = pd.to_numeric(feature_data, errors='coerce')
        
        # Calculate metrics
        n_unique = feature_data.nunique()
        n_total = len(feature_data)
        unique_ratio = n_unique / n_total if n_total > 0 else 0
        
        # Check if conversion failed (indicates categorical)
        n_non_numeric = numeric_data.isna().sum() - feature_data.isna().sum()
        is_string = feature_data.dtype == 'object'
        
        # Heuristic: categorical if:
        # 1. Contains non-numeric values, OR
        # 2. Low cardinality (< 20% unique values) and is object type
        is_categorical = (n_non_numeric > 0) or (is_string and unique_ratio < 0.2 and n_unique < 100)
        
        results[feature] = {
            'is_categorical': is_categorical,
            'n_unique': n_unique,
            'unique_ratio': unique_ratio,
            'n_non_numeric': n_non_numeric,
            'dtype': str(feature_data.dtype)
        }
    
    return results


def suggest_categorical_concepts(cfg: Dict) -> Dict[str, List[str]]:
    """
    Analyze all concepts and suggest which should be categorical.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        Dict mapping concept names to list of categorical features
    """
    concepts = cfg.get("concepts", [])
    suggestions = {}
    
    logger.info("Analyzing concepts for categorical features...")
    logger.info("="*80)
    
    for concept in concepts:
        concept_path = f"data/interim/concepts/{concept}.pkl"
        
        try:
            results = check_concept_type(concept_path)
            
            categorical_features = [
                feat for feat, info in results.items() 
                if info['is_categorical']
            ]
            
            if categorical_features:
                suggestions[concept] = categorical_features
                
                logger.info(f"\n{concept}: Found {len(categorical_features)} categorical features")
                for feat in categorical_features[:5]:  # Show first 5
                    info = results[feat]
                    logger.info(f"  - {feat}: {info['n_unique']} unique values, "
                              f"{info['n_non_numeric']} non-numeric")
                
                if len(categorical_features) > 5:
                    logger.info(f"  ... and {len(categorical_features) - 5} more")
        
        except Exception as e:
            logger.warning(f"Could not analyze {concept}: {str(e)}")
    
    logger.info("\n" + "="*80)
    logger.info("SUGGESTIONS:")
    logger.info("Add to your config:")
    logger.info("cfg['cat_time_series'] = {")
    logger.info("    'concepts': {")
    
    for concept, features in suggestions.items():
        if len(features) == len(results):
            logger.info(f"        '{concept}': ['all'],  # All features categorical")
        else:
            logger.info(f"        '{concept}': {features},")
    
    logger.info("    },")
    logger.info("    'multi_label': [],  # Add concepts that can have multiple values per bin")
    logger.info("}")
    logger.info("="*80)
    
    return suggestions

import time


def assign_bins_vectorized_fast(concept_df, bin_df):
    """
    Vectorized bin assignment using numpy searchsorted.
    
    This is MUCH faster than merge_asof for large datasets.
    
    Speed: ~1M timestamps in <5 seconds
    """
    logger.info(f"Vectorized bin assignment for {len(concept_df)} timestamps...")
    start_time = time.time()
    
    # Sort data
    concept_df = concept_df.sort_values(['PID', 'TIMESTAMP']).reset_index(drop=True)
    bin_df = bin_df.sort_values(['PID', 'bin_start']).reset_index(drop=True)
    
    # Process all patients at once using vectorized operations
    results = []
    
    unique_pids = concept_df['PID'].unique()
    n_pids = len(unique_pids)
    
    logger.info(f"Processing {n_pids} patients...")
    
    # Group once (much faster than repeated filtering)
    concept_grouped = concept_df.groupby('PID')
    bin_grouped = bin_df.groupby('PID')
    
    for i, pid in enumerate(unique_pids):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (n_pids - i) / rate
            logger.info(f"  Progress: {i}/{n_pids} ({100*i/n_pids:.1f}%) - ETA: {eta:.0f}s")
        
        # Get patient data (already grouped, very fast)
        try:
            patient_data = concept_grouped.get_group(pid)
            patient_bins = bin_grouped.get_group(pid)
        except KeyError:
            # Patient has no bins
            continue
        
        if len(patient_bins) == 0:
            continue
        
        # Vectorized search: find which bin each timestamp belongs to
        # This is MUCH faster than iterating
        timestamps = patient_data['TIMESTAMP'].values
        bin_starts = patient_bins['bin_start'].values
        bin_ends = patient_bins['bin_end'].values
        
        # Use searchsorted to find bin indices (O(n log n) instead of O(n^2))
        indices = np.searchsorted(bin_starts, timestamps, side='right') - 1
        
        # Validate: timestamp must be within [bin_start, bin_end)
        valid_mask = (indices >= 0) & (indices < len(patient_bins))
        
        if valid_mask.any():
            # Further validate against bin_end
            valid_indices = indices[valid_mask]
            valid_timestamps = timestamps[valid_mask]
            valid_bin_ends = bin_ends[valid_indices]
            
            # Check if timestamp < bin_end
            within_bin = valid_timestamps < valid_bin_ends
            
            # Create result for this patient
            patient_result = patient_data[valid_mask].copy()
            patient_result['bin_counter'] = patient_bins['bin_counter'].iloc[valid_indices[within_bin]].values
            patient_result['bin_freq'] = patient_bins['bin_freq'].iloc[valid_indices[within_bin]].values
            
            results.append(patient_result)
    
    # Combine all results
    if len(results) == 0:
        logger.error("No timestamps matched any bins!")
        return pd.DataFrame()
    
    merged = pd.concat(results, ignore_index=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Bin assignment complete in {elapsed:.1f}s")
    logger.info(f"  Input: {len(concept_df)} rows, {concept_df['PID'].nunique()} patients")
    logger.info(f"  Output: {len(merged)} rows, {merged['PID'].nunique()} patients")
    logger.info(f"  Rate: {len(concept_df)/elapsed:.0f} rows/second")
    
    return merged


def assign_bins_parallel(concept_df, bin_df, n_workers=4):
    """
    Parallel bin assignment using multiprocessing.
    
    Use this for VERY large datasets (>10M rows).
    """
    from multiprocessing import Pool
    import numpy as np
    
    logger.info(f"Parallel bin assignment using {n_workers} workers...")
    
    # Split patients into chunks
    unique_pids = concept_df['PID'].unique()
    pid_chunks = np.array_split(unique_pids, n_workers)
    
    def process_chunk(pid_list):
        """Process a chunk of patients."""
        chunk_data = concept_df[concept_df['PID'].isin(pid_list)]
        chunk_bins = bin_df[bin_df['PID'].isin(pid_list)]
        return assign_bins_vectorized_fast(chunk_data, chunk_bins)
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(process_chunk, pid_chunks)
    
    # Combine results
    merged = pd.concat(results, ignore_index=True)
    
    logger.info(f"Parallel processing complete: {len(merged)} rows")
    
    return merged



def merge_and_aggregate_optimized(
    bin_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    agg_func: str = "mean",
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> pd.DataFrame:
    """
    ULTRA-OPTIMIZED merge and aggregate.
    
    Performance target: ~1-2s per feature for 2M rows
    """
    start_time = time.time()
    
    # Validate inputs
    if 'TIMESTAMP' not in subset_df.columns:
        raise ValueError(f"subset_df missing TIMESTAMP column. Available: {subset_df.columns.tolist()}")
    
    # Ensure TIMESTAMP is datetime
    if not pd.api.types.is_datetime64_any_dtype(subset_df['TIMESTAMP']):
        subset_df = subset_df.copy()
        subset_df['TIMESTAMP'] = pd.to_datetime(subset_df['TIMESTAMP'])
    
    # Ensure PIDs are same type (use int32 for memory efficiency)
    bin_df = bin_df.copy()
    subset_df = subset_df.copy()
    bin_df["PID"] = bin_df["PID"].astype("int32")
    subset_df["PID"] = subset_df["PID"].astype("int32")
    
    # OPTIMIZATION: Pre-filter subset to only PIDs that have bins
    bin_pids = set(bin_df['PID'].unique())
    subset_df = subset_df[subset_df['PID'].isin(bin_pids)]
    
    if len(subset_df) == 0:
        logger.warning("No data after PID filtering")
        result_df = bin_df.copy()
        result_df['FEATURE'] = np.nan
        result_df['VALUE'] = np.nan
        return result_df
    
    logger.debug(f"  Assigning bins for {len(subset_df)} rows...")
    t0 = time.time()
    
    # OPTIMIZED: Vectorized bin assignment using searchsorted
    # Group once (much faster than repeated filtering)
    subset_grouped = subset_df.groupby('PID', sort=False)
    bin_grouped = bin_df.groupby('PID', sort=False)
    
    results = []
    unique_pids = subset_df['PID'].unique()
    
    # Progress tracking
    batch_size = 1000
    
    for i, pid in enumerate(unique_pids):
        if i > 0 and i % batch_size == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(unique_pids) - i) / rate
            logger.debug(f"    Progress: {i}/{len(unique_pids)} ({100*i/len(unique_pids):.0f}%) - ETA: {eta:.0f}s")
        
        try:
            patient_data = subset_grouped.get_group(pid)
            patient_bins = bin_grouped.get_group(pid)
        except KeyError:
            continue
        
        if len(patient_bins) == 0:
            continue
        
        # Vectorized: Use searchsorted to find bin indices
        timestamps = patient_data['TIMESTAMP'].values
        bin_starts = patient_bins['bin_start'].values
        bin_ends = patient_bins['bin_end'].values
        
        # Find bin index for each timestamp
        indices = np.searchsorted(bin_starts, timestamps, side='right') - 1
        
        # Validate: timestamp must be within bin
        valid_mask = (indices >= 0) & (indices < len(patient_bins))
        
        if not valid_mask.any():
            continue
        
        valid_indices = indices[valid_mask]
        valid_timestamps = timestamps[valid_mask]
        
        # Check if timestamp is before bin_end
        within_bin = valid_timestamps < bin_ends[valid_indices]
        
        if not within_bin.any():
            continue
        
        # Create result for this patient
        final_valid_mask = np.zeros(len(patient_data), dtype=bool)
        final_valid_mask[np.where(valid_mask)[0][within_bin]] = True
        
        patient_result = patient_data[final_valid_mask].copy()
        
        # Add bin information
        final_indices = valid_indices[within_bin]
        patient_result['bin_counter'] = patient_bins['bin_counter'].iloc[final_indices].values
        patient_result['bin_start'] = patient_bins['bin_start'].iloc[final_indices].values
        patient_result['bin_end'] = patient_bins['bin_end'].iloc[final_indices].values
        patient_result['bin_freq'] = patient_bins['bin_freq'].iloc[final_indices].values
        
        results.append(patient_result)
    
    if len(results) == 0:
        logger.warning("  No data matched any bins!")
        result_df = bin_df.copy()
        result_df['FEATURE'] = np.nan
        result_df['VALUE'] = np.nan
        return result_df
    
    # Combine all patient results
    filtered_df = pd.concat(results, ignore_index=True)
    
    elapsed = time.time() - t0
    logger.debug(f"  Bin assignment: {len(subset_df)} → {len(filtered_df)} rows in {elapsed:.1f}s")
    
    # === AGGREGATION ===
    t0 = time.time()
    
    if is_categorical:
        if is_multi_label:
            keep_cols = ["PID", "bin_counter", "bin_start", "bin_end", "FEATURE", "VALUE"]
            if "SUB_CODE" in filtered_df.columns:
                keep_cols.append("SUB_CODE")
            aggregated_df = filtered_df[keep_cols].drop_duplicates()
        else:
            if agg_func == "mode":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan})
                    .reset_index()
                )
            elif agg_func in ["last", "first"]:
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": agg_func})
                    .reset_index()
                )
            else:
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "last"})
                    .reset_index()
                )
    else:
        # Continuous: standard aggregation
        aggregation = {
            "first": "first", "mean": "mean", "max": "max", "min": "min",
            "std": "std", "sum": "sum", "count": "count", "last": "last",
        }
        agg_function = aggregation.get(agg_func, "mean")
        
        # Ensure VALUE is numeric
        filtered_df['VALUE'] = pd.to_numeric(filtered_df['VALUE'], errors='coerce')
        filtered_df = filtered_df[filtered_df['VALUE'].notna()]
        
        if len(filtered_df) == 0:
            logger.warning("  No numeric values!")
            result_df = bin_df.copy()
            result_df['FEATURE'] = np.nan
            result_df['VALUE'] = np.nan
            return result_df
        
        aggregated_df = (
            filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
            .agg({"VALUE": agg_function})
            .reset_index()
        )
    
    logger.debug(f"  Aggregation: {time.time()-t0:.1f}s")
    
    # Merge back to bin_df
    t0 = time.time()
    result_df = pd.merge(
        bin_df,
        aggregated_df,
        on=["PID", "bin_counter", "bin_start", "bin_end"],
        how="left",
    )
    logger.debug(f"  Merge back: {time.time()-t0:.1f}s")
    
    total_time = time.time() - start_time
    logger.debug(f"  Total feature processing: {total_time:.1f}s")
    
    return result_df


def clean_dataframe_ultra_fast(result_df):
    """
    Ultra-fast cleaning using agg + merge (no transform).
    
    Expected: ~5-10s for 2.5M rows
    """
    t0 = time.time()
    
    logger.debug(f"  Cleaning {len(result_df)} rows...")
    
    # Step 1: Aggregate to find groups with data (FAST!)
    t1 = time.time()
    group_info = result_df.groupby(['PID', 'bin_counter'], sort=False).agg({
        'FEATURE': lambda x: x.notna().any(),
        'VALUE': lambda x: x.notna().any()
    }).reset_index()
    
    group_info.columns = ['PID', 'bin_counter', '_has_feature', '_has_value']
    group_info['_has_any_data'] = group_info['_has_feature'] | group_info['_has_value']
    
    logger.debug(f"    Aggregation: {time.time()-t1:.1f}s")
    
    # Step 2: Merge back (FAST!)
    t1 = time.time()
    result_with_info = result_df.merge(
        group_info[['PID', 'bin_counter', '_has_any_data']], 
        on=['PID', 'bin_counter'],
        how='left'
    )
    logger.debug(f"    Merge: {time.time()-t1:.1f}s")
    
    # Step 3: Filter (FAST!)
    t1 = time.time()
    mask = (~result_with_info['_has_any_data']) | (
        result_with_info['FEATURE'].notna() & result_with_info['VALUE'].notna()
    )
    
    filtered_df = result_with_info[mask].drop(columns=['_has_any_data']).reset_index(drop=True)
    
    logger.debug(f"    Filter: {time.time()-t1:.1f}s")
    
    total = time.time() - t0
    logger.info(f"[{total:.1f}s] Cleaned: {len(filtered_df)} rows (removed {len(result_df) - len(filtered_df)} rows)")
    
    return filtered_df


def map_concept_optimized(
    cfg,
    concept: str,
    agg_func: str,
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> None:
    """
    FINAL ultra-optimized mapper with all fixes.
    
    Expected performance for 12M rows, 6 features:
    - Loading: ~80s
    - Processing: ~60s (10s per feature)
    - Concatenation: ~3s
    - Cleaning: ~10s
    - Saving: ~10s
    TOTAL: ~160s (2.7min) instead of 8.5min
    """
    from astra.utils import get_bin_df
    from astra.data.filters import collect_filter
    
    overall_start = time.time()
    
    logger.info(f"{'='*80}")
    logger.info(f"MAPPING: {concept} with {agg_func} (ULTRA-OPTIMIZED)")
    logger.info(f"{'='*80}")
    
    output_path = f"data/interim/mapped/{concept}"
    
    # Load binning DataFrame
    t0 = time.time()
    bin_df = get_bin_df()
    logger.info(f"[{time.time()-t0:.1f}s] Loaded bin_df: {bin_df['PID'].nunique()} patients")
    
    # Load and filter concept
    t0 = time.time()
    filter_function = collect_filter(concept)

    # Load Notater.pkl once for concepts that need it
    notes_concepts = ("ITAOversigtsrapport", "ISS_notes", "ISS_computed", "Events")
    notater_df = pd.read_pickle("data/interim/concepts/Notater.pkl") if concept in notes_concepts else None

    # Cross-concept augmentation
    if concept == "VitaleVaerdier":
        concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
        ews_df = pd.read_pickle("data/interim/concepts/EWS.pkl")
        concept_df = filter_function(concept_df, ews=ews_df)
    elif concept == "ITAOversigtsrapport":
        concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
        concept_df = filter_function(concept_df)
        from astra.data.notes_features import build_gcs_from_notes
        gcs_df = build_gcs_from_notes(notater_df)
        concept_df = pd.concat([concept_df, gcs_df], ignore_index=True)
        logger.info(f"Augmented ITAOversigtsrapport with {len(gcs_df)} GCS values from notes")
    elif concept in ("ISS_notes", "ISS_computed"):
        iss_pkl = f"data/interim/concepts/{concept}.pkl"
        concept_df = pd.read_pickle(iss_pkl)
        logger.info(f"Loaded {concept}: {len(concept_df)} rows")
        concept_df = filter_function(concept_df)
    elif concept == "Events":
        from astra.data.cardiac_arrest import build_cardiac_arrest_from_notes
        from astra.data.notes_features import build_intubation_from_notes
        ca_df = build_cardiac_arrest_from_notes(notater_df)
        intub_df = build_intubation_from_notes(notater_df)
        # Restrict intubation to within 24h of admission
        n_before = len(intub_df)
        start = bin_df.groupby("PID")["bin_start"].min()
        intub_df = intub_df.merge(start, on="PID", how="left")
        intub_df = intub_df[intub_df["TIMESTAMP"] <= intub_df["bin_start"] + pd.Timedelta(hours=24)]
        intub_df = intub_df.drop(columns=["bin_start"])
        logger.info(f"Intubation: {n_before} → {len(intub_df)} after 24h admission cutoff")
        concept_df = pd.concat([ca_df, intub_df], ignore_index=True).reset_index(drop=True)
        # Reformat for categorical: FEATURE=constant, VALUE=event_type
        concept_df["VALUE"] = concept_df["FEATURE"]  # 'cardiac_arrest' or 'INTUBATED'
        concept_df["FEATURE"] = "event"
        concept_df = filter_function(concept_df)
    else:
        concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
        concept_df = filter_function(concept_df)
    logger.info(f"[{time.time()-t0:.1f}s] Loaded concept: {len(concept_df)} rows, {concept_df['PID'].nunique()} patients")

    # Expand interval events (e.g., ADT with start+end timestamps) to per-bin rows
    if 'END_TIMESTAMP' in concept_df.columns:
        logger.info("Expanding interval events to per-bin rows...")
        t1 = time.time()
        concept_df = expand_interval_to_bins(concept_df, bin_df)
        logger.info(f"[{time.time()-t1:.1f}s] Expanded to {len(concept_df)} rows")

    # Validate columns
    if 'TIMESTAMP' not in concept_df.columns:
        raise ValueError(f"Concept {concept} missing TIMESTAMP column. Available: {concept_df.columns.tolist()}")

    # Process each feature
    dfs = []
    features = concept_df.FEATURE.unique()
    logger.info(f"Processing {len(features)} features...")
    
    feature_times = []
    
    for i, feat in enumerate(features):
        t0 = time.time()
        
        logger.info(f"  Feature {i+1}/{len(features)}: {feat}")
        subset = concept_df[concept_df.FEATURE == feat]
        
        # Use optimized merge and aggregate
        result_df = merge_and_aggregate_optimized(
            bin_df, 
            subset, 
            agg_func=agg_func,
            is_categorical=is_categorical,
            is_multi_label=is_multi_label
        )
        
        elapsed = time.time() - t0
        feature_times.append(elapsed)
        logger.info(f"    Time: {elapsed:.1f}s")
        
        dfs.append(result_df)
    
    # Report feature processing stats
    avg_time = np.mean(feature_times)
    logger.info(f"Average time per feature: {avg_time:.1f}s")
    
    logger.info("Concatenating feature dataframes...")
    t0 = time.time()
    
    if len(dfs) < 1:
        logger.warning(f"Concept {concept} failed - no features processed")
        bin_df["FEATURE"] = np.nan
        bin_df["VALUE"] = np.nan
        ensure_parent_dir(f"{output_path}_{agg_func}.pkl")
        bin_df.to_pickle(f"{output_path}_{agg_func}.pkl", protocol=4)
        bin_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
        return
    
    result_df = pd.concat(dfs, ignore_index=True).drop_duplicates().sort_values(["PID", "bin_counter"]).reset_index(drop=True)
    
    logger.info(f"[{time.time()-t0:.1f}s] Concatenated: {len(result_df)} rows")
    
    # Handle multi-label expansion
    if is_categorical and is_multi_label:
        t0 = time.time()
        logger.info("Expanding multi-label values...")
        
        is_list = result_df['VALUE'].apply(lambda x: isinstance(x, list) if pd.notna(x) else False)
        
        if is_list.any():
            list_rows = result_df[is_list].copy()
            non_list_rows = result_df[~is_list].copy()
            list_rows_expanded = list_rows.explode('VALUE').reset_index(drop=True)
            result_df = pd.concat([non_list_rows, list_rows_expanded], ignore_index=True)
            logger.info(f"[{time.time()-t0:.1f}s] Expanded {is_list.sum()} multi-label rows")
        else:
            logger.info(f"[{time.time()-t0:.1f}s] No multi-label values to expand")
    
    # CRITICAL: Use ultra-fast cleaning
    logger.info(f"Cleaning dataframe...")
    filtered_df = clean_dataframe_ultra_fast(result_df)
    
    # Log statistics
    logger.info(f"Final shape: {filtered_df.shape}")
    if is_categorical:
        n_unique_values = filtered_df['VALUE'].nunique()
        logger.info(f"Unique categorical values: {n_unique_values}")
        if is_multi_label:
            avg_values_per_bin = filtered_df.groupby(['PID', 'bin_counter']).size().mean()
            logger.info(f"Average values per bin: {avg_values_per_bin:.2f}")
    
    # Save
    t0 = time.time()
    logger.info(f"Saving to {output_path}...")
    ensure_parent_dir(f"{output_path}_{agg_func}.pkl")
    filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl", protocol=4)
    filtered_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
    logger.info(f"[{time.time()-t0:.1f}s] Saved")
    
    total_time = time.time() - overall_start
    logger.info(f"{'='*80}")
    logger.info(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Speedup estimate: {508.8/total_time:.1f}x faster than before")
    logger.info(f"{'='*80}\n")


