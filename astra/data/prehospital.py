"""
Pre-Hospital Journal (PPJ) data extraction pipeline.

Extracts vital signs, GCS, and ABCD assessments from raw PPJ data and
prepares them for integration into the ASTRA pipeline. Produces intermediate
pickle files that are consumed by filters.py (concat into existing concepts)
and build_patient_info.py (prehospital_start for bin_df).

Data flow:
    raw PPJ CSVs + CPR mapping  →  filter to study population
                                →  extract vitals  → prehospital_VitaleVaerdier.pkl
                                →  extract GCS     → prehospital_GCS.pkl
                                →  extract ABCD    → ppj_base_df.pkl
                                →  update base_df  (prehospital_start + ABCD columns)
"""
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from astra.utils import cfg, get_base_df, is_file_present, ensure_parent_dir
from astra.data.mappings import (
    PPJ_MONTH_DICT,
    PPJ_VITALS_MAP,
    PPJ_VITAL_EVENT_CODES,
    PPJ_ABCD_MAP,
    PPJ_VITAL_BOUNDS,
    ABCD_SEVERITY,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _normalize_journal_id(series: pd.Series) -> pd.Series:
    """Normalize JournalID to consistent string representation.

    Handles the float→string round-trip issue: pandas may read integer IDs
    as float64 (e.g. 12345 → 12345.0), producing "12345.0" via astype(str)
    instead of "12345". This strips trailing '.0' from numeric-looking IDs
    while preserving UUID-style string IDs unchanged.
    """
    s = series.astype(str)
    # Strip trailing .0 from float-like values (e.g. "12345.0" → "12345")
    s = s.str.replace(r'\.0$', '', regex=True)
    return s


# ============================================================================
# Config helpers
# ============================================================================

def _get_sources(cfg) -> list:
    """Return the list of PPJ source definitions from config.

    Supports new multi-source format (``prehospital_config.sources``) and
    falls back to wrapping the old flat keys into a single-element list for
    backward compatibility.
    """
    ph_cfg = cfg.get("prehospital_config", {})
    sources = ph_cfg.get("sources")
    if sources:
        return sources

    # Backward compat: wrap old flat keys into a single source
    return [{
        "name": "RegH",
        "ppj_mapping_path": ph_cfg.get("ppj_mapping_path", "data/raw/ppj_mapping.csv"),
        "ppj_data_paths": ph_cfg.get("ppj_data_paths", []),
        "mapping_sep": ";",
        "mapping_timestamp_format": "sas",
    }]


# ============================================================================
# Timestamp parsing
# ============================================================================

def parse_ppj_timestamps(series: pd.Series) -> pd.Series:
    """Convert PPJ timestamp strings to datetime.

    PPJ format: ``22FEB2018:13:40:02.2750`` — 3-letter month abbreviation
    embedded in a numeric date string.  We replace month names with numbers
    then parse with ``pd.to_datetime``.
    """
    s = series.astype(str)
    for month_abbr, month_num in PPJ_MONTH_DICT.items():
        s = s.str.replace(month_abbr, month_num, regex=False)
    return pd.to_datetime(s, format="%d%m%Y:%H:%M:%S.%f", errors="coerce")


# ============================================================================
# Data loading
# ============================================================================

def load_ppj_mapping(cfg) -> pd.DataFrame:
    """Load CPR_hash → JournalID mapping from all configured PPJ sources.

    Iterates over ``prehospital_config.sources``, loading each mapping CSV
    with its source-specific separator and timestamp format.  Adds a
    ``PrehospitalRegion`` column to track data origin.

    Returns DataFrame with columns
    [CPR_hash, JournalID, CreationTime_dt, PrehospitalRegion].
    """
    sources = _get_sources(cfg)
    dfs = []

    for source in sources:
        name = source.get("name", "Unknown")
        mapping_path = source.get("ppj_mapping_path")
        sep = source.get("mapping_sep", ";")
        ts_format = source.get("mapping_timestamp_format", "sas")

        if not mapping_path or not os.path.exists(mapping_path):
            logger.warning(f"PPJ mapping not found for source '{name}': {mapping_path}")
            continue

        logger.info(f"Loading PPJ mapping for '{name}' from {mapping_path} (sep='{sep}')")
        df = pd.read_csv(mapping_path, sep=sep)

        # Drop unnamed index columns if present
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.drop_duplicates(inplace=True)

        logger.info(f"  [{name}] columns: {df.columns.tolist()}, "
                     f"{len(df)} rows, dtypes:\n{df.dtypes.to_string()}")

        # Validate required columns
        required = {"CPR_hash", "JournalID"}
        missing = required - set(df.columns)
        if missing:
            logger.error(f"  [{name}] Missing required columns: {missing} — skipping source")
            continue
        # Check for CreationTime (required for temporal filtering)
        if "CreationTime" not in df.columns and "CreationTime_dt" not in df.columns:
            logger.warning(f"  [{name}] No CreationTime column — temporal filtering will be limited")

        # Log sample values for debugging
        logger.info(f"  [{name}] CPR_hash samples: {df['CPR_hash'].head(3).tolist()}")
        logger.info(f"  [{name}] JournalID samples (raw): {df['JournalID'].head(3).tolist()} "
                     f"(dtype={df['JournalID'].dtype})")

        # Parse timestamps
        if "CreationTime" in df.columns:
            if ts_format == "sas":
                df["CreationTime_dt"] = parse_ppj_timestamps(df["CreationTime"])
            else:
                df["CreationTime_dt"] = pd.to_datetime(df["CreationTime"], errors="coerce")
            n_parsed = df["CreationTime_dt"].notna().sum()
            logger.info(f"  [{name}] Parsed CreationTime: {n_parsed}/{len(df)} non-NaT")
            if n_parsed == 0:
                logger.error(f"  [{name}] All CreationTime values failed to parse!")

        # Normalize JournalID to string to prevent type mismatches across sources
        if "JournalID" in df.columns:
            df["JournalID"] = _normalize_journal_id(df["JournalID"])
            logger.info(f"  [{name}] JournalID samples (normalized): {df['JournalID'].head(3).tolist()}")

        logger.info(f"  [{name}] {len(df)} rows, {df['CPR_hash'].nunique()} unique CPR_hash, "
                     f"{df['JournalID'].nunique()} unique JournalID")

        df["PrehospitalRegion"] = name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No PPJ mapping files found for any configured source")

    ppj_map = pd.concat(dfs, ignore_index=True)
    ppj_map.drop_duplicates(subset=["CPR_hash", "JournalID"], inplace=True)

    logger.info(f"PPJ mapping combined: {len(ppj_map)} rows, "
                f"{ppj_map['CPR_hash'].nunique()} unique CPR_hash, "
                f"sources: {ppj_map['PrehospitalRegion'].value_counts().to_dict()}")

    return ppj_map


def load_ppj_data(cfg) -> pd.DataFrame:
    """Load raw PPJ CSV file(s) from all configured sources.

    Reads semicolon-delimited CSVs with columns:
    ``EventCodeName, CreationTime, ManualTime, ValueFloat, ValueString,
    ValueDateTime, ValueBool, JournalID``

    Parses timestamps, replaces empty quoted strings with NaN, and removes
    CPR identity entries (EventCodeName == 'PAT00013').
    """
    # Collect data paths from all sources
    sources = _get_sources(cfg)
    ppj_paths = []
    for source in sources:
        paths = source.get("ppj_data_paths", [])
        if isinstance(paths, str):
            paths = [paths]
        ppj_paths.extend(paths)

    if not ppj_paths:
        # Backward compat: try old flat key
        ph_cfg = cfg.get("prehospital_config", {})
        ppj_paths = ph_cfg.get("ppj_data_paths", [])
        if isinstance(ppj_paths, str):
            ppj_paths = [ppj_paths]

    dfs = []
    for path in ppj_paths:
        p = Path(path)
        if p.is_dir():
            csv_files = sorted(p.glob("*.csv"))
        else:
            csv_files = [p]

        for csv_file in csv_files:
            logger.info(f"Loading PPJ data from {csv_file}")
            df = pd.read_csv(csv_file, sep=";", encoding="utf-8",
                             low_memory=False, on_bad_lines="warn")
            # Drop unnamed index columns
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

            # Validate expected columns
            expected_cols = {"EventCodeName", "CreationTime", "ValueFloat",
                             "ValueString", "JournalID"}
            actual_cols = set(df.columns)
            missing = expected_cols - actual_cols
            if missing:
                logger.error(f"  {csv_file}: MISSING expected columns {missing}! "
                             f"Got: {df.columns.tolist()}. "
                             f"Possible column shift from bad delimiter handling.")
            extra = actual_cols - expected_cols - {"ManualTime", "ValueDateTime", "ValueBool"}
            if extra:
                logger.warning(f"  {csv_file}: Unexpected extra columns: {extra}")

            # Log per-file stats
            n_journals = df["JournalID"].nunique() if "JournalID" in df.columns else "?"
            logger.info(f"  → {len(df):,} rows, {n_journals} journals, "
                         f"JournalID dtype={df['JournalID'].dtype if 'JournalID' in df.columns else 'N/A'}, "
                         f"JournalID samples={df['JournalID'].head(3).tolist() if 'JournalID' in df.columns else []}")

            # Check for signs of column shifting (e.g. JournalID contains timestamps)
            if "JournalID" in df.columns:
                jid_sample = df["JournalID"].dropna().head(10).astype(str)
                if jid_sample.str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).any():
                    logger.error(f"  {csv_file}: JournalID contains date-like values — "
                                  f"likely column shift! Samples: {jid_sample.tolist()}")

            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No PPJ data files found at configured paths: {ppj_paths}"
        )

    n_before_concat = sum(len(d) for d in dfs)
    ppj = pd.concat(dfs, ignore_index=True)
    n_after_concat = len(ppj)
    ppj.drop_duplicates(inplace=True)
    n_after_dedup = len(ppj)
    logger.info(f"PPJ data: {n_before_concat:,} rows from {len(dfs)} files → "
                f"{n_after_concat:,} after concat → {n_after_dedup:,} after dedup "
                f"(dropped {n_after_concat - n_after_dedup:,} duplicates)")

    # Normalize JournalID to string (must match mapping dtype)
    if "JournalID" in ppj.columns:
        ppj["JournalID"] = _normalize_journal_id(ppj["JournalID"])
        logger.info(f"JournalID normalized: dtype={ppj['JournalID'].dtype}, "
                     f"nunique={ppj['JournalID'].nunique()}, "
                     f"samples={ppj['JournalID'].head(3).tolist()}")

    # Parse timestamps — try standard datetime first, fall back to PPJ SAS format
    for col in ["CreationTime", "ManualTime"]:
        if col in ppj.columns:
            # Log sample raw values before parsing
            raw_sample = ppj[col].dropna().head(3).tolist()
            logger.info(f"Raw {col} samples (before parsing): {raw_sample}")

            # Try standard datetime parsing (ISO, etc.)
            parsed = pd.to_datetime(ppj[col], errors="coerce")
            n_standard = parsed.notna().sum()

            if n_standard == 0:
                # Fall back to PPJ-specific SAS-style format (22FEB2018:13:40:02.2750)
                logger.info(f"Standard parse yielded 0 timestamps for {col}, trying PPJ SAS format")
                parsed = parse_ppj_timestamps(ppj[col])

            ppj[col] = parsed
            logger.info(f"Parsed {col}: {ppj[col].notna().sum()}/{len(ppj)} non-NaT")

    # Replace empty quoted strings with NaN
    ppj.replace('""', np.nan, inplace=True)

    # Remove CPR identity entries (privacy)
    ppj = ppj[ppj["EventCodeName"] != "PAT00013"].copy()

    logger.info(f"Loaded {len(ppj)} PPJ records across {ppj['JournalID'].nunique()} journals")
    logger.info(f"PPJ data columns: {ppj.columns.tolist()}")
    logger.info(f"PPJ dtypes:\n{ppj.dtypes}")
    logger.debug(f"PPJ sample (first 3 rows):\n{ppj.head(3).to_string()}")
    return ppj


def _prefilter_raw_sources(cfg, matched_jids) -> None:
    """Pre-filter large raw PPJ files by chunked reading.

    Some sources (e.g. RegSJ) provide a single large CSV containing all
    patients, not just the study population.  When a source defines
    ``raw_data_path``, this function reads it in chunks, filters to the
    matched JournalIDs, and saves the result to the first entry in
    ``ppj_data_paths`` so that ``load_ppj_data()`` can read it normally.

    Skips sources without ``raw_data_path`` or where the output already exists.
    """
    sources = _get_sources(cfg)
    # Normalize using the same logic as _normalize_journal_id
    matched_jids_set = set(_normalize_journal_id(pd.Series(matched_jids)))

    for source in sources:
        raw_path = source.get("raw_data_path")
        if not raw_path:
            continue

        # Output path = first entry in ppj_data_paths
        out_paths = source.get("ppj_data_paths", [])
        if isinstance(out_paths, str):
            out_paths = [out_paths]
        if not out_paths:
            logger.warning(f"[{source['name']}] raw_data_path set but no ppj_data_paths — skipping")
            continue
        out_path = out_paths[0]

        # Skip if already filtered
        if os.path.exists(out_path):
            logger.info(f"[{source['name']}] Pre-filtered file already exists: {out_path}")
            continue

        if not os.path.exists(raw_path):
            logger.warning(f"[{source['name']}] Raw data file not found: {raw_path}")
            continue

        logger.info(f"[{source['name']}] Pre-filtering {raw_path} → {out_path} "
                     f"({len(matched_jids_set)} matched JournalIDs)")

        chunk_size = 1_000_000
        all_chunks = []
        n_total = 0
        for chunk in pd.read_csv(
            raw_path, chunksize=chunk_size, sep=";",
            on_bad_lines="warn", low_memory=False
        ):
            n_total += len(chunk)
            chunk["JournalID"] = _normalize_journal_id(chunk["JournalID"])
            filtered = chunk[chunk["JournalID"].isin(matched_jids_set)]
            if len(filtered) > 0:
                all_chunks.append(filtered)
            logger.info(f"  Processed {n_total:,} rows, kept {sum(len(c) for c in all_chunks):,} so far")

        if not all_chunks:
            logger.warning(f"[{source['name']}] No matching records found in {raw_path}")
            continue

        result = pd.concat(all_chunks, ignore_index=True)
        ensure_parent_dir(out_path)
        result.to_csv(out_path, sep=";", index=False)
        logger.info(f"[{source['name']}] Saved {len(result):,} filtered records to {out_path}")


# ============================================================================
# Population filtering
# ============================================================================

def filter_ppj_to_population(
    ppj: pd.DataFrame,
    ppj_map: pd.DataFrame,
    base_df: pd.DataFrame,
    max_hours_before: float = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter PPJ records to patients in the study population.

    1. Links JournalIDs to CPR_hash via ppj_map.
    2. Merges with base_df to get hospital admission times.
    3. Filters to records within ``max_hours_before`` hours before admission
       and before hospital discharge.

    Returns:
        ppj_filtered: PPJ records with PID column added.
        ph_pop: One row per PID with prehospital_start/end times.
    """
    # Link PPJ mapping to base population
    ppj_pop = ppj_map[ppj_map["CPR_hash"].isin(base_df["CPR_hash"])].copy()
    logger.info(f"PPJ mapping: {ppj_pop['CPR_hash'].nunique()} patients matched to study population")
    if "PrehospitalRegion" in ppj_pop.columns:
        logger.info(f"  Per-source match: {ppj_pop.groupby('PrehospitalRegion')['CPR_hash'].nunique().to_dict()}")

    # Log JournalID overlap between mapping and raw PPJ data
    map_jids = set(ppj_pop["JournalID"].unique())
    ppj_jids = set(ppj["JournalID"].unique())
    overlap = map_jids & ppj_jids
    logger.info(f"  JournalID overlap: {len(overlap)} of {len(map_jids)} mapping JIDs "
                f"found in {len(ppj_jids)} data JIDs")
    if len(overlap) < len(map_jids):
        map_only = map_jids - ppj_jids
        logger.info(f"  JIDs in mapping but NOT in data: {len(map_only)} "
                     f"(samples: {list(map_only)[:5]})")
        ppj_only_sample = list(ppj_jids - map_jids)[:5]
        logger.info(f"  JIDs in data but NOT in mapping: {len(ppj_jids - map_jids)} "
                     f"(samples: {ppj_only_sample})")

    # Merge mapping with base_df to get admission/discharge times + PID
    base_cols = ["CPR_hash", "PID", "start", "end"]
    ph = base_df[base_cols].merge(ppj_pop, on="CPR_hash", how="inner")

    # Compute hours before admission
    if "CreationTime_dt" in ph.columns:
        # Ensure datetime types for subtraction
        ph["CreationTime_dt"] = pd.to_datetime(ph["CreationTime_dt"])
        ph["start"] = pd.to_datetime(ph["start"])
        ph["end"] = pd.to_datetime(ph["end"])

        ph["delta_hours_start"] = (
            ph["CreationTime_dt"] - ph["start"]
        ).dt.total_seconds() / 3600

        # Filter: within max_hours_before before admission, and before discharge
        ph = ph[
            (ph["CreationTime_dt"] <= ph["end"])
            & (ph["delta_hours_start"] >= -max_hours_before)
        ].drop_duplicates()

    logger.info(f"PPJ population after time filtering: {ph['PID'].nunique()} patients")
    if "PrehospitalRegion" in ph.columns:
        logger.info(f"  Per-source after time filter: "
                     f"{ph.groupby('PrehospitalRegion')['PID'].nunique().to_dict()}")

    # Filter raw PPJ to matched JournalIDs
    valid_jids = set(ph["JournalID"].unique())
    n_before_jid_filter = len(ppj)
    ppj_filtered = ppj[ppj["JournalID"].isin(valid_jids)].copy()
    logger.info(f"PPJ data filtered by JournalID: {n_before_jid_filter:,} → {len(ppj_filtered):,} rows "
                f"({len(valid_jids)} valid JIDs)")

    # Add PID and PrehospitalRegion via JournalID
    jid_cols = ["JournalID", "PID"]
    if "PrehospitalRegion" in ph.columns:
        jid_cols.append("PrehospitalRegion")
    jid_to_pid = ph[jid_cols].drop_duplicates()
    n_before_pid = len(ppj_filtered)
    ppj_filtered = ppj_filtered.merge(jid_to_pid, on="JournalID", how="left")
    n_no_pid = ppj_filtered["PID"].isna().sum()
    if n_no_pid > 0:
        logger.warning(f"  {n_no_pid:,} PPJ rows have no PID after JournalID merge — dropping")
    ppj_filtered = ppj_filtered[ppj_filtered["PID"].notnull()].copy()

    logger.info(f"ppj_filtered: {len(ppj_filtered):,} rows, columns: {ppj_filtered.columns.tolist()}")
    if "PrehospitalRegion" in ppj_filtered.columns:
        logger.info(f"  Per-source: {ppj_filtered.groupby('PrehospitalRegion').size().to_dict()}")
        logger.info(f"  Per-source unique PIDs: "
                     f"{ppj_filtered.groupby('PrehospitalRegion')['PID'].nunique().to_dict()}")
    if "EventCodeName" in ppj_filtered.columns and len(ppj_filtered) > 0:
        logger.info(
            f"ppj_filtered unique EventCodeNames ({ppj_filtered['EventCodeName'].nunique()}): "
            f"{ppj_filtered['EventCodeName'].value_counts().head(30).to_dict()}"
        )

        # Look up top event codes in event_descriptions (both sheets)
        ph_cfg = cfg.get("prehospital_config", {})
        ed_path = ph_cfg.get("event_descriptions_path")
        if ed_path and os.path.exists(ed_path):
            try:
                ed = _load_event_descriptions(ed_path)
                top_codes = ppj_filtered["EventCodeName"].value_counts().head(15).index.tolist()
                for code in top_codes:
                    row = ed[ed["Kode"] == code]
                    if not row.empty:
                        dt = row["Datatype"].iloc[0] if "Datatype" in row.columns else "?"
                        logger.info(
                            f"  {code}: Tekst='{row['Tekst'].iloc[0]}', Datatype='{dt}'"
                        )
            except Exception as e:
                logger.debug(f"Could not look up event codes in event_descriptions: {e}")

    # Build per-PID population summary (ph_pop)
    # Keep PrehospitalRegion for per-source tracking (ADT events, timing)
    pop_cols = ["CPR_hash", "PID", "start", "end"]
    dedup_cols = ["PID"]
    if "PrehospitalRegion" in ph.columns:
        pop_cols.append("PrehospitalRegion")
        dedup_cols.append("PrehospitalRegion")
    ph_pop = ph[pop_cols].drop_duplicates(subset=dedup_cols)

    return ppj_filtered, ph_pop


# ============================================================================
# Event code resolution from event_descriptions xlsx
# ============================================================================

def _load_event_descriptions(ed_path: str) -> pd.DataFrame:
    """Load both sheets from event_descriptions and merge them.

    The PPJ event_descriptions xlsx has two relevant sheets:
    - "Prædefinerede eventkoder" — predefined codes (ABCD, GCS components, etc.)
    - "Eventkoder Vitaldata" — vital sign / monitoring codes (HR, SBP, DBP, SpO2)

    Returns merged DataFrame with columns [Kode, Tekst, Datatype, ...].
    """
    try:
        ed_pre = pd.read_excel(ed_path, sheet_name="Prædefinerede eventkoder", engine="openpyxl")
        ed_vitals = pd.read_excel(ed_path, sheet_name="Eventkoder Vitaldata", engine="openpyxl")
        logger.info(f"event_descriptions: {len(ed_pre)} predefined + {len(ed_vitals)} vitaldata rows")
        ed = pd.concat([ed_pre, ed_vitals], ignore_index=True)
        return ed
    except Exception as e:
        logger.warning(f"Could not load event_descriptions from {ed_path}: {e}")
        return pd.DataFrame()


def _resolve_vital_event_codes() -> dict:
    """Resolve vital sign event codes from event_descriptions xlsx.

    Looks up the PPJ subset names (M_Puls, M_NInv Sys Blodtryk, etc.)
    in the "Eventkoder Vitaldata" sheet to find their actual event codes
    (typically OMI codes).

    Returns dict mapping {event_code: subset_name}, e.g. {"OMI00001": "M_Puls"}.
    Falls back to hardcoded PPJ_VITAL_EVENT_CODES if xlsx not available.
    """
    ph_cfg = cfg.get("prehospital_config", {})
    ed_path = ph_cfg.get("event_descriptions_path")

    if not ed_path or not os.path.exists(ed_path):
        logger.warning("event_descriptions not available, using hardcoded vital event codes")
        return dict(PPJ_VITAL_EVENT_CODES)

    try:
        ed_vitals = pd.read_excel(ed_path, sheet_name="Eventkoder Vitaldata", engine="openpyxl")
        logger.info(f"Vitaldata sheet: {len(ed_vitals)} rows, columns: {ed_vitals.columns.tolist()}")

        # Look up each subset name from PPJ_VITALS_MAP
        subset_names = list(PPJ_VITALS_MAP.keys())  # ["M_NInv Sys Blodtryk", "M_NInv Dia Blodtryk", "M_Puls", "M_SpO2"]
        code_to_subset = {}

        for subset_name in subset_names:
            matches = ed_vitals[ed_vitals["Tekst"] == subset_name]
            if not matches.empty:
                code = matches["Kode"].iloc[0]
                code_to_subset[code] = subset_name
                logger.info(f"  Vital '{subset_name}' → code '{code}'")
            else:
                logger.warning(f"  Vital '{subset_name}' not found in Eventkoder Vitaldata")

        if code_to_subset:
            logger.info(f"Resolved vital event codes: {code_to_subset}")
            return code_to_subset

    except Exception as e:
        logger.warning(f"Could not read Eventkoder Vitaldata: {e}")

    logger.warning("Falling back to hardcoded vital event codes (SVD — likely wrong!)")
    return dict(PPJ_VITAL_EVENT_CODES)


# ============================================================================
# Concept extraction
# ============================================================================

def extract_ppj_vitals(
    ppj_filtered: pd.DataFrame,
    ph_pop: pd.DataFrame,
) -> pd.DataFrame:
    """Extract pre-hospital vital signs from PPJ.

    Resolves vital sign event codes from the "Eventkoder Vitaldata" sheet,
    maps to ASTRA standard feature names (SBP, DBP, HR, SPO2), applies
    outlier bounds, and produces a DataFrame in standard ASTRA format.
    """
    # Dynamically resolve vital event codes from event_descriptions
    vital_code_to_subset = _resolve_vital_event_codes()
    vital_codes = list(vital_code_to_subset.keys())
    logger.info(f"Filtering for vital event codes: {vital_codes}")

    vitals = ppj_filtered[ppj_filtered["EventCodeName"].isin(vital_codes)].copy()
    logger.info(f"Vital sign matches: {len(vitals)} rows")

    if len(vitals) > 0:
        logger.info(f"  CreationTime non-null: {vitals['CreationTime'].notna().sum()}, "
                     f"ValueFloat non-null: {vitals['ValueFloat'].notna().sum()}")
        # Show value ranges per code
        for code in vital_codes:
            code_rows = vitals[vitals["EventCodeName"] == code]
            if len(code_rows) > 0:
                vals = pd.to_numeric(code_rows["ValueFloat"], errors="coerce").dropna()
                if len(vals) > 0:
                    subset_name = vital_code_to_subset.get(code, "?")
                    astra_name = PPJ_VITALS_MAP.get(subset_name, "?")
                    logger.info(f"  {code} ({subset_name} → {astra_name}): "
                                 f"n={len(vals)}, range=[{vals.min():.1f}, {vals.max():.1f}], "
                                 f"mean={vals.mean():.1f}")

    if vitals.empty:
        logger.warning("No pre-hospital vital signs found in PPJ data")
        return pd.DataFrame(columns=["TIMESTAMP", "PID", "FEATURE", "VALUE"])

    # Map event codes → subset names → ASTRA standard names
    vitals["FEATURE"] = vitals["EventCodeName"].map(vital_code_to_subset)
    vitals["FEATURE"] = vitals["FEATURE"].map(PPJ_VITALS_MAP)

    # Use ManualTime if available, else CreationTime
    vitals["TIMESTAMP"] = vitals["ManualTime"].fillna(vitals["CreationTime"])
    vitals["VALUE"] = pd.to_numeric(vitals["ValueFloat"], errors="coerce")

    # Select and clean
    vitals = vitals[["TIMESTAMP", "PID", "FEATURE", "VALUE"]].copy()
    vitals = vitals.dropna(subset=["VALUE", "TIMESTAMP", "PID"])

    # Apply outlier bounds
    for feature, (low, high) in PPJ_VITAL_BOUNDS.items():
        mask = vitals["FEATURE"] == feature
        vitals = vitals[~(mask & ((vitals["VALUE"] < low) | (vitals["VALUE"] > high)))]

    # Ensure VALUE is string to match in-hospital VitaleVaerdier format
    vitals["VALUE"] = vitals["VALUE"].astype(str)

    vitals = vitals.sort_values(["PID", "TIMESTAMP"]).reset_index(drop=True)
    logger.info(
        f"Extracted {len(vitals)} pre-hospital vital measurements "
        f"across {vitals['PID'].nunique()} patients"
    )

    # Save
    ensure_parent_dir("data/interim/prehospital_VitaleVaerdier.pkl")
    vitals.to_pickle("data/interim/prehospital_VitaleVaerdier.pkl", protocol=4)
    return vitals


def extract_ppj_gcs(
    ppj_filtered: pd.DataFrame,
    ph_pop: pd.DataFrame,
) -> pd.DataFrame:
    """Extract pre-hospital GCS from PPJ.

    GCS in the PPJ system is identified by the subset name 'GCS' (a numerical
    measurement). Uses ManualTime if available, otherwise CreationTime.

    Output format: [TIMESTAMP, PID, FEATURE="GCS", VALUE].
    """
    # GCS can appear under various event codes — filter by known GCS subset
    # In the PPJ system, GCS is typically extracted after subset collection.
    # We look for numeric GCS values in the PPJ data.
    # The event code for GCS needs to be identified from event_descriptions.
    # For now, filter by ValueFloat presence and known GCS event codes.
    gcs_codes = _get_gcs_event_codes(ppj_filtered)
    logger.info(f"GCS event codes resolved to: {gcs_codes}")

    if not gcs_codes:
        logger.warning("No GCS event codes identified in PPJ data")
        return pd.DataFrame(columns=["TIMESTAMP", "PID", "FEATURE", "VALUE"])

    gcs = ppj_filtered[ppj_filtered["EventCodeName"].isin(gcs_codes)].copy()
    logger.info(f"GCS matches: {len(gcs)} rows")

    if len(gcs) > 0:
        logger.info(f"  GCS non-null counts:\n{gcs.notna().sum().to_string()}")
        logger.info(f"  GCS ValueFloat non-null: {gcs['ValueFloat'].notna().sum()}, "
                     f"ValueString non-null: {gcs['ValueString'].notna().sum()}")
        logger.info(f"  GCS sample rows:\n{gcs.head(5).to_string()}")
        # Check value ranges per GCS code
        for code in gcs_codes:
            code_rows = gcs[gcs["EventCodeName"] == code]
            if len(code_rows) > 0:
                vals = pd.to_numeric(code_rows["ValueFloat"], errors="coerce").dropna()
                if len(vals) > 0:
                    logger.info(f"  {code}: n={len(vals)}, range=[{vals.min():.1f}, {vals.max():.1f}], "
                                 f"unique={sorted(vals.unique()[:15].tolist())}")

    if gcs.empty:
        logger.warning("No pre-hospital GCS records found")
        return pd.DataFrame(columns=["TIMESTAMP", "PID", "FEATURE", "VALUE"])

    # Use ManualTime if available, else CreationTime
    gcs["TIMESTAMP"] = gcs["ManualTime"].fillna(gcs["CreationTime"])
    gcs["VALUE"] = pd.to_numeric(gcs["ValueFloat"], errors="coerce")
    gcs["FEATURE"] = "GCS"

    gcs = gcs[["TIMESTAMP", "PID", "FEATURE", "VALUE"]].copy()
    gcs = gcs.dropna(subset=["VALUE", "TIMESTAMP", "PID"])

    # GCS bounds: 3-15
    gcs = gcs[(gcs["VALUE"] >= 3) & (gcs["VALUE"] <= 15)]

    gcs["VALUE"] = gcs["VALUE"].astype(str)
    gcs = gcs.sort_values(["PID", "TIMESTAMP"]).reset_index(drop=True)

    logger.info(
        f"Extracted {len(gcs)} pre-hospital GCS measurements "
        f"across {gcs['PID'].nunique()} patients"
    )

    ensure_parent_dir("data/interim/prehospital_GCS.pkl")
    gcs.to_pickle("data/interim/prehospital_GCS.pkl", protocol=4)
    return gcs


def _get_gcs_event_codes(ppj_filtered: pd.DataFrame) -> list:
    """Identify GCS event codes from PPJ data.

    Loads from BOTH event_descriptions sheets (predefined + vitaldata).
    Prefers codes with float/numeric datatype (actual GCS total scores)
    over listvalue codes (component scores that need decoding).
    """
    ph_cfg = cfg.get("prehospital_config", {})
    ed_path = ph_cfg.get("event_descriptions_path")

    logger.info(f"GCS code resolution — event_descriptions_path: {ed_path}")
    logger.info(f"  exists on disk: {ed_path and os.path.exists(ed_path)}")

    if ed_path and os.path.exists(ed_path):
        try:
            ed = _load_event_descriptions(ed_path)
            if ed.empty:
                raise ValueError("Empty event descriptions")

            gcs_rows = ed[ed["Tekst"].str.contains("GCS", case=False, na=False)]
            logger.info(f"  GCS rows found (both sheets): {len(gcs_rows)}")
            if not gcs_rows.empty:
                # Show all GCS rows with their datatype
                display_cols = [c for c in ["Kode", "Tekst", "Datatype"] if c in gcs_rows.columns]
                logger.info(f"  GCS rows:\n{gcs_rows[display_cols].to_string()}")

                # Prefer float/numeric datatype codes (total GCS) over listvalue (component scores)
                if "Datatype" in gcs_rows.columns:
                    float_gcs = gcs_rows[gcs_rows["Datatype"].str.contains("float|numeric|integer", case=False, na=False)]
                    if not float_gcs.empty:
                        codes = float_gcs["Kode"].tolist()
                        logger.info(f"  Preferring float-type GCS codes: {codes}")
                        return codes

                # Fall back to all GCS codes
                codes = gcs_rows["Kode"].tolist()
                flat_codes = []
                for c in codes:
                    if isinstance(c, list):
                        flat_codes.extend(c)
                    else:
                        flat_codes.append(c)
                logger.info(f"GCS event codes from event_descriptions: {flat_codes}")
                return flat_codes
        except Exception as e:
            logger.warning(f"Could not read event descriptions for GCS codes: {e}")

    # Fallback: scan PPJ data for likely GCS codes
    # GCS values are typically 3-15 (integers)
    logger.info("GCS fallback: scanning PPJ data for likely GCS codes")
    if "ValueFloat" in ppj_filtered.columns:
        numeric_vals = pd.to_numeric(ppj_filtered["ValueFloat"], errors="coerce")
        candidates = ppj_filtered[
            numeric_vals.between(3, 15)
        ]["EventCodeName"].value_counts()
        if not candidates.empty:
            logger.info(f"Candidate GCS event codes (by frequency): {candidates.head(10).to_dict()}")

    logger.warning(
        "Could not determine GCS event codes automatically. "
        "Set prehospital_config.gcs_event_codes in config or provide event_descriptions_path."
    )
    # Return configured codes if available
    configured = ph_cfg.get("gcs_event_codes", [])
    logger.info(f"GCS codes from config fallback: {configured}")
    return configured


def extract_ppj_abcd(
    ppj_filtered: pd.DataFrame,
    ph_pop: pd.DataFrame,
) -> pd.DataFrame:
    """Extract ABCD primary survey assessment from PPJ.

    ABCD assessments are categorical (listvalue type) in the PPJ system:
    - A: Luftveje (Airway) — Fri / Truede / Blokerede
    - B: Respiration (Breathing) — Normal / Let påvirket / Meget påvirket / Respirationsstop
    - C: Cirkulation (Circulation) — Normal / Let påvirket / Meget påvirket / Hjertestop
    - D: Bevidsthedsniveau (Consciousness) — Vågen / Bevidsthedspåvirket / Bevidstløs

    Takes the latest observation per PID for each component.
    Returns one row per PID with columns [PID, A, B, C, D].
    """
    ph_cfg = cfg.get("prehospital_config", {})
    ed_path = ph_cfg.get("event_descriptions_path")
    abcd_codes = ph_cfg.get("abcd_event_codes", {})

    logger.info(f"ABCD extraction — event_descriptions_path: {ed_path}")
    logger.info(f"  exists on disk: {ed_path and os.path.exists(ed_path)}")
    logger.info(f"  abcd_event_codes from config: {abcd_codes}")

    # Try to resolve ABCD event codes from event_descriptions
    if not abcd_codes and ed_path and os.path.exists(ed_path):
        abcd_codes = _get_abcd_event_codes(ed_path)

    logger.info(f"ABCD event codes resolved to: {abcd_codes}")

    if not abcd_codes:
        logger.warning("No ABCD event codes configured — skipping ABCD extraction")
        return pd.DataFrame(columns=["PID"] + list(PPJ_ABCD_MAP.values()))

    result_dfs = []
    for ppj_name, short_name in PPJ_ABCD_MAP.items():
        codes = abcd_codes.get(ppj_name, [])
        if isinstance(codes, str):
            codes = [codes]
        if not codes:
            logger.warning(f"No event code for ABCD component '{ppj_name}'")
            continue

        subset = ppj_filtered[ppj_filtered["EventCodeName"].isin(codes)].copy()
        logger.info(f"  ABCD '{ppj_name}' ({short_name}): {len(subset)} rows matching codes {codes}")
        if subset.empty:
            continue

        logger.info(f"    Non-null: ValueString={subset['ValueString'].notna().sum()}, "
                     f"ValueFloat={subset['ValueFloat'].notna().sum()}, "
                     f"ValueBool={subset['ValueBool'].notna().sum()}")
        logger.info(f"    Sample rows:\n{subset.head(3).to_string()}")

        # Use ValueString for categorical values; fall back to ValueFloat if ValueString is all NaN
        if "ValueString" in subset.columns and subset["ValueString"].notna().any():
            val_col = "ValueString"
        else:
            val_col = "ValueFloat"
        logger.info(f"    Using {val_col} for values")
        subset["value"] = subset[val_col].astype(str).str.replace('"', '')
        logger.info(f"    After str conversion, unique values: {subset['value'].unique()[:10].tolist()}")
        subset = subset[subset["value"].notna() & (subset["value"] != "nan")]
        logger.info(f"    After NaN filter: {len(subset)} rows")

        # Pick most severe observation per PID.
        # ABCD values can be numeric (1.0, 2.0, 4.0, 8.0 — higher = more severe)
        # or text (Fri, Truede, Blokerede). Try numeric first, fall back to
        # text-based severity ordering, then to latest observation.
        subset["_severity"] = pd.to_numeric(subset["value"], errors="coerce")
        if subset["_severity"].notna().any():
            # Numeric encoding: higher = more severe → take max per PID
            logger.info(f"    Using numeric severity (max): range [{subset['_severity'].min()}, {subset['_severity'].max()}]")
            subset = subset.sort_values("_severity").groupby("PID").last().reset_index()
            subset = subset.drop(columns=["_severity"])
        else:
            # Text encoding: use ABCD_SEVERITY ordering
            severity_order = ABCD_SEVERITY.get(short_name, [])
            subset = subset.drop(columns=["_severity"])
            if severity_order:
                unknown_vals = set(subset["value"].unique()) - set(severity_order)
                if unknown_vals:
                    logger.warning(f"    Unknown {short_name} values not in severity list: {unknown_vals}")
                subset["_sev"] = subset["value"].map(
                    {v: i for i, v in enumerate(severity_order)}
                ).fillna(-1).astype(int)
                subset = subset.sort_values("_sev").groupby("PID").last().reset_index()
                subset = subset.drop(columns=["_sev"])
            else:
                # Last resort: take latest observation
                subset["ts"] = subset["ManualTime"].fillna(subset["CreationTime"])
                subset = subset.sort_values("ts").groupby("PID").last().reset_index()
        subset = subset[["PID", "value"]].rename(columns={"value": short_name})

        result_dfs.append(subset)

    if not result_dfs:
        logger.warning("No ABCD data extracted from PPJ")
        return pd.DataFrame(columns=["PID"] + list(PPJ_ABCD_MAP.values()))

    # Merge all ABCD components on PID
    abcd = result_dfs[0]
    for df in result_dfs[1:]:
        abcd = abcd.merge(df, on="PID", how="outer")

    logger.info(
        f"Extracted ABCD assessment for {len(abcd)} patients "
        f"(A: {abcd['A'].notna().sum() if 'A' in abcd else 0}, "
        f"B: {abcd['B'].notna().sum() if 'B' in abcd else 0}, "
        f"C: {abcd['C'].notna().sum() if 'C' in abcd else 0}, "
        f"D: {abcd['D'].notna().sum() if 'D' in abcd else 0})"
    )

    ensure_parent_dir("data/interim/ppj_base_df.pkl")
    abcd.to_pickle("data/interim/ppj_base_df.pkl", protocol=4)
    return abcd


def _get_abcd_event_codes(ed_path: str) -> dict:
    """Load ABCD event codes from event_descriptions_modified.xlsx."""
    try:
        ed_pre = pd.read_excel(
            ed_path, sheet_name="Prædefinerede eventkoder", engine="openpyxl"
        )
        logger.info(f"ABCD code resolution — event_descriptions loaded: {len(ed_pre)} rows")
        logger.info(f"  Columns: {ed_pre.columns.tolist()}")
        logger.info(f"  Sample Tekst values: {ed_pre['Tekst'].head(20).tolist()}")
        codes = {}
        for ppj_name in PPJ_ABCD_MAP:
            matches = ed_pre[ed_pre["Tekst"] == ppj_name]
            logger.info(f"  ABCD lookup '{ppj_name}': {len(matches)} matches")
            if not matches.empty:
                code_val = matches["Kode"].iloc[0]
                codes[ppj_name] = [code_val] if not isinstance(code_val, list) else code_val
                logger.info(f"    → code: {codes[ppj_name]}")
        if codes:
            logger.info(f"ABCD event codes from event_descriptions: {codes}")
        else:
            logger.warning("No ABCD event codes found in event_descriptions")
        return codes
    except Exception as e:
        logger.warning(f"Could not read ABCD event codes from {ed_path}: {e}")
        return {}


# ============================================================================
# Prehospital timing
# ============================================================================

def compute_prehospital_times(
    ppj_filtered: pd.DataFrame,
    ph_pop: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-PID prehospital_start and prehospital_end from PPJ records.

    Computes per-source times (grouped by PID + PrehospitalRegion) and merges
    them into ph_pop.  Also computes global per-PID prehospital_start/end
    (min/max across all sources) for the universal timeline.

    Returns ph_pop with columns:
    - prehospital_start_source: per-(PID, PrehospitalRegion) earliest timestamp
    - prehospital_end_source: per-(PID, PrehospitalRegion) latest timestamp
    - prehospital_start: global earliest across all sources (per PID)
    - prehospital_end: global latest across all sources (per PID)
    """
    ts_col = "CreationTime"
    if ts_col not in ppj_filtered.columns:
        ts_col = "TIMESTAMP"

    has_region = "PrehospitalRegion" in ppj_filtered.columns

    # Per-source times (for ADT event generation)
    if has_region:
        source_times = ppj_filtered.groupby(["PID", "PrehospitalRegion"]).agg(
            prehospital_start_source=(ts_col, "min"),
            prehospital_end_source=(ts_col, "max"),
        ).reset_index()
        for col in ["prehospital_start_source", "prehospital_end_source"]:
            source_times[col] = pd.to_datetime(source_times[col])
        ph_pop = ph_pop.merge(source_times, on=["PID", "PrehospitalRegion"], how="left")

    # Global per-PID times (for universal timeline)
    global_times = ppj_filtered.groupby("PID").agg(
        prehospital_start=(ts_col, "min"),
        prehospital_end=(ts_col, "max"),
    ).reset_index()
    for col in ["prehospital_start", "prehospital_end"]:
        global_times[col] = pd.to_datetime(global_times[col])

    # Drop existing global columns to avoid _x/_y on re-merge
    for col in ["prehospital_start", "prehospital_end"]:
        if col in ph_pop.columns:
            ph_pop = ph_pop.drop(columns=[col])
    ph_pop = ph_pop.merge(global_times, on="PID", how="left")

    n_with_ph = ph_pop["prehospital_start"].notna().sum()
    n_unique_pid = ph_pop["PID"].nunique()
    logger.info(f"Prehospital times computed: {n_with_ph}/{len(ph_pop)} rows have PPJ data "
                f"({n_unique_pid} unique patients)")
    if has_region:
        logger.info(f"  Per-source breakdown: "
                     f"{ph_pop.groupby('PrehospitalRegion')['prehospital_start_source'].count().to_dict()}")

    return ph_pop


# ============================================================================
# Prehospital ADT events
# ============================================================================

def generate_prehospital_adt(
    base_df: pd.DataFrame,
    ph_pop: pd.DataFrame,
) -> pd.DataFrame:
    """Create interval-based ADT events for prehospital transport per source.

    For each (PID, PrehospitalRegion) with PPJ data, creates an ADT event:
    - FEATURE = "ADT"
    - VALUE = "PREHOSP_{region}" (e.g. PREHOSP_REGH, PREHOSP_REGSJ)
    - TIMESTAMP = per-source prehospital_start
    - END_TIMESTAMP = inhospital_start

    Saves to data/interim/prehospital_ADT.pkl and returns the DataFrame.
    """
    if "PrehospitalRegion" not in ph_pop.columns:
        logger.info("No PrehospitalRegion in ph_pop — skipping ADT generation")
        return pd.DataFrame(columns=["PID", "FEATURE", "VALUE", "TIMESTAMP", "END_TIMESTAMP"])

    # Use per-source start times; fall back to global prehospital_start
    start_col = "prehospital_start_source" if "prehospital_start_source" in ph_pop.columns else "prehospital_start"

    # Get inhospital_start from base_df
    inhospital_start_map = base_df.set_index("PID")["inhospital_start"] if "inhospital_start" in base_df.columns else base_df.set_index("PID")["start"]

    rows = []
    for _, row in ph_pop.iterrows():
        pid = row["PID"]
        region = row["PrehospitalRegion"]
        ph_start = row.get(start_col)

        if pd.isna(ph_start):
            continue

        ih_start = inhospital_start_map.get(pid)
        if pd.isna(ih_start):
            continue

        rows.append({
            "PID": pid,
            "FEATURE": "ADT",
            "VALUE": f"PREHOSP_{region.upper().replace(' ', '_')}",
            "TIMESTAMP": pd.to_datetime(ph_start),
            "END_TIMESTAMP": pd.to_datetime(ih_start),
        })

    adt_df = pd.DataFrame(rows)

    if adt_df.empty:
        logger.warning("No prehospital ADT events generated")
        return adt_df

    # Drop events where prehospital_start >= inhospital_start (no prehospital interval)
    adt_df = adt_df[adt_df["TIMESTAMP"] < adt_df["END_TIMESTAMP"]].copy()

    logger.info(
        f"Generated {len(adt_df)} prehospital ADT events "
        f"({adt_df['VALUE'].value_counts().to_dict()})"
    )

    ensure_parent_dir("data/interim/prehospital_ADT.pkl")
    adt_df.to_pickle("data/interim/prehospital_ADT.pkl", protocol=4)
    return adt_df


# ============================================================================
# Pipeline orchestrator
# ============================================================================

def run_prehospital_pipeline(cfg, base: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run the full pre-hospital data extraction pipeline.

    1. Load PPJ mapping and raw PPJ data.
    2. Filter to study population within time window.
    3. Extract vitals, GCS, and ABCD.
    4. Compute prehospital_start/end per patient.
    5. Merge prehospital_start and ABCD into base_df.

    Returns the updated base_df with new columns:
    - prehospital_start: earliest PPJ timestamp (NaT if no PPJ data)
    - prehospital_end: latest PPJ timestamp (NaT if no PPJ data)
    - inhospital_start: hospital admission time (formerly 'start')
    - start: universal earliest timestamp = min(prehospital_start, inhospital_start)
    - A, B, C, D: ABCD categorical assessment values

    Args:
        cfg: Global config dict.
        base: Optional base_df. If None, loads from disk.
    """
    logger.info("=" * 60)
    logger.info("Starting pre-hospital data extraction pipeline")
    logger.info("=" * 60)

    if base is None:
        base = get_base_df()

    ph_cfg = cfg.get("prehospital_config", {})
    max_hours = ph_cfg.get("max_hours_before_admission", 48)

    # Step 1: Load PPJ mapping
    ppj_map = load_ppj_mapping(cfg)

    # Step 1.5: Pre-filter large raw data files (e.g. RegSJ) using matched JournalIDs
    matched_jids = ppj_map[ppj_map["CPR_hash"].isin(base["CPR_hash"])]["JournalID"].unique()
    _prefilter_raw_sources(cfg, matched_jids)

    # Step 1.6: Load PPJ data (now including pre-filtered files)
    ppj_data = load_ppj_data(cfg)

    # Step 2: Filter to study population
    ppj_filtered, ph_pop = filter_ppj_to_population(
        ppj_data, ppj_map, base, max_hours_before=max_hours
    )

    if ppj_filtered.empty:
        logger.warning("No PPJ data matched study population — skipping extraction")
        base["prehospital_start"] = pd.NaT
        base["prehospital_end"] = pd.NaT
        if "inhospital_start" not in base.columns:
            base = base.rename(columns={"start": "inhospital_start"})
        base["start"] = base["inhospital_start"].copy()
        return base

    # Step 3: Extract concepts
    extract_ppj_vitals(ppj_filtered, ph_pop)
    extract_ppj_gcs(ppj_filtered, ph_pop)
    abcd = extract_ppj_abcd(ppj_filtered, ph_pop)

    # Step 4: Compute prehospital times (per-source and global)
    ph_pop = compute_prehospital_times(ppj_filtered, ph_pop)

    # Step 4.5: Generate prehospital ADT events (uses per-source times)
    generate_prehospital_adt(base, ph_pop)

    # Step 5: Merge into base_df
    # Drop existing prehospital columns to avoid _x/_y suffixing on re-runs
    for col in ["prehospital_start", "prehospital_end"]:
        if col in base.columns:
            base = base.drop(columns=[col])

    # Deduplicate ph_pop to global per-PID level (may have multiple rows per source)
    global_ph = ph_pop[["PID", "prehospital_start", "prehospital_end"]].drop_duplicates(subset=["PID"])

    # Add prehospital_start (NaT for patients without PPJ data — intentionally nullable)
    base = base.merge(global_ph, on="PID", how="left")

    # Rename hospital admission 'start' → 'inhospital_start',
    # then create universal 'start' = earliest of prehospital and inhospital
    if "inhospital_start" not in base.columns:
        base = base.rename(columns={"start": "inhospital_start"})
    base["start"] = base["prehospital_start"].fillna(base["inhospital_start"])
    # Guard: if prehospital_start is later than inhospital_start, use the earlier
    mask = base["prehospital_start"].notna() & (
        base["inhospital_start"] < base["prehospital_start"]
    )
    base.loc[mask, "start"] = base.loc[mask, "inhospital_start"]

    # Add ABCD as tabular features
    if not abcd.empty and len(abcd.columns) > 1:
        abcd_cols = [c for c in abcd.columns if c != "PID"]
        # Drop existing ABCD columns to avoid _x/_y suffixing on re-runs
        for col in abcd_cols:
            if col in base.columns:
                base = base.drop(columns=[col])
        base = base.merge(abcd, on="PID", how="left")
        for col in abcd_cols:
            base[col] = base[col].fillna("#na#")
        logger.info(f"Added ABCD columns to base_df: {abcd_cols}")

    # Optionally filter to patients with PPJ data only
    if cfg.get("prehospital_only", False):
        n_before = len(base)
        base = base[base["prehospital_end"].notna()].copy()
        logger.info(f"prehospital_only: filtered {n_before} → {len(base)} patients")

    logger.info(
        f"Pre-hospital pipeline complete. "
        f"{base['prehospital_end'].notna().sum()}/{len(base)} patients have PPJ data"
    )

    return base


if __name__ == "__main__":
    import argparse
    import traceback
    from astra.utils import ProjectManager, setup_logging

    parser = argparse.ArgumentParser(description="Run pre-hospital (PPJ) data extraction")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    pm = ProjectManager()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Verify logging works
    logger.debug(f"Logger handlers: {logging.getLogger('astra').handlers}")
    logger.debug(f"Logger level: {logging.getLogger('astra').level}")
    logger.info("About to call run_prehospital_pipeline...")

    try:
        cfg["prehospital"] = True  # force-enable for standalone run
        logger.debug("cfg set, calling pipeline now")
        base = run_prehospital_pipeline(cfg)
        logger.info(f"Pipeline returned, base shape: {base.shape}")
        ensure_parent_dir(cfg["base_df_path"])
        base.to_pickle(cfg["base_df_path"], protocol=4)
        logger.info(f"Saved to {cfg['base_df_path']}")
    except BaseException as e:
        logger.error(f"FATAL ERROR ({type(e).__name__}): {e}")
        traceback.print_exc()
