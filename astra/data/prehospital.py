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

from astra.utils import cfg, get_base_df, is_file_present
from astra.data.mappings import (
    PPJ_MONTH_DICT,
    PPJ_VITALS_MAP,
    PPJ_VITAL_EVENT_CODES,
    PPJ_ABCD_MAP,
    PPJ_VITAL_BOUNDS,
)

logger = logging.getLogger(__name__)


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
    """Load CPR_hash → JournalID mapping.

    The mapping file links hospital patient identifiers (CPR_hash) to
    pre-hospital journal identifiers (JournalID).  It also contains the
    CreationTime that is used to determine temporal overlap.

    Returns DataFrame with columns [CPR_hash, JournalID, CreationTime_dt].
    """
    ph_cfg = cfg.get("prehospital_config", {})
    mapping_path = ph_cfg.get("ppj_mapping_path", "data/raw/ppj_mapping.csv")
    logger.info(f"Loading PPJ mapping from {mapping_path}")

    ppj_map = pd.read_csv(mapping_path, sep=";")

    # Drop unnamed index columns if present
    ppj_map = ppj_map.loc[:, ~ppj_map.columns.str.startswith("Unnamed")]
    ppj_map.drop_duplicates(inplace=True)

    # Parse timestamps
    if "CreationTime" in ppj_map.columns:
        ppj_map["CreationTime_dt"] = parse_ppj_timestamps(ppj_map["CreationTime"])

    return ppj_map


def load_ppj_data(cfg) -> pd.DataFrame:
    """Load raw PPJ CSV file(s).

    Reads semicolon-delimited CSVs with columns:
    ``EventCodeName, CreationTime, ManualTime, ValueFloat, ValueString,
    ValueDateTime, ValueBool, JournalID``

    Parses timestamps, replaces empty quoted strings with NaN, and removes
    CPR identity entries (EventCodeName == 'PAT00013').
    """
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
            df = pd.read_csv(csv_file, sep=";", encoding="utf-8", low_memory=False)
            # Drop unnamed index columns
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No PPJ data files found at configured paths: {ppj_paths}"
        )

    ppj = pd.concat(dfs, ignore_index=True)
    ppj.drop_duplicates(inplace=True)

    # Parse timestamps
    for col in ["CreationTime", "ManualTime"]:
        if col in ppj.columns:
            ppj[col] = parse_ppj_timestamps(ppj[col])

    # Replace empty quoted strings with NaN
    ppj.replace('""', np.nan, inplace=True)

    # Remove CPR identity entries (privacy)
    ppj = ppj[ppj["EventCodeName"] != "PAT00013"].copy()

    logger.info(f"Loaded {len(ppj)} PPJ records across {ppj['JournalID'].nunique()} journals")
    return ppj


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

    # Merge mapping with base_df to get admission/discharge times + PID
    base_cols = ["CPR_hash", "PID", "start", "end"]
    ph = base_df[base_cols].merge(ppj_pop, on="CPR_hash", how="inner")

    # Compute hours before admission
    if "CreationTime_dt" in ph.columns:
        ph["delta_hours_start"] = (
            ph["CreationTime_dt"] - ph["start"]
        ).dt.total_seconds() / 3600

        # Filter: within max_hours_before before admission, and before discharge
        ph = ph[
            (ph["CreationTime_dt"] <= ph["end"])
            & (ph["delta_hours_start"] >= -max_hours_before)
        ].drop_duplicates()

    logger.info(f"PPJ population after time filtering: {ph['PID'].nunique()} patients")

    # Filter raw PPJ to matched JournalIDs
    valid_jids = ph["JournalID"].unique()
    ppj_filtered = ppj[ppj["JournalID"].isin(valid_jids)].copy()

    # Add PID via JournalID → CPR_hash → PID
    jid_to_pid = ph[["JournalID", "PID"]].drop_duplicates()
    ppj_filtered = ppj_filtered.merge(jid_to_pid, on="JournalID", how="left")
    ppj_filtered = ppj_filtered[ppj_filtered["PID"].notnull()].copy()

    # Build per-PID population summary (ph_pop)
    ph_pop = ph[["CPR_hash", "PID", "start", "end"]].drop_duplicates(subset=["PID"])

    return ppj_filtered, ph_pop


# ============================================================================
# Concept extraction
# ============================================================================

def extract_ppj_vitals(
    ppj_filtered: pd.DataFrame,
    ph_pop: pd.DataFrame,
) -> pd.DataFrame:
    """Extract pre-hospital vital signs from PPJ.

    Filters by vital sign event codes, maps PPJ subset names to ASTRA standard
    feature names (SBP, DBP, HR, SPO2), applies outlier bounds, and produces
    a DataFrame in the standard ASTRA format: [TIMESTAMP, PID, FEATURE, VALUE].
    """
    # Filter to vital sign event codes
    vital_codes = list(PPJ_VITAL_EVENT_CODES.keys())
    vitals = ppj_filtered[ppj_filtered["EventCodeName"].isin(vital_codes)].copy()

    if vitals.empty:
        logger.warning("No pre-hospital vital signs found in PPJ data")
        return pd.DataFrame(columns=["TIMESTAMP", "PID", "FEATURE", "VALUE"])

    # Map event codes to subset names, then to ASTRA standard names
    vitals["FEATURE"] = vitals["EventCodeName"].map(PPJ_VITAL_EVENT_CODES)
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

    if not gcs_codes:
        logger.warning("No GCS event codes identified in PPJ data")
        return pd.DataFrame(columns=["TIMESTAMP", "PID", "FEATURE", "VALUE"])

    gcs = ppj_filtered[ppj_filtered["EventCodeName"].isin(gcs_codes)].copy()

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

    gcs.to_pickle("data/interim/prehospital_GCS.pkl", protocol=4)
    return gcs


def _get_gcs_event_codes(ppj_filtered: pd.DataFrame) -> list:
    """Identify GCS event codes from PPJ data.

    Tries to load from event_descriptions_modified.xlsx if available,
    otherwise uses known GCS event code pattern.
    """
    ph_cfg = cfg.get("prehospital_config", {})
    ed_path = ph_cfg.get("event_descriptions_path")

    if ed_path and os.path.exists(ed_path):
        try:
            ed = pd.read_excel(ed_path, sheet_name="Prædefinerede eventkoder", engine="openpyxl")
            gcs_rows = ed[ed["Tekst"].str.contains("GCS", case=False, na=False)]
            if not gcs_rows.empty:
                codes = gcs_rows["Kode"].tolist()
                # Flatten lists if codes are stored as lists
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
    if "ValueFloat" in ppj_filtered.columns:
        candidates = ppj_filtered[
            ppj_filtered["ValueFloat"].between(3, 15)
        ]["EventCodeName"].value_counts()
        # Look for event codes that consistently have values 3-15
        # This is a heuristic — log for manual verification
        if not candidates.empty:
            logger.info(f"Candidate GCS event codes (by frequency): {candidates.head(5).to_dict()}")

    logger.warning(
        "Could not determine GCS event codes automatically. "
        "Set prehospital_config.gcs_event_codes in config or provide event_descriptions_path."
    )
    # Return configured codes if available
    return ph_cfg.get("gcs_event_codes", [])


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

    # Try to resolve ABCD event codes from event_descriptions
    if not abcd_codes and ed_path and os.path.exists(ed_path):
        abcd_codes = _get_abcd_event_codes(ed_path)

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
        if subset.empty:
            continue

        # Use ValueString for categorical values
        val_col = "ValueString" if "ValueString" in subset.columns else "ValueFloat"
        subset["value"] = subset[val_col].astype(str).str.replace('"', '')
        subset = subset[subset["value"].notna() & (subset["value"] != "nan")]

        # Take latest observation per PID
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

    abcd.to_pickle("data/interim/ppj_base_df.pkl", protocol=4)
    return abcd


def _get_abcd_event_codes(ed_path: str) -> dict:
    """Load ABCD event codes from event_descriptions_modified.xlsx."""
    try:
        ed_pre = pd.read_excel(
            ed_path, sheet_name="Prædefinerede eventkoder", engine="openpyxl"
        )
        codes = {}
        for ppj_name in PPJ_ABCD_MAP:
            matches = ed_pre[ed_pre["Tekst"] == ppj_name]
            if not matches.empty:
                code_val = matches["Kode"].iloc[0]
                codes[ppj_name] = [code_val] if not isinstance(code_val, list) else code_val
        if codes:
            logger.info(f"ABCD event codes from event_descriptions: {codes}")
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

    prehospital_start = earliest PPJ timestamp for the patient.
    prehospital_end = latest PPJ timestamp for the patient.

    These are merged into ph_pop and returned.
    """
    ts_col = "CreationTime"
    if ts_col not in ppj_filtered.columns:
        ts_col = "TIMESTAMP"

    times = ppj_filtered.groupby("PID").agg(
        prehospital_start=(ts_col, "min"),
        prehospital_end=(ts_col, "max"),
    ).reset_index()

    for col in ["prehospital_start", "prehospital_end"]:
        times[col] = pd.to_datetime(times[col])

    ph_pop = ph_pop.merge(times, on="PID", how="left")

    n_with_ph = ph_pop["prehospital_start"].notna().sum()
    logger.info(f"Prehospital times computed: {n_with_ph}/{len(ph_pop)} patients have PPJ data")

    return ph_pop


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
    - prehospital_start: earliest PPJ timestamp (or hospital start if no PPJ)
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

    # Step 1: Load PPJ mapping and data
    ppj_map = load_ppj_mapping(cfg)
    ppj_data = load_ppj_data(cfg)

    # Step 2: Filter to study population
    ppj_filtered, ph_pop = filter_ppj_to_population(
        ppj_data, ppj_map, base, max_hours_before=max_hours
    )

    if ppj_filtered.empty:
        logger.warning("No PPJ data matched study population — skipping extraction")
        base["prehospital_start"] = base["start"]
        return base

    # Step 3: Extract concepts
    extract_ppj_vitals(ppj_filtered, ph_pop)
    extract_ppj_gcs(ppj_filtered, ph_pop)
    abcd = extract_ppj_abcd(ppj_filtered, ph_pop)

    # Step 4: Compute prehospital times
    ph_pop = compute_prehospital_times(ppj_filtered, ph_pop)

    # Step 5: Merge into base_df
    # Add prehospital_start — fall back to hospital start for patients without PPJ
    base = base.merge(
        ph_pop[["PID", "prehospital_start", "prehospital_end"]],
        on="PID",
        how="left",
    )
    base.loc[base["prehospital_start"].isna(), "prehospital_start"] = base.loc[
        base["prehospital_start"].isna(), "start"
    ]

    # Add ABCD as tabular features
    if not abcd.empty and len(abcd.columns) > 1:
        abcd_cols = [c for c in abcd.columns if c != "PID"]
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
    from astra.utils import ProjectManager, setup_logging

    parser = argparse.ArgumentParser(description="Run pre-hospital (PPJ) data extraction")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    pm = ProjectManager()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    cfg["prehospital"] = True  # force-enable for standalone run
    base = run_prehospital_pipeline(cfg)
    base.to_pickle(cfg["base_df_path"], protocol=4)
    logger.info(f"Updated base_df saved at {cfg['base_df_path']}")
