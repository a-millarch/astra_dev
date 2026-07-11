import argparse
import logging
import os

import pandas as pd
import numpy as np

from astra.utils import ProjectManager, cfg, setup_logging, ensure_parent_dir
from astra.utils import is_file_present, are_files_present

logger = logging.getLogger(__name__)

# The Azure ML collector needs azureml/mltable; environments that receive the
# raw dump as flat CSVs in data/raw/ (e.g. the external team) don't have or
# need it.
try:
    from astra.data.collectors import collect_subsets
except ImportError:
    collect_subsets = None
import astra.data.build_patient_info as bpi
from astra.data.filters import filter_subsets_inhospital, mark_traumatext
from astra.data.mapper import map_concept, map_concept_optimized
from astra.data.caching import prepare_data_and_dls_cached


def proces_raw_concepts(cfg, base= None, reset=False): # move to construct data_sets?
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
            are_files_present("data/raw", subsets_filenames, extension=".csv")
            and reset == False
        ):
            logger.info("All subsets found, continuing")
    else:
            if collect_subsets is None:
                missing = [f for f in subsets_filenames
                           if not is_file_present(os.path.join("data/raw", f"{f}.csv"))]
                raise FileNotFoundError(
                    "Raw concept CSVs are missing from data/raw/ and the Azure ML "
                    "collector is unavailable (azureml/mltable not installed). "
                    "Place the raw data dump in data/raw/ first. "
                    f"Missing: {missing}"
                )
            logger.info("Subsets missing, collecting missing")
            collect_subsets(cfg, base=base)

def proces_inhospital_concepts(cfg, reset=False):
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
        are_files_present("data/interim/concepts", subsets_filenames, extension=".pkl")
        and reset == False
    ):
        logger.info("Interim subsets found, continuing")
    else:
        logger.info("Filtering subsets")
        filter_subsets_inhospital(cfg)
    
def map_data(cfg):
    logger.info("Mapping data to bins")
    map_dir = "data/interim/mapped/"
    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            if is_file_present(
                f"{map_dir}{concept}_{agg_func}.csv"
            ) and is_file_present(f"{map_dir}{concept}_{agg_func}.pkl"):
                pass
            else:
                logger.debug(f"Binning and mapping {concept} with agg_func: {agg_func}")
                if concept in cfg["dataset"]["ts_cat_names"]:
                    is_categorical = True
                    is_multi_label =True
                else:
                    is_categorical = False
                    is_multi_label =False                    
                map_concept(cfg, concept, agg_func, is_categorical, is_multi_label)

def map_data_optimized(cfg, overwrite=False):
    """Updated to use optimized mapper."""
    logger.info("Mapping data to bins")
    map_dir = "data/interim/mapped/"

    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            output_file = f"{map_dir}{concept}_{agg_func}.csv"

            if not overwrite and os.path.exists(output_file):
                logger.info(f"Skipping {concept}_{agg_func} (already exists)")
                continue

            logger.info(f"Processing {concept} with {agg_func}")

            is_categorical = concept in cfg["dataset"]["ts_cat_names"]
            is_multi_label = concept in cfg["dataset"]["ts_categorical_multi_label"]

            map_concept_optimized(
                cfg,
                concept,
                agg_func,
                is_categorical,
                is_multi_label
            )            


def process_prehospital(cfg, base, overwrite=False):
    """Extract pre-hospital PPJ data and merge into base_df when enabled."""
    if not cfg.get("prehospital"):
        return base

    from astra.data.prehospital import run_prehospital_pipeline

    if not overwrite and "prehospital_start" in base.columns:
        logger.info("Pre-hospital columns already present in base_df, skipping")
        return base

    logger.info("Pre-hospital pipeline enabled — extracting PPJ data")
    base = run_prehospital_pipeline(cfg, base=base)
    ensure_parent_dir(cfg["base_df_path"])
    base.to_pickle(cfg["base_df_path"], protocol=4)
    logger.info(f"Updated base_df saved at {cfg['base_df_path']}")
    return base


def _build_r_iss(base: pd.DataFrame) -> pd.DataFrame:
    """Compute ISS from ICD-10 diagnosis codes via R icdpicr and return as
    standard [PID, TIMESTAMP, FEATURE, VALUE] DataFrame.

    TIMESTAMP is the latest Noteret_dato among the diagnoses used per patient,
    i.e. the moment the ISS becomes fully known.
    """
    from astra.evaluation.trauma_scores import _prepare_long_df, compute_iss_from_r

    diag_path = "data/raw/Diagnoser.csv"
    if not os.path.exists(diag_path):
        logger.info("Diagnoser.csv not found — skipping R-computed ISS")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])

    # Ensure diagnoses_long.csv exists
    long_path = "data/interim/diagnoses_long.csv"
    if not is_file_present(long_path):
        logger.info("Creating long diagnosis DataFrame for R ISS computation...")
        _prepare_long_df(base)

    # Run R script to produce computed_iss_df.csv
    iss_csv_path = "data/interim/computed_iss_df.csv"
    if not is_file_present(iss_csv_path):
        try:
            compute_iss_from_r(base)
        except Exception as e:
            logger.warning(f"R ISS computation failed: {e}")
            return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])

    if not is_file_present(iss_csv_path):
        logger.warning("computed_iss_df.csv not produced — skipping R-computed ISS")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])

    # Load R-computed ISS values
    iss_r = pd.read_csv(iss_csv_path, low_memory=False)
    iss_r["riss"] = pd.to_numeric(iss_r.get("riss"), errors="coerce")
    iss_r["niss"] = pd.to_numeric(iss_r.get("niss"), errors="coerce")
    # Use riss preferentially, fall back to niss
    iss_r["iss_value"] = iss_r["riss"].fillna(iss_r["niss"])
    iss_r = iss_r.dropna(subset=["iss_value"])
    if iss_r.empty:
        logger.info("R ISS: no valid riss/niss values")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])

    # Compute latest diagnosis timestamp per patient from raw diagnoses
    diag = pd.read_csv(diag_path)
    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"])
    merged = base[["CPR_hash", "PID", "start", "end"]].merge(diag, on="CPR_hash", how="inner")
    merged = merged[
        (merged["Noteret_dato"] >= merged["start"] - pd.DateOffset(days=1))
        & (merged["Noteret_dato"] <= merged["end"] + pd.DateOffset(days=1))
    ]
    max_diag_date = merged.groupby("PID")["Noteret_dato"].max().reset_index()
    max_diag_date.columns = ["PID", "TIMESTAMP"]

    # Join ISS values with timestamps
    result = iss_r[["PID", "iss_value"]].merge(max_diag_date, on="PID", how="inner")
    result["FEATURE"] = "ISS_computed"
    result = result.rename(columns={"iss_value": "VALUE"})
    result = result[["PID", "TIMESTAMP", "FEATURE", "VALUE"]]

    # Filter to only patients in the cohort (base_df)
    cohort_pids = set(base["PID"].unique())
    result = result[result["PID"].isin(cohort_pids)]
    logger.info(f"R-computed ISS: {len(result)} patients with valid scores (after cohort filter)")
    return result


def _save_notater_derived_concepts(cfg, base: pd.DataFrame):
    """Extract ISS_notes/ISS_computed/Events and save as concept pickles.

    These concepts are derived from clinical notes or computed from diagnosis
    codes rather than having their own raw CSVs, so filter_subsets_inhospital()
    does not create them.  Saving them as standard concept pickles makes all
    downstream consumers (AggregatedDS, inference) work without special-casing.
    """
    notater_path = "data/interim/concepts/Notater.pkl"
    if not os.path.exists(notater_path):
        logger.warning("Notater.pkl not found, skipping derived concept extraction")
        return

    notater_df = pd.read_pickle(notater_path)
    bin_df = pd.read_pickle(cfg["bin_df_path"])

    if "ISS_notes" in cfg["concepts"]:
        from astra.data.notes_features import build_iss_from_notes

        iss_notes = build_iss_from_notes(notater_df)
        n_notes = iss_notes["PID"].nunique() if len(iss_notes) else 0
        ensure_parent_dir("data/interim/concepts/ISS_notes.pkl")
        iss_notes.to_pickle("data/interim/concepts/ISS_notes.pkl", protocol=4)
        logger.info(f"Saved ISS_notes: {len(iss_notes)} rows, {n_notes} patients")

    if "ISS_computed" in cfg["concepts"]:
        iss_r = _build_r_iss(base)
        n_r = iss_r["PID"].nunique() if len(iss_r) else 0
        ensure_parent_dir("data/interim/concepts/ISS_computed.pkl")
        iss_r.to_pickle("data/interim/concepts/ISS_computed.pkl", protocol=4)
        logger.info(f"Saved ISS_computed: {len(iss_r)} rows, {n_r} patients")

    if "Events" in cfg["concepts"]:
        from astra.data.cardiac_arrest import build_cardiac_arrest_from_notes
        from astra.data.notes_features import build_intubation_from_notes
        ca_df = build_cardiac_arrest_from_notes(notater_df)
        intub_df = build_intubation_from_notes(notater_df)
        # 24h admission cutoff for intubation
        n_before = len(intub_df)
        start_times = bin_df.groupby("PID")["bin_start"].min()
        intub_df = intub_df.merge(start_times, on="PID", how="left")
        intub_df = intub_df[intub_df["TIMESTAMP"] <= intub_df["bin_start"] + pd.Timedelta(hours=24)]
        intub_df = intub_df.drop(columns=["bin_start"])
        logger.info(f"Intubation: {n_before} → {len(intub_df)} after 24h admission cutoff")
        events_df = pd.concat([ca_df, intub_df], ignore_index=True)
        # Reformat for categorical: FEATURE=constant, VALUE=event_type
        events_df["VALUE"] = events_df["FEATURE"]  # 'cardiac_arrest' or 'INTUBATED'
        events_df["FEATURE"] = "event"
        ensure_parent_dir("data/interim/concepts/Events.pkl")
        events_df.to_pickle("data/interim/concepts/Events.pkl", protocol=4)
        logger.info(f"Saved Events concept: {len(events_df)} rows, {events_df['PID'].nunique() if len(events_df) else 0} patients")


def _forward_fill_concept(cfg: dict, concept: str) -> None:
    """Forward-fill time columns in mapped concept pickle.

    Ensures semi-static features (ISS) propagate forward from
    first observation. ffill on axis=1 is inherently forward-only.
    """
    map_dir = "data/interim/mapped/"
    for agg_func in cfg["agg_func"][concept]:
        path = f"{map_dir}{concept}_{agg_func}.pkl"
        if not os.path.exists(path):
            logger.warning(f"Forward-fill skipped: {path} not found")
            continue

        df = pd.read_pickle(path)
        time_cols = [c for c in df.columns if c not in ("PID", "FEATURE")]
        df[time_cols] = df[time_cols].ffill(axis=1)
        df.to_pickle(path, protocol=4)
        logger.info(f"Forward-filled {concept}_{agg_func}")

if __name__ =='__main__':
    pm = ProjectManager()
    setup_logging()
    
    parser = argparse.ArgumentParser(description="ASTRA data pipeline")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing intermediate files instead of skipping them",
    )
    args = parser.parse_args()
    overwrite = args.overwrite

    # Cohort mode
    if not overwrite and is_file_present(cfg['base_df_path']):
        base = pd.read_pickle(cfg['base_df_path'])
    else:
        base = bpi.create_base_df(cfg)

    proces_raw_concepts(cfg, base= base, reset=False)

    # Pre-hospital data extraction (when enabled)
    base = process_prehospital(cfg, base, overwrite=overwrite)

    # bin_df (now uses prehospital_start if available)
    if not overwrite and is_file_present(cfg['bin_df_path']):
        pass
    else:
        bpi.create_bin_df(cfg, base=base)

    # Inhospital filtering depends on base_df — must reset when overwriting
    proces_inhospital_concepts(cfg, reset=overwrite)

    # Mark trauma text keywords on base (requires Notater.pkl from step above)
    if cfg.get("traumatext_config", {}).get("enabled", False):
        base = mark_traumatext(base, cfg)
        ensure_parent_dir(cfg["base_df_path"])
        base.to_pickle(cfg["base_df_path"], protocol=4)
        logger.info(f"Updated base_df with TRAUMATEXT columns at {cfg['base_df_path']}")

    # Extract Notater-derived concepts (ISS, Events) before mapping
    _save_notater_derived_concepts(cfg, base)

    map_data_optimized(cfg, overwrite=overwrite)

    # Forward-fill ISS channels (semi-static features)
    if "ISS_notes" in cfg["concepts"]:
        _forward_fill_concept(cfg, "ISS_notes")
    if "ISS_computed" in cfg["concepts"]:
        _forward_fill_concept(cfg, "ISS_computed")
  
    data = prepare_data_and_dls_cached(cfg)