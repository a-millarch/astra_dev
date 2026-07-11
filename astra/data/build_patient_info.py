import logging

import pandas as pd
import numpy as np
import subprocess

from astra.utils import cfg, get_base_df, create_enumerated_id, is_file_present, ensure_parent_dir
from astra.utils import ensure_datetime,count_csv_rows, inches_to_cm, ounces_to_kg
try:
    from astra.data.collectors import collect_procedures, population_filter_parquet
except ImportError:
    collect_procedures = None
    population_filter_parquet = None
from astra.data.mapper import map_concept
from astra.data.mappings import (
    standardize_hospital as _standardize_hospital,
    first_hospital as _first_hospital,
)

from typing import List, Dict, Optional, Union
from datetime import timedelta

try:
    from azureml.core import Dataset
except ImportError:
    Dataset = None

logger = logging.getLogger(__name__)

def create_base_df(cfg, result_path=None):
    if result_path is None:
        result_path = cfg["base_df_path"]
    logger.info("Creating base dataframe")

    population = load_or_collect_population(cfg)
    df_ad = load_or_collect_adt(population)
    of = build_trajectories(df_ad)

    population = ensure_datetime(population, "ServiceDate")
    matched = match_population_to_trajectories(of, population)
    
    merged_df = add_first_contacts(matched, df_ad)
    merged_df =  add_first_hospital(merged_df)
    
    result = add_patient_info(merged_df, population)
    result = add_patient_id(result)
    result = mask_mortality(result)
    result = final_cleanup(result)

    # Add statics
    result = add_to_base(result)
    # add Elixhauser
    result = add_elixhauser(result) 
    
    logger.info(f"Saving file at{result_path}")
    ensure_parent_dir(result_path)
    result.to_pickle(result_path, protocol=4)
    return result
    
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

def define_historic_population(cfg=cfg):
    """Derive the trauma cohort seed (CPR_hash, ServiceDate): every patient
    with a BWST1F trauma-call procedure.

    Reads the Azure ML parquet dataset when azureml is available; otherwise
    falls back to the raw ``data/raw/Procedurer.csv`` dump so the cohort can
    be (re)defined in non-Azure environments.
    """
    if Dataset is not None:
        path = f'{cfg["raw_file_path"]}CPMI_Procedurer.parquet'
        df_procedure = Dataset.Tabular.from_parquet_files(path=path)
        dtr_procedure = df_procedure.to_pandas_dataframe()
    else:
        csv_path = "data/raw/Procedurer.csv"
        if not is_file_present(csv_path):
            raise FileNotFoundError(
                f"Population seed {cfg['population_file_path']} is missing and "
                f"cannot be derived: azureml is not installed and {csv_path} "
                "does not exist. Either provide the seed file (columns: "
                "CPR_hash, ServiceDate — one row per trauma call) or place "
                "the raw Procedurer.csv dump in data/raw/."
            )
        logger.info("Deriving population from %s (ProcedureCode == 'BWST1F')",
                    csv_path)
        dtr_procedure = pd.read_csv(csv_path, low_memory=False)
        if ("ServiceDate" not in dtr_procedure.columns
                and "ServiceDatetime" in dtr_procedure.columns):
            dtr_procedure = dtr_procedure.rename(
                columns={"ServiceDatetime": "ServiceDate"})

    traumepatienter = dtr_procedure[dtr_procedure["ProcedureCode"] == "BWST1F"][
        ["CPR_hash", "ServiceDate"]
    ]
    ensure_parent_dir(cfg["population_file_path"])
    traumepatienter.to_csv(cfg["population_file_path"])
    logger.info("Wrote population seed: %d trauma calls -> %s",
                len(traumepatienter), cfg["population_file_path"])

def define_single_patient(cfg):
    """Write a placeholder single-patient seed file (used by
    load_or_collect_population in single_patient_mode). Replace the
    placeholder hash/date with the target patient before running."""
    ensure_parent_dir(cfg["trauma_call_file_path"])
    pd.DataFrame.from_dict(
        {'CPR_hash': ['0' * 64],  # placeholder — replace with a real CPR hash
         'ServiceDate': [np.datetime64('2030-01-01T12:00:00.000000000')]},
        orient='columns',
    ).to_csv(cfg["trauma_call_file_path"])
    logger.warning(
        "Wrote PLACEHOLDER single-patient seed to %s — edit it with the "
        "target patient's CPR_hash and ServiceDate.",
        cfg["trauma_call_file_path"],
    )

def load_or_collect_population(cfg):
    if cfg["single_patient_mode"] is True:
        logger.debug("Single patient mode")
        while True:
            try:
                logger.debug("Read patient seed file")
                return pd.read_csv(cfg["trauma_call_file_path"], index_col=0)
            except FileNotFoundError:
                logger.warning("Patient seed file not found!")
                define_single_patient(cfg)
    else:
        logger.debug("Population mode")
        while True:
            try:
                logger.debug("Read population seed file")
                return pd.read_csv(cfg["population_file_path"], index_col=0)
            except FileNotFoundError:
                logger.warning("Population seed file not found!")
                define_historic_population(cfg)        

def load_or_collect_adt( population):
    path = "data/raw/ADTHaendelser.csv"
    while True:
        try:
            logger.debug("Loading ADT")
            df_ad = pd.read_csv(path, dtype={"CPR_hash": str}, index_col=0)
            break
        except FileNotFoundError:
            logger.warning("ADT file not found. Loading.")
            population_filter_parquet("ADTHaendelser", base=population)

    df_ad[["Flyt_ind", "Flyt_ud"]] = df_ad[["Flyt_ind", "Flyt_ud"]].apply(
        pd.to_datetime, format="mixed", errors="coerce"
    )
    df_ad.loc[df_ad.ADT_haendelse == "Flyt Ind", "Flyt_ind"] += pd.Timedelta(seconds=1)

    return df_ad.sort_values(["CPR_hash", "Flyt_ind"]).reset_index(drop=True)


def build_trajectories(df_ad):
    """
    Builds patient trajectories from ADT admission events by assigning trajectory numbers and
    collapsing consecutive admissions per patient using a time gap threshold (default: 1 hour).
    Parameters:
    - df_ad (pd.DataFrame): ADT events with Flyt_ind, Flyt_ud, and CPR_hash.
 
    Returns:
    - pd.DataFrame: Collapsed trajectories with start, end, duration, and combined trajectory IDs.
    """
    logger.info(">Building trajectories")
 
    # Tildel unik trajectory ID til hver 'Indlæggelse' hændelse
    df_ad["trajectory"] = (
        df_ad[df_ad["ADT_haendelse"] == "Indlæggelse"]
        .groupby("CPR_hash")
        .cumcount() + 1
    )
    df_ad["trajectory"] = df_ad["trajectory"].ffill()
 
    # Filtrer nødvendige kolonner og sorter
    df_ad = df_ad[["CPR_hash", "trajectory", "Flyt_ind", "Flyt_ud"]].copy()
    df_ad = df_ad.rename(columns={"Flyt_ind": "start", "Flyt_ud": "end"})
 
    # Kør den optimerede collapse-admissions
    of = collapse_admissions(df_ad, time_gap_hours=1)
 
    return of


def match_population_to_trajectories(of, population):
    logger.info("Matching population to trajectories")
    fdf = find_forløb(of, population, "ServiceDate")
    df = pd.merge(
        fdf[["CPR_hash", "trajectory", "ServiceDate"]],
        of,
        on=["CPR_hash", "trajectory"],
        how="left"
    )
    return df


def add_first_contacts(df, df_adt):
    """
    Matches department admissions to trajectories and classifies visitation type. Finds the first department contact and first RH contact within each trajectory. Calculates time to first RH contact 
    and assigns visitation type: 'primær', 'sekundær', or 'primær ingen RH'.
    """ 
    logger.info("Adding first contacts and visitation type")
    merged = df_adt.merge(df[["CPR_hash", "ServiceDate", "start", "end"]], on="CPR_hash")
    filtered = merged[(merged["Flyt_ind"] >= merged["start"]) & (merged["Flyt_ind"] <= merged["end"])]

    first_afsnit = filtered.groupby(["CPR_hash", "ServiceDate", "start"]).first().reset_index()

    first_RH = filtered[
        filtered["Afsnit"].str.contains("RH ", case=False, na=False)
    ].groupby(["CPR_hash", "ServiceDate", "start"]).first().reset_index()

    first_RH = first_RH[["CPR_hash", "Flyt_ind", "ServiceDate", "start"]].rename(columns={"Flyt_ind": "first_RH"})

    result = pd.merge(first_afsnit, first_RH, on=["CPR_hash", "ServiceDate", "start"], how="left")
    result = result.rename(columns={"Afsnit": "first_afsnit"})

    result["time_to_RH"] = result["first_RH"] - result["start"]

    # Visitationstype
    result["type_visitation"] = "primær ingen RH" #default
    result.loc[result["first_afsnit"].str.contains("RH TRAUMECENTER", na=False), "type_visitation"] = "primær"
    result.loc[
        (~result["first_afsnit"].str.contains("RH TRAUMECENTER", na=False)) & result["first_RH"].notnull(),
        "type_visitation"
    ] = "sekundær"

    return result



# Delegated to astra.data.mappings (single source of truth)
standardize_hospital = _standardize_hospital
first_hospital = _first_hospital

def add_first_hospital(df):
    # First, remove commas from FIRST_HOSPITAL (or source column before extraction)
    df['first_afsnit'] = df['first_afsnit'].str.replace(',', '', regex=False)
    df['FIRST_HOSPITAL'] = df['first_afsnit'].apply(first_hospital)
    df['FIRST_HOSPITAL'] = df['FIRST_HOSPITAL'].apply(standardize_hospital)
    return df


def add_patient_info(df, population):
    logger.info("Adding patient info file")
    population_filter_parquet("PatientInfo", base=population)
    pi = pd.read_csv("data/raw/PatientInfo.csv", index_col=0)
    pi = pi.rename(columns={"Fødselsdato": "DOB", "Dødsdato": "DOD", "Køn": "SEX"})
    pi["SEX"] = pi["SEX"].replace({"Mand": "Male", "Kvinde": "Female"})

    df = df.merge(pi[["CPR_hash", "DOB", "DOD", "SEX"]], on="CPR_hash", how="left")
    df["overlap"] = df.groupby("CPR_hash", group_keys=False).apply(check_overlaps).explode().values

    return df


def add_patient_id(df):
    logger.info("Creating PID")
    return create_enumerated_id(df, "CPR_hash", "ServiceDate")


def final_cleanup(df):
    logger.info("Base dataframe final cleanup")
    df = df[df["start"].notnull() & df["end"].notnull()]
    df = df.drop(columns=["Flyt_ind", "Flyt_ud", "ADT_haendelse"], errors='ignore')
    df = df.drop_duplicates(subset="PID").reset_index(drop=True)
    df = df.drop_duplicates(subset=["CPR_hash", "start", "end"]).reset_index(drop=True)

    return df


def collapse_admissions(df: pd.DataFrame, time_gap_hours: int = 1) -> pd.DataFrame:
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["CPR_hash", "start"]).reset_index(drop=True)

    # include_groups only exists in pandas >= 2.2; older versions forward
    # unknown kwargs to the lambda and raise TypeError.
    try:
        collapsed = df.groupby("CPR_hash").apply(
            lambda group: _collapse_patient_admissions(group, time_gap_hours),
            include_groups=True,
        ).reset_index(drop=True)
    except TypeError:
        collapsed = df.groupby("CPR_hash").apply(
            lambda group: _collapse_patient_admissions(group, time_gap_hours)
        ).reset_index(drop=True)

    return collapsed


def _collapse_patient_admissions(group: pd.DataFrame, time_gap_hours: int) -> pd.DataFrame:
    group = group.sort_values("start").copy()
    group["prev_end"] = group["end"].shift()
    group["gap"] = group["start"] - group["prev_end"]

    gap_thresh = pd.Timedelta(hours=time_gap_hours)
    group["group_id"] = (group["gap"] >= gap_thresh).cumsum()

    collapsed = (
        group.groupby("group_id")
        .agg({
            "CPR_hash": "first",
            "start": "min",
            "end": "max",
            "trajectory": lambda x: ",".join(map(str, x)),
        })
        .reset_index(drop=True)
    )
    collapsed["duration"] = collapsed["end"] - collapsed["start"]
    return collapsed


def find_forløb(
    base: pd.DataFrame, df: pd.DataFrame, dt_name: str, offset=1
) -> pd.DataFrame:
    """
    Matcher observations in df to trajectories in base based on date overlap with optional offset. Filters the observations so that only rows where the date (dt_name) falls within the trajectory's 
start and end dates (extended by the specified offset in both directions) are retained.
    """
    
    # save colnames for return
    colnames = df.columns.to_list()
    # ensure datetime format for input df
    df = ensure_datetime(df, dt_name)
    # merge and filter
    merged_df = base.merge(df, on="CPR_hash", how="left")

    filtered_df = merged_df[
        (merged_df[dt_name] >= merged_df["start"] - pd.DateOffset(days=offset))
        & (merged_df[dt_name] <= merged_df["end"] + pd.DateOffset(days=offset))
    ]
    filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    return filtered_df[colnames + ["trajectory"]]


def check_overlaps(group):
    """Checks for overlapping in start - end of trajectories for same patient in groupby object

    usage example:
    df['overlap'] = df.groupby('CPR_hash').apply(check_overlaps).explode().reset_index(drop=True)

    """
    overlaps = []
    for i in range(len(group) - 1):
        # Check if the current end_time overlaps with the next start_time
        if group.iloc[i]["end"] > group.iloc[i + 1]["start"]:
            overlaps.append(True)
        else:
            overlaps.append(False)
    # Append False for the last entry as it has no next entry to compare
    overlaps.append(False)
    return overlaps

def create_bin_df(cfg, base=None):
    """
    Generate time bins for each patient trajectory based on configurable binning intervals.

    For each patient (PID) in the base dataset, the function iterates over the trajectory start and end times,
    and divides the trajectory into time intervals ("bins") according to rules defined in cfg["bin_intervals"].
    These bins can have varying frequencies depending on the duration of the trajectory.

    After the prehospital pipeline, ``start`` is the universal earliest timestamp
    (= min(prehospital_start, inhospital_start)), so bins always begin from ``start``.
    """
    logger.info("Generating bin_df")
    bin_list = []
    if base is None:
        base = get_base_df()

    # Load bin intervals from cfg
    bin_intervals = cfg["bin_intervals"]

    # Margin: use the largest bin frequency so the last bin is always created,
    # even for coarse intervals (e.g. 1D, 7D). The old +10min margin failed
    # for bin frequencies larger than 10min.
    freq_values = [f for f in bin_intervals.values() if f != 'end']
    max_freq = max(pd.Timedelta(f) for f in freq_values) if freq_values else pd.Timedelta(minutes=10)

    # 'start' is the universal earliest timestamp (incorporates prehospital when available)
    start_col = "start"

    for _, row in base.iterrows():
        start_time = row[start_col]
        # Safety fallback if start is unexpectedly NaT
        if pd.isna(start_time):
            start_time = row.get("inhospital_start", row.get("start"))
        end_time = row["end"] + max_freq
        pid = row["PID"]

        current_time = start_time
        bin_counter = 1

        for interval, freq in bin_intervals.items():
            if current_time >= end_time:
                break

            # Determine the end time for this interval
            if interval == "end":
                interval_end = end_time
            else:
                interval_end = start_time + pd.Timedelta(interval)

            # Create bins for this interval
            bins = pd.date_range(
                start=current_time,
                end=min(interval_end, end_time),
                freq=freq,
            )

            # Add bins to the list
            bin_list.extend(
                [
                    (pid, bin_start, bin_end, bin_counter + i, freq)
                    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:]))
                ]
            )

            # Update the current time and bin counters
            current_time = bins[-1]
            bin_counter += len(bins) - 1

    # Create DataFrame from bin list
    bin_df = pd.DataFrame(
        bin_list, columns=["PID", "bin_start", "bin_end", "bin_counter", "bin_freq"]
    )

    # Save DataFrame to pickle file
    ensure_parent_dir(cfg["bin_df_path"])
    bin_df.to_pickle(cfg["bin_df_path"], protocol=4)
    logger.info(f'>> Saved at {cfg["bin_df_path"]}')

    return bin_df


def create_bin_df_with_mortality_masking(cfg, base):
    """
    Create bin_df with option to drop last bin for deceased patients.
    
    This is an alternative to adjusting end times.
    """
    logger.info("Generating bin_df with mortality masking")
    
    bin_list = []
    bin_intervals = cfg["bin_intervals"]
    
    drop_last_bin = base.get('drop_last_bin', pd.Series([False] * len(base)))
    
    for idx, row in base.iterrows():
        pid = row["PID"]
        start_time = row["start"]
        end_time = row["end"] + pd.Timedelta(minutes=10)
        should_drop_last = drop_last_bin.iloc[idx] if idx < len(drop_last_bin) else False
        
        # Validate
        if pd.isna(start_time) or pd.isna(end_time):
            logger.warning(f"Patient {pid} has NULL timestamps, skipping")
            continue
        
        if end_time <= start_time:
            logger.warning(f"Patient {pid} has invalid trajectory (end <= start), skipping")
            continue
        
        # Create bins
        current_time = start_time
        bin_counter = 1
        patient_bins = []
        
        for interval, freq in bin_intervals.items():
            if current_time >= end_time:
                break
            
            if interval == "end":
                interval_end = end_time
            else:
                interval_end = start_time + pd.Timedelta(interval)
            
            bins = pd.date_range(
                start=current_time,
                end=min(interval_end, end_time),
                freq=freq,
            )

            if len(bins) < 2:
                continue
            
            # Create bin tuples
            for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                patient_bins.append((pid, bin_start, bin_end, bin_counter + i, freq))
            
            current_time = bins[-1]
            bin_counter += len(bins) - 1
        
        # Drop last bin if patient died
        if should_drop_last and len(patient_bins) > 1:
            logger.debug(f"Dropping last bin for deceased patient {pid}")
            patient_bins = patient_bins[:-1]
        
        bin_list.extend(patient_bins)
    
    # Create DataFrame
    bin_df = pd.DataFrame(
        bin_list, columns=["PID", "bin_start", "bin_end", "bin_counter", "bin_freq"]
    )
    
    # Validation
    base_pids = set(base['PID'].unique())
    bin_pids = set(bin_df['PID'].unique())
    missing = base_pids - bin_pids
    
    logger.info(f"Created bins for {len(bin_pids)}/{len(base_pids)} patients")
    
    if len(missing) > 0:
        logger.error(f"⚠️  {len(missing)} patients missing from bin_df!")
    
    ensure_parent_dir(cfg["bin_df_path"])
    bin_df.to_pickle(cfg["bin_df_path"])
    logger.info(f'Saved to {cfg["bin_df_path"]}')
    
    return bin_df


def add_survival_labels(base):
    """Add discrete-time survival labels to base_df.

    Creates columns:
        event_time_hours: continuous time-to-event (or censoring) in hours
        event_time_steps: discrete timestep index of event/censoring
        event_indicator: 1 if death observed within observation window, 0 if censored
    """
    from astra.evaluation.utils import time_to_step, step_to_time, get_total_steps

    max_steps = get_total_steps()
    max_time_min = step_to_time(max_steps - 1)
    max_hours = max_time_min / 60 if max_time_min is not None else float("inf")

    start = pd.to_datetime(base["start"])
    dod = pd.to_datetime(base["DOD"])
    end = pd.to_datetime(base["end"])

    # Time from admission to death (hours), NaN if no DOD
    time_to_death_hours = (dod - start).dt.total_seconds() / 3600

    # Time from admission to end of observation (hours) — already mortality-masked
    time_to_end_hours = (end - start).dt.total_seconds() / 3600

    # Event indicator: died AND death within observation window
    has_death = base["DOD"].notnull()
    within_window = time_to_death_hours <= max_hours
    base["event_indicator"] = (has_death & within_window).astype(int)

    # Event time: death time for events, end-of-observation for censored
    base["event_time_hours"] = np.where(
        base["event_indicator"] == 1,
        time_to_death_hours,
        time_to_end_hours.clip(upper=max_hours),
    )
    # Ensure non-negative
    base["event_time_hours"] = base["event_time_hours"].clip(lower=0)

    # Convert to discrete timestep index
    base["event_time_steps"] = base["event_time_hours"].apply(
        lambda h: time_to_step(h, time_unit="h") if pd.notnull(h) else 0
    )
    # Clamp to valid range
    base["event_time_steps"] = base["event_time_steps"].clip(upper=max_steps - 1).astype(int)

    n_events = base["event_indicator"].sum()
    n_censored = len(base) - n_events
    logger.info(
        f"Survival labels: {n_events} events, {n_censored} censored "
        f"(max observation: {max_hours:.0f}h / {max_steps} steps)"
    )

    return base


############
def add_to_base(base):

    base["DURATION"] = (base.end - base.start) / np.timedelta64(1, "D")

    base["AGE"] = (
        np.floor(
            (pd.to_datetime(base["start"]) - pd.to_datetime(base.DOB)).dt.days / 365.25
        )
    ).astype(int)

    base = add_height_weight(base)
    # Mortality
    base.loc[
        (pd.to_datetime(base.DOD) - pd.to_datetime(base.start))
        <= pd.Timedelta(days=30),
        "deceased_30d",
    ] = 1
    base["deceased_30d"] = base["deceased_30d"].fillna(0)

    base.loc[
        (pd.to_datetime(base.DOD) - pd.to_datetime(base.start))
        <= pd.Timedelta(days=90),
        "deceased_90d",
    ] = 1
    base["deceased_90d"] = base["deceased_90d"].fillna(0)

    # Survival labels (time-to-event)
    base = add_survival_labels(base)

    # If trauma bay RH
    base["LVL1TC"] = 0
    base.loc[base.first_RH.notnull(), "LVL1TC"] = 1

    return base


def prepare_long_df(base):
    diag = pd.read_csv("data/raw/Diagnoser.csv")

    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"])

    merged_df = base[["CPR_hash", "PID", "AGE", "start", "end"]].merge(
        diag, on="CPR_hash", how="left"
    )

    # Filtering rows where Noteret_dato is between start and end
    filtered_df = merged_df[
        (merged_df["Noteret_dato"] >= merged_df["start"] - pd.DateOffset(days=1))
        & (merged_df["Noteret_dato"] <= merged_df["end"] + pd.DateOffset(days=1))
    ]

    # Adjust Diagnosekode by removing the first and last character for ICD10 conversion
    filtered_df["Diagnosekode"] = filtered_df["Diagnosekode"].str.slice(1, -1)

    # Now, checking how many unique combinations are there
    logger.info(
        f"Unique CPR_hash-ServiceDate combinations in df1:{base.groupby('PID').ngroups}"
    )
    logger.info(
        f"Result after merging and filtering: {filtered_df.groupby('PID').ngroups}"
    )

    # Group by CPR_hash and apply a function to create new columns for each Diagnosekode
    def enumerate_diagnoses(group):
        diagnoses = group["Diagnosekode"].tolist()
        for i, diag in enumerate(diagnoses, start=1):
            group[f"ICD10_{i}"] = diag
        return group

    # Applying the function
    result_df = filtered_df.groupby("PID").apply(enumerate_diagnoses)

    # Dropping duplicates if necessary (since each row is expanded per group)
    result_df = result_df.drop_duplicates(subset="PID").reset_index(drop=True)

    ensure_parent_dir("data/interim/ISS_ELIX/diagnoses_long.csv")
    result_df.to_csv("data/interim/ISS_ELIX/diagnoses_long.csv")




def add_iss(base):
    """Add ISS and Elixhauser by R"""
    #output_df["TRISS"] = np.nan
    
    # Create long df if not there
    if is_file_present("data/interim/diagnoses_long.csv"):
        logger.info("Long diagnose df dataframe found, continuing")
    else:
        logger.info("No long diagnose file, creating.")
        prepare_long_df(base)
    logger.info("Calling R script to create ISS df at data/interim/ISS_ELIX/iss_df.csv")
    subprocess.call("Rscript src/R/iss.r", shell=True)
    logger.info("R subprocess finished")




def prepare_height_weight(base):
    from astra.data.collectors import population_filter_parquet
    
    if not is_file_present("data/raw/VitaleVaerdier.csv"):
        logger.info("No local vitale vaerdier file, creating.")
        population_filter_parquet("VitaleVaerdier", base=base)  

    vit_raw = pd.read_csv("data/raw/VitaleVaerdier.csv") 
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
    hw = vit_raw[(vit_raw.FEATURE.isin(list(set(hw_map.values()))))]
    assert len(hw)>0
    hw = hw.merge(base[["PID", "CPR_hash", "start", "end"]], on="CPR_hash", how="left")
    hw["TIMESTAMP"] = pd.to_datetime(hw.TIMESTAMP)
    hw = hw[hw.TIMESTAMP <= hw.end]
    hw = hw.sort_values(["CPR_hash", "TIMESTAMP"], ascending=False).drop_duplicates(
        subset=["CPR_hash", "FEATURE"], keep="first"
    )
    #hw = hw[hw.delta.dt.days < 365 * 2]
    return hw[["TIMESTAMP", "PID", "FEATURE", "VALUE"]]


def add_height_weight(base):

    hw = prepare_height_weight(base)
    hw_df = hw.sort_values("TIMESTAMP").drop_duplicates(
        subset=["PID", "FEATURE"], keep="first"
    )
    pivot_df = hw_df.pivot(
        index=["PID"], columns="FEATURE", values="VALUE"
    ).reset_index()
    base = base.merge(pivot_df, how="left", on="PID")

    return base


def prepare_elix_df(base):
    diag = pd.read_csv("data/raw/Diagnoser.csv")
    assert len(diag) >0
    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"])
    diag["Løst_dato"] = pd.to_datetime(diag["Løst_dato"])

    merged_df = base[["CPR_hash", "PID", "AGE", "start", "end"]].merge(
        diag, on="CPR_hash", how="left"
    )
    
    logger.info("Preparing Elixhauser Df")
    # Where noted date is before trauma AND not solved before trauma.
    e_df = merged_df[
        (merged_df["Noteret_dato"] <= merged_df["start"] - pd.DateOffset(days=1))
        &     ((merged_df["Løst_dato"].isnull() |
              ( merged_df["Løst_dato"].notnull() & 
               (merged_df["Løst_dato"] >= merged_df["start"] + pd.DateOffset(days=1)))
             )
        )
    ]

    # Adjust Diagnosekode by removing the first and last character for ICD10 conversion
    e_df["Diagnosekode"] = e_df["Diagnosekode"].str.slice(1, -1)

    # Now, checking how many unique combinations are there
    logger.info(
        f"Unique CPR_hash-ServiceDate combinations in df1: {base.groupby('PID').ngroups}"
    )

    logger.info(f"Result after merging and filtering: {e_df.groupby('PID').ngroups}")
    ensure_parent_dir("data/interim/pre_elix_df.csv")
    e_df[["PID", "AGE", "Diagnosekode"]].to_csv("data/interim/pre_elix_df.csv")



def create_elixhauser(base):
    """Compute cohort Elixhauser scores → ``data/interim/computed_elix_df.csv``.

    Uses the pure-Python implementation shared with inference
    (``astra/inference/comorbidity.py`` — Quan ICD-10 mapping, van Walraven
    weights), so training and inference score comorbidity identically by
    construction. Set ``ASTRA_ELIX_USE_R=1`` to fall back to the original R
    ``comorbidity`` package (``astra/R/elixhauser.r``).
    """
    import os
    if os.environ.get("ASTRA_ELIX_USE_R"):
        return create_elixhauser_r(base)
    return create_elixhauser_python(base)


def create_elixhauser_python(base):
    """Cohort version of the inference Elixhauser scorer (single source of truth)."""
    from astra.inference.comorbidity import compute_elixhauser_vw

    pre_elix_path = "data/interim/pre_elix_df.csv"
    if is_file_present(pre_elix_path):
        logger.info("Elixhauser diagnose df dataframe found, continuing")
    else:
        logger.info("No Elixhauser diagnose file, creating.")
        prepare_elix_df(base)

    pre = pd.read_csv(pre_elix_path, index_col=0)
    pre = pre.dropna(subset=["Diagnosekode"])
    if len(pre) == 0:
        logger.info(">No prior diagnoses, elixscore is null")
        output_df = base[["CPR_hash", "PID"]].copy(deep=True)
        output_df["elixscore"] = np.nan
    else:
        logger.info(
            "Computing Elixhauser (van Walraven, Python) for %d patients",
            pre["PID"].nunique(),
        )
        scores = (
            pre.groupby("PID")["Diagnosekode"]
            .apply(lambda codes: compute_elixhauser_vw(list(codes.astype(str))))
        )
        output_df = scores.rename("elixscore").reset_index()

    ensure_parent_dir("data/interim/computed_elix_df.csv")
    output_df.to_csv("data/interim/computed_elix_df.csv")
    logger.info("Saved computed_elix_df.csv (%d rows, Python implementation)",
                len(output_df))


def create_elixhauser_r(base):
    """Original R path (``comorbidity`` package) — kept as an escape hatch."""
    pre_elix_path = "data/interim/pre_elix_df.csv"
    if is_file_present(pre_elix_path):
        logger.info("Elixhauser diagnose df dataframe found, continuing")

    else:
        logger.info("No Elixhauser diagnose file, creating.")
        prepare_elix_df(base)

    if count_csv_rows(pre_elix_path) > 0:
        logger.info(">Calling R script to create Elixhauser df at data/interim/")
        subprocess.call("Rscript astra/R/elixhauser.r", shell=True)
        logger.info("R subprocess finished")
    else:
        logger.info(">No prior diagnoses, elixscore is null")
        output_df=base[['CPR_hash',"PID"]].copy(deep=True) #also CPR_hash?
        output_df["elixscore"] = np.nan

        ensure_parent_dir("data/interim/computed_elix_df.csv")
        output_df.to_csv("data/interim/computed_elix_df.csv")

def add_elixhauser(base, cols_to_add=["ASMT_ELIX", ]):
    """ Check if elix_df is present, if not then compute it with prepare elix_df (requires data/raw/Diagnoser.csv)

    If the score computation cannot produce the file (missing
    data/raw/Diagnoser.csv; or, on the R escape-hatch path, no R runtime /
    CRAN access — subprocess.call does not raise when Rscript is missing),
    ASMT_ELIX degrades to 0.0 for all patients instead of retrying forever.
    """
    for attempt in (1, 2):
        try:
            elix = pd.read_csv(
                "data/interim/computed_elix_df.csv", low_memory=False
            )
        except FileNotFoundError:
            if attempt == 1:
                logger.info("DF missing.")
                try:
                    create_elixhauser(base)
                except Exception:
                    logger.warning(
                        "Elixhauser computation failed (missing "
                        "data/raw/Diagnoser.csv?)", exc_info=True,
                    )
                continue
            logger.warning(
                "computed_elix_df.csv could not be produced — filling "
                "ASMT_ELIX with 0.0 for all patients. Comorbidity signal "
                "will be absent."
            )
            base["ASMT_ELIX"] = 0.0
            return base

        logger.info("Elixhauser df dataframe found, continuing")
        baselen = len(base)
        # merge
        elix=elix.rename(columns={'elixscore':'ASMT_ELIX'})
        base = base.merge(
            elix[["PID", ]+cols_to_add], how="left", on="PID"
        )
        assert baselen - len(base) == 0
        n_missing = int(base["ASMT_ELIX"].isna().sum())
        base["ASMT_ELIX"] = base["ASMT_ELIX"].fillna(0.0)
        logger.info(f"Merged Elix onto base (filled {n_missing} missing ASMT_ELIX with 0.0)")
        return base
        
def mask_mortality(df, method='percentage', min_duration_hours=0.5):
    logger.info(f"Masking mortality using method: {method}")
    
    # Convert to datetime
    for col in ["start", "end", "DOD"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Get patients with DOD
    dod_mask = df["DOD"].notnull()
    n_deaths = dod_mask.sum()
    
    if not dod_mask.any():
        logger.info("No patients with DOD, skipping masking")
        return df
    
    logger.info(f"Masking {n_deaths} patients with DOD")
    
    # Calculate duration
    duration = df["end"] - df["start"]
    duration_hours = duration.dt.total_seconds() / 3600
    
    # Store original for comparison
    original_end = df["end"].copy()
    
    if method == 'percentage':
        # Mask by percentage of trajectory duration
        # More robust: avoids end <= start issues
        
        # Short trajectories (<6h): mask last 10%
        cond_short = dod_mask & (duration_hours <= 6)
        df.loc[cond_short, "end"] = df.loc[cond_short, "start"] + 0.9 * duration.loc[cond_short]
        
        # Medium trajectories (6-72h): mask last 5%
        cond_medium = dod_mask & (duration_hours > 6) & (duration_hours <= 72)
        df.loc[cond_medium, "end"] = df.loc[cond_medium, "start"] + 0.95 * duration.loc[cond_medium]
        
        # Long trajectories (>72h): mask last 2%
        cond_long = dod_mask & (duration_hours > 72)
        df.loc[cond_long, "end"] = df.loc[cond_long, "start"] + 0.98 * duration.loc[cond_long]
    
    elif method == 'absolute':
        # Your original approach but with validation
        
        # <3h: minus 10 minutes
        cond1 = dod_mask & (duration_hours < 3)
        new_end = df.loc[cond1, "DOD"] - pd.Timedelta(minutes=10)
        df.loc[cond1, "end"] = new_end
        
        # 3-72h: minus 30 minutes
        cond2 = dod_mask & (duration_hours >= 3) & (duration_hours <= 72)
        new_end = df.loc[cond2, "DOD"] - pd.Timedelta(minutes=30)
        df.loc[cond2, "end"] = new_end
        
        # 72h-7d: minus 3 hours
        cond3 = dod_mask & (duration_hours > 72) & (duration_hours <= 168)
        new_end = df.loc[cond3, "DOD"] - pd.Timedelta(hours=3)
        df.loc[cond3, "end"] = new_end
        
        # >7d: minus 1 day
        cond4 = dod_mask & (duration_hours > 168)
        new_end = df.loc[cond4, "DOD"] - pd.Timedelta(days=1)
        df.loc[cond4, "end"] = new_end
    
    elif method == 'drop_last_bin':
        # Don't adjust end time here - flag for bin_df creation
        # Mark patients to drop last bin
        df['drop_last_bin'] = dod_mask
        logger.info(f"Marked {n_deaths} patients to drop last bin during bin creation")
        return df
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # CRITICAL VALIDATION: Ensure end > start
    min_duration = pd.Timedelta(hours=min_duration_hours)
    
    invalid_mask = dod_mask & (df["end"] <= df["start"] + min_duration)
    n_invalid = invalid_mask.sum()
    
    if n_invalid > 0:
        logger.warning(f"⚠️  {n_invalid} patients would have end <= start after masking!")
        logger.warning(f"   Adjusting to minimum duration: {min_duration_hours}h")
        
        # Force minimum duration
        df.loc[invalid_mask, "end"] = df.loc[invalid_mask, "start"] + min_duration
        
        # Log examples
        if n_invalid > 0:
            examples = df[invalid_mask].head(3)
            for _, row in examples.iterrows():
                orig_dur = (original_end.loc[row.name] - row['start']).total_seconds() / 3600
                new_dur = (row['end'] - row['start']).total_seconds() / 3600
                logger.warning(f"   PID {row['PID']}: {orig_dur:.1f}h → {new_dur:.1f}h")
    
    # Report changes
    masked_patients = df[dod_mask]
    time_removed = (original_end - df["end"]).loc[dod_mask]
    avg_removed_hours = time_removed.dt.total_seconds().mean() / 3600
    
    logger.info(f"Masking complete:")
    logger.info(f"  Patients masked: {n_deaths}")
    logger.info(f"  Avg time removed: {avg_removed_hours:.1f} hours")
    logger.info(f"  Invalid trajectories fixed: {n_invalid}")
    
    return df

def _mask_mortality(df):
    """Adjust end times based on DOD and trajectory duration.
    Input base_df after DOD added"""
    for col in ["start", "end", "DOD"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    dod_mask = df["DOD"].notnull()
    if not dod_mask.any():
        return df
    
    duration_hours = (df["end"] - df["start"]).dt.total_seconds() / 3600
    logger.debug("duration_hours=%s", duration_hours)
    # < 3 hour: minus 10 minutes
    cond1 = dod_mask & (duration_hours < 3)
    df.loc[cond1, "end"] = df.loc[cond1, "DOD"] - pd.Timedelta(minutes=10)
    
    # >3 and <=72 hours: minus 1 hour
    cond2 = dod_mask & (duration_hours > 3) & (duration_hours <= 72)
    df.loc[cond2, "end"] = df.loc[cond2, "DOD"] - pd.Timedelta(minutes=30)
    
    # >72 hours: minus 6 hours
    cond3 = dod_mask & (duration_hours > 72)
    df.loc[cond3, "end"] = df.loc[cond3, "DOD"] - pd.Timedelta(hours=3)
    
    # >7 days: minus 1 day
    cond5 = dod_mask & (duration_hours > 168)
    df.loc[cond5, "end"] = df.loc[cond5, "DOD"] - pd.Timedelta(days=1)
    
    return df


if __name__ == "__main__":
    create_base_df(cfg)
    create_bin_df(cfg)
