import pandas as pd
import numpy as np
import gc

from astra.utils import logger, cfg, get_base_df, mark_keywords_in_df
from astra.utils import ensure_datetime, is_file_present, inches_to_cm, ounces_to_kg

from astra.data.mappings import (
    VITALS_MAP, TEMP_FAHRENHEIT,BP_TYPES, HEIGHT_WEIGHT_MAP,
    LABS_FEATURE_MAP, LABS_REVERSE_MAP,
    ICU_MAP,
    ATC_LVL3_MAP, ATC_LVL4_MAP, MEDICATION_ACTION_LIST,
    PROCEDURE_MAP, PROCEDURE_REVERSE_MAP, PROCEDURE_INCLUDE_LIST,
    ADT_PATTERNS, classify_department,
)


def filter_base(base):
    # filter by notes
    logger.info("Loading Notes")
    df = pd.read_csv("data/raw/Notater.csv")
    dff = filter_inhospital(base, df, cfg, "Redigeringstidspunkt", offset=0)
    dff = dff.merge(base[["PID", "start"]], on="PID", how="left")

    keywords = ["traume", "trauma", "tilskadekomst", "traumemodtagelse", "traumecenter"]
    dff = mark_keywords_in_df(
        dff,
        "Note",
        keywords,
        "Oprettelsestidspunkt",
        "start",
        t_delta=12,
        new_column="TRAUMATEXT",
    )

    # Mark in base
    pids_wtt_12 = dff.loc[
        (dff["TRAUMATEXT"] == True) & (dff["within_12_hours"] == True)
    ].PID.unique()
    logger.info(
        f"{len(pids_wtt_12)} of {len(base)} patients have trauma keywords in notes."
    )
    base.loc[base.PID.isin(pids_wtt_12), "TRAUMATEXT"] = True
    return base


def filter_subsets_inhospital(cfg, base=None):
    metadata = pd.read_csv("data/external/metadata.csv")

    # Make sure that we have metadata for all files intended
    missing_files = [
        file
        for file in cfg["default_load_filenames"]
        if file not in metadata["filename"].values
    ]
    assert (
        len(missing_files) == 0
    ), f"{missing_files} are not present in data/external/metadata.csv"

    df = pd.DataFrame()

    if base is None:
        base = get_base_df()

    for filename in metadata.filename:
        del df
        gc.collect()
        logger.debug(f"Filtering {filename}")
        df = pd.read_csv(f"data/raw/{filename}.csv", low_memory=False, index_col=0)
        
        dt_name = str(
            metadata.loc[metadata["filename"] == filename]["dt_colname"].iat[0]
        )
        offset = int(
            metadata.loc[metadata["filename"] == filename]["ts_offset"].iat[0]
        )

        filtered_df = filter_inhospital(base, df, cfg, dt_name, offset=offset)
        # TODO: add valuefilter layer
        filtered_df.to_pickle(f"data/interim/concepts/{filename}.pkl", protocol=4)


def filter_inhospital(
    base: pd.DataFrame, df: pd.DataFrame, cfg, dt_name: str, offset=1
) -> pd.DataFrame:
    # save colnames for return
    colnames = df.columns.to_list()
    # ensure datetime format for input df
    df = ensure_datetime(df, dt_name)
    # merge and filter
    merged_df = base[["PID", "CPR_hash", "start", "end"]].merge(
        df, on="CPR_hash", how="left"
    )

    filtered_df = merged_df[
        (merged_df[dt_name] >= merged_df["start"] - pd.DateOffset(days=offset))
        & (merged_df[dt_name] <= merged_df["end"] + pd.DateOffset(days=offset))
    ]
    filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    #logger.info(f"Reduced df with {len(df)-len(filtered_df)} rows.")
    logger.debug(f">Original df len: {len(df)}, new df len: {len(filtered_df)}")
    return filtered_df[colnames + ["PID"]]


### CONCEPT SPECIFICS

def filter_vitals(vit):
    # Create a copy to avoid SettingWithCopyWarning
    vit = vit.copy()

    def fahrenheit_to_celsius(f):
        return (f - 32) * 5.0 / 9.0

    for f in TEMP_FAHRENHEIT:
        numeric_vals = pd.to_numeric(vit.loc[vit.Vital_parametre == f, 'Værdi'], errors='coerce')
        vit.loc[numeric_vals.index, 'Værdi'] = numeric_vals.apply(fahrenheit_to_celsius)

    # Value-based F→C for 'Temperatur': bimodal distribution — values >50 are in °F
    temp_numeric = pd.to_numeric(vit.loc[vit.Vital_parametre == 'Temperatur', 'Værdi'], errors='coerce')
    f_idx = temp_numeric[temp_numeric > 50].index
    vit.loc[f_idx, 'Værdi'] = temp_numeric.loc[f_idx].apply(fahrenheit_to_celsius)

    # rename cols to standard and reduce
    vit.rename(columns={"Værdi":"VALUE", "Vital_parametre":"FEATURE", "Registreringstidspunkt":"TIMESTAMP"}, inplace=True)
    vit = vit[["TIMESTAMP","PID", "FEATURE", "VALUE"]]

    # split BP — uses BP_TYPES from mappings
    for bt in BP_TYPES:
        mask = vit['FEATURE'] == bt
        if len(vit.loc[mask])>0:
            split_values = vit.loc[mask, 'VALUE'].str.split('/', n=1, expand=True)
            vit.loc[mask, 'FEATURE'] = 'SBP'
            vit.loc[mask, 'VALUE'] = split_values[0]
            diastolic_rows = vit[mask].copy()
            diastolic_rows['FEATURE'] = 'DBP'
            diastolic_rows['VALUE'] = split_values[1]
            vit = pd.concat([vit, diastolic_rows], ignore_index=True)
            vit.loc[vit['FEATURE'].isin(['SBP', 'DBP']), 'VALUE'] = pd.to_numeric(
                vit.loc[vit['FEATURE'].isin(['SBP', 'DBP']), 'VALUE'],
                errors='coerce'
            )
            vit['VALUE'] = vit['VALUE'].astype(str)

    # Map parameter names — uses VITALS_MAP and HEIGHT_WEIGHT_MAP from mappings
    vit["FEATURE"] = vit["FEATURE"].replace(to_replace=VITALS_MAP)
    vit["FEATURE"] = vit["FEATURE"].replace(to_replace=HEIGHT_WEIGHT_MAP)
    vit.loc[vit.FEATURE == 'HEIGHT', 'VALUE'] = inches_to_cm(vit[vit.FEATURE == 'HEIGHT'].VALUE.astype(float))
    vit.loc[vit.FEATURE == 'WEIGHT','VALUE'] = ounces_to_kg(vit[vit.FEATURE == 'WEIGHT'].VALUE.astype(float))
    vit[(vit.FEATURE.isin(list(set(HEIGHT_WEIGHT_MAP.values()))))]

    pattern = r'([<>]\s*)?[-+]?\d*\.\d+|\d+\.?\d*'
    vit = vit[(vit.FEATURE.isin(list(set(VITALS_MAP.values()))))
                & (vit.VALUE.notnull())
               & ((vit['VALUE'].str.contains(pattern, regex=True) ) | (vit['VALUE'].dtype==float))].copy(deep=True)

    return vit

def filter_procedures(proc):
    # Uses PROCEDURE_INCLUDE_LIST and PROCEDURE_REVERSE_MAP from mappings
    proc = proc[proc["ProcedureCode"].isin(PROCEDURE_INCLUDE_LIST)].copy(deep=True)

    proc.rename(
        columns={"ProcedureCode": "VALUE", "ServiceDatetime": "TIMESTAMP"},
        inplace=True,
    )

    proc.VALUE = proc.VALUE.replace(PROCEDURE_REVERSE_MAP)
    logger.info(f"Using {len(proc)} observations of procedures")
    proc["FEATURE"] = "procedures"
    return proc


def filter_labs(lab):
    """Filter by value and by type of lab test.

    Uses LABS_FEATURE_MAP and LABS_REVERSE_MAP from mappings.
    """
    lab["Resultatværdi"] = lab["Resultatværdi"].str.replace(",", ".")
    lab["Resultatværdi"] = lab["Resultatværdi"].str.replace("*", "")
    pattern = r"([<>]\s*)?[-+]?\d*\.\d+|\d+\.?\d*"
    lab = lab[lab["Resultatværdi"].notnull()].copy(deep=True)
    lab = lab[lab["Resultatværdi"].str.contains(pattern, regex=True)].copy(deep=True)

    # Keep relevant features only — uses LABS_FEATURE_MAP from mappings
    include_list = [name for names in LABS_FEATURE_MAP.values() for name in names]
    lab = lab[lab["BestOrd"].isin(include_list)].copy(deep=True)

    lab.rename(
        columns={
            "BestOrd": "FEATURE",
            "Resultatværdi": "VALUE",
            "Prøvetagningstidspunkt": "TIMESTAMP",
        },
        inplace=True,
    )

    lab.VALUE = lab.VALUE.replace({"<": "", ">": ""}, regex=True)
    lab["VALUE"] = pd.to_numeric(lab["VALUE"], errors="coerce")
    lab = lab.dropna(subset=["VALUE"])
    lab.FEATURE = lab.FEATURE.replace(LABS_REVERSE_MAP)
    logger.info(f"Using {len(lab)} observations of labs")
    return lab


def filter_ita(ita):
    """Uses ICU_MAP from mappings."""
    ita = ita.copy()
    ita.rename(
        columns={
            "ITAOversigt_Måling": "FEATURE",
            "Værdi": "VALUE",
            "Målingstidspunkt": "TIMESTAMP",
        },
        inplace=True,
    )

    ita["FEATURE"] = ita["FEATURE"].replace(to_replace=ICU_MAP)
    return ita


def reverse_dict_replace(original_dict, df, atc_level):
    # Invert the dictionary
    inverted_dict = {}
    for key, value in original_dict.items():
        # Ensure value is a list for consistent processing
        if isinstance(value, list):
            for item in value:
                inverted_dict[item] = key
        else:
            inverted_dict[value] = key
    # Replace values in the 'ID' column using the inverted dictionary
    df["FEATURE"] = (
        df[f"ATC{atc_level}"]
        .replace(inverted_dict)
        .where(df[f"ATC{atc_level}"].isin(inverted_dict.keys()), np.nan)
    )
    logger.info(
        f">>Medicine: found {len(df[df.FEATURE.notnull()])} administrations of a ATC level {atc_level} drug"
    )
    return df


def filter_medicin(med):
    """Uses MEDICATION_ACTION_LIST, ATC_LVL3_MAP, ATC_LVL4_MAP from mappings."""
    med = med[med.Handling.isin(MEDICATION_ACTION_LIST)].copy()
    med["ATC3"] = med.ATC.str[:3]
    med["ATC4"] = med.ATC.str[:4]

    med3 = reverse_dict_replace(ATC_LVL3_MAP, med.copy(deep=True), 3)
    med4 = reverse_dict_replace(ATC_LVL4_MAP, med.copy(deep=True), 4)
    med = pd.concat([med3, med4]).drop_duplicates().copy()
    med = med[med["FEATURE"].notnull()].copy()
    med["VALUE"] = med["FEATURE"]
    med["FEATURE"] = "medication"

    med.rename(
        columns={"Administrationstidspunkt": "start", "Seponeringstidspunkt": "end"},
        inplace=True,
    )
    med["TIMESTAMP"] = med["start"]
    logger.info(f"Using {len(med)} observations of medicine")
    return med




def filter_adt(adt, base_df=None):
    """Filter ADT events: classify department types and prepare interval timestamps.

    Uses classify_department() and ADT_PATTERNS from mappings.

    Args:
        adt: Raw ADT DataFrame (after filter_inhospital).
        base_df: Optional base DataFrame for filling missing Flyt_ud.
            If None, loads from disk via get_base_df().
    """
    adt = adt.copy()

    adt["Flyt_ind"] = pd.to_datetime(adt["Flyt_ind"], errors="coerce")
    adt["Flyt_ud"] = pd.to_datetime(adt["Flyt_ud"], errors="coerce")

    # Classify departments using shared classify_department()
    adt["VALUE"] = adt["Afsnit"].apply(classify_department)

    adt["FEATURE"] = "ADT"

    # Drop unrecognized departments
    adt = adt[adt["VALUE"].notna()].copy()
    logger.info(f"ADT: {len(adt)} events after department classification")

    # Sort for forward-fill logic
    adt = adt.sort_values(["PID", "Flyt_ind"]).reset_index(drop=True)

    # Handle missing Flyt_ud: fill from next event's Flyt_ind per patient
    adt["next_flyt_ind"] = adt.groupby("PID")["Flyt_ind"].shift(-1)
    mask_missing_end = adt["Flyt_ud"].isna()
    adt.loc[mask_missing_end, "Flyt_ud"] = adt.loc[mask_missing_end, "next_flyt_ind"]

    # For remaining NaN (last event per patient), fill from base_df end time
    if mask_missing_end.any() and adt["Flyt_ud"].isna().any():
        if base_df is None:
            base_df = get_base_df()
        end_map = base_df.set_index("PID")["end"]
        still_missing = adt["Flyt_ud"].isna()
        adt.loc[still_missing, "Flyt_ud"] = adt.loc[still_missing, "PID"].map(end_map).values

    adt = adt.drop(columns=["next_flyt_ind"])

    # Drop any rows still missing end timestamp
    n_before = len(adt)
    adt = adt.dropna(subset=["Flyt_ud"])
    if len(adt) < n_before:
        logger.warning(f"ADT: dropped {n_before - len(adt)} events with missing Flyt_ud")

    adt["TIMESTAMP"] = adt["Flyt_ind"]
    adt["END_TIMESTAMP"] = adt["Flyt_ud"]

    logger.info(f"Using {len(adt)} ADT observations")
    return adt


def collect_filter(concept: str):
    filter_funcs = {
        "VitaleVaerdier": filter_vitals,
        "ITAOversigtsrapport": filter_ita,
        "Labsvar": filter_labs,
        "Medicin": filter_medicin,
        "Procedurer": filter_procedures,
        "ADTHaendelser": filter_adt,
    }

    return filter_funcs[concept]


if __name__ == "__main__":
    pass