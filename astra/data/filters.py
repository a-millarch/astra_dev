import pandas as pd
import numpy as np
import gc

from astra.utils import logger, cfg, get_base_df, mark_keywords_in_df
from astra.utils import ensure_datetime, is_file_present, inches_to_cm, ounces_to_kg


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
        filtered_df.to_pickle(f"data/interim/concepts/{filename}.pkl")


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

    # Fix temp in fahrenheit first
    vit.loc[vit.Vital_parametre == 'Temp.', 'Værdi'] = vit["Værdi_Omregnet"]
    
    
    # rename cols to standard and reduce
    vit.rename(columns={"Værdi":"VALUE", "Vital_parametre":"FEATURE", "Registreringstidspunkt":"TIMESTAMP"}, inplace=True)
    vit = vit[["TIMESTAMP","PID", "FEATURE", "VALUE"]]
    
    # split BP
    for bt in ["BT", 
               "ART inv BT", 
               "Invasivt BT - ABP (sys/dia)", 
               "NIBP", 
               "ABP inv BT",
              "Invasivt BT - ART (sys/dia)"]:
        mask = vit['FEATURE'] == bt
        if len(vit.loc[mask])>0:
            # Split the VALUE column
            split_values = vit.loc[mask, 'VALUE'].str.split('/', n=1, expand=True)
    
            # Update existing rows for systolic
            vit.loc[mask, 'FEATURE'] = 'SBP'
            vit.loc[mask, 'VALUE'] = split_values[0]
    
            # Create new rows for diastolic
            diastolic_rows = vit[mask].copy()
            diastolic_rows['FEATURE'] = 'DBP'
            diastolic_rows['VALUE'] = split_values[1]
    
            # Concatenate the original DataFrame with the new diastolic rows
            vit = pd.concat([vit, diastolic_rows], ignore_index=True)
    
            # Optional: Convert VALUE to numeric for both SYSTOLIC and DIASTOLIC
            vit.loc[vit['FEATURE'].isin(['SBP', 'DBP']), 'VALUE'] = pd.to_numeric(
                vit.loc[vit['FEATURE'].isin(['SBP', 'DBP']), 'VALUE'], 
                errors='coerce'
            )
            vit['VALUE'] = vit['VALUE'].astype(str)
    
            # Optional: Sort the DataFrame to keep SYSTOLIC and DIASTOLIC rows together
            #vit = vit.sort_values(by=['FEATURE', 'VALUE'])
        
    measurements_map = {
        'Saturation': 'SPO2',
        'ABP Puls (fra A-kanyle)':'HR',
        'Puls': 'HR',
        'Puls (fra SAT-måler)': 'HR',
        'Resp.frekvens': 'RESPIRATORYRATE',
        #'BT': 'BP',
        #'ART inv BT': 'BP',
        'SYSTOLIC':'SBP',
        'ART mean inv BT': 'MAP',
        'Temp (in Celsius)': 'TEMP',
        'Temp.':'TEMP',
        'DBP':'DBP',
        'SBP':'SBP',
    }

    hw_map = {
        "Højde": "HEIGHT",
        "Vægt": "WEIGHT"
    }

    vit["FEATURE"] = vit["FEATURE"].replace(to_replace=measurements_map)
    
    vit["FEATURE"] = vit["FEATURE"].replace(to_replace=hw_map)
    vit.loc[vit.FEATURE == 'HEIGHT', 'VALUE'] = inches_to_cm(vit[vit.FEATURE == 'HEIGHT'].VALUE.astype(float))
    vit.loc[vit.FEATURE == 'WEIGHT','VALUE'] =ounces_to_kg(vit[vit.FEATURE == 'WEIGHT'].VALUE.astype(float))
    vit[(vit.FEATURE.isin(list(set(hw_map.values()))))].to_pickle('data/interim/Height_Weight.pkl')
        
    pattern = r'([<>]\s*)?[-+]?\d*\.\d+|\d+\.?\d*'
    vit =  vit[(vit.FEATURE.isin(list(set(measurements_map.values())))) 
                & (vit.VALUE.notnull()) 
               & ((vit['VALUE'].str.contains(pattern, regex=True) ) | (vit['VALUE'].dtype==float))].copy(deep=True)



    return vit

def filter_procedures(proc):

    feature_map = {
        "neuro_major": (
            "KAAA27",
            "KAAD05",
            "KAAF00A",
            "KAAD00",
            "KAAD15",
            "KAAA20",
            "KAAA40",
            "KAAC00",
            "KAAA99",
            "KAAD40",
            "KAAL11",
            "KAAB30",
            "KAAD10",
            "KABC60",
            "KAAD30",
            "KAWD00",
            "KAAK35",
            "KAAK00",
            "KAAK10",
        ),
        "abdominal_major": (
            "KNHJ63",
            "KJBA00",
            "KPCT20",
            "KPCT99",
            "KJDH70",
            "KJJA96",
            "KKBV02A",
            "KJJW96",
            "KKAH00",
            "KJKB30",
            "KKAD10",
            "KKAC00",
            "KPCT30",
            "KJJA50",
            "KJJB00",
        ),
        "vascular_major": (
            "KFNG05A",
            "KFNG02A",
            "KPBH20",
            "KPET11",
            "KPEA12",
            "KPBC30",
            "KPHC23",
            "KPDC30",
            "KPBB30",
            "KPDG10",
            "KPDT30",
            "KPEH12",
            "KPBC10",
            "KPBN20",
            "KACB22",
            "KPAC20",
            "KPBE30",
            "KPDF10",
            "KPEA10",
            "KPBA20",
            "KPHH99",
            "KFCA70",
            "KFCA50",
            "KPBU82",
            "KPHP30",
            "KPEN11",
            "KPEH20",
            "KPFN30",
            "KPEC12",
            "KNDL41",
            "KPDQ10",
            "KPAP21",
            "KPCH30",
            "KPFC10",
            "KPHC22",
            "KPAQ21",
            "KPBC20",
            "KPEP11",
            "KPEU87",
            "KPFE10",
        ),
        "thorax_major": (
            "KGAB10",
            "KGAA31",
            "KGAB20",
            "KGDB11",
            "KGAC10",
            "KFLC00",
            "KFXE00",
            "KFEB10",
            "KFXD00",
            "KFWW96",
            "KGDA40",
            "KGAE30",
            "KUGC02",
            "KFJB00",
            "KGAE03",
            "KGDB10",
            "KGDA41",
            "KFEW96",
            "KGDB96",
            "KGAE96",
        ),
        "orto_major": (
            "KNGJ22",
            "KNAG73",
            "KNAG40",
            "KNFJ54",
            "KNAG70",
            "KNGM09",
            "KNEJ29",
            "KNGJ29",
            "KNGJ52",
            "KNFJ25",
            "KNDL40A",
            "KNEJ69",
            "KACB23",
            "KNGJ21",
            "KNCJ45",
            "KNCJ27",
            "KACB29",
            "KNAG71",
            "KNDA02",
            "KACC51",
            "KNHJ45",
            "KNFJ51",
            "KNAG72",
            "KNDM09",
            "KNHJ62",
            "KNDJ42",
            "KNFJ43",
            "KNBQ03",
            "KNCJ65",
            "KNGQ19",
            "KNAG76",
            "KNGJ40",
            "KABC56",
            "KPBB99",
            "KACB21",
            "KNGJ61",
            "KNDL40",
            "KNFQ19",
            "KNAN00",
            "KNBJ41",
            "KNBJ61",
            "KNCJ88",
            "KNBA02",
            "KNHJ80",
            "KNDJ43",
            "KNHJ47",
            "KNGE29",
            "KNHJ23",
            "KNHJ71",
            "KACA13",
            "KNFJ10",
            "KNFJ70",
            "KNFJ73",
            "KNHN09",
            "KNCJ67",
            "KNGJ71",
            "KNCJ26",
            "KNCJ60",
            "KNCJ42",
            "KNAN03",
            "KNFJ52",
            "KNCE22",
            "KNDQ99",
            "KNHQ22",
            "KNCL49",
            "KQCG30",
            "KNCJ64",
            "KNAN02",
            "KNAK12",
            "KNHJ72",
            "KABA00",
            "KNCJ28",
            "KNCJ80",
            "KNFJ44",
            "KNHJ82",
            "KNFJ55",
            "KNEJ89",
            "KNAJ12",
            "KACC29",
            "KNDJ11",
            "KNDU39",
            "KNDJ70",
            "KNBJ51",
            "KNHJ22",
            "KNHL49",
            "KNHE99",
            "KNFM09",
            "KNGJ80",
            "KQAA10",
            "KNHJ14",
            "KNHJ44",
            "KNDL41A",
            "KNAK10",
            "KNBJ62",
            "KNBJ21",
            "KNCJ47",
            "KNAJ00",
            "KACA19",
            "KNFQ99",
            "KNFJ50",
            "KNGJ73",
            "KNHJ81",
            "KNGM99",
            "KECB40",
            "KNGD22",
            "KNCJ05",
            "KNHJ25",
            "KACC53",
            "KNHJ24",
            "KNCM09",
            "KNDH12",
            "KNAN04",
            "KNFJ65",
            "KNDH02",
            "KNHJ41",
            "KNHJ74",
            "KNCJ66",
            "KNGJ63",
            "KNHJ42",
            "KNFJ45",
            "KNGJ42",
            "KNAG41",
            "KNFA02A",
        ),
        "ønh_major": (
            "KEFB20",
            "KEDC38",
            "KEEC25",
            "KEEC35",
            "KDLD30",
            "KEWE00",
            "KECB20A",
            "KDQE00",
            "KEDC36",
            "KGBA00",
            "KGAB00",
            "KDWE00",
            "KENC00",
            "KDHD30",
            "KDJD20",
            "KDAD30",
            "KDWA00",
            "KDQW99",
            "KEMC00",
            "KEDC39B",
            "KDLD20",
        ),
    }

    reversed_feature_map = {}
    for key, values in feature_map.items():
        for value in values:
            reversed_feature_map[value] = key
    # keep relevant features only
    include_list = [item for sublist in feature_map.values() for item in sublist]
    proc = proc[proc["ProcedureCode"].isin(include_list)].copy(deep=True)

    # rename cols to adherence
    proc.rename(
        columns={"ProcedureCode": "VALUE", "ServiceDatetime": "TIMESTAMP"},
        inplace=True,
    )
    
    # Replace feature
    proc.VALUE = proc.VALUE.replace(reversed_feature_map)
    logger.info(f"Using {len(proc)} observations of procedures")
    proc["FEATURE"] = "procedures"
    return proc


def filter_labs(lab):
    """Filter by value and by type of lab test"""
    # TODO: Add TIMESTAMP
    ####### !!!!!!!! ############

    feature_map = {
        "LACTATE": (
            "LAKTAT(POC);P(AB)",
            "LAKTAT;P(AB)",
            "LAKTAT;P(VB)",
            "LAKTAT(POC);P(VB)",
            "LAKTAT;CSV",
            "LAKTAT(POC);CSV",
            "LAKTAT(POC);P(KB)",
        ),
        "BASE_EXCESS": ("BASE EXCESS;ECV", "ECV-BASE EXCESS;(POC)"),
        "HEMOGLOBIN": ("HÆMOGLOBIN;B", "HÆMOGLOBIN(POC);B", "HÆMOGLOBIN (POC);B"),
        "LEUKOCYTES": ("LEUKOCYTTER;B",),
        "B-GROUP-LEUKOCYTES": (
            "LEUKOCYTTYPE (MIKR.) GRUPPE;B",
            "LEUKOCYTTYPE GRUPPE;B",
            "LEUKOCYTTYPE; ANTALK. (LISTE);B",
        ),
        "TEG-R": ("TEG-R",),
        "TEG-MA": ("TEG-MA",),
        "TEG-LY30": ("TEG-LY30",),
    }
    # for replacing later
    reversed_feature_map = {}
    for key, values in feature_map.items():
        for value in values:
            reversed_feature_map[value] = key
    lab["Resultatværdi"] = lab["Resultatværdi"].str.replace(",", ".")
    lab["Resultatværdi"] = lab["Resultatværdi"].str.replace("*", "")
    # where either float or prefix with < or > such as ">30.0"
    pattern = r"([<>]\s*)?[-+]?\d*\.\d+|\d+\.?\d*"
    # remove nulls and apply float filter
    lab = lab[lab["Resultatværdi"].notnull()].copy(deep=True)
    lab = lab[lab["Resultatværdi"].str.contains(pattern, regex=True)].copy(deep=True)

    # keep relevant features only
    include_list = [item for sublist in feature_map.values() for item in sublist]
    lab = lab[lab["BestOrd"].isin(include_list)].copy(deep=True)

    # rename cols to adherence
    lab.rename(
        columns={
            "BestOrd": "FEATURE",
            "Resultatværdi": "VALUE",
            "Prøvetagningstidspunkt": "TIMESTAMP",
        },
        inplace=True,
    )

    # Replace value and feature
    lab.VALUE = lab.VALUE.replace({"<": "", ">": ""}, regex=True)
    # ensure float
    lab["VALUE"] = pd.to_numeric(lab["VALUE"], errors="coerce")
    lab = lab.dropna(subset=["VALUE"])
    lab.FEATURE = lab.FEATURE.replace(reversed_feature_map)
    logger.info(f"Using {len(lab)} observations of labs")
    return lab


def filter_ita(ita):
    # Create a copy to avoid SettingWithCopyWarning
    ita = ita.copy()
    ita.rename(
        columns={
            "ITAOversigt_Måling": "FEATURE",
            "Værdi": "VALUE",
            "Målingstidspunkt": "TIMESTAMP",
        },
        inplace=True,
    )

    feature_map = {
        "GLASGOW COMA SCORE": "GCS",
        "Glasgow Coma Score": "GCS",
        "SAPS 3 SCORE": "SAPS3",
        "SOFA total score": "SOFA",
    }
    ita["FEATURE"] = ita["FEATURE"].replace(to_replace=feature_map)

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
    # use only medications given
    action_list = [
        "Administreret",
        "Ny pose",
        "Selvadministration",
        "Adm. ernæring/sterilt vand",
        "Genstartet",
        "Infusion/pose skiftet",
        "Selvmedicinering",
        "Status, indgift",
    ]
    med = med[med.Handling.isin(action_list)].copy()
    # reduce ATC code
    med["ATC3"] = med.ATC.str[:3]
    med["ATC4"] = med.ATC.str[:4]

    lvl3_map = {
        "cardiovascular_drugs": ["C01", "C02", "C07"],
        "antibiotics": "J01",
        "neuro_drugs": ["N05", "M03"],
        "anti_thrombotic": "B01",
        "diuretics": "C03",
        "hemostatics": "B02",
    }

    lvl4_map = {
        "infusion": ["B05B", "B05X"],
        "blood": ["B05A"],
        "opiods": "N02A",
        "local_anastethics": "N01B",
        "anastethics": "N01A",
        "insulin": "A10A",
    }

    med3 = reverse_dict_replace(lvl3_map, med.copy(deep=True), 3)
    med4 = reverse_dict_replace(lvl4_map, med.copy(deep=True), 4)
    med = pd.concat([med3, med4]).drop_duplicates().copy()
    med = med[med["FEATURE"].notnull()].copy()
    # set value to binary, consider changing to actual values of daily dosage or smth
    med["VALUE"] = med["FEATURE"] #CHANGED!
    med["FEATURE"] = "medication" #CHANGED!

    # rename for convenience
    med.rename(
        columns={"Administrationstidspunkt": "start", "Seponeringstidspunkt": "end"},
        inplace=True,
    )
    med["TIMESTAMP"] = med["start"]
    logger.info(f"Using {len(med)} observations of medicine")
    # WIP: create new rows based on interval between administration and seponation
    # med = expand_datetime_rows(med)
    return med




def collect_filter(concept: str):
    filter_funcs = {
        "VitaleVaerdier": filter_vitals,
        "ITAOversigtsrapport": filter_ita,
        "Labsvar": filter_labs,
        "Medicin": filter_medicin,
        "Procedurer": filter_procedures,
    }

    return filter_funcs[concept]


if __name__ == "__main__":
    pass