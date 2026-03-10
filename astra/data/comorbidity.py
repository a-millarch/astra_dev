from azureml.core import Dataset

import pandas as pd

from astra.utils import add_categorical_variable, cfg, ensure_parent_dir

pd.options.mode.chained_assignment = None
# move to config
cc_col_names = [
    "CC_ADHD",
    "CC_ANGINAPECTORIS",
    "CC_COPD",
    "CC_DIABETES",
    "CC_BLEEDING",
    "CC_CIRRHOSIS",
    "CC_DEMENTIA",
    "CC_CVA",
    "CC_CHF",
    "CC_HYPERTENSION",
    "CC_MI",
    "CC_PAD",
    "CC_MENTALPERSONALITY",
    "CC_RENAL",
    "CC_SUBSTANCEABUSE",
    "CC_ALCOHOLISM",
    "CC_SMOKING",
]

cc_sks = [
    ("DF90", "DF900", "DF901", "DF909"),
    (
        "DI20",
        "DI200",
        "DI200B",
        "DI200C",
        "DI201",
        "DI208",
        "DI208D",
        "DI208E",
        "DI208E1",
        "DI208E2",
        "DI209",
    ),
    ("DJ44"),
    ("DE10", "DE11"),
    ("DD66"),
    ("DK703", "DK717", "DK7732E", "DK743", "DK744", "DK745", "DK746", "DP788A"),
    ("DF00", "DF01", "DF02", "DF03"),
    ("DI61", "DI63"),
    ("DI110", "DI50"),
    ("DI1"),
    ("DI21"),
    ("DI70"),
    ("DF20", "DF31", "DF33", "DF401", "DF431", "DF602"),
    ("DN17"),
    ("DZ722"),
    ("DF101", "DF102", "DF103", "DF104", "DF105", "DF106", "DF107", "DF108", "DF109"),
    ("DZ720E", "DZ720A", "DF17"),
]

hc_dict = {
    "HC_SUPERFICIALINCISIONSSI": [("DT814G"), 30],
    "HC_DEEPSSI": [("DT814H"), 30],
    "HC_ORGANSPACESSI": [("DT814B", "DT814C", "DT814I", "DT814J"), 30],
    # Check if respirator!
    "HC_VAPNEUMONIA": [("DJ12", "DJ13", "DJ14", "DJ15", "DJ16", "DJ17", "DJ18"), 30],
    "HC_EMBOLISM": [("DT817D", "DI26"), 30],
    "HC_CAUTI": [("DT814U", "DN30"), 30],
    "HC_STROKECVA": [("DT817Y1", "DI61", "DI63"), 30],
    "HC_CARDARREST": [("DI460", "DI461"), 30],
    "HC_MI": [("DT817Y2", "DI21"), 30],
    "HC_DVTHROMBOSIS": [("DT817C", "DI80"), 30],
    "HC_SEPSIS": [("DT814D", "DA40", "DA41"), 30],
    # "SEPSHOCK"  : [['DR572'], -30]
}


def add_from_diagnoses(
    diag,
    base_df,
    dt_col_name="ServiceDate",
    cc_col_names=cc_col_names,
    cc_sks=cc_sks,
    hc_dict=hc_dict,
):
    # Diagnose

    cc_df = pd.DataFrame(list(zip(cc_col_names, cc_sks)), columns=["varname", "sks"])
    cc_df.head()

    # also med dependent var
    diag_med_dependent = ["CC_HYPERTENSION"]

    all_sks = tuple(j for i in cc_df.sks for j in (i if isinstance(i, tuple) else (i,)))

    cc_diag = diag[diag.Diagnosekode.str.startswith(all_sks, na=False)].copy(deep=True)

    cc_res = base_df.copy(deep=True)

    for index, row in cc_df.iterrows():
        cc_res = add_categorical_variable(
            cc_res,
            cc_diag,
            base_df_date_col_name=dt_col_name,
            filt_column="Diagnosekode",
            filt_date="Noteret_dato",
            solved_date="Løst_dato",
            var_name=row["varname"],
            filt_val=row["sks"],
            delta_var="days",
            max_val=0,
            min_val=None,
        )

    hc_df = (
        pd.DataFrame.from_dict(hc_dict, orient="index", columns=["sks", "max_val"])
        .reset_index()
        .rename(columns={"index": "TQIP_name"})
    )

    hc_res = cc_res.copy(deep=True)

    for index, row in hc_df.iterrows():
        hc_res = add_categorical_variable(
            hc_res,
            diag,
            base_df_date_col_name=dt_col_name,
            filt_column="Diagnosekode",
            filt_date="Noteret_dato",
            solved_date="Løst_dato",
            var_name=row["TQIP_name"],
            filt_val=row["sks"],
            min_val=0,
            max_val=row["max_val"] * 24,
            delta_var="days",
        )

    return hc_res


def add_cc_hc(path=cfg["raw_file_path"], base=None, save_file=False):
    if base is None:
        base = pd.read_csv("data/interim/base_df.csv")

    diag_ds = Dataset.Tabular.from_parquet_files(path=path + "CPMI_Diagnoser.parquet")
    diag = diag_ds.to_pandas_dataframe()
    diag = diag[diag.CPR_hash.isin(base.CPR_hash)].copy(deep=True)

    res = add_from_diagnoses(diag, base)
    if save_file:
        ensure_parent_dir("data/interim/base_df.csv")
        res.to_csv("data/interim/base_df.csv")
    else:
        return res


if __name__ == "__main__":
    add_cc_hc(save_file=True)