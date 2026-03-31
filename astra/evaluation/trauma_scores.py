"""
Traditional trauma risk scores (RTS, ISS, TRISS) for baseline comparison.

Computes static clinical risk scores and evaluates them against the ASTRA model's
time-dependent predictions. Requires Azure infrastructure (DTR data, R scripts)
for full ISS/TRISS computation — behind --trauma-scores CLI flag.
"""

import os
import logging
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from astra.evaluation.utils import (
    calculate_roc_auc_ci,
    calculate_average_precision_ci,
    delong_test_paired,
    benjamini_hochberg,
    step_to_time,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SCORE COMPUTATION
# ============================================================================

def compute_rts(df: pd.DataFrame, gcs_col: str = 'GCS', sbp_col: str = 'SBP',
                rr_col: str = 'RR') -> pd.DataFrame:
    """Add RTS and RTSc columns using coded GCS, SBP, RR values.

    RTS = 0.9368 * GCS_code + 0.7326 * SBP_code + 0.2908 * RR_code
    """
    gcs = pd.to_numeric(df[gcs_col], errors='coerce').values
    sbp = pd.to_numeric(df[sbp_col], errors='coerce').values
    rr = pd.to_numeric(df[rr_col], errors='coerce').values

    gcs_code = np.select(
        [gcs >= 13, gcs >= 9, gcs >= 6, gcs >= 4],
        [4, 3, 2, 1], default=0
    ).astype(float)
    sbp_code = np.select(
        [sbp > 89, sbp >= 76, sbp >= 50, sbp >= 1],
        [4, 3, 2, 1], default=0
    ).astype(float)
    rr_code = np.select(
        [(rr >= 10) & (rr <= 29), rr >= 30, rr >= 6, rr >= 1],
        [4, 3, 2, 1], default=0
    ).astype(float)

    # Propagate NaN: if any component is NaN, result is NaN
    any_nan = np.isnan(gcs) | np.isnan(sbp) | np.isnan(rr)
    rts = 0.9368 * gcs_code + 0.7326 * sbp_code + 0.2908 * rr_code
    rtsc = gcs_code + sbp_code + rr_code
    rts[any_nan] = np.nan
    rtsc[any_nan] = np.nan

    df = df.copy()
    df['RTS'] = rts
    df['RTSc'] = rtsc
    return df


def compute_triss(df: pd.DataFrame, iss_col: str = 'ISS', age_col: str = 'AGE',
                  mechanism_col: str = 'mechanism', rts_col: str = 'RTS') -> pd.DataFrame:
    """Add TRISS survival and mortality probability columns.

    Uses MTOS coefficients for blunt vs penetrating trauma.
    """
    df = df.copy()
    iss = pd.to_numeric(df[iss_col], errors='coerce').values
    age = pd.to_numeric(df[age_col], errors='coerce').values
    rts = pd.to_numeric(df[rts_col], errors='coerce').values
    mechanism = df[mechanism_col].astype(str).str.lower().values

    age_index = np.where(age >= 55, 1.0, 0.0)

    # MTOS coefficients
    b0_blunt, b1_blunt, b2_blunt, b3_blunt = -0.4499, 0.8085, -0.0835, -1.7430
    b0_pen, b1_pen, b2_pen, b3_pen = -2.5355, 0.9934, -0.0651, -1.1360

    is_pen = np.isin(mechanism, ['penetrating', 'p', 'pen'])
    b0 = np.where(is_pen, b0_pen, b0_blunt)
    b1 = np.where(is_pen, b1_pen, b1_blunt)
    b2 = np.where(is_pen, b2_pen, b2_blunt)
    b3 = np.where(is_pen, b3_pen, b3_blunt)

    b = b0 + b1 * rts + b2 * iss + b3 * age_index
    survival_prob = expit(b)

    # NaN where any component missing
    any_nan = np.isnan(iss) | np.isnan(age) | np.isnan(rts)
    # Also NaN where mechanism is unknown/nan
    mech_invalid = ~np.isin(mechanism, ['blunt', 'b', 'blunt_tr', 'penetrating', 'p', 'pen'])
    invalid = any_nan | mech_invalid
    survival_prob[invalid] = np.nan

    df['TRISS_survival_prob'] = survival_prob
    df['TRISS_mors_prob'] = 1.0 - survival_prob
    return df


def compound_iss(df: pd.DataFrame) -> pd.DataFrame:
    """Create unified ISS column: max across ISS_DTR, riss, and niss."""
    df = df.copy()
    candidates = []
    for col in ['ISS_DTR', 'riss', 'niss']:
        if col in df.columns:
            candidates.append(pd.to_numeric(df[col], errors='coerce'))

    if candidates:
        df['ISS_COMPOUND'] = pd.concat(candidates, axis=1).max(axis=1)
    else:
        df['ISS_COMPOUND'] = np.nan
    return df


# ============================================================================
# DTR MATCHING (Azure-only)
# ============================================================================

BUFFER_DAYS = 2


def load_dtr_and_match(base_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Match patients to Danish Trauma Registry records.

    Returns DataFrame with PID + ISS + mechanism columns, or None if DTR unavailable.
    """
    try:
        from azureml.core import Dataset
    except ImportError:
        logger.warning("Azure ML SDK not available — skipping DTR matching")
        return None

    DTR_URL = (
        "https://forskerpln0ybkrdls01.blob.core.windows.net/"
        "researcher-data/maskerede_data/dtr.csv"
    )

    try:
        df_dtr = Dataset.Tabular.from_delimited_files(path=DTR_URL, separator=";")
        dtr = df_dtr.to_pandas_dataframe()
    except Exception as e:
        logger.warning(f"Could not load DTR data: {e}")
        return None

    dtr.insert(0, "dtr_row_nr", range(1, len(dtr) + 1))
    logger.info(f"Loaded DTR: {len(dtr)} rows")

    # Prepare base
    base = base_df[["CPR_hash", "ServiceDate", "start", "PID"]].copy()
    base["start"] = pd.to_datetime(base["start"], errors="coerce")
    base["ServiceDate"] = pd.to_datetime(base["ServiceDate"], errors="coerce")
    base["CPR_hash"] = base["CPR_hash"].astype(str).str.strip()
    base = base.reset_index(drop=True)
    base["_row_id"] = base.index

    # Prepare DTR
    dtr["BWSTDateTime"] = pd.to_datetime(dtr["BWSTDateTime"], errors="coerce")
    dtr["CPR_hash"] = dtr["CPR_hash"].astype(str).str.strip()
    dtr["date"] = dtr["BWSTDateTime"].dt.date
    dtr_keyed = (
        dtr.sort_values("BWSTDateTime")
        .drop_duplicates(subset=["CPR_hash", "date"], keep="last")
    )

    # Match: start date first, then ServiceDate ± buffer
    base["date_start"] = base["start"].dt.date
    m_start = base.merge(
        dtr_keyed,
        left_on=["CPR_hash", "date_start"],
        right_on=["CPR_hash", "date"],
        how="left",
        suffixes=("", "_dtr"),
    )
    m_start["dtr_row_nr_start"] = m_start["dtr_row_nr"]

    base["date_service"] = base["ServiceDate"].dt.date
    parts = []
    for off in range(-BUFFER_DAYS, BUFFER_DAYS + 1):
        parts.append(
            base[["_row_id", "CPR_hash", "date_service"]].assign(
                date_key=base["date_service"] + pd.Timedelta(days=off),
                off=off,
                abs_off=abs(off),
                sign_pref=(0 if off == 0 else (1 if off < 0 else 2)),
            )
        )
    service_long = pd.concat(parts, ignore_index=True)
    service_merged = service_long.merge(
        dtr_keyed,
        left_on=["CPR_hash", "date_key"],
        right_on=["CPR_hash", "date"],
        how="left",
    )
    service_merged["has_match"] = service_merged["dtr_row_nr"].notna().astype(int)
    service_best = (
        service_merged.sort_values(
            ["_row_id", "has_match", "abs_off", "sign_pref"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates(subset=["_row_id"], keep="first")
    )

    # Build final dataset
    dtr_out_cols = ["hospital", "TraumeCodeUnitCode", "BWSTDateTime", "dtr_row_nr"]
    service_best2 = service_best[["_row_id"] + dtr_out_cols].rename(
        columns={c: f"{c}_service" for c in dtr_out_cols}
    )
    m = m_start.merge(service_best2, on="_row_id", how="left")
    m["dtr_row_nr_service"] = m["dtr_row_nr_service"]
    for c in dtr_out_cols:
        m[c] = m[c].combine_first(m[f"{c}_service"])

    # Keep PID + all DTR columns
    final_cols = ["PID"] + [c for c in dtr.columns if c in m.columns]
    final = m[[c for c in final_cols if c in m.columns]].copy()
    for drop_col in ["dtr_row_nr", "Column1"]:
        if drop_col in final.columns:
            final.drop(columns=[drop_col], inplace=True)

    matched = final.dropna(subset=["BWSTDateTime"]) if "BWSTDateTime" in final.columns else final
    logger.info(f"DTR matched: {len(matched)}/{len(base_df)} patients")

    # Extract mechanism from AIS codes
    final = _preprocess_mechanism(final)
    return final


def _preprocess_mechanism(df: pd.DataFrame) -> pd.DataFrame:
    """Determine mechanism (blunt/penetrating) from AIS codes."""
    ais_cols = [
        col for col in df.columns
        if 'ais' in col.lower() and col.lower() not in ['mais', 'nais']
    ]

    df['mechanism'] = 'blunt'  # default

    for col in ais_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).astype(str)
        if col[2:4] == '60':
            penetrating_mask = df[col].astype(int) > 0
            df.loc[penetrating_mask, 'mechanism'] = 'penetrating'

    return df


def update_mechanism_from_mechmaj(df: pd.DataFrame) -> pd.DataFrame:
    """Update mechanism to penetrating if mechmaj columns indicate Cut/pierce or Firearm."""
    mechmaj_cols = [c for c in ['mechmaj1', 'mechmaj2', 'mechmaj3', 'mechmaj4'] if c in df.columns]
    if not mechmaj_cols:
        return df

    penetrating_mask = pd.Series(False, index=df.index)
    for col in mechmaj_cols:
        col_mask = df[col].astype(str).str.contains(
            'Cut/pierce|Firearm', case=False, na=False
        )
        penetrating_mask |= col_mask

    df.loc[penetrating_mask, 'mechanism'] = 'penetrating'
    n_updated = penetrating_mask.sum()
    if n_updated:
        logger.info(f"Updated {n_updated} rows to 'penetrating' from mechmaj columns")
    return df


# ============================================================================
# ISS COMPUTATION
# ============================================================================

def compute_iss_from_r(base: pd.DataFrame) -> None:
    """Generate ISS via R icdpicr package from ICD-10 diagnosis codes."""
    from astra.utils import is_file_present, ensure_parent_dir

    # Create long diagnosis df if not present
    long_path = "data/interim/diagnoses_long.csv"
    if not is_file_present(long_path):
        logger.info("Creating long diagnosis DataFrame for ISS computation...")
        _prepare_long_df(base)

    logger.info("Calling R script to compute ISS...")
    subprocess.call("Rscript astra/R/iss.r", shell=True)
    logger.info("R ISS computation finished")


def _prepare_long_df(base: pd.DataFrame) -> None:
    """Create long-format diagnosis DataFrame for R ISS computation."""
    from astra.utils import ensure_parent_dir

    diag = pd.read_csv("data/raw/Diagnoser.csv")
    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"])

    merged_df = base[["CPR_hash", "PID", "AGE", "start", "end"]].merge(
        diag, on="CPR_hash", how="left"
    )
    filtered_df = merged_df[
        (merged_df["Noteret_dato"] >= merged_df["start"] - pd.DateOffset(days=1))
        & (merged_df["Noteret_dato"] <= merged_df["end"] + pd.DateOffset(days=1))
    ]
    filtered_df["Diagnosekode"] = filtered_df["Diagnosekode"].str.slice(1, -1)

    def enumerate_diagnoses(group):
        diagnoses = group["Diagnosekode"].tolist()
        for i, diag_code in enumerate(diagnoses, start=1):
            group[f"ICD10_{i}"] = diag_code
        return group

    result_df = filtered_df.groupby("PID").apply(enumerate_diagnoses)
    result_df = result_df.drop_duplicates(subset="PID").reset_index(drop=True)

    ensure_parent_dir("data/interim/diagnoses_long.csv")
    result_df.to_csv("data/interim/diagnoses_long.csv")
    logger.info(f"Saved long diagnosis df: {len(result_df)} patients")


def add_iss_to_df(base: pd.DataFrame) -> pd.DataFrame:
    """Add ISS columns to base_df from ISS_notes and ISS_computed concept pickles
    and R-computed auxiliary columns (maxais, niss, mechmaj).

    For TRISS evaluation, take the max ISS per patient across both sources.
    """
    from astra.utils import is_file_present

    # Load both ISS sources and take max per PID for TRISS evaluation
    iss_frames = []
    for pkl_name in ("ISS_notes.pkl", "ISS_computed.pkl"):
        pkl_path = f"data/interim/concepts/{pkl_name}"
        if os.path.exists(pkl_path):
            iss_frames.append(pd.read_pickle(pkl_path))
    if iss_frames:
        iss_combined = pd.concat(iss_frames, ignore_index=True)
        iss_seq = (
            iss_combined.groupby("PID")["VALUE"]
            .max()
            .reset_index()
            .rename(columns={"VALUE": "riss"})
        )
        base = base.merge(iss_seq, how="left", on="PID")
        logger.info(f"Merged ISS (notes + R-computed): {len(iss_seq)} patients")
    else:
        base["riss"] = np.nan
        logger.warning("ISS_notes.pkl and ISS_computed.pkl not found — no ISS data available")

    # Auxiliary R-computed columns (maxais, niss, mechmaj) for TRISS mechanism
    iss_r_path = "data/interim/computed_iss_df.csv"
    if not is_file_present(iss_r_path):
        try:
            compute_iss_from_r(base)
        except Exception as e:
            logger.warning(f"ISS R computation failed: {e}")

    aux_cols = ["maxais", "niss", "mechmaj1", "mechmaj2", "mechmaj3", "mechmaj4"]
    if is_file_present(iss_r_path):
        iss_r = pd.read_csv(iss_r_path, low_memory=False)
        available_cols = [c for c in aux_cols if c in iss_r.columns]
        if available_cols:
            base = base.merge(iss_r[["PID"] + available_cols], how="left", on="PID")
            for col in available_cols:
                base[col] = base[col].replace("None", np.nan)
            logger.info(f"Merged R auxiliary columns: {available_cols}")

    return base


# ============================================================================
# DATA ASSEMBLY — MAIN ENTRY POINT
# ============================================================================

def build_trauma_score_df(data: dict, cfg: dict) -> pd.DataFrame:
    """Build DataFrame with trauma risk scores (RTS, ISS, TRISS) for holdout patients.

    Args:
        data: Data dict from prepare_data_and_dls (contains holdout TSDS).
        cfg: Configuration dict.

    Returns:
        DataFrame indexed by PID with columns: RTS, RTSc, ISS, ISS_COMPOUND,
        mechanism, TRISS_survival_prob, TRISS_mors_prob, plus first vitals.
    """
    from astra.data.datasets import AggregatedDS

    holdout_base = data["holdout"].base.copy()
    logger.info(f"Building trauma scores for {len(holdout_base)} holdout patients...")

    # Step 1: Extract first vitals via AggregatedDS
    logger.info("Extracting first vitals via AggregatedDS...")
    ads = AggregatedDS(
        cfg, holdout_base,
        agg_funcs=['first'],
        concepts=['VitaleVaerdier', 'ITAOversigtsrapport'],
        use_gpu=False,
    )
    vitals_df = ads.final_df[['PID']].copy()

    # Map expected column names from AggregatedDS output
    col_map = {
        'GCS_ITAOversigtsrapport_first': 'first_GCS',
        'SBP_VitaleVaerdier_first': 'first_SBP',
        'RESPIRATORYRATE_VitaleVaerdier_first': 'first_RR',
    }
    for ads_col, target_col in col_map.items():
        if ads_col in ads.final_df.columns:
            vitals_df[target_col] = pd.to_numeric(ads.final_df[ads_col], errors='coerce')
            # Replace 0 with NaN (AggregatedDS fills NaN with 0)
            vitals_df.loc[vitals_df[target_col] == 0.0, target_col] = np.nan
        else:
            vitals_df[target_col] = np.nan
            logger.warning(f"Column {ads_col} not found in AggregatedDS output")

    # Step 2: Compute RTS
    vitals_df = compute_rts(vitals_df, gcs_col='first_GCS', sbp_col='first_SBP', rr_col='first_RR')

    # Step 3: Add ISS from R + notes
    logger.info("Loading/computing ISS...")
    holdout_with_iss = add_iss_to_df(holdout_base)

    # Step 4: DTR matching for ISS_DTR and mechanism
    dtr_df = load_dtr_and_match(holdout_base)
    if dtr_df is not None and 'ISS' in dtr_df.columns:
        holdout_with_iss = holdout_with_iss.merge(
            dtr_df[['PID', 'ISS']].rename(columns={'ISS': 'ISS_DTR'}),
            on='PID', how='left',
        )
        holdout_with_iss['ISS_DTR'] = pd.to_numeric(holdout_with_iss['ISS_DTR'], errors='coerce')
    else:
        holdout_with_iss['ISS_DTR'] = np.nan

    # Step 5: Compound ISS
    holdout_with_iss = compound_iss(holdout_with_iss)

    # Step 6: Mechanism
    if dtr_df is not None and 'mechanism' in dtr_df.columns:
        holdout_with_iss = holdout_with_iss.merge(
            dtr_df[['PID', 'mechanism']], on='PID', how='left',
        )
    else:
        holdout_with_iss['mechanism'] = np.nan

    # Update mechanism from mechmaj columns if available
    holdout_with_iss = update_mechanism_from_mechmaj(holdout_with_iss)

    # Step 7: Merge everything into one df
    score_cols = ['PID', 'AGE']
    for col in ['riss', 'niss', 'ISS_DTR', 'ISS_COMPOUND', 'mechanism',
                 'mechmaj1', 'mechmaj2', 'mechmaj3', 'mechmaj4']:
        if col in holdout_with_iss.columns:
            score_cols.append(col)

    result = vitals_df.merge(
        holdout_with_iss[list(set(score_cols))],
        on='PID', how='left',
    )

    # Step 8: Compute TRISS where possible
    if 'mechanism' in result.columns and 'ISS_COMPOUND' in result.columns:
        result = compute_triss(
            result,
            iss_col='ISS_COMPOUND',
            age_col='AGE',
            mechanism_col='mechanism',
            rts_col='RTS',
        )

    # Log coverage stats
    _log_coverage(result)

    return result


def _log_coverage(df: pd.DataFrame) -> None:
    """Log coverage statistics for trauma scores."""
    n = len(df)
    for col, label in [
        ('first_GCS', 'GCS'), ('first_SBP', 'SBP'), ('first_RR', 'RR'),
        ('RTS', 'RTS'), ('ISS_COMPOUND', 'ISS'), ('TRISS_mors_prob', 'TRISS'),
    ]:
        if col in df.columns:
            valid = df[col].notna().sum()
            logger.info(f"  {label}: {valid}/{n} ({100*valid/n:.1f}%) patients")

    if 'mechanism' in df.columns:
        mech_counts = df['mechanism'].value_counts(dropna=False)
        logger.info(f"  Mechanism distribution: {mech_counts.to_dict()}")


# ============================================================================
# EVALUATION OF STATIC SCORES
# ============================================================================

def evaluate_static_scores(
    scores_df: pd.DataFrame,
    holdout_y: np.ndarray,
    holdout_pids: np.ndarray,
) -> Dict[str, dict]:
    """Compute AUROC/AUPRC with CIs for each available static score.

    Args:
        scores_df: DataFrame with PID + score columns from build_trauma_score_df().
        holdout_y: Binary target array for holdout patients.
        holdout_pids: PID array for holdout patients (same order as holdout_y).

    Returns:
        Dict mapping score name to {auroc, auroc_ci, auprc, auprc_ci, n}.
    """
    # Align scores with holdout order
    pid_to_idx = {pid: i for i, pid in enumerate(holdout_pids)}
    scores_aligned = scores_df.set_index('PID').reindex(holdout_pids).reset_index()

    score_configs = [
        ('RTS', 'RTS', True),          # negate: higher RTS = better prognosis
        ('ISS_COMPOUND', 'ISS', False), # higher ISS = worse → correlates with mortality
        ('TRISS_mors_prob', 'TRISS', False),  # already probability of death
    ]

    results = {}
    for col, label, negate in score_configs:
        if col not in scores_aligned.columns:
            continue

        valid_mask = scores_aligned[col].notna().values & ~np.isnan(holdout_y)
        n_valid = valid_mask.sum()
        if n_valid < 10:
            logger.warning(f"Skipping {label}: only {n_valid} valid cases")
            continue

        y_true = np.array(holdout_y)[valid_mask]
        y_score = scores_aligned[col].values[valid_mask].astype(float)

        if negate:
            y_score = -y_score

        if len(np.unique(y_true)) < 2:
            logger.warning(f"Skipping {label}: only one class in valid subset")
            continue

        try:
            auroc, auroc_lo, auroc_hi = calculate_roc_auc_ci(y_true, y_score)
            auprc, auprc_lo, auprc_hi = calculate_average_precision_ci(y_true, y_score)
        except Exception as e:
            logger.warning(f"Error computing metrics for {label}: {e}")
            continue

        results[label] = {
            'auroc': auroc,
            'auroc_ci': (auroc_lo, auroc_hi),
            'auprc': auprc,
            'auprc_ci': (auprc_lo, auprc_hi),
            'n': int(n_valid),
        }
        logger.info(
            f"  {label}: AUROC={auroc:.3f} [{auroc_lo:.3f}-{auroc_hi:.3f}], "
            f"AUPRC={auprc:.3f} [{auprc_lo:.3f}-{auprc_hi:.3f}] (n={n_valid})"
        )

    return results


def evaluate_static_scores_over_time(
    scores_df: pd.DataFrame,
    preds_df_active: pd.DataFrame,
    holdout_y: np.ndarray,
    holdout_pids: np.ndarray,
    valid_pids: Optional[np.ndarray] = None,
    delong: bool = False,
) -> Dict[str, Dict[str, "List[TimeMetricResult]"]]:
    """Compute score and model AUROC/AUPRC at each censor step on identical patients.

    At each censor step, for each score: identifies patients with that score available,
    then computes BOTH the score's metric AND the HNN model's metric on that exact
    same patient set. This ensures a fair apples-to-apples comparison.

    Args:
        scores_df: DataFrame with PID + score columns from build_trauma_score_df().
        preds_df_active: Active-only predictions DataFrame with [PID, censor_step, pred].
        holdout_y: Full holdout binary targets.
        holdout_pids: Full holdout PIDs (same order as holdout_y).
        valid_pids: Optional further filter (e.g., patients with RTS scores).
        delong: If True, run paired DeLong test at each timestep (HNN vs score)
                and apply Benjamini-Hochberg FDR correction.

    Returns:
        Dict mapping score label to {"score": List[TimeMetricResult],
            "model": List[TimeMetricResult], "counts": List[TimeMetricResult]}.
        When delong=True, also includes "delong_p": list of raw p-values,
            "delong_p_adj": FDR-adjusted p-values, "delong_z": z-statistics,
            "delong_hours": corresponding time in hours, and
            "delong_significant": boolean mask of significance at alpha=0.05.
    """
    from astra.evaluation.predictive_performance import TimeMetricResult

    pid_to_y = dict(zip(holdout_pids, holdout_y))
    scores_by_pid = scores_df.set_index('PID')

    score_configs = [
        ('RTS', 'RTS', True),
        ('ISS_COMPOUND', 'ISS', False),
        ('TRISS_mors_prob', 'TRISS', False),
    ]

    # Pre-filter valid_pids
    if valid_pids is not None:
        valid_set = set(valid_pids)
        preds_filtered = preds_df_active[preds_df_active['PID'].isin(valid_set)]
    else:
        preds_filtered = preds_df_active

    # Index model predictions by (PID, censor_step) for fast lookup
    preds_indexed = preds_filtered.set_index(['PID', 'censor_step'])['pred']

    all_results = {}
    for col, label, negate in score_configs:
        if col not in scores_by_pid.columns:
            continue

        score_results = []
        model_results = []
        count_results = []
        delong_raw_p = []
        delong_z_vals = []
        delong_se_vals = []
        delong_delta_vals = []
        delong_hours = []
        for step, group in preds_filtered.groupby('censor_step'):
            # Active PIDs at this step (intersection with score availability)
            active_pids = group['PID'].values
            score_vals = scores_by_pid.reindex(active_pids)[col]
            has_score = score_vals.notna()

            pids_with_score = active_pids[has_score.values]

            time_min = step_to_time(step)
            if time_min is None:
                continue

            y_true = np.array([pid_to_y[pid] for pid in pids_with_score])
            n_samples = len(pids_with_score)
            n_positive = int(y_true.sum())

            time_kwargs = dict(
                time_min=time_min,
                time_hours=time_min / 60.0,
                time_days=time_min / (60.0 * 24.0),
                censor_step=int(step),
                n_samples=n_samples,
                n_positive=n_positive,
            )

            # Always record counts (no skipping — for bottom panels)
            count_results.append(TimeMetricResult(
                **time_kwargs,
                auroc=float('nan'),
                auroc_ci=(float('nan'), float('nan')),
                auprc=float('nan'),
                auprc_ci=(float('nan'), float('nan')),
            ))

            # Skip metric computation if insufficient data
            if n_samples < 10 or len(np.unique(y_true)) < 2:
                continue

            y_score = score_vals[has_score].values.astype(float)
            if negate:
                y_score = -y_score

            # Get model predictions for the SAME patients
            model_preds = np.array([
                preds_indexed.loc[(pid, step)]
                for pid in pids_with_score
            ])

            try:
                auroc, auroc_lo, auroc_hi = calculate_roc_auc_ci(y_true, y_score)
                auprc, auprc_lo, auprc_hi = calculate_average_precision_ci(
                    y_true, y_score, n_bootstraps=200
                )
                m_auroc, m_auroc_lo, m_auroc_hi = calculate_roc_auc_ci(
                    y_true, model_preds
                )
                m_auprc, m_auprc_lo, m_auprc_hi = calculate_average_precision_ci(
                    y_true, model_preds, n_bootstraps=200
                )
            except Exception:
                continue

            score_results.append(TimeMetricResult(
                **time_kwargs,
                auroc=auroc,
                auroc_ci=(auroc_lo, auroc_hi),
                auprc=auprc,
                auprc_ci=(auprc_lo, auprc_hi),
            ))

            model_results.append(TimeMetricResult(
                **time_kwargs,
                auroc=m_auroc,
                auroc_ci=(m_auroc_lo, m_auroc_hi),
                auprc=m_auprc,
                auprc_ci=(m_auprc_lo, m_auprc_hi),
            ))

            # DeLong paired test (model vs score AUROC)
            if delong:
                try:
                    z, p, se = delong_test_paired(y_true, model_preds, y_score)
                    delong_z_vals.append(z)
                    delong_raw_p.append(p)
                    delong_se_vals.append(se)
                    delong_delta_vals.append(m_auroc - auroc)
                    delong_hours.append(time_min / 60.0)
                except Exception:
                    delong_z_vals.append(0.0)
                    delong_raw_p.append(1.0)
                    delong_se_vals.append(0.0)
                    delong_delta_vals.append(0.0)
                    delong_hours.append(time_min / 60.0)

        if score_results:
            result_entry = {
                "score": score_results,
                "model": model_results,
                "counts": count_results,
            }

            # Apply FDR correction across all timesteps for this score
            if delong and delong_raw_p:
                p_arr = np.array(delong_raw_p)
                rejected, p_adj = benjamini_hochberg(p_arr, alpha=0.05)
                result_entry["delong_p"] = delong_raw_p
                result_entry["delong_p_adj"] = p_adj.tolist()
                result_entry["delong_z"] = delong_z_vals
                result_entry["delong_se"] = delong_se_vals
                result_entry["delong_delta"] = delong_delta_vals
                result_entry["delong_hours"] = delong_hours
                result_entry["delong_significant"] = rejected.tolist()

                n_sig = int(rejected.sum())
                logger.info(
                    f"  {label}: DeLong test at {len(delong_raw_p)} time points, "
                    f"{n_sig}/{len(delong_raw_p)} significant (FDR<0.05)"
                )

            all_results[label] = result_entry
            logger.info(f"  {label}: evaluated at {len(score_results)} time points")

    return all_results


# ============================================================================
# POST-HOC FILTERED EVALUATION
# ============================================================================

def recompute_metrics_for_subset(
    preds_df: pd.DataFrame,
    holdout_y: np.ndarray,
    holdout_pids: np.ndarray,
    valid_pids: np.ndarray,
) -> "List[TimeMetricResult]":
    """Recompute AUROC/AUPRC from existing predictions filtered to a patient subset.

    Args:
        preds_df: DataFrame with columns [PID, censor_step, pred] from evaluate_over_time.
        holdout_y: Full holdout binary targets.
        holdout_pids: Full holdout PIDs (same order as holdout_y).
        valid_pids: PIDs to keep for filtered evaluation.

    Returns:
        List of TimeMetricResult for each censor_step in the filtered subset.
    """
    from astra.evaluation.predictive_performance import TimeMetricResult

    # Build PID → y mapping
    pid_to_y = dict(zip(holdout_pids, holdout_y))

    # Filter predictions to valid PIDs
    valid_set = set(valid_pids)
    preds_filtered = preds_df[preds_df['PID'].isin(valid_set)].copy()

    if len(preds_filtered) == 0:
        logger.warning("No predictions found for valid PIDs")
        return []

    results = []
    for step, group in preds_filtered.groupby('censor_step'):
        pids = group['PID'].values
        y_pred = group['pred'].values
        y_true = np.array([pid_to_y[pid] for pid in pids])

        n_samples = len(y_true)
        n_positive = int(y_true.sum())

        if len(np.unique(y_true)) < 2 or n_samples < 10:
            continue

        time_min = step_to_time(step)
        if time_min is None:
            continue

        try:
            auroc, auroc_lo, auroc_hi = calculate_roc_auc_ci(y_true, y_pred)
            auprc, auprc_lo, auprc_hi = calculate_average_precision_ci(y_true, y_pred)
        except Exception:
            continue

        results.append(TimeMetricResult(
            time_min=time_min,
            time_hours=time_min / 60.0,
            time_days=time_min / (60.0 * 24.0),
            censor_step=int(step),
            auroc=auroc,
            auroc_ci=(auroc_lo, auroc_hi),
            auprc=auprc,
            auprc_ci=(auprc_lo, auprc_hi),
            n_samples=n_samples,
            n_positive=n_positive,
        ))

    logger.info(f"Recomputed metrics for {len(results)} time points "
                f"on {len(valid_pids)} patients")
    return results


def compute_counts_for_subset(
    preds_df: pd.DataFrame,
    holdout_y: np.ndarray,
    holdout_pids: np.ndarray,
    valid_pids: np.ndarray,
) -> "List[TimeMetricResult]":
    """Compute patient counts at each censor step for a filtered subset.

    Unlike recompute_metrics_for_subset, this does NOT skip steps with
    insufficient data — it returns entries for ALL censor steps with NaN
    metrics but valid n_samples/n_positive. Used for the count panels.
    """
    from astra.evaluation.predictive_performance import TimeMetricResult

    pid_to_y = dict(zip(holdout_pids, holdout_y))
    valid_set = set(valid_pids)
    preds_filtered = preds_df[preds_df['PID'].isin(valid_set)]

    results = []
    for step, group in preds_filtered.groupby('censor_step'):
        pids = group['PID'].values
        y_true = np.array([pid_to_y[pid] for pid in pids])
        n_samples = len(y_true)
        n_positive = int(y_true.sum())

        time_min = step_to_time(step)
        if time_min is None:
            continue

        results.append(TimeMetricResult(
            time_min=time_min,
            time_hours=time_min / 60.0,
            time_days=time_min / (60.0 * 24.0),
            censor_step=int(step),
            auroc=float('nan'),
            auroc_ci=(float('nan'), float('nan')),
            auprc=float('nan'),
            auprc_ci=(float('nan'), float('nan')),
            n_samples=n_samples,
            n_positive=n_positive,
        ))

    return results
