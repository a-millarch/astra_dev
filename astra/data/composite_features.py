"""Composite medication feature computation.

Derives 12 dense, clinically grounded medication channels from raw
medication records per time bin.  Replaces the 46 per-category
tier/binary features with cross-category composite tiers that capture
clinical intensity patterns.

Toggle: ``composite_mode: true`` under Medicin in profiles.yaml.
When false, the old per-category 46-feature path is used.
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =====================================================================
# Canonical feature names (used across batch and inference pipelines)
# =====================================================================

COMPOSITE_FEATURE_NAMES = [
    "abx_tier",               # 0-6  max antibiotic escalation
    "abx_n_concurrent",       # 0+   distinct J01 codes per bin
    "hemodynamic_tier",       # 0-3  vasopressor/hemodynamic intensity
    "sedation_tier",          # 0-5  sedation + NMBA + delirium
    "coagulation_tier",       # 0-4  anticoagulation + hemostatic mgmt
    "organ_support_tier",     # 0-4  diuretics + insulin + lytes + nutrition
    "opioid_tier",            # 0-3  opioid intensity
    "surgical_tier",          # 0-3  surgical/procedural markers
    "acute_deterioration",    # 0-2  reversal agent events
    "comorbidity_med_count",  # 0-5  chronic medication classes
    "n_active_dimensions",    # 0-9  count of non-zero features 1,3-10
    "max_severity_signal",    # 0-1  normalised max severity
]

# =====================================================================
# ATC code sets — signal group membership
# =====================================================================

# -- Hemodynamic --
PERIOP_VASOPRESSOR = frozenset({"C01CA26", "C01CA06"})
ICU_VASOPRESSOR = frozenset({
    "C01CA03", "C01CA24", "C01CA04", "C01CA07", "C01CA02",
    "H01BA01", "H01BA04",
})
AMIODARONE = frozenset({"C01BD01"})

# -- Sedation --
WARD_SEDATION = frozenset({
    "N05CH01", "N05CF01", "N05BA04", "N05BA01", "N05BA02", "C02AC01",
})
ICU_LIGHT_SEDATION = frozenset({
    "N05CM18", "N05AD01", "N05AH03", "N05AH04",
})
PROPOFOL = frozenset({"N01AX10"})
DEEP_BENZO = frozenset({"N05CD08", "N05BA06"})
ETOMIDATE = frozenset({"N01AX07"})
NMBA_ALL = frozenset({"M03AC09", "M03AC11", "M03AB01", "M03AC03"})

# -- Coagulation --
LMWH_TINZAPARIN = frozenset({"B01AB10"})
LMWH_DALTEPARIN = frozenset({"B01AB04"})
LMWH_ENOXAPARIN = frozenset({"B01AB05"})
LMWH_ALL = LMWH_TINZAPARIN | LMWH_DALTEPARIN | LMWH_ENOXAPARIN
UFH = frozenset({"B01AB01"})
DOAC = frozenset({"B01AF02", "B01AF01", "B01AE07"})
VKA = frozenset({"B01AA03"})
ANTIPLATELET = frozenset({"B01AC06", "B01AC04", "B01AC24", "B01AC22"})
TXA = frozenset({"B02AA02"})
FIBRINOGEN = frozenset({"B02BB01"})
PCC = frozenset({"B02BD01"})
VITAMIN_K = frozenset({"B02BA01"})
DESMOPRESSIN = frozenset({"H01BA02"})
PROTAMINE = frozenset({"V03AB14"})
RFVIIA = frozenset({"B02BD08"})
# Countable hemostatics for tier 4 check (≥2 concurrent)
HEMOSTATIC_COUNTABLE = FIBRINOGEN | PCC | VITAMIN_K | DESMOPRESSIN | PROTAMINE

# -- Organ support --
LOOP_DIURETIC = frozenset({"C03CA01", "C03CA02"})
METOLAZONE = frozenset({"C03BA08"})
ALBUMIN = frozenset({"B05AA01"})
BICARBONATE = frozenset({"B05XA02"})
CALCIUM = frozenset({"B05XA03"})
POTASSIUM = frozenset({"B05XA01"})
MAGNESIUM = frozenset({"B05XA05", "B05XA06"})
ELECTROLYTE_ALL = CALCIUM | POTASSIUM | MAGNESIUM
RAPID_INSULIN = frozenset({"A10AB01", "A10AB05"})
# Prefix-matched groups (checked via str.startswith)
ORAL_DIURETIC_PREFIXES = ("C03AA", "C03AB", "C03DA")
CHRONIC_INSULIN_PREFIXES = ("A10AE", "A10AC")
TPN_PREFIXES = ("B05BA",)

# -- Opioid --
MILD_OPIOID = frozenset({"N02AX02", "N02AA59", "N02AX06"})
STRONG_OPIOID = frozenset({
    "N02AA01", "N02AA05", "N02AB01", "N02AG01", "N02AG02", "N02AB02",
})
ICU_OPIOID_PREFIX = "N01AH"

# -- Surgical --
ESKETAMINE = frozenset({"N01AX14", "N01AX03"})
THIOPENTAL = frozenset({"N01AF03"})
VOLATILE_PREFIX = "N01AB"
REGIONAL_PREFIX = "N01BB"
ANESTHETIC_OPIOID = frozenset({"N01AH06", "N01AH02", "N01AH03"})

# -- Acute deterioration --
NALOXONE = frozenset({"V03AB15"})
FLUMAZENIL = frozenset({"V03AB25"})
ACETYLCYSTEINE = frozenset({"V03AB23"})
REVERSAL_AGENTS = NALOXONE | FLUMAZENIL | ACETYLCYSTEINE

# -- Comorbidity --
BETA_BLOCKER_PREFIX = "C07"
CV_COMORBIDITY = frozenset({"C01AA05", "C01DA14", "C02AB01", "C01CA17"})
ANTIPLATELET_BROAD_PREFIX = "B01AC"
ORAL_ANTICOAG_VKA_PREFIX = "B01AA"
ORAL_ANTICOAG_DOAC_PREFIX = "B01AF"
ORAL_ANTICOAG_DABIGATRAN = frozenset({"B01AE07"})

# =====================================================================
# Dose / unit classification
# =====================================================================

INFUSION_UNITS = frozenset({
    "mg/kg/t", "ml/time", "mg/time", "mg/t.", "mg/t",
    "mikrog/kg/min", "mg/kg/min", "mg/min", "mg/kg/time",
    "ie/time", "ie/t", "enhed/t", "ie/kg/time",
    "ml/t", "ml/t.", "mcg/kg/min", "mikrog/min",
    "mg/kg/t.", "ml/t./time",
})

# LMWH prophylactic dose ceilings
_LMWH_THRESHOLDS = {
    "B01AB10": 4500,   # tinzaparin IU
    "B01AB04": 5000,   # dalteparin IU
    "B01AB05": 40,     # enoxaparin mg
}


def _lower_strip(series: pd.Series) -> pd.Series:
    """Lowercase + strip a string series, NaN-safe."""
    return series.astype(str).str.strip().str.lower().where(series.notna(), other=np.nan)


# =====================================================================
# Step 1: Tag every record with boolean signal membership
# =====================================================================

def _tag_records(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean signal columns to each medication record.

    Expects columns: ATC (str), _bin_position (int).
    Optional columns: Administrationsdosis, Dosisenhed.
    """
    atc = df["ATC"]
    dose_col = "Administrationsdosis"
    unit_col = "Dosisenhed"

    has_dose = dose_col in df.columns
    has_unit = unit_col in df.columns

    # Pre-compute dose/unit helpers
    if has_unit:
        unit_lower = _lower_strip(df[unit_col])
        is_infusion = unit_lower.isin(INFUSION_UNITS)
        is_bolus_mg = unit_lower == "mg"
    else:
        is_infusion = pd.Series(False, index=df.index)
        is_bolus_mg = pd.Series(False, index=df.index)

    if has_dose:
        dose_numeric = pd.to_numeric(df[dose_col], errors="coerce")
    else:
        dose_numeric = pd.Series(np.nan, index=df.index)

    # --- Antibiotics ---
    df["_j01"] = atc.str.startswith("J01", na=False)
    # Exclude low-dose erythromycin prokinetic use (≤250 mg)
    erythro_prokinetic = (
        (atc == "J01FA01")
        & (dose_numeric <= 250)
        & (is_bolus_mg | (unit_lower == "mg") if has_unit else False)
    )
    df.loc[erythro_prokinetic, "_j01"] = False

    # Classify antibiotic tiers via registered mapping
    from astra.data.tier_mappings import get_mapping
    abx_map = get_mapping("antibiotic_escalation")
    df["_abx_tier_val"] = np.nan
    df.loc[df["_j01"], "_abx_tier_val"] = (
        df.loc[df["_j01"], "ATC"].map(abx_map.classify)
    )

    # --- Hemodynamic ---
    df["_periop_vaso"] = atc.isin(PERIOP_VASOPRESSOR)
    df["_icu_vaso"] = atc.isin(ICU_VASOPRESSOR)
    df["_amiodarone"] = atc.isin(AMIODARONE)

    # --- Sedation ---
    df["_ward_sed"] = atc.isin(WARD_SEDATION)
    df["_icu_light_sed"] = atc.isin(ICU_LIGHT_SEDATION)
    df["_propofol"] = atc.isin(PROPOFOL)
    df["_deep_benzo"] = atc.isin(DEEP_BENZO)
    df["_etomidate"] = atc.isin(ETOMIDATE)
    df["_nmba"] = atc.isin(NMBA_ALL)

    # --- Coagulation ---
    is_lmwh = atc.isin(LMWH_ALL)
    # Therapeutic LMWH: dose above prophylactic ceiling
    therapeutic_lmwh = pd.Series(False, index=df.index)
    for code, threshold in _LMWH_THRESHOLDS.items():
        therapeutic_lmwh |= (atc == code) & (dose_numeric > threshold)
    df["_prophylactic_lmwh"] = is_lmwh & ~therapeutic_lmwh
    df["_therapeutic_lmwh"] = is_lmwh & therapeutic_lmwh
    df["_iv_heparin"] = atc.isin(UFH) & is_infusion
    df["_doac"] = atc.isin(DOAC)
    df["_vka"] = atc.isin(VKA)
    df["_antiplatelet"] = atc.isin(ANTIPLATELET)
    df["_txa"] = atc.isin(TXA)
    df["_fibrinogen"] = atc.isin(FIBRINOGEN)
    df["_pcc"] = atc.isin(PCC)
    df["_vitamin_k"] = atc.isin(VITAMIN_K)
    df["_desmopressin"] = atc.isin(DESMOPRESSIN)
    df["_protamine"] = atc.isin(PROTAMINE)
    df["_rfviia"] = atc.isin(RFVIIA)
    df["_hemostatic_countable"] = atc.isin(HEMOSTATIC_COUNTABLE)

    # Therapeutic anticoag: any of {therapeutic LMWH, iv heparin, DOAC, VKA}
    df["_therapeutic_anticoag"] = (
        df["_therapeutic_lmwh"] | df["_iv_heparin"]
        | df["_doac"] | df["_vka"]
    )

    # --- Organ support ---
    df["_loop_diuretic"] = atc.isin(LOOP_DIURETIC)
    df["_continuous_loop"] = df["_loop_diuretic"] & is_infusion
    df["_oral_diuretic"] = atc.str.startswith(ORAL_DIURETIC_PREFIXES, na=False)
    df["_iv_insulin"] = atc.isin(RAPID_INSULIN) & is_infusion
    df["_sc_insulin"] = atc.str.startswith(CHRONIC_INSULIN_PREFIXES, na=False)
    df["_electrolyte"] = atc.isin(ELECTROLYTE_ALL)
    df["_albumin"] = atc.isin(ALBUMIN)
    df["_tpn"] = atc.str.startswith(TPN_PREFIXES, na=False)
    df["_bicarbonate"] = atc.isin(BICARBONATE)
    df["_metolazone"] = atc.isin(METOLAZONE)

    # --- Opioid ---
    df["_mild_opioid"] = atc.isin(MILD_OPIOID)
    df["_strong_opioid"] = atc.isin(STRONG_OPIOID)
    df["_icu_opioid"] = atc.str.startswith(ICU_OPIOID_PREFIX, na=False)

    # --- Surgical ---
    df["_regional"] = atc.str.startswith(REGIONAL_PREFIX, na=False)
    df["_esketamine"] = atc.isin(ESKETAMINE)
    df["_thiopental"] = atc.isin(THIOPENTAL)
    df["_volatile"] = atc.str.startswith(VOLATILE_PREFIX, na=False)
    df["_anesthetic_opioid"] = atc.isin(ANESTHETIC_OPIOID)

    # --- Acute deterioration ---
    df["_naloxone"] = atc.isin(NALOXONE)
    df["_flumazenil"] = atc.isin(FLUMAZENIL)
    df["_acetylcysteine"] = atc.isin(ACETYLCYSTEINE)
    df["_reversal"] = atc.isin(REVERSAL_AGENTS)

    # --- Comorbidity ---
    df["_beta_blocker"] = atc.str.startswith(BETA_BLOCKER_PREFIX, na=False)
    df["_cv_comorbidity"] = atc.isin(CV_COMORBIDITY)
    df["_antiplatelet_broad"] = atc.str.startswith(
        ANTIPLATELET_BROAD_PREFIX, na=False
    )
    df["_oral_anticoag_broad"] = (
        atc.str.startswith(ORAL_ANTICOAG_VKA_PREFIX, na=False)
        | atc.str.startswith(ORAL_ANTICOAG_DOAC_PREFIX, na=False)
        | atc.isin(ORAL_ANTICOAG_DABIGATRAN)
    )

    # --- Count helper columns (ATC when in group, else NaN) ---
    df["_j01_atc"] = atc.where(df["_j01"])
    df["_icu_vaso_atc"] = atc.where(df["_icu_vaso"])
    df["_antiplatelet_atc"] = atc.where(df["_antiplatelet"])
    df["_electrolyte_atc"] = atc.where(df["_electrolyte"])
    df["_hemostatic_count_atc"] = atc.where(df["_hemostatic_countable"])
    df["_reversal_atc"] = atc.where(df["_reversal"])
    df["_strong_icu_atc"] = atc.where(df["_strong_opioid"] | df["_icu_opioid"])

    return df


# =====================================================================
# Step 2: Aggregate signals per (PID, bin_position)
# =====================================================================

def _aggregate_signals(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tagged records into per-(PID, bin) signal summary."""
    g = tagged_df.groupby(["PID", "_bin_position"], sort=False)

    agg_spec = {
        # Boolean "any" signals
        "_periop_vaso": "any",
        "_icu_vaso": "any",
        "_amiodarone": "any",
        "_ward_sed": "any",
        "_icu_light_sed": "any",
        "_propofol": "any",
        "_deep_benzo": "any",
        "_etomidate": "any",
        "_nmba": "any",
        "_prophylactic_lmwh": "any",
        "_therapeutic_anticoag": "any",
        "_txa": "any",
        "_fibrinogen": "any",
        "_pcc": "any",
        "_vitamin_k": "any",
        "_desmopressin": "any",
        "_protamine": "any",
        "_rfviia": "any",
        "_loop_diuretic": "any",
        "_continuous_loop": "any",
        "_oral_diuretic": "any",
        "_iv_insulin": "any",
        "_sc_insulin": "any",
        "_albumin": "any",
        "_tpn": "any",
        "_bicarbonate": "any",
        "_metolazone": "any",
        "_mild_opioid": "any",
        "_strong_opioid": "any",
        "_icu_opioid": "any",
        "_regional": "any",
        "_esketamine": "any",
        "_thiopental": "any",
        "_volatile": "any",
        "_anesthetic_opioid": "any",
        "_naloxone": "any",
        "_flumazenil": "any",
        "_acetylcysteine": "any",
        "_beta_blocker": "any",
        "_cv_comorbidity": "any",
        "_sc_insulin": "any",
        "_antiplatelet_broad": "any",
        "_oral_anticoag_broad": "any",
        # Numeric aggregations
        "_abx_tier_val": "max",
        "_j01_atc": "nunique",
        "_icu_vaso_atc": "nunique",
        "_antiplatelet_atc": "nunique",
        "_electrolyte_atc": "nunique",
        "_hemostatic_count_atc": "nunique",
        "_reversal_atc": "nunique",
        "_strong_icu_atc": "nunique",
    }

    signals = g.agg(agg_spec)
    signals.columns = signals.columns.get_level_values(0)
    return signals


# =====================================================================
# Step 3: Derive 12 composite tiers (vectorised)
# =====================================================================

def _derive_composite_tiers(s: pd.DataFrame) -> pd.DataFrame:
    """Derive composite feature values from aggregated signal DataFrame.

    Operates in-place on *s* and returns it with 12 new columns matching
    ``COMPOSITE_FEATURE_NAMES``.
    """
    # ---- Feature 1: abx_tier ----
    s["abx_tier"] = s["_abx_tier_val"].fillna(0).astype(int)

    # ---- Feature 2: abx_n_concurrent ----
    s["abx_n_concurrent"] = s["_j01_atc"].fillna(0).astype(int)

    # ---- Feature 3: hemodynamic_tier ----
    s["hemodynamic_tier"] = 0
    s.loc[s["_periop_vaso"] & ~s["_icu_vaso"], "hemodynamic_tier"] = 1
    s.loc[s["_icu_vaso"], "hemodynamic_tier"] = 2
    s.loc[
        (s["_icu_vaso_atc"] >= 2)
        | (s["_icu_vaso"] & s["_amiodarone"]),
        "hemodynamic_tier",
    ] = 3

    # ---- Feature 4: sedation_tier ----
    # Co-occurrence-based: depth inferred from which agents share a bin
    s["sedation_tier"] = 0
    s.loc[s["_ward_sed"], "sedation_tier"] = 1
    s.loc[s["_icu_light_sed"], "sedation_tier"] = 2
    esketamine_alone = (
        s["_esketamine"] & ~s["_volatile"] & ~s["_nmba"]
        & ~s["_propofol"] & ~s["_anesthetic_opioid"]
    )
    s.loc[esketamine_alone, "sedation_tier"] = 2
    propofol_alone = (
        s["_propofol"] & ~s["_volatile"] & ~s["_nmba"]
        & ~s["_anesthetic_opioid"]
    )
    s.loc[propofol_alone | s["_deep_benzo"], "sedation_tier"] = 3
    tier4 = (
        s["_volatile"]
        | s["_thiopental"]
        | s["_etomidate"]
        | (s["_propofol"] & s["_nmba"])
        | (s["_propofol"] & s["_volatile"])
        | (s["_propofol"] & s["_anesthetic_opioid"])
        | (s["_esketamine"] & s["_nmba"])
        | (s["_esketamine"] & s["_volatile"])
        | (s["_esketamine"] & s["_propofol"])
        | (s["_esketamine"] & s["_anesthetic_opioid"])
    )
    s.loc[tier4, "sedation_tier"] = 4
    s.loc[tier4 & s["_nmba"], "sedation_tier"] = 5

    # ---- Feature 5: coagulation_tier ----
    s["coagulation_tier"] = 0
    # Tier 1: prophylactic LMWH, single antiplatelet, or TXA alone
    s.loc[
        s["_prophylactic_lmwh"]
        | ((s["_antiplatelet_atc"] >= 1) & (s["_antiplatelet_atc"] < 2))
        | s["_txa"],
        "coagulation_tier",
    ] = 1
    # Tier 2: therapeutic anticoag, DAPT, or IV heparin
    s.loc[
        s["_therapeutic_anticoag"]
        | (s["_antiplatelet_atc"] >= 2),
        "coagulation_tier",
    ] = 2
    # Tier 3: active hemorrhage management
    s.loc[
        s["_fibrinogen"] | s["_pcc"] | s["_vitamin_k"]
        | s["_desmopressin"] | s["_protamine"],
        "coagulation_tier",
    ] = 3
    # Tier 4: massive/refractory hemorrhage
    s.loc[
        s["_rfviia"] | (s["_hemostatic_count_atc"] >= 2),
        "coagulation_tier",
    ] = 4

    # ---- Feature 6: organ_support_tier ----
    os_multi_lyte = s["_electrolyte_atc"] >= 2
    os_sub_count = (
        s["_loop_diuretic"].astype(int)
        + s["_iv_insulin"].astype(int)
        + os_multi_lyte.astype(int)
        + s["_albumin"].astype(int)
        + s["_tpn"].astype(int)
        + s["_bicarbonate"].astype(int)
        + s["_metolazone"].astype(int)
        + s["_continuous_loop"].astype(int)
    )

    s["organ_support_tier"] = 0
    # Tier 1: oral comorbidity diuretics, SC insulin, or single electrolyte
    s.loc[
        s["_oral_diuretic"] | s["_sc_insulin"]
        | ((s["_electrolyte_atc"] == 1) & ~os_multi_lyte),
        "organ_support_tier",
    ] = 1
    # Tier 2: intermittent loop diuretic, iv insulin, or multi-electrolyte
    s.loc[
        s["_loop_diuretic"] | s["_iv_insulin"] | os_multi_lyte,
        "organ_support_tier",
    ] = 2
    # Tier 3: albumin/tpn/metolazone, or loop + one of {iv_insulin, multi_lyte, albumin}
    s.loc[
        s["_albumin"] | s["_tpn"] | s["_metolazone"]
        | (s["_loop_diuretic"] & (s["_iv_insulin"] | os_multi_lyte | s["_albumin"])),
        "organ_support_tier",
    ] = 3
    # Tier 4: continuous loop, bicarbonate, or ≥3 sub-signals
    s.loc[
        s["_continuous_loop"] | s["_bicarbonate"] | (os_sub_count >= 3),
        "organ_support_tier",
    ] = 4

    # ---- Feature 7: opioid_tier ----
    s["opioid_tier"] = 0
    s.loc[s["_mild_opioid"], "opioid_tier"] = 1
    s.loc[s["_strong_opioid"] | s["_icu_opioid"], "opioid_tier"] = 2
    s.loc[s["_strong_icu_atc"] >= 2, "opioid_tier"] = 3

    # ---- Feature 8: surgical_tier ----
    s["surgical_tier"] = 0
    s.loc[s["_regional"], "surgical_tier"] = 1
    surgical_ga = (
        s["_volatile"]
        | (s["_esketamine"] & s["_propofol"])
        | (s["_propofol"] & s["_anesthetic_opioid"])
    )
    s.loc[surgical_ga, "surgical_tier"] = 2
    s.loc[s["_thiopental"], "surgical_tier"] = 3

    # ---- Feature 9: acute_deterioration ----
    s["acute_deterioration"] = 0
    s.loc[s["_reversal_atc"] == 1, "acute_deterioration"] = 1
    s.loc[s["_reversal_atc"] >= 2, "acute_deterioration"] = 2

    # ---- Feature 10: comorbidity_med_count ----
    s["comorbidity_med_count"] = (
        s["_beta_blocker"].astype(int)
        + s["_sc_insulin"].astype(int)
        + s["_antiplatelet_broad"].astype(int)
        + s["_oral_anticoag_broad"].astype(int)
        + s["_cv_comorbidity"].astype(int)
    ).clip(upper=5)

    # ---- Feature 11: n_active_dimensions ----
    s["n_active_dimensions"] = (
        (s["abx_tier"] > 0).astype(int)
        + (s["hemodynamic_tier"] > 0).astype(int)
        + (s["sedation_tier"] > 0).astype(int)
        + (s["coagulation_tier"] > 0).astype(int)
        + (s["organ_support_tier"] > 0).astype(int)
        + (s["opioid_tier"] > 0).astype(int)
        + (s["surgical_tier"] > 0).astype(int)
        + (s["acute_deterioration"] > 0).astype(int)
        + (s["comorbidity_med_count"] > 0).astype(int)
    )

    # ---- Feature 12: max_severity_signal ----
    s["max_severity_signal"] = np.maximum.reduce([
        s["abx_tier"] / 6.0,
        s["hemodynamic_tier"] / 3.0,
        s["sedation_tier"] / 5.0,
        s["coagulation_tier"] / 4.0,
        s["organ_support_tier"] / 4.0,
        s["opioid_tier"] / 3.0,
        s["surgical_tier"] / 3.0,
        s["acute_deterioration"] / 2.0,
    ])

    return s


# =====================================================================
# Batch entry point
# =====================================================================

def compute_composite_features(
    binned_df: pd.DataFrame,
    base_df: pd.DataFrame,
    cfg: dict,
    ts_cols: List[str],
    base_pids: set,
) -> Optional[pd.DataFrame]:
    """Compute 12 composite medication features from pre-binned records.

    Returns a long-format DataFrame matching the output schema of
    ``CategoricalProfileEncoder.compute_tier_features()``:
    ``[PID, FEATURE, target, '0', '1', '2', ...]``

    Returns None if *binned_df* is empty.
    """
    if binned_df.empty:
        logger.info("No binned medication records — skipping composite features")
        return None

    target = cfg["target"]
    n_ts = len(ts_cols)
    sorted_pids = sorted(base_pids)
    target_map = base_df.set_index("PID")[target]

    # Deduplicate: one record per (PID, bin, ATC), keep highest dose
    df = binned_df.copy()
    if "Administrationsdosis" in df.columns:
        df["_dose_sort"] = pd.to_numeric(
            df["Administrationsdosis"], errors="coerce"
        ).fillna(0)
        df = df.sort_values("_dose_sort", ascending=False)
        df = df.drop_duplicates(
            subset=["PID", "_bin_position", "ATC"], keep="first"
        )
        df = df.drop(columns=["_dose_sort"])
    else:
        df = df.drop_duplicates(
            subset=["PID", "_bin_position", "ATC"], keep="first"
        )

    logger.info(
        f"Composite features: {len(df)} records after dedup, "
        f"{df['PID'].nunique()} patients"
    )

    # Tag → aggregate → derive
    df = _tag_records(df)
    signals = _aggregate_signals(df)
    signals = _derive_composite_tiers(signals)

    # Pivot each feature into wide format
    all_dfs = []
    for feat_name in COMPOSITE_FEATURE_NAMES:
        feat_values = signals[feat_name].reset_index()
        feat_values.columns = ["PID", "position", "value"]

        pivot = feat_values.pivot_table(
            index="PID", columns="position", values="value", aggfunc="first"
        )
        pivot = pivot.reindex(
            index=sorted_pids,
            columns=range(n_ts),
            fill_value=0.0,
        )
        pivot.columns = ts_cols

        row_df = pivot.reset_index()
        row_df.rename(columns={"index": "PID"}, inplace=True)
        row_df["FEATURE"] = feat_name
        row_df[target] = row_df["PID"].map(target_map).astype(int)
        all_dfs.append(row_df)

        n_nonzero_patients = (pivot.values > 0).any(axis=1).sum()
        logger.info(
            f"  {feat_name}: {n_nonzero_patients}/{len(sorted_pids)} patients "
            f"with non-zero ({100*n_nonzero_patients/len(sorted_pids):.1f}%)"
        )

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(["PID", "FEATURE"]).reset_index(drop=True)
    logger.info(
        f"Composite features complete: {len(result)} rows, "
        f"{len(COMPOSITE_FEATURE_NAMES)} features, {n_ts} timesteps"
    )
    return result


# =====================================================================
# Inference entry point
# =====================================================================

def compute_composite_features_for_patient(
    med_events: list,
    bin_df: pd.DataFrame,
    ts_channel_names: List[str],
) -> Dict[str, np.ndarray]:
    """Compute composite features for a single patient (inference).

    Args:
        med_events: List of medication event dicts. Each must have
            ``timestamp`` and ``atc_code``; optionally ``dose`` and
            ``unit``.
        bin_df: Patient's bin grid with ``bin_start``, ``bin_end``,
            ``position`` columns.
        ts_channel_names: Model's channel name list (to skip features
            the model doesn't use).

    Returns:
        Dict mapping feature name → 1-D numpy array of length
        ``len(bin_df)``.
    """
    channel_set = set(ts_channel_names)
    needed = [f for f in COMPOSITE_FEATURE_NAMES if f in channel_set]
    if not needed:
        return {}

    n_positions = len(bin_df)
    result = {f: np.zeros(n_positions) for f in needed}

    if not med_events:
        return result

    # Build DataFrame from events
    rows = []
    for ev in med_events:
        atc = ev.get("atc_code", "")
        if not atc:
            continue
        rows.append({
            "ATC": str(atc),
            "timestamp": pd.Timestamp(ev["timestamp"]),
            "Administrationsdosis": ev.get("dose"),
            "Dosisenhed": ev.get("unit"),
        })

    if not rows:
        return result

    df = pd.DataFrame(rows)

    # Assign records to bins
    bin_starts = bin_df["bin_start"].values
    bin_ends = bin_df["bin_end"].values
    positions = (
        bin_df["position"].values
        if "position" in bin_df.columns
        else np.arange(n_positions)
    )

    assigned = []
    for _, row in df.iterrows():
        ts = np.datetime64(row["timestamp"])
        idx = np.searchsorted(bin_starts, ts, side="right") - 1
        if 0 <= idx < n_positions and ts < bin_ends[idx]:
            r = row.to_dict()
            r["_bin_position"] = int(positions[idx])
            r["PID"] = 0  # single patient
            assigned.append(r)

    if not assigned:
        return result

    assigned_df = pd.DataFrame(assigned)

    # Deduplicate per (bin, ATC), keep highest dose
    if "Administrationsdosis" in assigned_df.columns:
        assigned_df["_ds"] = pd.to_numeric(
            assigned_df["Administrationsdosis"], errors="coerce"
        ).fillna(0)
        assigned_df = assigned_df.sort_values("_ds", ascending=False)
        assigned_df = assigned_df.drop_duplicates(
            subset=["_bin_position", "ATC"], keep="first"
        )
        assigned_df = assigned_df.drop(columns=["_ds"])
    else:
        assigned_df = assigned_df.drop_duplicates(
            subset=["_bin_position", "ATC"], keep="first"
        )

    # Tag → aggregate → derive (same pipeline as batch)
    assigned_df = _tag_records(assigned_df)
    signals = _aggregate_signals(assigned_df)
    signals = _derive_composite_tiers(signals)

    # Extract per-bin values into arrays
    for feat_name in needed:
        if feat_name not in signals.columns:
            continue
        for (pid, pos), row in signals[feat_name].items():
            if 0 <= pos < n_positions:
                result[feat_name][pos] = float(row)

    return result
