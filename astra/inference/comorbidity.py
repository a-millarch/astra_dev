"""
Pure-Python Elixhauser comorbidity scoring for inference.

Reimplements the R `comorbidity` package's `elixhauser_icd10_quan` mapping
with van Walraven weights, eliminating the R subprocess dependency.

References:
    Quan H et al. (2005). Coding algorithms for defining comorbidities in
    ICD-9-CM and ICD-10 administrative data. Medical Care 43(11):1130-1139.

    van Walraven C et al. (2009). A modification of the Elixhauser comorbidity
    measures into a point system for hospital death using administrative data.
    Medical Care 47(6):626-633.
"""

import re
import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elixhauser ICD-10 Quan mapping
# ---------------------------------------------------------------------------
# Each category maps to a list of ICD-10 code prefixes. A diagnosis matches
# a category if it starts with any of the listed prefixes (after stripping
# dots and converting to uppercase).
#
# Source: Quan et al. (2005), Table 2 — ICD-10 coding algorithms.
# Verified against the `comorbidipy` Python package (vvcb/comorbidipy).
# ---------------------------------------------------------------------------

ELIXHAUSER_ICD10_QUAN: Dict[str, List[str]] = {
    "chf": [
        "I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426",
        "I427", "I428", "I429", "I43", "I50", "P290",
    ],
    "carit": [
        "I441", "I442", "I443", "I456", "I459", "I47", "I48", "I49",
        "R000", "R001", "R008", "T821", "Z450", "Z950",
    ],
    "valv": [
        "A520", "I05", "I06", "I07", "I08", "I091", "I098",
        "I34", "I35", "I36", "I37", "I38", "I39",
        "Q230", "Q231", "Q232", "Q233", "Z952", "Z953", "Z954",
    ],
    "pcd": ["I26", "I27", "I280", "I288", "I289"],
    "pvd": [
        "I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
        "K551", "K558", "K559", "Z958", "Z959",
    ],
    "hypunc": ["I10"],
    "hypc": ["I11", "I12", "I13", "I15"],
    "para": [
        "G041", "G114", "G801", "G802", "G81", "G82",
        "G830", "G831", "G832", "G833", "G834", "G839",
    ],
    "ond": [
        "G10", "G11", "G12", "G13", "G20", "G21", "G22",
        "G254", "G255", "G312", "G318", "G319", "G32",
        "G35", "G36", "G37", "G40", "G41", "G931", "G934",
        "R470", "R56",
    ],
    "cpd": [
        "I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45",
        "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65",
        "J66", "J67", "J684", "J701", "J703",
    ],
    "diabunc": [
        "E100", "E101", "E109", "E110", "E111", "E119",
        "E120", "E121", "E129", "E130", "E131", "E139",
        "E140", "E141", "E149",
    ],
    "diabc": [
        "E102", "E103", "E104", "E105", "E106", "E107", "E108",
        "E112", "E113", "E114", "E115", "E116", "E117", "E118",
        "E122", "E123", "E124", "E125", "E126", "E127", "E128",
        "E132", "E133", "E134", "E135", "E136", "E137", "E138",
        "E142", "E143", "E144", "E145", "E146", "E147", "E148",
    ],
    "hypothy": ["E00", "E01", "E02", "E03", "E890"],
    "rf": [
        "I120", "I131", "N18", "N19", "N250",
        "Z490", "Z491", "Z492", "Z940", "Z992",
    ],
    "ld": [
        "B18", "I85", "I864", "I982", "K70", "K711",
        "K713", "K714", "K715", "K716", "K717",
        "K72", "K73", "K74", "K760", "K762", "K763", "K764",
        "K765", "K766", "K767", "K768", "K769", "Z944",
    ],
    "pud": ["K257", "K259", "K267", "K269", "K277", "K279", "K287", "K289"],
    "aids": ["B20", "B21", "B22", "B24"],
    "lymph": ["C81", "C82", "C83", "C84", "C85", "C88", "C96", "C900", "C902"],
    "metacanc": ["C77", "C78", "C79", "C80"],
    "solidtum": [
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
        "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
        "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
        "C30", "C31", "C32", "C33", "C34",
        "C37", "C38", "C39", "C40", "C41", "C43",
        "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53",
        "C54", "C55", "C56", "C57", "C58", "C59", "C60", "C61", "C62",
        "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71",
        "C72", "C73", "C74", "C75", "C76", "C97",
    ],
    "rheumd": [
        "L940", "L941", "L943", "M05", "M06", "M08", "M120", "M123",
        "M30", "M31", "M32", "M33", "M34", "M35", "M45",
        "M461", "M468", "M469",
    ],
    "coag": ["D65", "D66", "D67", "D68", "D691", "D693", "D694", "D695", "D696"],
    "obes": ["E66"],
    "wloss": ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64"],
    "fed": ["E222", "E86", "E87"],
    "blane": ["D500"],
    "dane": ["D508", "D509", "D51", "D52", "D53"],
    "alcohol": [
        "F10", "E52", "G621", "I426", "K292", "K700", "K703", "K709",
        "T51", "Z502", "Z714", "Z721",
    ],
    "drug": ["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715", "Z722"],
    "psycho": ["F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312", "F315"],
    "depre": ["F204", "F313", "F314", "F315", "F32", "F33", "F341", "F412", "F432"],
}

# ---------------------------------------------------------------------------
# Van Walraven weights
# ---------------------------------------------------------------------------
# Source: van Walraven et al. (2009), Table 1.

VAN_WALRAVEN_WEIGHTS: Dict[str, int] = {
    "chf": 7,
    "carit": 5,
    "valv": -1,
    "pcd": 4,
    "pvd": 2,
    "hypunc": 0,
    "hypc": 0,
    "para": 7,
    "ond": 6,
    "cpd": 3,
    "diabunc": 0,
    "diabc": 0,
    "hypothy": 0,
    "rf": 5,
    "ld": 11,
    "pud": 0,
    "aids": 0,
    "lymph": 9,
    "metacanc": 12,
    "solidtum": 4,
    "rheumd": 0,
    "coag": 3,
    "obes": -4,
    "wloss": 6,
    "fed": 5,
    "blane": -2,
    "dane": -2,
    "alcohol": 0,
    "drug": -7,
    "psycho": 0,
    "depre": -3,
}

# Category names for human-readable output
CATEGORY_NAMES: Dict[str, str] = {
    "chf": "Congestive heart failure",
    "carit": "Cardiac arrhythmias",
    "valv": "Valvular disease",
    "pcd": "Pulmonary circulation disorders",
    "pvd": "Peripheral vascular disorders",
    "hypunc": "Hypertension, uncomplicated",
    "hypc": "Hypertension, complicated",
    "para": "Paralysis",
    "ond": "Other neurological disorders",
    "cpd": "Chronic pulmonary disease",
    "diabunc": "Diabetes, uncomplicated",
    "diabc": "Diabetes, complicated",
    "hypothy": "Hypothyroidism",
    "rf": "Renal failure",
    "ld": "Liver disease",
    "pud": "Peptic ulcer disease",
    "aids": "AIDS/HIV",
    "lymph": "Lymphoma",
    "metacanc": "Metastatic cancer",
    "solidtum": "Solid tumor without metastasis",
    "rheumd": "Rheumatoid arthritis / collagen vascular diseases",
    "coag": "Coagulopathy",
    "obes": "Obesity",
    "wloss": "Weight loss",
    "fed": "Fluid and electrolyte disorders",
    "blane": "Blood loss anemia",
    "dane": "Deficiency anemia",
    "alcohol": "Alcohol abuse",
    "drug": "Drug abuse",
    "psycho": "Psychoses",
    "depre": "Depression",
}


# ---------------------------------------------------------------------------
# Hierarchical exclusion rules (matches R comorbidity::assign0 = TRUE)
# ---------------------------------------------------------------------------
# When a more specific/severe category is present, the less specific one is
# suppressed.  These rules come from the original Elixhauser definition and
# are enforced by the R `comorbidity` package when `assign0 = TRUE`.

ELIXHAUSER_EXCLUSIONS: Dict[str, str] = {
    # If present → remove
    "metacanc": "solidtum",   # metastatic cancer  → suppress solid tumor
    "diabc":    "diabunc",    # complicated DM      → suppress uncomplicated DM
    "hypc":     "hypunc",     # complicated HTN     → suppress uncomplicated HTN
}


def _normalize_icd10(code: str) -> str:
    """Strip dots and whitespace, uppercase."""
    return code.replace(".", "").replace(" ", "").upper()


def _apply_exclusions(categories: Set[str]) -> Set[str]:
    """Apply hierarchical exclusion rules (assign0 logic)."""
    for present, suppress in ELIXHAUSER_EXCLUSIONS.items():
        if present in categories:
            categories.discard(suppress)
    return categories


def map_elixhauser_categories(diagnoses: List[str]) -> Set[str]:
    """
    Map a list of ICD-10 diagnosis codes to Elixhauser comorbidity categories.

    Applies hierarchical exclusion rules (matching R's ``assign0 = TRUE``):
    metastatic cancer suppresses solid tumor, complicated diabetes suppresses
    uncomplicated, complicated hypertension suppresses uncomplicated.

    Args:
        diagnoses: List of ICD-10 codes (dots optional, case-insensitive).

    Returns:
        Set of matched category abbreviations (e.g. {'chf', 'rf', 'diabunc'}).
    """
    normalized = [_normalize_icd10(d) for d in diagnoses if d and isinstance(d, str)]
    matched: Set[str] = set()

    for category, prefixes in ELIXHAUSER_ICD10_QUAN.items():
        for code in normalized:
            if any(code.startswith(prefix) for prefix in prefixes):
                matched.add(category)
                break  # One match per category is enough

    return _apply_exclusions(matched)


def compute_elixhauser_vw(diagnoses: List[str]) -> float:
    """
    Compute the van Walraven weighted Elixhauser comorbidity score.

    Args:
        diagnoses: List of ICD-10 codes for the patient.

    Returns:
        Weighted Elixhauser score (float). Returns 0.0 if no diagnoses match.
    """
    categories = map_elixhauser_categories(diagnoses)
    score = sum(VAN_WALRAVEN_WEIGHTS.get(cat, 0) for cat in categories)
    return float(score)


def compute_elixhauser_for_patient(
    base_df: pd.DataFrame,
    data_dir: str,
    patient_dir: str = 'data/patients',
) -> pd.DataFrame:
    """
    Compute Elixhauser comorbidity score for a single patient from raw CSVs.

    Replicates the logic in prepare_elix_df() + R script chain from
    build_patient_info.py, but operates entirely in memory.

    The filtering logic keeps diagnoses that were:
    - Noted BEFORE trauma admission (>= 1 day before start)
    - NOT resolved before trauma (resolved date is null or >= 1 day after start)

    Args:
        base_df: Single-patient base DataFrame with CPR_hash, PID, start, end.
        data_dir: Directory containing raw CSV files (Diagnoser.csv).
        patient_dir: Directory with pre-split per-patient CSVs.

    Returns:
        base_df with ASMT_ELIX column added.
    """
    from astra.inference.patient_store import load_patient_csv

    try:
        cpr_hash = base_df['CPR_hash'].iloc[0]
        diag = load_patient_csv(cpr_hash, 'Diagnoser', data_dir, patient_dir)
    except FileNotFoundError:
        logger.info("Diagnoser.csv not found — setting ASMT_ELIX=0.0")
        base_df["ASMT_ELIX"] = 0.0
        return base_df

    if len(diag) == 0:
        logger.info("Empty Diagnoser.csv — setting ASMT_ELIX=0.0")
        base_df["ASMT_ELIX"] = 0.0
        return base_df

    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"], errors="coerce")
    diag["Løst_dato"] = pd.to_datetime(diag["Løst_dato"], errors="coerce")

    # Merge with patient info
    merged = base_df[["CPR_hash", "PID", "start", "end"]].merge(
        diag, on="CPR_hash", how="left"
    )

    if merged["Diagnosekode"].isna().all():
        logger.info("No diagnoses found for patient — setting ASMT_ELIX=0.0")
        base_df["ASMT_ELIX"] = 0.0
        return base_df

    # Filter: noted before trauma AND not resolved before trauma
    start = merged["start"].iloc[0]
    e_df = merged[
        (merged["Noteret_dato"] <= start - pd.DateOffset(days=1))
        & (
            merged["Løst_dato"].isnull()
            | (merged["Løst_dato"] >= start + pd.DateOffset(days=1))
        )
    ]

    if len(e_df) == 0:
        logger.info("No pre-existing diagnoses — setting ASMT_ELIX=0.0")
        base_df["ASMT_ELIX"] = 0.0
        return base_df

    # Convert Danish diagnosis codes to ICD-10: strip first and last character
    # e.g. "DA10.5B" → "A10.5"
    icd10_codes = (
        e_df["Diagnosekode"]
        .dropna()
        .str.slice(1, -1)
        .tolist()
    )

    score = compute_elixhauser_vw(icd10_codes)
    categories = map_elixhauser_categories(icd10_codes)

    logger.info(
        f"Elixhauser score: {score} "
        f"(categories: {', '.join(sorted(categories)) or 'none'})"
    )

    base_df["ASMT_ELIX"] = score
    return base_df


def compute_iss(diagnoses: List[str]) -> float:
    """
    Placeholder for ISS (Injury Severity Score) computation.

    ISS is not currently used as a model feature in inference. This placeholder
    exists for potential future reimplementation if ISS becomes needed.

    Args:
        diagnoses: List of ICD-10 codes for the patient.

    Returns:
        NaN (not implemented).
    """
    logger.info("ISS computation not implemented for inference — returning NaN")
    return float("nan")
