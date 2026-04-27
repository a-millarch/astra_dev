import re
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union


# ============================================================================
# Vital signs: Danish parameter names → standardized feature names
# (source: filter_vitals in filters.py)
# ============================================================================
TEMP_FAHRENHEIT = ["Kernetemperatur", "Blæretemperatur", "Rektaltemperatur", "Axiltemperatur", "Esophagustemperatur"]
VITALS_MAP = {
    'Saturation': 'SPO2',
    'ABP Puls (fra A-kanyle)': 'HR',
    'Puls': 'HR',
    'Puls (fra SAT-måler)': 'HR',
    'Resp.frekvens': 'RESPIRATORYRATE',
    'SYSTOLIC': 'SBP',
    'ART mean inv BT': 'MAP',
    'Temperatur': 'TEMP',
    'Temp.': 'TEMP',
    'Blæretemperatur': 'TEMP',
    'Esophagustemperatur': 'TEMP',
    'DBP': 'DBP',
    'SBP': 'SBP',
}

# Blood pressure parameter names that carry "sys/dia" strings to split
BP_TYPES = [
    'BT',
    'ART inv BT',
    'Invasivt BT - ABP (sys/dia)',
    'NIBP',
    'ABP inv BT',
    'Invasivt BT - ART (sys/dia)',
]

# Invasive blood pressure parameter names (subset of BP_TYPES)
INVASIVE_BP_TYPES = frozenset({
    'ART inv BT',
    'Invasivt BT - ABP (sys/dia)',
    'ABP inv BT',
    'Invasivt BT - ART (sys/dia)',
})

# Raw vital parameter names that indicate invasive measurement → category label
INVASIVE_VITALS_MAP = {
    'ABP Puls (fra A-kanyle)': 'arterial_hr',
    'Blæretemperatur': 'invasive_temp',
    'Esophagustemperatur': 'invasive_temp',
}

# Height/weight parameter names → standardized
HEIGHT_WEIGHT_MAP = {
    'Højde': 'HEIGHT',
    'Vægt': 'WEIGHT',
}

# EWS measurements → Vital_parametre names (original VitaleVaerdier column names)
# These are then standardized by VITALS_MAP in filter_vitals()
EWS_TO_VITAL_PARAMETRE = {
    "SAT (score)": "Saturation",
    "Temp. (score)": "Temperatur",
    "BT (score)": "BT",
    "RF (score)": "Resp.frekvens",
}


# ============================================================================
# Lab tests: Danish test names → standardized feature names
# (source: filter_labs in filters.py)
# ============================================================================

# Canonical form: standardized name → tuple of raw names
LABS_FEATURE_MAP = {
    'LACTATE': (
        'LAKTAT(POC);P(AB)',
        'LAKTAT;P(AB)',
        'LAKTAT;P(VB)',
        'LAKTAT(POC);P(VB)',
        'LAKTAT;CSV',
        'LAKTAT(POC);CSV',
        'LAKTAT(POC);P(KB)',
    ),
    'BASE_EXCESS': ('BASE EXCESS;ECV', 'ECV-BASE EXCESS;(POC)'),
    'HEMOGLOBIN': ('HÆMOGLOBIN;B', 'HÆMOGLOBIN(POC);B', 'HÆMOGLOBIN (POC);B'),
    'LEUKOCYTES': ('LEUKOCYTTER;B',),
    'B-GROUP-LEUKOCYTES': (
        'LEUKOCYTTYPE (MIKR.) GRUPPE;B',
        'LEUKOCYTTYPE GRUPPE;B',
        'LEUKOCYTTYPE; ANTALK. (LISTE);B',
    ),
    'TEG-R': ('TEG-R',),
    'TEG-MA': ('TEG-MA',),
    'TEG-LY30': ('TEG-LY30',),
}

# Reverse lookup: raw test name → standardized name
LABS_REVERSE_MAP: Dict[str, str] = {}
for _std_name, _raw_names in LABS_FEATURE_MAP.items():
    for _raw in _raw_names:
        LABS_REVERSE_MAP[_raw] = _std_name


# ============================================================================
# ICU scores: measurement names → standardized feature names
# (source: filter_ita in filters.py)
# ============================================================================

ICU_MAP = {
    'GLASGOW COMA SCORE': 'GCS',
    'Glasgow Coma Score': 'GCS',
    'SAPS 3 SCORE': 'SAPS3',
    'SOFA total score': 'SOFA',
}

# ============================================================================
# EWS: measurement names → standardized feature names
# (source: filter_ews in filters.py)
# ============================================================================

EWS_MAP = {
    'EWS korr. total score': 'EWS_SCORE',
}


# ============================================================================
# Medications: ATC code prefixes → category names
# (source: filter_medicin in filters.py)
# ============================================================================

ATC_LVL3_MAP = {
    'cardiovascular_drugs': ['C01', 'C02', 'C07'],
    'antibiotics': ['J01'],
    'neuro_drugs': ['N05', 'M03'],
    'anti_thrombotic': ['B01'],
    'diuretics': ['C03'],
    'hemostatics': ['B02'],
    'hormone_drugs': ['H01'],
    'antidotes': ['V03'],
}

ATC_LVL4_MAP = {
    'infusion': ['B05B', 'B05X'],
    'blood': ['B05A'],
    'opiods': ['N02A'],
    'local_anastethics': ['N01B'],
    'anastethics': ['N01A'],
    'insulin': ['A10A'],
}

# Reverse lookups: ATC prefix → category name
ATC_LVL3_REVERSE: Dict[str, str] = {}
for _cat, _codes in ATC_LVL3_MAP.items():
    for _code in _codes:
        ATC_LVL3_REVERSE[_code] = _cat

ATC_LVL4_REVERSE: Dict[str, str] = {}
for _cat, _codes in ATC_LVL4_MAP.items():
    for _code in _codes:
        ATC_LVL4_REVERSE[_code] = _cat

# Valid medication administration actions (from filter_medicin)
MEDICATION_ACTION_LIST = [
    'Administreret',
    'Ny pose',
    'Selvadministration',
    'Adm. ernæring/sterilt vand',
    'Genstartet',
    'Infusion/pose skiftet',
    'Selvmedicinering',
    'Status, indgift',
]


# ============================================================================
# Procedures: procedure codes → surgical category names
# (source: filter_procedures in filters.py)
# ============================================================================

PROCEDURE_MAP = {
    'KA': 'neuro',
    'KB': 'endokrin',
    'KC': 'øje',
    'KD': 'ønh',
    'KE': 'oral',
    'KF': 'kardio',
    'KG': 'thorax',
    'KH': 'mamma',
    'KJ': 'abdomen',
    'KK': 'uro',
    'KL': 'gyn',
    'KM': 'obstetrik',
    'KN': 'orto',
    'KP': 'vaskulær',
    'KQ': 'hud',
    'BGD': 'respirator',
    'BGA': 'sonde_tube',
}

# Ordered list of prefixes (longest first so 3-char prefixes match before 2-char)
PROCEDURE_PREFIXES: List[str] = sorted(PROCEDURE_MAP.keys(), key=len, reverse=True)


# ============================================================================
# ADT (department) classification patterns
# (source: filter_adt in filters.py)
#
# Each entry: (location_type, list_of_patterns)
# A pattern is either:
#   - a string: simple regex match (case-insensitive)
#   - a tuple of strings: compound AND match (all patterns must match)
# ============================================================================

ADT_PATTERNS: List[Tuple[str, list]] = [
    ('TB', ['traumecenter']),
    ('ED', [
        r'(?i)(?!.*traumecenter)(?!.*psyk)(?!.*børnemod).*\bakutmodtagelse\b',
        r'(?i)(?!.*traumecenter)(?!.*psyk)(?!.*børnemod).*\bakutklinik\b',
        r'(?i)(?!.*traumecenter)(?!.*psyk)(?!.*børnemod).*\b[\w.]+\s+modtagelse\b',  # [\w.] allows "F."
    ]),
    ('OR', [
        'operationsgang',
        'operationsklinik',
        'operationsafsnit',
        'dagkirurgi',
        'op afs',
        'op-afsnit',
        'centraloperation',
        r'operationsanæstesi',
        r'\bBEDØVELSE OG OPERATION\b',
        ('øjenkl', 'operation'),  # compound: both must match
        r'\bREUM/RYG OP\b',
        'kirurgisk endo',
    ]),
    ('ICU', [r'\bintensiv\b', r'\bita\s', r'\bita,']),
    ('WARD', ['seng']),
    ('OPD', ['amb']),
]


# ============================================================================
# Hospital name standardization
# (source: standardize_hospital, first_hospital in build_patient_info.py)
# ============================================================================

VALID_HOSPITALS = [
    'RH', 'AHH', 'HGH', 'NOH', 'BFH', 'BOH', 'RHP',
    'SJ KØGE', 'SJ HOLBÆK', 'SJ NYKØBING', 'SJ ROSKILDE',
    'SJ VORDINGBORG', 'SJ NÆSTVED', 'SJ SLAGELSE',
]

# Sex value normalization
SEX_MAP = {
    'Mand': 'Male',
    'Kvinde': 'Female',
    'M': 'Male',
    'F': 'Female',
    'K': 'Female',
    'Male': 'Male',
    'Female': 'Female',
}


# ============================================================================
# Utility functions (pure — no file I/O, no heavy deps)
# ============================================================================

def classify_department(dept_name: str) -> Optional[str]:
    """
    Classify a department name into a location type.

    Uses ADT_PATTERNS. Supports both simple regex patterns and compound
    (tuple) patterns where all sub-patterns must match.

    Returns:
        'TC', 'OR', 'ICU', 'BED', 'AMB', or None.
    """
    if pd.isna(dept_name) or not dept_name:
        return None
    for location_type, patterns in ADT_PATTERNS:
        for pattern in patterns:
            if isinstance(pattern, tuple):
                # Compound: all sub-patterns must match
                if all(re.search(p, dept_name, re.IGNORECASE) for p in pattern):
                    return location_type
            else:
                if re.search(pattern, dept_name, re.IGNORECASE):
                    return location_type
    return None


def first_hospital(name: str) -> str:
    """
    Extract hospital identifier from a department name.

    Takes the first word, or "SJ <second_word>" for Sjælland hospitals.
    """
    if pd.isna(name):
        return name
    words = str(name).strip().split()
    if words and words[0] == 'SJ':
        return ' '.join(words[:2])
    elif words:
        return words[0]
    return ''


def standardize_hospital(
    name: str,
    valid_hospitals: List[str] = VALID_HOSPITALS,
) -> str:
    """
    Standardize a hospital name to a canonical form.

    Handles partial matches for SJ hospitals and returns 'MISC' for
    unrecognized names.
    """
    if pd.isna(name):
        return np.nan
    name = str(name).strip().upper()

    if name.startswith('SJ HOL'):
        return 'SJ HOLBÆK'
    if name.startswith('SJ ROS'):
        return 'SJ ROSKILDE'

    for h in valid_hospitals:
        if name == h.upper():
            return h

    return 'MISC'


def derive_first_hospital(dept_name: str) -> str:
    """Extract and standardize hospital name from a department name.

    Convenience wrapper: first_hospital() → standardize_hospital().
    Equivalent to add_first_hospital() pipeline in build_patient_info.py.
    """
    if pd.isna(dept_name) or not dept_name:
        return np.nan
    cleaned = str(dept_name).strip().replace(',', '')
    extracted = first_hospital(cleaned)
    return standardize_hospital(extracted)


def parse_numeric(value_str: str) -> Optional[float]:
    """Parse a numeric value from a string, stripping <, > prefixes."""
    if not value_str:
        return None
    cleaned = re.sub(r'^[<>]\s*', '', str(value_str).strip())
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def classify_atc(atc_code: str) -> Optional[str]:
    """Map an ATC code to a medication category.

    Tries level 3 first (first 3 chars), then level 4 (first 4 chars).
    Returns None if no match.
    """
    if not atc_code:
        return None
    atc = str(atc_code)
    cat = ATC_LVL3_REVERSE.get(atc[:3])
    if cat is None:
        cat = ATC_LVL4_REVERSE.get(atc[:4])
    return cat


# ============================================================================
# Pre-Hospital Journal (PPJ) mappings
# (source: prehospital.py — PPJ extraction pipeline)
# ============================================================================

# PPJ timestamps use 3-letter English month abbreviations (e.g. "22FEB2018:13:40:02.2750")
PPJ_MONTH_DICT = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# PPJ vital sign subset names → ASTRA standard feature names
PPJ_VITALS_MAP = {
    "M_NInv Sys Blodtryk": "SBP",
    "M_NInv Dia Blodtryk": "DBP",
    "M_Puls": "HR",
    "M_SpO2": "SPO2",
}

# PPJ EventCodeName → subset name for vital signs
# NOTE: These are FALLBACK codes only, used when event_descriptions_modified.xlsx
# is not available. The correct codes are resolved dynamically from the
# "Eventkoder Vitaldata" sheet (typically OMI codes, not SVD codes).
# SVD codes are secondary assessment listvalues, NOT numeric vital measurements.
PPJ_VITAL_EVENT_CODES: Dict[str, str] = {
    "SVD00029": "M_Puls",
    "SVD00030": "M_NInv Sys Blodtryk",
    "SVD00031": "M_NInv Dia Blodtryk",
    "SVD00032": "M_SpO2",
}

# PPJ EventCodeName for GCS (legacy, resolved dynamically now)
PPJ_GCS_EVENT_CODE = "GCS"  # subset name after ppjDataset.collect_subsets()

# PPJ ABCD categorical assessment → short column names
PPJ_ABCD_MAP: Dict[str, str] = {
    "A: Luftveje": "A",
    "B: Respiration": "B",
    "C: Cirkulation": "C",
    "D: Bevidsthedsniveau": "D",
}

# Outlier bounds for in-hospital vital signs
VITALS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "SBP":             (0.0, 300.0),
    "DBP":             (0.0, 200.0),
    "MAP":             (0.0, 300.0),
    "HR":              (0.0, 250.0),
    "SPO2":            (0.0, 100.0),
    "RESPIRATORYRATE": (0.0, 80.0),
    "TEMP":            (25.0, 43.0),
    "HEIGHT":          (50.0, 230.0),
    "WEIGHT":          (2.0, 300.0),
}

# Outlier bounds for PPJ vital signs (same as triAIge clean_sequentials)
PPJ_VITAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HR": (0.0, 220.0),
    "SPO2": (0.0, 100.0),
    "SBP": (0.0, 300.0),
    "DBP": (0.0, 200.0),
}

# ABCD severity ordering (index 0 = least severe, last = most severe)
# Used to pick the most severe assessment when a patient has PPJ data from multiple sources
ABCD_SEVERITY: Dict[str, list] = {
    "A": ["Fri", "Truede", "Blokerede"],
    "B": ["Normal", "Let påvirket", "Meget påvirket", "Respirationsstop"],
    "C": ["Normal", "Let påvirket", "Meget påvirket", "Hjertestop"],
    "D": ["Vågen", "Bevidsthedspåvirket", "Bevidstløs"],
}
