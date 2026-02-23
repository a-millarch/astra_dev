import re
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union


# ============================================================================
# Vital signs: Danish parameter names → standardized feature names
# (source: filter_vitals in filters.py)
# ============================================================================

VITALS_MAP = {
    'Saturation': 'SPO2',
    'ABP Puls (fra A-kanyle)': 'HR',
    'Puls': 'HR',
    'Puls (fra SAT-måler)': 'HR',
    'Resp.frekvens': 'RESPIRATORYRATE',
    'SYSTOLIC': 'SBP',
    'ART mean inv BT': 'MAP',
    'Temp (in Celsius)': 'TEMP',
    'Temp.': 'TEMP',
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

# Height/weight parameter names → standardized
HEIGHT_WEIGHT_MAP = {
    'Højde': 'HEIGHT',
    'Vægt': 'WEIGHT',
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
    'neuro_major': (
        'KAAA27', 'KAAD05', 'KAAF00A', 'KAAD00', 'KAAD15', 'KAAA20',
        'KAAA40', 'KAAC00', 'KAAA99', 'KAAD40', 'KAAL11', 'KAAB30',
        'KAAD10', 'KABC60', 'KAAD30', 'KAWD00', 'KAAK35', 'KAAK00', 'KAAK10',
    ),
    'abdominal_major': (
        'KNHJ63', 'KJBA00', 'KPCT20', 'KPCT99', 'KJDH70', 'KJJA96',
        'KKBV02A', 'KJJW96', 'KKAH00', 'KJKB30', 'KKAD10', 'KKAC00',
        'KPCT30', 'KJJA50', 'KJJB00',
    ),
    'vascular_major': (
        'KFNG05A', 'KFNG02A', 'KPBH20', 'KPET11', 'KPEA12', 'KPBC30',
        'KPHC23', 'KPDC30', 'KPBB30', 'KPDG10', 'KPDT30', 'KPEH12',
        'KPBC10', 'KPBN20', 'KACB22', 'KPAC20', 'KPBE30', 'KPDF10',
        'KPEA10', 'KPBA20', 'KPHH99', 'KFCA70', 'KFCA50', 'KPBU82',
        'KPHP30', 'KPEN11', 'KPEH20', 'KPFN30', 'KPEC12', 'KNDL41',
        'KPDQ10', 'KPAP21', 'KPCH30', 'KPFC10', 'KPHC22', 'KPAQ21',
        'KPBC20', 'KPEP11', 'KPEU87', 'KPFE10',
    ),
    'thorax_major': (
        'KGAB10', 'KGAA31', 'KGAB20', 'KGDB11', 'KGAC10', 'KFLC00',
        'KFXE00', 'KFEB10', 'KFXD00', 'KFWW96', 'KGDA40', 'KGAE30',
        'KUGC02', 'KFJB00', 'KGAE03', 'KGDB10', 'KGDA41', 'KFEW96',
        'KGDB96', 'KGAE96',
    ),
    'orto_major': (
        'KNGJ22', 'KNAG73', 'KNAG40', 'KNFJ54', 'KNAG70', 'KNGM09',
        'KNEJ29', 'KNGJ29', 'KNGJ52', 'KNFJ25', 'KNDL40A', 'KNEJ69',
        'KACB23', 'KNGJ21', 'KNCJ45', 'KNCJ27', 'KACB29', 'KNAG71',
        'KNDA02', 'KACC51', 'KNHJ45', 'KNFJ51', 'KNAG72', 'KNDM09',
        'KNHJ62', 'KNDJ42', 'KNFJ43', 'KNBQ03', 'KNCJ65', 'KNGQ19',
        'KNAG76', 'KNGJ40', 'KABC56', 'KPBB99', 'KACB21', 'KNGJ61',
        'KNDL40', 'KNFQ19', 'KNAN00', 'KNBJ41', 'KNBJ61', 'KNCJ88',
        'KNBA02', 'KNHJ80', 'KNDJ43', 'KNHJ47', 'KNGE29', 'KNHJ23',
        'KNHJ71', 'KACA13', 'KNFJ10', 'KNFJ70', 'KNFJ73', 'KNHN09',
        'KNCJ67', 'KNGJ71', 'KNCJ26', 'KNCJ60', 'KNCJ42', 'KNAN03',
        'KNFJ52', 'KNCE22', 'KNDQ99', 'KNHQ22', 'KNCL49', 'KQCG30',
        'KNCJ64', 'KNAN02', 'KNAK12', 'KNHJ72', 'KABA00', 'KNCJ28',
        'KNCJ80', 'KNFJ44', 'KNHJ82', 'KNFJ55', 'KNEJ89', 'KNAJ12',
        'KACC29', 'KNDJ11', 'KNDU39', 'KNDJ70', 'KNBJ51', 'KNHJ22',
        'KNHL49', 'KNHE99', 'KNFM09', 'KNGJ80', 'KQAA10', 'KNHJ14',
        'KNHJ44', 'KNDL41A', 'KNAK10', 'KNBJ62', 'KNBJ21', 'KNCJ47',
        'KNAJ00', 'KACA19', 'KNFQ99', 'KNFJ50', 'KNGJ73', 'KNHJ81',
        'KNGM99', 'KECB40', 'KNGD22', 'KNCJ05', 'KNHJ25', 'KACC53',
        'KNHJ24', 'KNCM09', 'KNDH12', 'KNAN04', 'KNFJ65', 'KNDH02',
        'KNHJ41', 'KNHJ74', 'KNCJ66', 'KNGJ63', 'KNHJ42', 'KNFJ45',
        'KNGJ42', 'KNAG41', 'KNFA02A',
    ),
    'ønh_major': (
        'KEFB20', 'KEDC38', 'KEEC25', 'KEEC35', 'KDLD30', 'KEWE00',
        'KECB20A', 'KDQE00', 'KEDC36', 'KGBA00', 'KGAB00', 'KDWE00',
        'KENC00', 'KDHD30', 'KDJD20', 'KDAD30', 'KDWA00', 'KDQW99',
        'KEMC00', 'KEDC39B', 'KDLD20',
    ),
}

# Reverse lookup: procedure code → category name
PROCEDURE_REVERSE_MAP: Dict[str, str] = {}
for _cat, _codes in PROCEDURE_MAP.items():
    for _code in _codes:
        PROCEDURE_REVERSE_MAP[_code] = _cat

# Flat include list of all procedure codes
PROCEDURE_INCLUDE_LIST: List[str] = [
    code for codes in PROCEDURE_MAP.values() for code in codes
]


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
    ('TC', ['traumecenter']),
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
    ('BED', ['seng']),
    ('AMB', ['amb']),
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
