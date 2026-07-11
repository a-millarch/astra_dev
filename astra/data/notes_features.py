"""
Extract clinical features from unstructured notes: GCS, ISS, Intubation.

These are fitted into the standard pipeline format [PID, TIMESTAMP, FEATURE, VALUE]
and used by mapper.py for cross-concept augmentation.
"""

import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# GCS Extraction
# ============================================================================

GCS_KEYWORD = r"(?:gcs|glasgow\s+coma\s+(?:scale|score))"
GCS_FILLER = r"(?:\s+\w+){0,3}?"
GCS_VALUE = r"[\s:=]*\b([1-9]|1[0-5])\b(?!\s*/\w)"

GCS_PATTERN = re.compile(rf"{GCS_KEYWORD}{GCS_FILLER}{GCS_VALUE}", re.IGNORECASE)
REPEAT_PREFIX = r"(?:rp\.?\s*|gentag\w*\s*)"
SKIP_PATTERN = re.compile(rf"{REPEAT_PREFIX}$", re.IGNORECASE)

FALL_PATTERN = re.compile(
    rf"(?:{GCS_KEYWORD}?\s*(?:er\s+)?(?:falder?|faldet|stiger?|steget)\s+fra|fald\s+(?:i\s+{GCS_KEYWORD}\s+)?fra)"
    rf"\s+(?:{GCS_KEYWORD}\s+)?\b(\d+)\b\s+ti?l\s+(?:{GCS_KEYWORD}\s+)?\b(\d+)\b",
    re.IGNORECASE,
)

ARROW_PATTERN = re.compile(
    rf"{GCS_KEYWORD}[^.\n\r]?(?:fra\s+)?\d+(?:\s(?:→|->|->)\s*\d+)+",
    re.IGNORECASE,
)

SUBSCALE_TOTAL_PATTERN = re.compile(
    rf"{GCS_KEYWORD}[\s:=]*[ØEøe]\d[^.\n\r]*?=\s*\b([1-9]|1[0-5])\b",
    re.IGNORECASE,
)

THRESHOLD_RE = re.compile(r"til\s+under\s+(?:gcs\s*)?\d+", re.IGNORECASE)
HISTORICAL_RE = re.compile(
    r"(?:gcs[^.\n\r]{0,50}for\s+\d+\s+dage?\s+siden|for\s+\d+\s+dage?\s+siden[^.\n\r]{0,50}gcs)",
    re.IGNORECASE,
)


def extract_gcs_from_text(text: str) -> list[int]:
    """Extract GCS values from a text note, handling various formats."""
    results = []
    for line in re.split(r"[\n\r.]+", text):
        # Remove threshold fragments
        if THRESHOLD_RE.search(line):
            line = THRESHOLD_RE.sub("", line)
        # Skip lines with historical references
        if HISTORICAL_RE.search(line):
            continue

        # Process: fra X til Y → keep only Y
        line = FALL_PATTERN.sub(lambda m: f"GCS {m.group(2)}", line)

        # Process: arrow notation → keep only last value
        def replace_arrow(m):
            last = re.findall(r"\d+", m.group(0))[-1]
            return f"GCS {last}"

        line = ARROW_PATTERN.sub(replace_arrow, line)

        # Process: subskala notation with total → keep total
        line = SUBSCALE_TOTAL_PATTERN.sub(lambda m: f"GCS {m.group(1)}", line)

        # Extract GCS values
        for m in GCS_PATTERN.finditer(line):
            window = line[max(0, m.start() - 30) : m.start()]
            if SKIP_PATTERN.search(window.rstrip()):
                continue
            val = int(m.group(1))
            if 3 <= val <= 15:
                results.append(val)

    return results


def build_gcs_from_notes(notater_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract GCS from notes and format as [PID, TIMESTAMP, FEATURE, VALUE].

    Pre-filters notes containing GCS keyword for efficiency.
    One row per GCS value found.
    """
    df = notater_df.copy()
    df["Redigeringstidspunkt"] = pd.to_datetime(df["Redigeringstidspunkt"], errors="coerce")

    # Pre-filter
    mask = df["Note"].str.contains(r"gcs|glasgow", case=False, na=False, regex=True)
    df = df[mask]
    logger.info(f"GCS: {mask.sum()} notes containing GCS keyword")

    # Extract
    df["_gcs_scores"] = df["Note"].astype(str).apply(extract_gcs_from_text)
    df = df[df["_gcs_scores"].map(len) > 0]
    df = df.explode("_gcs_scores").reset_index(drop=True)

    result = df[["PID", "Redigeringstidspunkt", "_gcs_scores"]].rename(
        columns={
            "_gcs_scores": "VALUE",
            "Redigeringstidspunkt": "TIMESTAMP",
        }
    )
    result["FEATURE"] = "GCS"
    result["VALUE"] = result["VALUE"].astype(float)

    logger.info(f"GCS: {len(result)} values extracted from {result['PID'].nunique()} patients")
    return result[["PID", "TIMESTAMP", "FEATURE", "VALUE"]]


# ============================================================================
# ISS Extraction
# ============================================================================

ISS_KEYWORD = r"(?:\()?\bISS(?:-?sco+re)?\.?"  # tillader "(ISS" og "ISS."
ISS_SEP     = r"[\s:=\-]*"                    # tillader bindestreg og newline som separator
ISS_FILLER  = r"(?:\s*\S+\s+){0,3}?"         # op til 3 vilkårlige tokens — fanger "ca.", "er", "på" osv.
ISS_VALUE   = r"\(?(\b[0-9]|[1-6]\d|7[0-5])\b"  # tillader evt. parentes og tal

ISS_PATTERN = re.compile(
    rf"{ISS_KEYWORD}{ISS_SEP}{ISS_FILLER}{ISS_VALUE}",
    re.IGNORECASE,
)

# Bagud: score efterfulgt af ISS — "pt scorer 19 på ISS"
# \S* fanger alt ikke-whitespace så danske tegn (å,ø,æ) ikke er et problem
ISS_BEFORE = re.compile(
    rf"\b([0-9]|[1-6]\d|7[0-5])\b\s*\S*\s*\bISS\b",
    re.IGNORECASE,
)


def extract_iss_from_text(text: str) -> list[int]:
    """Extract ISS scores (0-75) from text, handling various formats."""
    scores = []
    for m in ISS_PATTERN.finditer(text):
        scores.append(int(m.group(1)))
    for m in ISS_BEFORE.finditer(text):
        scores.append(int(m.group(1)))
    return scores


def build_iss_from_notes(notater_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ISS from notes and format as [PID, TIMESTAMP, FEATURE, VALUE].

    Keeps only ONE ISS per patient: the max value at the earliest timestamp.
    ISS is a one-time trauma severity assessment — forward-fill propagates it.
    """
    df = notater_df.copy()
    logger.info(f"ISS extraction: Notater shape={df.shape}, columns={df.columns.tolist()}")
    if "Redigeringstidspunkt" not in df.columns:
        logger.warning("ISS: 'Redigeringstidspunkt' column missing from Notater — returning empty")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])
    if "Note" not in df.columns:
        logger.warning("ISS: 'Note' column missing from Notater — returning empty")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])
    df["Redigeringstidspunkt"] = pd.to_datetime(df["Redigeringstidspunkt"], errors="coerce")

    # Combine notes per patient+timestamp (multi-row notes)
    df = (
        df.fillna({"Note": ""})
        .groupby(["PID", "Redigeringstidspunkt"], as_index=False)
        .agg({"Note": lambda x: "\n".join(x.astype(str))})
    )
    # Empty grouped result (e.g. all timestamps NaT → dropped as group keys)
    # can lose the key columns on some pandas versions.
    if df.empty or "PID" not in df.columns:
        logger.info("ISS: no notes with usable timestamps")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])
    df = df.sort_values(["PID", "Redigeringstidspunkt"]).reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        note_text = str(row["Note"]) if pd.notna(row["Note"]) else ""
        for score in extract_iss_from_text(note_text):
            records.append(
                {
                    "PID": row["PID"],
                    "TIMESTAMP": row["Redigeringstidspunkt"],
                    "FEATURE": "ISS_notes",
                    "VALUE": float(score),
                }
            )

    result = pd.DataFrame(records)
    if len(result) == 0:
        logger.info("ISS: No values found")
        return result

    # Keep only earliest ISS per patient (max value if multiple at same timestamp)
    n_raw = len(result)
    result = result.sort_values(["PID", "TIMESTAMP", "VALUE"], ascending=[True, True, False])
    result = result.drop_duplicates(subset=["PID"], keep="first").reset_index(drop=True)
    logger.info(f"ISS: {len(result)} patients with ISS (reduced from {n_raw} raw extractions)")
    return result


# ============================================================================
# Intubation Extraction
# ============================================================================

PRIMARY_NOTETYPES = {
    "AKA-vurderingsnotater",
    "AKA afsluttende notat for akut ambulante",
    "AKA-Skadenotat",
    "AKA-notater",
    "AKA-Traumenotat",
    "AKA Epikrise",
    "Præhopital patient notat (PPJ)",
}
FALLBACK_NOTETYPES = {"AOP"}

INTUBATION_PATTERN = re.compile(
    r"\b(?:"
    r"intub\w+"
    r"|[Tt]uben?\b"
    r"|tubet"
    r"|tubeplacering\w*"
    r"|OTI"
    r"|RSI"
    r"|ETT"
    r"|respirator\b\w*"
    r"|ventilator\w+"
    r"|mekanisk\s+ventil\w+"
    r")\b",
    re.IGNORECASE,
)

NEGATION_PATTERN = re.compile(
    r"\b(?:ikke|ej|uden|planlæg\w+|overvej\w+|forsøg\w+|forsøgt|evt.?|eventuelt|undlad\w+|undladt)\b",
    re.IGNORECASE,
)

SPECIFIC_INTUBATION = re.compile(
    r"intuberet[s]?\s+(?:i\s+TC|præhospitalt)",
    re.IGNORECASE,
)


def is_intubated(text: str) -> bool:
    """Check if text mentions intubation (without negation)."""
    for m in INTUBATION_PATTERN.finditer(text):
        window = text[max(0, m.start() - 15) : m.start()]
        if not NEGATION_PATTERN.search(window):
            return True
    return False


def _empty_intubation_frame() -> pd.DataFrame:
    """Typed empty result — 'intubated' MUST be bool dtype: an object-dtype
    empty column makes ``df[df["intubated"]]`` degrade to column-label
    indexing instead of row masking, dropping every column."""
    return pd.DataFrame({
        "PID": pd.Series(dtype=object),
        "intubated": pd.Series(dtype=bool),
        "Redigeringstidspunkt": pd.Series(dtype="datetime64[ns]"),
    })


def get_first_intubation_note(notes: pd.DataFrame, notetypes: set) -> pd.DataFrame:
    """
    Return one row per patient with intubation status and timestamp.
    For intubated patients: first note where intubation matches.
    For others: first note of given type.
    """
    if notes.empty:
        return _empty_intubation_frame()

    typed = notes[notes["Notetype"].isin(notetypes)]
    if typed.empty:
        # Patient has notes, but none of these notetypes.
        return _empty_intubation_frame()

    subset = (
        typed
        .fillna({"Note": ""})
        .groupby(["PID", "Redigeringstidspunkt", "Notetype"], as_index=False)
        .agg({"Note": lambda x: "\n".join(x.astype(str))})
    )
    # An empty grouped result (e.g. every Redigeringstidspunkt is NaT and
    # dropped as a group key) loses the key columns on some pandas versions —
    # bail out before PID is referenced.
    if subset.empty or "PID" not in subset.columns:
        return _empty_intubation_frame()
    subset = subset.sort_values(["PID", "Redigeringstidspunkt"])

    subset["intubated"] = subset["Note"].apply(lambda t: is_intubated(t))

    # Intubated patients: first matching note
    first_match = (
        subset[subset["intubated"]]
        .groupby("PID", as_index=False)
        .first()[["PID", "intubated", "Redigeringstidspunkt"]]
    )

    # Others: first note of the type
    first_any = (
        subset[~subset["PID"].isin(first_match["PID"])]
        .groupby("PID", as_index=False)
        .first()[["PID", "intubated", "Redigeringstidspunkt"]]
    )

    return pd.concat([first_match, first_any], ignore_index=True)


def build_intubation_from_notes(notater_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract intubation from notes as [PID, TIMESTAMP, FEATURE, VALUE].

    Only returns rows for intubated patients (VALUE=1.0).
    Non-intubated patients have no row — consistent with pipeline convention
    where absence = NaN (same as no medication, no procedure, etc.).
    """
    notes = notater_df.copy()
    logger.info(f"Intubation extraction: Notater shape={notes.shape}")
    if "Redigeringstidspunkt" not in notes.columns or "Note" not in notes.columns:
        logger.warning("Intubation: required columns missing from Notater — returning empty")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])
    notes["Redigeringstidspunkt"] = pd.to_datetime(
        notes["Redigeringstidspunkt"], errors="coerce"
    )

    # Primary notetypes
    primary = get_first_intubation_note(notes, PRIMARY_NOTETYPES)

    # Fallback — only for patients without primary notetype
    pids_with_primary = set(primary["PID"])
    fallback = get_first_intubation_note(
        notes[~notes["PID"].isin(pids_with_primary)], FALLBACK_NOTETYPES
    )

    result = pd.concat([primary, fallback], ignore_index=True)

    # Supplement: check ALL non-intubated PIDs (including those without
    # primary/fallback notetypes) for specific patterns in other notetypes
    # (`== True` keeps this a boolean ROW mask even if concat degraded the
    # dtype to object)
    pids_intubated = set(result.loc[result["intubated"] == True, "PID"])  # noqa: E712
    all_pids = set(notes["PID"].unique())
    pids_to_check = all_pids - pids_intubated
    exclude = PRIMARY_NOTETYPES | FALLBACK_NOTETYPES
    other_notes = (
        notes[notes["PID"].isin(pids_to_check) & ~notes["Notetype"].isin(exclude)]
        .fillna({"Note": ""})
        .sort_values(["PID", "Redigeringstidspunkt"])
    )

    specific_hits = other_notes[
        other_notes["Note"].apply(lambda t: bool(SPECIFIC_INTUBATION.search(str(t))))
    ]
    if len(specific_hits) > 0:
        first_specific = specific_hits.groupby("PID", as_index=False).first()
        supplement = first_specific[["PID", "Redigeringstidspunkt"]].copy()
        supplement["intubated"] = True
        result = pd.concat([result, supplement], ignore_index=True)

    # Keep only intubated patients
    result = result[result["intubated"] == True].copy()

    # Format output
    result = result[["PID", "Redigeringstidspunkt"]].rename(
        columns={"Redigeringstidspunkt": "TIMESTAMP"}
    )
    result["FEATURE"] = "INTUBATED"
    result["VALUE"] = 1.0

    logger.info(f"Intubation: {len(result)} intubated patients")
    return result[["PID", "TIMESTAMP", "FEATURE", "VALUE"]]
