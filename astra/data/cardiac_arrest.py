"""
Extract cardiac arrest events from unstructured clinical notes.

Patients can have multiple cardiac arrests during their stay.
Output format: [PID, TIMESTAMP, FEATURE='cardiac_arrest', VALUE=1.0]

Used by mapper.py for on-the-fly Events concept building.
"""

import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

_STOPWORDS = frozenset({
    "og", "i", "er", "en", "et", "til", "på", "med", "af", "de", "den",
    "det", "som", "der", "har", "pt", "pt.", "var", "ved", "fra", "om",
    "for", "ikke", "men", "han", "hun", "sin", "sit", "sig", "da", "så",
})

_ANON_RE = re.compile(r"\bx{2,}\b", re.IGNORECASE)

def _tokenize(text: str) -> frozenset:
    t = _ANON_RE.sub("ANON", text)
    t = re.sub(r"[^\w\s]", "", t)
    return frozenset(t.split()) - _STOPWORDS

def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _extract_date_from_line(line: str) -> str | None:
    m = re.search(r"(?:^|\s)(\d{1,2})[/.\-](\d{1,2})(?:\s*[:.]|\s|$)", line)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}"
    return None

def _extract_year_from_line(line: str) -> int | None:
    m = re.search(r"\b(19|20)(\d{2})\b", line)
    if m:
        return int(m.group(0))
    return None

_ARREST_MINUTES_RE = re.compile(
    r"hjertestop[^.\n]{0,30}?(\d+)\s*min|(\d+)\s*min[^.\n]{0,30}?hjertestop",
    re.IGNORECASE,
)

def _extract_arrest_minutes(line: str) -> int | None:
    m = _ARREST_MINUTES_RE.search(line)
    if m:
        return int(m.group(1) or m.group(2))
    return None

_DATE_PREFIX_RE = re.compile(
    r"^(?:\d{1,2}|xx)[\/\-\.](?:\d{1,2}|xx)\s*[\.:\-]?\s",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Excluded note types
# ---------------------------------------------------------------------------
EXCLUDED_NOTATTYPER = re.compile(
    r"mors-?notat|forskningsnotat|udskrivningsresum[eé]|samtalenotat|historisk"
    r"|behandlingsniveau|konferencenotat|helbredsforhold",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Notetype = hjertestopsnotat — only these phrases disqualify
# ---------------------------------------------------------------------------
NOTETYPE_NEGATION_RE = re.compile(
    r"""
      hjertestop\s+kald\s+ved\s+fejl
    | ikke\s+klinisk\s+hjertestop
    | ikke\s+regelret\s+hjertestop
    | (?:der\s+har\s+)?ikke\s+v[æe]ret\s+hjertestop
    | har\s+ikke\s+v[æe]ret\s+(?:klinisk\s+)?hjertestop
    | vurderer\s+ikke[^.\n]{0,60}hjertestop
    | formentlig\s+ikke[^.\n]{0,60}hjertestop
    | n[æe]ppe[^.\n]{0,40}hjertestop
    | vel\s+ikke[^.\n]{0,40}hjertestop
    | egentlig[^.\n]{0,60}hjertestop[^.\n]{0,40}ikke
    | ikke\s+(?:v[æe]ret\s+)?tale\s+om[^.\n]{0,60}hjertestop
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Diagnosis codes / standard phrases that are never a current arrest
# ---------------------------------------------------------------------------
DIAGNOSIS_CODE_RE = re.compile(
    r"hjertestop\s+med\s+vellykket\s+genoplivning",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Uncertain / suspected arrests
# ---------------------------------------------------------------------------
UNCERTAIN_RE = re.compile(
    r"""
      formodet\s+hjertestop
    | muligt?\s+hjertestop
    | usikkert\s+om[^.\n]{0,60}hjertestop
    | usikker[^.\n]{0,40}hjertestop
    | ikke\s+(?:sikkert|afklaret)[^.\n]{0,40}hjertestop
    | (?:genuint|reelt?)\s+hjertestop
    | om\s+pt\.?\s+har\s+haft[^.\n]{0,40}hjertestop
    | efter\s+formodet\s+hjertestop
    | m[åa]ske\s+(?:\w+\s+){0,3}hjertestop
    | muligvis\s+(?:\w+\s+){0,3}hjertestop
    | sandsynligvis\s+(?:\w+\s+){0,3}hjertestop
    | sandsynligt\s+(?:\w+\s+){0,3}hjertestop
    | t[æe]nkes\s+(?:at\s+)?(?:v[æe]re\s+)?(?:\w+\s+){0,3}hjertestop
    | kunne\s+(?:dreje\s+sig\s+om|v[æe]re)[^.\n]{0,40}hjertestop
    | evt\.?\s+(?:\w+\s+){0,3}hjertestop
    | hjertestop\s*\?
    | \?\s*hjertestop
    | differentialdiagnose[^.\n]{0,60}hjertestop
    | ddx[^.\n]{0,40}hjertestop
    | udelukke[^.\n]{0,40}hjertestop
    | sp[øo]rgsm[åa]l\s+om[^.\n]{0,40}hjertestop
    | uvist\s+om[^.\n]{0,60}hjertestop
    | (?:haft\s+)?sikkert\s+hjertestop
    | mist[æe]nkt\s+hjertestop
    | mist[æe]nkes?[^.\n]{0,40}hjertestop
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Hard negation at line level
# ---------------------------------------------------------------------------
NEGATIVE_LINE_RE = re.compile(
    r"""
    # Explicit denial
      ikke\s+(?:klinisk\s+)?hjertestop(?!\s+(?:ved|p[åa]|under))
    | ikke\s+regelret\s+hjertestop
    | hjertestop\s+kald\s+ved\s+fejl
    | n[æe]ppe\s+(?:\w+\s+){0,3}hjertestop
    | om\s+der\s+overhovedet\s+har\s+v[æe]ret\s+hjertestop
    | der\s+har\s+ikke\s+v[æe]ret\s+hjertestop
    | ikke\s+havde\s+hjertestop
    | ikke\s+haft\s+hjertestop
    | ingen\s+(?:respirations[\s-](?:el(?:ler)?[\s-])?)?hjertestop
    | hjertestopholdet\b

    # HLR / resuscitation opt-out
    | ingen\s+hlr\s+v(?:ed)?\s+hjertestop
    | genoplivning\s+ved\s+hjertestop
    | ingen\s+(?:hjertemassage|medicin)\s+ved\s+hjertestop
    | ikke\s+indikation\s+for\s+(?:hjerte[/]?lungeredning|genoplivning)\s+ved\s+hjertestop
    | der\s+er\s+ikke\s+indikation\s+for[^.\n]{0,60}hjertestop
    | vil\s+der\s+ikke\s+v[æe]re\s+indikation\s+for\s+genoplivning
    | fravalg[^.\n]{0,60}hjertestop
    | i\s+tilf[æe]lde\s+af\s+hjertestop[^.\n]{0,80}(?:ikke|ej|ingen)[^.\n]{0,40}genopliv(?:ning)?
    | ej\s+(?:hlr|genoplivning)\s+(?:hvis|ved)\s+hjertestop
    | ingen\s+hlr\s+(?:hvis|ved)\s+hjertestop
    | behandlingsloft[^.\n]{0,60}hjertestop
    | skal\s+ikke\s+genoplives[^.\n]{0,60}hjertestop
    | ved\s+hjertestop[^.\n]{0,60}ikke\s+iv[æe]rks[æe]ttes
    | ønsker?\s+ikke[^.\n]{0,60}genoplivet[^.\n]{0,60}hjertestop
    | ikke\s+(?:at\s+)?(?:pt\.?\s+)?genoplives?[^.\n]{0,60}hjertestop
    | livsforl[æe]ngende\s+behandling[^.\n]{0,80}hjertestop
    | hjertestop\s+el\.?\s*lign\.?
    | ift\.?\s+hjertestop
    | ifht\.?\s+hjertestop

    # Cause / explanation / genesis
    | (?:\w+\s+){0,4}som\s+[åa]rsag\s+til\s+hjertestop
    | [åa]rsag\s+til\s+hjertestop
    | genese\s+til\s+hjertestop
    | forklaring\s+p[åa][^.\n]{0,40}hjertestop
    | hvad\s+der\s+udl[øo]ser\s+hjertestop
    | hvorfor[^.\n]{0,40}hjertestop
    | hjertestop\s+uns\b
    | hjertestop\s+p[åa]\s+grund\s+af
    | hjertestop\s+pga\.?(?!\s+(?:p[åa]|ved|under))

    # Treatment concepts / post-arrest state
    | hjertestopbehandling
    | \d+\s+dage?\s+efter\s+(?:trauma\s+med\s+)?hjertestop
    | post[\s-]hjertestop
    | posthjertestop
    | konstatering\s+af\s+hjertestop
    | prognose[^.\n]{0,60}hjertestop
    | f[øo]lger?\s+(?:efter\s+)?hjertestop
    | sekvel[^.\n]{0,40}hjertestop
    | rehabiliter[^.\n]{0,60}hjertestop
    | skader\s+fra[^.\n]{0,60}hjertestop
    | (?:kognitive?\s+)?deficits?\s+efter\s+hjertestop
    | opv[åa]gning\s+fra\s+hjertestop
    | efter\s+opv[åa]gning[^.\n]{0,40}hjertestop
    | i\s+lyset\s+af[^.\n]{0,40}hjertestop

    # Historical with explicit time marker
    | inden\s+hjertestop
    | har\s+v[æe]ret\s+kaldt?\s+til\s+hjertestop
    | tidligere\s+(?:\w+\s+){0,3}hjertestop
    | tidl\.?\s+hjertestop
    | kendt\s+med\s+hjertestop
    | anamnestisk[^.\n]{0,60}hjertestop
    | i\s+sygehistorien[^.\n]{0,60}hjertestop
    | i\s+g[åa]r[^.\n]{0,60}hjertestop
    | igår[^.\n]{0,60}hjertestop
    | hjertestop\s+(?:19|20)\d{2}\b
    | (?:19|20)\d{2}[^.\n]{0,40}hjertestop
    | \d+\s+dages?\s+(?:tid\s+)?siden[^.\n]{0,60}hjertestop
    | \d+\s*[åa]r\s+siden[^.\n]{0,60}hjertestop
    | hjertestop\s+\d+\s+dage\s+senere
    | \d+\s+dage\s+senere[^.\n]{0,40}hjertestop
    | seneste\s+livstegn[^.\n]{0,60}hjertestop

    # Family / other persons
    | fam(?:iliære?\s+)?(?:disp\.?|dispositioner?)[^.\n]{0,80}hjertestop
    | for[æa]ldre[^.\n]{0,80}hjertestop
    | familie[^.\n]{0,60}hjertestop
    | sl[æe]gtning[^.\n]{0,60}hjertestop
    | mor\s+(?:d[øo]de?|fik|har)[^.\n]{0,60}hjertestop
    | far\s+(?:d[øo]de?|fik|har)[^.\n]{0,60}hjertestop
    | (?:nabo(?:patient)?|hustru|[æe]gtefælle|kone|s[øo]n|datter)[^.\n]{0,60}hjertestop
    | hjertestop\s+hos\s+(?:nabo|anden|medpatient)

    # Hypothetical / future
    | hvis\s+[^.\n]{0,40}f[åa]r\s+hjertestop
    | s[åa]fremt\s+(?:pt\.?\s+)?f[åa]r\s+hjertestop
    | hvis\s+(?:pt\.?\s+)?(?:igen\s+)?f[åa]r\s+hjertestop
    | ved\s+(?:fornyet|nyt|evt\.?)\s+hjertestop
    | skulle\s+f[åa]\s+hjertestop
    | t[æe]nker[^.\n]{0,60}hjertestop
    | ikke\s+vil\s+genoplives[^.\n]{0,60}hjertestop
    | vil\s+ikke\s+genoplives[^.\n]{0,60}hjertestop

    # Headings / section phrases
    | omst[æe]ndigheder\s+omkring\s+hjertestop
    | p[åa]\s+baggrund\s+af\s+hjertestop
    | grundet\s+hjertestop
    | sekundaert\s+til\s+hjertestop
    | sekund[æe]rt\s+til\s+hjertestop
    | [åa]rsager?\s+til\s+hjertestop
    | information\s+til\s+p[åa]r[øo]rende[^.\n]{0,80}hjertestop
    | problemstilling\s+vedr\.?[^.\n]{0,60}hjertestop

    # i forbindelse med — only when hjertestop follows directly
    | i\s+forbindelse\s+med\s+(?:\w+\s+){0,2}hjertestop
    | ifm\.?\s+hjertestop
    | ifbm\.?\s+hjertestop

    # Concern / fear
    | bekymret[^.\n]{0,60}hjertestop
    | bange\s+for[^.\n]{0,60}hjertestop
    | frygter?[^.\n]{0,60}hjertestop

    # Rhetorical / corrective denial
    | egentlig[^.\n]{0,60}hjertestop[^.\n]{0,40}ikke
    | ikke\s+(?:rigtig|egentlig|reelt?)\s+(?:et\s+)?hjertestop
    | var\s+der\s+vel\s+ikke\s+tale\s+om[^.\n]{0,40}hjertestop
    | ikke\s+(?:v[æe]ret\s+)?tale\s+om[^.\n]{0,40}hjertestop
    | hjertestop[^.\n]{0,60}vel\s+ikke
    | formentlig\s+ikke[^.\n]{0,60}hjertestop
    | vurderes?\s+(?:ikke|at\s+der\s+ikke)[^.\n]{0,60}hjertestop

    # Research / study
    | studie[^.\n]{0,60}hjertestop
    | randomiserer[^.\n]{0,60}hjertestop
    | inkluderer[^.\n]{0,60}hjertestop
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# "Called to arrest, but not really"
# ---------------------------------------------------------------------------
CALLED_BUT_NEGATED_RE = re.compile(
    r"kaldt?\s+(?:til\s+)?hjertestop[^.\n]{0,100}"
    r"(?:ikke|ingen|men\s+(?:pt\.?\s+)?(?:der\s+)?(?:ikke|ingen|har\s+ikke|havde\s+ikke)|"
    r"ikke\s+regelret|ikke\s+klinisk|ikke\s+haft)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# "hjertestop x N"
# ---------------------------------------------------------------------------
HJERTESTOP_XNUM_RE = re.compile(
    r"hjertestop\s*(?:\([^)]{1,20}\)\s*)?x\s*(\d+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Positive capture — current cardiac arrest
# ---------------------------------------------------------------------------
POSITIVE_RE = re.compile(
    r"""
      (?:f[åa]r|g[åa]r\s+i|udvikler|f[åa]et|har\s+(?:f[åa]et|udviklet))\s+hjertestop
    | idet\s+(?:patienten|pt\.?)\s+har\s+(?:f[åa]et|haft)\s+hjertestop
    | (?:patienten|pt\.?)\s+har\s+(?:f[åa]et|haft)\s+hjertestop(?:\s+(?:ved|p[åa]|under))
    | har\s+pt\.?\s+udviklet\s+hjertestop
    | kaldt?\s+(?:til\s+)?hjertestop
    | kaldes\s+(?:til\s+|hjertestop)
    | hjertestop\s+(?:på|p[åa])\s+(?:skadestedet|stedet|gaden)
    | p[åa]\s+(?:skadestedet|stedet)[^.\n]{0,50}hjertestop
    | hjertestop[/]cirkulationssvigt\s+p[åa]\s+skadestedet
    | (?:u)?bevidnet\s+hjertestop
    | (?:pt\.?\s*)?\d{1,3}[- ]?[åa]rig[^.\n]{0,100}hjertestop
    | bringes\s+ind[^.\n]{0,80}hjertestop
    | indbringes[^.\n]{0,60}hjertestop
    | indbragt[^.\n]{0,60}hjertestop
    | (?:ind)?l[æe]gges[^.\n]{0,60}hjertestop
    | hjertestop\s+(?:som\s+)?traumemekanisme
    | hjertestop\s+under\s+transport\s+hertil
    | under\s+transport[^.\n]{0,40}hjertestop
    | hjertestop\s*(?:\([^)]{1,20}\)\s*)?x\s*\d+
    | igen[^.\n]{0,15}hjertestop
    | hjertestop[^.\n]{0,15}igen
    | atter[^.\n]{0,15}hjertestop
    | hjertestop[^.\n]{0,15}atter
    | nyt\s+hjertestop
    | fornyet\s+hjertestop
    | endnu\s+(?:et|ét)\s+hjertestop
    | konstateret\s+hjertestop
    | hjertestop\s+konstateres
    | (?:umiddelbart|kort)\s+efter[^.\n]{0,60}hjertestop
    | (?:ankomst|ankommer|ankom|indkom)[^.\n]{0,60}hjertestop
    | meldes\s+om[^.\n]{0,40}hjertestop
    | svindende\s+puls[^.\n]{0,40}hjertestop
    | ved\s+ankomst[^\n]{0,80}hjertestop
    | klinisk\s+hjertestop
    | \bhjertestop\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

CONTEXT_NEGATION_RE = re.compile(
    r"""
      [åa]rsag | forklaring | overskrift | tidligere | anamnese
    | p[åa]\s+baggrund | grundet | inden
    | (?:d\.?\s*\d{1,2}[./-]\d{1,2}) | udl[øo]ser | hvorfor
    | omst[æe]ndigheder | diagnose | post[\s-]?hjertestop
    | sekundaert | sekund[æe]rt | historik
    """,
    re.IGNORECASE | re.VERBOSE,
)

TRANSPORT_RE  = re.compile(r"hjertestop\s+under\s+transport\s+hertil", re.IGNORECASE)
UBEVIDNET_RE  = re.compile(r"(?:u)?bevidnet\s+hjertestop", re.IGNORECASE)

PAST_TENSE_AMBIGUOUS_RE = re.compile(
    r"""
      (?:patienten|pt\.?)\s+har\s+(?:haft|v[æe]ret\s+i)\s+hjertestop
    | har\s+(?:haft|v[æe]ret)\s+hjertestop
    | der\s+har\s+v[æe]ret\s+hjertestop
    | har\s+v[æe]ret\s+(?:klinisk\s+)?hjertestop
    """,
    re.IGNORECASE | re.VERBOSE,
)

PAST_TENSE_TRAUMA_RE = re.compile(
    r"""
      har\s+(?:haft|f[åa]et)\s+hjertestop
        [^.\n]{0,60}
        (?:p[åa]\s+(?:stedet|skadestedet|gaden|adressen)|ved\s+ankomst|under\s+transport)
    |
      (?:p[åa]\s+(?:stedet|skadestedet|gaden|adressen)|ved\s+ankomst|under\s+transport)
        [^.\n]{0,60}
        har\s+(?:haft|f[åa]et)\s+hjertestop
    """,
    re.IGNORECASE | re.VERBOSE,
)

TRAUMA_CONTEXT_RE = re.compile(
    r"""
    (?:
      ved\s+ankomst[^\n]{0,80}hjertestop
    | hjertestop[^\n]{0,40}ved\s+ankomst
    | bringes\s+ind
    | indbringes[^\n]{0,60}hjertestop
    | indbragt[^\n]{0,60}hjertestop
    | (?:ind)?l[æe]gges[^\n]{0,60}hjertestop
    | indlagt[^\n]{0,60}hjertestop
    | ankommet[^\n]{0,60}hjertestop
    | overflyttet[^\n]{0,60}hjertestop
    | ankom[^\n]{0,40}hjertestop
    | indkom[^\n]{0,40}hjertestop
    | hjertestop\s+p[åa]\s+(?:gaden|stedet|skadestedet|adressen)
    | p[åa]\s+skadestedet[^\n]{0,50}hjertestop
    | ubevidnet\s+hjertestop
    | hjertestop\s+under\s+transport
    | under\s+transport[^\n]{0,40}hjertestop
    | hjertestop\s+(?:som\s+)?traumemekanisme
    | traumamekanisme[^\n]{0,60}hjertestop
    | traumatisk\s+hjertestop
    | (?:\d{1,3}|x+)[- ]?[åa]rig[^\n]{0,100}hjertestop
    | (?:ned|kollaps(?:ede)?)\s+(?:til|med)\s+hjertestop
    | meldes\s+om[^\n]{0,40}hjertestop
    | svindende\s+puls[^\n]{0,40}hjertestop
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

IGEN_RE = re.compile(r"\bigen\b|\batter\b|\bfornyet\b|\bnyt\b|\bendnu\b", re.IGNORECASE)

STRONG_POSITIVE_RE = re.compile(
    r"(?:f[åa]r|g[åa]r\s+i|udvikler|kaldt?|kaldes|bringes|indbringes|indbragt|ankommer|ankom|indkom|"
    r"idet\s+(?:pt|patienten)|har\s+(?:pt\.?\s+)?(?:f[åa]et|udviklet)|"
    r"traumemekanisme|skadestedet|stedet|igen|atter|fornyet|nyt|endnu|ubevidnet|"
    r"\d{1,3}[- ]?[åa]rig|ankomst|meldes|konstateret|konstateres|"
    r"svindende|umiddelbart|indlægges|under\s+transport|"
    r"ved\s+ankomst|p[åa]\s+(?:skadestedet|stedet)|klinisk\s+hjertestop)",
    re.IGNORECASE,
)

EXCLUDED_SECTION_RE = re.compile(
    r"famili[æe]re?\s+dispositioner?|fam\.?\s*disp\.?|"
    r"information\s+til\s+p[åa]r[øo]rende|hereditet",
    re.IGNORECASE,
)

DATE_REFERENCE_RE = re.compile(
    r"(?:hjertestop|arresthjertet|arresteret)\s+(?:d\.?|den)\s+(\d{1,2})[/.](\d{1,2})",
    re.IGNORECASE,
)

PRESENT_TENSE_NEW_EVENT_RE = re.compile(
    r"(?:f[åa]r|g[åa]r\s+i|udvikler|kaldes)\s+hjertestop"
    r"|hjertestop\s+konstateres"
    r"|hjertestop\s+(?:i\s+dag|i\s+nat|i\s+morges|i\s+aftes|til\s+morgen|her\s+til\s+morgen)"
    r"|(?:i\s+dag|i\s+nat|i\s+morges|i\s+aftes|til\s+morgen|her\s+til\s+morgen)[^.\n]{0,30}hjertestop"
    r"|hjertestop\s+kl\.?\s*\d{1,2}",
    re.IGNORECASE,
)

RECAP_INTRO_RE = re.compile(
    r"""
      (?:kort\s+)?sygehistorie | kort\s+anamnese | resumé | resume
    | som\s+(?:bekendt|tidligere\s+beskrevet|n[æe]vnt)
    | anamnestisk
    | pt\.?\s+(?:er|var|blev|har)\s+(?:indlagt|indbragt|overflyttet)
    | tidligere\s+i\s+forl[øo]bet
    | som\s+det\s+fremg[åa]r\s+(?:af|fra)
    | som\s+beskrevet\s+(?:ovenfor|tidligere)
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Core extraction per patient
# ---------------------------------------------------------------------------

def _extract_events_for_patient(df_patient: pd.DataFrame) -> list[dict]:
    pid = df_patient["PID"].iloc[0]
    events: list[dict] = []

    seen_trauma_lines:     list[str] = []
    seen_inhospital_lines: list[str] = []
    transport_seen         = False
    ubevidnet_seen         = False
    seen_notetype_keys:    set[int]  = set()
    any_arrest_registered  = False
    registered_xnums:      set[int]  = set()
    registered_note_tids:  set       = set()

    for _, row in df_patient.iterrows():
        note_text  = str(row["Note"])
        tidspunkt  = row["Redigeringstidspunkt"]
        notetype   = str(row.get("Notetype", "")).strip()

        note_day_month = (
            f"{tidspunkt.day:02d}/{tidspunkt.month:02d}"
            if pd.notna(tidspunkt) else None
        )

        note_is_fresh = pd.isna(tidspunkt) or (tidspunkt not in registered_note_tids)

        # ----------------------------------------------------------------
        # Branch 1: Notetype = hjertestopsnotat
        # ----------------------------------------------------------------
        if re.search(r"hjertestop", notetype, re.IGNORECASE):
            content_key = hash(note_text.strip())
            if content_key in seen_notetype_keys:
                if not IGEN_RE.search(note_text):
                    continue
            seen_notetype_keys.add(content_key)

            if NOTETYPE_NEGATION_RE.search(note_text):
                continue

            igen_count = len(IGEN_RE.findall(note_text))
            for _ in range(1 + igen_count):
                events.append({"PID": pid, "TIMESTAMP": tidspunkt})
            any_arrest_registered = True
            if pd.notna(tidspunkt):
                registered_note_tids.add(tidspunkt)
            continue

        # ----------------------------------------------------------------
        # Branch 2: Line-by-line analysis
        # ----------------------------------------------------------------
        raw_lines = re.split(r"[\n\r]+", note_text)
        lines = []
        for rl in raw_lines:
            if _DATE_PREFIX_RE.match(_normalize(rl)):
                continue
            sub = re.split(r"(?<=\w)\.(?=\s+(?:[A-ZÆØÅ]|[Hh]jertestop\b))", rl)
            lines.extend(s.strip() for s in sub if s.strip())

        note_hjertestop_count  = 0
        note_starts_with_recap = bool(RECAP_INTRO_RE.search(_normalize(note_text[:300])))
        in_excluded_section    = False

        for line in lines:
            if not line:
                continue

            if EXCLUDED_SECTION_RE.search(line):
                in_excluded_section = True
                continue
            if re.match(r"^[A-ZÆØÅ\w][^\n]{0,40}:\s*$", line):
                in_excluded_section = False
            if in_excluded_section:
                continue

            line_norm = _normalize(line)

            if DIAGNOSIS_CODE_RE.search(line_norm):
                continue

            if "hjertestop" not in line_norm:
                continue

            # Date mismatch → historical reference
            line_date = _extract_date_from_line(line_norm)
            if line_date and note_day_month and line_date != note_day_month:
                continue

            date_ref = DATE_REFERENCE_RE.search(line_norm)
            if date_ref and note_day_month:
                ref = f"{int(date_ref.group(1)):02d}/{int(date_ref.group(2)):02d}"
                if ref != note_day_month:
                    continue

            if pd.notna(tidspunkt):
                yr = _extract_year_from_line(line_norm)
                if yr and yr != tidspunkt.year:
                    continue

            if UNCERTAIN_RE.search(line_norm):
                continue

            if NEGATIVE_LINE_RE.search(line_norm):
                continue

            if CALLED_BUT_NEGATED_RE.search(line_norm):
                continue

            if not POSITIVE_RE.search(line_norm):
                continue

            if not STRONG_POSITIVE_RE.search(line_norm):
                m = re.search(r"hjertestop", line_norm)
                if m:
                    ctx = line_norm[max(0, m.start() - 100): m.end() + 100]
                    if CONTEXT_NEGATION_RE.search(ctx):
                        continue

            if PAST_TENSE_AMBIGUOUS_RE.search(line_norm) and not STRONG_POSITIVE_RE.search(line_norm):
                continue

            # Transport / ubevidnet — only first time per patient
            if TRANSPORT_RE.search(line_norm):
                if transport_seen:
                    continue
                transport_seen = True
            if UBEVIDNET_RE.search(line_norm):
                if ubevidnet_seen:
                    continue
                ubevidnet_seen = True

            # Classify line
            is_trauma_line = bool(
                TRAUMA_CONTEXT_RE.search(line_norm)
                or PAST_TENSE_TRAUMA_RE.search(line_norm)
            )

            xnum_match = HJERTESTOP_XNUM_RE.search(line_norm)
            xnum_val = int(xnum_match.group(1)) if xnum_match else None
            is_fresh_xnum = xnum_val is not None and xnum_val not in registered_xnums

            has_repeat_marker = IGEN_RE.search(line_norm) is not None
            is_fresh_note_event = note_is_fresh and not note_starts_with_recap
            has_time_marker = PRESENT_TENSE_NEW_EVENT_RE.search(line_norm) is not None

            # "hjertestop x N"
            if is_fresh_xnum:
                for _ in range(xnum_val):
                    events.append({"PID": pid, "TIMESTAMP": tidspunkt})
                registered_xnums.add(xnum_val)
                note_hjertestop_count += xnum_val
                any_arrest_registered = True
                if pd.notna(tidspunkt):
                    registered_note_tids.add(tidspunkt)
                if is_trauma_line:
                    seen_trauma_lines.append(line_norm)
                else:
                    seen_inhospital_lines.append(line_norm)
                continue

            # "igen/atter/fornyet/nyt/endnu"
            if has_repeat_marker:
                events.append({"PID": pid, "TIMESTAMP": tidspunkt})
                note_hjertestop_count += 1
                any_arrest_registered = True
                if pd.notna(tidspunkt):
                    registered_note_tids.add(tidspunkt)
                if is_trauma_line:
                    seen_trauma_lines.append(line_norm)
                else:
                    seen_inhospital_lines.append(line_norm)
                continue

            # Time marker in fresh note
            if has_time_marker and is_fresh_note_event:
                events.append({"PID": pid, "TIMESTAMP": tidspunkt})
                note_hjertestop_count += 1
                any_arrest_registered = True
                if pd.notna(tidspunkt):
                    registered_note_tids.add(tidspunkt)
                if is_trauma_line:
                    seen_trauma_lines.append(line_norm)
                else:
                    seen_inhospital_lines.append(line_norm)
                continue

            # Standard deduplication
            if any_arrest_registered:
                continue

            if note_hjertestop_count > 0:
                continue

            # Jaccard deduplication
            if is_trauma_line:
                threshold = 0.20 if note_starts_with_recap else 0.25
                best_j = max((_jaccard(line_norm, s) for s in seen_trauma_lines), default=0.0)
                if best_j >= threshold:
                    continue
                seen_trauma_lines.append(line_norm)
            else:
                threshold_inh = 0.30 if registered_note_tids else 0.40
                line_min = _extract_arrest_minutes(line_norm)
                best_j = max((_jaccard(line_norm, s) for s in seen_inhospital_lines), default=0.0)
                is_dup = best_j >= threshold_inh or any(
                    line_min is not None and line_min == _extract_arrest_minutes(s)
                    for s in seen_inhospital_lines
                )
                if is_dup:
                    continue
                seen_inhospital_lines.append(line_norm)

            # Register event
            events.append({"PID": pid, "TIMESTAMP": tidspunkt})
            note_hjertestop_count += 1
            any_arrest_registered = True
            if pd.notna(tidspunkt):
                registered_note_tids.add(tidspunkt)

    return events


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def build_cardiac_arrest_from_notes(notater_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cardiac arrest events from notes.

    Returns [PID, TIMESTAMP, FEATURE='cardiac_arrest', VALUE=1.0].
    Patients can have multiple events.
    """
    notes = notater_df.copy()
    logger.info(f"Cardiac arrest extraction: Notater shape={notes.shape}, columns={notes.columns.tolist()}")
    if "Redigeringstidspunkt" not in notes.columns or "Note" not in notes.columns:
        logger.warning("Cardiac arrest: required columns missing from Notater — returning empty")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])
    notes["Redigeringstidspunkt"] = pd.to_datetime(
        notes["Redigeringstidspunkt"], errors="coerce"
    )

    mask_excl = notes["Notetype"].str.strip().str.contains(EXCLUDED_NOTATTYPER, na=False)
    notes = notes[~mask_excl].copy()
    notes = notes.sort_values(["PID", "Redigeringstidspunkt"]).reset_index(drop=True)

    mask_kw = (
        notes["Note"].str.contains(r"hjertestop", case=False, na=False)
        | notes["Notetype"].str.contains(r"hjertestop", case=False, na=False)
    )
    notes_filtered = notes[mask_kw].copy()
    logger.info(f"Cardiac arrest: {mask_kw.sum()} notes with keyword out of {len(notes)}")

    all_events: list[dict] = []
    for pid, df_grp in notes_filtered.groupby("PID", sort=False):
        all_events.extend(_extract_events_for_patient(df_grp))

    if not all_events:
        logger.info("Cardiac arrest: no events found")
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE"])

    result = pd.DataFrame(all_events)
    result["FEATURE"] = "cardiac_arrest"
    result["VALUE"] = 1.0

    logger.info(
        f"Cardiac arrest: {len(result)} events from {result['PID'].nunique()} patients"
    )
    return result[["PID", "TIMESTAMP", "FEATURE", "VALUE"]]
