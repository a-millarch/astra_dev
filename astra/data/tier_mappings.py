"""Tier mapping registry for ordinal profiling of medication categories.

Each TierMapping defines a priority-based classification of ATC codes into
ordinal tiers.  Mappings are registered by name and referenced from
profiles.yaml via ``tier_mapping.mapping``.

Adding a new medication group:
    1. Define exact/prefix/default_prefix tables below.
    2. Call ``register_mapping(TierMapping(name=..., ...))``.
    3. Reference the name in profiles.yaml under the relevant category.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TierMapping class
# ---------------------------------------------------------------------------


class TierMapping:
    """Classifies ATC codes into ordinal tiers via a priority chain.

    Priority order:
        1. Exact code match  (``exact`` dict)
        2. Prefix match      (``prefix`` list, longest prefix first)
        3. Default prefixes   (``default_prefixes`` dict)
        4. None               (unmatched)
    """

    def __init__(
        self,
        name: str,
        n_levels: int,
        exact: Dict[str, int],
        prefix: List[Tuple[str, int]],
        default_prefixes: Optional[Dict[str, int]] = None,
    ):
        self.name = name
        self.n_levels = n_levels
        self.exact = exact
        # Sort by prefix length descending so longest match wins
        self.prefix = sorted(prefix, key=lambda x: len(x[0]), reverse=True)
        self.default_prefixes = default_prefixes or {}

    def classify(self, atc_code: str) -> Optional[int]:
        """Map an ATC code to a tier (1-based) or None if unmatched."""
        if not atc_code or len(atc_code) < 4:
            return None

        # 1. Exact match
        tier = self.exact.get(atc_code)
        if tier is not None:
            return tier

        # 2. Prefix match (longest first)
        for pfx, tier in self.prefix:
            if atc_code.startswith(pfx):
                return tier

        # 3. Default prefix fallbacks
        for pfx, tier in self.default_prefixes.items():
            if atc_code.startswith(pfx):
                return tier

        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TIER_MAPPINGS: Dict[str, TierMapping] = {}


def register_mapping(mapping: TierMapping) -> None:
    TIER_MAPPINGS[mapping.name] = mapping


def get_mapping(name: str) -> TierMapping:
    if name not in TIER_MAPPINGS:
        raise KeyError(
            f"Tier mapping '{name}' not registered. "
            f"Available: {list(TIER_MAPPINGS.keys())}"
        )
    return TIER_MAPPINGS[name]


# ---------------------------------------------------------------------------
# Antibiotic escalation tiers
# ---------------------------------------------------------------------------

_ABX_EXACT: Dict[str, int] = {
    # Tier 1 — Narrow first-line
    "J01CE01": 1,  # Benzylpenicillin
    "J01CE02": 1,  # Phenoxymethylpenicillin
    "J01XE01": 1,  # Nitrofurantoin
    "J01EB02": 1,  # Sulfamethizole
    "J01EA01": 1,  # Trimethoprim
    "J01XX05": 1,  # Methenamine
    # Tier 2 — Standard Access
    "J01XD01": 2,  # Metronidazole
    "J01CA08": 2,  # Pivmecillinam
    "J01CF01": 2,  # Dicloxacillin
    "J01CF02": 2,  # Flucloxacillin
    "J01CF05": 2,  # Flucloxacillin (alt code)
    "J01CA04": 2,  # Amoxicillin
    "J01GB03": 2,  # Gentamicin
    "J01DB01": 2,  # Cefalexin
    "J01DB04": 2,  # Cefazolin
    "J01CA01": 2,  # Ampicillin
    "J01CA02": 2,  # Pivampicillin
    "J01GB01": 2,  # Tobramycin
    "J01CA11": 2,  # Mecillinam
    # Tier 3 — Moderate Watch
    "J01DC02": 3,  # Cefuroxime
    "J01FA01": 3,  # Erythromycin (see exclusion for low-dose prokinetic use)
    "J01FA09": 3,  # Clarithromycin
    "J01FA06": 3,  # Roxithromycin
    "J01FA10": 3,  # Azithromycin
    "J01EE01": 3,  # Co-trimoxazole
    "J01AA02": 3,  # Doxycycline
    "J01AA07": 3,  # Tetracycline
    "J01FF01": 3,  # Clindamycin
    "J01CR02": 3,  # Amoxicillin-clavulanate
    # Tier 4 — Broad Watch
    "J01XA01": 4,  # Vancomycin
    "J01MA02": 4,  # Ciprofloxacin
    "J01MA14": 4,  # Moxifloxacin
    "J01DD04": 4,  # Ceftriaxone
    "J01DD01": 4,  # Cefotaxime
    # Tier 5 — Very broad / critical (exceptions that override prefix rules)
    "J01CR05": 5,  # Piperacillin-tazobactam
    "J01DH02": 5,  # Meropenem
    "J01DD02": 5,  # Ceftazidime (antipseudomonal, not tier 4 like other J01DD)
    "J01DE01": 5,  # Cefepime
    "J01DI54": 5,  # Ceftolozane-tazobactam
    "J01DH01": 5,  # Imipenem-cilastatin
    "J01DH03": 5,  # Ertapenem
    # Tier 6 — Reserve / last-resort
    "J01XX08": 6,  # Linezolid
    "J01XB01": 6,  # Colistin
    "J01DD52": 6,  # Ceftazidime-avibactam
    "J01AA12": 6,  # Tigecycline (exception: not tier 3 like other J01AA)
    "J01DI04": 6,  # Cefiderocol
    "J01XX09": 6,  # Daptomycin
}

_ABX_PREFIX: List[Tuple[str, int]] = [
    # Tier 1
    ("J01CE", 1),  # Narrow-spectrum penicillins
    # Tier 2
    ("J01CF", 2),  # Anti-staphylococcal penicillins
    ("J01DB", 2),  # First-gen cephalosporins
    ("J01GB", 2),  # Aminoglycosides
    # Tier 3
    ("J01DC", 3),  # Second-gen cephalosporins
    ("J01FA", 3),  # Macrolides
    ("J01AA", 3),  # Tetracyclines (except J01AA12 → 6, caught by exact match)
    ("J01EE", 3),  # Sulfonamide combinations
    ("J01FF", 3),  # Lincosamides
    # Tier 4
    ("J01MA", 4),  # Fluoroquinolones
    ("J01XA", 4),  # Glycopeptides
    # Tier 5
    ("J01DH", 5),  # Carbapenems
    ("J01DE", 5),  # Fourth-gen cephalosporins
    # Tier 6
    ("J01XB", 6),  # Polymyxins
]

_ABX_DEFAULT_PREFIXES: Dict[str, int] = {
    "J01DD": 4,  # Unrecognized third-gen cephalosporins default to tier 4
}

register_mapping(
    TierMapping(
        name="antibiotic_escalation",
        n_levels=6,
        exact=_ABX_EXACT,
        prefix=_ABX_PREFIX,
        default_prefixes=_ABX_DEFAULT_PREFIXES,
    )
)


# ---------------------------------------------------------------------------
# Vasopressor escalation (simplified — no dose data)
# All vasopressor/inotrope codes → tier 1.  Multi-agent escalation (tier 2)
# is captured via the n_distinct feature rather than code-level logic.
# ---------------------------------------------------------------------------

_VASO_EXACT: Dict[str, int] = {
    "C01CA03": 1,  # Norepinephrine
    "C01CA24": 1,  # Epinephrine
    "C01CA04": 1,  # Dopamine
    "C01CA07": 1,  # Dobutamine (inotrope)
    "C01CA02": 1,  # Isoprenaline (chronotropic)
    "H01BA01": 1,  # Vasopressin
    "H01BA04": 1,  # Terlipressin
}

register_mapping(
    TierMapping(
        name="vasopressor_escalation",
        n_levels=1,
        exact=_VASO_EXACT,
        prefix=[],
    )
)

# ---------------------------------------------------------------------------
# Sedation escalation (co-occurrence-based unified design)
# Tier 1 = anxiolysis / mild CNS effect
# Tier 2 = moderate sedation (dexmedetomidine, antipsychotics, esketamine alone)
# Tier 3 = deep sedation (propofol alone, midazolam, lorazepam)
# Tier 4 = general anesthesia (volatile, thiopental, etomidate)
# Tier 5 = GA + NMB (composite mode only; flat mapping cannot represent
#          co-occurrence, so Tier 5 is not reachable via this mapping)
# ---------------------------------------------------------------------------

_SEDATION_EXACT: Dict[str, int] = {
    # Tier 1 — Anxiolysis
    "N05CH01": 1,  # Melatonin
    "N05CF01": 1,  # Zopiclone
    "N05BA04": 1,  # Oxazepam
    "N05BA01": 1,  # Diazepam
    "N05BA02": 1,  # Chlordiazepoxide
    "C02AC01": 1,  # Clonidine
    # Tier 2 — Moderate sedation
    "N05CM18": 2,  # Dexmedetomidine (capped)
    "N05AD01": 2,  # Haloperidol
    "N05AH03": 2,  # Olanzapine
    "N05AH04": 2,  # Quetiapine
    "N01AX14": 2,  # Esketamine (alone; co-occurrence elevates in composite mode)
    "N01AX03": 2,  # Ketamine (alone)
    # Tier 3 — Deep sedation
    "N01AX10": 3,  # Propofol (alone; co-occurrence elevates in composite mode)
    "N05CD08": 3,  # Midazolam
    "N05BA06": 3,  # Lorazepam
    # Tier 4 — General anesthesia (fixed agents)
    "N01AF03": 4,  # Thiopental
    "N01AX07": 4,  # Etomidate
}

_SEDATION_PREFIX: List[Tuple[str, int]] = [
    ("N01AB", 4),  # Volatile anesthetics
]

register_mapping(
    TierMapping(
        name="sedation_escalation",
        n_levels=5,
        exact=_SEDATION_EXACT,
        prefix=_SEDATION_PREFIX,
    )
)

# ---------------------------------------------------------------------------
# NMBA escalation (unified — no bolus/infusion distinction)
# All peripheral NMBAs → tier 1.  Context inferred via co-occurrence
# with sedation agents in the composite feature pipeline.
# ---------------------------------------------------------------------------

_NMBA_EXACT: Dict[str, int] = {
    "M03AB01": 1,  # Suxamethonium
    "M03AC09": 1,  # Rocuronium
    "M03AC11": 1,  # Cisatracurium
    "M03AC03": 1,  # Vecuronium
}

register_mapping(
    TierMapping(
        name="nmba_escalation",
        n_levels=1,
        exact=_NMBA_EXACT,
        prefix=[],
    )
)

# ---------------------------------------------------------------------------
# Anticoagulation escalation (no dose — can't distinguish prophylactic
# from therapeutic LMWH without dose data)
# Tier 1 = heparins (LMWH + UFH, assumed prophylactic)
# Tier 2 = DOACs, warfarin (always therapeutic)
# ---------------------------------------------------------------------------

_ANTICOAG_EXACT: Dict[str, int] = {
    # Tier 1 — Heparins (assume prophylactic without dose)
    "B01AB10": 1,  # Tinzaparin
    "B01AB04": 1,  # Dalteparin
    "B01AB05": 1,  # Enoxaparin
    "B01AB01": 1,  # UFH
    # Tier 2 — Therapeutic oral anticoagulants
    "B01AF02": 2,  # Apixaban
    "B01AF01": 2,  # Rivaroxaban
    "B01AE07": 2,  # Dabigatran
    "B01AA03": 2,  # Warfarin
}

register_mapping(
    TierMapping(
        name="anticoag_escalation",
        n_levels=2,
        exact=_ANTICOAG_EXACT,
        prefix=[
            ("B01AB", 1),  # Other heparins → tier 1
            ("B01AF", 2),  # Other DOACs → tier 2
        ],
    )
)

# ---------------------------------------------------------------------------
# Antiplatelet escalation
# All antiplatelets → tier 1.  DAPT (dual) detected via n_distinct >= 2.
# ---------------------------------------------------------------------------

_ANTIPLATELET_EXACT: Dict[str, int] = {
    "B01AC06": 1,  # ASA
    "B01AC04": 1,  # Clopidogrel
    "B01AC24": 1,  # Ticagrelor
}

register_mapping(
    TierMapping(
        name="antiplatelet_escalation",
        n_levels=1,
        exact=_ANTIPLATELET_EXACT,
        prefix=[],
    )
)

# ---------------------------------------------------------------------------
# Diuretic escalation (no dose)
# Tier 1 = thiazide / K-sparing (chronic comorbidity)
# Tier 2 = loop diuretic (active fluid management)
# ---------------------------------------------------------------------------

_DIURETIC_EXACT: Dict[str, int] = {
    # Tier 2 — Loop diuretics
    "C03CA01": 2,  # Furosemide
    "C03CA02": 2,  # Bumetanide
}

register_mapping(
    TierMapping(
        name="diuretic_escalation",
        n_levels=2,
        exact=_DIURETIC_EXACT,
        prefix=[
            ("C03AA", 1),  # Thiazides
            ("C03AB", 1),  # Thiazide combinations
            ("C03DA", 1),  # K-sparing (spironolactone, eplerenone)
            ("C03CA", 2),  # Other loop diuretics
        ],
    )
)

# ---------------------------------------------------------------------------
# Insulin escalation (simplified — no unit checking in v1)
# All insulin → tier 1 (presence).  IV vs SC discrimination requires
# dose unit inspection (Phase 2).
# ---------------------------------------------------------------------------

register_mapping(
    TierMapping(
        name="insulin_escalation",
        n_levels=1,
        exact={},
        prefix=[
            ("A10A", 1),  # All insulin subtypes
        ],
    )
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_SPOT_CHECKS = {
    "antibiotic_escalation": {
        "J01DH02": 5,  # Meropenem
        "J01XA01": 4,  # Vancomycin
        "J01DC02": 3,  # Cefuroxime
        "J01XB01": 6,  # Colistin
        "J01CE01": 1,  # Benzylpenicillin
        "J01DD02": 5,  # Ceftazidime (antipseudomonal exception)
        "J01AA12": 6,  # Tigecycline (tetracycline exception)
        "J01CR05": 5,  # Pip-taz (not caught by J01CR prefix)
        "J01DD52": 6,  # Ceftazidime-avibactam (not tier 4 J01DD default)
    },
    "vasopressor_escalation": {
        "C01CA03": 1,  # Norepinephrine
        "C01CA24": 1,  # Epinephrine
        "H01BA01": 1,  # Vasopressin
    },
    "sedation_escalation": {
        "N05CH01": 1,  # Melatonin (anxiolysis)
        "C02AC01": 1,  # Clonidine (anxiolysis)
        "N05CM18": 2,  # Dexmedetomidine (moderate)
        "N01AX10": 3,  # Propofol (deep, alone)
        "N05CD08": 3,  # Midazolam (deep)
        "N01AF03": 4,  # Thiopental (GA)
        "N01AX07": 4,  # Etomidate (GA)
    },
    "nmba_escalation": {
        "M03AB01": 1,  # Suxamethonium
        "M03AC09": 1,  # Rocuronium
        "M03AC11": 1,  # Cisatracurium
        "M03AC03": 1,  # Vecuronium
    },
    "anticoag_escalation": {
        "B01AB10": 1,  # Tinzaparin (heparin)
        "B01AF02": 2,  # Apixaban (DOAC)
        "B01AA03": 2,  # Warfarin
    },
    "diuretic_escalation": {
        "C03CA01": 2,  # Furosemide (loop)
        "C03DA01": 1,  # Spironolactone (K-sparing) via C03DA prefix
    },
    "insulin_escalation": {
        "A10AE04": 1,  # Insulin glargine
        "A10AB05": 1,  # Insulin aspart
    },
}


def validate_mapping(mapping_name: str) -> bool:
    """Run spot-check assertions for a registered mapping.

    Returns True if all checks pass, raises AssertionError otherwise.
    """
    mapping = get_mapping(mapping_name)
    checks = _SPOT_CHECKS.get(mapping_name, {})
    for atc, expected_tier in checks.items():
        result = mapping.classify(atc)
        assert result == expected_tier, (
            f"Spot-check failed for {mapping_name}: "
            f"{atc} → {result}, expected {expected_tier}"
        )
    logger.info(f"All {len(checks)} spot-checks passed for '{mapping_name}'")
    return True
