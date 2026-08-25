"""Unit tests for sedation and surgical tier co-occurrence logic.

Tests the _tag_records → _aggregate_signals → _derive_composite_tiers
pipeline from composite_features.py against 22 specification test cases.
"""

import pandas as pd
import pytest

from astra.data.composite_features import (
    _aggregate_signals,
    _derive_composite_tiers,
    _tag_records,
)

# ATC code reference for readability
MELATONIN = "N05CH01"
ZOPICLONE = "N05CF01"
PROPOFOL = "N01AX10"
SEVOFLURANE = "N01AB08"
ROCURONIUM = "M03AC09"
CISATRACURIUM = "M03AC11"
ESKETAMINE = "N01AX14"
MIDAZOLAM = "N05CD08"
THIOPENTAL = "N01AF03"
DEXMEDETOMIDINE = "N05CM18"
HALOPERIDOL = "N05AD01"
BUPIVACAINE = "N01BB01"
REMIFENTANIL = "N01AH06"
ALFENTANIL = "N01AH02"
SUFENTANIL = "N01AH03"
FENTANYL = "N01AH01"
OLANZAPINE = "N05AH03"
QUETIAPINE = "N05AH04"


def _make_bin(atc_codes: list[str]) -> pd.DataFrame:
    """Build a single-bin DataFrame for one patient from ATC codes."""
    return pd.DataFrame({
        "PID": [0] * len(atc_codes),
        "_bin_position": [0] * len(atc_codes),
        "ATC": atc_codes,
    })


def _compute_tiers(atc_codes: list[str]) -> tuple[int, int]:
    """Run the full tier pipeline and return (sedation_tier, surgical_tier)."""
    df = _make_bin(atc_codes)
    df = _tag_records(df)
    signals = _aggregate_signals(df)
    signals = _derive_composite_tiers(signals)
    row = signals.iloc[0]
    return int(row["sedation_tier"]), int(row["surgical_tier"])


# -- Specification test cases --

def test_01_melatonin_only():
    assert _compute_tiers([MELATONIN]) == (1, 0)


def test_02_no_sedation_agents():
    # Empty bin — need at least one record to flow through pipeline,
    # so we use a non-sedation agent (paracetamol N02BE01)
    sed, surg = _compute_tiers(["N02BE01"])
    assert sed == 0
    assert surg == 0


def test_03_propofol_alone():
    assert _compute_tiers([PROPOFOL]) == (3, 0)


def test_04_propofol_sevoflurane():
    assert _compute_tiers([PROPOFOL, SEVOFLURANE]) == (4, 2)


def test_05_propofol_sevoflurane_rocuronium():
    assert _compute_tiers([PROPOFOL, SEVOFLURANE, ROCURONIUM]) == (5, 2)


def test_06_esketamine_rocuronium():
    assert _compute_tiers([ESKETAMINE, ROCURONIUM]) == (5, 0)


def test_07_esketamine_alone():
    assert _compute_tiers([ESKETAMINE]) == (2, 0)


def test_08_midazolam_cisatracurium():
    assert _compute_tiers([MIDAZOLAM, CISATRACURIUM]) == (3, 0)


def test_09_propofol_cisatracurium():
    assert _compute_tiers([PROPOFOL, CISATRACURIUM]) == (5, 0)


def test_10_sevoflurane_alone():
    assert _compute_tiers([SEVOFLURANE]) == (4, 2)


def test_11_thiopental_alone():
    assert _compute_tiers([THIOPENTAL]) == (4, 3)


def test_12_dexmedetomidine():
    assert _compute_tiers([DEXMEDETOMIDINE]) == (2, 0)


def test_13_haloperidol_zopiclone():
    assert _compute_tiers([HALOPERIDOL, ZOPICLONE]) == (2, 0)


def test_14_bupivacaine_only():
    assert _compute_tiers([BUPIVACAINE]) == (0, 1)


def test_15_propofol_esketamine():
    assert _compute_tiers([PROPOFOL, ESKETAMINE]) == (4, 2)


def test_16_propofol_remifentanil():
    assert _compute_tiers([PROPOFOL, REMIFENTANIL]) == (4, 2)


def test_17_propofol_remifentanil_rocuronium():
    assert _compute_tiers([PROPOFOL, REMIFENTANIL, ROCURONIUM]) == (5, 2)


def test_18_esketamine_propofol_rocuronium():
    assert _compute_tiers([ESKETAMINE, PROPOFOL, ROCURONIUM]) == (5, 2)


def test_19_propofol_fentanyl():
    assert _compute_tiers([PROPOFOL, FENTANYL]) == (3, 0)


def test_20_propofol_fentanyl_rocuronium():
    assert _compute_tiers([PROPOFOL, FENTANYL, ROCURONIUM]) == (5, 0)


def test_21_esketamine_remifentanil():
    assert _compute_tiers([ESKETAMINE, REMIFENTANIL]) == (4, 0)


def test_22_propofol_sufentanil():
    assert _compute_tiers([PROPOFOL, SUFENTANIL]) == (4, 2)
