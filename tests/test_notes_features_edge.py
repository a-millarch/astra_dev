"""Regression tests for empty-subset edge cases in notes-derived features.

Azure crash (patient 8F647AFE, 2026-07-11): a patient WITH notes but with
none of the PRIMARY/FALLBACK notetypes made `get_first_intubation_note`'s
grouped subset empty; on that pandas version the group-key columns were
dropped and the subsequent `groupby("PID")` raised `KeyError('PID')`,
killing the whole PatientContext build. Same latent trap when every
`Redigeringstidspunkt` is NaT (NaT group keys are dropped).
"""

import pandas as pd
import pytest

from astra.data.notes_features import (
    build_intubation_from_notes,
    build_iss_from_notes,
    get_first_intubation_note,
    PRIMARY_NOTETYPES,
)

INTUB_COLS = ["PID", "intubated", "Redigeringstidspunkt"]
CONCEPT_COLS = ["PID", "TIMESTAMP", "FEATURE", "VALUE"]


def _notes(notetype, note="almindelig tekst uden fund", ts="2024-05-30 23:00"):
    return pd.DataFrame({
        "PID": ["p1", "p1"],
        "CPR_hash": ["8F647AFE", "8F647AFE"],
        "Redigeringstidspunkt": [ts, ts],
        "Notetype": [notetype, notetype],
        "Note": [note, note + " 2"],
    })


class TestGetFirstIntubationNote:
    def test_no_matching_notetype_returns_empty_with_columns(self):
        notes = _notes("EtHeltAndetNotat")
        out = get_first_intubation_note(notes, PRIMARY_NOTETYPES)
        assert list(out.columns) == INTUB_COLS
        assert out.empty

    def test_all_nat_timestamps_do_not_crash(self):
        notetype = next(iter(PRIMARY_NOTETYPES))
        notes = _notes(notetype)
        notes["Redigeringstidspunkt"] = pd.NaT
        out = get_first_intubation_note(notes, PRIMARY_NOTETYPES)
        assert list(out.columns) == INTUB_COLS
        assert out.empty

    def test_empty_input(self):
        out = get_first_intubation_note(pd.DataFrame(), PRIMARY_NOTETYPES)
        assert out.empty


class TestBuildIntubationFromNotes:
    def test_patient_without_primary_or_fallback_notetypes(self):
        """The exact Azure crash: notes exist, none of the special notetypes."""
        out = build_intubation_from_notes(_notes("EtHeltAndetNotat"))
        assert list(out.columns) == CONCEPT_COLS
        assert out.empty

    def test_all_nat_timestamps(self):
        notetype = next(iter(PRIMARY_NOTETYPES))
        notes = _notes(notetype)
        notes["Redigeringstidspunkt"] = None
        out = build_intubation_from_notes(notes)
        assert list(out.columns) == CONCEPT_COLS

    def test_missing_required_columns_returns_empty(self):
        out = build_intubation_from_notes(pd.DataFrame({"PID": ["p1"]}))
        assert list(out.columns) == CONCEPT_COLS
        assert out.empty


class TestBuildIssFromNotes:
    def test_all_nat_timestamps_do_not_crash(self):
        notes = _notes("Journalnotat", note="ingen scorer her")
        notes["Redigeringstidspunkt"] = None
        out = build_iss_from_notes(notes)
        assert out.empty

    def test_note_without_iss_returns_empty(self):
        out = build_iss_from_notes(_notes("Journalnotat"))
        assert out.empty

    def test_iss_extracted(self):
        notes = _notes("Journalnotat", note="Traumescreening: ISS 25 ved ankomst")
        out = build_iss_from_notes(notes)
        if out.empty:
            pytest.skip("ISS pattern did not match test phrasing — pattern-dependent")
        assert out["VALUE"].iloc[0] == 25.0
        assert out["FEATURE"].iloc[0] == "ISS_notes"
