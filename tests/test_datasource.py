"""Tests for the PatientDataSource seam: registry dispatch in load_patient_csv,
byte-identical legacy behavior when unregistered, and the prehospital hook."""

import numpy as np
import pandas as pd
import pytest

from astra.inference.datasource import CSVDataSource, InMemoryDataSource, PatientDataSource
from astra.inference.patient_store import (
    load_patient_csv,
    set_data_source,
    get_data_source,
    clear_data_source,
)

CPR = "abcdef1234567890"


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_data_source()
    yield
    clear_data_source()


@pytest.fixture
def vitals_df():
    return pd.DataFrame({
        "CPR_hash": [CPR, CPR],
        "Registreringstidspunkt": ["2023-08-15 11:00:00", "2023-08-15 11:10:00"],
        "Vital_parametre": ["Puls", "BT systolisk"],
        "Værdi": ["82", "118"],
    })


@pytest.fixture
def csv_dirs(tmp_path, vitals_df):
    """Population CSV containing our patient + one other."""
    data_dir = tmp_path / "raw"
    patient_dir = tmp_path / "patients"
    data_dir.mkdir()
    other = vitals_df.copy()
    other["CPR_hash"] = "other000"
    pd.concat([vitals_df, other]).to_csv(data_dir / "VitaleVaerdier.csv", index=True)
    return str(data_dir), str(patient_dir)


class TestRegistry:
    def test_set_get_clear(self, vitals_df):
        src = InMemoryDataSource({CPR: {"VitaleVaerdier": vitals_df}})
        set_data_source(src)
        assert get_data_source() is src
        clear_data_source()
        assert get_data_source() is None

    def test_registered_source_is_authoritative(self, vitals_df):
        set_data_source(InMemoryDataSource({CPR: {"VitaleVaerdier": vitals_df}}))
        # Bogus dirs prove the filesystem is never touched.
        df = load_patient_csv(CPR, "VitaleVaerdier",
                              data_dir="Z:/does/not/exist",
                              patient_dir="Z:/nope")
        assert len(df) == 2
        assert set(df["CPR_hash"]) == {CPR}

    def test_missing_concept_raises_filenotfound(self, vitals_df):
        set_data_source(InMemoryDataSource({CPR: {"VitaleVaerdier": vitals_df}}))
        with pytest.raises(FileNotFoundError):
            load_patient_csv(CPR, "Labsvar", data_dir="Z:/nope")

    def test_source_result_filtered_to_patient(self):
        mixed = pd.DataFrame({
            "CPR_hash": [CPR, "someone_else"],
            "Værdi": ["1", "2"],
        })
        set_data_source(InMemoryDataSource({CPR: {"VitaleVaerdier": mixed}}))
        df = load_patient_csv(CPR, "VitaleVaerdier")
        assert set(df["CPR_hash"]) == {CPR}


class TestLegacyFileBehavior:
    def test_population_fallback_and_filter(self, csv_dirs):
        data_dir, patient_dir = csv_dirs
        df = load_patient_csv(CPR, "VitaleVaerdier",
                              data_dir=data_dir, patient_dir=patient_dir,
                              cache=False, index_col=0)
        assert len(df) == 2
        assert set(df["CPR_hash"]) == {CPR}

    def test_cache_writes_per_patient_file(self, csv_dirs):
        import os
        data_dir, patient_dir = csv_dirs
        load_patient_csv(CPR, "VitaleVaerdier",
                         data_dir=data_dir, patient_dir=patient_dir,
                         cache=True, index_col=0)
        assert os.path.isfile(os.path.join(patient_dir, CPR, "VitaleVaerdier.csv"))

    def test_missing_file_raises(self, csv_dirs):
        data_dir, patient_dir = csv_dirs
        with pytest.raises(FileNotFoundError):
            load_patient_csv(CPR, "Labsvar",
                             data_dir=data_dir, patient_dir=patient_dir)

    def test_csv_datasource_bypasses_registered_source(self, csv_dirs, vitals_df):
        """CSVDataSource.fetch must hit the files even when another source
        is registered (no recursion, no shadowing)."""
        data_dir, patient_dir = csv_dirs
        decoy = pd.DataFrame({"CPR_hash": [CPR], "Værdi": ["999"]})
        set_data_source(InMemoryDataSource({CPR: {"VitaleVaerdier": decoy}}))

        csv_src = CSVDataSource(data_dir=data_dir, patient_dir=patient_dir, cache=False)
        df = csv_src.fetch(CPR, "VitaleVaerdier")
        assert len(df) == 2                      # file rows, not the decoy
        assert csv_src.fetch(CPR, "Labsvar") is None


class TestInMemoryDataSource:
    def test_fetch_returns_copy(self, vitals_df):
        src = InMemoryDataSource({CPR: {"VitaleVaerdier": vitals_df}})
        out = src.fetch(CPR, "VitaleVaerdier")
        out.loc[:, "Værdi"] = "mutated"
        assert list(vitals_df["Værdi"]) == ["82", "118"]

    def test_satisfies_protocol(self, vitals_df):
        src = InMemoryDataSource({CPR: {"VitaleVaerdier": vitals_df}})
        assert isinstance(src, PatientDataSource)

    def test_fetch_prehospital(self):
        ts = pd.Timestamp("2023-08-15 09:45:00")
        src = InMemoryDataSource({}, prehospital={CPR: {"prehospital_start": ts, "A": "Fri"}})
        rec = src.fetch_prehospital(CPR)
        assert rec["prehospital_start"] == ts
        assert src.fetch_prehospital("unknown") is None


class TestComputedIssMatching:
    """computed_iss_df.csv is keyed by cohort PIDs (sequential ints); the
    inference PID is hash8+date — matching must go through CPR_hash."""

    def _iss_df(self):
        return pd.DataFrame({
            "PID": [101, 102, 103],                    # cohort enumeration
            "CPR_hash": [CPR, CPR, "other000"],        # CPR has two encounters
            "start": ["2023-08-15 10:30:00", "2024-01-02 08:00:00",
                      "2023-05-01 00:00:00"],
            "riss": [17.0, 9.0, 25.0],
            "niss": [22.0, 12.0, 30.0],
        })

    def test_matches_by_cpr_hash_not_pid(self):
        from astra.inference.data_prep import _match_computed_iss
        rows = _match_computed_iss(
            self._iss_df(), CPR, "abcdef1220230815",
            pd.Timestamp("2023-08-15 10:30:00"))
        assert len(rows) == 1
        assert rows["riss"].iloc[0] == 17.0

    def test_picks_nearest_encounter(self):
        from astra.inference.data_prep import _match_computed_iss
        rows = _match_computed_iss(
            self._iss_df(), CPR, "abcdef1220240102",
            pd.Timestamp("2024-01-02 07:45:00"))
        assert rows["riss"].iloc[0] == 9.0

    def test_legacy_pid_fallback_without_cpr_column(self):
        from astra.inference.data_prep import _match_computed_iss
        df = self._iss_df().drop(columns=["CPR_hash"])
        df.loc[0, "PID"] = "abcdef1220230815"          # regenerated with inference PIDs
        rows = _match_computed_iss(
            df, CPR, "abcdef1220230815", pd.Timestamp("2023-08-15"))
        assert len(rows) == 1
        assert rows["riss"].iloc[0] == 17.0

    def test_unknown_patient_returns_empty(self):
        from astra.inference.data_prep import _match_computed_iss
        rows = _match_computed_iss(
            self._iss_df(), "nobody", "nope", pd.Timestamp("2023-01-01"))
        assert rows.empty


class TestPrehospitalHook:
    def _base_df(self):
        start = pd.Timestamp("2023-08-15 10:30:00")
        return pd.DataFrame({
            "PID": ["p1"],
            "CPR_hash": [CPR],
            "start": [start],
            "end": [start + pd.Timedelta(days=2)],
        })

    def test_hook_shifts_start_and_sets_abcd(self):
        from astra.inference.data_prep import _apply_prehospital_start

        ph_start = pd.Timestamp("2023-08-15 09:40:00")
        set_data_source(InMemoryDataSource(
            {}, prehospital={CPR: {"prehospital_start": ph_start, "A": "Fri", "C": "Puls normal"}},
        ))
        out = _apply_prehospital_start(self._base_df(), CPR, cfg={})
        assert out["start"].iloc[0] == ph_start
        assert out["inhospital_start"].iloc[0] == pd.Timestamp("2023-08-15 10:30:00")
        assert out["A"].iloc[0] == "Fri"
        assert out["C"].iloc[0] == "Puls normal"

    def test_hook_no_record_keeps_start(self):
        from astra.inference.data_prep import _apply_prehospital_start

        set_data_source(InMemoryDataSource({}, prehospital={}))
        out = _apply_prehospital_start(self._base_df(), CPR, cfg={})
        assert out["start"].iloc[0] == pd.Timestamp("2023-08-15 10:30:00")
        assert out["prehospital_start"].iloc[0] is None

    def test_source_without_capability_uses_legacy_path(self, tmp_path):
        """A source lacking fetch_prehospital falls through to the batch
        base_df lookup — which warns and no-ops when the pkl is absent."""
        from astra.inference.data_prep import _apply_prehospital_start

        set_data_source(InMemoryDataSource({}))          # has fetch_prehospital
        src = CSVDataSource(str(tmp_path), str(tmp_path))  # does NOT
        set_data_source(src)
        base = self._base_df()
        out = _apply_prehospital_start(
            base, CPR, cfg={"base_df_path": str(tmp_path / "missing.pkl")})
        assert out["start"].iloc[0] == base["start"].iloc[0]
        assert "prehospital_start" not in out.columns or pd.isna(out.get("prehospital_start", pd.Series([None])).iloc[0])
