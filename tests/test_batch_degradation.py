"""Non-Azure batch pipeline enablement: the retraining lifecycle must run in
an environment with only flat raw CSVs — no azureml/mltable, possibly no R.

Covers the three fixes that make `python -m astra.make_data` viable for the
receiving team:
  1. make_data imports without the Azure ML SDK (collectors made optional)
  2. add_elixhauser degrades to ASMT_ELIX=0.0 instead of looping forever
     when R cannot produce computed_elix_df.csv
  3. define_historic_population derives the cohort seed from the raw
     Procedurer.csv (BWST1F trauma calls) when azureml is unavailable
"""

import os

import pandas as pd
import pytest

import astra.data.build_patient_info as bpi


class TestMakeDataImport:
    def test_imports_without_mltable(self):
        import astra.make_data  # noqa: F401 — would raise before the fix

    def test_missing_raw_files_raise_clear_error(self, tmp_path, monkeypatch):
        import astra.make_data as md
        monkeypatch.setattr(md, "collect_subsets", None)
        monkeypatch.chdir(tmp_path)
        cfg = {"default_load_filenames": ["VitaleVaerdier"],
               "large_load_filenames": ["Notater"]}
        with pytest.raises(FileNotFoundError, match="data/raw"):
            md.proces_raw_concepts(cfg)


class TestAddElixhauserDegradation:
    def test_degrades_to_zero_without_r(self, tmp_path, monkeypatch):
        """create_elixhauser (subprocess.call) fails silently without R —
        add_elixhauser must not retry forever."""
        monkeypatch.chdir(tmp_path)                       # no interim csv here
        monkeypatch.setattr(bpi, "create_elixhauser", lambda base: None)
        base = pd.DataFrame({"PID": [1, 2], "CPR_hash": ["a", "b"]})
        out = bpi.add_elixhauser(base)
        assert (out["ASMT_ELIX"] == 0.0).all()
        assert len(out) == 2

    def test_merges_when_csv_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs("data/interim", exist_ok=True)
        pd.DataFrame({"PID": [1], "elixscore": [5.0]}).to_csv(
            "data/interim/computed_elix_df.csv", index=False)
        base = pd.DataFrame({"PID": [1, 2]})
        out = bpi.add_elixhauser(base)
        assert out.loc[out["PID"] == 1, "ASMT_ELIX"].iloc[0] == 5.0
        assert out.loc[out["PID"] == 2, "ASMT_ELIX"].iloc[0] == 0.0  # missing → 0


class TestPythonElixhauser:
    """Batch Elixhauser now uses the same Python implementation as inference
    (train/serve consistency by construction); R is an env-gated escape hatch."""

    def _cohort(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs("data/raw", exist_ok=True)
        start = pd.Timestamp("2030-06-01 12:00")
        base = pd.DataFrame({
            "PID": [1, 2],
            "CPR_hash": ["pat_a", "pat_b"],
            "AGE": [60, 40],
            "start": [start, start],
            "end": [start + pd.Timedelta(days=3)] * 2,
        })
        # pat_a: CHF (DI509 -> I50) + metastatic cancer (DC784 -> C78),
        #        noted pre-admission, unresolved
        # pat_b: diagnosis noted AFTER admission -> excluded by the window
        pd.DataFrame({
            "CPR_hash": ["pat_a", "pat_a", "pat_b"],
            "Diagnosekode": ["DI509", "DC784", "DI509"],
            "Noteret_dato": ["2029-01-01", "2029-06-01", "2030-06-02"],
            "Løst_dato": [None, None, None],
        }).to_csv("data/raw/Diagnoser.csv", index=False)
        return base

    def test_batch_score_equals_inference_scorer(self, tmp_path, monkeypatch):
        from astra.inference.comorbidity import compute_elixhauser_vw

        base = self._cohort(tmp_path, monkeypatch)
        out = bpi.add_elixhauser(base)

        expected = compute_elixhauser_vw(["I50", "C78"])   # same codes, post-strip
        assert expected != 0.0                              # sanity: categories matched
        assert out.loc[out["PID"] == 1, "ASMT_ELIX"].iloc[0] == expected
        # pat_b's diagnosis is post-admission -> no qualifying codes -> 0.0
        assert out.loc[out["PID"] == 2, "ASMT_ELIX"].iloc[0] == 0.0

    def test_missing_diagnoser_degrades(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)                        # no Diagnoser.csv at all
        base = pd.DataFrame({
            "PID": [1], "CPR_hash": ["x"], "AGE": [50],
            "start": [pd.Timestamp("2030-01-01")],
            "end": [pd.Timestamp("2030-01-02")],
        })
        out = bpi.add_elixhauser(base)
        assert (out["ASMT_ELIX"] == 0.0).all()

    def test_env_var_routes_to_r_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASTRA_ELIX_USE_R", "1")
        called = {}
        monkeypatch.setattr(bpi, "create_elixhauser_r",
                            lambda base: called.setdefault("r", True))
        bpi.create_elixhauser(pd.DataFrame())
        assert called.get("r") is True


class TestDefineHistoricPopulationFallback:
    def _cfg(self, tmp_path):
        return {"population_file_path": str(tmp_path / "seed.csv"),
                "raw_file_path": "unused/"}

    def test_derives_from_raw_procedurer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bpi, "Dataset", None)
        monkeypatch.chdir(tmp_path)
        os.makedirs("data/raw", exist_ok=True)
        pd.DataFrame({
            "CPR_hash": ["t1", "t1", "other"],
            "ProcedureCode": ["BWST1F", "KABC10", "BWST1F"],
            "ServiceDatetime": ["2030-01-01 12:00"] * 3,   # alt column name
        }).to_csv("data/raw/Procedurer.csv", index=False)

        bpi.define_historic_population(self._cfg(tmp_path))
        seed = pd.read_csv(tmp_path / "seed.csv", index_col=0)
        assert len(seed) == 2                              # BWST1F rows only
        assert set(seed.columns) == {"CPR_hash", "ServiceDate"}

    def test_clear_error_without_any_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bpi, "Dataset", None)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Procedurer.csv"):
            bpi.define_historic_population(self._cfg(tmp_path))
