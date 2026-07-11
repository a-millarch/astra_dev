"""AstraPredictor facade tests with a stubbed InferenceSession.

No data, no model artifacts: the fake session returns deterministic
predictions and SHAP values; contexts are faked at the _build_entry seam.
Covers curve unification (temporal vs simulation), LRU caching,
earlier-timestamp censoring, error mapping and payload JSON-safety.
"""

import json

import numpy as np
import pandas as pd
import pytest

from astra.evaluation.utils import get_total_steps, time_to_step
from astra.inference.api import (
    AstraPredictor,
    TimestampBeforeAdmissionError,
)
from astra.inference.pipeline import InferenceResult, SHAPResult

SMALL_CONFIG = {
    "bin_intervals": {"1h": "10min", "3h": "30min", "end": "1h"},
    "bin_freq_include": ["10min", "30min", "1h"],
    # Real bundles store per-channel metadata (see _build_channel_map)
    "channel_map": {
        "HR": {"concept": "VitaleVaerdier", "feature": "HR",
               "agg_func": "mean", "type": "continuous"},
        "SBP": {"concept": "VitaleVaerdier", "feature": "SBP",
                "agg_func": "mean", "type": "continuous"},
    },
}
SEQ_LEN = get_total_steps(data_config=SMALL_CONFIG)   # 10
ADMISSION = pd.Timestamp("2030-01-01 12:00:00")
CHANNELS = ["HR", "SBP"]


def make_bundle(temporal=True):
    return {
        "model_name": "fake",
        "model_params": {
            "seq_len": SEQ_LEN,
            "temporal_head": temporal,
            "survival_mode": False,
            "classes": {"SEX": ["#na#", "Male", "Female"]},
        },
        "ts_channel_names": list(CHANNELS),
        "tab_feature_names": ["AGE"],
        "encoding_info": {
            "feature_ranges": {"Medicin": (0, 2)},
            "category_labels": {"Medicin": ["ATC_A", "ATC_B"]},
        },
        "data_config": SMALL_CONFIG,
        "shap_background": None,
    }


class FakeCtx:
    def __init__(self, current_time=None):
        self.pid = "fakepid"
        self.admission_time = ADMISSION
        self.current_time = current_time or ADMISSION
        self.patient_end_time = ADMISSION + pd.Timedelta(days=30)
        self.demographics = {}
        self.tab_df = pd.DataFrame({"SEX": ["Male"], "AGE": [54.0]})
        # Raw continuous TS: NaN = missing within trajectory, 0 = padding
        self.x_ts = np.zeros((len(CHANNELS), SEQ_LEN))
        self.x_ts[0, 0] = 82.0
        self.x_ts[1, 0] = 120.0
        self.x_ts[0, 1] = np.nan
        self.x_ts_cat = np.zeros((2, SEQ_LEN))
        self._sync_length()

    def _sync_length(self):
        delta_min = (self.current_time - self.admission_time).total_seconds() / 60.0
        step = time_to_step(delta_min, "min", data_config=SMALL_CONFIG)
        step = SEQ_LEN - 1 if step is None else step
        self.trajectory_length = int(min(step + 1, SEQ_LEN))

    def refresh(self, current_time, new_data=None):
        assert current_time >= self.current_time, "facade must never move backwards"
        self.current_time = pd.Timestamp(current_time)
        self._sync_length()


class FakeRunner:
    """Mimics SimulationRunner: fills the prediction curve while advancing."""

    def __init__(self, ctx):
        self.context = ctx
        self._prediction_curve = np.full(SEQ_LEN, np.nan)

    def advance_to(self, hours=None, time=None):
        target = (self.context.admission_time + pd.Timedelta(hours=hours)
                  if hours is not None else pd.Timestamp(time))
        self.context.refresh(target)
        n = self.context.trajectory_length
        self._prediction_curve[:n] = _ramp()[:n]


def _ramp():
    return np.linspace(0.05, 0.95, SEQ_LEN)


class FakeSession:
    def __init__(self, temporal=True):
        self.bundle = make_bundle(temporal)
        self.is_temporal = temporal
        self.device = "cpu"
        self._calibration_method = None
        self.predict_calls = 0

    def predict_from_context(self, ctx, censor_step=None, profiling=None):
        self.predict_calls += 1
        step = censor_step if censor_step is not None else ctx.trajectory_length - 1
        traj = min(step + 1, ctx.trajectory_length) if censor_step is not None else ctx.trajectory_length
        probs = _ramp()
        return InferenceResult(
            pid=ctx.pid,
            probability=float(probs[step]),
            trajectory_length=traj,
            censor_step=step,
            predictions_over_time=probs if self.is_temporal else None,
        )

    def explain_from_context(self, ctx, censor_step=None):
        step = censor_step if censor_step is not None else ctx.trajectory_length - 1
        arr = np.arange(SEQ_LEN, dtype=float) / 100.0
        return SHAPResult(
            pid=ctx.pid,
            ts_shap={"HR": arr, "SBP": -arr},
            cat_ts_shap={"ATC_A": arr * 0.1},
            static_cat_shap={"SEX": 0.05},
            static_cont_shap={"AGE": 0.3},
            top_features=[("AGE", 0.3), ("HR", 0.2)],
            eval_timestep=step,
        )

    # Reuse the real array assembler so the facade is tested against the
    # exact structure the dashboard consumes.
    def shap_to_viz_dict(self, shap_result, x_ts, x_ts_cat, tab_df):
        from astra.inference.pipeline import InferenceSession
        return InferenceSession.shap_to_viz_dict(
            self, shap_result, x_ts=x_ts, x_ts_cat=x_ts_cat, tab_df=tab_df)


@pytest.fixture
def temporal_predictor(monkeypatch):
    predictor = AstraPredictor(FakeSession(temporal=True), context_cache_size=2)
    predictor._build_entry = lambda pid, sd: {"ctx": FakeCtx(), "runner": None}
    return predictor


@pytest.fixture
def simulation_predictor():
    predictor = AstraPredictor(FakeSession(temporal=False), context_cache_size=2)

    def _build(pid, sd):
        ctx = FakeCtx()
        return {"ctx": ctx, "runner": FakeRunner(ctx)}

    predictor._build_entry = _build
    return predictor


class TestTemporalPredict:
    def test_curve_and_probability(self, temporal_predictor):
        ts = ADMISSION + pd.Timedelta(minutes=125)   # inside step 8 (bin 120-150min)
        resp = temporal_predictor.predict("pat1", ts, "2030-01-01")
        d = resp.to_dict()
        json.dumps(d)
        assert d["is_temporal"] is True
        assert d["curve"]["source"] == "temporal_head"
        assert len(d["curve"]["probabilities"]) == d["trajectory_length"]
        assert d["probability"] == pytest.approx(_ramp()[d["eval_step"]])
        assert d["seq_len"] == SEQ_LEN
        assert d["curve"]["hours"] == temporal_predictor.time_axis.hours_end[:d["trajectory_length"]]

    def test_earlier_timestamp_uses_censoring_not_rebuild(self, temporal_predictor):
        late = ADMISSION + pd.Timedelta(hours=2.9)
        early = ADMISSION + pd.Timedelta(minutes=35)  # step 3
        r_late = temporal_predictor.predict("pat1", late, "2030-01-01")
        ctx_before = temporal_predictor._entries[("pat1", "2030-01-01")]["ctx"]
        r_early = temporal_predictor.predict("pat1", early, "2030-01-01")
        ctx_after = temporal_predictor._entries[("pat1", "2030-01-01")]["ctx"]
        assert ctx_before is ctx_after                     # cached, not rebuilt
        assert r_early.eval_step < r_late.eval_step
        assert r_early.probability == pytest.approx(_ramp()[r_early.eval_step])
        # context never moved backwards
        assert ctx_after.current_time == min(late, ctx_after.patient_end_time)

    def test_pre_admission_timestamp_raises(self, temporal_predictor):
        with pytest.raises(TimestampBeforeAdmissionError):
            temporal_predictor.predict(
                "pat1", ADMISSION - pd.Timedelta(hours=1), "2030-01-01")

    def test_include_curve_false(self, temporal_predictor):
        resp = temporal_predictor.predict(
            "pat1", ADMISSION + pd.Timedelta(hours=1), "2030-01-01",
            include_curve=False)
        assert resp.curve is None
        assert 0.0 <= resp.probability <= 1.0

    def test_service_date_required(self, temporal_predictor):
        with pytest.raises(ValueError):
            temporal_predictor.predict("pat1", ADMISSION, None)


class TestLRU:
    def test_eviction(self, temporal_predictor):
        builds = []
        orig = temporal_predictor._build_entry

        def counting(pid, sd):
            builds.append(pid)
            return orig(pid, sd)

        temporal_predictor._build_entry = counting
        t = ADMISSION + pd.Timedelta(hours=1)
        for pid in ["a", "b", "c"]:                 # cache_size=2 → "a" evicted
            temporal_predictor.predict(pid, t, "2030-01-01")
        assert ("a", "2030-01-01") not in temporal_predictor._entries
        temporal_predictor.predict("a", t, "2030-01-01")
        assert builds == ["a", "b", "c", "a"]

    def test_clear_cache(self, temporal_predictor):
        t = ADMISSION + pd.Timedelta(hours=1)
        temporal_predictor.predict("a", t, "2030-01-01")
        temporal_predictor.clear_cache("a")
        assert not temporal_predictor._entries
        temporal_predictor.predict("a", t, "2030-01-01")
        temporal_predictor.clear_cache()
        assert not temporal_predictor._entries


class TestSimulationPredict:
    def test_curve_source_and_lookup(self, simulation_predictor):
        ts = ADMISSION + pd.Timedelta(minutes=125)
        resp = simulation_predictor.predict("pat1", ts, "2030-01-01")
        d = resp.to_dict()
        json.dumps(d)
        assert d["curve"]["source"] == "simulation"
        assert d["probability"] == pytest.approx(_ramp()[d["eval_step"]])
        # Steps beyond the advanced point are None, within are floats
        assert all(p is None or isinstance(p, float)
                   for p in d["curve"]["probabilities"])

    def test_historical_query_reads_stored_curve(self, simulation_predictor):
        late = ADMISSION + pd.Timedelta(hours=2.9)
        simulation_predictor.predict("pat1", late, "2030-01-01")
        session = simulation_predictor.session
        calls_before = session.predict_calls
        early = ADMISSION + pd.Timedelta(minutes=35)
        resp = simulation_predictor.predict("pat1", early, "2030-01-01")
        # Served from the stored curve — no extra model forward
        assert session.predict_calls == calls_before
        assert resp.probability == pytest.approx(_ramp()[resp.eval_step])


class TestExplain:
    def test_payload_structure(self, temporal_predictor):
        ts = ADMISSION + pd.Timedelta(hours=1)
        resp = temporal_predictor.explain("pat1", ts, "2030-01-01", top_n=1)
        d = resp.to_dict()
        json.dumps(d)
        assert d["channels"] == CHANNELS
        assert len(d["ts_shap"]) == len(CHANNELS)
        assert len(d["ts_shap"][0]) == SEQ_LEN
        assert d["ts_values"][0][0] == 82.0
        assert d["ts_values"][0][1] is None               # NaN → None
        assert d["channel_map"]["HR"]["concept"] == "VitaleVaerdier"
        assert d["cat_ts"]["labels"] == ["ATC_A", "ATC_B"]
        assert d["static_cat"]["values"] == ["Male"]
        assert d["static_cont"]["values"] == [54.0]
        assert d["top_features"] == [{"name": "AGE", "importance": 0.3}]
        assert 0.0 <= d["completeness"]["overall"] <= 1.0
        assert d["time_axis"]["steps"] == list(range(SEQ_LEN))
        assert d["eval_hours"] == temporal_predictor.time_axis.hours_end[d["eval_step"]]

    def test_completeness_counts_measured_fraction(self, temporal_predictor):
        ts = ADMISSION + pd.Timedelta(minutes=15)         # only 2 steps visible
        resp = temporal_predictor.explain("pat1", ts, "2030-01-01")
        per = resp.completeness["per_channel"]
        assert set(per) == set(CHANNELS)
        assert all(0.0 <= v <= 1.0 for v in per.values())


class TestConfigPlumbing:
    def test_cfg_passed_to_from_csv(self, monkeypatch):
        """A config given to the predictor must reach PatientContext.from_csv —
        the global cfg is NOT updated by get_cfg(path), so explicit plumbing
        is the only way a custom config takes effect."""
        from astra.inference.patient_context import PatientContext

        sentinel_cfg = {"model_name": "fake", "sentinel": True}
        captured = {}

        def fake_from_csv(**kwargs):
            captured.update(kwargs)
            return FakeCtx()

        monkeypatch.setattr(PatientContext, "from_csv", staticmethod(fake_from_csv))
        predictor = AstraPredictor(FakeSession(temporal=True), cfg=sentinel_cfg)
        predictor.predict("pat1", ADMISSION + pd.Timedelta(hours=1), "2030-01-01")
        assert captured["cfg"] is sentinel_cfg
        assert captured["start_hours"] == 0.0

    def test_load_requires_model_name_or_config(self):
        with pytest.raises(ValueError, match="model_name is required"):
            AstraPredictor.load(None, artifacts_dir="Z:/nowhere")

    def test_load_reads_model_name_from_config(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "exp.yaml"
        cfg_file.write_text("model_name: cfg_model\n", encoding="utf-8")

        captured = {}

        def fake_session_load(model_name, device=None, bundle_dir=None,
                              weights_dir=None):
            captured["model_name"] = model_name
            return FakeSession(temporal=True)

        from astra.inference import pipeline
        monkeypatch.setattr(pipeline.InferenceSession, "load",
                            staticmethod(fake_session_load))
        predictor = AstraPredictor.load(config_path=str(cfg_file))
        assert captured["model_name"] == "cfg_model"
        assert predictor._cfg["model_name"] == "cfg_model"


class TestExplainViz:
    def test_returns_dashboard_format(self, temporal_predictor):
        ts = ADMISSION + pd.Timedelta(hours=1)
        shap_dict, ch2f, ncat, ncont = temporal_predictor.explain_viz(
            "pat1", ts, "2030-01-01")
        assert shap_dict["ts_shap"].shape == (1, len(CHANNELS), SEQ_LEN)
        assert ch2f == {i: c for i, c in enumerate(CHANNELS)}
        assert ncat == ["SEX"] and ncont == ["AGE"]
        assert "test_data" in shap_dict and "trajectory_length" in shap_dict


class TestModelInfo:
    def test_json_safe_and_content(self, temporal_predictor):
        info = temporal_predictor.model_info()
        json.dumps(info)
        assert info["model_name"] == "fake"
        assert info["seq_len"] == SEQ_LEN
        assert info["channels"] == CHANNELS
        assert info["time_axis"]["steps"] == list(range(SEQ_LEN))
        assert info["static_categorical"] == ["SEX"]
