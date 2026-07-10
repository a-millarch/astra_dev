"""Tests for the FastAPI reference service (``astra.service``).

Self-contained: an injected mock predictor stands in for the real
:class:`AstraPredictor` — no model artifacts, no patient data, no network.
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from astra.inference.api import (  # noqa: E402
    ArtifactError,
    PatientNotFoundError,
    TimestampBeforeAdmissionError,
)
from astra.service.app import create_app  # noqa: E402
from astra.service.settings import ServiceSettings  # noqa: E402

SEQ_LEN = 10

VALID_PREDICT_BODY = {
    "patient_id": "abc123",
    "service_date": "2023-08-15",
    "timestamp": "2023-08-16 06:00",
}


# ---------------------------------------------------------------------------
# Mock payloads (minimal but complete w.r.t. the response models)
# ---------------------------------------------------------------------------

def _time_axis():
    return {
        "steps": list(range(SEQ_LEN)),
        "hours_start": [float(i) for i in range(SEQ_LEN)],
        "hours_end": [float(i + 1) for i in range(SEQ_LEN)],
        "bin_freq": ["1h"] * SEQ_LEN,
    }


def _prediction_payload():
    return {
        "patient_id": "abc123",
        "pid": "abc12320230815",
        "model_name": "mock",
        "is_temporal": True,
        "survival_mode": False,
        "calibration_method": None,
        "admission_time": "2023-08-15T12:00:00",
        "requested_time": "2023-08-16T06:00:00",
        "eval_hours": 5.0,
        "eval_step": 4,
        "trajectory_length": 6,
        "seq_len": SEQ_LEN,
        "probability": 0.42,
        "curve": {
            "steps": list(range(6)),
            "hours": [float(i + 1) for i in range(6)],
            # None = NaN gap, as produced by to_jsonable()
            "probabilities": [0.10, 0.15, None, 0.30, 0.38, 0.42],
            "source": "temporal_head",
            "survival": None,
        },
        "inhospital_start_hours": None,
        "compute_ms": 12.3,
    }


def _explanation_payload():
    zeros = [0.0] * SEQ_LEN
    return {
        "patient_id": "abc123",
        "pid": "abc12320230815",
        "model_name": "mock",
        "eval_step": 4,
        "eval_hours": 5.0,
        "trajectory_length": 6,
        "seq_len": SEQ_LEN,
        "time_axis": _time_axis(),
        "channels": ["HR", "SBP"],
        "ts_shap": [zeros, zeros],
        "ts_values": [[1.0, None] + [None] * (SEQ_LEN - 2), zeros],
        "channel_map": {"VitaleVaerdier": ["HR", "SBP"]},
        "cat_ts": {
            "labels": ["Medicin_A"],
            "shap_per_category": [zeros],
            "shap_aggregate": zeros,
            "values_per_category": None,
        },
        "static_cat": {"names": ["Sex"], "shap": [0.01], "values": ["M"]},
        "static_cont": {"names": ["Age"], "shap": [0.2], "values": [54.0]},
        "encoding_info": None,
        "top_features": [{"name": "SBP", "importance": 0.5}],
        "completeness": {"per_channel": {"HR": 0.5, "SBP": 0.0}, "overall": 0.25},
        "inhospital_start_step": None,
        "compute_ms": 45.6,
    }


def _differential_payload():
    zeros = [0.0] * SEQ_LEN
    return {
        "patient_id": "abc123",
        "pid": "abc12320230815",
        "model_name": "mock",
        "t1_hours": 1.0,
        "t2_hours": 6.0,
        "t1_step": 0,
        "t2_step": 5,
        "t1_probability": 0.2,
        "t2_probability": 0.4,
        "time_axis": _time_axis(),
        "channels": ["HR", "SBP"],
        "delta_ts_shap": [zeros, zeros],
        "delta_cat_ts": None,
        "delta_static_cat": None,
        "delta_static_cont": None,
        "top_delta_features": [{"name": "HR", "importance": 0.1}],
        "shap_t1": None,
        "shap_t2": None,
        "compute_ms": 99.0,
    }


# ---------------------------------------------------------------------------
# Mock predictor
# ---------------------------------------------------------------------------

class _ToDict:
    """Minimal stand-in for the facade's response dataclasses."""

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _MockSession:
    device = "cpu"


class MockPredictor:
    """Plain mock of the ``AstraPredictor`` surface used by the app."""

    model_name = "mock"
    is_temporal = True
    seq_len = SEQ_LEN

    def __init__(self, error=None):
        self.session = _MockSession()
        self.bundle = {"model_name": "mock"}
        self.ebm_result = None  # EBM disabled by default
        self._error = error

    def _maybe_raise(self):
        if self._error is not None:
            raise self._error

    def predict(self, patient_id, timestamp, service_date, *, include_curve=True):
        self._maybe_raise()
        payload = _prediction_payload()
        payload["patient_id"] = patient_id
        if not include_curve:
            payload["curve"] = None
        return _ToDict(payload)

    def explain(self, patient_id, timestamp, service_date, *,
                top_n=20, include_values=True):
        self._maybe_raise()
        # Returns a plain dict: exercises the app's dict-tolerant path.
        return _explanation_payload()

    def explain_differential(self, patient_id, service_date, t1_hours, t2_hours, *,
                             include_endpoints=False):
        self._maybe_raise()
        return _ToDict(_differential_payload())

    def explain_ebm(self, patient_id, timestamp, service_date):
        self._maybe_raise()
        return self.ebm_result

    def model_info(self):
        self._maybe_raise()
        return {"model_name": "mock", "is_temporal": True,
                "seq_len": SEQ_LEN, "ebm_enabled": False}


def _client(predictor=None) -> TestClient:
    app = create_app(
        settings=ServiceSettings(model_name="mock"),
        predictor=predictor if predictor is not None else MockPredictor(),
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_ok_with_injected_predictor():
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "mock"
    assert body["is_temporal"] is True
    assert body["device"] == "cpu"
    assert body["seq_len"] == SEQ_LEN


def test_health_degraded_when_model_load_fails(monkeypatch):
    def _boom(*args, **kwargs):
        raise ArtifactError("no artifacts on this machine")

    monkeypatch.setattr("astra.inference.api.AstraPredictor.load", _boom)
    app = create_app(settings=ServiceSettings(model_name="missing-model"))
    with TestClient(app) as client:  # context manager runs the lifespan -> load fails
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        # Inference endpoints must report 503 while degraded.
        r = client.post("/predict", json=VALID_PREDICT_BODY)
        assert r.status_code == 503
        assert "detail" in r.json()


def test_predict_happy_path():
    r = _client().post("/predict", json=VALID_PREDICT_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "abc123"
    assert body["probability"] == pytest.approx(0.42)
    assert body["is_temporal"] is True
    curve = body["curve"]
    assert curve["source"] == "temporal_head"
    assert len(curve["steps"]) == len(curve["hours"]) == len(curve["probabilities"]) == 6
    assert curve["probabilities"][2] is None  # NaN gap survives as null
    assert curve["probabilities"][-1] == pytest.approx(0.42)


def test_predict_without_curve():
    r = _client().post("/predict", json={**VALID_PREDICT_BODY, "include_curve": False})
    assert r.status_code == 200
    assert r.json()["curve"] is None


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (PatientNotFoundError("no data for patient abc123"), 404),
        (TimestampBeforeAdmissionError("timestamp precedes trajectory start"), 422),
        (ArtifactError("bundle unreadable"), 503),
        (ValueError("Unparseable timestamp: 'garbage'"), 422),
    ],
)
def test_predict_error_mapping(error, expected_status):
    client = _client(MockPredictor(error=error))
    r = client.post("/predict", json=VALID_PREDICT_BODY)
    assert r.status_code == expected_status
    assert str(error) in r.json()["detail"]


def test_model_info_returns_mock_dict():
    r = _client().get("/model/info")
    assert r.status_code == 200
    assert r.json() == {"model_name": "mock", "is_temporal": True,
                        "seq_len": SEQ_LEN, "ebm_enabled": False}


def test_predict_missing_patient_id_is_422():
    body = {k: v for k, v in VALID_PREDICT_BODY.items() if k != "patient_id"}
    r = _client().post("/predict", json=body)
    assert r.status_code == 422  # FastAPI request validation


def test_explain_happy_path():
    r = _client().post("/explain", json={**VALID_PREDICT_BODY, "top_n": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["channels"] == ["HR", "SBP"]
    assert body["time_axis"]["steps"] == list(range(SEQ_LEN))
    assert body["ts_values"][0][1] is None  # unmeasured value survives as null
    assert body["top_features"][0] == {"name": "SBP", "importance": 0.5}
    assert body["static_cat"]["values"] == ["M"]


def test_explain_differential_happy_path():
    r = _client().post(
        "/explain/differential",
        json={"patient_id": "abc123", "service_date": "2023-08-15",
              "t1_hours": 1.0, "t2_hours": 6.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["t1_probability"] == pytest.approx(0.2)
    assert body["t2_probability"] == pytest.approx(0.4)
    assert body["t2_step"] == 5
    assert len(body["delta_ts_shap"]) == 2


def test_explain_ebm_disabled_returns_400():
    r = _client().post("/explain/ebm", json=VALID_PREDICT_BODY)
    assert r.status_code == 400
    assert "EBM" in r.json()["detail"]


def test_explain_ebm_enabled_returns_payload():
    predictor = MockPredictor()
    predictor.ebm_result = {
        "intervals": [{"hours": 1.0, "contributions": {"HR": 0.1}}]
    }
    r = _client(predictor).post("/explain/ebm", json=VALID_PREDICT_BODY)
    assert r.status_code == 200
    assert r.json() == predictor.ebm_result
