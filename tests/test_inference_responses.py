"""Unit tests for astra.inference.responses — JSON safety of the API wire format."""

import json

import numpy as np
import pandas as pd
import pytest

from astra.inference.responses import (
    to_jsonable,
    format_top_features,
    TimeAxis,
    ProbabilityCurve,
    PredictionResponse,
    ExplanationResponse,
    StaticFeatureBlock,
    CategoricalTSBlock,
)


class TestToJsonable:
    def test_scalars(self):
        assert to_jsonable(None) is None
        assert to_jsonable("x") == "x"
        assert to_jsonable(True) is True
        assert to_jsonable(3) == 3
        assert to_jsonable(3.5) == 3.5

    def test_nan_and_inf_become_none(self):
        assert to_jsonable(float("nan")) is None
        assert to_jsonable(float("inf")) is None
        assert to_jsonable(np.float32("nan")) is None
        assert to_jsonable(np.float64("-inf")) is None

    def test_numpy_scalars(self):
        assert to_jsonable(np.int64(7)) == 7
        assert isinstance(to_jsonable(np.int64(7)), int)
        assert to_jsonable(np.float32(1.5)) == 1.5
        assert to_jsonable(np.bool_(True)) is True

    def test_arrays_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        assert to_jsonable(arr) == [1.0, None, 3.0]

    def test_nested_2d_array(self):
        arr = np.array([[1.0, np.nan], [np.inf, 4.0]])
        assert to_jsonable(arr) == [[1.0, None], [None, 4.0]]

    def test_timestamps(self):
        ts = pd.Timestamp("2023-08-15 10:30:00")
        assert to_jsonable(ts) == "2023-08-15T10:30:00"
        assert to_jsonable(pd.NaT) is None

    def test_dict_with_numpy_keys_and_tuples(self):
        d = {np.int64(0): "HR", "range": (3, 7)}
        out = to_jsonable(d)
        assert out == {0: "HR", "range": [3, 7]}
        json.dumps(out)  # must not raise

    def test_timedelta(self):
        assert to_jsonable(pd.Timedelta(hours=1)) == 3600.0


class TestFormatTopFeatures:
    def test_basic(self):
        out = format_top_features([("HR", np.float32(0.5)), ("SBP", 0.25)])
        assert out == [
            {"name": "HR", "importance": 0.5},
            {"name": "SBP", "importance": 0.25},
        ]

    def test_empty_and_none(self):
        assert format_top_features([]) == []
        assert format_top_features(None) == []


def _mini_time_axis(n=4):
    return TimeAxis(
        steps=list(range(n)),
        hours_start=[i * 0.5 for i in range(n)],
        hours_end=[(i + 1) * 0.5 for i in range(n)],
        bin_freq=["30min"] * n,
    )


class TestResponseRoundTrips:
    def test_probability_curve(self):
        curve = ProbabilityCurve(
            steps=[0, 1, 2],
            hours=[0.5, 1.0, 1.5],
            probabilities=np.array([0.1, np.nan, 0.3]),
            source="simulation",
        )
        d = curve.to_dict()
        json.dumps(d)
        assert d["probabilities"] == [0.1, None, 0.3]
        assert d["source"] == "simulation"

    def test_prediction_response(self):
        resp = PredictionResponse(
            patient_id="abc123",
            pid="abc12320230815",
            model_name="m",
            is_temporal=True,
            survival_mode=False,
            calibration_method=None,
            admission_time="2023-08-15T10:30:00",
            requested_time="2023-08-16T06:00:00",
            eval_hours=19.5,
            eval_step=42,
            trajectory_length=43,
            seq_len=124,
            probability=np.float32(0.234),
            curve=ProbabilityCurve(
                steps=[0], hours=[0.5],
                probabilities=np.array([np.nan]), source="temporal_head",
            ),
        )
        d = resp.to_dict()
        json.dumps(d)
        assert d["probability"] == pytest.approx(0.234, abs=1e-6)
        assert d["curve"]["probabilities"] == [None]

    def test_explanation_response_with_numpy_fields(self):
        resp = ExplanationResponse(
            patient_id="abc",
            pid="p",
            model_name="m",
            eval_step=1,
            eval_hours=1.0,
            trajectory_length=2,
            seq_len=4,
            time_axis=_mini_time_axis(),
            channels=["HR", "SBP"],
            ts_shap=np.array([[0.1, 0.2, 0.0, 0.0], [0.0, np.nan, 0.0, 0.0]]),
            ts_values=np.array([[80.0, np.nan, 0.0, 0.0], [120.0, 118.0, 0.0, 0.0]]),
            cat_ts=CategoricalTSBlock(
                labels=["ATC_A", "ATC_B"],
                shap_per_category=np.zeros((2, 4)),
                shap_aggregate=np.zeros(4),
            ),
            static_cat=StaticFeatureBlock(
                names=["SEX"], shap=np.array([0.05]), values=["Male"]),
            static_cont=StaticFeatureBlock(
                names=["AGE"], shap=np.array([0.3]), values=[np.float32(54.0)]),
            encoding_info={"feature_ranges": {"Medicin": (0, 2)}},
            top_features=format_top_features([("AGE", 0.3)]),
            completeness={"per_channel": {"HR": 0.5, "SBP": 1.0}, "overall": 0.75},
        )
        d = resp.to_dict()
        json.dumps(d)
        assert d["ts_shap"][1][1] is None            # NaN scrubbed
        assert d["ts_values"][0][1] is None
        assert d["static_cat"]["values"] == ["Male"]
        assert d["encoding_info"]["feature_ranges"]["Medicin"] == [0, 2]
        assert d["time_axis"]["hours_end"][-1] == 2.0
