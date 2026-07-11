"""JSON-safe response objects for the ASTRA inference API.

These dataclasses are the wire format between :class:`~astra.inference.api.AstraPredictor`
and any consumer (the bundled FastAPI reference service, a custom service, or a
frontend). Every object serializes with ``to_dict()`` → plain dict of JSON types
(numpy arrays → lists, ``NaN``/``inf`` → ``None``, timestamps → ISO 8601 strings).

The payloads carry everything needed to recreate the SHAP panels from
``dashboard/app_shap.py`` without access to the model or the bin configuration:
the :class:`TimeAxis` maps step indices to elapsed hours, so frontends never
need to know the variable-width bin grid.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reference epoch used to materialize the bin grid for the TimeAxis. The grid
# is admission-relative, so any fixed naive timestamp yields identical
# elapsed-hour offsets.
REFERENCE_EPOCH = pd.Timestamp("2000-01-01 00:00:00")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert *obj* to JSON-serializable types.

    numpy scalars/arrays → python types/lists; ``NaN``/``inf`` → None;
    Timestamps/datetimes → ISO 8601 strings; tuples/sets → lists.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, (pd.Timestamp,)) or hasattr(obj, "isoformat"):
        try:
            if obj is pd.NaT or (isinstance(obj, pd.Timestamp) and pd.isna(obj)):
                return None
        except (TypeError, ValueError):
            pass
        return obj.isoformat()
    if isinstance(obj, dict):
        return {
            (int(k) if isinstance(k, np.integer) else k): to_jsonable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    if hasattr(obj, "item"):  # 0-d numpy leftovers
        return to_jsonable(obj.item())
    logger.debug("to_jsonable: falling back to str() for %s", type(obj))
    return str(obj)


class _JsonableMixin:
    """Shared ``to_dict`` for all response dataclasses."""

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))


@dataclass
class TimeAxis(_JsonableMixin):
    """Maps step indices on the model's bin grid to elapsed hours.

    Built from the deployment bundle's ``data_config`` (never the global cfg),
    so it always matches the grid the tensors were built on.
    """

    steps: List[int]
    hours_start: List[float]   # bin start, hours since admission (t=0)
    hours_end: List[float]     # bin end, hours since admission
    bin_freq: List[str]        # bin width label per step (e.g. '10min', '1h')

    @classmethod
    def from_data_config(cls, data_config: dict) -> "TimeAxis":
        from astra.inference.data_prep import _create_patient_bins

        bin_df = _create_patient_bins(REFERENCE_EPOCH, data_config)
        elapsed = lambda col: (
            (bin_df[col] - REFERENCE_EPOCH).dt.total_seconds() / 3600.0
        ).tolist()
        return cls(
            steps=bin_df["position"].astype(int).tolist(),
            hours_start=elapsed("bin_start"),
            hours_end=elapsed("bin_end"),
            bin_freq=bin_df["bin_freq"].astype(str).tolist(),
        )

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class ProbabilityCurve(_JsonableMixin):
    """Outcome probability over time, truncated to the visible trajectory."""

    steps: List[int]
    hours: List[float]                          # elapsed hours per step (bin end)
    probabilities: List[Optional[float]]        # None where no prediction exists
    source: str                                 # 'temporal_head' | 'simulation'
    survival: Optional[List[Optional[float]]] = None  # S(t), survival mode only


@dataclass
class PredictionResponse(_JsonableMixin):
    """Response for a single predict(patient, timestamp) call."""

    patient_id: str
    pid: Any
    model_name: str
    is_temporal: bool
    survival_mode: bool
    calibration_method: Optional[str]
    admission_time: str                         # ISO 8601
    requested_time: str                         # ISO 8601, as passed by caller
    eval_hours: float                           # hours since admission actually evaluated
    eval_step: int
    trajectory_length: int
    seq_len: int
    probability: float
    curve: Optional[ProbabilityCurve] = None
    inhospital_start_hours: Optional[float] = None  # prehospital patients only
    compute_ms: Optional[float] = None


@dataclass
class StaticFeatureBlock(_JsonableMixin):
    """SHAP + raw values for static (tabular) features."""

    names: List[str]
    shap: List[float]
    values: List[Any]                           # raw display values (str or float)


@dataclass
class CategoricalTSBlock(_JsonableMixin):
    """SHAP for the multi-hot categorical time series (Medicin/Procedurer/ADT)."""

    labels: List[str]                           # category labels, row order
    shap_per_category: List[List[float]]        # [n_categories, seq_len]
    shap_aggregate: List[float]                 # [seq_len] mean |SHAP| across categories
    values_per_category: Optional[List[List[float]]] = None  # multi-hot counts [n_categories, seq_len]


@dataclass
class ExplanationResponse(_JsonableMixin):
    """SHAP explanation of the prediction at ``eval_step``.

    ``ts_shap[c][t]`` is the attribution of channel ``c`` at timestep ``t``
    toward the probability at ``eval_step``. Together with ``ts_values``,
    ``time_axis`` and the static blocks this reproduces every SHAP panel in
    ``dashboard/app_shap.py``.
    """

    patient_id: str
    pid: Any
    model_name: str
    eval_step: int
    eval_hours: float
    trajectory_length: int
    seq_len: int
    time_axis: TimeAxis
    channels: List[str]                         # continuous TS channel names, row order
    ts_shap: List[List[float]]                  # [n_channels, seq_len]
    ts_values: List[List[Optional[float]]]      # raw values, None = not measured
    # Per-channel metadata from the bundle's data_config:
    # {channel: {concept, feature, agg_func, type}} — group channels by
    # source concept via entry['concept'].
    channel_map: Optional[Dict[str, Any]] = None
    cat_ts: Optional[CategoricalTSBlock] = None
    static_cat: Optional[StaticFeatureBlock] = None
    static_cont: Optional[StaticFeatureBlock] = None
    encoding_info: Optional[dict] = None        # feature_ranges + category_labels
    top_features: List[Dict[str, Any]] = field(default_factory=list)  # [{name, importance}]
    completeness: Optional[Dict[str, Any]] = None  # {per_channel: {name: frac}, overall: frac}
    inhospital_start_step: Optional[int] = None
    compute_ms: Optional[float] = None


@dataclass
class DifferentialExplanationResponse(_JsonableMixin):
    """ΔSHAP between two timepoints: what changed between T1 and T2."""

    patient_id: str
    pid: Any
    model_name: str
    t1_hours: float
    t2_hours: float
    t1_step: int
    t2_step: int
    t1_probability: float
    t2_probability: float
    time_axis: TimeAxis
    channels: List[str]
    delta_ts_shap: List[List[float]]            # [n_channels, seq_len]
    delta_cat_ts: Optional[CategoricalTSBlock] = None
    delta_static_cat: Optional[StaticFeatureBlock] = None
    delta_static_cont: Optional[StaticFeatureBlock] = None
    top_delta_features: List[Dict[str, Any]] = field(default_factory=list)
    shap_t1: Optional[ExplanationResponse] = None
    shap_t2: Optional[ExplanationResponse] = None
    compute_ms: Optional[float] = None


def format_top_features(pairs) -> List[Dict[str, Any]]:
    """Normalize ``[(name, importance), ...]`` to ``[{'name':…, 'importance':…}]``."""
    return [
        {"name": str(name), "importance": to_jsonable(float(imp))}
        for name, imp in (pairs or [])
    ]
