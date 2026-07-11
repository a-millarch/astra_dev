"""Pydantic (v2) request/response schemas for the ASTRA reference service.

Request models define the POST bodies. Response models mirror the response
dataclasses in :mod:`astra.inference.responses` **field-for-field** so the
OpenAPI schema documents the exact wire format and outgoing payloads are
validated.

Two deliberate loosenings versus the dataclasses:

- All response models use ``extra="allow"`` — additive fields in future facade
  versions pass through instead of breaking validation.
- Numeric lists have ``Optional[float]`` elements: the facade's ``to_jsonable``
  serializes ``NaN``/``inf`` to ``None`` (e.g. curve gaps, unmeasured values).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Numeric list/matrix aliases (NaN -> None after facade serialization).
FloatList = List[Optional[float]]
FloatMatrix = List[List[Optional[float]]]


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Body for ``POST /predict``."""

    patient_id: str = Field(description="CPR hash identifying the patient")
    service_date: str = Field(
        description="Trauma admission date (identifies the encounter), e.g. '2023-08-15'")
    timestamp: str = Field(
        description="Evaluation time (ISO 8601); clamped to trajectory end if beyond it")
    include_curve: bool = True


class ExplainRequest(BaseModel):
    """Body for ``POST /explain``."""

    patient_id: str
    service_date: str
    timestamp: str
    top_n: int = 20
    include_values: bool = True


class DifferentialRequest(BaseModel):
    """Body for ``POST /explain/differential`` (delta-SHAP T2 - T1)."""

    patient_id: str
    service_date: str
    t1_hours: float
    t2_hours: float
    include_endpoints: bool = False


class EbmRequest(BaseModel):
    """Body for ``POST /explain/ebm``."""

    patient_id: str
    service_date: str
    timestamp: str


# ---------------------------------------------------------------------------
# Response models (mirror astra/inference/responses.py dataclasses)
# ---------------------------------------------------------------------------

class _ResponseModel(BaseModel):
    """Base for response models.

    ``extra="allow"`` tolerates additive facade fields;
    ``protected_namespaces=()`` permits the ``model_name`` field.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class TimeAxisModel(_ResponseModel):
    """Mirrors :class:`astra.inference.responses.TimeAxis`."""

    steps: List[int]
    hours_start: FloatList
    hours_end: FloatList
    bin_freq: List[str]


class ProbabilityCurveModel(_ResponseModel):
    """Mirrors :class:`astra.inference.responses.ProbabilityCurve`."""

    steps: List[int]
    hours: FloatList
    probabilities: FloatList
    source: str
    survival: Optional[FloatList] = None


class PredictionResponseModel(_ResponseModel):
    """Mirrors :class:`astra.inference.responses.PredictionResponse`."""

    patient_id: str
    pid: Any
    model_name: str
    is_temporal: bool
    survival_mode: bool
    calibration_method: Optional[str] = None
    admission_time: str
    requested_time: str
    eval_hours: float
    eval_step: int
    trajectory_length: int
    seq_len: int
    probability: float
    curve: Optional[ProbabilityCurveModel] = None
    inhospital_start_hours: Optional[float] = None
    compute_ms: Optional[float] = None


class StaticFeatureBlockModel(_ResponseModel):
    """Mirrors :class:`astra.inference.responses.StaticFeatureBlock`."""

    names: List[str]
    shap: FloatList
    values: List[Any]


class CategoricalTSBlockModel(_ResponseModel):
    """Mirrors :class:`astra.inference.responses.CategoricalTSBlock`."""

    labels: List[str]
    shap_per_category: FloatMatrix
    shap_aggregate: FloatList
    values_per_category: Optional[FloatMatrix] = None


class ExplanationResponseModel(_ResponseModel):
    """Mirrors :class:`astra.inference.responses.ExplanationResponse`."""

    patient_id: str
    pid: Any
    model_name: str
    eval_step: int
    eval_hours: float
    trajectory_length: int
    seq_len: int
    time_axis: TimeAxisModel
    channels: List[str]
    ts_shap: FloatMatrix
    ts_values: FloatMatrix
    # Per-channel metadata from the bundle:
    # {channel: {concept, feature, agg_func, type}}
    channel_map: Optional[Dict[str, Any]] = None
    cat_ts: Optional[CategoricalTSBlockModel] = None
    static_cat: Optional[StaticFeatureBlockModel] = None
    static_cont: Optional[StaticFeatureBlockModel] = None
    encoding_info: Optional[Dict[str, Any]] = None
    top_features: List[Dict[str, Any]] = Field(default_factory=list)
    completeness: Optional[Dict[str, Any]] = None
    inhospital_start_step: Optional[int] = None
    compute_ms: Optional[float] = None


class DifferentialExplanationResponseModel(_ResponseModel):
    """Mirrors :class:`astra.inference.responses.DifferentialExplanationResponse`."""

    patient_id: str
    pid: Any
    model_name: str
    t1_hours: float
    t2_hours: float
    t1_step: int
    t2_step: int
    t1_probability: float
    t2_probability: float
    time_axis: TimeAxisModel
    channels: List[str]
    delta_ts_shap: FloatMatrix
    delta_cat_ts: Optional[CategoricalTSBlockModel] = None
    delta_static_cat: Optional[StaticFeatureBlockModel] = None
    delta_static_cont: Optional[StaticFeatureBlockModel] = None
    top_delta_features: List[Dict[str, Any]] = Field(default_factory=list)
    shap_t1: Optional[ExplanationResponseModel] = None
    shap_t2: Optional[ExplanationResponseModel] = None
    compute_ms: Optional[float] = None


class HealthResponse(_ResponseModel):
    """Body for ``GET /health``.

    ``status`` is ``'ok'`` when the model is loaded, ``'degraded'`` when the
    service is up but the model failed to load (all inference endpoints then
    return 503).
    """

    status: str
    model_name: Optional[str] = None
    model_loaded: bool
    is_temporal: Optional[bool] = None
    device: Optional[str] = None
    seq_len: Optional[int] = None
