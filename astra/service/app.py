"""FastAPI reference service wrapping :class:`astra.inference.api.AstraPredictor`.

This is a thin, deliberately minimal HTTP layer for external teams to build
on. Authentication, TLS, rate limiting and horizontal scaling are **out of
scope** — deploy behind your own gateway.

SINGLE-WORKER CONSTRAINT
    Run exactly one uvicorn worker (the default of ``python -m astra.service``).
    A module-global :data:`PREDICTOR_LOCK` serializes every predictor call
    because (a) the model and SHAP explainer are not re-entrant and (b) the
    CSV data layer writes a shared interim pickle during patient preparation.
    Multiple workers would each load a full model copy *and* race on that
    shared file. Endpoints are plain ``def`` (threadpool) so ``GET /health``
    stays responsive while a slow SHAP call holds the lock.

Endpoints
    - ``GET  /health``               liveness + model status (works when degraded)
    - ``GET  /model/info``           static model/deployment metadata
    - ``POST /predict``              probability (+ probability-over-time curve)
    - ``POST /explain``              SHAP explanation at a timestamp
    - ``POST /explain/differential`` delta-SHAP between two timepoints
    - ``POST /explain/ebm``          local EBM contributions (400 if EBM disabled)

Error mapping (JSON ``{"detail": str}``)
    - ``PatientNotFoundError``            -> 404
    - ``TimestampBeforeAdmissionError``   -> 422
    - ``ValueError``                      -> 422
    - ``ArtifactError``                   -> 503 (also: model not loaded)
    - ``AstraPredictorError`` (other)     -> 500
    - unexpected exceptions               -> 500 (generic detail, traceback logged)

Construction
    :func:`create_app` loads the predictor from :class:`ServiceSettings` in the
    lifespan handler, or accepts an injected ``predictor`` (used by the test
    suite — no artifacts/data required on the machine).
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from astra.inference.api import (
    ArtifactError,
    AstraPredictor,
    AstraPredictorError,
    PatientNotFoundError,
    TimestampBeforeAdmissionError,
)
from astra.service.schemas import (
    DifferentialExplanationResponseModel,
    DifferentialRequest,
    EbmRequest,
    ExplainRequest,
    ExplanationResponseModel,
    HealthResponse,
    PredictionResponseModel,
    PredictRequest,
)
from astra.service.settings import ServiceSettings

logger = logging.getLogger(__name__)

#: Serializes every predictor call. The model/SHAP explainer is not
#: re-entrant and the data layer writes a shared interim pickle — see the
#: single-worker constraint in the module docstring.
PREDICTOR_LOCK = threading.Lock()

# (exception class, HTTP status) — most specific class wins via MRO lookup.
_ERROR_STATUS = (
    (PatientNotFoundError, 404),
    (TimestampBeforeAdmissionError, 422),
    (ArtifactError, 503),
    (AstraPredictorError, 500),
    (ValueError, 422),
)

_METADATA_CSV = os.path.join("data", "external", "metadata.csv")
_CONCEPTS_DIR = os.path.join("data", "interim", "concepts")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _configure_logging(settings: ServiceSettings) -> None:
    """Set up the ``astra`` logger hierarchy (console-only, idempotent)."""
    try:
        from astra.utils import setup_logging

        setup_logging(level=settings.log_level_int)
    except Exception:  # noqa: BLE001 — logging must never block startup
        logging.basicConfig(level=settings.log_level_int)
        logger.warning(
            "astra.utils.setup_logging unavailable — falling back to basicConfig",
            exc_info=True,
        )


def _sanity_check_data_layout() -> None:
    """Log (warnings only, never crash) whether the CSV data layer can work.

    Irrelevant when a custom :class:`PatientDataSource` or an injected
    predictor is used — hence warnings, not errors.
    """
    if os.path.isfile(_METADATA_CSV):
        logger.info("Found patient metadata: %s", _METADATA_CSV)
    else:
        logger.warning(
            "%s not found — CSV-based patient lookup may fail "
            "(ignore if using a custom data source).", _METADATA_CSV,
        )

    if not os.path.isdir(_CONCEPTS_DIR):
        logger.warning(
            "%s does not exist — interim concept caching is unavailable "
            "(ignore if using a custom data source).", _CONCEPTS_DIR,
        )
    elif not os.access(_CONCEPTS_DIR, os.W_OK):
        logger.warning(
            "%s is not writable — the data layer's shared interim pickle "
            "writes will fail.", _CONCEPTS_DIR,
        )
    else:
        logger.info("Interim concepts dir is writable: %s", _CONCEPTS_DIR)


def _load_predictor(settings: ServiceSettings):
    """Load the predictor from settings; return ``None`` on failure (degraded)."""
    if not settings.model_name:
        logger.warning(
            "No model configured (ASTRA_MODEL_NAME unset and configs/defaults.yaml "
            "unreadable) — starting degraded."
        )
        return None
    try:
        predictor = AstraPredictor.load(
            settings.model_name,
            settings.artifacts_dir,
            config_path=settings.config_path,
            device=settings.device,
            data_dir=settings.data_dir,
            patient_dir=settings.patient_dir,
            context_cache_size=settings.cache_size,
        )
    except Exception:  # noqa: BLE001 — degraded /health beats a crash loop
        logger.exception(
            "Failed to load model '%s' from '%s' — starting degraded.",
            settings.model_name, settings.artifacts_dir,
        )
        return None
    logger.info(
        "Loaded model '%s' (temporal=%s, device=%s, seq_len=%d)",
        predictor.model_name, predictor.is_temporal,
        predictor.session.device, predictor.seq_len,
    )
    return predictor


# ---------------------------------------------------------------------------
# Request-time helpers
# ---------------------------------------------------------------------------

def _require_predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded (service degraded) — see GET /health.",
        )
    return predictor


def _payload(resp: Any) -> Any:
    """Facade responses expose ``to_dict()``; plain dicts pass through."""
    return resp.to_dict() if hasattr(resp, "to_dict") else resp


def _make_error_handler(status_code: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        if status_code >= 500:
            logger.error(
                "%s %s failed (%d)", request.method, request.url.path,
                status_code, exc_info=exc,
            )
        else:
            logger.info(
                "%s %s -> %d: %s", request.method, request.url.path,
                status_code, exc,
            )
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


async def _unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the traceback; return a generic detail (don't leak internals).
    logger.error(
        "Unhandled exception on %s %s", request.method, request.url.path,
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Optional[ServiceSettings] = None, predictor=None) -> FastAPI:
    """Build the reference service.

    Args:
        settings: Service configuration; ``ServiceSettings.from_env()`` if None.
        predictor: Pre-built predictor (e.g. a mock in tests, or an
            ``AstraPredictor`` wired to a custom data source). When given, it
            is used as-is and nothing is loaded at startup.
    """
    settings = settings or ServiceSettings.from_env()
    injected = predictor is not None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _configure_logging(settings)
        _sanity_check_data_layout()
        if not injected:
            app.state.predictor = _load_predictor(settings)
        yield

    app = FastAPI(
        title="ASTRA inference service",
        version="0.0.1",
        description=__doc__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.predictor = predictor  # None until lifespan loads it (unless injected)

    for exc_type, status in _ERROR_STATUS:
        app.add_exception_handler(exc_type, _make_error_handler(status))
    app.add_exception_handler(Exception, _unexpected_error_handler)

    # ------------------------------------------------------------------
    # Endpoints (plain `def` -> threadpool; PREDICTOR_LOCK serializes work)
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Liveness + model status.

        Always returns 200 — ``status='degraded'`` with ``model_loaded=False``
        when the model failed to load. Note: this service is single-worker by
        design (one model copy, non-re-entrant explainer); scale by running
        independent instances behind your own gateway.
        """
        pred = getattr(request.app.state, "predictor", None)
        if pred is None:
            return HealthResponse(
                status="degraded",
                model_name=settings.model_name,
                model_loaded=False,
            )
        session = getattr(pred, "session", None)
        device = getattr(session, "device", None)
        return HealthResponse(
            status="ok",
            model_name=getattr(pred, "model_name", None),
            model_loaded=True,
            is_temporal=getattr(pred, "is_temporal", None),
            device=str(device) if device is not None else None,
            seq_len=getattr(pred, "seq_len", None),
        )

    @app.get("/model/info")
    def model_info(request: Request) -> dict:
        """Static model/deployment metadata (JSON-safe passthrough)."""
        pred = _require_predictor(request)
        with PREDICTOR_LOCK:
            return pred.model_info()

    @app.post("/predict", response_model=PredictionResponseModel)
    def predict(req: PredictRequest, request: Request):
        """Outcome probability for a patient as of a timestamp."""
        pred = _require_predictor(request)
        with PREDICTOR_LOCK:
            resp = pred.predict(
                req.patient_id, req.timestamp, req.service_date,
                include_curve=req.include_curve,
            )
        return _payload(resp)

    @app.post("/explain", response_model=ExplanationResponseModel)
    def explain(req: ExplainRequest, request: Request):
        """SHAP explanation of the prediction at a timestamp."""
        pred = _require_predictor(request)
        with PREDICTOR_LOCK:
            resp = pred.explain(
                req.patient_id, req.timestamp, req.service_date,
                top_n=req.top_n, include_values=req.include_values,
            )
        return _payload(resp)

    @app.post("/explain/differential", response_model=DifferentialExplanationResponseModel)
    def explain_differential(req: DifferentialRequest, request: Request):
        """Delta-SHAP between two elapsed-hour timepoints (T2 - T1)."""
        pred = _require_predictor(request)
        with PREDICTOR_LOCK:
            resp = pred.explain_differential(
                req.patient_id, req.service_date, req.t1_hours, req.t2_hours,
                include_endpoints=req.include_endpoints,
            )
        return _payload(resp)

    @app.post("/explain/ebm")
    def explain_ebm(req: EbmRequest, request: Request):
        """Local EBM feature contributions (400 if the model has no EBM channel)."""
        pred = _require_predictor(request)
        with PREDICTOR_LOCK:
            result = pred.explain_ebm(req.patient_id, req.timestamp, req.service_date)
        if result is None:
            raise HTTPException(
                status_code=400,
                detail="EBM explanations are not available: this model has no "
                       "'_ebm_pred' input channel (ebm_feature disabled).",
            )
        return result

    return app
