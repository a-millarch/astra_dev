"""High-level inference facade: patient identifier + timestamp → prediction + SHAP.

:class:`AstraPredictor` is the single entry point external deployments build
on. It composes the existing inference stack (:class:`InferenceSession`,
:class:`PatientContext`, :class:`SimulationRunner`) behind a small, typed,
JSON-serializable surface:

    predictor = AstraPredictor.load("my_model", artifacts_dir="handoff/models")
    resp = predictor.predict(cpr_hash, "2023-08-16 06:00", service_date="2023-08-15")
    payload = resp.to_dict()   # probability + probability-over-time curve
    shap = predictor.explain(cpr_hash, "2023-08-16 06:00", service_date="2023-08-15")

Patient data is read through the pluggable
:class:`~astra.inference.datasource.PatientDataSource` seam — pass
``data_source=`` to serve data from SQL/parquet/memory instead of CSV files
(see ``docs/HANDOFF.md`` for the per-concept schema contract).

Both model types are supported transparently:

- **Temporal-head models**: the probability-over-time curve comes from a
  single forward pass.
- **Non-temporal models**: the curve is built by stepping a
  :class:`SimulationRunner` through the bin grid — one forward pass per bin,
  order-of-magnitude more expensive. Per-patient contexts are LRU-cached so
  repeated/advancing queries are incremental.
"""

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from astra.inference.responses import (
    TimeAxis,
    ProbabilityCurve,
    PredictionResponse,
    ExplanationResponse,
    DifferentialExplanationResponse,
    StaticFeatureBlock,
    CategoricalTSBlock,
    format_top_features,
    to_jsonable,
)
from astra.inference.datasource import CSVDataSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors (the reference service maps these to HTTP statuses)
# ---------------------------------------------------------------------------

class AstraPredictorError(Exception):
    """Base class for predictor errors."""


class PatientNotFoundError(AstraPredictorError):
    """No data could be resolved for the requested patient. → HTTP 404"""


class TimestampBeforeAdmissionError(AstraPredictorError):
    """Requested timestamp precedes the patient's trajectory start. → HTTP 422"""


class ArtifactError(AstraPredictorError):
    """Model artifacts are missing or unloadable. → HTTP 503"""


class AstraPredictor:
    """Facade for single-patient prediction and SHAP explanation.

    Construct via :meth:`load`. Thread-safe: a lock serializes inference
    (the underlying model and SHAP explainer are not re-entrant).
    """

    def __init__(self, session, *, data_dir='data/raw',
                 patient_dir='data/patients', ebm_models_dir='models/ebm',
                 context_cache_size=8, cfg=None):
        self.session = session
        self.bundle = session.bundle
        # Project config governing data preparation (concepts, filters,
        # prehospital toggle). None -> downstream defaults (configs/defaults.yaml).
        # The bin grid always comes from the bundle's data_config, not from here.
        self._cfg = cfg
        self.model_name = self.bundle.get('model_name', '?')
        self.is_temporal = session.is_temporal
        self.survival_mode = self.bundle.get('model_params', {}).get(
            'survival_mode', False)
        self.data_dir = data_dir
        self.patient_dir = patient_dir
        self.ebm_models_dir = ebm_models_dir

        data_config = self.bundle['data_config']
        self.time_axis = TimeAxis.from_data_config(data_config)
        self.seq_len = self.bundle['model_params']['seq_len']
        if len(self.time_axis) != self.seq_len:
            logger.warning(
                "TimeAxis has %d steps but model seq_len is %d — "
                "bundle data_config and model params disagree (stale bundle?)",
                len(self.time_axis), self.seq_len,
            )

        self._cache_size = max(1, int(context_cache_size))
        self._entries = OrderedDict()   # (patient_id, service_date) -> entry dict
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_name: Optional[str] = None,
             artifacts_dir: str = 'models', *,
             config_path: Optional[str] = None,
             device: Optional[str] = None,
             data_source=None,
             data_dir: str = 'data/raw',
             patient_dir: str = 'data/patients',
             context_cache_size: int = 8) -> "AstraPredictor":
        """Load model + deployment bundle and configure the data source.

        Args:
            model_name: Name the model was trained/exported under. May be
                omitted when *config_path* is given — it is then read from the
                config's ``model_name`` key (the config-first pattern used by
                ``python -m astra.training.train --config ...``).
            artifacts_dir: Root artifacts directory containing
                ``deployment/deployment_{model_name}.pkl``,
                ``{model_name}.pth``, optional ``calibrators/{model_name}/``
                and optional ``ebm/`` (the layout produced by training and by
                ``python -m astra.inference.export_artifacts``).
            config_path: Project config YAML governing data preparation
                (concepts, filters, prehospital toggle). None -> the
                pipeline's default ``configs/defaults.yaml``. The bin grid is
                always taken from the deployment bundle, never from here.
            device: 'cuda', 'cpu' or None (auto-detect).
            data_source: A :class:`PatientDataSource` for non-file data feeds
                (registered process-globally). Passing a
                :class:`CSVDataSource` (or None) selects the built-in
                file-based loading using *data_dir*/*patient_dir*.
            data_dir: Population CSV directory (file-based mode).
            patient_dir: Pre-split per-patient CSV directory (file-based mode).
            context_cache_size: Number of patient contexts kept warm (LRU).
        """
        from astra.inference.pipeline import InferenceSession

        cfg = None
        if config_path is not None:
            from astra.utils import get_cfg
            cfg = get_cfg(config_path)
            logger.info("Loaded config %s (model_name=%r)",
                        config_path, cfg.get('model_name'))
        if model_name is None:
            model_name = (cfg or {}).get('model_name')
            if not model_name:
                raise ValueError(
                    "model_name is required — pass it explicitly or provide "
                    "config_path to a YAML with a 'model_name' key")

        # Accept both artifact layouts:
        #   training root:   <dir>/deployment/deployment_<M>.pkl, <dir>/<M>.pth
        #   handoff bundle:  <dir>/models/deployment/..., <dir>/models/<M>.pth
        #     (the layout `export_artifacts export --out <dir>` produces —
        #      pointing at the unzipped bundle directory just works)
        bundle_name = f'deployment_{model_name}.pkl'
        nested = os.path.join(artifacts_dir, 'models')
        if (not os.path.isfile(os.path.join(artifacts_dir, 'deployment', bundle_name))
                and os.path.isfile(os.path.join(nested, 'deployment', bundle_name))):
            logger.info("Using nested artifacts root %s (handoff bundle layout)",
                        nested)
            artifacts_dir = nested

        bundle_dir = os.path.join(artifacts_dir, 'deployment')
        try:
            session = InferenceSession.load(
                model_name, device=device,
                bundle_dir=bundle_dir, weights_dir=artifacts_dir,
            )
        except (FileNotFoundError, OSError) as e:
            raise ArtifactError(
                f"Could not load model '{model_name}' from '{artifacts_dir}': {e}"
            ) from e

        if isinstance(data_source, CSVDataSource):
            # CSV sources are directory configuration for the built-in file
            # path (which honors per-call read_csv kwargs exactly).
            data_dir = data_source.data_dir
            patient_dir = data_source.patient_dir
            data_source = None
        if data_source is not None:
            from astra.inference.patient_store import set_data_source
            set_data_source(data_source)

        return cls(
            session,
            data_dir=data_dir,
            patient_dir=patient_dir,
            ebm_models_dir=os.path.join(artifacts_dir, 'ebm'),
            context_cache_size=context_cache_size,
            cfg=cfg,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, patient_id: str, timestamp, service_date, *,
                include_curve: bool = True) -> PredictionResponse:
        """Predict outcome probability for *patient_id* as of *timestamp*.

        Args:
            patient_id: CPR hash identifying the patient.
            timestamp: Evaluation time (ISO string or datetime). Clamped to
                the patient's trajectory end if beyond it.
            service_date: Trauma admission date — identifies which trajectory
                to build when a patient has several encounters.
            include_curve: Attach the probability-over-time curve (for
                non-temporal models the curve is computed either way; this
                only controls the payload).
        """
        ts = self._parse_timestamp(timestamp)
        t0 = time.perf_counter()
        with self._lock:
            entry = self._get_entry(patient_id, service_date, ts)
            ctx = entry['ctx']
            result, curve, eval_step = self._predict_at(entry, ts)

            response = PredictionResponse(
                patient_id=patient_id,
                pid=ctx.pid,
                model_name=self.model_name,
                is_temporal=self.is_temporal,
                survival_mode=self.survival_mode,
                calibration_method=getattr(self.session, '_calibration_method', None),
                admission_time=ctx.admission_time.isoformat(),
                requested_time=ts.isoformat(),
                eval_hours=self._step_hours(eval_step),
                eval_step=eval_step,
                trajectory_length=int(result.trajectory_length),
                seq_len=self.seq_len,
                probability=float(result.probability),
                curve=curve if include_curve else None,
                inhospital_start_hours=self._inhospital_start_hours(ctx),
                compute_ms=(time.perf_counter() - t0) * 1000.0,
            )
        return response

    def explain(self, patient_id: str, timestamp, service_date, *,
                top_n: int = 20,
                include_values: bool = True) -> ExplanationResponse:
        """SHAP explanation of the prediction at *timestamp*.

        Returns per-channel per-timestep SHAP for the continuous TS,
        per-category SHAP for the categorical TS, static feature importances,
        raw feature values and a data-completeness summary — everything the
        SHAP panels in ``dashboard/app_shap.py`` render.
        """
        ts = self._parse_timestamp(timestamp)
        t0 = time.perf_counter()
        with self._lock:
            entry = self._get_entry(patient_id, service_date, ts)
            ctx = entry['ctx']
            # Ensure the context has advanced to the requested time.
            self._advance(entry, ts)
            eval_step = self._eval_step(ctx, ts)

            shap_result = self.session.explain_from_context(
                ctx, censor_step=eval_step)
            shap_dict, channel2feature, names_cat, names_cont = (
                self.session.shap_to_viz_dict(
                    shap_result, x_ts=ctx.x_ts, x_ts_cat=ctx.x_ts_cat,
                    tab_df=ctx.tab_df,
                )
            )
            response = self._build_explanation_response(
                patient_id, ctx, shap_result, shap_dict,
                names_cat, names_cont, eval_step,
                top_n=top_n, include_values=include_values,
            )
            response.compute_ms = (time.perf_counter() - t0) * 1000.0
        return response

    def explain_differential(self, patient_id: str, service_date,
                             t1_hours: float, t2_hours: float, *,
                             include_endpoints: bool = False
                             ) -> DifferentialExplanationResponse:
        """ΔSHAP between two elapsed-hour timepoints (T2 − T1)."""
        t0 = time.perf_counter()
        with self._lock:
            # Advance the context to cover T2 before differencing.
            later = max(t1_hours, t2_hours)
            entry = self._get_entry_by_hours(patient_id, service_date, later)
            ctx = entry['ctx']

            diff = self.session.explain_differential(
                ctx, t1_hours=t1_hours, t2_hours=t2_hours)

            channels = list(self.bundle['ts_channel_names'])
            zeros = np.zeros(self.seq_len)
            delta_ts = np.stack([
                np.asarray(diff.delta_ts_shap.get(ch, zeros)).squeeze()
                for ch in channels
            ])

            delta_cat_ts = self._categorical_block(
                diff.delta_cat_ts_shap, x_ts_cat=None)
            delta_static_cat = self._static_block_from_dict(
                diff.delta_static_cat_shap, ctx.tab_df)
            delta_static_cont = self._static_block_from_dict(
                diff.delta_static_cont_shap, ctx.tab_df)

            endpoints = {}
            if include_endpoints:
                for key, sr in (('shap_t1', diff.shap_t1), ('shap_t2', diff.shap_t2)):
                    if sr is None:
                        continue
                    sd, _, ncat, ncont = self.session.shap_to_viz_dict(
                        sr, x_ts=ctx.x_ts, x_ts_cat=ctx.x_ts_cat, tab_df=ctx.tab_df)
                    endpoints[key] = self._build_explanation_response(
                        patient_id, ctx, sr, sd, ncat, ncont,
                        eval_step=sr.eval_timestep,
                        top_n=20, include_values=False,
                    )

            response = DifferentialExplanationResponse(
                patient_id=patient_id,
                pid=ctx.pid,
                model_name=self.model_name,
                t1_hours=float(diff.t1_hours),
                t2_hours=float(diff.t2_hours),
                t1_step=int(diff.t1_step),
                t2_step=int(diff.t2_step),
                t1_probability=float(diff.t1_probability),
                t2_probability=float(diff.t2_probability),
                time_axis=self.time_axis,
                channels=channels,
                delta_ts_shap=delta_ts,
                delta_cat_ts=delta_cat_ts,
                delta_static_cat=delta_static_cat,
                delta_static_cont=delta_static_cont,
                top_delta_features=format_top_features(diff.top_delta_features),
                shap_t1=endpoints.get('shap_t1'),
                shap_t2=endpoints.get('shap_t2'),
                compute_ms=(time.perf_counter() - t0) * 1000.0,
            )
        return response

    def explain_viz(self, patient_id: str, timestamp, service_date):
        """SHAP explanation in the dashboard visualization format.

        Returns exactly what ``InferenceSession.shap_to_viz_dict`` produces —
        ``(shap_dict, channel2feature, feature_names_cat, feature_names_cont)``
        — for drop-in use with the existing plotting helpers in
        ``astra.evaluation.behavior``::

            sd, ch2f, ncat, ncont = predictor.explain_viz(pid, ts, service_date)
            visualize_shap_individual_interactive(sd, 0, channel2feature=ch2f,
                feature_names_cat=ncat, feature_names_cont=ncont)

        Use :meth:`explain` instead when you need the JSON-safe payload for a
        frontend; this method is for notebooks/dashboards reusing the built-in
        matplotlib/plotly panels.
        """
        ts = self._parse_timestamp(timestamp)
        with self._lock:
            entry = self._get_entry(patient_id, service_date, ts)
            ctx = entry['ctx']
            self._advance(entry, ts)
            eval_step = self._eval_step(ctx, ts)
            shap_result = self.session.explain_from_context(
                ctx, censor_step=eval_step)
            return self.session.shap_to_viz_dict(
                shap_result, x_ts=ctx.x_ts, x_ts_cat=ctx.x_ts_cat,
                tab_df=ctx.tab_df,
            )

    def explain_ebm(self, patient_id: str, timestamp, service_date) -> Optional[dict]:
        """Local EBM feature contributions (only if the model uses `_ebm_pred`)."""
        if '_ebm_pred' not in self.bundle.get('ts_channel_names', []):
            return None
        ts = self._parse_timestamp(timestamp)
        with self._lock:
            entry = self._get_entry(patient_id, service_date, ts)
            self._advance(entry, ts)
            explanations = self.session.explain_ebm(entry['ctx'], save_path=None)
        return to_jsonable(explanations) if explanations else None

    def model_info(self) -> dict:
        """Static model/deployment metadata for consumers (JSON-safe)."""
        import torch
        params = self.bundle.get('model_params', {})
        dc = self.bundle.get('data_config', {})
        classes = params.get('classes', {}) or {}
        return to_jsonable({
            'model_name': self.model_name,
            'is_temporal': self.is_temporal,
            'survival_mode': self.survival_mode,
            'calibration_method': getattr(self.session, '_calibration_method', None),
            'device': str(self.session.device),
            'seq_len': self.seq_len,
            'channels': list(self.bundle.get('ts_channel_names', [])),
            'channel_map': dc.get('channel_map'),
            'static_categorical': list(classes.keys()),
            'static_continuous': list(self.bundle.get('tab_feature_names', [])),
            'ts_cat_features': list(
                (self.bundle.get('encoding_info') or {}).get('feature_ranges', {}).keys()
            ),
            'ebm_enabled': '_ebm_pred' in self.bundle.get('ts_channel_names', []),
            'bin_intervals': dc.get('bin_intervals'),
            'bin_freq_include': dc.get('bin_freq_include'),
            'time_axis': self.time_axis.to_dict(),
            'versions': {
                'torch': torch.__version__,
                'numpy': np.__version__,
                'pandas': pd.__version__,
            },
        })

    def clear_cache(self, patient_id: Optional[str] = None) -> None:
        """Drop cached patient contexts (all, or a single patient's)."""
        with self._lock:
            if patient_id is None:
                self._entries.clear()
            else:
                for key in [k for k in self._entries if k[0] == patient_id]:
                    del self._entries[key]

    # ------------------------------------------------------------------
    # Context/entry management
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(timestamp) -> pd.Timestamp:
        try:
            ts = pd.Timestamp(timestamp)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Unparseable timestamp: {timestamp!r}") from e
        if pd.isna(ts):
            raise ValueError(f"Unparseable timestamp: {timestamp!r}")
        return ts

    def _get_entry(self, patient_id: str, service_date, ts: pd.Timestamp) -> dict:
        """Return the cache entry for a patient, validating *ts* against admission."""
        entry = self._get_or_build(patient_id, service_date)
        admission = entry['ctx'].admission_time
        if ts < admission:
            raise TimestampBeforeAdmissionError(
                f"timestamp {ts.isoformat()} precedes trajectory start "
                f"{admission.isoformat()}"
            )
        return entry

    def _get_entry_by_hours(self, patient_id: str, service_date, hours: float) -> dict:
        """Entry advanced to at least ``admission + hours``."""
        entry = self._get_or_build(patient_id, service_date)
        target = entry['ctx'].admission_time + pd.Timedelta(hours=float(hours))
        self._advance(entry, target)
        return entry

    def _get_or_build(self, patient_id: str, service_date) -> dict:
        if service_date is None:
            raise ValueError("service_date is required to identify the trauma encounter")
        key = (str(patient_id), str(service_date))
        entry = self._entries.get(key)
        if entry is None:
            entry = self._build_entry(patient_id, service_date)
            self._entries[key] = entry
            while len(self._entries) > self._cache_size:
                evicted_key, _ = self._entries.popitem(last=False)
                logger.info("Evicted cached context for %s", evicted_key[0][:8])
        self._entries.move_to_end(key)
        return entry

    def _build_entry(self, patient_id: str, service_date) -> dict:
        """Build a fresh context (temporal) or simulation runner (non-temporal).

        Contexts are always built AT admission (``start_hours=0.0`` derives
        the true admission time from the patient's ADT trajectory — the
        service date alone may precede it) and then advanced forward.
        """
        logger.info("Building patient context: %s... (service=%s)",
                    str(patient_id)[:8], service_date)
        try:
            if self.is_temporal:
                from astra.inference.patient_context import PatientContext
                ctx = PatientContext.from_csv(
                    cpr_hash=patient_id,
                    service_date=service_date,
                    current_time=pd.Timestamp(service_date),
                    bundle=self.bundle,
                    cfg=self._cfg,
                    data_dir=self.data_dir,
                    patient_dir=self.patient_dir,
                    ebm_models_dir=self.ebm_models_dir,
                    start_hours=0.0,
                )
                return {'ctx': ctx, 'runner': None}

            from astra.inference.simulation import SimulationRunner
            runner = SimulationRunner(self.session)
            runner.setup(
                cpr_hash=patient_id,
                service_date=service_date,
                cfg=self._cfg,
                data_dir=self.data_dir,
                patient_dir=self.patient_dir,
                ebm_models_dir=self.ebm_models_dir,
            )
            return {'ctx': runner.context, 'runner': runner}
        except FileNotFoundError as e:
            raise PatientNotFoundError(
                f"No data found for patient {str(patient_id)[:8]}... "
                f"(service_date={service_date}): {e}"
            ) from e

    def _advance(self, entry: dict, ts: pd.Timestamp) -> None:
        """Move the entry's context forward to *ts* (never backwards)."""
        ctx = entry['ctx']
        target = ts
        end = getattr(ctx, 'patient_end_time', None)
        if end is not None:
            target = min(target, end)
        if target <= ctx.current_time:
            return
        if entry['runner'] is not None:
            entry['runner'].advance_to(time=target)
        else:
            ctx.refresh(target)

    # ------------------------------------------------------------------
    # Prediction internals
    # ------------------------------------------------------------------

    def _eval_step(self, ctx, ts: pd.Timestamp) -> int:
        """Step index for *ts* on the bundle's bin grid, clamped to visibility."""
        from astra.evaluation.utils import time_to_step
        delta_min = (min(ts, ctx.current_time) - ctx.admission_time
                     ).total_seconds() / 60.0
        step = time_to_step(delta_min, 'min',
                            data_config=self.bundle['data_config'])
        if step is None:
            step = ctx.trajectory_length - 1
        return int(max(0, min(step, ctx.trajectory_length - 1)))

    def _predict_at(self, entry: dict, ts: pd.Timestamp):
        """Predict at *ts*, returning (InferenceResult, ProbabilityCurve, eval_step)."""
        ctx = entry['ctx']
        self._advance(entry, ts)
        eval_step = self._eval_step(ctx, ts)

        if self.is_temporal:
            result = self.session.predict_from_context(ctx, censor_step=eval_step)
            curve = self._curve_from_temporal(result)
            return result, curve, int(result.censor_step)

        # Non-temporal: the runner has stepped bin-by-bin up to the target;
        # the stored curve is authoritative for any step ≤ current.
        runner = entry['runner']
        raw_curve = runner._prediction_curve
        probability = self._curve_value_at(raw_curve, eval_step)
        if probability is None:
            # No stepped prediction at/below this step (e.g. very early
            # query) — run a single direct forward pass instead.
            result = self.session.predict_from_context(ctx)
            probability = result.probability
        from astra.inference.pipeline import InferenceResult
        result = InferenceResult(
            pid=ctx.pid,
            probability=float(probability),
            trajectory_length=ctx.trajectory_length,
            censor_step=eval_step,
        )
        curve = self._curve_from_simulation(raw_curve, ctx.trajectory_length)
        return result, curve, eval_step

    def _curve_from_temporal(self, result) -> ProbabilityCurve:
        n = int(result.trajectory_length)
        probs = np.asarray(result.predictions_over_time)[:n]
        survival = None
        if result.survival_curve is not None:
            survival = np.asarray(result.survival_curve)[:n]
        return ProbabilityCurve(
            steps=list(range(n)),
            hours=self.time_axis.hours_end[:n],
            probabilities=probs,
            source='temporal_head',
            survival=survival,
        )

    def _curve_from_simulation(self, raw_curve, trajectory_length: int) -> ProbabilityCurve:
        n = int(trajectory_length)
        probs = (np.asarray(raw_curve, dtype=float)[:n]
                 if raw_curve is not None else np.full(n, np.nan))
        return ProbabilityCurve(
            steps=list(range(n)),
            hours=self.time_axis.hours_end[:n],
            probabilities=probs,
            source='simulation',
        )

    @staticmethod
    def _curve_value_at(curve, step: int) -> Optional[float]:
        """Last non-NaN curve value at or before *step*."""
        if curve is None:
            return None
        arr = np.asarray(curve, dtype=float)[: step + 1]
        valid = np.where(np.isfinite(arr))[0]
        if len(valid) == 0:
            return None
        return float(arr[valid[-1]])

    def _step_hours(self, step: int) -> float:
        idx = max(0, min(int(step), len(self.time_axis) - 1))
        return float(self.time_axis.hours_end[idx])

    @staticmethod
    def _inhospital_start_hours(ctx) -> Optional[float]:
        ihs = (ctx.demographics or {}).get('inhospital_start')
        if ihs is None or pd.isna(pd.Timestamp(ihs) if not isinstance(ihs, pd.Timestamp) else ihs):
            return None
        delta = (pd.Timestamp(ihs) - ctx.admission_time).total_seconds() / 3600.0
        return float(delta) if delta > 0 else None

    # ------------------------------------------------------------------
    # Explanation internals
    # ------------------------------------------------------------------

    def _build_explanation_response(self, patient_id, ctx, shap_result,
                                    shap_dict, names_cat, names_cont,
                                    eval_step, *, top_n, include_values):
        channels = list(self.bundle['ts_channel_names'])
        ts_shap = shap_dict['ts_shap'][0]                      # [C, L]
        x_ts = shap_dict['test_data']['ts'][0]                 # [C, L] raw values
        traj_len = int(shap_dict['trajectory_length'])

        cat_block = self._categorical_block(
            shap_dict.get('cat_ts_shap_per_category'),
            x_ts_cat=shap_dict['test_data'].get('ts_cat') if include_values else None,
            aggregate=shap_dict.get('cat_ts_shap'),
        )
        static_cat = self._static_block(
            names_cat, shap_dict.get('cat_shap'), ctx.tab_df)
        static_cont = self._static_block(
            names_cont, shap_dict.get('cont_shap'), ctx.tab_df)

        # Completeness within the visible trajectory (missing = NaN in raw x_ts)
        visible = np.asarray(x_ts)[:, :max(traj_len, 1)]
        with np.errstate(invalid='ignore'):
            per_channel = np.mean(np.isfinite(visible), axis=1)
        completeness = {
            'per_channel': {ch: float(frac) for ch, frac in zip(channels, per_channel)},
            'overall': float(np.mean(per_channel)) if len(per_channel) else 0.0,
        }

        ihs_step = None
        ihs_hours = self._inhospital_start_hours(ctx)
        if ihs_hours is not None:
            from astra.evaluation.utils import time_to_step
            ihs_step = time_to_step(ihs_hours, 'h',
                                    data_config=self.bundle['data_config'])

        eval_step = int(eval_step if eval_step is not None
                        else (shap_result.eval_timestep or traj_len - 1))
        return ExplanationResponse(
            patient_id=patient_id,
            pid=ctx.pid,
            model_name=self.model_name,
            eval_step=eval_step,
            eval_hours=self._step_hours(eval_step),
            trajectory_length=traj_len,
            seq_len=self.seq_len,
            time_axis=self.time_axis,
            channels=channels,
            ts_shap=ts_shap,
            ts_values=x_ts if include_values else np.zeros((0, 0)),
            channel_map=self.bundle.get('data_config', {}).get('channel_map'),
            cat_ts=cat_block,
            static_cat=static_cat,
            static_cont=static_cont,
            encoding_info=self.bundle.get('encoding_info'),
            top_features=format_top_features(
                (shap_result.top_features or [])[:top_n]),
            completeness=completeness,
            inhospital_start_step=ihs_step,
        )

    def _categorical_labels(self) -> List[str]:
        """Category labels in the row order used by shap_to_viz_dict."""
        encoding_info = self.bundle.get('encoding_info') or {}
        labels = []
        for feat, (start, end) in encoding_info.get('feature_ranges', {}).items():
            feat_labels = encoding_info.get('category_labels', {}).get(
                feat, [f'{feat}_{i}' for i in range(end - start)])
            labels.extend(feat_labels)
        return labels

    def _categorical_block(self, per_category, x_ts_cat=None, aggregate=None
                           ) -> Optional[CategoricalTSBlock]:
        labels = self._categorical_labels()
        if not labels:
            return None
        seq_len = self.seq_len
        zeros = np.zeros(seq_len)

        if isinstance(per_category, dict):
            # {label: [seq_len]} (differential path)
            if not per_category:
                return None
            per_cat = np.stack([
                np.asarray(per_category.get(lbl, zeros)).squeeze()
                for lbl in labels
            ])
        elif per_category is not None:
            per_cat = np.asarray(per_category)[0]              # [K, L]
        else:
            return None

        agg = (np.asarray(aggregate)[0] if aggregate is not None
               else np.abs(per_cat).mean(axis=0))
        values = None
        if x_ts_cat is not None:
            values = np.asarray(x_ts_cat)[0]
        return CategoricalTSBlock(
            labels=labels,
            shap_per_category=per_cat,
            shap_aggregate=agg,
            values_per_category=values,
        )

    def _static_block(self, names, shap_arr, tab_df) -> Optional[StaticFeatureBlock]:
        if not names or shap_arr is None:
            return None
        shap_vals = np.asarray(shap_arr)[0]
        values = [
            (tab_df[n].iloc[0] if n in tab_df.columns else None)
            for n in names
        ]
        return StaticFeatureBlock(names=list(names), shap=shap_vals, values=values)

    def _static_block_from_dict(self, shap_map, tab_df) -> Optional[StaticFeatureBlock]:
        if not shap_map:
            return None
        names = list(shap_map.keys())
        return StaticFeatureBlock(
            names=names,
            shap=[float(shap_map[n]) for n in names],
            values=[(tab_df[n].iloc[0] if n in tab_df.columns else None)
                    for n in names],
        )
