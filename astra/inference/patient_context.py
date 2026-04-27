"""
PatientContext: encapsulates all patient state for repeated inference.

Supports three modes:

1. **First-time inference** — Create a context via ``PatientContext.create()``
   or ``PatientContext.from_csv()``.  This builds the fixed 30-day bin grid,
   constructs initial tensors, and stores accumulated raw data.

2. **Re-inference (real-time)** — Call ``context.refresh(current_time, new_data)``
   to incorporate new measurements and advance the time horizon.

3. **Simulation (historical)** — When created via ``from_csv()``, the full
   trajectory is stored internally.  Calling ``refresh(new_time)`` without
   ``new_data`` automatically reveals measurements up to *new_time* from
   the stored trajectory, using incremental tensor updates.

Usage::

    session = InferenceSession.load("model_v2")

    # First time
    ctx = PatientContext.create(raw_data, session.bundle)
    result = session.predict_from_context(ctx)

    # Later — new data arrives (keys are concept names)
    new_data = {'VitaleVaerdier': [...], 'Labsvar': [...]}
    ctx.refresh(current_time="2026-02-18 14:00", new_data=new_data)
    result = session.predict_from_context(ctx)

    # Persist / restore
    ctx.save("patients/patient_abc.pkl")
    ctx = PatientContext.load("patients/patient_abc.pkl", bundle=session.bundle)
"""

import copy
import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatientContext:
    """Encapsulates all patient state for inference across repeated calls.

    The bin grid is fixed at creation (admission + 30 days) so that tensor
    positions are stable across re-inferences.  ``current_time`` controls
    which bins are "visible" (i.e. ``trajectory_length``).
    """

    # ---- Identity ----------------------------------------------------------
    pid: Any
    admission_time: pd.Timestamp
    max_time: pd.Timestamp  # admission + 30 days

    # ---- Static (set once at creation) -------------------------------------
    demographics: dict
    tab_df: pd.DataFrame
    bin_df: pd.DataFrame  # full 30-day grid, never changes

    # ---- Dynamic (updated on refresh) --------------------------------------
    current_time: pd.Timestamp
    trajectory_length: int
    x_ts: np.ndarray       # [n_channels, seq_len]
    x_ts_cat: np.ndarray   # [n_cat_dims, seq_len]

    # ---- Internal bookkeeping ----------------------------------------------
    _raw_data: dict = field(repr=False)  # accumulated raw_data dict (time-filtered)
    _bundle_name: Optional[str] = field(default=None, repr=False)
    _bundle_ref: Optional[dict] = field(default=None, repr=False)
    _ebm_context: Optional[dict] = field(default=None, repr=False)

    # ---- New fields for incremental / simulation ----------------------------
    patient_end_time: Optional[pd.Timestamp] = field(default=None, repr=False)
    _full_trajectory_data: Optional[dict] = field(default=None, repr=False)
    _bin_cache: Optional[Any] = field(default=None, repr=False)  # BinCache
    _last_refresh_time: Optional[pd.Timestamp] = field(default=None, repr=False)
    _ebm_cache: Optional[Dict[float, float]] = field(default=None, repr=False)
    _timing: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------------- #
    # Construction
    # ---------------------------------------------------------------------- #

    @classmethod
    def create(
        cls,
        raw_data: dict,
        bundle: dict,
    ) -> "PatientContext":
        """Create a PatientContext from a standardised raw_data dict.

        This is the primary constructor.  It builds the fixed 30-day bin grid,
        constructs all tensors, and stores the raw data for future refreshes.

        ``raw_data`` follows the same schema as
        :func:`~astra.inference.data_prep.prepare_single_patient`.
        """
        from astra.inference.data_prep import (
            _create_patient_bins,
            _build_continuous_ts,
            _build_categorical_ts,
            _build_tab_df,
            _populate_cache_from_raw_data,
            timed_stage,
        )
        from astra.evaluation.utils import time_to_step

        timing = {}

        # Parse timestamps
        raw_data = copy.deepcopy(raw_data)
        raw_data['admission_time'] = pd.Timestamp(raw_data['admission_time'])
        raw_data['current_time'] = pd.Timestamp(raw_data['current_time'])

        admission_time = raw_data['admission_time']
        current_time = raw_data['current_time']
        data_config = bundle['data_config']

        # 1. Fixed bin grid (window derived from bin_intervals config)
        bin_df = _create_patient_bins(admission_time, data_config)

        # Convert elapsed time to step index using cfg bin intervals
        delta_minutes = (current_time - admission_time).total_seconds() / 60
        step = time_to_step(delta_minutes, 'min', data_config=data_config)
        visible_bins = min(step + 1, len(bin_df)) if step is not None else len(bin_df)

        logger.info(
            f"PatientContext: {len(bin_df)} bins, "
            f"{visible_bins} visible at {current_time}"
        )

        # 2. Tensors
        with timed_stage(timing, 'continuous_build'):
            x_ts, trajectory_length = _build_continuous_ts(
                raw_data, bin_df, bundle, trajectory_length=visible_bins,
            )
        with timed_stage(timing, 'categorical_build'):
            x_ts_cat = _build_categorical_ts(raw_data, bin_df, bundle)

        # Zero out categorical beyond visibility
        seq_len = bundle['model_params']['seq_len']
        if trajectory_length < seq_len:
            x_ts_cat[:, trajectory_length:] = 0.0

        # 3. Static tabular
        tab_df = _build_tab_df(raw_data, bundle)

        # 4. Populate bin cache for future incremental updates
        bin_cache = _populate_cache_from_raw_data(raw_data, bin_df, bundle)
        bin_cache.dirty_continuous.clear()
        bin_cache.dirty_categorical.clear()

        from astra.inference.data_prep import _max_prediction_window

        return cls(
            pid=raw_data.get('pid'),
            admission_time=admission_time,
            max_time=admission_time + _max_prediction_window(data_config),
            demographics=raw_data.get('demographics', {}),
            tab_df=tab_df,
            bin_df=bin_df,
            current_time=current_time,
            trajectory_length=trajectory_length,
            x_ts=x_ts,
            x_ts_cat=x_ts_cat,
            _raw_data=raw_data,
            _bundle_name=bundle.get('model_name'),
            _bundle_ref=bundle,
            _bin_cache=bin_cache,
            _last_refresh_time=current_time,
            _timing=timing,
        )

    @classmethod
    def from_raw_ehr(
        cls,
        raw_ehr: dict,
        bundle: dict,
    ) -> "PatientContext":
        """Create from raw Danish EHR data (same schema as ``prepare_from_raw_ehr``).

        Standardises feature names then delegates to :meth:`create`.
        """
        from astra.inference.data_prep import (
            _standardize_vitals,
            _standardize_labs,
            _standardize_icu,
            _standardize_ews,
            _standardize_medications,
            _standardize_procedures,
            _standardize_adt,
            SEX_MAP,
        )
        from astra.data.mappings import derive_first_hospital

        admission_time = pd.Timestamp(raw_ehr['admission_time'])

        age = raw_ehr.get('age')
        if age is None and raw_ehr.get('dob') is not None:
            dob = pd.Timestamp(raw_ehr['dob'])
            age = int((admission_time - dob).days / 365.25)

        sex_raw = raw_ehr.get('sex', np.nan)
        sex = SEX_MAP.get(str(sex_raw), sex_raw)

        hosp = raw_ehr.get('first_hospital')
        if hosp is None and raw_ehr.get('first_department'):
            hosp = derive_first_hospital(raw_ehr['first_department'])

        # Map API short keys → concept names used throughout the pipeline.
        # The external API uses readable keys (vitals, labs, etc.); internally
        # we use concept names (VitaleVaerdier, Labsvar, etc.) for config-driven
        # processing.
        raw_data = {
            'pid': raw_ehr.get('pid'),
            'admission_time': raw_ehr['admission_time'],
            'current_time': raw_ehr['current_time'],
            'demographics': {
                'AGE': age,
                'SEX': sex,
                'FIRST_HOSPITAL': hosp,
                'HEIGHT': raw_ehr.get('height_cm'),
                'WEIGHT': raw_ehr.get('weight_kg'),
                'ASMT_ELIX': raw_ehr.get('elixhauser_score'),
            },
            'VitaleVaerdier': _standardize_vitals(raw_ehr.get('vitals', [])),
            'Labsvar': _standardize_labs(raw_ehr.get('labs', [])),
            'ITAOversigtsrapport': _standardize_icu(raw_ehr.get('icu_scores', [])),
            'Medicin': _standardize_medications(raw_ehr.get('medications', [])),
            'Procedurer': _standardize_procedures(raw_ehr.get('procedures', [])),
            'ADTHaendelser': _standardize_adt(raw_ehr.get('adt', [])),
            'EWS': _standardize_ews(raw_ehr.get('ews', [])),
        }

        return cls.create(raw_data, bundle)

    @classmethod
    def from_csv(
        cls,
        cpr_hash: str,
        service_date,
        current_time,
        bundle: dict,
        cfg: dict = None,
        data_dir: str = 'data/raw',
        ebm_models_dir: str = 'models/ebm',
        start_hours: float = None,
        patient_dir: str = 'data/patients',
    ) -> "PatientContext":
        """Create from raw CSV files (stateless — no shared file writes).

        Loads the **full trajectory** from CSVs and stores it in
        ``_full_trajectory_data`` for simulation time-stepping.  Only data
        up to *current_time* is used for the initial tensor build.

        Args:
            start_hours: If provided, derive *current_time* as
                ``admission_time + start_hours`` using the actual admission
                time from ``base_df['start']``.  This avoids misalignment
                when ``service_date`` (date-only) differs from the true
                admission timestamp (e.g. with prehospital data).
            patient_dir: Directory with pre-split per-patient CSVs.
        """
        from astra.inference.data_prep import (
            _build_single_patient_base_df,
            _filter_concepts_for_patient,
            _filtered_dfs_to_raw_data,
            _filter_raw_data_by_time,
            timed_stage,
        )

        timing = {}

        if cfg is None:
            from astra.utils import get_cfg
            cfg = get_cfg()

        # Phase 1: Build base_df (stateless)
        with timed_stage(timing, 'csv_load'):
            base_df = _build_single_patient_base_df(cpr_hash, service_date, cfg,
                                                    data_dir, patient_dir=patient_dir)
            logger.info(
                f"Built base_df for patient {cpr_hash[:8]}...: "
                f"trajectory {base_df['start'].iloc[0]} -> {base_df['end'].iloc[0]}"
            )

            # Phase 2: Filter concepts (stateless) — loads full trajectory
            filtered_concepts = _filter_concepts_for_patient(base_df, cfg, data_dir,
                                                             patient_dir=patient_dir)
            logger.info(
                f"Filtered {len(filtered_concepts)} concepts: "
                f"{list(filtered_concepts.keys())}"
            )

        # If start_hours given, derive current_time from actual admission
        # (avoids mismatch when service_date differs from base_df['start'],
        # e.g. prehospital shifting the trajectory start earlier).
        if start_hours is not None:
            admission_time = pd.Timestamp(base_df['start'].iloc[0])
            current_time = admission_time + pd.Timedelta(hours=start_hours)

        # Clamp current_time to patient's actual trajectory end so that
        # visible bins match the batch path (which is bounded by data extent).
        patient_end = base_df['end'].iloc[0]
        if pd.notna(patient_end):
            clamped = min(pd.Timestamp(current_time), pd.Timestamp(patient_end))
            if clamped < pd.Timestamp(current_time):
                logger.info(f"Clamped current_time from {current_time} to {clamped} (patient end)")
            current_time = clamped

        # Phase 3a: Build full trajectory raw_data (unfiltered by time)
        raw_data_full = _filtered_dfs_to_raw_data(
            base_df, filtered_concepts, current_time=current_time,
            cfg=cfg, filter_by_time=False,
        )

        # Phase 3b: Build time-filtered raw_data for initial tensors
        with timed_stage(timing, 'time_filter'):
            raw_data = _filter_raw_data_by_time(raw_data_full, current_time)

        # Phase 4: Create context from time-filtered data
        ctx = cls.create(raw_data, bundle)
        ctx.patient_end_time = pd.Timestamp(patient_end) if pd.notna(patient_end) else None
        ctx._full_trajectory_data = raw_data_full
        # Merge timing from csv_load into context timing
        for k, v in timing.items():
            ctx._timing.setdefault(k, []).extend(v)

        # Phase 5: Inject EBM predictions if model expects them
        if '_ebm_pred' in bundle.get('ts_channel_names', []):
            from astra.inference.ebm import (
                compute_ebm_predictions, inject_ebm_into_x_ts,
            )

            with timed_stage(ctx._timing, 'ebm_compute'):
                ebm_preds = compute_ebm_predictions(
                    raw_data, filtered_concepts, base_df, cfg, ebm_models_dir,
                )
            with timed_stage(ctx._timing, 'ebm_inject'):
                ctx.x_ts = inject_ebm_into_x_ts(
                    ctx.x_ts, ebm_preds, ctx.bin_df,
                    raw_data['admission_time'], bundle,
                    trajectory_length=ctx.trajectory_length,
                )

            # Store context for EBM re-injection on refresh()
            ctx._ebm_context = {
                'filtered_concepts': filtered_concepts,
                'base_df': base_df,
                'cfg': cfg,
                'ebm_models_dir': ebm_models_dir,
            }
            ctx._ebm_cache = dict(ebm_preds)

        return ctx

    # ---------------------------------------------------------------------- #
    # Refresh (re-inference with updated data / time)
    # ---------------------------------------------------------------------- #

    def refresh(
        self,
        current_time,
        new_data: Optional[dict] = None,
        profiling: Optional[dict] = None,
    ) -> dict:
        """Update patient state and return model-ready tensors.

        **Incremental mode** (default when ``_bin_cache`` is available):
        only processes newly arriving data and updates dirty bins.

        **Simulation mode** (when ``_full_trajectory_data`` is set and no
        *new_data* is provided): automatically reveals measurements from
        the stored full trajectory up to *current_time*.

        Args:
            current_time: New time horizon for visibility masking.
            new_data: Optional dict with new measurements to append.
                Same keys as ``raw_data`` (vitals, labs, icu, medications,
                procedures, adt).  New entries are appended to accumulated
                data; existing entries are preserved.
            profiling: Optional dict to collect sub-stage timing (for perf analysis).

        Returns:
            Dict with x_ts, x_ts_cat, tab_df, trajectory_length, bin_df —
            same schema as ``prepare_single_patient`` output.
        """
        from astra.inference.data_prep import (
            _build_continuous_ts,
            _build_categorical_ts,
            _build_continuous_ts_incremental,
            _build_categorical_ts_incremental,
            _filter_raw_data_by_time,
            timed_stage,
        )
        from contextlib import nullcontext as _nullctx
        from astra.evaluation.utils import time_to_step

        def _ts(name):
            return timed_stage(profiling, name) if profiling is not None else _nullctx()

        new_current_time = pd.Timestamp(current_time)
        if self.patient_end_time is not None and new_current_time > self.patient_end_time:
            logger.info(f"Clamped refresh current_time from {new_current_time} to {self.patient_end_time}")
            new_current_time = self.patient_end_time
        old_trajectory_length = self.trajectory_length

        bundle = self._bundle_ref
        if bundle is None:
            raise RuntimeError(
                "Bundle reference lost (was this context deserialized without "
                "re-attaching the bundle? Use PatientContext.load(path, bundle=...))"
            )

        # Recompute visibility using cfg bin intervals
        data_config = bundle['data_config']
        delta_minutes = (new_current_time - self.admission_time).total_seconds() / 60
        step = time_to_step(delta_minutes, 'min', data_config=data_config)
        visible_bins = min(step + 1, len(self.bin_df)) if step is not None else len(self.bin_df)

        # ----- Determine new measurements to process -----
        # Classification is config-driven: ts_cat_names distinguishes
        # categorical concepts from continuous; interval vs point is detected
        # from data structure ('start' key = interval, 'feature' key = continuous).
        from astra.inference.data_prep import _RAW_DATA_META_KEYS
        ts_cat_names = set(data_config.get('ts_cat_names', []))

        incremental_records = []  # continuous point events
        incremental_cat_events = {}  # {concept_name: [event_dicts]}

        if new_data is not None:
            # Real-time mode: use provided new_data (filter by time)
            for key, entries in new_data.items():
                if key in _RAW_DATA_META_KEYS or not isinstance(entries, list):
                    continue
                for m in entries:
                    is_interval = 'start' in m
                    ts_val = pd.Timestamp(m['start'] if is_interval else m['timestamp'])
                    if ts_val > new_current_time:
                        continue
                    if is_interval:
                        m = dict(m)
                        m['end'] = min(pd.Timestamp(m['end']), new_current_time)
                    if key in ts_cat_names:
                        incremental_cat_events.setdefault(key, []).append(m)
                    else:
                        incremental_records.append(m)
                    self._raw_data.setdefault(key, []).append(m)

        elif self._full_trajectory_data is not None:
            # Simulation mode: reveal data from stored trajectory
            last_time = self._last_refresh_time or self.admission_time
            with timed_stage(self._timing, 'time_filter'):
                with _ts('reveal_scan'):
                    for key, items in self._full_trajectory_data.items():
                        if key in _RAW_DATA_META_KEYS or not isinstance(items, list):
                            continue
                        if not items:
                            continue

                        sample = items[0]
                        is_interval = 'start' in sample
                        is_categorical = key in ts_cat_names

                        if is_interval:
                            # Interval events: include intervals that started in
                            # (last_time, new_time] OR ongoing intervals extending
                            # into newly visible bins.
                            new_entries = []
                            for evt in items:
                                start = pd.Timestamp(evt['start'])
                                end = pd.Timestamp(evt['end'])
                                if start > new_current_time:
                                    continue
                                if start > last_time:
                                    clamped = dict(evt)
                                    clamped['end'] = min(end, new_current_time)
                                    new_entries.append(clamped)
                                    self._raw_data.setdefault(key, []).append(clamped)
                                elif end > last_time:
                                    clamped = {
                                        'start': evt['start'],
                                        'end': min(end, new_current_time),
                                        'value': evt['value'],
                                    }
                                    new_entries.append(clamped)
                            if new_entries:
                                incremental_cat_events[key] = new_entries
                        else:
                            # Point events (continuous or categorical)
                            new_entries = []
                            for m in items:
                                ts = pd.Timestamp(m['timestamp'])
                                if ts > last_time and ts <= new_current_time:
                                    new_entries.append(m)
                                    self._raw_data.setdefault(key, []).append(m)
                            if is_categorical and new_entries:
                                incremental_cat_events[key] = new_entries
                            elif new_entries:
                                incremental_records.extend(new_entries)

        self.current_time = new_current_time
        self._raw_data['current_time'] = self.current_time

        # ----- Incremental or full tensor update -----
        has_new_data = bool(incremental_records) or bool(incremental_cat_events)

        if self._bin_cache is not None:
            # Incremental path
            with timed_stage(self._timing, 'continuous_build'):
                self.x_ts, self.trajectory_length = _build_continuous_ts_incremental(
                    new_records=incremental_records,
                    bin_df=self.bin_df,
                    bundle=bundle,
                    cache=self._bin_cache,
                    x_ts_existing=self.x_ts,
                    old_trajectory_length=old_trajectory_length,
                    trajectory_length=visible_bins,
                    admission_time=self.admission_time,
                    profiling=profiling,
                    raw_data=self._raw_data,
                )
            with timed_stage(self._timing, 'categorical_build'):
                self.x_ts_cat = _build_categorical_ts_incremental(
                    new_events=incremental_cat_events,
                    bin_df=self.bin_df,
                    bundle=bundle,
                    cache=self._bin_cache,
                    x_ts_cat_existing=self.x_ts_cat,
                    trajectory_length=visible_bins,
                )
        else:
            # Fallback: full rebuild (e.g. deserialized context without cache)
            with timed_stage(self._timing, 'continuous_build'):
                self.x_ts, self.trajectory_length = _build_continuous_ts(
                    self._raw_data, self.bin_df, bundle,
                    trajectory_length=visible_bins,
                )
            with timed_stage(self._timing, 'categorical_build'):
                self.x_ts_cat = _build_categorical_ts(
                    self._raw_data, self.bin_df, bundle,
                )
            # Zero out beyond visibility
            seq_len = bundle['model_params']['seq_len']
            if self.trajectory_length < seq_len:
                self.x_ts_cat[:, self.trajectory_length:] = 0.0

        # ----- EBM: only compute new intervals -----
        if '_ebm_pred' in bundle.get('ts_channel_names', []) and self._ebm_context is not None:
            self._refresh_ebm(bundle)

        self._last_refresh_time = self.current_time

        logger.info(
            f"Refreshed context: trajectory_length={self.trajectory_length}, "
            f"current_time={self.current_time}, "
            f"incremental={'yes' if self._bin_cache is not None else 'no'}, "
            f"new_records={len(incremental_records)}"
        )

        return self.to_dict()

    def _refresh_ebm(self, bundle: dict) -> None:
        """Incrementally update EBM predictions — only compute new intervals."""
        from astra.inference.ebm import (
            compute_ebm_predictions, inject_ebm_into_x_ts,
        )
        from astra.inference.data_prep import timed_stage

        if self._ebm_cache is None:
            self._ebm_cache = {}

        with timed_stage(self._timing, 'ebm_compute'):
            new_preds = compute_ebm_predictions(
                self._raw_data,
                self._ebm_context['filtered_concepts'],
                self._ebm_context['base_df'],
                self._ebm_context['cfg'],
                self._ebm_context['ebm_models_dir'],
                cached_predictions=self._ebm_cache,
            )

        self._ebm_cache.update(new_preds)

        with timed_stage(self._timing, 'ebm_inject'):
            self.x_ts = inject_ebm_into_x_ts(
                self.x_ts, self._ebm_cache, self.bin_df,
                self.admission_time, bundle,
                trajectory_length=self.trajectory_length,
            )

    # ---------------------------------------------------------------------- #
    # Timing
    # ---------------------------------------------------------------------- #

    def get_timing_summary(self) -> dict:
        """Return ``{stage: {mean_ms, total_ms, count, last_ms}}``."""
        summary = {}
        for stage, durations in self._timing.items():
            total = sum(durations)
            count = len(durations)
            summary[stage] = {
                'mean_ms': (total / count) * 1000 if count else 0,
                'total_ms': total * 1000,
                'count': count,
                'last_ms': durations[-1] * 1000 if durations else 0,
            }
        return summary

    # ---------------------------------------------------------------------- #
    # Output helpers
    # ---------------------------------------------------------------------- #

    def to_dict(self) -> dict:
        """Return model-ready tensors in the same format as ``prepare_single_patient``."""
        return {
            'x_ts': self.x_ts,
            'x_ts_cat': self.x_ts_cat,
            'tab_df': self.tab_df,
            'trajectory_length': self.trajectory_length,
            'bin_df': self.bin_df,
        }

    # ---------------------------------------------------------------------- #
    # Serialization
    # ---------------------------------------------------------------------- #

    def save(self, path: Union[str, Path]) -> None:
        """Persist context to disk.

        The deployment bundle reference is NOT saved (too large).  When
        loading, pass the bundle explicitly via ``PatientContext.load()``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'pid': self.pid,
            'admission_time': self.admission_time,
            'max_time': self.max_time,
            'demographics': self.demographics,
            'tab_df': self.tab_df,
            'bin_df': self.bin_df,
            'current_time': self.current_time,
            'trajectory_length': self.trajectory_length,
            'x_ts': self.x_ts,
            'x_ts_cat': self.x_ts_cat,
            '_raw_data': self._raw_data,
            '_bundle_name': self._bundle_name,
            '_ebm_context': self._ebm_context,
            '_full_trajectory_data': self._full_trajectory_data,
            '_bin_cache': self._bin_cache,
            '_last_refresh_time': self._last_refresh_time,
            '_ebm_cache': self._ebm_cache,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved PatientContext to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        bundle: dict,
    ) -> "PatientContext":
        """Load a persisted context and re-attach the deployment bundle.

        Args:
            path: Path to the saved context file.
            bundle: Deployment bundle (must match the one used at creation).
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        ctx = cls(
            pid=state['pid'],
            admission_time=state['admission_time'],
            max_time=state['max_time'],
            demographics=state['demographics'],
            tab_df=state['tab_df'],
            bin_df=state['bin_df'],
            current_time=state['current_time'],
            trajectory_length=state['trajectory_length'],
            x_ts=state['x_ts'],
            x_ts_cat=state['x_ts_cat'],
            _raw_data=state['_raw_data'],
            _bundle_name=state.get('_bundle_name'),
            _bundle_ref=bundle,
            _ebm_context=state.get('_ebm_context'),
            _full_trajectory_data=state.get('_full_trajectory_data'),
            _bin_cache=state.get('_bin_cache'),
            _last_refresh_time=state.get('_last_refresh_time'),
            _ebm_cache=state.get('_ebm_cache'),
        )

        logger.info(f"Loaded PatientContext from {path} (pid={ctx.pid})")
        return ctx
