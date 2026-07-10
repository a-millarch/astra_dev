"""
Simulation runner for ASTRA inference pipeline.

Steps a historical patient through time (bin by bin), collects predictions
and per-step timing data to benchmark real-world deployment performance.

Usage::

    from astra.inference import InferenceSession
    from astra.inference.simulation import SimulationRunner

    session = InferenceSession.load("model_v2", device="cpu")
    runner = SimulationRunner(session)
    result = runner.run("abc123hash", "2025-06-15")
    print(result.to_dataframe())
    result.plot_trajectory("simulation_output.png")

Interactive (pause & inspect)::

    runner = SimulationRunner(session)
    runner.setup("abc123hash", "2025-06-15")
    runner.advance_to(hours=12)
    runner.inspect()              # SHAP, trajectory, data completeness
    runner.advance_to(hours=24)
    runner.inspect()

Real-time benchmark (mimics ETL-fed data delivery)::

    result_rt = runner.run_realtime("abc123hash", "2025-06-15")
    result_sim = runner.run("abc123hash", "2025-06-15")
    # Compare predictions and timing between the two paths
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from astra.utils import ensure_parent_dir

logger = logging.getLogger(__name__)


# ============================================================================
# Result dataclasses
# ============================================================================

@dataclass
class SimulationStep:
    """Result of a single simulation time step."""
    current_time: pd.Timestamp
    elapsed_hours: float
    trajectory_length: int
    probability: float
    predictions_over_time: Optional[np.ndarray] = None  # temporal head only
    survival_curve: Optional[np.ndarray] = None         # survival mode: S(t) at each step
    step_timing: dict = field(default_factory=dict)
    n_new_measurements: int = 0


@dataclass
class SimulationResult:
    """Complete output of a simulation run."""
    pid: Any
    admission_time: pd.Timestamp
    steps: List[SimulationStep]
    total_timing: dict = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    prediction_curve: Optional[np.ndarray] = None  # [seq_len], NaN where unpredicted
    inhospital_start_hours: Optional[float] = None  # hours after admission
    profiling: Optional[dict] = field(default=None, repr=False)  # fine-grained sub-stage timing

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def print_profile(self):
        """Log a comprehensive profiling summary.

        Shows three sections:
        1. **Step-level**: mean/total for coarse refresh vs predict per step.
        2. **Context-level**: accumulated timing from PatientContext._timing
           (continuous_build, categorical_build, time_filter, ebm_*).
        3. **Sub-stage**: fine-grained breakdown within refresh and predict
           (reveal_scan, temporal_features, normalization, model_forward, etc.).
        """
        lines = [
            f"\n{'='*70}",
            f"SIMULATION PROFILE: {self.n_steps} steps, "
            f"{self.wall_clock_seconds:.1f}s wall clock",
            f"{'='*70}",
        ]

        # --- Section 1: Step-level (refresh vs predict) ---
        refresh_ms = []
        predict_ms = []
        for s in self.steps:
            for stage, durations in s.step_timing.items():
                val = sum(durations) * 1000
                if stage == 'refresh':
                    refresh_ms.append(val)
                elif stage == 'predict':
                    predict_ms.append(val)

        lines.append("\n-- Step-level (per-step mean / total) --")
        for label, vals in [('refresh', refresh_ms), ('predict', predict_ms)]:
            if vals:
                arr = np.array(vals)
                lines.append(
                    f"  {label:20s}: mean={arr.mean():7.1f}ms  "
                    f"total={arr.sum():8.0f}ms  "
                    f"min={arr.min():7.1f}ms  max={arr.max():7.1f}ms"
                )

        accounted = sum(refresh_ms) + sum(predict_ms)
        overhead = self.wall_clock_seconds * 1000 - accounted
        lines.append(f"  {'loop overhead':20s}: {overhead:8.0f}ms")

        # --- Section 2: Context-level (PatientContext._timing) ---
        if self.total_timing:
            lines.append("\n-- Context-level (accumulated across all steps) --")
            for stage, durations in sorted(self.total_timing.items()):
                total = sum(durations) * 1000
                count = len(durations)
                mean = total / count if count else 0
                lines.append(
                    f"  {stage:20s}: mean={mean:7.1f}ms  "
                    f"total={total:8.0f}ms  count={count}"
                )

        # --- Section 3: Sub-stage profiling ---
        if self.profiling:
            lines.append("\n-- Sub-stage profiling (fine-grained breakdown) --")
            # Sort by total time descending
            stage_totals = []
            for stage, durations in self.profiling.items():
                total = sum(durations) * 1000
                count = len(durations)
                mean = total / count if count else 0
                stage_totals.append((stage, mean, total, count))
            stage_totals.sort(key=lambda x: x[2], reverse=True)

            for stage, mean, total, count in stage_totals:
                pct = total / (self.wall_clock_seconds * 1000) * 100
                lines.append(
                    f"  {stage:25s}: mean={mean:7.2f}ms  "
                    f"total={total:8.0f}ms  count={count:4d}  "
                    f"({pct:5.1f}%)"
                )

            # Unaccounted time
            profiled_total = sum(
                sum(d) * 1000 for d in self.profiling.values()
            )
            # Note: sub-stages overlap with step-level, so just show totals
            lines.append(f"\n  Sub-stage total: {profiled_total:.0f}ms "
                         f"(may overlap with step-level timings)")

        lines.append(f"{'='*70}")
        logger.info('\n'.join(lines))

    def to_dataframe(self) -> pd.DataFrame:
        """One row per step: elapsed_hours, probability, timing breakdown."""
        rows = []
        for s in self.steps:
            row = {
                'current_time': s.current_time,
                'elapsed_hours': s.elapsed_hours,
                'trajectory_length': s.trajectory_length,
                'probability': s.probability,
                'n_new_measurements': s.n_new_measurements,
            }
            # Flatten step timing into columns
            for stage, durations in s.step_timing.items():
                row[f'timing_{stage}_ms'] = sum(durations) * 1000
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_trajectory(self, save_path=None, show: bool = False):
        """Plot P(deceased) over time with optional timing subplot.

        Args:
            save_path: If provided, save figure to this path.
            show: Whether to call plt.show().
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        timing_cols = [c for c in df.columns if c.startswith('timing_')]

        fig, axes = plt.subplots(
            2 if timing_cols else 1, 1,
            figsize=(12, 6 if timing_cols else 4),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1]} if timing_cols else None,
        )
        if not timing_cols:
            axes = [axes]

        # -- Prediction trajectory --
        ax = axes[0]
        ax.plot(df['elapsed_hours'], df['probability'], 'b-o', markersize=2, linewidth=1)
        if self.inhospital_start_hours is not None and self.inhospital_start_hours > 0:
            ax.axvline(x=self.inhospital_start_hours, color='#2196F3', linewidth=1.5,
                       linestyle=':', alpha=0.8,
                       label=f'Hospital arrival ({self.inhospital_start_hours:.1f}h)')
            ax.legend()
        has_survival = any(s.survival_curve is not None for s in self.steps)
        ylabel = 'Cumulative risk (1 - S(t))' if has_survival else 'P(deceased 30d)'
        ax.set_ylabel(ylabel)
        ax.set_title(f'Simulation: patient {self.pid} ({self.n_steps} steps, '
                      f'{self.wall_clock_seconds:.1f}s total)')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # -- Timing breakdown --
        if timing_cols:
            ax2 = axes[1]
            bottom = np.zeros(len(df))
            for col in timing_cols:
                label = col.replace('timing_', '').replace('_ms', '')
                vals = df[col].fillna(0).values
                ax2.bar(df['elapsed_hours'], vals, bottom=bottom,
                        width=0.3, label=label, alpha=0.7)
                bottom += vals
            ax2.set_ylabel('Time (ms)')
            ax2.set_xlabel('Elapsed hours')
            ax2.legend(fontsize=7, ncol=3)
            ax2.grid(True, alpha=0.3)
        else:
            axes[0].set_xlabel('Elapsed hours')

        plt.tight_layout()
        if save_path:
            ensure_parent_dir(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved simulation plot to {save_path}")
        if show:
            plt.show()
        plt.close(fig)

        return fig


# ============================================================================
# SimulationRunner
# ============================================================================

class SimulationRunner:
    """Steps a historical patient through time, collecting predictions and timings.

    Uses :class:`~astra.inference.patient_context.PatientContext` with
    incremental tensor updates for optimal performance.  All computation
    is CPU-only.
    """

    def __init__(self, session):
        """
        Args:
            session: An :class:`~astra.inference.pipeline.InferenceSession`.
                Should be loaded with ``device='cpu'``.
        """
        self.session = session
        self.context = None
        self._time_points: List[pd.Timestamp] = []
        self._step_idx: int = 0
        self._steps: List[SimulationStep] = []
        self._prev_raw_counts: int = 0
        self._prediction_curve: Optional[np.ndarray] = None

    # ---- Interactive (step-through) API ----

    def setup(
        self,
        cpr_hash: str,
        service_date,
        cfg: dict = None,
        data_dir: str = 'data/raw',
        ebm_models_dir: str = 'models/ebm',
        start_hours: float = 0.0,
        patient_dir: str = 'data/patients',
    ):
        """Create PatientContext at admission and prepare for stepping.

        After calling this, use :meth:`advance_to` to step through time
        and :meth:`inspect` to visualize model behavior at the current time.

        Args:
            cpr_hash: Patient identifier hash.
            service_date: Admission date (for base_df lookup).
            cfg: Configuration dict (loaded from defaults.yaml if None).
            data_dir: Path to raw CSV data.
            ebm_models_dir: Path to saved EBM models.
            start_hours: Start at this many hours after admission.
            patient_dir: Directory with pre-split per-patient CSVs.
        """
        from astra.inference.patient_context import PatientContext

        self.context = PatientContext.from_csv(
            cpr_hash=cpr_hash,
            service_date=service_date,
            current_time=pd.Timestamp(service_date),
            bundle=self.session.bundle,
            cfg=cfg,
            data_dir=data_dir,
            ebm_models_dir=ebm_models_dir,
            start_hours=start_hours,
            patient_dir=patient_dir,
        )

        self._time_points = _generate_bin_aligned_times(
            self.context.bin_df,
            self.context.admission_time,
            start_time=self.context.current_time,
        )
        self._step_idx = 0
        self._steps = []
        self._prev_raw_counts = _count_raw_data(self.context._raw_data)
        self._prediction_curve = np.full(len(self.context.bin_df), np.nan)

        # Make context available for default_session_plot
        self.session.ctx = self.context

        logger.info(
            f"Setup complete: pid={self.context.pid}, "
            f"{len(self._time_points)} time points, "
            f"start={self.context.current_time}"
        )

    def advance_to(
        self,
        hours: Optional[float] = None,
        time: Optional[pd.Timestamp] = None,
    ) -> List[SimulationStep]:
        """Advance simulation to a target time, returning steps taken.

        Incrementally steps through bin boundaries up to the target.
        After advancing, ``session.ctx`` is updated for inspection via
        :func:`~astra.inference.run_inference.default_session_plot`.

        Args:
            hours: Target elapsed hours from admission.
            time: Target absolute timestamp. Provide one of *hours* or *time*.

        Returns:
            List of :class:`SimulationStep` for the steps just taken.
        """
        if self.context is None:
            raise RuntimeError("Call setup() before advance_to()")

        if hours is not None:
            target = self.context.admission_time + pd.Timedelta(hours=hours)
        elif time is not None:
            target = pd.Timestamp(time)
        else:
            raise ValueError("Provide either hours= or time=")

        from astra.inference.data_prep import timed_stage

        new_steps = []
        while self._step_idx < len(self._time_points):
            tp = self._time_points[self._step_idx]
            if tp > target:
                break

            step_timing = {}

            with timed_stage(step_timing, 'refresh'):
                self.context.refresh(tp)

            new_counts = _count_raw_data(self.context._raw_data)
            n_new = new_counts - self._prev_raw_counts
            self._prev_raw_counts = new_counts

            with timed_stage(step_timing, 'predict'):
                result = self.session.predict_from_context(self.context)

            # Store prediction at current bin position
            bin_idx = self.context.trajectory_length - 1
            if 0 <= bin_idx < len(self._prediction_curve):
                self._prediction_curve[bin_idx] = result.probability

            elapsed = (tp - self.context.admission_time).total_seconds() / 3600

            step = SimulationStep(
                current_time=tp,
                elapsed_hours=elapsed,
                trajectory_length=self.context.trajectory_length,
                probability=result.probability,
                predictions_over_time=result.predictions_over_time,
                survival_curve=getattr(result, 'survival_curve', None),
                step_timing=step_timing,
                n_new_measurements=n_new,
            )
            new_steps.append(step)
            self._steps.append(step)
            self._step_idx += 1

        # Update session context for inspection
        self.session.ctx = self.context

        if new_steps:
            logger.info(
                f"Advanced {len(new_steps)} steps to "
                f"{new_steps[-1].elapsed_hours:.1f}h "
                f"(P={new_steps[-1].probability:.4f})"
            )
        else:
            logger.info("No new steps to advance (already at or past target)")

        return new_steps

    def inspect(self):
        """Run default_session_plot on the current context.

        Uses the accumulated prediction curve so the trajectory plot
        shows per-timestep predictions for both temporal and non-temporal
        models (non-temporal models would otherwise need a pre-computed CSV).
        """
        if self.context is None:
            raise RuntimeError("Call setup() before inspect()")

        from astra.visualize.inference import default_session_plot
        self.session.ctx = self.context
        default_session_plot(self.session, prediction_curve=self._prediction_curve)

    @property
    def result(self) -> Optional[SimulationResult]:
        """Build a SimulationResult from steps accumulated so far."""
        if not self._steps or self.context is None:
            return None
        ihs_hours = _get_inhospital_start_hours(self.context)
        return SimulationResult(
            pid=self.context.pid,
            admission_time=self.context.admission_time,
            steps=list(self._steps),
            total_timing=dict(self.context._timing),
            prediction_curve=self._prediction_curve.copy() if self._prediction_curve is not None else None,
            inhospital_start_hours=ihs_hours,
        )

    @property
    def elapsed_hours(self) -> Optional[float]:
        """Current elapsed hours (from last step taken)."""
        if not self._steps:
            return 0.0 if self.context else None
        return self._steps[-1].elapsed_hours

    @property
    def remaining_steps(self) -> int:
        """Number of time points not yet advanced through."""
        return len(self._time_points) - self._step_idx

    # ---- Real-time benchmark API ----

    def run_realtime(
        self,
        cpr_hash: str,
        service_date,
        cfg: dict = None,
        data_dir: str = 'data/raw',
        ebm_models_dir: str = 'models/ebm',
        start_hours: float = 0.0,
        end_hours: Optional[float] = None,
        patient_dir: str = 'data/patients',
    ) -> SimulationResult:
        """Run simulation mimicking real-time ETL-fed data delivery.

        Unlike :meth:`run` which pre-loads the full trajectory and reveals
        data internally via ``_full_trajectory_data``, this method:

        1. Loads CSV data the same way (for apples-to-apples comparison)
        2. Creates a :class:`PatientContext` with only initial demographics
           (no clinical events)
        3. At each bin boundary, extracts events in ``(last_time, current_time]``
           from the pre-loaded data and passes them as ``new_data`` to
           :meth:`~PatientContext.refresh` — mimicking how an external SQL ETL
           pipeline would deliver genuinely new measurements at each poll

        This benchmarks the real-time code path (``refresh(t, new_data=...)``)
        against the retrospective simulation path (``refresh(t)`` with
        ``_full_trajectory_data``).

        Args:
            cpr_hash: Patient identifier hash.
            service_date: Admission date (for base_df lookup).
            cfg: Configuration dict (loaded from defaults.yaml if None).
            data_dir: Path to raw CSV data.
            ebm_models_dir: Path to saved EBM models.
            start_hours: Start simulation at this many hours after admission.
            end_hours: Stop simulation at this many hours (None = full trajectory).
            patient_dir: Directory with pre-split per-patient CSVs.

        Returns:
            :class:`SimulationResult` with per-step predictions and timing.
        """
        from astra.inference.patient_context import PatientContext
        from astra.inference.data_prep import (
            _build_single_patient_base_df,
            _filter_concepts_for_patient,
            _filtered_dfs_to_raw_data,
            _RAW_DATA_META_KEYS,
            timed_stage,
        )

        wall_start = time.perf_counter()

        if cfg is None:
            from astra.utils import get_cfg
            cfg = get_cfg()

        setup_timing = {}

        # Phase 1-3: Load CSV data (identical to from_csv)
        with timed_stage(setup_timing, 'csv_load'):
            base_df = _build_single_patient_base_df(
                cpr_hash, service_date, cfg, data_dir,
                patient_dir=patient_dir,
            )
            filtered_concepts = _filter_concepts_for_patient(base_df, cfg, data_dir,
                                                             patient_dir=patient_dir)

        admission_time = pd.Timestamp(base_df['start'].iloc[0])
        current_time = admission_time + pd.Timedelta(hours=start_hours)

        # Clamp to patient end
        patient_end = base_df['end'].iloc[0]
        if pd.notna(patient_end):
            current_time = min(current_time, pd.Timestamp(patient_end))

        # Build full raw_data (unfiltered by time) for slicing
        raw_data_full = _filtered_dfs_to_raw_data(
            base_df, filtered_concepts, current_time=current_time,
            cfg=cfg, filter_by_time=False,
        )

        # Create initial context with only demographics (no clinical events)
        initial_raw_data = {
            'pid': raw_data_full.get('pid'),
            'admission_time': raw_data_full['admission_time'],
            'current_time': current_time,
            'demographics': raw_data_full.get('demographics', {}),
        }
        for key, items in raw_data_full.items():
            if key in _RAW_DATA_META_KEYS or not isinstance(items, list):
                continue
            initial_raw_data[key] = []

        ctx = PatientContext.create(initial_raw_data, self.session.bundle)
        ctx.patient_end_time = (
            pd.Timestamp(patient_end) if pd.notna(patient_end) else None
        )
        for k, v in setup_timing.items():
            ctx._timing.setdefault(k, []).extend(v)

        # EBM setup: identical to from_csv() — bolt on EBM context so that
        # refresh() can incrementally compute EBM predictions at each step.
        bundle = self.session.bundle
        if '_ebm_pred' in bundle.get('ts_channel_names', []):
            from astra.inference.ebm import (
                compute_ebm_predictions, inject_ebm_into_x_ts,
            )

            with timed_stage(ctx._timing, 'ebm_compute'):
                ebm_preds = compute_ebm_predictions(
                    initial_raw_data, filtered_concepts, base_df,
                    cfg, ebm_models_dir,
                )
            with timed_stage(ctx._timing, 'ebm_inject'):
                ctx.x_ts = inject_ebm_into_x_ts(
                    ctx.x_ts, ebm_preds, ctx.bin_df,
                    ctx.admission_time, bundle,
                    trajectory_length=ctx.trajectory_length,
                )

            ctx._ebm_context = {
                'filtered_concepts': filtered_concepts,
                'base_df': base_df,
                'cfg': cfg,
                'ebm_models_dir': ebm_models_dir,
            }
            ctx._ebm_cache = dict(ebm_preds)

        # Generate bin-aligned time points (same grid as run())
        time_points = _generate_bin_aligned_times(
            ctx.bin_df, ctx.admission_time,
            start_time=current_time, end_hours=end_hours,
        )

        # Step through, delivering data as an ETL would
        steps = []
        prev_raw_counts = _count_raw_data(ctx._raw_data)
        prediction_curve = np.full(len(ctx.bin_df), np.nan)
        last_delivery_time = current_time

        for tp in time_points:
            step_timing = {}

            # Extract events in (last_delivery_time, tp] — mimics ETL poll
            with timed_stage(step_timing, 'data_slice'):
                new_data = _extract_events_in_window(
                    raw_data_full, last_delivery_time, tp,
                    _RAW_DATA_META_KEYS,
                )

            # Refresh with genuinely new data (real-time code path)
            with timed_stage(step_timing, 'refresh'):
                ctx.refresh(tp, new_data=new_data)

            new_counts = _count_raw_data(ctx._raw_data)
            n_new = new_counts - prev_raw_counts
            prev_raw_counts = new_counts

            with timed_stage(step_timing, 'predict'):
                result = self.session.predict_from_context(ctx)

            bin_idx = ctx.trajectory_length - 1
            if 0 <= bin_idx < len(prediction_curve):
                prediction_curve[bin_idx] = result.probability

            elapsed = (tp - ctx.admission_time).total_seconds() / 3600

            steps.append(SimulationStep(
                current_time=tp,
                elapsed_hours=elapsed,
                trajectory_length=ctx.trajectory_length,
                probability=result.probability,
                predictions_over_time=result.predictions_over_time,
                survival_curve=getattr(result, 'survival_curve', None),
                step_timing=step_timing,
                n_new_measurements=n_new,
            ))

            last_delivery_time = tp

        # Build result
        ihs_hours = _get_inhospital_start_hours(ctx)

        sim_result = SimulationResult(
            pid=ctx.pid,
            admission_time=ctx.admission_time,
            steps=steps,
            total_timing=dict(ctx._timing),
            wall_clock_seconds=time.perf_counter() - wall_start,
            prediction_curve=prediction_curve,
            inhospital_start_hours=ihs_hours,
        )

        # Store state for inspect() / .result
        self.context = ctx
        self._steps = list(steps)
        self._prediction_curve = prediction_curve.copy()
        self.session.ctx = ctx

        logger.info(
            f"Real-time simulation complete: {len(steps)} steps, "
            f"{sim_result.wall_clock_seconds:.1f}s wall clock, pid={ctx.pid}"
        )

        return sim_result

    # ---- Batch API (unchanged) ----

    def run(
        self,
        cpr_hash: str,
        service_date,
        cfg: dict = None,
        data_dir: str = 'data/raw',
        ebm_models_dir: str = 'models/ebm',
        start_hours: float = 0.0,
        end_hours: Optional[float] = None,
        patient_dir: str = 'data/patients',
    ) -> SimulationResult:
        """Run full simulation for a patient loaded from CSV.

        Creates a :class:`PatientContext` at admission time (or
        *start_hours* after admission), then steps through every bin
        boundary up to *end_hours* (default: full 30-day trajectory).

        Args:
            cpr_hash: Patient identifier hash.
            service_date: Admission date (for base_df lookup).
            cfg: Configuration dict (loaded from defaults.yaml if None).
            data_dir: Path to raw CSV data.
            ebm_models_dir: Path to saved EBM models.
            start_hours: Start simulation at this many hours after admission.
            end_hours: Stop simulation at this many hours (None = full trajectory).
            patient_dir: Directory with pre-split per-patient CSVs.

        Returns:
            :class:`SimulationResult` with per-step predictions and timing.
        """
        from astra.inference.patient_context import PatientContext

        wall_start = time.perf_counter()

        # Use from_csv which stores _full_trajectory_data for simulation.
        # Pass start_hours so from_csv derives current_time from the actual
        # admission_time (base_df['start']), avoiding mismatch when
        # service_date differs (e.g. prehospital shifts start earlier).
        ctx = PatientContext.from_csv(
            cpr_hash=cpr_hash,
            service_date=service_date,
            current_time=pd.Timestamp(service_date),
            bundle=self.session.bundle,
            cfg=cfg,
            data_dir=data_dir,
            ebm_models_dir=ebm_models_dir,
            start_hours=start_hours,
            patient_dir=patient_dir,
        )

        result = self.run_from_context(ctx, end_hours=end_hours)
        result.wall_clock_seconds = time.perf_counter() - wall_start

        # Store state so inspect() and .result work after run()
        self.context = ctx
        self._steps = list(result.steps)
        self._prediction_curve = result.prediction_curve.copy() if result.prediction_curve is not None else None
        self.session.ctx = ctx

        return result

    def run_from_context(
        self,
        context,
        time_points: Optional[List[pd.Timestamp]] = None,
        end_hours: Optional[float] = None,
    ) -> SimulationResult:
        """Run simulation using an existing PatientContext.

        Args:
            context: A :class:`PatientContext` (should have
                ``_full_trajectory_data`` set for simulation mode).
            time_points: Explicit list of timestamps to evaluate at.
                If None, uses bin-aligned boundaries from ``context.bin_df``.
            end_hours: Stop at this many hours after admission (only used
                when *time_points* is None).
        """
        from astra.inference.data_prep import timed_stage

        wall_start = time.perf_counter()

        if time_points is None:
            time_points = _generate_bin_aligned_times(
                context.bin_df,
                context.admission_time,
                start_time=context.current_time,
                end_hours=end_hours,
            )

        steps = []
        prev_raw_counts = _count_raw_data(context._raw_data)
        prediction_curve = np.full(len(context.bin_df), np.nan)
        profiling = {}  # fine-grained sub-stage timing

        for tp in time_points:
            step_timing = {}

            # Refresh context (incremental)
            with timed_stage(step_timing, 'refresh'):
                context.refresh(tp, profiling=profiling)

            # Count new measurements
            with timed_stage(profiling, 'count_raw'):
                new_counts = _count_raw_data(context._raw_data)
            n_new = new_counts - prev_raw_counts
            prev_raw_counts = new_counts

            # Predict
            with timed_stage(step_timing, 'predict'):
                result = self.session.predict_from_context(
                    context, profiling=profiling,
                )

            # Store prediction at current bin position
            bin_idx = context.trajectory_length - 1
            if 0 <= bin_idx < len(prediction_curve):
                prediction_curve[bin_idx] = result.probability

            elapsed = (tp - context.admission_time).total_seconds() / 3600

            steps.append(SimulationStep(
                current_time=tp,
                elapsed_hours=elapsed,
                trajectory_length=context.trajectory_length,
                probability=result.probability,
                predictions_over_time=result.predictions_over_time,
                survival_curve=getattr(result, 'survival_curve', None),
                step_timing=step_timing,
                n_new_measurements=n_new,
            ))

        # Aggregate timing from context
        total_timing = dict(context._timing)

        ihs_hours = _get_inhospital_start_hours(context)

        sim_result = SimulationResult(
            pid=context.pid,
            admission_time=context.admission_time,
            steps=steps,
            total_timing=total_timing,
            wall_clock_seconds=time.perf_counter() - wall_start,
            prediction_curve=prediction_curve,
            inhospital_start_hours=ihs_hours,
            profiling=profiling,
        )

        logger.info(
            f"Simulation complete: {len(steps)} steps, "
            f"{sim_result.wall_clock_seconds:.1f}s wall clock, "
            f"pid={context.pid}"
        )
        sim_result.print_profile()

        return sim_result


# ============================================================================
# Helpers
# ============================================================================

def _get_inhospital_start_hours(context) -> Optional[float]:
    """Extract inhospital start hours from context demographics (if prehospital)."""
    ihs_time = context.demographics.get('inhospital_start')
    if ihs_time is not None:
        ihs_ts = pd.Timestamp(ihs_time)
        if pd.notna(ihs_ts):
            h = (ihs_ts - context.admission_time).total_seconds() / 3600
            if h > 0:
                return h
    return None


def _extract_events_in_window(
    raw_data_full: dict,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    meta_keys: frozenset,
) -> dict:
    """Extract events from *raw_data_full* in the time window ``(start, end]``.

    Mimics how an ETL pipeline would deliver data arriving since the last poll:

    - **Point events** (continuous or categorical): include if
      ``window_start < timestamp <= window_end``.
    - **Interval events**: include if the interval started in the window.
      Ongoing intervals (started before *window_start* but extending past it)
      are also included with clamped end, matching the simulation reveal logic
      so that newly visible bins get the interval's multi-hot encoding.
    """
    new_data: Dict[str, list] = {}

    for key, items in raw_data_full.items():
        if key in meta_keys or not isinstance(items, list) or not items:
            continue

        sample = items[0]

        if 'start' in sample:
            # Interval events (e.g. ADTHaendelser)
            entries = []
            for evt in items:
                start = pd.Timestamp(evt['start'])
                end = pd.Timestamp(evt['end'])
                if start > window_end:
                    continue
                if start > window_start:
                    # New interval: clamp end to delivery time
                    entries.append({
                        'start': evt['start'],
                        'end': min(end, window_end),
                        'value': evt['value'],
                    })
                elif end > window_start:
                    # Ongoing interval extending into new bins: deliver update
                    entries.append({
                        'start': evt['start'],
                        'end': min(end, window_end),
                        'value': evt['value'],
                    })
            if entries:
                new_data[key] = entries

        elif 'timestamp' in sample:
            # Point events (continuous or categorical)
            entries = [
                m for m in items
                if window_start < pd.Timestamp(m['timestamp']) <= window_end
            ]
            if entries:
                new_data[key] = entries

    return new_data


def _generate_bin_aligned_times(
    bin_df: pd.DataFrame,
    admission_time: pd.Timestamp,
    start_time: Optional[pd.Timestamp] = None,
    end_hours: Optional[float] = None,
) -> List[pd.Timestamp]:
    """Generate evaluation timestamps aligned to bin boundaries.

    Uses ``bin_df['bin_end']`` values so each step crosses into the next
    bin, ensuring the model output differs from the previous step.
    """
    # Use bin_end as the time point (the bin is fully observed)
    times = bin_df['bin_end'].tolist()

    if start_time is not None:
        times = [t for t in times if t > start_time]

    if end_hours is not None:
        cutoff = admission_time + pd.Timedelta(hours=end_hours)
        times = [t for t in times if t <= cutoff]

    return sorted(times)


def _count_raw_data(raw_data: dict) -> int:
    """Total record count across all event types in raw_data."""
    from astra.inference.data_prep import _RAW_DATA_META_KEYS
    total = 0
    for key, items in raw_data.items():
        if key in _RAW_DATA_META_KEYS or not isinstance(items, list):
            continue
        total += len(items)
    return total
