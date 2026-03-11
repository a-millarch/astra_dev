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

    @property
    def n_steps(self) -> int:
        return len(self.steps)

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
        ax.set_ylabel('P(deceased 30d)')
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
        """
        from astra.inference.patient_context import PatientContext

        admission_start = pd.Timestamp(service_date)

        self.context = PatientContext.from_csv(
            cpr_hash=cpr_hash,
            service_date=service_date,
            current_time=admission_start + pd.Timedelta(hours=start_hours),
            bundle=self.session.bundle,
            cfg=cfg,
            data_dir=data_dir,
            ebm_models_dir=ebm_models_dir,
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

        from astra.inference.run_inference import default_session_plot
        self.session.ctx = self.context
        default_session_plot(self.session, prediction_curve=self._prediction_curve)

    @property
    def result(self) -> Optional[SimulationResult]:
        """Build a SimulationResult from steps accumulated so far."""
        if not self._steps or self.context is None:
            return None
        # Compute inhospital start hours for prehospital boundary
        ihs_hours = None
        ihs_time = self.context.demographics.get('inhospital_start')
        if ihs_time is not None:
            ihs_ts = pd.Timestamp(ihs_time)
            if pd.notna(ihs_ts):
                h = (ihs_ts - self.context.admission_time).total_seconds() / 3600
                if h > 0:
                    ihs_hours = h
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

        Returns:
            :class:`SimulationResult` with per-step predictions and timing.
        """
        from astra.inference.patient_context import PatientContext

        wall_start = time.perf_counter()

        # Create context at admission time — loads full trajectory
        admission_start = pd.Timestamp(service_date)

        # Use from_csv which stores _full_trajectory_data for simulation
        ctx = PatientContext.from_csv(
            cpr_hash=cpr_hash,
            service_date=service_date,
            current_time=admission_start + pd.Timedelta(hours=start_hours),
            bundle=self.session.bundle,
            cfg=cfg,
            data_dir=data_dir,
            ebm_models_dir=ebm_models_dir,
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

        for tp in time_points:
            step_timing = {}

            # Refresh context (incremental)
            with timed_stage(step_timing, 'refresh'):
                context.refresh(tp)

            # Count new measurements
            new_counts = _count_raw_data(context._raw_data)
            n_new = new_counts - prev_raw_counts
            prev_raw_counts = new_counts

            # Predict
            with timed_stage(step_timing, 'predict'):
                result = self.session.predict_from_context(context)

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
                step_timing=step_timing,
                n_new_measurements=n_new,
            ))

        # Aggregate timing from context
        total_timing = dict(context._timing)

        # Compute inhospital start hours for prehospital boundary plotting
        ihs_hours = None
        ihs_time = context.demographics.get('inhospital_start')
        if ihs_time is not None:
            ihs_ts = pd.Timestamp(ihs_time)
            if pd.notna(ihs_ts):
                h = (ihs_ts - context.admission_time).total_seconds() / 3600
                if h > 0:
                    ihs_hours = h

        sim_result = SimulationResult(
            pid=context.pid,
            admission_time=context.admission_time,
            steps=steps,
            total_timing=total_timing,
            wall_clock_seconds=time.perf_counter() - wall_start,
            prediction_curve=prediction_curve,
            inhospital_start_hours=ihs_hours,
        )

        logger.info(
            f"Simulation complete: {len(steps)} steps, "
            f"{sim_result.wall_clock_seconds:.1f}s wall clock, "
            f"pid={context.pid}"
        )

        return sim_result


# ============================================================================
# Helpers
# ============================================================================

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
    total = 0
    for key in ('vitals', 'labs', 'icu', 'medications', 'procedures', 'adt'):
        total += len(raw_data.get(key, []))
    return total
