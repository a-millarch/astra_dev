"""
ASTRA Patient SHAP Dashboard

Run from repo root:
    streamlit run dashboard/app_shap.py

Requires: pip install streamlit plotly
"""

import sys
import os
import logging

# ── Ensure repo root is on path ──────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS — avoid circular import by not going through run_inference
# ═════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_astra_modules():
    """Import heavy astra modules once, lazily."""
    from astra.utils import get_cfg, get_base_df
    from astra.inference import InferenceSession
    from astra.inference.patient_context import PatientContext
    from astra.inference.simulation import SimulationRunner
    from astra.evaluation.behavior import (
        plot_continuous_ts_shap_plotly,
        plot_categorical_ts_shap_plotly,
        plot_ebm_contributions_plotly,
        plot_prediction_trajectory_plotly,
        plot_data_completeness_plotly,
        plot_shap_budget_plotly,
        plot_shap_temporal_plotly,
        plot_top_channels_plotly,
        plot_static_features_plotly,
        plot_unified_shap_heatmap_plotly,
        plot_delta_shap_temporal_plotly,
        visualize_data_completeness,
    )
    from astra.visualize.inference import plot_prediction_trajectory
    from astra.evaluation.utils import time_to_step

    return {
        "get_cfg": get_cfg,
        "get_base_df": get_base_df,
        "InferenceSession": InferenceSession,
        "PatientContext": PatientContext,
        "SimulationRunner": SimulationRunner,
        "plot_continuous_ts_shap_plotly": plot_continuous_ts_shap_plotly,
        "plot_categorical_ts_shap_plotly": plot_categorical_ts_shap_plotly,
        "plot_ebm_contributions_plotly": plot_ebm_contributions_plotly,
        "plot_prediction_trajectory_plotly": plot_prediction_trajectory_plotly,
        "plot_data_completeness_plotly": plot_data_completeness_plotly,
        "plot_shap_budget_plotly": plot_shap_budget_plotly,
        "plot_shap_temporal_plotly": plot_shap_temporal_plotly,
        "plot_top_channels_plotly": plot_top_channels_plotly,
        "plot_static_features_plotly": plot_static_features_plotly,
        "plot_unified_shap_heatmap_plotly": plot_unified_shap_heatmap_plotly,
        "plot_delta_shap_temporal_plotly": plot_delta_shap_temporal_plotly,
        "plot_prediction_trajectory": plot_prediction_trajectory,
        "visualize_data_completeness": visualize_data_completeness,
        "time_to_step": time_to_step,
    }


# ═════════════════════════════════════════════════════════════════════════
# CONFIG & CACHED RESOURCES
# ═════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ASTRA SHAP Dashboard",
    page_icon="🔬",
    layout="wide",
)


@st.cache_data
def list_config_files():
    """Find all YAML config files in configs/."""
    configs_dir = os.path.join(REPO_ROOT, "configs")
    if not os.path.isdir(configs_dir):
        return ["configs/defaults.yaml"]
    files = []
    for f in sorted(os.listdir(configs_dir)):
        if f.endswith((".yaml", ".yml")):
            files.append(f"configs/{f}")
    return files if files else ["configs/defaults.yaml"]


@st.cache_resource
def load_config(config_path):
    mods = _load_astra_modules()
    cfg = mods["get_cfg"](config_path)
    return cfg


@st.cache_resource
def load_base_df():
    mods = _load_astra_modules()
    return mods["get_base_df"]()


@st.cache_resource
def load_session(model_name):
    """Load InferenceSession once (cached across reruns)."""
    mods = _load_astra_modules()
    session = mods["InferenceSession"].load(
        model_name=model_name,
        bundle_dir="models/deployment",
    )
    return session


def get_patient_info(base_df, cohort_pid):
    """Extract CPR, ServiceDate, start from base_df for a given PID."""
    row = base_df[base_df.PID == cohort_pid]
    if row.empty:
        return None, None, None
    cpr = row["CPR_hash"].values[0]
    sd = row["ServiceDate"].values[0]
    start = row["start"].values[0]
    return cpr, sd, start


@st.cache_data
def get_pid_list(_base_df):
    """Get sorted list of available PIDs."""
    return sorted(_base_df["PID"].unique().tolist())


# ═════════════════════════════════════════════════════════════════════════
# SIMULATION-BASED INFERENCE
# ═════════════════════════════════════════════════════════════════════════

def _get_or_create_runner(cfg, cpr, sd, actual_start, hours_offset):
    """
    Get cached SimulationRunner or create/re-create as needed.

    - Patient/model change -> fresh setup + advance
    - Time increased -> incremental advance_to (fast)
    - Time already covered -> no-op (lookup from stored steps)
    """
    mods = _load_astra_modules()
    session = load_session(cfg["model_name"])

    patient_key = f"{cpr}_{sd}_{cfg['model_name']}"
    prev_key = st.session_state.get("runner_key")
    runner = st.session_state.get("runner")

    need_setup = runner is None or prev_key != patient_key

    if need_setup:
        runner = mods["SimulationRunner"](session)
        runner.setup(
            cpr_hash=cpr,
            service_date=sd,
            data_dir="data/raw",
        )
        st.session_state["runner_key"] = patient_key
        # Invalidate stale SHAP when patient changes
        st.session_state.pop("shap_data", None)
        st.session_state.pop("shap_hours", None)
        st.session_state.pop("diff_shap_data", None)

    # Only advance if target is beyond what we've already simulated
    runner.advance_to(hours=hours_offset)

    st.session_state["runner"] = runner

    return session, runner


def _lookup_step_at_hours(sim_result, hours_offset):
    """Find the simulation step closest to (but not exceeding) hours_offset."""
    if sim_result is None or not sim_result.steps:
        return None
    best = None
    for step in sim_result.steps:
        if step.elapsed_hours <= hours_offset + 0.01:
            best = step
        else:
            break
    return best


def run_simulation_predict(cfg, cpr, sd, actual_start, hours_offset, progress_bar=None):
    """
    Setup/advance SimulationRunner to target time. Fast (no SHAP).
    Returns prediction result and simulation state.
    """
    if progress_bar:
        progress_bar.progress(10, text="Setting up simulation...")

    session, runner = _get_or_create_runner(cfg, cpr, sd, actual_start, hours_offset)

    if progress_bar:
        progress_bar.progress(70, text="Running prediction...")
    result = session.predict_from_context(runner.context)

    if progress_bar:
        progress_bar.progress(100, text="Done.")

    return {
        "session": session,
        "runner": runner,
        "result": result,
        "sim_result": runner.result,
        "ctx": runner.context,
    }


def run_shap_explanation(session, runner, progress_bar=None):
    """
    Compute SHAP for the current simulation state. Expensive.
    """
    ctx = runner.context

    if progress_bar:
        progress_bar.progress(10, text="Computing SHAP values (this takes a moment)...")
    shap_result = session.explain_from_context(ctx)

    if progress_bar:
        progress_bar.progress(60, text="Building visualization dict...")
    shap_dict, ch2feat, cat_names, cont_names = session.shap_to_viz_dict(
        shap_result, ctx.x_ts, ctx.x_ts_cat, ctx.tab_df
    )

    ebm_explanations = None
    if "_ebm_pred" in session.bundle.get("ts_channel_names", []):
        if progress_bar:
            progress_bar.progress(80, text="Computing EBM explanations...")
        ebm_explanations = session.explain_ebm(ctx, save_path=None)

    if progress_bar:
        progress_bar.progress(100, text="SHAP complete.")

    return {
        "shap_dict": shap_dict,
        "channel2feature": ch2feat,
        "feature_names_cat": cat_names,
        "feature_names_cont": cont_names,
        "ebm_explanations": ebm_explanations,
    }


def run_differential_shap(session, runner, t1_hours, t2_hours, progress_bar=None):
    """
    Compute differential SHAP between T1 and T2. Expensive (2x SHAP).
    """
    # Ensure runner is advanced to at least T2
    runner.advance_to(hours=t2_hours)
    ctx = runner.context

    if progress_bar:
        progress_bar.progress(5, text=f"Computing SHAP at T1={t1_hours:.1f}h...")

    diff_result = session.explain_differential(ctx, t1_hours, t2_hours)

    if progress_bar:
        progress_bar.progress(85, text="Building visualization dict...")

    diff_dict, ch2feat, cat_names, cont_names = session.differential_shap_to_viz_dict(
        diff_result, ctx.x_ts, ctx.x_ts_cat, ctx.tab_df
    )

    if progress_bar:
        progress_bar.progress(100, text="Differential SHAP complete.")

    return {
        "diff_result": diff_result,
        "shap_dict": diff_dict,
        "channel2feature": ch2feat,
        "feature_names_cat": cat_names,
        "feature_names_cont": cont_names,
    }


# ═════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═════════════════════════════════════════════════════════════════════════

_TRAJECTORY_TICK_HOURS = [0, 1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 504, 672]

TAB_NAMES = ["SHAP Heatmaps", "SHAP Overview", "Differential SHAP", "Data Completeness"]


def _hours_to_label(hours):
    """Format hours to human-readable string."""
    if hours < 1:
        return f"{hours * 60:.0f}min"
    elif hours < 24:
        return f"{int(hours)}h" if hours == int(hours) else f"{hours:.1f}h"
    else:
        d = hours / 24
        return f"{int(d)}D" if d == int(d) else f"{d:.1f}D"


def _plot_simulation_trajectory(sim_result, shap_hours=None, viewed_hours=None,
                                diff_data=None):
    """Build Plotly prediction trajectory from SimulationResult."""
    if sim_result is None or not sim_result.steps:
        return None

    df = sim_result.to_dataframe()

    # Filter to observation window
    if viewed_hours is not None:
        df = df[df["elapsed_hours"] <= viewed_hours + 0.01].copy()
    if df.empty:
        return None

    fig = go.Figure()

    # Prediction curve
    fig.add_trace(go.Scatter(
        x=df["elapsed_hours"],
        y=df["probability"],
        mode="lines+markers",
        marker=dict(size=3),
        line=dict(color="#1f77b4", width=2),
        name="P(deceased 30d)",
    ))

    # Inhospital arrival boundary
    if sim_result.inhospital_start_hours is not None and sim_result.inhospital_start_hours > 0:
        if viewed_hours is None or sim_result.inhospital_start_hours <= viewed_hours:
            fig.add_vline(
                x=sim_result.inhospital_start_hours,
                line_dash="dot",
                line_color="#2196F3",
                annotation_text=f"Hospital arrival ({sim_result.inhospital_start_hours:.1f}h)",
                annotation_position="top left",
            )

    # Mark SHAP evaluation timepoint
    if shap_hours is not None and (viewed_hours is None or shap_hours <= viewed_hours + 0.01):
        closest_idx = (df["elapsed_hours"] - shap_hours).abs().idxmin()
        shap_prob = df.loc[closest_idx, "probability"]
        fig.add_trace(go.Scatter(
            x=[shap_hours],
            y=[shap_prob],
            mode="markers",
            marker=dict(size=12, color="#0068C9", symbol="diamond"),
            name=f"SHAP eval ({shap_hours:.1f}h)",
            showlegend=True,
        ))

    # Mark differential SHAP interval [T1, T2]
    if diff_data is not None:
        dr = diff_data["diff_result"]
        t1h, t2h = dr.t1_hours, dr.t2_hours
        # Shaded region
        fig.add_vrect(
            x0=t1h, x1=t2h,
            fillcolor="rgba(255, 165, 0, 0.12)", line_width=0,
            annotation_text="ΔSHAP window",
            annotation_position="top left",
        )
        # T1 and T2 markers
        for th, prob, label, symbol in [
            (t1h, dr.t1_probability, f"T1 ({t1h:.1f}h)", "triangle-left"),
            (t2h, dr.t2_probability, f"T2 ({t2h:.1f}h)", "triangle-right"),
        ]:
            if viewed_hours is None or th <= viewed_hours + 0.01:
                fig.add_trace(go.Scatter(
                    x=[th], y=[prob], mode="markers",
                    marker=dict(size=11, color="#FF8C00", symbol=symbol),
                    name=label, showlegend=True,
                ))

    # Build meaningful x-axis tick labels (Change D)
    max_h = df["elapsed_hours"].iloc[-1]
    tick_hours = [h for h in _TRAJECTORY_TICK_HOURS if h <= max_h]
    if not tick_hours or (max_h - tick_hours[-1]) > max_h * 0.05:
        tick_hours.append(max_h)
    tick_labels = [_hours_to_label(h) for h in tick_hours]

    # Clamp x-axis to start at the first data point (avoid phantom hover at 0.0)
    min_h = df["elapsed_hours"].iloc[0]
    n_visible = len(df)
    fig.update_layout(
        title=f"Prediction Trajectory ({n_visible} steps up to {df['elapsed_hours'].iloc[-1]:.1f}h)",
        xaxis_title="Time since admission",
        yaxis_title="P(deceased 30d)",
        yaxis=dict(range=[-0.05, 1.05]),
        xaxis=dict(
            tickvals=tick_hours, ticktext=tick_labels, tickangle=0,
            range=[max(0, min_h - 0.1), max_h + max_h * 0.02],
        ),
        hovermode="x",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def _plot_simulation_diagnostics(sim_result, viewed_hours=None):
    """Build diagnostic Plotly charts from SimulationResult."""
    if sim_result is None or not sim_result.steps:
        return None, None

    df = sim_result.to_dataframe()

    # Filter to observation window
    if viewed_hours is not None:
        df = df[df["elapsed_hours"] <= viewed_hours + 0.01].copy()
    if df.empty:
        return None, None

    # Timing breakdown (stacked bar)
    timing_cols = [c for c in df.columns if c.startswith("timing_")]
    fig_timing = None
    if timing_cols:
        fig_timing = go.Figure()
        for col in timing_cols:
            label = col.replace("timing_", "").replace("_ms", "")
            fig_timing.add_trace(go.Bar(
                x=df["elapsed_hours"],
                y=df[col].fillna(0),
                name=label,
            ))
        fig_timing.update_layout(
            barmode="stack",
            title="Per-step Timing Breakdown",
            xaxis_title="Elapsed hours",
            yaxis_title="Time (ms)",
            height=250,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

    # Measurement arrival
    fig_meas = go.Figure()
    fig_meas.add_trace(go.Bar(
        x=df["elapsed_hours"],
        y=df["n_new_measurements"],
        marker_color="#2ca02c",
        name="New measurements",
    ))
    fig_meas.update_layout(
        title="New Measurements per Step",
        xaxis_title="Elapsed hours",
        yaxis_title="Count",
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig_timing, fig_meas


# ═════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════

def main():

    # ── Read URL query params for state persistence (Change F) ───────
    qp = st.query_params
    qp_config = qp.get("config", None)
    qp_pid = qp.get("pid", None)
    qp_hours = qp.get("hours", None)
    qp_diff_t1 = qp.get("diff_t1", None)
    qp_diff_t2 = qp.get("diff_t2", None)
    qp_tab = qp.get("tab", None)

    # ── Sidebar: Config selection ─────────────────────────────────────
    st.sidebar.header("Configuration")
    config_files = list_config_files()

    if qp_config and qp_config in config_files:
        default_config_idx = config_files.index(qp_config)
    else:
        default_config_idx = next(
            (i for i, f in enumerate(config_files) if f.endswith("defaults.yaml")), 0
        )
    config_path = st.sidebar.selectbox(
        "Config file",
        options=config_files,
        index=default_config_idx,
        help="YAML config from configs/ folder",
    )

    # ── Load config & base data (fast, cached) ───────────────────────
    with st.spinner("Loading config & base data..."):
        cfg = load_config(config_path)
        base_df = load_base_df()
    pid_list = get_pid_list(base_df)

    st.sidebar.markdown(f"**Model:** `{cfg.get('model_name', 'unknown')}`")
    st.sidebar.markdown("---")

    # ── Sidebar: Patient selection ────────────────────────────────────
    st.sidebar.header("Patient Selection")

    pid_default_idx = 0
    if qp_pid is not None:
        try:
            pid_val = int(qp_pid)
            if pid_val in pid_list:
                pid_default_idx = pid_list.index(pid_val)
        except (ValueError, TypeError):
            pass

    cohort_pid = st.sidebar.selectbox(
        "Patient ID (PID)",
        options=pid_list,
        index=pid_default_idx,
    )

    cpr, sd, actual_start = get_patient_info(base_df, cohort_pid)

    if cpr is None:
        st.error(f"PID {cohort_pid} not found in base_df")
        return

    # Compute full trajectory length for this patient
    row = base_df[base_df.PID == cohort_pid].iloc[0]
    patient_start = pd.Timestamp(row["start"])
    patient_end = pd.Timestamp(row.get("end", row.get("stop", patient_start + pd.Timedelta(hours=168))))
    max_hours = max(1.0, (patient_end - patient_start).total_seconds() / 3600)
    max_hours = min(max_hours, 720.0)  # cap at 30 days

    # Time offset
    st.sidebar.markdown("---")
    st.sidebar.subheader("Observation Time")

    initial_hours = float(max_hours)
    if qp_hours is not None:
        try:
            initial_hours = max(0.5, min(float(qp_hours), float(max_hours)))
        except (ValueError, TypeError):
            pass

    hours_offset = st.sidebar.slider(
        "Hours after admission",
        min_value=0.5,
        max_value=float(max_hours),
        value=initial_hours,
        step=0.5,
        key="slider_hours",
        help="How many hours of data to include from admission",
    )

    st.sidebar.markdown(f"""
    **Patient Info:**
    - PID: `{cohort_pid}`
    - Service Date: `{sd}`
    - Observation: `{hours_offset}h` after admission
    """)

    # ── Sidebar: Action buttons (no Run Simulation - Change A) ───────
    st.sidebar.markdown("---")
    with st.sidebar.container(key="shap_container"):
        shap_clicked = st.button(
            "Compute SHAP",
            type="primary",
            use_container_width=True,
            help="Compute SHAP explanations at the current time point",
        )

    # ── Sidebar: Differential SHAP ──────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Differential SHAP")
    diff_col1, diff_col2 = st.sidebar.columns(2)

    diff_t1_default = max(0.0, hours_offset - 2.0)
    if qp_diff_t1 is not None:
        try:
            diff_t1_default = max(0.0, min(float(qp_diff_t1), float(max_hours)))
        except (ValueError, TypeError):
            pass

    diff_t2_default = float(hours_offset)
    if qp_diff_t2 is not None:
        try:
            diff_t2_default = max(0.0, min(float(qp_diff_t2), float(max_hours)))
        except (ValueError, TypeError):
            pass

    with diff_col1:
        diff_t1 = st.number_input(
            "T1 (hours)", min_value=0.0, max_value=float(max_hours),
            value=diff_t1_default, step=0.5, key="diff_t1_input",
        )
    with diff_col2:
        diff_t2 = st.number_input(
            "T2 (hours)", min_value=0.0, max_value=float(max_hours),
            value=diff_t2_default, step=0.5, key="diff_t2_input",
        )
    diff_clicked = st.sidebar.button(
        "Compute Differential SHAP",
        type="primary",
        use_container_width=True,
        help="Compute ΔSHAP = SHAP(T2) - SHAP(T1) to see what drives the change",
    )

    # ── Update URL query params (Change F) ───────────────────────────
    st.query_params["config"] = config_path
    st.query_params["pid"] = str(cohort_pid)
    st.query_params["hours"] = str(hours_offset)
    st.query_params["diff_t1"] = str(diff_t1)
    st.query_params["diff_t2"] = str(diff_t2)

    # ═════════════════════════════════════════════════════════════════
    # SESSION INITIALIZATION GATE (Change B)
    # ═════════════════════════════════════════════════════════════════

    session_initialized = st.session_state.get("session_initialized", False)
    patient_key = f"{cpr}_{sd}_{cfg.get('model_name')}"

    if not session_initialized:
        if shap_clicked or diff_clicked:
            st.session_state["session_initialized"] = True
            session_initialized = True
        else:
            st.info(
                "Select a **config**, **patient**, and **observation time** in the sidebar, "
                "then click **Compute SHAP** or **Compute Differential SHAP** to begin."
            )
            return

    # ═════════════════════════════════════════════════════════════════
    # PREDICT: auto-run on every rerun (Change A — no Run Sim button)
    # ═════════════════════════════════════════════════════════════════

    # Invalidate cached results if the patient changed
    if st.session_state.get("pred_patient_key") != patient_key:
        st.session_state.pop("pred_data", None)
        st.session_state.pop("shap_data", None)
        st.session_state.pop("shap_hours", None)
        st.session_state.pop("diff_shap_data", None)

    try:
        pred_data = run_simulation_predict(
            cfg, cpr, sd, actual_start, hours_offset
        )
        st.session_state["pred_data"] = pred_data
        st.session_state["pred_patient_key"] = patient_key
        st.session_state["cfg_used"] = cfg
    except Exception as e:
        st.error(f"Error running simulation: {e}")
        st.exception(e)
        return

    # Look up the step at the current slider position from stored results
    viewed_step = _lookup_step_at_hours(pred_data["sim_result"], hours_offset)
    st.session_state["viewed_hours"] = hours_offset

    # ═════════════════════════════════════════════════════════════════
    # SHAP: only on button click
    # ═════════════════════════════════════════════════════════════════

    if shap_clicked:
        try:
            progress = st.progress(0, text="Computing SHAP...")
            shap_data = run_shap_explanation(
                pred_data["session"], pred_data["runner"], progress_bar=progress
            )
            progress.empty()
            st.session_state["shap_data"] = shap_data
            st.session_state["shap_hours"] = hours_offset
        except Exception as e:
            st.error(f"Error computing SHAP: {e}")
            st.exception(e)

    # ═════════════════════════════════════════════════════════════════
    # DIFFERENTIAL SHAP: only on button click
    # ═════════════════════════════════════════════════════════════════

    if diff_clicked:
        try:
            progress = st.progress(0, text="Computing Differential SHAP (2x SHAP)...")
            diff_data = run_differential_shap(
                pred_data["session"], pred_data["runner"],
                diff_t1, diff_t2, progress_bar=progress,
            )
            progress.empty()
            st.session_state["diff_shap_data"] = diff_data
            # Auto-switch to Differential SHAP tab (Change H)
            st.session_state["active_tab"] = "Differential SHAP"
        except Exception as e:
            st.error(f"Error computing differential SHAP: {e}")
            st.exception(e)

    mods = _load_astra_modules()

    # ── Header metrics ────────────────────────────────────────────────
    sim_result = pred_data["sim_result"]
    # Show metrics for the slider position (viewed_step), not the simulation endpoint
    if viewed_step is not None:
        prob = viewed_step.probability
        traj_len = viewed_step.trajectory_length
    else:
        prob = pred_data["result"].probability
        traj_len = pred_data["result"].trajectory_length

    session = pred_data["session"]
    cal_label = ""
    if getattr(session, '_calibration_method', None):
        cal_label = f" ({session._calibration_method})"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"P(deceased 30d){cal_label}", f"{prob:.3f}")
    with col2:
        st.metric("Trajectory Length", f"{traj_len} steps")
    with col3:
        st.metric("Simulation Steps", f"{sim_result.n_steps}" if sim_result else "0")
    with col4:
        if viewed_step is not None:
            st.metric("Viewing Time", f"{viewed_step.elapsed_hours:.1f}h")
        elif sim_result and sim_result.steps:
            st.metric("Viewing Time", f"{sim_result.steps[-1].elapsed_hours:.1f}h")
        else:
            st.metric("Viewing Time", "N/A")

    # ── Patient Context (expandable) ──────────────────────────────────
    ctx = pred_data["ctx"]
    with st.expander("Patient Context"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
**PID:** `{ctx.pid}`
**Admission:** `{ctx.admission_time}`
**Trajectory:** {traj_len} steps
**Model:** `{cfg.get('model_name')}`
**Temporal head:** `{session.is_temporal}`
""")
        with col_b:
            if hasattr(ctx, 'tab_df') and ctx.tab_df is not None:
                st.markdown("**Static features:**")
                st.dataframe(ctx.tab_df.T.rename(columns={ctx.tab_df.index[0]: "Value"}),
                             use_container_width=True, height=200)

    # ── Prediction Trajectory (from simulation) ──────────────────────
    shap_hours = st.session_state.get("shap_hours")
    diff_shap_data = st.session_state.get("diff_shap_data")
    fig_traj = _plot_simulation_trajectory(
        sim_result, shap_hours=shap_hours, viewed_hours=hours_offset,
        diff_data=diff_shap_data,
    )
    if fig_traj:
        fig_traj.update_layout(width=None)
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.info("No trajectory data available. Move the slider to generate predictions.")

    # ── Simulation Diagnostics (expandable) ──────────────────────────
    with st.expander("Simulation Details"):
        if sim_result and sim_result.steps:
            # Summary metrics
            diag_col1, diag_col2, diag_col3 = st.columns(3)
            runner = pred_data["runner"]
            with diag_col1:
                st.metric("Total Steps", sim_result.n_steps)
            with diag_col2:
                wc = sim_result.wall_clock_seconds
                st.metric("Wall Clock", f"{wc:.2f}s" if wc > 0 else "N/A")
            with diag_col3:
                st.metric("Remaining Steps", runner.remaining_steps)

            # Diagnostic charts
            fig_timing, fig_meas = _plot_simulation_diagnostics(sim_result, viewed_hours=hours_offset)

            if fig_timing:
                fig_timing.update_layout(width=None)
                st.plotly_chart(fig_timing, use_container_width=True)

            if fig_meas:
                fig_meas.update_layout(width=None)
                st.plotly_chart(fig_meas, use_container_width=True)

            # Raw data table
            with st.expander("Step data (raw)"):
                st.dataframe(
                    sim_result.to_dataframe(),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No simulation steps recorded yet.")

    st.markdown("---")

    # ── SHAP staleness / re-slicing logic (Change G) ─────────────────
    shap_data = st.session_state.get("shap_data")
    shap_hours_cached = st.session_state.get("shap_hours")

    time_to_step = mods["time_to_step"]
    current_eval_step = time_to_step(hours_offset, 'h')

    if shap_data and shap_hours_cached is not None:
        if hours_offset > shap_hours_cached + 0.01:
            st.warning(
                f"SHAP was computed at **{shap_hours_cached:.1f}h** but current time is "
                f"**{hours_offset:.1f}h** (beyond SHAP range). Click **Compute SHAP** to update."
            )
        elif abs(hours_offset - shap_hours_cached) > 0.01:
            st.info(
                f"Showing SHAP truncated to **{hours_offset:.1f}h** "
                f"(computed at {shap_hours_cached:.1f}h)."
            )

    # ── Tab selector via st.radio (Change H — supports programmatic switch) ──
    if "active_tab" not in st.session_state:
        if qp_tab and qp_tab in TAB_NAMES:
            st.session_state["active_tab"] = qp_tab
        else:
            st.session_state["active_tab"] = TAB_NAMES[0]

    active_tab = st.radio(
        "View",
        options=TAB_NAMES,
        horizontal=True,
        key="active_tab",
        label_visibility="collapsed",
    )

    st.query_params["tab"] = active_tab

    # ── Tab: SHAP Heatmaps (unified, grouped by concept) ───────────
    if active_tab == "SHAP Heatmaps":
        if shap_data is None:
            st.info("Click **Compute SHAP** in the sidebar to generate explanations for the current time point.")
        else:
            fig_unified = mods["plot_unified_shap_heatmap_plotly"](
                shap_data["shap_dict"],
                sample_idx=0,
                channel2feature=shap_data["channel2feature"],
                channel_map=session.bundle.get('data_config', {}).get('channel_map', {}),
                eval_timestep=current_eval_step,
                title="SHAP Heatmap (all channels, grouped by concept)",
            )
            if fig_unified:
                fig_unified.update_layout(width=None)
                st.plotly_chart(fig_unified, use_container_width=True)
            else:
                st.warning("No SHAP heatmap data available")

            # EBM contributions (supplementary)
            if shap_data["ebm_explanations"]:
                with st.expander("EBM Contributions"):
                    fig_ebm = mods["plot_ebm_contributions_plotly"](
                        shap_data["ebm_explanations"],
                        height=600,
                    )
                    if fig_ebm:
                        fig_ebm.update_layout(width=None)
                        st.plotly_chart(fig_ebm, use_container_width=True)

            st.markdown("---")

            fig_channels = mods["plot_top_channels_plotly"](
                shap_data["shap_dict"], sample_idx=0,
                channel2feature=shap_data["channel2feature"],
                eval_timestep=current_eval_step,
            )
            if fig_channels:
                fig_channels.update_layout(width=None)
                st.plotly_chart(fig_channels, use_container_width=True)

    # ── Tab: SHAP Overview ────────────────────────────────────────────
    elif active_tab == "SHAP Overview":
        if shap_data is None:
            st.info("Click **Compute SHAP** in the sidebar to generate explanations.")
        else:
            fig_budget = mods["plot_shap_budget_plotly"](
                shap_data["shap_dict"], sample_idx=0,
                channel2feature=shap_data["channel2feature"],
                eval_timestep=current_eval_step,
            )
            if fig_budget:
                fig_budget.update_layout(width=None)
                st.plotly_chart(fig_budget, use_container_width=True)

            fig_temporal = mods["plot_shap_temporal_plotly"](
                shap_data["shap_dict"], sample_idx=0,
                channel2feature=shap_data["channel2feature"],
                eval_timestep=current_eval_step,
            )
            if fig_temporal:
                fig_temporal.update_layout(width=None)
                st.plotly_chart(fig_temporal, use_container_width=True)

            st.markdown("---")

            fig_static = mods["plot_static_features_plotly"](
                shap_data["shap_dict"], sample_idx=0,
                feature_names_cat=shap_data["feature_names_cat"],
                feature_names_cont=shap_data["feature_names_cont"])
            if fig_static:
                fig_static.update_layout(width=None)
                st.plotly_chart(fig_static, use_container_width=True)

    # ── Tab: Differential SHAP ───────────────────────────────────────
    elif active_tab == "Differential SHAP":
        if diff_shap_data is None:
            st.info(
                "Set **T1** and **T2** in the sidebar, then click "
                "**Compute Differential SHAP** to see what drives the "
                "prediction change between two timepoints."
            )
        else:
            dr = diff_shap_data["diff_result"]
            delta_p = dr.t2_probability - dr.t1_probability

            # Summary metrics
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("P(T1)", f"{dr.t1_probability:.3f}",
                          help=f"Prediction at T1 = {dr.t1_hours:.1f}h")
            with mc2:
                st.metric("P(T2)", f"{dr.t2_probability:.3f}",
                          help=f"Prediction at T2 = {dr.t2_hours:.1f}h")
            with mc3:
                st.metric("ΔP", f"{delta_p:+.3f}",
                          delta=f"{delta_p:+.3f}",
                          delta_color="inverse",
                          help="P(T2) - P(T1)")

            st.caption(
                f"ΔSHAP = SHAP({dr.t2_hours:.1f}h) - SHAP({dr.t1_hours:.1f}h). "
                f"Red = increased risk attribution, Blue = decreased. "
                f"Note: P(T1)/P(T2) are computed on the T2 context with censoring "
                f"and may differ slightly from the trajectory (which builds context incrementally)."
            )

            # Delta unified heatmap (all channels grouped by concept)
            fig_delta_unified = mods["plot_unified_shap_heatmap_plotly"](
                diff_shap_data["shap_dict"],
                sample_idx=0,
                channel2feature=diff_shap_data["channel2feature"],
                channel_map=session.bundle.get('data_config', {}).get('channel_map', {}),
                start_timestep=dr.t1_step,
                title=f"ΔSHAP Heatmap ({dr.t1_hours:.1f}h -> {dr.t2_hours:.1f}h)",
            )
            if fig_delta_unified:
                fig_delta_unified.update_layout(width=None)
                st.plotly_chart(fig_delta_unified, use_container_width=True)

            st.markdown("---")

            # Delta SHAP over time (continuous + categorical)
            fig_delta_temporal = mods["plot_delta_shap_temporal_plotly"](
                diff_shap_data["shap_dict"],
                sample_idx=0,
                channel2feature=diff_shap_data["channel2feature"],
                title=f"ΔSHAP Over Time ({dr.t1_hours:.1f}h -> {dr.t2_hours:.1f}h)",
            )
            if fig_delta_temporal:
                fig_delta_temporal.update_layout(width=None)
                st.plotly_chart(fig_delta_temporal, use_container_width=True)

            st.markdown("---")

            # Top changed features
            fig_delta_top = mods["plot_top_channels_plotly"](
                diff_shap_data["shap_dict"], sample_idx=0,
                channel2feature=diff_shap_data["channel2feature"],
            )
            if fig_delta_top:
                fig_delta_top.update_layout(
                    width=None,
                    title="Top Changed Channels (by |ΔSHAP|)",
                )
                st.plotly_chart(fig_delta_top, use_container_width=True)

            st.markdown("---")

            # Delta static features
            fig_delta_static = mods["plot_static_features_plotly"](
                diff_shap_data["shap_dict"], sample_idx=0,
                feature_names_cat=diff_shap_data["feature_names_cat"],
                feature_names_cont=diff_shap_data["feature_names_cont"],
            )
            if fig_delta_static:
                fig_delta_static.update_layout(
                    width=None,
                    title="ΔSHAP: Static Features",
                )
                st.plotly_chart(fig_delta_static, use_container_width=True)

    # ── Tab: Data Completeness ───────────────────────────────────────
    elif active_tab == "Data Completeness":
        # Data completeness uses shap_dict but can also work from prediction data
        completeness_source = shap_data["shap_dict"] if shap_data else None
        completeness_ch2feat = shap_data["channel2feature"] if shap_data else None

        if completeness_source is None:
            st.info("Click **Compute SHAP** in the sidebar to generate data completeness analysis.")
        else:
            # Override eval_timestep with current slider position (Change G)
            view_dict = {**completeness_source, "eval_timestep": current_eval_step}

            comp_figs = mods["plot_data_completeness_plotly"](
                view_dict,
                sample_idx=0,
                channel2feature=completeness_ch2feat,
            )

            if comp_figs:
                # Density timeline
                if comp_figs.get('density'):
                    comp_figs['density'].update_layout(width=None)
                    st.plotly_chart(comp_figs['density'], use_container_width=True)

                # Presence heatmap
                if comp_figs.get('presence'):
                    comp_figs['presence'].update_layout(width=None)
                    st.plotly_chart(comp_figs['presence'], use_container_width=True)

                # Categorical activity
                if comp_figs.get('categorical'):
                    comp_figs['categorical'].update_layout(width=None)
                    st.plotly_chart(comp_figs['categorical'], use_container_width=True)

                # Completeness bars
                if comp_figs.get('bars'):
                    comp_figs['bars'].update_layout(width=None)
                    st.plotly_chart(comp_figs['bars'], use_container_width=True)

                # Summary
                if comp_figs.get('summary'):
                    s = comp_figs['summary']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trajectory", f"{s['trajectory_steps']} steps ({s['trajectory_hours']}h)")
                    with col2:
                        st.metric("Channels", s['n_channels'])
                    with col3:
                        st.metric("Overall completeness", f"{s['overall_completeness']}%")

                    with st.expander("Completeness per channel (raw values)"):
                        comp_df = pd.DataFrame(
                            list(s['completeness'].items()),
                            columns=["Channel", "Completeness (%)"],
                        ).sort_values("Completeness (%)", ascending=False)
                        comp_df["Completeness (%)"] = (comp_df["Completeness (%)"] * 100).round(1)
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not generate completeness plots")


if __name__ == "__main__":
    main()
