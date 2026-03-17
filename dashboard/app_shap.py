"""
ASTRA Patient SHAP Dashboard

Run from repo root:
    streamlit run dashboard/app_shap.py

Requires: pip install streamlit plotly
"""

import sys
import os
import logging
import time as _time

# ── Ensure repo root is on path ──────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
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
        return ["configs/defaults_ebm.yaml"]
    files = []
    for f in sorted(os.listdir(configs_dir)):
        if f.endswith((".yaml", ".yml")):
            files.append(f"configs/{f}")
    return files if files else ["configs/defaults_ebm.yaml"]


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

    - Patient/model change → fresh setup + advance
    - Time increased → incremental advance_to (fast)
    - Time decreased → re-setup (runner is forward-only) + advance
    """
    mods = _load_astra_modules()
    session = load_session(cfg["model_name"])

    patient_key = f"{cpr}_{sd}_{cfg['model_name']}"
    prev_key = st.session_state.get("runner_key")
    prev_hours = st.session_state.get("runner_hours", 0.0)
    runner = st.session_state.get("runner")

    need_setup = (
        runner is None
        or prev_key != patient_key
        or hours_offset < prev_hours  # backward requires re-setup
    )

    if need_setup:
        runner = mods["SimulationRunner"](session)
        runner.setup(
            cpr_hash=cpr,
            service_date=sd,
            data_dir="data/raw",
        )
        st.session_state["runner_key"] = patient_key
        # Invalidate stale SHAP when patient changes or time goes backward
        st.session_state.pop("shap_data", None)
        st.session_state.pop("shap_hours", None)

    # Advance to target time (incremental if already partially there)
    runner.advance_to(hours=hours_offset)

    st.session_state["runner"] = runner
    st.session_state["runner_hours"] = hours_offset

    return session, runner


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


# ═════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═════════════════════════════════════════════════════════════════════════

def _plot_simulation_trajectory(sim_result, shap_hours=None):
    """Build Plotly prediction trajectory from SimulationResult."""
    if sim_result is None or not sim_result.steps:
        return None

    df = sim_result.to_dataframe()

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
        fig.add_vline(
            x=sim_result.inhospital_start_hours,
            line_dash="dot",
            line_color="#2196F3",
            annotation_text=f"Hospital arrival ({sim_result.inhospital_start_hours:.1f}h)",
            annotation_position="top left",
        )

    # Mark SHAP evaluation timepoint
    if shap_hours is not None:
        # Find the closest step to the SHAP evaluation time
        closest_idx = (df["elapsed_hours"] - shap_hours).abs().idxmin()
        shap_prob = df.loc[closest_idx, "probability"]
        fig.add_trace(go.Scatter(
            x=[shap_hours],
            y=[shap_prob],
            mode="markers",
            marker=dict(size=12, color="red", symbol="diamond"),
            name=f"SHAP eval ({shap_hours:.1f}h)",
            showlegend=True,
        ))

    fig.update_layout(
        title=f"Prediction Trajectory ({sim_result.n_steps} simulation steps)",
        xaxis_title="Elapsed hours",
        yaxis_title="P(deceased 30d)",
        yaxis=dict(range=[-0.05, 1.05]),
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def _plot_simulation_diagnostics(sim_result):
    """Build diagnostic Plotly charts from SimulationResult."""
    if sim_result is None or not sim_result.steps:
        return None, None

    df = sim_result.to_dataframe()

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

    # ── Sidebar: Config selection ─────────────────────────────────────
    st.sidebar.header("Configuration")
    config_files = list_config_files()
    default_idx = next(
        (i for i, f in enumerate(config_files) if "defaults_ebm" in f), 0
    )
    config_path = st.sidebar.selectbox(
        "Config file",
        options=config_files,
        index=default_idx,
        help="YAML config from configs/ folder",
    )

    # ── Load resources ────────────────────────────────────────────────
    with st.spinner("Loading config & base data..."):
        cfg = load_config(config_path)
        base_df = load_base_df()
    pid_list = get_pid_list(base_df)

    st.sidebar.markdown(f"**Model:** `{cfg.get('model_name', 'unknown')}`")
    st.sidebar.markdown("---")

    # ── Sidebar: Patient selection ────────────────────────────────────
    st.sidebar.header("Patient Selection")

    cohort_pid = st.sidebar.selectbox(
        "Patient ID (PID)",
        options=pid_list,
        index=0,
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

    hours_offset = st.sidebar.slider(
        "Hours after admission",
        min_value=0.5,
        max_value=float(max_hours),
        value=float(max_hours),
        step=0.5,
        help="How many hours of data to include from admission",
    )

    st.sidebar.markdown(f"""
    **Patient Info:**
    - PID: `{cohort_pid}`
    - Service Date: `{sd}`
    - Observation: `{hours_offset}h` after admission
    """)

    # ── Playback controls ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Playback")

    play_col1, play_col2 = st.sidebar.columns(2)
    with play_col1:
        play_clicked = st.button(
            "Play" if not st.session_state.get("play_active", False) else "Pause",
            use_container_width=True,
        )
    with play_col2:
        play_speed = st.selectbox(
            "Speed",
            options=["1 step", "5 steps", "10 steps"],
            index=0,
            label_visibility="collapsed",
        )

    if play_clicked:
        st.session_state["play_active"] = not st.session_state.get("play_active", False)

    # ── Sidebar: SHAP button ─────────────────────────────────────────
    st.sidebar.markdown("---")
    shap_clicked = st.sidebar.button(
        "Compute SHAP",
        type="primary",
        use_container_width=True,
        help="Compute SHAP explanations at the current time point",
    )

    # ── Apply playback override BEFORE prediction ─────────────────────
    # During playback, the previous rerun set play_target_hours.
    # Override hours_offset so the prediction runs at the playback time.
    if "play_target_hours" in st.session_state:
        hours_offset = st.session_state.pop("play_target_hours")

    # ═════════════════════════════════════════════════════════════════
    # AUTO-PREDICT: only when patient or time actually changed
    # ═════════════════════════════════════════════════════════════════

    pred_key = f"{cpr}_{sd}_{cfg.get('model_name')}_{hours_offset}"
    if st.session_state.get("pred_key") != pred_key or "pred_data" not in st.session_state:
        try:
            with st.spinner("Running simulation prediction..."):
                pred_data = run_simulation_predict(
                    cfg, cpr, sd, actual_start, hours_offset
                )
            st.session_state["pred_data"] = pred_data
            st.session_state["pred_key"] = pred_key
            st.session_state["cfg_used"] = cfg
        except Exception as e:
            st.error(f"Error running simulation: {e}")
            st.exception(e)
            return

    pred_data = st.session_state["pred_data"]

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
    # PLAYBACK: schedule next advance and trigger rerun
    # ═════════════════════════════════════════════════════════════════

    if st.session_state.get("play_active", False):
        runner = pred_data["runner"]
        if runner.remaining_steps > 0:
            # Determine step size from speed setting
            steps_map = {"1 step": 1, "5 steps": 5, "10 steps": 10}
            n_advance = steps_map.get(play_speed, 1)

            # Find the next bin-aligned time point(s) to advance to
            next_idx = min(
                runner._step_idx + n_advance,
                len(runner._time_points)
            )
            if next_idx > runner._step_idx and next_idx <= len(runner._time_points):
                next_tp = runner._time_points[next_idx - 1]
                next_hours = (next_tp - runner.context.admission_time).total_seconds() / 3600
                st.session_state["play_target_hours"] = next_hours
            else:
                st.session_state["play_active"] = False

            _time.sleep(0.3)
            st.rerun()
        else:
            st.session_state["play_active"] = False

    mods = _load_astra_modules()

    # ── Header metrics ────────────────────────────────────────────────
    result = pred_data["result"]
    sim_result = pred_data["sim_result"]
    prob = result.probability

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("P(deceased 30d)", f"{prob:.3f}")
    with col2:
        st.metric("Trajectory Length", f"{result.trajectory_length} steps")
    with col3:
        st.metric("Simulation Steps", f"{sim_result.n_steps}" if sim_result else "0")
    with col4:
        if sim_result and sim_result.steps:
            effective_h = sim_result.steps[-1].elapsed_hours
            st.metric("Effective Time", f"{effective_h:.1f}h")
        else:
            st.metric("Effective Time", "N/A")

    # ── Patient Context (expandable) ──────────────────────────────────
    ctx = pred_data["ctx"]
    session = pred_data["session"]
    with st.expander("Patient Context"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
**PID:** `{ctx.pid}`
**Admission:** `{ctx.admission_time}`
**Trajectory:** {result.trajectory_length} steps
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
    fig_traj = _plot_simulation_trajectory(sim_result, shap_hours=shap_hours)
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
            fig_timing, fig_meas = _plot_simulation_diagnostics(sim_result)

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

    # ── SHAP staleness warning ────────────────────────────────────────
    shap_data = st.session_state.get("shap_data")
    shap_hours_cached = st.session_state.get("shap_hours")
    if shap_data and shap_hours_cached is not None and abs(shap_hours_cached - hours_offset) > 0.01:
        st.warning(
            f"SHAP was computed at **{shap_hours_cached:.1f}h** but current time is "
            f"**{hours_offset:.1f}h**. Click **Compute SHAP** to update."
        )

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_shap, tab_overview, tab_comp = st.tabs([
        "SHAP Heatmaps",
        "SHAP Overview",
        "Data Completeness",
    ])

    # ── Tab 1: SHAP Heatmaps ─────────────────────────────────────────
    with tab_shap:
        if shap_data is None:
            st.info("Click **Compute SHAP** in the sidebar to generate explanations for the current time point.")
        else:
            fig_cont = mods["plot_continuous_ts_shap_plotly"](
                shap_data["shap_dict"],
                sample_idx=0,
                channel2feature=shap_data["channel2feature"],
                height=max(400, len(shap_data["channel2feature"]) * 14),
            )
            if fig_cont:
                fig_cont.update_layout(width=None)
                st.plotly_chart(fig_cont, use_container_width=True)
            else:
                st.warning("No continuous TS SHAP data available")

            st.markdown("---")

            fig_cat = mods["plot_categorical_ts_shap_plotly"](
                shap_data["shap_dict"],
                sample_idx=0,
                height=500,
            )
            if fig_cat:
                fig_cat.update_layout(width=None)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No categorical TS SHAP data available")

            st.markdown("---")

            if shap_data["ebm_explanations"]:
                fig_ebm = mods["plot_ebm_contributions_plotly"](
                    shap_data["ebm_explanations"],
                    height=600,
                )
                if fig_ebm:
                    fig_ebm.update_layout(width=None)
                    st.plotly_chart(fig_ebm, use_container_width=True)
                else:
                    st.info("Could not generate EBM plot")
            else:
                st.info("No EBM explanations available for this patient/timepoint")

            st.markdown("---")

            fig_channels = mods["plot_top_channels_plotly"](
                shap_data["shap_dict"], sample_idx=0,
                channel2feature=shap_data["channel2feature"])
            if fig_channels:
                fig_channels.update_layout(width=None)
                st.plotly_chart(fig_channels, use_container_width=True)

    # ── Tab: SHAP Overview ────────────────────────────────────────────
    with tab_overview:
        if shap_data is None:
            st.info("Click **Compute SHAP** in the sidebar to generate explanations.")
        else:
            fig_budget = mods["plot_shap_budget_plotly"](
                shap_data["shap_dict"], sample_idx=0,
                channel2feature=shap_data["channel2feature"])
            if fig_budget:
                fig_budget.update_layout(width=None)
                st.plotly_chart(fig_budget, use_container_width=True)

            fig_temporal = mods["plot_shap_temporal_plotly"](
                shap_data["shap_dict"], sample_idx=0,
                channel2feature=shap_data["channel2feature"])
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

    # ── Tab 3: Data Completeness ──────────────────────────────────────
    with tab_comp:
        # Data completeness uses shap_dict but can also work from prediction data
        completeness_source = shap_data["shap_dict"] if shap_data else None
        completeness_ch2feat = shap_data["channel2feature"] if shap_data else None

        if completeness_source is None:
            st.info("Click **Compute SHAP** in the sidebar to generate data completeness analysis.")
        else:
            comp_figs = mods["plot_data_completeness_plotly"](
                completeness_source,
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
