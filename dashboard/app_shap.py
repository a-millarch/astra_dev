"""
ASTRA Patient SHAP Dashboard

Run from repo root:
    streamlit run dashboard/app.py

Requires: pip install streamlit plotly
"""

import sys
import os
import logging

# ── Ensure repo root is on path ──────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
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
    from astra.utils import get_cfg, get_base_df, ProjectManager
    from astra.inference import InferenceSession
    from astra.inference.patient_context import PatientContext
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
        "ProjectManager": ProjectManager,
        "InferenceSession": InferenceSession,
        "PatientContext": PatientContext,
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

st.cache_resource.clear()

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
    os.chdir(REPO_ROOT)
    mods["ProjectManager"]()
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
# INFERENCE + SHAP (no run_inference.py dependency)
# ═════════════════════════════════════════════════════════════════════════

def _get_or_create_context(cfg, cpr, sd, actual_start, hours_offset):
    """
    Get cached PatientContext or create a new one.
    If the patient changed, rebuild from CSV.
    If only time changed, use refresh (much faster).
    """
    mods = _load_astra_modules()
    session = load_session(cfg["model_name"])

    current_time = pd.Timestamp(actual_start) + pd.Timedelta(hours=hours_offset)
    patient_key = f"{cpr}_{sd}_{cfg['model_name']}"

    # Check if we already have a context for this patient
    if (st.session_state.get("_ctx_patient_key") == patient_key
            and st.session_state.get("_ctx") is not None):
        # Same patient, just refresh to new time (fast!)
        ctx = st.session_state["_ctx"]
        session.refresh_and_predict(ctx, current_time=current_time)
    else:
        # New patient — build from CSV
        ctx = mods["PatientContext"].from_csv(
            cpr_hash=cpr,
            service_date=sd,
            current_time=current_time,
            bundle=session.bundle,
            data_dir="data/raw",
        )
        st.session_state["_ctx_patient_key"] = patient_key
        st.session_state["_ctx"] = ctx

    return session, ctx


def run_inference_and_shap(cfg, cpr, sd, actual_start, hours_offset, progress_bar=None):
    """
    Build/refresh PatientContext, run prediction + SHAP for a patient.
    Uses refresh_and_predict when only time changes (fast).
    Rebuilds from CSV only when patient or model changes.
    """
    if progress_bar:
        progress_bar.progress(10, text="Building patient context...")
    session, ctx = _get_or_create_context(cfg, cpr, sd, actual_start, hours_offset)

    if progress_bar:
        progress_bar.progress(25, text="Running prediction...")
    result = session.predict_from_context(ctx)

    if progress_bar:
        progress_bar.progress(35, text="Computing SHAP values (this takes a moment)...")
    shap_result = session.explain_from_context(ctx)

    if progress_bar:
        progress_bar.progress(75, text="Building visualizations...")
    shap_dict, ch2feat, cat_names, cont_names = session.shap_to_viz_dict(
        shap_result, ctx.x_ts, ctx.x_ts_cat, ctx.tab_df
    )

    ebm_explanations = None
    if "_ebm_pred" in session.bundle.get("ts_channel_names", []):
        if progress_bar:
            progress_bar.progress(85, text="Computing EBM explanations...")
        ebm_explanations = session.explain_ebm(ctx, save_path=None)

    if progress_bar:
        progress_bar.progress(95, text="Finalizing...")

    return {
        "session": session,
        "ctx": ctx,
        "result": result,
        "shap_dict": shap_dict,
        "channel2feature": ch2feat,
        "feature_names_cat": cat_names,
        "feature_names_cont": cont_names,
        "ebm_explanations": ebm_explanations,
    }


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

    # ── Run button ────────────────────────────────────────────────────
    run_clicked = st.sidebar.button(
        "Run Inference + SHAP",
        type="primary",
        use_container_width=True,
    )

    if not run_clicked:
        if "data" not in st.session_state:
            st.info("Select a patient and time, then click **Run Inference + SHAP** in the sidebar.")
            return
        else:
            # Re-use cached results from last run
            data = st.session_state["data"]
            cfg_used = st.session_state["cfg_used"]
            # Skip to rendering (below)
    

    # ── Compute (only when button clicked) ────────────────────────────
    if run_clicked:
        try:
            progress = st.progress(0, text="Loading patient data...")
            data = run_inference_and_shap(cfg, cpr, sd, actual_start, hours_offset, progress_bar=progress)
            progress.empty()
            st.session_state["data"] = data
            st.session_state["cfg_used"] = cfg
        except Exception as e:
            st.error(f"Error running inference: {e}")
            st.exception(e)
            return
    else:
        data = st.session_state.get("data")
        cfg_used = st.session_state.get("cfg_used")
        if data is None:
            return
        cfg = cfg_used if cfg_used else cfg

    mods = _load_astra_modules()

    # ── Header metrics ────────────────────────────────────────────────
    result = data["result"]
    prob = result.probability
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P(deceased 30d)", f"{prob:.3f}")
    with col2:
        st.metric("Trajectory Length", f"{result.trajectory_length} steps")
    with col3:
        st.metric("Eval Step", f"{result.censor_step}")

    # ── Tabs ──────────────────────────────────────────────────────────
    # ── Patient Context (expandable) ──────────────────────────────────
    with st.expander("Patient Context"):
        ctx = data["ctx"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
**PID:** `{ctx.pid}`  
**Admission:** `{ctx.admission_time}`  
**Trajectory:** {result.trajectory_length} steps  
**Model:** `{cfg.get('model_name')}`  
**Temporal head:** `{data['session'].is_temporal}`
""")
        with col_b:
            if hasattr(ctx, 'tab_df') and ctx.tab_df is not None:
                st.markdown("**Static features:**")
                st.dataframe(ctx.tab_df.T.rename(columns={ctx.tab_df.index[0]: "Value"}),
                             use_container_width=True, height=200)

    # ── Prediction Trajectory (always visible) ───────────────────────
    fig_traj = mods["plot_prediction_trajectory_plotly"](
        data["result"], data["ctx"], model_name=cfg.get("model_name"))
    if fig_traj:
        fig_traj.update_layout(width=None)
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.info("No trajectory data available.")

    st.markdown("---")

    tab_shap, tab_overview, tab_comp = st.tabs([
        "SHAP Heatmaps",
        "SHAP Overview",
        "Data Completeness",
    ])

    # ── Tab 1: SHAP Heatmaps ─────────────────────────────────────────
    with tab_shap:
        fig_cont = mods["plot_continuous_ts_shap_plotly"](
            data["shap_dict"],
            sample_idx=0,
            channel2feature=data["channel2feature"],
            height=max(400, len(data["channel2feature"]) * 14),
        )
        if fig_cont:
            fig_cont.update_layout(width=None)
            st.plotly_chart(fig_cont, use_container_width=True)
        else:
            st.warning("No continuous TS SHAP data available")

        st.markdown("---")

        fig_cat = mods["plot_categorical_ts_shap_plotly"](
            data["shap_dict"],
            sample_idx=0,
            height=500,
        )
        if fig_cat:
            fig_cat.update_layout(width=None)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No categorical TS SHAP data available")

        st.markdown("---")

        if data["ebm_explanations"]:
            fig_ebm = mods["plot_ebm_contributions_plotly"](
                data["ebm_explanations"],
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
            data["shap_dict"], sample_idx=0, channel2feature=data["channel2feature"])
        if fig_channels:
            fig_channels.update_layout(width=None)
            st.plotly_chart(fig_channels, use_container_width=True)

    # ── Tab: SHAP Overview ────────────────────────────────────────────
    with tab_overview:
        fig_budget = mods["plot_shap_budget_plotly"](
            data["shap_dict"], sample_idx=0, channel2feature=data["channel2feature"])
        if fig_budget:
            fig_budget.update_layout(width=None)
            st.plotly_chart(fig_budget, use_container_width=True)

        fig_temporal = mods["plot_shap_temporal_plotly"](
            data["shap_dict"], sample_idx=0, channel2feature=data["channel2feature"])
        if fig_temporal:
            fig_temporal.update_layout(width=None)
            st.plotly_chart(fig_temporal, use_container_width=True)

        st.markdown("---")

        fig_static = mods["plot_static_features_plotly"](
            data["shap_dict"], sample_idx=0,
            feature_names_cat=data["feature_names_cat"],
            feature_names_cont=data["feature_names_cont"])
        if fig_static:
            fig_static.update_layout(width=None)
            st.plotly_chart(fig_static, use_container_width=True)

    # ── Tab 3: Data Completeness ──────────────────────────────────────
    with tab_comp:
        comp_figs = mods["plot_data_completeness_plotly"](
            data["shap_dict"],
            sample_idx=0,
            channel2feature=data["channel2feature"],
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