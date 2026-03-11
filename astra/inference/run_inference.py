"""
Example: end-to-end inference with PatientContext.

Demonstrates:
  1. First-time inference from CSV files
  2. Prediction trajectory over time (all visible timesteps)
  3. SHAP explanation + visualization
  4. Re-inference with new data (refresh)

Run in Azure where data and models exist:

    python -m astra.inference.example_usage \
        --cpr-hash <HASH> \
        --service-date "2023-08-15 10:30:00" \
        --current-time "2023-08-16 06:00:00" \
        --model-name <MODEL>
"""

import argparse
import logging
import os

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")  # headless backend for script execution only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astra.inference import InferenceSession, PatientContext
from astra.evaluation.behavior import visualize_shap_individual, visualize_data_completeness, visualize_shap_individual_interactive, plot_prediction_trajectory_plotly
from astra.visualize.inference import plot_prediction_trajectory
from astra.utils import make_inference_pid

logger = logging.getLogger(__name__)

def run(cpr_hash, service_date, current_time, model_name,
                data_dir="data/raw", save_dir="reports/inference", device=None):
    """Full end-to-end example."""

    os.makedirs(save_dir, exist_ok=True)
    pid_short = make_inference_pid(cpr_hash, service_date)

    # ---- 1. Load session ----
    logger.info("Loading model: %s", model_name)
    session = InferenceSession.load(model_name, device=device)
    logger.info("Temporal head: %s | Channels: %d",
                session.is_temporal, len(session.bundle['ts_channel_names']))

    # ---- 2. Create PatientContext from CSV ----
    logger.info("Building PatientContext for %s (service=%s, current=%s)",
                pid_short, service_date, current_time)

    ctx = PatientContext.from_csv(
        cpr_hash=cpr_hash,
        service_date=service_date,
        current_time=current_time,
        bundle=session.bundle,
        data_dir=data_dir,
    )

    logger.info("PID=%s admission=%s traj_len=%d bins=%d",
                ctx.pid, ctx.admission_time, ctx.trajectory_length, len(ctx.bin_df))
    logger.debug("x_ts=%s x_ts_cat=%s", ctx.x_ts.shape, ctx.x_ts_cat.shape)

    # ---- 3. Predict ----
    logger.info("Running prediction...")
    result = session.predict_from_context(ctx)
    logger.info("P(deceased_30d)=%.4f censor_step=%s traj_len=%d",
                result.probability, result.censor_step, result.trajectory_length)

    # ---- 4. Plot prediction trajectory ----
    plot_prediction_trajectory(
        result, ctx,
        save_path=f"{save_dir}/trajectory_{pid_short}.png",
        model_name=model_name,
    )

    # ---- 5. SHAP explanation ----
    logger.info("Computing SHAP explanation...")
    shap_result = session.explain_from_context(ctx)

    shap_dict, channel2feature, feature_names_cat, feature_names_cont = (
        session.shap_to_viz_dict(
            shap_result,
            x_ts=ctx.x_ts,
            x_ts_cat=ctx.x_ts_cat,
            tab_df=ctx.tab_df,
        )
    )

    shap_save_path = f"{save_dir}/shap_patient_{pid_short}.png"
    visualize_shap_individual(
        shap_dict,
        sample_idx=0,
        channel2feature=channel2feature,
        feature_names_cat=feature_names_cat,
        feature_names_cont=feature_names_cont,
        save_path=shap_save_path,
    )
    logger.info("Saved SHAP plot to %s", shap_save_path)

    # ---- 5b. EBM feature importance (if EBM is enabled) ----
    if '_ebm_pred' in session.bundle.get('ts_channel_names', []):
        logger.info("Computing EBM feature importance...")
        ebm_save = f"{save_dir}/ebm_importance_{pid_short}.png"
        ebm_explanations = session.explain_ebm(ctx, save_path=ebm_save)
        if ebm_explanations:
            logger.info("Saved EBM importance plot to %s (%d timeframes)",
                        ebm_save, len(ebm_explanations))

    # ---- 6. Demonstrate re-inference (refresh) ----
    later_time = pd.Timestamp(current_time) + pd.Timedelta(hours=6)
    logger.info("Refreshing context to %s (6 hours later)", later_time)
    result2 = session.refresh_and_predict(ctx, current_time=later_time)

    logger.info("After refresh: traj_len=%d P(deceased)=%.4f delta_P=%+.4f",
                result2.trajectory_length, result2.probability,
                result2.probability - result.probability)

    # ---- 7. Save context for later ----
    ctx_path = f"{save_dir}/context_{pid_short}.pkl"
    ctx.save(ctx_path)
    logger.info("Saved PatientContext to %s", ctx_path)

    logger.info("Done!")


def initialize_session(cpr_hash, service_date, current_time, model_name,
                data_dir="data/raw", save_dir="reports/inference", device=None):

    os.makedirs(save_dir, exist_ok=True)
    pid_short = make_inference_pid(cpr_hash, service_date)

    # ---- 1. Load session ----
    logger.info("Loading model: %s", model_name)
    session = InferenceSession.load(model_name, device=device)
    logger.info("Temporal head: %s | Channels: %d",
                session.is_temporal, len(session.bundle['ts_channel_names']))

    # ---- 2. Create PatientContext from CSV ----
    logger.info("Building PatientContext for %s (service=%s, current=%s)",
                pid_short, service_date, current_time)

    ctx = PatientContext.from_csv(
        cpr_hash=cpr_hash,
        service_date=service_date,
        current_time=current_time,
        bundle=session.bundle,
        data_dir=data_dir,
    )

    logger.info("PID=%s admission=%s traj_len=%d bins=%d",
                ctx.pid, ctx.admission_time, ctx.trajectory_length, len(ctx.bin_df))
    logger.debug("x_ts=%s x_ts_cat=%s", ctx.x_ts.shape, ctx.x_ts_cat.shape)

    # ---- 3. Predict ----
    logger.info("Running prediction...")
    result = session.predict_from_context(ctx)
    logger.info("P(deceased_30d)=%.4f censor_step=%s traj_len=%d",
                result.probability, result.censor_step, result.trajectory_length)

    session.ctx = ctx
    session.result = result

    return session

def default_session_plot(session, prediction_curve=None, interactive =False):
    ctx = session.ctx
    result = session.predict_from_context(ctx)

    # If a simulation prediction curve is available, use it for trajectory
    # plotting (enables non-temporal models to show per-timestep predictions
    # from the simulation step-through without needing a pre-computed CSV).
    if prediction_curve is not None and result.predictions_over_time is None:
        from astra.inference.pipeline import InferenceResult
        plot_result = InferenceResult(
            pid=result.pid,
            probability=result.probability,
            trajectory_length=result.trajectory_length,
            censor_step=result.censor_step,
            predictions_over_time=prediction_curve,
        )
    else:
        plot_result = result

    # ---- 4. Plot prediction trajectory ----
    model_name = session.bundle.get('model_name')
    traj_fig = plot_prediction_trajectory(
        plot_result, ctx,
        save_path=None,
        model_name=model_name,
    )

    # ---- SHAP explanation ----
    logger.info("Computing SHAP explanation...")
    shap_result = session.explain_from_context(ctx)

    shap_dict, channel2feature, feature_names_cat, feature_names_cont = (
        session.shap_to_viz_dict(
            shap_result,
            x_ts=ctx.x_ts,
            x_ts_cat=ctx.x_ts_cat,
            tab_df=ctx.tab_df,
        )
    )

    # Inject inhospital boundary info for SHAP plots (prehospital patients)
    ihs_time = ctx.demographics.get('inhospital_start')
    if ihs_time is not None:
        ihs_ts = pd.Timestamp(ihs_time)
        if pd.notna(ihs_ts):
            from astra.evaluation.utils import time_to_step
            delta_min = (ihs_ts - ctx.admission_time).total_seconds() / 60
            ihs_step = time_to_step(delta_min, 'min')
            if ihs_step is not None:
                shap_dict['test_data']['inhospital_start_steps'] = np.array([ihs_step])
    if interactive:

    
    # ---- EBM explanations (if available) ----
        ebm_explanations = None
        if '_ebm_pred' in session.bundle.get('ts_channel_names', []):
            logger.info("Computing EBM feature importance...")
            ebm_explanations = session.explain_ebm(ctx, save_path=None)

        # ---- Interactive Plotly heatmaps + matplotlib fallback ----
        visualize_shap_individual_interactive(
            shap_dict,
            sample_idx=0,
            channel2feature=channel2feature,
            feature_names_cat=feature_names_cat,
            feature_names_cont=feature_names_cont,
            ebm_explanations=ebm_explanations,
        )
    else:
        visualize_shap_individual(            shap_dict,
            sample_idx=0,
            channel2feature=channel2feature,
            feature_names_cat=feature_names_cat,
            feature_names_cont=feature_names_cont)

    # ---- Data completeness (matplotlib — unchanged) ----
    visualize_data_completeness(
        shap_dict,
        channel2feature=channel2feature,
        save_path=None,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end inference example with PatientContext"
    )
    parser.add_argument("--cpr-hash", required=True, help="Patient CPR hash")
    parser.add_argument("--service-date", required=True, help="Trauma service date")
    parser.add_argument("--current-time", required=True, help="Inference time")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--data-dir", default="data/raw", help="Raw CSV directory")
    parser.add_argument("--save-dir", default="reports/inference", help="Output directory")
    parser.add_argument("--device", default=None, help="Force device (default: auto-detect)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging")

    args = parser.parse_args()

    from astra.utils import setup_logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    run(
        cpr_hash=args.cpr_hash,
        service_date=args.service_date,
        current_time=args.current_time,
        model_name=args.model_name,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        device=args.device,
    )
