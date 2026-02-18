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
import os

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")  # headless backend for script execution only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astra.inference import InferenceSession, PatientContext
from astra.evaluation.behavior import visualize_shap_individual
from astra.visualize.inference import plot_prediction_trajectory

def run(cpr_hash, service_date, current_time, model_name,
                data_dir="data/raw", save_dir="reports/inference", device=None):
    """Full end-to-end example."""

    os.makedirs(save_dir, exist_ok=True)
    pid_short = cpr_hash[:8] + service_date.astype(str)[:10].replace('-','')

    # ---- 1. Load session ----
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    session = InferenceSession.load(model_name, device=device)
    print(f"  Temporal head: {session.is_temporal}")
    print(f"  Channels: {len(session.bundle['ts_channel_names'])}")

    # ---- 2. Create PatientContext from CSV ----
    print(f"\n{'='*60}")
    print(f"Building PatientContext for {pid_short}...")
    print(f"  Service date:  {service_date}")
    print(f"  Current time:  {current_time}")
    print(f"{'='*60}")

    ctx = PatientContext.from_csv(
        cpr_hash=cpr_hash,
        service_date=service_date,
        current_time=current_time,
        bundle=session.bundle,
        data_dir=data_dir,
    )

    print(f"  PID:               {ctx.pid}")
    print(f"  Admission:         {ctx.admission_time}")
    print(f"  Max window:        {ctx.max_time}")
    print(f"  Trajectory length: {ctx.trajectory_length}")
    print(f"  Total bins:        {len(ctx.bin_df)}")
    print(f"  x_ts shape:        {ctx.x_ts.shape}")
    print(f"  x_ts_cat shape:    {ctx.x_ts_cat.shape}")

    # ---- 3. Predict ----
    print(f"\n{'='*60}")
    print("Running prediction...")
    print(f"{'='*60}")

    result = session.predict_from_context(ctx)
    print(f"  P(deceased_30d):   {result.probability:.4f}")
    print(f"  Censor step:       {result.censor_step}")
    print(f"  Trajectory length: {result.trajectory_length}")

    # ---- 4. Plot prediction trajectory ----
    plot_prediction_trajectory(
        result, ctx,
        save_path=f"{save_dir}/trajectory_{pid_short}.png",
    )

    # ---- 5. SHAP explanation ----
    print(f"\n{'='*60}")
    print("Computing SHAP explanation...")
    print(f"{'='*60}")

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
    print(f"  Saved SHAP plot to {shap_save_path}")

    # ---- 6. Demonstrate re-inference (refresh) ----
    print(f"\n{'='*60}")
    print("Demonstrating refresh (simulating 6 hours later)...")
    print(f"{'='*60}")

    later_time = pd.Timestamp(current_time) + pd.Timedelta(hours=6)
    result2 = session.refresh_and_predict(ctx, current_time=later_time)

    print(f"  New current_time:  {ctx.current_time}")
    print(f"  New traj length:   {result2.trajectory_length}")
    print(f"  New P(deceased):   {result2.probability:.4f}")
    print(f"  Delta P:           {result2.probability - result.probability:+.4f}")

    # ---- 7. Save context for later ----
    ctx_path = f"{save_dir}/context_{pid_short}.pkl"
    ctx.save(ctx_path)
    print(f"\n  Saved PatientContext to {ctx_path}")
    print(f"  (Reload with: PatientContext.load('{ctx_path}', bundle=session.bundle))")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


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

    args = parser.parse_args()
    run(
        cpr_hash=args.cpr_hash,
        service_date=args.service_date,
        current_time=args.current_time,
        model_name=args.model_name,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        device=args.device,
    )
