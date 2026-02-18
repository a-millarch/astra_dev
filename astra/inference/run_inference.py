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


def plot_prediction_trajectory(result, ctx, save_path=None):
    """Plot P(deceased_30d) at each visible timestep."""
    if result.predictions_over_time is None:
        print("Model does not have temporal head — skipping trajectory plot")
        return

    probs = result.predictions_over_time  # [seq_len]
    traj_len = result.trajectory_length

    # Bin midpoints as hours since admission
    bin_df = ctx.bin_df
    admission = ctx.admission_time
    hours = [
        (row.bin_start - admission).total_seconds() / 3600
        + (row.bin_end - row.bin_start).total_seconds() / 7200
        for _, row in bin_df.iterrows()
    ]
    hours = hours[:traj_len]
    probs_visible = probs[:traj_len]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hours, probs_visible, "o-", color="steelblue", markersize=3, linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.fill_between(hours, probs_visible, alpha=0.15, color="steelblue")

    ax.set_xlabel("Hours since admission")
    ax.set_ylabel("P(deceased 30d)")
    ax.set_title(f"Prediction trajectory — Patient {ctx.pid}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory plot to {save_path}")
    plt.close(fig)


def run(cpr_hash, service_date, current_time, model_name,
                data_dir="data/raw", save_dir="reports/inference", device=None):
    """Full end-to-end example."""

    os.makedirs(save_dir, exist_ok=True)
    pid_short = cpr[:8] + sd.astype(str)[:10].replace('-','')

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

    shap_result = session.explain(
        x_ts=ctx.x_ts,
        x_ts_cat=ctx.x_ts_cat,
        tab_df=ctx.tab_df,
    )

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
