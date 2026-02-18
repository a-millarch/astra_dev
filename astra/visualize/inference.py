import os

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")  # headless backend for script execution only
import matplotlib.pyplot as plt


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
