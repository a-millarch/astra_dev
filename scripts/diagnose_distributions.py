"""
Diagnostic script for analyzing feature distributions before normalization.

Characterizes each continuous TS channel and tabular numeric feature to determine
whether StandardScaler (z-score) is appropriate or if alternatives (RobustScaler,
PowerTransformer, QuantileTransformer) would be better suited.

Outputs:
  - Console summary with per-channel statistics and recommendations
  - reports/distributions/summary.csv — full stats table
  - reports/distributions/<channel>.png — per-channel histograms with KDE
  - reports/distributions/worst_skew_grid.png — 3x3 grid of most-skewed channels
  - reports/distributions/recommendation_summary.png — bar chart of recommendations

Usage:
    python scripts/diagnose_distributions.py
    python scripts/diagnose_distributions.py --no-plots   # skip plot generation
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from astra.utils import get_cfg

logger = logging.getLogger(__name__)

SAVE_DIR = "reports/distributions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def _measured_mask(X, trajectory_lengths):
    """Build boolean mask [n_samples, n_channels, seq_len] for measured values.

    Measured = not NaN AND within trajectory (position < trajectory_length).
    """
    n_samples, n_channels, seq_len = X.shape
    pos = np.arange(seq_len)[np.newaxis, :]
    tl = trajectory_lengths[:, np.newaxis]
    padding_2d = pos >= tl  # [n_samples, seq_len]
    padding_3d = np.broadcast_to(
        padding_2d[:, np.newaxis, :], (n_samples, n_channels, seq_len)
    )
    return ~np.isnan(X) & ~padding_3d


def _classify_distribution(skewness, kurtosis_excess, values=None):
    """Classify distribution shape and recommend scaler."""
    from astra.data.dataloader import AstraScaler

    abs_skew = abs(skewness)
    if abs_skew < 0.5 and kurtosis_excess < 3:
        return "normal-ish", "StandardScaler"
    elif abs_skew < 2 and kurtosis_excess < 7:
        return "moderate skew", "PowerTransformer"
    else:
        if values is not None and AstraScaler._is_boundary_concentrated(values):
            return "boundary-concentrated", "RobustScaler"
        return "heavy skew/tails", "QuantileTransformer"


# ---------------------------------------------------------------------------
# Channel analysis
# ---------------------------------------------------------------------------

def analyze_channel(values, name):
    """Compute distribution statistics for a single channel's measured values.

    Args:
        values: 1D array of measured (non-NaN, non-padding) values.
        name: Channel name for reporting.

    Returns:
        Dict of statistics.
    """
    n = len(values)
    if n < 3:
        return {
            "channel": name, "n_measured": n,
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "min": np.nan, "max": np.nan,
            "p1": np.nan, "p5": np.nan, "p25": np.nan,
            "p75": np.nan, "p95": np.nan, "p99": np.nan,
            "skewness": np.nan, "kurtosis": np.nan,
            "shapiro_p": np.nan,
            "distribution_class": "insufficient data",
            "recommendation": "N/A",
        }

    skewness = float(sp_stats.skew(values, nan_policy="omit"))
    kurtosis = float(sp_stats.kurtosis(values, nan_policy="omit"))  # excess kurtosis

    # Shapiro-Wilk: subsample to 5000 (limit of the test)
    if n > 5000:
        rng = np.random.RandomState(42)
        subsample = rng.choice(values, size=5000, replace=False)
    else:
        subsample = values
    try:
        _, shapiro_p = sp_stats.shapiro(subsample)
    except Exception:
        shapiro_p = np.nan

    dist_class, recommendation = _classify_distribution(skewness, kurtosis, values)

    return {
        "channel": name,
        "n_measured": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "shapiro_p": shapiro_p,
        "distribution_class": dist_class,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_channel_histogram(values, name, stats, save_dir):
    """Plot histogram + KDE for a single channel and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Clip to p1-p99 for visualization (show inset text for full range)
    p1, p99 = np.percentile(values, [1, 99])
    clipped = values[(values >= p1) & (values <= p99)]

    ax.hist(clipped, bins=100, density=True, alpha=0.7, color="steelblue",
            edgecolor="none", label="Histogram (p1–p99)")

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(clipped)
        x_kde = np.linspace(p1, p99, 300)
        ax.plot(x_kde, kde(x_kde), color="darkred", linewidth=1.5, label="KDE")
    except Exception:
        pass

    ax.set_title(f"{name}  (n={stats['n_measured']:,})", fontsize=11)
    ax.set_xlabel("Raw value (p1–p99 range)")
    ax.set_ylabel("Density")

    # Stats inset
    info = (f"skew={stats['skewness']:.2f}  kurt={stats['kurtosis']:.2f}\n"
            f"mean={stats['mean']:.3f}  std={stats['std']:.3f}\n"
            f"full range: [{stats['min']:.2f}, {stats['max']:.2f}]\n"
            f"→ {stats['recommendation']}")
    ax.text(0.97, 0.95, info, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.legend(fontsize=8)
    plt.tight_layout()
    safe_name = name.replace("/", "_").replace("\\", "_")
    fig.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=120)
    plt.close(fig)


def plot_worst_skew_grid(all_stats, measured_values_by_ch, save_dir):
    """Plot 3x3 grid of the 9 most-skewed channels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort by |skewness| descending
    ranked = sorted(
        [(s, measured_values_by_ch[s["channel"]]) for s in all_stats
         if not np.isnan(s["skewness"])],
        key=lambda x: abs(x[0]["skewness"]),
        reverse=True,
    )[:9]

    if len(ranked) == 0:
        return

    n = len(ranked)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (s, vals) in enumerate(ranked):
        ax = axes[i]
        p1, p99 = np.percentile(vals, [1, 99])
        clipped = vals[(vals >= p1) & (vals <= p99)]
        ax.hist(clipped, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        ax.set_title(f"{s['channel']}\nskew={s['skewness']:.2f}  kurt={s['kurtosis']:.2f}",
                     fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top 9 Most-Skewed Channels (p1–p99 range)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "worst_skew_grid.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_recommendation_summary(all_stats, save_dir):
    """Bar chart showing count of channels per recommended scaler."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [s["recommendation"] for s in all_stats if s["recommendation"] != "N/A"]
    from collections import Counter
    counts = Counter(recs)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]
    colors = {"StandardScaler": "#4CAF50", "PowerTransformer": "#FF9800",
              "QuantileTransformer": "#F44336"}
    bar_colors = [colors.get(l, "#9E9E9E") for l in labels]

    ax.barh(labels, values, color=bar_colors, edgecolor="none")
    ax.set_xlabel("Number of channels")
    ax.set_title("Recommended Normalization Method per Channel")
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "recommendation_summary.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tabular analysis
# ---------------------------------------------------------------------------

def analyze_tabular(data, cfg):
    """Analyze distribution of tabular numeric columns."""
    _section("TABULAR NUMERIC FEATURES")

    num_cols = cfg["dataset"]["num_cols"]
    trainval_base = data["trainval"].base

    results = []
    for col in num_cols:
        if col not in trainval_base.columns:
            print(f"  {col}: not found in base_df")
            continue
        vals = trainval_base[col].dropna().values.astype(float)
        s = analyze_channel(vals, f"tab:{col}")
        results.append(s)
        print(f"  {col:<20s}  n={s['n_measured']:>6d}  "
              f"mean={s['mean']:>8.2f}  std={s['std']:>7.2f}  "
              f"skew={s['skewness']:>6.2f}  kurt={s['kurtosis']:>6.2f}  "
              f"→ {s['recommendation']}")

    return results


# ---------------------------------------------------------------------------
# Current normalization impact analysis
# ---------------------------------------------------------------------------

def analyze_normalization_impact(data):
    """Compare raw vs normalized distributions to show StandardScaler impact."""
    _section("STANDARDSCALER IMPACT: Raw vs Normalized")
    print("  Shows how the current z-score normalization handles each channel.")
    print("  Key concern: skewed channels remain skewed after z-score — only shifted/scaled.\n")

    X_raw = data["X_raw"]
    X_norm = data["X"]
    traj_lengths = data["trajectory_lengths"]
    channel_names = data.get("ts_channel_names", [])
    mask = _measured_mask(X_raw, traj_lengths)

    header = (f"  {'Channel':<26s} {'raw_skew':>9s} {'norm_skew':>10s} "
              f"{'raw_range':>20s} {'norm_range':>20s} {'z>3':>6s} {'z>5':>6s}")
    print(header)
    print(f"  {'-'*26} {'-'*9} {'-'*10} {'-'*20} {'-'*20} {'-'*6} {'-'*6}")

    for ch in range(X_raw.shape[1]):
        name = channel_names[ch] if ch < len(channel_names) else f"ch_{ch}"
        m = mask[:, ch, :]

        raw_vals = X_raw[:, ch, :][m]
        norm_vals = X_norm[:, ch, :][m]

        if len(raw_vals) < 10:
            continue

        raw_skew = float(sp_stats.skew(raw_vals))
        norm_skew = float(sp_stats.skew(norm_vals))

        raw_range = f"[{raw_vals.min():.2f}, {raw_vals.max():.2f}]"
        norm_range = f"[{norm_vals.min():.2f}, {norm_vals.max():.2f}]"

        # Extreme z-scores in normalized data
        pct_gt3 = 100.0 * np.sum(np.abs(norm_vals) > 3) / len(norm_vals)
        pct_gt5 = 100.0 * np.sum(np.abs(norm_vals) > 5) / len(norm_vals)

        print(f"  {name:<26s} {raw_skew:>9.2f} {norm_skew:>10.2f} "
              f"{raw_range:>20s} {norm_range:>20s} {pct_gt3:>5.1f}% {pct_gt5:>5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze feature distributions for normalization")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    cfg = get_cfg()

    print("Loading data...")
    from astra.data.caching import prepare_data_and_dls_cached
    data = prepare_data_and_dls_cached(cfg)

    X_raw = data["X_raw"]
    traj_lengths = data["trajectory_lengths"]
    channel_names = data.get("ts_channel_names", [])

    print(f"X_raw shape: {X_raw.shape}")
    print(f"Channels: {len(channel_names)}")
    print(f"Trainval samples: {X_raw.shape[0]}")

    # Auxiliary channels to flag (not exclude from analysis — still interesting to see)
    tf_cfg = cfg.get("temporal_features", {})
    aux_names = set(tf_cfg.get("features", []))

    # ---- Build measured mask ----
    mask = _measured_mask(X_raw, traj_lengths)

    # ---- Analyze each TS channel ----
    _section("CONTINUOUS TIME SERIES CHANNELS")
    print(f"  Analyzing {len(channel_names)} channels on trainval (pre-normalization)\n")

    header = (f"  {'Channel':<26s} {'n':>8s} {'sparsity':>9s} {'mean':>9s} "
              f"{'std':>9s} {'skew':>7s} {'kurt':>7s} {'shapiro_p':>10s} "
              f"{'class':>16s} {'recommendation':>20s}")
    print(header)
    print(f"  {'-' * 140}")

    all_stats = []
    measured_values_by_ch = {}

    for ch in range(X_raw.shape[1]):
        name = channel_names[ch] if ch < len(channel_names) else f"ch_{ch}"
        m = mask[:, ch, :]
        vals = X_raw[:, ch, :][m]

        # Sparsity: fraction of within-trajectory positions that are NaN
        n_within_traj = sum(int(tl) for tl in traj_lengths)
        n_measured = len(vals)
        sparsity = 100.0 * (1.0 - n_measured / max(n_within_traj, 1))

        s = analyze_channel(vals, name)
        s["sparsity_pct"] = sparsity
        s["is_auxiliary"] = name in aux_names

        all_stats.append(s)
        measured_values_by_ch[name] = vals

        aux_flag = " [AUX]" if name in aux_names else ""
        shapiro_str = f"{s['shapiro_p']:.2e}" if not np.isnan(s['shapiro_p']) else "N/A"
        print(f"  {name:<26s} {s['n_measured']:>8,d} {sparsity:>8.1f}% "
              f"{s['mean']:>9.3f} {s['std']:>9.3f} "
              f"{s['skewness']:>7.2f} {s['kurtosis']:>7.2f} "
              f"{shapiro_str:>10s} {s['distribution_class']:>16s} "
              f"{s['recommendation']:>20s}{aux_flag}")

    # ---- Tabular features ----
    tab_stats = analyze_tabular(data, cfg)

    # ---- Normalization impact ----
    analyze_normalization_impact(data)

    # ---- Summary ----
    _section("RECOMMENDATION SUMMARY")

    non_aux = [s for s in all_stats if not s.get("is_auxiliary", False)]
    from collections import Counter
    rec_counts = Counter(s["recommendation"] for s in non_aux if s["recommendation"] != "N/A")
    for rec, count in sorted(rec_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(non_aux)
        print(f"  {rec:<25s} {count:>3d} channels ({pct:.0f}%)")

    heavy = [s for s in non_aux
             if s["distribution_class"] == "heavy skew/tails" and s["recommendation"] != "N/A"]
    if heavy:
        print(f"\n  Channels with heavy skew/tails ({len(heavy)}):")
        for s in sorted(heavy, key=lambda x: abs(x["skewness"]), reverse=True):
            print(f"    {s['channel']:<26s}  skew={s['skewness']:>7.2f}  "
                  f"kurt={s['kurtosis']:>7.2f}  range=[{s['min']:.2f}, {s['max']:.2f}]")

    # ---- Save CSV ----
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_results = all_stats + tab_stats
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(SAVE_DIR, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved summary to {csv_path}")

    # ---- Plots ----
    if not args.no_plots:
        _section("GENERATING PLOTS")
        print(f"  Saving to {SAVE_DIR}/")

        for s in all_stats:
            vals = measured_values_by_ch.get(s["channel"])
            if vals is not None and len(vals) >= 10:
                plot_channel_histogram(vals, s["channel"], s, SAVE_DIR)
                print(f"    {s['channel']}.png")

        plot_worst_skew_grid(all_stats, measured_values_by_ch, SAVE_DIR)
        print("    worst_skew_grid.png")

        plot_recommendation_summary(all_stats, SAVE_DIR)
        print("    recommendation_summary.png")

    # ---- Interpretation guide ----
    _section("INTERPRETATION GUIDE")
    print("""
  Distribution Classes:
    normal-ish:       |skew| < 0.5, kurtosis < 3  → StandardScaler is fine
    moderate skew:    |skew| 0.5–2, kurtosis < 7   → PowerTransformer (Yeo-Johnson)
                      makes data more Gaussian while preserving monotonic relationships
    heavy skew/tails: |skew| >= 2 or kurtosis >= 7 → QuantileTransformer (→ normal)
                      forces exact normal output; handles extreme distributions

  Why this matters:
    StandardScaler only shifts and scales — a right-skewed distribution stays
    right-skewed. Extreme values (e.g., lactate=20) produce extreme z-scores
    (z=8+) that dominate gradient updates. With 80% sparsity, the model sees
    mostly 0.0 with occasional extreme spikes.

  What to do with these results:
    1. If most channels are "normal-ish": current StandardScaler is fine
    2. If many are "moderate skew": switch to PowerTransformer globally
    3. If mixed: use adaptive per-channel normalization
    4. Check the z>3 and z>5 columns — high percentages mean outlier influence

  Next step: share summary.csv and worst_skew_grid.png so we can decide
  on the best normalization strategy.
    """)


if __name__ == "__main__":
    main()
