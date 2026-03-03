"""
Diagnostic script for SHAP anomalies with temporal head + EBM injection.

Three checks:
  1. Per-timestep model predictions — does the model output near-zero probs early?
  2. EBM channel over time — when does it first become non-zero?
  3. Gradient test — compare |grad| profile for eval at position -1 vs last valid step.
     (Cheaper proxy for SHAP; reveals recency bias from Bug #1.)

Usage:
    python scripts/diagnose_shap_temporal.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg" / remove for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from astra.utils import get_cfg
from astra.data.caching import prepare_data_and_dls_cached
from astra.evaluation.utils import prepare_learner, step_to_time
from astra.evaluation.behavior import extract_data_from_dataloader, create_channel_mapping


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_PATIENTS       = 8          # patients to plot in diagnostics 1 & 2
N_GRAD_SAMPLES   = 32         # samples for gradient profile (diag 3) — keep small, slow
OUTPUT_DIR       = "outputs/diagnose_shap_temporal"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _steps_to_hours(steps):
    """Convert array of step indices to hours (best-effort via step_to_time)."""
    hours = []
    for s in steps:
        t_min = step_to_time(int(s))
        hours.append(t_min / 60.0 if t_min is not None else float(s))
    return np.array(hours)


def _get_holdout_batches(data, n_samples, device):
    """Extract up to n_samples from holdout dataloader."""
    return extract_data_from_dataloader(
        data["holdout_mixed_dls"], max_samples=n_samples, device=device
    )


# ---------------------------------------------------------------------------
# Diagnostic 1: Per-timestep model predictions
# ---------------------------------------------------------------------------

def diag1_per_timestep_predictions(model, data, device, n_patients=N_PATIENTS):
    _section("DIAGNOSTIC 1 — Per-timestep model predictions")
    print(f"  Plotting per-timestep sigmoid(logit) for {n_patients} holdout patients.")
    print("  If probabilities are near-zero early and spike only at the end,")
    print("  the model is not learning from early data (EBM or gradient issue).\n")

    if not (hasattr(model, 'temporal_head_enabled') and model.temporal_head_enabled):
        print("  SKIP: model does not have a temporal head (not applicable).")
        return

    x_ts, x_ts_cat, x_cat, x_cont, y, _ = _get_holdout_batches(
        data, n_samples=n_patients, device=device
    )
    traj_lengths = data["holdout_trajectory_lengths"][:n_patients]
    seq_len = x_ts.shape[2]

    # Build time axis (hours)
    time_ax = _steps_to_hours(np.arange(seq_len))

    model.eval()
    with torch.no_grad():
        # Build the packed input tuple that model.forward() expects
        x_tab = (x_cat, x_cont)
        logits = model((x_ts, x_tab, x_ts_cat))   # [batch, seq_len]
        probs  = torch.sigmoid(logits).cpu().numpy()

    n_cols = 2
    n_rows = (n_patients + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), squeeze=False)
    fig.suptitle("Per-timestep predicted P(mortality) [temporal head]", fontsize=13, y=1.01)

    y_np = y.cpu().numpy().astype(int)

    for i in range(n_patients):
        ax = axes[i // n_cols][i % n_cols]
        tlen = int(traj_lengths[i]) if i < len(traj_lengths) else seq_len
        label_str = "DECEASED" if y_np[i] else "SURVIVED"
        color     = "crimson"  if y_np[i] else "steelblue"

        # Full trajectory (including padding — should stay near 0)
        ax.plot(time_ax, probs[i], color="gray", linewidth=0.8, alpha=0.5, label="all steps")
        # Valid trajectory only
        ax.plot(time_ax[:tlen], probs[i, :tlen], color=color, linewidth=1.5, label=f"valid ({label_str})")
        # Mark trajectory end
        if tlen < seq_len:
            ax.axvline(x=time_ax[tlen - 1], color="black", linestyle="--", linewidth=0.8,
                       alpha=0.7, label=f"traj end ({time_ax[tlen-1]:.1f}h)")

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Hours since admission")
        ax.set_ylabel("P(mortality)")
        ax.set_title(f"Patient {i} (traj={tlen} steps, {time_ax[min(tlen,seq_len)-1]:.1f}h)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    for i in range(n_patients, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "diag1_per_timestep_predictions.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    # Summary stats
    probs_valid = []
    for i in range(n_patients):
        tlen = int(traj_lengths[i]) if i < len(traj_lengths) else seq_len
        probs_valid.append(probs[i, :tlen])

    early_cutoff = max(1, seq_len // 4)   # first 25% of seq
    print(f"\n  Summary (first {n_patients} holdout patients):")
    for i, pv in enumerate(probs_valid):
        early_mean = pv[:early_cutoff].mean() if len(pv) > 0 else float('nan')
        late_mean  = pv[max(0, len(pv) - early_cutoff):].mean() if len(pv) > 0 else float('nan')
        print(f"    Patient {i}: early_25%_mean={early_mean:.4f}  "
              f"late_25%_mean={late_mean:.4f}  label={int(y_np[i])}")


# ---------------------------------------------------------------------------
# Diagnostic 2: EBM channel over time
# ---------------------------------------------------------------------------

def diag2_ebm_channel(data, channel2feature, n_patients=N_PATIENTS):
    _section("DIAGNOSTIC 2 — EBM channel values over time")

    # Prefer the tracked index; fall back to scanning channel2feature for _ebm_pred
    # (ebm_channel_idx can be None in cached data even when the channel exists)
    ebm_idx = data.get("ebm_channel_idx")
    if ebm_idx is None:
        feature2channel = {v: k for k, v in channel2feature.items()}
        ebm_idx = feature2channel.get("_ebm_pred")
    if ebm_idx is None:
        print("  SKIP: _ebm_pred not found in channel2feature (EBM not injected).")
        return

    ebm_name = channel2feature.get(ebm_idx, "_ebm_pred")
    print(f"  EBM channel: index={ebm_idx}, name={ebm_name}")

    tX_raw   = data["tX_raw"]    # [n_holdout, n_channels, seq_len]  — raw (unscaled)
    tX_norm  = data["tX"]        # [n_holdout, n_channels, seq_len]  — normalized
    traj_lengths = np.array(data["holdout_trajectory_lengths"])
    seq_len  = tX_raw.shape[2]
    n_act    = min(n_patients, tX_raw.shape[0])

    time_ax = _steps_to_hours(np.arange(seq_len))

    fig, axes = plt.subplots(n_act, 2, figsize=(14, 3.0 * n_act), squeeze=False)
    fig.suptitle(f"EBM channel ({ebm_name}) — raw (left) and normalized (right)", fontsize=12)

    # Overall stats: when does EBM first become non-zero?
    first_nonzero_steps = []
    for i in range(tX_raw.shape[0]):
        tlen = int(traj_lengths[i])
        vals = tX_raw[i, ebm_idx, :tlen]
        nz = np.where(vals != 0.0)[0]
        first_nonzero_steps.append(nz[0] if len(nz) > 0 else None)

    valid_first = [s for s in first_nonzero_steps if s is not None]
    if valid_first:
        median_step = int(np.median(valid_first))
        print(f"\n  EBM first non-zero step across holdout:")
        print(f"    Patients with EBM data: {len(valid_first)}/{len(first_nonzero_steps)} "
              f"({100*len(valid_first)/len(first_nonzero_steps):.1f}%)")
        print(f"    Median first non-zero step: {median_step}  "
              f"({_steps_to_hours([median_step])[0]:.1f}h)")
        print(f"    P5/P25/P75/P95 first non-zero step: "
              f"{np.percentile(valid_first, [5,25,75,95])}")
    else:
        print("  WARNING: EBM is all-zero for every holdout patient!")

    for i in range(n_act):
        tlen = int(traj_lengths[i])
        raw  = tX_raw[i,  ebm_idx, :]
        norm = tX_norm[i, ebm_idx, :]

        for ax, vals, title in [
            (axes[i][0], raw,  "raw"),
            (axes[i][1], norm, "normalized"),
        ]:
            ax.plot(time_ax, vals, color="gray", linewidth=0.7, alpha=0.5)
            ax.plot(time_ax[:tlen], vals[:tlen], color="darkorange", linewidth=1.5)
            # Mark first non-zero
            fn = first_nonzero_steps[i]
            if fn is not None:
                ax.axvline(x=time_ax[fn], color="green", linestyle=":", linewidth=1.2,
                           label=f"1st EBM@{time_ax[fn]:.1f}h")
                ax.legend(fontsize=7)
            # Mark traj end
            if tlen < seq_len:
                ax.axvline(x=time_ax[tlen - 1], color="black", linestyle="--", linewidth=0.8,
                           alpha=0.6)
            ax.set_xlabel("Hours since admission")
            ax.set_ylabel(f"EBM ({title})")
            ax.set_title(f"Patient {i} [{title}]  traj={tlen}steps")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "diag2_ebm_channel.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {path}")

    # Additional: plot mean EBM value over time across all holdout patients
    # (aligned by step index, not hours — shows population-level availability)
    mean_raw  = np.zeros(seq_len)
    mean_norm = np.zeros(seq_len)
    pct_nonzero = np.zeros(seq_len)
    n_total = tX_raw.shape[0]

    for step in range(seq_len):
        vals_r = tX_raw[:, ebm_idx, step]
        vals_n = tX_norm[:, ebm_idx, step]
        # Only include patients whose trajectory reaches this step
        alive_mask = traj_lengths > step
        if alive_mask.sum() > 0:
            mean_raw[step]   = vals_r[alive_mask].mean()
            mean_norm[step]  = vals_n[alive_mask].mean()
            pct_nonzero[step] = (vals_r[alive_mask] != 0).mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, vals, title, color in [
        (axes[0], mean_raw,    "Mean raw EBM value",          "darkorange"),
        (axes[1], mean_norm,   "Mean normalized EBM value",   "steelblue"),
        (axes[2], pct_nonzero, "% patients with EBM > 0",     "green"),
    ]:
        ax.plot(time_ax, vals, color=color)
        ax.set_xlabel("Hours since admission")
        ax.set_ylabel(title)
        ax.set_title(title + " [by step, patients alive]")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Population-level EBM availability (holdout)", fontsize=12)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "diag2_ebm_population.png")
    plt.savefig(path2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path2}")


# ---------------------------------------------------------------------------
# Diagnostic 3: Gradient profile — eval at -1 vs last valid step
# ---------------------------------------------------------------------------

def diag3_gradient_profile(model, data, channel2feature, device, n_samples=N_GRAD_SAMPLES):
    _section("DIAGNOSTIC 3 — Gradient magnitude profile (cheap SHAP proxy)")
    print("  Computes mean |∂output/∂x_ts| across timesteps.")
    print("  Compares eval at position -1 (Bug #1) vs last VALID step per patient.")
    print("  If gradient collapses to zero early for -1 but not for last-valid,")
    print("  Bug #1 (eval_timestep=-1 recency bias) is the dominant cause.\n")

    if not (hasattr(model, 'temporal_head_enabled') and model.temporal_head_enabled):
        print("  SKIP: model does not have a temporal head (not applicable).")
        return

    x_ts, x_ts_cat, x_cat, x_cont, y, _ = _get_holdout_batches(
        data, n_samples=n_samples, device=device
    )
    traj_lengths = np.array(data["holdout_trajectory_lengths"][:n_samples])
    n_actual = x_ts.shape[0]
    traj_lengths = traj_lengths[:n_actual]
    seq_len  = x_ts.shape[2]
    n_channels = x_ts.shape[1]

    model.eval()
    x_tab = (x_cat, x_cont)

    def compute_mean_grad_profile(eval_step_fn):
        """
        eval_step_fn(i) -> int : which output position to evaluate for patient i.
        Returns mean |grad| per timestep, averaged over patients.
        """
        grad_profiles = []

        for i in range(n_actual):
            # leaf holds the gradient; non-leaf clone is passed to model so that
            # the in-place x[mask]=0 inside _key_padding_mask doesn't error
            xi_ts_leaf = x_ts[i:i+1].detach().clone()
            xi_ts_leaf.requires_grad_(True)
            xi_ts_for_model = xi_ts_leaf.clone()   # non-leaf; grad flows to xi_ts_leaf

            xi_ts_cat_i = x_ts_cat[i:i+1].detach().clone()
            xi_cat  = x_cat[i:i+1].detach().clone()
            xi_cont = x_cont[i:i+1].detach().clone()
            xi_tab  = (xi_cat, xi_cont)

            logits = model((xi_ts_for_model, xi_tab, xi_ts_cat_i))   # [1, seq_len]
            step = eval_step_fn(i)
            scalar = logits[0, step]
            scalar.backward()

            grad = xi_ts_leaf.grad.detach().cpu().numpy()[0]   # [c_in, seq_len]
            profile = np.abs(grad).mean(axis=0)                 # [seq_len]
            grad_profiles.append(profile)

        return np.stack(grad_profiles, axis=0).mean(axis=0)  # [seq_len]

    # --- Profile A: always evaluate at position -1 ---
    print("  Computing gradient profile A: eval at position -1 ...")
    prof_last = compute_mean_grad_profile(lambda i: -1)

    # --- Profile B: evaluate at last VALID step per patient ---
    print("  Computing gradient profile B: eval at last valid step per patient ...")
    def last_valid_step(i):
        tlen = int(traj_lengths[i]) if i < len(traj_lengths) else seq_len
        return max(0, tlen - 1)

    prof_valid = compute_mean_grad_profile(last_valid_step)

    # --- Profile C (optional): fixed clinical timepoint, e.g. 24h ---
    step_24h = None
    try:
        from astra.evaluation.utils import time_to_step
        step_24h = time_to_step(24, 'h')
    except Exception:
        pass

    if step_24h is not None and step_24h < seq_len:
        print(f"  Computing gradient profile C: eval at 24h (step={step_24h}) ...")
        prof_24h = compute_mean_grad_profile(lambda i: step_24h)
    else:
        prof_24h = None
        print("  Skipping profile C (24h step not resolvable).")

    # --- Plot ---
    time_ax = _steps_to_hours(np.arange(seq_len))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all profiles overlaid
    ax = axes[0]
    ax.plot(time_ax, prof_last,  label="eval @ -1 (Bug #1)", color="crimson",   linewidth=1.8)
    ax.plot(time_ax, prof_valid, label="eval @ last valid",  color="steelblue",  linewidth=1.8)
    if prof_24h is not None:
        ax.plot(time_ax, prof_24h, label=f"eval @ 24h (step {step_24h})",
                color="seagreen", linewidth=1.8)
    ax.set_xlabel("Hours since admission")
    ax.set_ylabel("Mean |∂output / ∂x_ts| (over channels & patients)")
    ax.set_title("Gradient magnitude profile\n(proxy for SHAP — higher = more influence)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: ratio valid/last — shows how much recency bias -1 adds
    ratio = np.divide(prof_valid, prof_last + 1e-15)
    ax2 = axes[1]
    ax2.plot(time_ax, ratio, color="purple", linewidth=1.5)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Hours since admission")
    ax2.set_ylabel("Ratio: grad(last valid) / grad(-1)")
    ax2.set_title("Recency bias amplification\nRatio >> 1 = -1 severely underweights early steps")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Gradient profiles (n={n_actual} holdout patients)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "diag3_gradient_profiles.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {path}")

    # --- Per-channel gradient importance at specific steps ---
    print("\n  Computing per-channel gradient importance at key steps ...")
    def compute_channel_importance(eval_step_fn):
        """Returns mean |grad| per channel, averaged over patients."""
        channel_grads = []
        for i in range(n_actual):
            xi_ts_leaf = x_ts[i:i+1].detach().clone()
            xi_ts_leaf.requires_grad_(True)
            xi_ts_for_model = xi_ts_leaf.clone()   # non-leaf

            xi_ts_cat_i = x_ts_cat[i:i+1].detach().clone()
            xi_cat  = x_cat[i:i+1].detach().clone()
            xi_cont = x_cont[i:i+1].detach().clone()
            xi_tab  = (xi_cat, xi_cont)
            logits = model((xi_ts_for_model, xi_tab, xi_ts_cat_i))
            step = eval_step_fn(i)
            logits[0, step].backward()
            grad = xi_ts_leaf.grad.detach().cpu().numpy()[0]   # [c_in, seq_len]
            channel_grads.append(np.abs(grad).mean(axis=1))    # [c_in]
        return np.stack(channel_grads).mean(axis=0)             # [c_in]

    ch_imp_last  = compute_channel_importance(lambda i: -1)
    ch_imp_valid = compute_channel_importance(last_valid_step)

    # Get channel names from channel2feature (index → feature name)
    ch_names = [channel2feature.get(i, f"ch_{i}") for i in range(n_channels)]

    # Sort by importance under eval@last
    sort_idx = np.argsort(ch_imp_last)[::-1]
    top_n = min(20, n_channels)
    sort_idx = sort_idx[:top_n]

    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(top_n)
    width = 0.35
    ax.bar(x_pos - width/2, ch_imp_last[sort_idx],  width, label="eval @ -1",         color="crimson",  alpha=0.8)
    ax.bar(x_pos + width/2, ch_imp_valid[sort_idx], width, label="eval @ last valid", color="steelblue", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ch_names[j] if j < len(ch_names) else f"ch{j}" for j in sort_idx],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |gradient| (mean over time & patients)")
    ax.set_title(f"Per-channel gradient importance (top {top_n} by eval@-1)\n"
                 "If EBM dominates here, the model relies on EBM")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "diag3_channel_importance.png")
    plt.savefig(path2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path2}")

    # Print summary table
    print(f"\n  Top-10 channels by gradient importance (eval @ -1):")
    print(f"  {'Channel':<28s} {'grad@-1':>10s} {'grad@valid':>12s} {'ratio':>8s}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*8}")
    for j in sort_idx[:10]:
        name = ch_names[j] if j < len(ch_names) else f"ch_{j}"
        g1 = ch_imp_last[j]
        gv = ch_imp_valid[j]
        r  = gv / (g1 + 1e-15)
        print(f"  {name:<28s} {g1:>10.5f} {gv:>12.5f} {r:>8.2f}")

    print()
    print("  NOTE — Why gradient ranking != SHAP ranking for EBM:")
    print("    Gradient (this table): mean|∂output/∂x| averaged over ALL timesteps.")
    print("      EBM gradient is non-zero even at early steps (attention is global),")
    print("      so its per-step gradient is diluted when averaged over 600+ steps → rank ~5.")
    print("    SHAP = (x − x_background) × gradient.")
    print("      At early steps x_ebm ≈ 0 and x_bg_ebm ≈ 0 → (x−bg) ≈ 0 → SHAP ≈ 0.")
    print("      At late steps x_ebm ≠ 0 AND gradient is inflated by Bug #1 (eval@-1).")
    print("      → EBM's SHAP concentrates at the same late steps where Bug #1 is strongest,")
    print("        pushing it to rank #1 in SHAP despite rank #5 in raw gradient.")
    print("    Fix: use a meaningful eval_timestep (e.g. 24h) to break the compounding.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = get_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data ...")
    data = prepare_data_and_dls_cached(cfg)
    print(f"  Holdout X shape: {data['tX'].shape}  "
          f"(n_channels={data['tX'].shape[1]}, seq_len={data['tX'].shape[2]})")
    print(f"  EBM channel idx: {data.get('ebm_channel_idx')}")

    print("Loading model ...")
    learn = prepare_learner(data, cfg)
    model = learn.model
    model.eval()

    channel2feature, _ = create_channel_mapping(data)
    print(f"  Model: temporal_head={model.temporal_head_enabled}, causal={model.causal}")
    print(f"  Channels ({len(channel2feature)}): { {i: channel2feature[i] for i in sorted(channel2feature)} }")
    print(f"  ebm_channel_idx in data dict: {data.get('ebm_channel_idx')}  "
          f"(resolved via channel2feature: {next((i for i,n in channel2feature.items() if n=='_ebm_pred'), None)})")

    diag1_per_timestep_predictions(model, data, device, n_patients=N_PATIENTS)
    diag2_ebm_channel(data, channel2feature, n_patients=N_PATIENTS)
    diag3_gradient_profile(model, data, channel2feature, device, n_samples=N_GRAD_SAMPLES)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("\nInterpretation guide:")
    print("  Diag 1 — If probs are near 0 for most of the trajectory and spike at end,")
    print("           the model isn't learning from early data (EBM or causal head issue).")
    print("  Diag 2 — If EBM is 0 for the first N hours and the prob spike in Diag 1")
    print("           aligns with when EBM becomes non-zero, EBM dominance is the cause.")
    print("  Diag 3 — If 'eval@-1' profile has near-zero gradients early but 'eval@last valid'")
    print("           does not, Bug #1 (eval_timestep=-1) is the dominant SHAP issue.")
    print("           If BOTH profiles are near-zero early, it's a model issue (EBM dominance).")


if __name__ == "__main__":
    main()
