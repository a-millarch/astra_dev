"""
Regenerate all paper-submission figures from saved eval artifacts — no model inference.

Reads CSVs written by ``run_eval`` and pickled calibrators, recomputes trauma scores
+ DeLong in-memory, then calls the same plotting functions used in production.
Each regeneration is isolated so a missing artifact skips that figure rather than
aborting the run. All outputs honour journal submission caps (longest side ≤ 1200 px,
≤ 5 MB) with auto-computed DPI unless ``--dpi`` is passed explicitly.

Covered (from saved artifacts):
    predictive performance
        time_metrics, n_active,
        time_metrics_comparison (plain + with trauma static lines),
        pred_distribution,
        baseline_eval, dca_baseline,
        multi_curves, dca_multicurve,
        trauma_rts_comparison, trauma_triss_comparison,
        delong_rts_comparison, delong_triss_comparison
    calibration
        calibration_analysis, reliability_diagrams, dca_comparison,
        dca_calibrated, per_timepoint_vs_global
    SHAP
        delegated to ``python -m astra.evaluation.shap_paper_figures --figures-only``

Not covered — methodology cannot be preserved without rerunning the eval pipeline:
    cm_F1, cm_F5  — production finds F-beta thresholds on calibrated TRAINVAL
        predictions and applies an inline isotonic calibrator to holdout. Neither
        the calibrator nor the threshold is persisted by run_eval; recomputing
        either from holdout would constitute a methodology change.
    multi_percentile_recall  — PercentileRecallResult list not saved
    shap_class1  — tied to a fresh SHAP computation in train.py

Usage:
    python -m astra.evaluation.replot_paper_figures                # default: all figures, auto-dpi
    python -m astra.evaluation.replot_paper_figures --suffix _rev1 # versioned filenames
    python -m astra.evaluation.replot_paper_figures --dpi 300      # fixed DPI (warn if over cap)
    python -m astra.evaluation.replot_paper_figures --skip-shap    # skip SHAP subprocess
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

from astra.utils import cfg, get_cfg, save_figure, setup_logging
from astra.data.caching import prepare_data_and_dls_cached

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Artifact loaders
# ═══════════════════════════════════════════════════════════════════════════

def _time_metrics_from_csv(path: str):
    from astra.evaluation.predictive_performance import TimeMetricResult
    df = pd.read_csv(path)
    results = []
    for _, r in df.iterrows():
        results.append(TimeMetricResult(
            time_min=float(r["time_min"]),
            time_hours=float(r["time_hours"]),
            time_days=float(r["time_days"]),
            censor_step=int(r["censor_step"]),
            auroc=float(r["auroc"]),
            auroc_ci=(float(r["auroc_ci_lower"]), float(r["auroc_ci_upper"])),
            auprc=float(r["auprc"]),
            auprc_ci=(float(r["auprc_ci_lower"]), float(r["auprc_ci_upper"])),
            n_samples=int(r["n_samples"]),
            n_positive=int(r["n_positive"]),
        ))
    return results


def _optional_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    logger.warning(f"Missing optional artifact: {path}")
    return None


def _build_timepoint_preds(preds_df_active: pd.DataFrame,
                           holdout_y: np.ndarray,
                           holdout_pids: np.ndarray):
    """Reconstruct ``{censor_step: TimepointPredictions}`` from active preds."""
    from astra.evaluation.posthoc_calibration import TimepointPredictions
    pid_to_y = dict(zip(holdout_pids, holdout_y))
    out: Dict[int, "TimepointPredictions"] = {}
    for step, group in preds_df_active.groupby("censor_step"):
        mask = group["PID"].isin(pid_to_y)
        group = group[mask]
        if len(group) == 0:
            continue
        y_true = np.array([pid_to_y[p] for p in group["PID"].values])
        y_prob = group["pred"].values.astype(float)
        out[int(step)] = TimepointPredictions(
            censor_step=int(step),
            time_hours=float(group["time_hours"].iloc[0]),
            y_true=y_true,
            y_prob=y_prob,
            n_samples=int(len(y_true)),
            n_positive=int(y_true.sum()),
        )
    return out


def _build_preds_matrix(preds_df: pd.DataFrame, holdout_pids: np.ndarray):
    """Reconstruct ``[N, seq_len]`` predictions matrix from long preds_df."""
    if preds_df.empty:
        return None, None
    seq_len = int(preds_df["censor_step"].max()) + 1
    pid_to_row = {p: i for i, p in enumerate(holdout_pids)}
    n = len(holdout_pids)
    mat = np.full((n, seq_len), np.nan)
    filled = np.zeros((n, seq_len), dtype=bool)
    for _, r in preds_df.iterrows():
        i = pid_to_row.get(r["PID"])
        if i is None:
            continue
        step = int(r["censor_step"])
        mat[i, step] = float(r["pred"])
        filled[i, step] = True
    # Trajectory length: last step where any prediction exists, +1
    traj_lengths = np.array([filled[i].nonzero()[0].max() + 1 if filled[i].any() else 0
                             for i in range(n)])
    return mat, traj_lengths


def _load_calibrators(calibrator_dir: str):
    """Load pickled per-timepoint and global calibrators from disk."""
    if not os.path.isdir(calibrator_dir):
        logger.warning(f"No calibrator directory at {calibrator_dir}")
        return None, None
    per_tp: Dict[int, Dict[str, object]] = {}
    globals_: Dict[str, object] = {}
    for fname in os.listdir(calibrator_dir):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(calibrator_dir, fname)
        with open(path, "rb") as f:
            cal = pickle.load(f)
        stem = fname[:-4]
        if stem.endswith("_global"):
            method = stem[: -len("_global")]
            globals_[method] = cal
        elif "_step" in stem:
            method, _, step_s = stem.partition("_step")
            try:
                step = int(step_s)
            except ValueError:
                continue
            per_tp.setdefault(step, {})[method] = cal
    if not per_tp and not globals_:
        return None, None
    return per_tp, globals_


# ═══════════════════════════════════════════════════════════════════════════
# Figure regenerators — each try/except isolated so one failure doesn't block others
# ═══════════════════════════════════════════════════════════════════════════

def _safe(label: str):
    """Context decorator: log + continue on exception instead of aborting."""
    def deco(fn):
        def wrapped(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except FileNotFoundError as e:
                logger.warning(f"[{label}] skipped — missing input: {e}")
            except Exception as e:
                logger.error(f"[{label}] failed: {e}", exc_info=True)
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


@_safe("time_metrics")
def _regen_time_metrics(results_all, out_dir, suffix, kw):
    from astra.evaluation.predictive_performance import plot_time_metrics
    fig = plot_time_metrics(results_all, cut_hours=72)
    save_figure(fig, f"time_metrics{suffix}", save_dir=out_dir, **kw)


@_safe("n_active")
def _regen_n_active(results_active, target, out_dir, suffix, kw):
    from astra.evaluation.predictive_performance import plot_n_active_over_time
    fig = plot_n_active_over_time(results_active, target_name=target)
    save_figure(fig, f"n_active{suffix}", save_dir=out_dir, **kw)


@_safe("time_metrics_comparison")
def _regen_tm_comparison(results_all, results_active, target, out_dir, suffix, kw,
                          static_scores=None, stem="time_metrics_comparison"):
    from astra.evaluation.predictive_performance import plot_time_metrics_comparison
    fig = plot_time_metrics_comparison(
        results_all, results_active,
        target_name=target, static_scores=static_scores,
    )
    save_figure(fig, f"{stem}{suffix}", save_dir=out_dir, **kw)


@_safe("pred_distribution")
def _regen_pred_distribution(preds_df_active, holdout_y, holdout_pids,
                              out_dir, suffix, kw):
    from astra.evaluation.predictive_performance import plot_prediction_distribution
    fig = plot_prediction_distribution(preds_df_active, holdout_y, holdout_pids)
    save_figure(fig, f"pred_distribution{suffix}", save_dir=out_dir, **kw)


@_safe("baseline_eval + dca_baseline")
def _regen_baseline(preds_df, holdout_y, holdout_pids, target, out_dir, suffix, kw):
    """Derive last-step predictions per patient from preds_df and plot baseline figures."""
    from astra.evaluation.predictive_performance import plot_decision_curve
    from astra.visualize.evaluation import plot_evaluation
    import torch

    # For each PID, take prediction at max censor_step present for that patient
    last_preds_df = (preds_df.sort_values("censor_step")
                              .groupby("PID").tail(1)
                              .set_index("PID"))
    pid_to_label = dict(zip(holdout_pids, holdout_y))
    aligned = [(pid, last_preds_df.at[pid, "pred"])
               for pid in holdout_pids if pid in last_preds_df.index]
    if len(aligned) < 10:
        logger.warning("Too few patients with baseline preds; skipping")
        return
    pids_aligned, preds_aligned = zip(*aligned)
    preds_np = np.array(preds_aligned, dtype=float)
    y_np = np.array([pid_to_label[p] for p in pids_aligned], dtype=float)

    fig_eval = plot_evaluation(torch.tensor(preds_np), torch.tensor(y_np), target)
    save_figure(fig_eval, f"baseline_eval{suffix}", save_dir=out_dir, **kw)

    fig_dca = plot_decision_curve(y_np, preds_np, model_name="HNN")
    save_figure(fig_dca, f"dca_baseline{suffix}", save_dir=out_dir, **kw)


@_safe("multi_curves + dca_multicurve")
def _regen_multicurve(preds_df, holdout_y, holdout_pids,
                       out_dir, suffix, kw):
    """Reuse the from-arrays helpers with predictions assembled from preds_df."""
    from astra.evaluation.predictive_performance import (
        _plot_roc_pr_curves_from_arrays,
        _plot_decision_curves_temporal,
        format_step_label,
        time_to_step,
        get_total_steps,
    )

    max_step = get_total_steps() - 1
    raw_timepoints = [
        time_to_step(1, "h"), time_to_step(6, "h"),
        time_to_step(12, "h"), time_to_step(72, "h"),
        time_to_step(7, "D"), time_to_step(14, "D"),
        time_to_step(30, "D"), time_to_step(90, "D"),
    ]
    key_timepoints = sorted({min(t, max_step) for t in raw_timepoints if t is not None})
    labels = [format_step_label(s) for s in key_timepoints]

    pid_to_y = dict(zip(holdout_pids, holdout_y))
    preds_list, targs_list, valid_steps, valid_labels = [], [], [], []
    for step, lbl in zip(key_timepoints, labels):
        grp = preds_df[preds_df["censor_step"] == step]
        # Filter to PIDs present in holdout set (preds_df can contain stragglers).
        grp = grp[grp["PID"].isin(pid_to_y)]
        if len(grp) < 10:
            continue
        y = np.array([pid_to_y[p] for p in grp["PID"].values])
        if len(set(y)) < 2:
            continue
        p = grp["pred"].values.astype(float)
        preds_list.append(p)
        targs_list.append(y)
        valid_steps.append(step)
        valid_labels.append(lbl)

    if preds_list:
        fig_mc = _plot_roc_pr_curves_from_arrays(valid_steps, preds_list, targs_list, valid_labels)
        save_figure(fig_mc, f"multi_curves{suffix}", save_dir=out_dir, **kw)

    preds_mat, traj_lens = _build_preds_matrix(preds_df, holdout_pids)
    if preds_mat is not None:
        fig_dca = _plot_decision_curves_temporal(
            preds_mat, holdout_y, traj_lens, key_timepoints, labels=labels,
        )
        save_figure(fig_dca, f"dca_multicurve{suffix}", save_dir=out_dir, **kw)


@_safe("trauma score + DeLong figures")
def _regen_trauma_and_delong(data, trauma_cfg, results_all, results_active,
                              preds_df_active, holdout_y, holdout_pids,
                              target, out_dir, suffix, kw):
    from astra.evaluation.trauma_scores import (
        build_trauma_score_df,
        evaluate_static_scores,
        evaluate_static_scores_over_time,
    )
    from astra.evaluation.predictive_performance import (
        plot_time_metrics_comparison,
        plot_trauma_score_comparison,
        plot_delong_comparison,
    )

    trauma_df = build_trauma_score_df(data, trauma_cfg)
    static_scores = evaluate_static_scores(trauma_df, holdout_y, holdout_pids)
    if static_scores and results_all and results_active:
        fig = plot_time_metrics_comparison(
            results_all, results_active, target_name=target, static_scores=static_scores,
        )
        save_figure(fig, f"time_metrics_comparison_trauma{suffix}",
                    save_dir=out_dir, **kw)

    rts_valid = trauma_df.dropna(subset=["RTS"])
    valid_pids = rts_valid["PID"].values
    if len(valid_pids) < 20:
        logger.warning(f"Only {len(valid_pids)} patients with RTS — skipping trauma comparisons")
        return

    score_results = evaluate_static_scores_over_time(
        trauma_df, preds_df_active, holdout_y, holdout_pids,
        valid_pids=valid_pids, delong=True,
    )
    for sname, paired in score_results.items():
        if sname == "ISS":
            continue
        fig_trauma = plot_trauma_score_comparison(
            sname, paired, target_name=target, results_counts=paired.get("counts"),
        )
        save_figure(fig_trauma, f"trauma_{sname.lower()}_comparison{suffix}",
                    save_dir=out_dir, **kw)
        if "delong_significant" in paired:
            fig_dl = plot_delong_comparison(sname, paired)
            save_figure(fig_dl, f"delong_{sname.lower()}_comparison{suffix}",
                        save_dir=out_dir, **kw)


@_safe("calibration figures")
def _regen_calibration(model_name, holdout_preds, out_dir, suffix):
    """Reload pickled calibrators, apply to holdout preds, replot the 4 calibration figures.

    The calibration plot helpers internally apply their own ``_SUBMISSION_KW``
    (auto-DPI to fit 1200 px) via ``save_figure`` — this path intentionally ignores
    an explicit ``--dpi`` override. Use the standard save_figure path for everything else.
    """
    from astra.evaluation.posthoc_calibration import (
        apply_calibrator,
        _plot_calibration_metrics_over_time,
        _plot_reliability_diagrams,
        _plot_dca_comparison,
        _plot_dca_calibrated,
        _plot_per_timepoint_vs_global,
    )
    from astra.evaluation.utils import time_to_step, get_total_steps

    calibrator_dir = f"models/calibrators/{model_name}"
    per_tp, globals_ = _load_calibrators(calibrator_dir)
    if per_tp is None:
        logger.warning("No calibrators on disk — skipping calibration figures")
        return

    methods = sorted({m for methods in per_tp.values() for m in methods.keys()})
    if not methods:
        return

    # Filter to the same 8 key timepoints production uses — otherwise the grid
    # helpers build a (ncols * 3.5, nrows * 3.5) figure that grows unboundedly.
    # Max 14 days — matches the original production output. The 30D/90D
    # timepoints have too few non-survivors active (min_positive filter dropped
    # them in the original run), so we omit them upfront here for consistency.
    max_step = get_total_steps() - 2
    raw_key = [
        time_to_step(1, 'h'), time_to_step(6, 'h'), time_to_step(12, 'h'),
        time_to_step(72, 'h'), time_to_step(7, 'D'), time_to_step(14, 'D'),
    ]
    key_timepoints = sorted({min(t, max_step) for t in raw_key if t is not None})
    holdout_preds = {s: tp for s, tp in holdout_preds.items() if s in key_timepoints}
    if not holdout_preds:
        logger.warning("No predictions at key calibration timepoints — skipping")
        return

    # Build calibrated_preds dict matching what the plot helpers expect
    calibrated: Dict[int, Dict[str, np.ndarray]] = {}
    for step, tp in holdout_preds.items():
        calibrated[step] = {}
        for method in methods:
            cal = per_tp.get(step, {}).get(method) or globals_.get(method)
            if cal is None:
                continue
            calibrated[step][method] = apply_calibrator(cal, tp.y_prob, method)

    # Best method: fall back to first (calibration_summary CSV would tell us
    # definitively, but picking any present method still lets us replot)
    summary_path = f"reports/eval/{model_name}/calibration/calibration_summary_{model_name}.csv"
    best_method = methods[0]
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            per_tp_rows = df[df["calibrator_type"] == "per_timepoint"]
            if len(per_tp_rows):
                best_method = (per_tp_rows.groupby("method")["ece_reduction"]
                                           .mean().idxmax())
        except Exception as e:
            logger.debug(f"Could not parse calibration summary: {e}")

    # Plot helpers hardcode filenames as f"{stem}_{model_name}.{ext}". We pass the
    # bare model_name so the first part matches disk layout; then rename outputs
    # so they line up with the predictive-perf naming convention ``{stem}{suffix}``.
    _plot_reliability_diagrams(holdout_preds, calibrated, best_method, model_name, out_dir)
    _plot_dca_comparison(holdout_preds, calibrated, best_method, model_name, out_dir)
    _plot_dca_calibrated(holdout_preds, calibrated, best_method, model_name, out_dir)

    calibration_stems = ["reliability_diagrams", "dca_comparison", "dca_calibrated"]

    # calibration_analysis + per_timepoint_vs_global both read from a
    # CalibratorResult list. Reconstruct once from the summary CSV and call both.
    if os.path.exists(summary_path):
        try:
            from astra.evaluation.posthoc_calibration import CalibratorResult
            df = pd.read_csv(summary_path)
            all_results = [
                CalibratorResult(
                    censor_step=int(r["censor_step"]),
                    time_hours=float(r["time_hours"]),
                    time_label=str(r["time_label"]),
                    method=str(r["method"]),
                    calibrator_type=str(r["calibrator_type"]),
                    ece_raw=float(r["ece_raw"]), ece_cal=float(r["ece_cal"]),
                    brier_raw=float(r["brier_raw"]), brier_cal=float(r["brier_cal"]),
                    auroc_raw=float(r["auroc_raw"]), auroc_cal=float(r["auroc_cal"]),
                    auprc_raw=float(r["auprc_raw"]), auprc_cal=float(r["auprc_cal"]),
                    n_samples=int(r["n_samples"]), n_positive=int(r["n_positive"]),
                )
                for _, r in df.iterrows()
            ]
            _plot_calibration_metrics_over_time(
                all_results, methods, model_name, out_dir,
            )
            calibration_stems.append("calibration_analysis")
            _plot_per_timepoint_vs_global(all_results, best_method, model_name, out_dir)
            calibration_stems.append("per_timepoint_vs_global")
        except Exception as e:
            logger.warning(f"calibration_analysis / per_timepoint_vs_global skipped: {e}")

    # Rename to match {stem}{suffix} convention (e.g. reliability_diagrams_rev20260430)
    if suffix:
        import shutil
        for stem in calibration_stems:
            src = os.path.join(out_dir, f"{stem}_{model_name}.png")
            dst = os.path.join(out_dir, f"{stem}{suffix}.png")
            if os.path.exists(src):
                shutil.move(src, dst)
            # Also move base64 sidecar
            src_b64 = os.path.join(out_dir, "base64", f"{stem}_{model_name}_base64.txt")
            dst_b64 = os.path.join(out_dir, "base64", f"{stem}{suffix}_base64.txt")
            if os.path.exists(src_b64):
                shutil.move(src_b64, dst_b64)


@_safe("SHAP paper figures")
def _regen_shap(skip: bool, config: Optional[str]):
    """Invoke the existing shap_paper_figures --figures-only entry point."""
    if skip:
        logger.info("SHAP replot skipped (--skip-shap)")
        return
    logger.info("Delegating SHAP replot to shap_paper_figures --figures-only")
    cmd = [sys.executable, "-m", "astra.evaluation.shap_paper_figures", "--figures-only"]
    if config:
        cmd.extend(["--config", config])
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.warning(
            f"shap_paper_figures returned non-zero exit code {result.returncode}. "
            f"If you see an IndexError about 'axis 1', the cached shap_cache.pkl "
            f"was computed against a different config (different channel count) "
            f"than the one currently loaded. Pass --config <same-as-original-run> "
            f"to both this script and the SHAP replot."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def replot(
    model_name: str,
    suffix: str,
    dpi: Optional[int],
    output_dir: Optional[str],
    skip_shap: bool,
    config: Optional[str] = None,
) -> None:
    preds_dir = f"reports/eval/{model_name}/predictions"
    out_dir = output_dir or f"reports/eval/{model_name}/revision{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    # Submission kwargs: explicit dpi if given, else auto-fit to 1200 px
    if dpi is None:
        submission_kw = dict(
            fit_long_side_px=1200, max_long_side_px=1200, max_bytes=5_000_000,
        )
        logger.info("Auto-DPI: computing highest DPI that fits 1200 px long side")
    else:
        submission_kw = dict(
            dpi=dpi, max_long_side_px=1200, max_bytes=5_000_000,
        )
        logger.info(f"Fixed DPI: {dpi}")

    logger.info(f"Reloading saved artifacts from {preds_dir}")
    preds_df = _optional_csv(f"{preds_dir}/preds_df_{model_name}.csv")
    preds_df_active = _optional_csv(f"{preds_dir}/preds_df_{model_name}_active.csv")
    results_all_csv = f"{preds_dir}/time_metrics_{model_name}.csv"
    results_active_csv = f"{preds_dir}/time_metrics_{model_name}_active.csv"
    results_all = _time_metrics_from_csv(results_all_csv) if os.path.exists(results_all_csv) else None
    results_active = _time_metrics_from_csv(results_active_csv) if os.path.exists(results_active_csv) else None

    if preds_df is None and results_all is None:
        raise FileNotFoundError(
            f"Neither predictions nor time metrics CSVs found under {preds_dir}. "
            f"Ensure run_eval has been executed for model '{model_name}'."
        )

    logger.info("Loading data dict (cached; no model inference)")
    data = prepare_data_and_dls_cached(cfg)
    holdout_pids = data["holdout"].base.PID.values
    holdout_y = np.array(data["ty"])
    target = cfg["target"]

    # ── Predictive performance ──────────────────────────────────────────
    if results_all:
        _regen_time_metrics(results_all, out_dir, suffix, submission_kw)
    if results_active:
        _regen_n_active(results_active, target, out_dir, suffix, submission_kw)
    if results_all and results_active:
        _regen_tm_comparison(results_all, results_active, target,
                             out_dir, suffix, submission_kw)
    if preds_df_active is not None:
        _regen_pred_distribution(preds_df_active, holdout_y, holdout_pids,
                                  out_dir, suffix, submission_kw)
    if preds_df is not None:
        _regen_baseline(preds_df, holdout_y, holdout_pids, target,
                        out_dir, suffix, submission_kw)
    if preds_df is not None:
        _regen_multicurve(preds_df, holdout_y, holdout_pids,
                          out_dir, suffix, submission_kw)

    # ── Trauma scores + DeLong ──────────────────────────────────────────
    if results_all and results_active and preds_df_active is not None:
        _regen_trauma_and_delong(
            data, cfg, results_all, results_active, preds_df_active,
            holdout_y, holdout_pids, target, out_dir, suffix, submission_kw,
        )

    # ── Calibration ─────────────────────────────────────────────────────
    if preds_df_active is not None:
        holdout_preds = _build_timepoint_preds(preds_df_active, holdout_y, holdout_pids)
        _regen_calibration(model_name, holdout_preds, out_dir, suffix)

    # ── SHAP ────────────────────────────────────────────────────────────
    _regen_shap(skip_shap, config)

    logger.info(f"Replot complete. Output: {out_dir}")
    logger.info(
        "Not regenerated — methodology cannot be preserved from saved artifacts "
        "alone. Rerun `python -m astra.training.train --eval --comprehensive-eval "
        "[--calibrate] [--shap]` to regenerate these with the updated formatting:"
    )
    logger.info(
        "  - cm_F1, cm_F5: production finds the F-beta threshold on calibrated "
        "TRAINVAL predictions and applies an inline isotonic calibrator to holdout. "
        "Neither the calibrator nor the threshold is persisted; recomputing either "
        "from holdout would constitute a methodology change."
    )
    logger.info(
        "  - multi_percentile_recall: PercentileRecallResult list is only held in "
        "memory during the eval run."
    )
    logger.info(
        "  - shap_class1 (from train.py): tied to a fresh SHAP computation inside "
        "the training pipeline."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate paper-submission figures from saved eval artifacts.",
    )
    parser.add_argument(
        "--config", type=str, default="defaults.yaml",
        help="Config YAML filename in configs/ dir (default: defaults.yaml). "
             "Must match the config used during the original eval run. "
             "Passed through to the SHAP replot subprocess.",
    )
    parser.add_argument("--model-name", type=str, default=None,
                        help="Defaults to cfg['model_name'].")
    parser.add_argument("--suffix", type=str, default="",
                        help='Filename suffix, e.g. "_rev20260430".')
    parser.add_argument(
        "--dpi", type=int, default=None,
        help="Explicit output DPI. If omitted, auto-computes the highest DPI "
             "that keeps the tight-bbox output within 1200 px on the long side.",
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output dir; defaults to "
                             "reports/eval/<model>/revision<suffix>/")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip the SHAP replot subprocess.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Load config from configs/ dir (mutate in place so imported references stay valid)
    import astra.utils as _utils
    _cfg = get_cfg(_utils.PROJECT_ROOT / "configs" / args.config)
    _utils.cfg.clear()
    _utils.cfg.update(_cfg)

    model_name = args.model_name or cfg["model_name"]
    replot(
        model_name=model_name,
        suffix=args.suffix,
        dpi=args.dpi,
        output_dir=args.output_dir,
        skip_shap=args.skip_shap,
        config=args.config,
    )


if __name__ == "__main__":
    main()
