"""
Regenerate the four paper-revision figures from already-saved evaluation artifacts.

Reads the CSVs written by ``run_eval`` (predictions + serialized TimeMetricResult) and
recomputes the trauma-score/DeLong context in-memory (no model inference), then calls
the same plotting functions used in production. Use this to restyle figures for
journal revisions without rerunning the full eval.

Figures produced:
    - time_metrics_comparison{suffix}.png
    - time_metrics_comparison_trauma{suffix}.png    (when --trauma-scores)
    - delong_rts_comparison{suffix}.png             (when --trauma-scores)
    - delong_triss_comparison{suffix}.png           (when --trauma-scores)
    - pred_distribution{suffix}.png

Usage:
    python -m astra.evaluation.replot_paper_figures \
        --model-name <model> \
        --suffix _rev20260430 \
        --dpi 300 \
        --trauma-scores
"""

import argparse
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from astra.utils import cfg, save_figure, setup_logging
from astra.data.caching import prepare_data_and_dls_cached
from astra.evaluation.predictive_performance import (
    TimeMetricResult,
    plot_delong_comparison,
    plot_prediction_distribution,
    plot_time_metrics_comparison,
)

logger = logging.getLogger(__name__)


def _time_metrics_from_csv(path: str) -> List[TimeMetricResult]:
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


def _require(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def replot(
    model_name: str,
    suffix: str,
    dpi: int,
    output_dir: Optional[str],
    trauma_scores: bool,
) -> None:
    preds_dir = f"reports/eval/{model_name}/predictions"
    out_dir = output_dir or f"reports/eval/{model_name}/revision{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    submission_kw = dict(dpi=dpi, max_long_side_px=1200, max_bytes=5_000_000)

    logger.info(f"Loading saved eval artifacts from {preds_dir}")
    preds_df = pd.read_csv(_require(
        f"{preds_dir}/preds_df_{model_name}.csv", "all-patient predictions"
    ))
    preds_df_active = pd.read_csv(_require(
        f"{preds_dir}/preds_df_{model_name}_active.csv", "active-only predictions"
    ))
    results_all = _time_metrics_from_csv(_require(
        f"{preds_dir}/time_metrics_{model_name}.csv", "all-patient time metrics"
    ))
    results_active = _time_metrics_from_csv(_require(
        f"{preds_dir}/time_metrics_{model_name}_active.csv", "active-only time metrics"
    ))
    logger.info(
        f"Loaded {len(results_all)} all-patient and {len(results_active)} active-only "
        f"time points; {len(preds_df_active)} active prediction rows"
    )

    logger.info("Loading data dict (cached; no model inference)")
    data = prepare_data_and_dls_cached(cfg)
    holdout_pids = data["holdout"].base.PID.values
    holdout_y = np.array(data["ty"])

    # ── Figure 1: time_metrics_comparison (plain, no trauma baselines) ──
    fig_cmp = plot_time_metrics_comparison(
        results_all, results_active, target_name=cfg["target"]
    )
    save_figure(fig_cmp, f"time_metrics_comparison{suffix}", save_dir=out_dir, **submission_kw)
    logger.info(f"Saved time_metrics_comparison{suffix}.png")

    # ── Figure 2: pred_distribution ──
    fig_dist = plot_prediction_distribution(preds_df_active, holdout_y, holdout_pids)
    save_figure(fig_dist, f"pred_distribution{suffix}", save_dir=out_dir, **submission_kw)
    logger.info(f"Saved pred_distribution{suffix}.png")

    # ── Trauma-score-dependent figures ──
    if not trauma_scores:
        logger.info("Trauma-score figures skipped (pass --trauma-scores to include).")
        return

    from astra.evaluation.trauma_scores import (
        build_trauma_score_df,
        evaluate_static_scores,
        evaluate_static_scores_over_time,
    )

    logger.info("Rebuilding trauma scores from raw vitals/diagnoses")
    trauma_df = build_trauma_score_df(data, cfg)

    static_scores_all = evaluate_static_scores(trauma_df, holdout_y, holdout_pids)
    if static_scores_all:
        fig_cmp_ts = plot_time_metrics_comparison(
            results_all, results_active,
            target_name=cfg["target"],
            static_scores=static_scores_all,
        )
        save_figure(fig_cmp_ts, f"time_metrics_comparison_trauma{suffix}",
                    save_dir=out_dir, **submission_kw)
        logger.info(f"Saved time_metrics_comparison_trauma{suffix}.png")

    rts_valid = trauma_df.dropna(subset=["RTS"])
    valid_pids = rts_valid["PID"].values
    if len(valid_pids) < 20:
        logger.warning(
            f"Only {len(valid_pids)} patients with RTS — skipping DeLong figures"
        )
        return

    logger.info("Recomputing paired DeLong comparisons (HNN vs RTS, TRISS)")
    score_results = evaluate_static_scores_over_time(
        trauma_df, preds_df_active, holdout_y, holdout_pids,
        valid_pids=valid_pids, delong=True,
    )

    for sname, paired in score_results.items():
        if sname == "ISS":
            continue
        if "delong_significant" not in paired:
            logger.warning(f"No DeLong results for {sname}; skipping")
            continue
        fig_dl = plot_delong_comparison(sname, paired)
        save_figure(fig_dl, f"delong_{sname.lower()}_comparison{suffix}",
                    save_dir=out_dir, **submission_kw)
        logger.info(f"Saved delong_{sname.lower()}_comparison{suffix}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate paper-revision figures from saved eval artifacts.",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model name (defaults to cfg['model_name']). Must match the directory "
             "under reports/eval/ where predictions CSVs live.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help='Filename suffix, e.g. "_rev20260430". Prepend an underscore yourself.',
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI. At the shipped figsizes (longest side ~10 inches for the "
             "4-panel figures, ~7.5 for pred_distribution), dpi=300 produces PNGs "
             "that exceed the 1200 px cap (~3000 px on long side) — save_figure will "
             "warn. If the cap is strict, pass --dpi 120 for the 4-panel figures or "
             "--dpi 160 for pred_distribution to land exactly on the cap.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory. Defaults to "
             "reports/eval/<model>/revision<suffix>/",
    )
    parser.add_argument(
        "--trauma-scores", action="store_true",
        help="Also regenerate trauma-score and DeLong comparison figures.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    model_name = args.model_name or cfg["model_name"]
    replot(
        model_name=model_name,
        suffix=args.suffix,
        dpi=args.dpi,
        output_dir=args.output_dir,
        trauma_scores=args.trauma_scores,
    )


if __name__ == "__main__":
    main()
