"""Replot the paper's cohort SHAP figures from saved temporal-SHAP artifacts.

No model, no data dict, no SHAP recomputation — everything comes out of the
pickled CohortTemporalSHAPResults written by run_cohort_temporal_shap().

This is deliberately NOT the `shap_paper_figures --figures-only` path: that one
reads reports/shap_paper/shap_cache.pkl, which is a different artifact from a
different code route and can be stale relative to the paper run.

Regenerates:
    cohort_temporal_comparison<suffix>.png   (+ base64)
    cohort_feature_trajectory<suffix>.png    (+ base64)
    figure_shap_summary_panel.png/.pdf       (+ base64)

Usage:
    python -m scripts.replot_shap_paper_figures \
        --shap-dir reports/eval/rev20260430_BEFORERERUN/temporal_shap \
        --out-dir  reports/eval/rev20260430/revision_typeset/shap
"""
import argparse
import logging
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Must live under the "astra" hierarchy: setup_logging() only attaches handlers
# there, so a bare __name__ logger (== "__main__" under `python -m`) is silent.
logger = logging.getLogger("astra.scripts.replot_shap_paper_figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap-dir", required=True,
                    help="Directory holding cohort_temporal_shap_results<suffix>.pkl "
                         "and cohort_shap_all_features<suffix>.csv")
    ap.add_argument("--out-dir", required=True, help="Where to write the figures")
    ap.add_argument("--suffix", default="_active_dn",
                    help="Artifact suffix (default: _active_dn)")
    ap.add_argument("--skip-panel", action="store_true",
                    help="Skip figure_shap_summary_panel")
    args = ap.parse_args()

    from astra.utils import setup_logging
    setup_logging(logging.INFO)

    pkl = os.path.join(args.shap_dir, f"cohort_temporal_shap_results{args.suffix}.pkl")
    csv = os.path.join(args.shap_dir, f"cohort_shap_all_features{args.suffix}.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Results pickle not found: {pkl}")

    logger.info(f"Loading {pkl} (this is a large file, be patient)")
    with open(pkl, "rb") as f:
        results = pickle.load(f)
    # Printed, not just logged - this is the verification output the caller needs.
    print()
    print("=" * 62)
    print("LOADED COHORT SHAP RESULTS  (check these against the paper)")
    print("=" * 62)
    print(f"  source            : {pkl}")
    print(f"  n_patients        : {results.n_patients}")
    print(f"  timeframes        : {results.get_available_timeframes()}")
    print(f"  patient_counts    : {results.patient_counts}")
    print(f"  active_only       : {results.active_only}")
    print(f"  density_normalize : {results.density_normalize}")
    print(f"  n_channels        : {len(results.channel2feature)}")
    print("=" * 62)
    print()

    # Both plotters are pure functions of `results` (zero `self.` references),
    # so they can be called unbound — no analyzer, no model, no data dict.
    from astra.evaluation.behavior import TemporalSHAPAnalyzer

    for name, method in [
        ("cohort_temporal_comparison", TemporalSHAPAnalyzer.plot_cohort_temporal_comparison),
        ("cohort_feature_trajectory", TemporalSHAPAnalyzer.plot_cohort_feature_trajectory),
    ]:
        out = os.path.join(args.out_dir, f"{name}{args.suffix}.png")
        logger.info(f"Regenerating {name}{args.suffix}")
        fig = method(None, results, save_path=out)
        if fig:
            plt.close(fig)

    if not args.skip_panel:
        from astra.evaluation.shap_paper_figures import figure_shap_summary_panel
        logger.info("Regenerating figure_shap_summary_panel")
        if not os.path.exists(csv):
            logger.warning(f"CSV not found ({csv}); relying on pickle alone")
        figure_shap_summary_panel(csv_path=csv, save_dir=args.out_dir, pickle_path=pkl)

    print()
    print(f"Done. Output written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
