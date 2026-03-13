# evaluate_ebm_over_time.py
"""
Evaluate deployment EBM models on holdout data.

Evaluates the exact models used by the inference module (ebm_model_*.pkl from
generate_ebm_feature.py) on the holdout dataset, computing AUROC/AUPRC with
confidence intervals at each EBM interval.

Usage:
    python -m astra.models.ebm.evaluate_ebm_over_time --models_dir models/ebm
    python -m astra.models.ebm.evaluate_ebm_over_time --hybrid_results models/eval/hybrid_results.csv
"""

import logging
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass

from astra.utils import get_base_df, get_train_test_split, cfg
from astra.evaluation.utils import calculate_roc_auc_ci, calculate_average_precision_ci
from astra.models.ebm.generate_ebm_feature import (
    generate_ebm_intervals,
    _model_filename,
    _format_hours,
    _create_aggregated_dataset,
    preprocess_features,
    preload_concept_cache,
)
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class EBMTimeMetricResult:
    """Container for EBM time-dependent evaluation results."""
    time_label: str
    masking_hours: float
    auroc: float
    auroc_ci: Tuple[float, float]
    auprc: float
    auprc_ci: Tuple[float, float]
    n_samples: int
    n_positive: int
    time_hours: float
    time_days: float


def load_deployment_model(model_path: str) -> Dict:
    """Load a deployment EBM model (ebm_model_*.pkl)."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def evaluate_deployment_model_at_timepoint(
    holdout_df: pd.DataFrame,
    cfg_dict: dict,
    masking_hours: float,
    model_dict: dict,
    concept_cache: Optional[dict] = None,
) -> Optional[EBMTimeMetricResult]:
    """
    Evaluate a single deployment EBM model at its masking point on holdout data.

    Uses the same preprocessing path as the inference module:
    AggregatedDS -> preprocess_features(encoder, expected_cat/cont_feats).

    Args:
        holdout_df: Holdout patient base DataFrame.
        cfg_dict: Configuration dictionary.
        masking_hours: Masking time in hours for this model.
        model_dict: Loaded deployment model dict with keys:
            model, encoder, expected_cat_feats, expected_cont_feats, feature_names.
        concept_cache: Pre-loaded concept data to avoid repeated disk I/O.

    Returns:
        EBMTimeMetricResult or None if evaluation failed.
    """
    time_label = _format_hours(masking_hours)

    X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
        holdout_df, cfg_dict, masking_hours,
        concept_cache=concept_cache,
    )

    id_col = cfg_dict.get('dataset', {}).get('id_col', 'PID')
    X_features = X_full.drop(columns=[id_col], errors='ignore')
    y = np.asarray(y_full)

    if len(set(y)) < 2:
        logger.warning(f"Skipping {time_label}: only one class in holdout set")
        return None

    # Preprocess using the deployment model's encoder (matches inference path)
    X_processed, _, _ = preprocess_features(
        X_features,
        cat_feats=cat_feats,
        cont_feats=cont_feats,
        encoder=model_dict['encoder'],
        fit=False,
        expected_cat_feats=model_dict['expected_cat_feats'],
        expected_cont_feats=model_dict['expected_cont_feats'],
    )

    y_proba = model_dict['model'].predict_proba(X_processed)[:, 1]
    y_bin = y.round().astype(int)

    auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(y_bin, y_proba)
    auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(y_bin, y_proba)

    logger.info(
        f"{time_label}: AUROC={auroc:.3f} [{auroc_lower:.3f}-{auroc_upper:.3f}], "
        f"AUPRC={auprc:.3f} [{auprc_lower:.3f}-{auprc_upper:.3f}] "
        f"(n={len(y)}, pos={int(y.sum())})"
    )

    return EBMTimeMetricResult(
        time_label=time_label,
        masking_hours=masking_hours,
        auroc=auroc,
        auroc_ci=(auroc_lower, auroc_upper),
        auprc=auprc,
        auprc_ci=(auprc_lower, auprc_upper),
        n_samples=len(y),
        n_positive=int(y.sum()),
        time_hours=masking_hours,
        time_days=masking_hours / 24.0,
    )


def evaluate_all_deployment_ebms(
    models_dir: str = "models/ebm",
    save_dir: str = "models/eval",
) -> Tuple[List[EBMTimeMetricResult], pd.DataFrame]:
    """
    Evaluate all deployment EBM models on the holdout set.

    Loads models in the same format as the inference module (ebm_model_*.pkl),
    preprocesses holdout data identically to compute_ebm_predictions() in
    astra/inference/ebm.py, and computes AUROC/AUPRC with CIs.

    Args:
        models_dir: Directory containing deployment models (ebm_model_*.pkl).
        save_dir: Directory to save evaluation results.

    Returns:
        Tuple of (results list, predictions DataFrame).
    """
    logger.info("=" * 80)
    logger.info("EVALUATING DEPLOYMENT EBM MODELS ON HOLDOUT")
    logger.info("=" * 80)

    # Load data with same split as hybrid model
    base_df_full = get_base_df()
    train_df, holdout_df = get_train_test_split(cfg, base_df_full)

    logger.info(f"Trainval: {len(train_df)} patients")
    logger.info(f"Holdout:  {len(holdout_df)} patients")

    # Find all deployment model files
    intervals = generate_ebm_intervals(cfg)
    available = []
    for h in intervals:
        model_path = os.path.join(models_dir, _model_filename(h))
        if os.path.exists(model_path):
            available.append((h, model_path))

    if not available:
        raise ValueError(
            f"No deployment models found in {models_dir}. "
            f"Expected files like ebm_model_10min.pkl, ebm_model_1h.pkl, etc."
        )

    logger.info(f"Found {len(available)}/{len(intervals)} deployment models")

    # Pre-load concept data for holdout to avoid repeated disk reads
    logger.info("Pre-loading holdout concept data...")
    holdout_concept_cache = preload_concept_cache(holdout_df, cfg)

    # Evaluate each model
    results = []
    predictions = []

    for i, (masking_hours, model_path) in enumerate(available):
        logger.info(f"[{i + 1}/{len(available)}] {os.path.basename(model_path)}")

        try:
            model_dict = load_deployment_model(model_path)

            result = evaluate_deployment_model_at_timepoint(
                holdout_df=holdout_df,
                cfg_dict=cfg,
                masking_hours=masking_hours,
                model_dict=model_dict,
                concept_cache=holdout_concept_cache,
            )

            if result is not None:
                results.append(result)

                # Save per-patient predictions
                id_col = cfg.get('dataset', {}).get('id_col', 'PID')
                X_full, y_full, cat_feats, cont_feats = _create_aggregated_dataset(
                    holdout_df, cfg, masking_hours,
                    concept_cache=holdout_concept_cache,
                )
                pids = X_full[id_col].values
                X_features = X_full.drop(columns=[id_col], errors='ignore')

                X_processed, _, _ = preprocess_features(
                    X_features,
                    cat_feats=cat_feats,
                    cont_feats=cont_feats,
                    encoder=model_dict['encoder'],
                    fit=False,
                    expected_cat_feats=model_dict['expected_cat_feats'],
                    expected_cont_feats=model_dict['expected_cont_feats'],
                )

                y_proba = model_dict['model'].predict_proba(X_processed)[:, 1]

                for pid, pred, yt in zip(pids, y_proba, np.asarray(y_full)):
                    predictions.append({
                        'PID': pid,
                        'masking_hours': masking_hours,
                        'time_hours': masking_hours,
                        'time_days': masking_hours / 24.0,
                        'pred': float(pred),
                        'y_true': int(round(yt)),
                    })

        except Exception as e:
            logger.error(f"Failed to evaluate {os.path.basename(model_path)}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successfully evaluated: {len(results)}/{len(available)} models")

    results_df = pd.DataFrame([
        {
            'masking_hours': r.masking_hours,
            'time_label': r.time_label,
            'time_hours': r.time_hours,
            'time_days': r.time_days,
            'auroc': r.auroc,
            'auroc_lower': r.auroc_ci[0],
            'auroc_upper': r.auroc_ci[1],
            'auprc': r.auprc,
            'auprc_lower': r.auprc_ci[0],
            'auprc_upper': r.auprc_ci[1],
            'n_samples': r.n_samples,
            'n_positive': r.n_positive,
        }
        for r in results
    ])

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, 'ebm_deployment_eval_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved: {results_path}")

    predictions_df = pd.DataFrame(predictions)
    preds_path = os.path.join(save_dir, 'ebm_deployment_predictions.csv')
    predictions_df.to_csv(preds_path, index=False)
    predictions_df.to_pickle(os.path.join(save_dir, 'ebm_deployment_predictions.pkl'))
    logger.info(f"Predictions saved: {preds_path}")

    # Summary table
    logger.info("\nPerformance Summary (deployment models on holdout):")
    logger.info("-" * 80)
    logger.info(f"{'Time':>12} | {'AUROC':>22} | {'AUPRC':>22}")
    logger.info("-" * 80)
    for _, row in results_df.iterrows():
        logger.info(
            f"{row['time_label']:>12} | "
            f"{row['auroc']:>5.3f} [{row['auroc_lower']:>5.3f}-{row['auroc_upper']:>5.3f}] | "
            f"{row['auprc']:>5.3f} [{row['auprc_lower']:>5.3f}-{row['auprc_upper']:>5.3f}]"
        )
    logger.info("-" * 80)
    logger.info(f"Mean AUROC: {results_df['auroc'].mean():.3f} +/- {results_df['auroc'].std():.3f}")
    logger.info(f"Mean AUPRC: {results_df['auprc'].mean():.3f} +/- {results_df['auprc'].std():.3f}")

    return results, predictions_df


def plot_ebm_vs_hybrid_comparison(
    ebm_results: List[EBMTimeMetricResult],
    hybrid_results_path: Optional[str] = None,
    cut_hours: int = 72,
    max_days: int = 30,
    save_dir: str = 'reports/eval'
):
    """
    Plot EBM performance over time and compare with hybrid model if available.

    Uses 2-panel layout: hours (left), days (right).
    """
    logger.info("\nCreating comparison plots...")

    times_h = np.array([r.time_hours for r in ebm_results])
    times_d = np.array([r.time_days for r in ebm_results])
    auroc_vals = np.array([r.auroc for r in ebm_results])
    auroc_lower = np.array([r.auroc_ci[0] for r in ebm_results])
    auroc_upper = np.array([r.auroc_ci[1] for r in ebm_results])
    auprc_vals = np.array([r.auprc for r in ebm_results])
    auprc_lower = np.array([r.auprc_ci[0] for r in ebm_results])
    auprc_upper = np.array([r.auprc_ci[1] for r in ebm_results])

    # Load hybrid results if available
    hybrid_data = None
    if hybrid_results_path and os.path.exists(hybrid_results_path):
        logger.info(f"Loading hybrid model results from: {hybrid_results_path}")
        hybrid_df = pd.read_csv(hybrid_results_path)

        if all(col in hybrid_df.columns for col in ['time_hours', 'time_days', 'auroc', 'auprc']):
            hybrid_data = {
                'times_h': hybrid_df['time_hours'].values,
                'times_d': hybrid_df['time_days'].values,
                'auroc': hybrid_df['auroc'].values,
                'auprc': hybrid_df['auprc'].values,
                'auroc_lower': hybrid_df.get('auroc_lower', hybrid_df['auroc']).values,
                'auroc_upper': hybrid_df.get('auroc_upper', hybrid_df['auroc']).values,
                'auprc_lower': hybrid_df.get('auprc_lower', hybrid_df['auprc']).values,
                'auprc_upper': hybrid_df.get('auprc_upper', hybrid_df['auprc']).values,
            }
            logger.info("Hybrid model results loaded")
        else:
            logger.warning("Hybrid results file doesn't have expected columns")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: hours
    mask_cut = times_h <= cut_hours

    ax1.plot(times_h[mask_cut], auroc_vals[mask_cut], 'o-', color='C0',
             label='AUROC (EBM)', markersize=3, linewidth=2)
    ax1.fill_between(times_h[mask_cut], auroc_lower[mask_cut], auroc_upper[mask_cut],
                      color='C0', alpha=0.2)

    ax1.plot(times_h[mask_cut], auprc_vals[mask_cut], 's-', color='C1',
             label='AUPRC (EBM)', markersize=3, linewidth=2)
    ax1.fill_between(times_h[mask_cut], auprc_lower[mask_cut], auprc_upper[mask_cut],
                      color='C1', alpha=0.2)

    if hybrid_data:
        mask_hybrid = hybrid_data['times_h'] <= cut_hours
        ax1.plot(hybrid_data['times_h'][mask_hybrid], hybrid_data['auroc'][mask_hybrid],
                 'o--', color='C0', label='AUROC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)
        ax1.plot(hybrid_data['times_h'][mask_hybrid], hybrid_data['auprc'][mask_hybrid],
                 's--', color='C1', label='AUPRC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)

    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours + 1, 4))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("A)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.0, 1.0)

    # Panel B: days
    ax2.plot(times_d, auroc_vals, 'o-', color='C0',
             label='AUROC (EBM)', markersize=3, linewidth=2)
    ax2.fill_between(times_d, auroc_lower, auroc_upper, color='C0', alpha=0.2)

    ax2.plot(times_d, auprc_vals, 's-', color='C1',
             label='AUPRC (EBM)', markersize=3, linewidth=2)
    ax2.fill_between(times_d, auprc_lower, auprc_upper, color='C1', alpha=0.2)

    if hybrid_data:
        ax2.plot(hybrid_data['times_d'], hybrid_data['auroc'],
                 'o--', color='C0', label='AUROC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)
        ax2.plot(hybrid_data['times_d'], hybrid_data['auprc'],
                 's--', color='C1', label='AUPRC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)

    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days + 1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("B)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ebm_deployment_vs_hybrid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved: {save_path}")

    return fig


def plot_global_importance_heatmap(
    models_dir: str = "models/ebm",
    top_n: int = 25,
    save_dir: str = "reports/eval",
):
    """
    Heatmap of global feature importances across all timepoints.

    Works with deployment models (ebm_model_*.pkl).
    """
    intervals = generate_ebm_intervals(cfg)
    model_files = []
    for h in intervals:
        p = Path(models_dir) / _model_filename(h)
        if p.exists():
            model_files.append((h, p))

    if not model_files:
        raise ValueError(f"No deployment EBM models found in {models_dir}")

    all_importances = {}
    for masking_hours, mf in model_files:
        model_data = load_deployment_model(str(mf))
        ebm = model_data['model']
        time_label = _format_hours(masking_hours)

        imp = pd.Series(ebm.term_importances(), index=ebm.term_names_)
        all_importances[time_label] = imp

    heatmap_df = pd.DataFrame(all_importances).T.fillna(0.0)

    top_features = heatmap_df.max(axis=0).nlargest(top_n).index
    heatmap_df = heatmap_df[top_features]

    # Transpose: features on y-axis, timepoints on x-axis
    heatmap_df = heatmap_df.T

    n_time = heatmap_df.shape[1]
    n_feat = heatmap_df.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, n_time * 0.55), max(4, n_feat * 0.3)))
    sns.heatmap(
        heatmap_df,
        cmap="YlOrRd",
        linewidths=0.3,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xlabel("Time point", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"Global Feature Importance Over Time (top {top_n})", fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ebm_global_importance_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved global importance heatmap: {save_path}")

    return fig, heatmap_df


def plot_local_importance_heatmap(
    pids: List[int],
    base_df: pd.DataFrame,
    models_dir: str = "models/ebm",
    top_n: int = 25,
    save_dir: str = "reports/eval",
):
    """
    Heatmap of local feature contributions for specific PIDs across timepoints.

    Works with deployment models (ebm_model_*.pkl). Uses preprocess_features()
    path matching inference.
    """
    intervals = generate_ebm_intervals(cfg)
    model_files = []
    for h in intervals:
        p = Path(models_dir) / _model_filename(h)
        if p.exists():
            model_files.append((h, p))

    if not model_files:
        raise ValueError(f"No deployment EBM models found in {models_dir}")

    pid_base = base_df[base_df['PID'].isin(pids)].copy()
    if len(pid_base) == 0:
        raise ValueError(f"None of the specified PIDs found in base_df")
    found_pids = pid_base['PID'].unique().tolist()
    logger.info(f"Found {len(found_pids)}/{len(pids)} PIDs in base_df")

    # Pre-load concept data
    concept_cache = preload_concept_cache(pid_base, cfg)

    all_local = {}
    for masking_hours, mf in model_files:
        model_data = load_deployment_model(str(mf))
        ebm = model_data['model']
        time_label = _format_hours(masking_hours)

        X_full, _, cat_feats, cont_feats = _create_aggregated_dataset(
            pid_base, cfg, masking_hours, concept_cache=concept_cache,
        )

        id_col = cfg.get('dataset', {}).get('id_col', 'PID')
        row_pids = X_full[id_col].values
        X_features = X_full.drop(columns=[id_col], errors='ignore')

        X_processed, _, _ = preprocess_features(
            X_features,
            cat_feats=cat_feats,
            cont_feats=cont_feats,
            encoder=model_data['encoder'],
            fit=False,
            expected_cat_feats=model_data['expected_cat_feats'],
            expected_cont_feats=model_data['expected_cont_feats'],
        )

        contributions = ebm.eval_terms(X_processed)
        local_df = pd.DataFrame(contributions, columns=ebm.term_names_)
        local_df.index = row_pids
        all_local[time_label] = local_df

    # Mean across PIDs heatmap
    mean_contributions = {}
    for time_label, local_df in all_local.items():
        mean_contributions[time_label] = local_df.mean(axis=0)

    mean_df = pd.DataFrame(mean_contributions).T.fillna(0.0)

    top_features = mean_df.abs().max(axis=0).nlargest(top_n).index
    mean_plot = mean_df[top_features].T

    n_time = mean_plot.shape[1]
    n_feat = mean_plot.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, n_time * 0.55), max(4, n_feat * 0.3)))
    sns.heatmap(
        mean_plot,
        cmap="RdBu_r",
        center=0,
        linewidths=0.3,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xlabel("Time point", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"Mean Local Contributions Over Time (n={len(found_pids)} PIDs, top {top_n})", fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ebm_local_mean_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved mean local heatmap: {save_path}")

    # Per-PID heatmaps
    pid_figs = {}
    for pid in found_pids:
        pid_contributions = {}
        for time_label, local_df in all_local.items():
            if pid in local_df.index:
                pid_contributions[time_label] = local_df.loc[pid]

        if not pid_contributions:
            continue

        pid_df = pd.DataFrame(pid_contributions).T.fillna(0.0)
        pid_plot = pid_df[top_features] if all(f in pid_df.columns for f in top_features) else pid_df
        pid_plot = pid_plot.T

        n_time_p = pid_plot.shape[1]
        n_feat_p = pid_plot.shape[0]
        fig_pid, ax_pid = plt.subplots(figsize=(max(4, n_time_p * 0.55), max(4, n_feat_p * 0.3)))
        sns.heatmap(
            pid_plot,
            cmap="RdBu_r",
            center=0,
            linewidths=0.3,
            ax=ax_pid,
            xticklabels=True,
            yticklabels=True,
        )
        ax_pid.set_xlabel("Time point", fontsize=12)
        ax_pid.set_ylabel("Feature", fontsize=12)
        ax_pid.set_title(f"Local Contributions Over Time - PID {pid}", fontsize=13)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        pid_path = os.path.join(save_dir, f"ebm_local_pid{pid}_heatmap.png")
        fig_pid.savefig(pid_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PID {pid} heatmap: {pid_path}")
        pid_figs[pid] = fig_pid

    return fig, mean_df, all_local, pid_figs


def main():
    """Evaluate deployment EBM models on holdout and optionally compare with hybrid."""
    parser = argparse.ArgumentParser(
        description='Evaluate deployment EBM models on holdout data'
    )
    parser.add_argument('--models_dir', type=str, default='models/ebm',
                        help='Directory containing deployment models (ebm_model_*.pkl)')
    parser.add_argument('--hybrid_results', type=str, default=None,
                        help='Path to hybrid model evaluation results CSV (optional)')
    parser.add_argument('--cut_hours', type=int, default=72,
                        help='Hours cutoff for first plot panel')
    parser.add_argument('--max_days', type=int, default=30,
                        help='Maximum days for second plot panel')
    parser.add_argument('--save_dir', type=str, default='models/eval',
                        help='Directory to save evaluation outputs')
    parser.add_argument('--plot_dir', type=str, default='reports/eval',
                        help='Directory to save plots')
    parser.add_argument('--heatmap', action='store_true',
                        help='Also generate global importance heatmap')

    args = parser.parse_args()

    results, predictions_df = evaluate_all_deployment_ebms(
        models_dir=args.models_dir,
        save_dir=args.save_dir,
    )

    if results:
        plot_ebm_vs_hybrid_comparison(
            ebm_results=results,
            hybrid_results_path=args.hybrid_results,
            cut_hours=args.cut_hours,
            max_days=args.max_days,
            save_dir=args.plot_dir,
        )

    if args.heatmap:
        plot_global_importance_heatmap(
            models_dir=args.models_dir,
            save_dir=args.plot_dir,
        )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
