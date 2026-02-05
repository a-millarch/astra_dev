# evaluate_ebm_over_time.py
"""
Evaluate trained EBM models at different time points and compare with hybrid model.

This script:
1. Loads all trained EBM models from train_ebm_over_time.py
2. Evaluates each model on a test/holdout set with the corresponding masking point
3. Calculates metrics (AUROC, AUPRC) with confidence intervals
4. Creates comparison plots with hybrid model results (if available)
5. Saves predictions and visualizations

Usage:
    python evaluate_ebm_over_time.py --models_dir models/ebm --test_frac 0.20
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from astra.utils import get_base_df, get_train_test_split, cfg, logger, save_figure
from astra.data.datasets import AggregatedDS
from astra.evaluation.utils import calculate_roc_auc_ci, calculate_average_precision_ci
import seaborn as sns


@dataclass
class EBMTimeMetricResult:
    """Container for EBM time-dependent evaluation results"""
    time_label: str
    step: int
    masking_point: str
    auroc: float
    auroc_ci: Tuple[float, float]
    auprc: float
    auprc_ci: Tuple[float, float]
    n_samples: int
    n_positive: int
    time_hours: float
    time_days: float


def load_ebm_model(model_path: str) -> Dict:
    """Load a trained EBM model and its metadata."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def evaluate_ebm_at_timepoint(
    test_df: pd.DataFrame,
    cfg: dict,
    model_path: str,
) -> Optional[EBMTimeMetricResult]:
    """
    Evaluate a single EBM model at its trained time point.

    Args:
        test_df: Test dataframe (already split, only test patients)
        cfg: Configuration dictionary
        model_path: Path to trained EBM model

    Returns:
        EBMTimeMetricResult or None if evaluation failed
    """
    # Load model
    logger.info(f"Loading model: {os.path.basename(model_path)}")
    model_data = load_ebm_model(model_path)

    ebm = model_data['model']
    preprocessor = model_data['preprocessor']
    step = model_data['step']
    masking_point = model_data['masking_point']
    time_label = model_data['time_label']
    logger.info(f"Evaluating at {time_label} (masking={masking_point})")

    # ============================================================================
    # CREATE TEST DATASET WITH SAME MASKING POINT (only test patients)
    # ============================================================================
    # Convert masking_point string back to Timedelta
    if isinstance(masking_point, str):
        masking_point = pd.Timedelta(masking_point)

    # Create aggregated dataset ONLY on test patients
    agg_ds = AggregatedDS(
        cfg=cfg,
        base_df=test_df,  # Only test patients
        masking_point=masking_point,
        agg_funcs=['first', 'last', 'min', 'max', 'mean', 'std'],
        concepts=cfg["concepts"],
        default_mode=True,
    )

    X_test, y_test = agg_ds.get_X_y()

    logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} positive)")

    # Check if we have both classes
    if len(set(y_test)) < 2:
        logger.warning(f"Skipping {time_label}: only one class in test set")
        return None

    # ============================================================================
    # PREPROCESS AND PREDICT
    # ============================================================================
    X_test_proc = preprocessor.transform(X_test)
    y_proba = ebm.predict_proba(X_test_proc)[:, 1]
    y_test_bin = np.array(y_test).round().astype(int)

    # ============================================================================
    # CALCULATE METRICS WITH CONFIDENCE INTERVALS
    # ============================================================================
    auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(y_test_bin, y_proba)
    auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(y_test_bin, y_proba)

    logger.info(f"Results: AUROC={auroc:.3f} [{auroc_lower:.3f}-{auroc_upper:.3f}], "
                f"AUPRC={auprc:.3f} [{auprc_lower:.3f}-{auprc_upper:.3f}]")

    # Calculate time in hours and days
    time_min = masking_point.total_seconds() / 60
    time_hours = time_min / 60
    time_days = time_min / (24 * 60)

    return EBMTimeMetricResult(
        time_label=time_label,
        step=step,
        masking_point=str(masking_point),
        auroc=auroc,
        auroc_ci=(auroc_lower, auroc_upper),
        auprc=auprc,
        auprc_ci=(auprc_lower, auprc_upper),
        n_samples=len(y_test),
        n_positive=int(y_test.sum()),
        time_hours=time_hours,
        time_days=time_days
    )


def evaluate_all_ebms(
    models_dir: str = "models/ebm",
) -> Tuple[List[EBMTimeMetricResult], pd.DataFrame]:
    """
    Evaluate all trained EBM models on test set.

    Uses the same train/test split as the hybrid model (from config).

    Args:
        models_dir: Directory containing trained models

    Returns:
        Tuple of (results list, predictions DataFrame)
    """
    logger.info("="*80)
    logger.info("EVALUATING EBM MODELS OVER TIME")
    logger.info("="*80)

    # ============================================================================
    # LOAD BASE DATA AND APPLY SAME SPLIT AS HYBRID MODEL
    # ============================================================================
    logger.info("Loading base dataframe and applying train/test split...")
    base_df_full = get_base_df()

    # Use the same split as hybrid model (from config)
    train_df, test_df = get_train_test_split(cfg, base_df_full)

    logger.info(f"✓ Total patients: {len(base_df_full)}")
    logger.info(f"  Training set: {len(train_df)} patients")
    logger.info(f"  Test set:     {len(test_df)} patients")
    logger.info(f"Split strategy: {cfg.get('holdout_type', 'temporal')}")

    # Use only test_df for evaluation (no need for full dataset)
    test_df = test_df.sort_values('start').reset_index(drop=True)

    # ============================================================================
    # FIND ALL TRAINED MODELS
    # ============================================================================
    logger.info(f"\nSearching for models in: {models_dir}")
    model_files = sorted(Path(models_dir).glob("ebm_step*.pkl"))

    if not model_files:
        raise ValueError(f"No EBM models found in {models_dir}")

    logger.info(f"✓ Found {len(model_files)} trained models")

    # ============================================================================
    # EVALUATE EACH MODEL
    # ============================================================================
    results = []
    predictions = []

    for i, model_path in enumerate(model_files):
        logger.info(f"\n[{i+1}/{len(model_files)}] Evaluating {model_path.name}...")

        try:
            result = evaluate_ebm_at_timepoint(
                test_df=test_df,
                cfg=cfg,
                model_path=str(model_path),
            )

            if result is not None:
                results.append(result)

                # Also save predictions for this timepoint
                # (Re-load model and predict to save predictions)
                model_data = load_ebm_model(str(model_path))
                ebm = model_data['model']
                preprocessor = model_data['preprocessor']
                masking_point = pd.Timedelta(model_data['masking_point'])
                # Create dataset (only test patients)
                agg_ds = AggregatedDS(
                    cfg=cfg,
                    base_df=test_df,  # Only test patients
                    masking_point=masking_point,
                    agg_funcs=['first', 'last', 'min', 'max', 'mean', 'std'],
                    concepts=cfg["concepts"],
                    default_mode=True,
                )
                X_test, y_test = agg_ds.get_X_y()

                # Get patient IDs
                patient_ids = test_df['PID'].values

                # Predict
                X_test_proc = preprocessor.transform(X_test)
                y_proba = ebm.predict_proba(X_test_proc)[:, 1]

                # Calculate time in minutes (matching hybrid model structure)
                time_min = masking_point.total_seconds() / 60

                # Store predictions (matching hybrid model structure)
                for pid, pred in zip(patient_ids, y_proba):
                    predictions.append({
                        'PID': pid,
                        'censor_step': result.step,  # Match hybrid model field name
                        'time_min': time_min,  # Add time_min field
                        'time_hours': result.time_hours,
                        'time_days': result.time_days,
                        'pred': float(pred)
                    })

        except Exception as e:
            logger.error(f"Failed to evaluate {model_path.name}: {e}")
            continue

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Successfully evaluated: {len(results)}/{len(model_files)} models")

    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'step': r.step,
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

    # Save results
    os.makedirs('models/eval', exist_ok=True)
    results_path = 'models/eval/ebm_evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n✓ Evaluation results saved: {results_path}")

    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    preds_path = 'models/eval/ebm_predictions.csv'
    predictions_df.to_csv(preds_path, index=False)
    predictions_df.to_pickle('models/eval/ebm_predictions.pkl')
    logger.info(f"✓ Predictions saved: {preds_path}")

    # Print summary
    logger.info("\nPerformance Summary:")
    logger.info("-"*80)
    logger.info(f"{'Time':>12} | {'AUROC':>17} | {'AUPRC':>17}")
    logger.info("-"*80)
    for _, row in results_df.iterrows():
        logger.info(
            f"{row['time_label']:>12} | "
            f"{row['auroc']:>5.3f} [{row['auroc_lower']:>5.3f}-{row['auroc_upper']:>5.3f}] | "
            f"{row['auprc']:>5.3f} [{row['auprc_lower']:>5.3f}-{row['auprc_upper']:>5.3f}]"
        )
    logger.info("-"*80)

    logger.info(f"\nMean AUROC: {results_df['auroc'].mean():.3f} ± {results_df['auroc'].std():.3f}")
    logger.info(f"Mean AUPRC: {results_df['auprc'].mean():.3f} ± {results_df['auprc'].std():.3f}")

    return results, predictions_df


def plot_ebm_vs_hybrid_comparison(
    ebm_results: List[EBMTimeMetricResult],
    hybrid_results_path: Optional[str] = None,
    cut_hours: int = 72,
    max_days: int = 30,
    save_dir: str = 'reports'
):
    """
    Plot EBM performance over time and compare with hybrid model if available.

    Uses the same plotting style as the hybrid model evaluation (2-panel layout).

    Args:
        ebm_results: List of EBMTimeMetricResult objects
        hybrid_results_path: Path to hybrid model predictions CSV (optional)
        cut_hours: Hours cutoff for first plot
        max_days: Maximum days for second plot
        save_dir: Directory to save plots
    """
    logger.info("\nCreating comparison plots...")

    # Extract EBM data
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

        # Try to extract time and metrics
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
            logger.info("✓ Hybrid model results loaded successfully")
        else:
            logger.warning("Hybrid results file doesn't have expected columns")

    # ============================================================================
    # CREATE PLOTS (2-panel layout matching hybrid model evaluation style)
    # ============================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ============================================================================
    # PLOT 1: AUROC and AUPRC over hours
    # ============================================================================
    mask_cut = times_h <= cut_hours

    # AUROC - EBM
    ax1.plot(times_h[mask_cut], auroc_vals[mask_cut], 'o-', color='C0',
             label='AUROC (EBM)', markersize=3, linewidth=2)
    ax1.fill_between(times_h[mask_cut], auroc_lower[mask_cut], auroc_upper[mask_cut],
                      color='C0', alpha=0.2)

    # AUPRC - EBM
    ax1.plot(times_h[mask_cut], auprc_vals[mask_cut], 's-', color='C1',
             label='AUPRC (EBM)', markersize=3, linewidth=2)
    ax1.fill_between(times_h[mask_cut], auprc_lower[mask_cut], auprc_upper[mask_cut],
                      color='C1', alpha=0.2)

    # Add hybrid model if available
    if hybrid_data:
        mask_hybrid = hybrid_data['times_h'] <= cut_hours
        ax1.plot(hybrid_data['times_h'][mask_hybrid], hybrid_data['auroc'][mask_hybrid],
                 'o--', color='C0', label='AUROC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)
        ax1.plot(hybrid_data['times_h'][mask_hybrid], hybrid_data['auprc'][mask_hybrid],
                 's--', color='C1', label='AUPRC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)

    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours+1, 4))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("A)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.0, 1.0)

    # ============================================================================
    # PLOT 2: AUROC and AUPRC over days
    # ============================================================================
    # AUROC - EBM
    ax2.plot(times_d, auroc_vals, 'o-', color='C0',
             label='AUROC (EBM)', markersize=3, linewidth=2)
    ax2.fill_between(times_d, auroc_lower, auroc_upper, color='C0', alpha=0.2)

    # AUPRC - EBM
    ax2.plot(times_d, auprc_vals, 's-', color='C1',
             label='AUPRC (EBM)', markersize=3, linewidth=2)
    ax2.fill_between(times_d, auprc_lower, auprc_upper, color='C1', alpha=0.2)

    # Add hybrid model if available
    if hybrid_data:
        ax2.plot(hybrid_data['times_d'], hybrid_data['auroc'],
                 'o--', color='C0', label='AUROC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)
        ax2.plot(hybrid_data['times_d'], hybrid_data['auprc'],
                 's--', color='C1', label='AUPRC (Hybrid)', markersize=2, linewidth=1.5, alpha=0.5)

    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days+1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("B)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ebm_vs_hybrid_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Comparison plot saved: {save_path}")

    return fig


def _align_X_to_ebm(X: pd.DataFrame, ebm) -> pd.DataFrame:
    """Align feature matrix to EBM's expected features. Adds missing cols as 0."""
    expected = list(ebm.feature_names)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=missing)], axis=1)
    return X[expected]


def plot_global_importance_heatmap(
    models_dir: str = "models/ebm",
    top_n: int = 25,
    save_dir: str = "reports",
):
    """
    Heatmap of global feature importances across all timepoints.

    Each cell = mean absolute score for that feature at that timepoint
    (from ebm.term_importances()).

    Args:
        models_dir: Directory containing trained EBM model pickles
        top_n: Number of top features to show (by max importance across time)
        save_dir: Directory to save the figure
    """
    model_files = sorted(Path(models_dir).glob("ebm_step*.pkl"))
    if not model_files:
        raise ValueError(f"No EBM models found in {models_dir}")

    all_importances = {}
    for mf in model_files:
        model_data = load_ebm_model(str(mf))
        ebm = model_data['model']
        time_label = model_data['time_label']

        imp = pd.Series(ebm.term_importances(), index=ebm.term_names_)
        all_importances[time_label] = imp

    # Rows = timepoints, columns = features; NaN for features absent in a model
    heatmap_df = pd.DataFrame(all_importances).T.fillna(0.0)

    # Select top_n features by max importance across all timepoints
    top_features = heatmap_df.max(axis=0).nlargest(top_n).index
    heatmap_df = heatmap_df[top_features]

    # Transpose: features on y-axis, timepoints on x-axis
    heatmap_df = heatmap_df.T

    # Plot — compact cells
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
    save_dir: str = "reports",
):
    """
    Heatmap of local feature contributions for specific PIDs across timepoints.

    Each cell = mean additive log-odds contribution for that feature across
    the selected PIDs at that timepoint (from ebm.eval_terms()).

    Produces one heatmap with the mean across PIDs, plus individual per-PID
    heatmaps if len(pids) is small.

    Args:
        pids: List of patient IDs to explain
        base_df: Base dataframe (must contain these PIDs)
        models_dir: Directory containing trained EBM model pickles
        top_n: Number of top features to show (by max absolute contribution)
        save_dir: Directory to save figures
    """
    model_files = sorted(Path(models_dir).glob("ebm_step*.pkl"))
    if not model_files:
        raise ValueError(f"No EBM models found in {models_dir}")

    pid_base = base_df[base_df['PID'].isin(pids)].copy()
    if len(pid_base) == 0:
        raise ValueError(f"None of the specified PIDs found in base_df")
    found_pids = pid_base['PID'].unique().tolist()
    logger.info(f"Found {len(found_pids)}/{len(pids)} PIDs in base_df")

    # Collect local contributions: {time_label: DataFrame(PIDs x features)}
    all_local = {}
    for mf in model_files:
        model_data = load_ebm_model(str(mf))
        ebm = model_data['model']
        time_label = model_data['time_label']
        masking_point = model_data['masking_point']

        if isinstance(masking_point, str):
            masking_point = pd.Timedelta(masking_point)

        # Build dataset for these PIDs at this timepoint
        agg_ds = AggregatedDS(
            cfg=cfg,
            base_df=pid_base,
            masking_point=masking_point,
            agg_funcs=['first', 'last', 'min', 'max', 'mean', 'std'],
            concepts=cfg["concepts"],
            default_mode=True,
        )
        X, _ = agg_ds.get_X_y()

        # Align to model features (PID subset may be missing categoricals)
        X = _align_X_to_ebm(X, ebm)

        # Manual imputation (preprocessor is just SimpleImputer(most_frequent))
        X = X.fillna(X.mode().iloc[0] if len(X) > 0 else 0)
        contributions = ebm.eval_terms(X)

        local_df = pd.DataFrame(contributions, columns=ebm.term_names_)
        local_df.index = pid_base['PID'].values
        all_local[time_label] = local_df

    # =========================================================================
    # Mean across PIDs heatmap (rows=time, cols=features)
    # =========================================================================
    mean_contributions = {}
    for time_label, local_df in all_local.items():
        mean_contributions[time_label] = local_df.mean(axis=0)

    mean_df = pd.DataFrame(mean_contributions).T.fillna(0.0)

    # Select top features by max absolute contribution
    top_features = mean_df.abs().max(axis=0).nlargest(top_n).index
    mean_plot = mean_df[top_features].T  # Transpose: features on y, time on x

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

    # =========================================================================
    # Per-PID heatmaps (rows=time, cols=features)
    # =========================================================================
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
        pid_plot = pid_plot.T  # Transpose: features on y, time on x

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
        ax_pid.set_title(f"Local Contributions Over Time — PID {pid}", fontsize=13)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        pid_path = os.path.join(save_dir, f"ebm_local_pid{pid}_heatmap.png")
        fig_pid.savefig(pid_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PID {pid} heatmap: {pid_path}")
        pid_figs[pid] = fig_pid

    return fig, mean_df, all_local, pid_figs


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Evaluate EBM models over time')
    parser.add_argument('--models_dir', type=str, default='models/ebm',
                        help='Directory containing trained models')
    parser.add_argument('--hybrid_results', type=str, default=None,
                        help='Path to hybrid model evaluation results CSV (optional)')
    parser.add_argument('--cut_hours', type=int, default=72,
                        help='Hours cutoff for visualization')
    parser.add_argument('--max_days', type=int, default=30,
                        help='Maximum days for visualization')
    parser.add_argument('--save_dir', type=str, default='reports',
                        help='Directory to save plots')

    args = parser.parse_args()

    # Run evaluation (uses config-based train/test split)
    results, predictions_df = evaluate_all_ebms(
        models_dir=args.models_dir,
    )

    # Create comparison plots
    if results:
        plot_ebm_vs_hybrid_comparison(
            ebm_results=results,
            hybrid_results_path=args.hybrid_results,
            cut_hours=args.cut_hours,
            max_days=args.max_days,
            save_dir=args.save_dir,
        )

    logger.info("\n" + "="*80)
    logger.info("ALL DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
