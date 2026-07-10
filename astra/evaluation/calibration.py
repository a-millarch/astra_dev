# evaluation_with_calibration.py
"""
Extended evaluation workflow that includes calibration analysis.

"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from astra.utils import cfg, ensure_parent_dir
from astra.utils import save_figure
from astra.evaluation.utils import prepare_model
from astra.visualize.evaluation import plot_evaluation

from astra.evaluation.predictive_performance import (
    TimeDependentEvaluator, time_to_step, format_step_label,
    plot_multiple_roc_pr_curves, generate_time_thresholds, plot_time_metrics)

logger = logging.getLogger(__name__)

# ── Figure style constants for readability ───────────────────────────────
_FIG_STYLE = dict(
    title=16,
    axis_label=14,
    tick_label=12,
    legend=12,
    annotation=11,
    suptitle=18,
)


def run_eval_with_calibration(
    data, 
    model_name: str, 
    multicurve: bool = True, 
    comprehensive_eval: bool = True,
    calibration_analysis: bool = True
):
    """
    Enhanced evaluation with calibration analysis.
    
    This is a drop-in replacement for run_eval() that adds calibration plots.
    
    Args:
        data: Output from prepare_data_and_dls()
        model_name: Name of saved model
        multicurve: Plot multiple ROC/PR curves
        comprehensive_eval: Run time-dependent evaluation
        calibration_analysis: Add calibration plots (NEW!)
        
    Returns:
        results: Time-dependent results (if comprehensive_eval=True)
        preds_df: Predictions DataFrame (if comprehensive_eval=True)
        cal_metrics: Calibration metrics (if calibration_analysis=True)
    """

    
    mixed_dls = data["mixed_dls"]
    holdout_mixed_dls = data["holdout_mixed_dls"]
    holdout = data["holdout"]
    
    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    model, device = prepare_model(data, cfg)
    logger.info("Model loaded and moved to device")

    # ============================================================================
    # BASELINE EVALUATION (Full Time Series)
    # ============================================================================
    logger.info("Running baseline evaluation with full time series...")
    from astra.evaluation.predictive_performance import _get_predictions
    preds, targs = _get_predictions(model, holdout_mixed_dls.train, device)
    
    # Plot and save baseline evaluation
    evalplt = plot_evaluation(preds[:, 1], targs, cfg["target"])
    save_figure(evalplt, f"baseline_eval_{model_name}", save_dir=f'reports/eval/{model_name}')
    logger.info("✓ Baseline ROC/PR plot saved")
    
    # ============================================================================
    # CALIBRATION ANALYSIS (NEW!)
    # ============================================================================
    cal_metrics = None
    if calibration_analysis:
        logger.info("="*80)
        logger.info("CALIBRATION ANALYSIS")
        logger.info("="*80)
        
        cal_metrics = add_calibration_to_eval(
            model,
            holdout_mixed_dls,
            device=device,
            model_name=model_name,
            save_dir='reports/calibration'
        )

        logger.info(f"✓ Calibration plot saved to reports/calibration/calibration_{model_name}.png")
        logger.info(f"  ECE: {cal_metrics['ece']:.4f}")
        logger.info(f"  Brier Score: {cal_metrics['brier_score']:.4f}")
    
    # ============================================================================
    # TIME-DEPENDENT EVALUATION (same as before)
    # ============================================================================
    evaluator = TimeDependentEvaluator(data, model, cfg, device=device, active_only=True)
    
    if multicurve:
        logger.info("Creating multiple ROC/PR curves at key timepoints...")

        
        key_timepoints = [
            time_to_step(1, 'h'),
            time_to_step(6, 'h'),
            time_to_step(12, 'h'),
            time_to_step(24, 'h'),
            time_to_step(72, 'h'),
            time_to_step(7, 'D'),
            time_to_step(14, 'D'),
            time_to_step(30, 'D')
        ]
        
        key_timepoints = [t for t in key_timepoints if t is not None]
        key_timepoints.reverse()
        labels = [format_step_label(step) for step in key_timepoints]
        
        fig_curves = plot_multiple_roc_pr_curves(
            evaluator,
            key_timepoints,
            labels=labels
        )
        save_figure(fig_curves, f"multi_curves_{model_name}", save_dir=f'reports/eval/{model_name}')
        logger.info("✓ Multiple curves plot saved")
    
    if comprehensive_eval:
        logger.info("="*80)
        logger.info("COMPREHENSIVE TIME-DEPENDENT EVALUATION")
        logger.info("="*80)
        
        censor_thresholds = generate_time_thresholds(
            cut_hours=72,
            step_hours=1,
            step_days=1
        )
        
        logger.info(f"Generated {len(censor_thresholds)} time thresholds")
        
        # Run evaluation
        results, preds_df = evaluator.evaluate_over_time_ultra_fast(
            censor_thresholds,
            save_predictions=True,
            model_name=model_name
        )
        
        if not results:
            logger.error("No valid results from time-dependent evaluation!")
            return None, None, cal_metrics
        
        logger.info(f"✓ Evaluated at {len(results)} time points")
        
        # Save predictions
        ensure_parent_dir(f'data/processed/preds_df_{model_name}.csv')
        preds_df.to_csv(f'data/processed/preds_df_{model_name}.csv', index=False)
        logger.info(f"✓ Predictions saved to CSV")
        
        # ========================================================================
        # CALIBRATION OVER TIME (NEW!)
        # ========================================================================
        if calibration_analysis:
            logger.info("="*80)
            logger.info("CALIBRATION OVER TIME ANALYSIS")
            logger.info("="*80)
            
            # Add true labels to predictions DataFrame
            logger.info("Adding true labels to predictions DataFrame...")
            preds_df['true_label'] = preds_df['PID'].map(
                dict(zip(holdout.base.PID, holdout.base[cfg["target"]]))
            )
            
            # Plot calibration over time
            logger.info("Generating calibration over time plot...")
            fig_cal_time = plot_calibration_over_time(
                preds_df,
                n_bins=4,
                save_path=f'reports/calibration/calibration_over_time_{model_name}.png'
            )
            logger.info(f"✓ Calibration over time plot saved")
        
        # ========================================================================
        # TIME METRICS PLOT
    
        
        logger.info("Creating time-dependent metrics plot...")
        fig_time = plot_time_metrics(results, cut_hours=72)
        save_figure(fig_time, f"time_metrics_{model_name}", save_dir=f'reports/eval/{model_name}')
        logger.info("✓ Time metrics plot saved")
        
        # ========================================================================
        # SUMMARY
        # ========================================================================
        logger.info("="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Baseline Performance:")
        logger.info(f"  Predictions: {len(preds)} samples")
        
        if cal_metrics:
            logger.info(f"Calibration Metrics:")
            logger.info(f"  ECE: {cal_metrics['ece']:.4f} (lower is better)")
            logger.info(f"  Brier Score: {cal_metrics['brier_score']:.4f} (lower is better)")
        
        logger.info(f"Time-dependent Evaluation:")
        logger.info(f"  Time points evaluated: {len(results)}")
        logger.info(f"  Predictions saved: {len(preds_df)} patient-timepoint pairs")
        
        # Print key metrics at important time points
        if multicurve:
            logger.info("\nPerformance at key time points:")
            for step in key_timepoints[::-1]:
                matching = [r for r in results if r.censor_step == step]
                if matching:
                    r = matching[0]
                    logger.info(
                        f"  {format_step_label(step):>12s}: "
                        f"AUROC={r.auroc:.3f} [{r.auroc_ci[0]:.3f}-{r.auroc_ci[1]:.3f}], "
                        f"AUPRC={r.auprc:.3f} [{r.auprc_ci[0]:.3f}-{r.auprc_ci[1]:.3f}]"
                    )
        
        logger.info("="*80)
        logger.info("✓ Comprehensive evaluation with calibration complete!")
        logger.info("="*80)
        
        return results, preds_df, cal_metrics
    
    else:
        logger.info("Skipping comprehensive evaluation")
        return None, None, cal_metrics


# ============================================================================
# STANDALONE CALIBRATION ANALYSIS
# ============================================================================

def analyze_calibration_only(data, model_name: str):
    """
    Run only calibration analysis (no full evaluation).
    
    Useful for quick calibration checks on already-evaluated models.
    
    Args:
        data: Output from prepare_data_and_dls()
        model_name: Name of saved model
        
    Returns:
        cal_metrics: Dictionary with calibration metrics
    """

    holdout_mixed_dls = data["holdout_mixed_dls"]

    # Load model
    model, device = prepare_model(data, cfg)
    logger.info("Model loaded")

    # Run calibration analysis
    cal_metrics = add_calibration_to_eval(
        model,
        holdout_mixed_dls,
        device=device,
        model_name=model_name,
        save_dir='reports/calibration'
    )

    return cal_metrics


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    logger.info(__doc__)
    logger.info("\n" + "="*80)
    logger.info("USAGE")
    logger.info("="*80)
    logger.info("""
# Option 1: Full evaluation with calibration (recommended)
from evaluation_with_calibration import run_eval_with_calibration

results, preds_df, cal_metrics = run_eval_with_calibration(
    data,
    model_name="my_model",
    multicurve=True,
    comprehensive_eval=True,
    calibration_analysis=True  # NEW!
)

# This creates:
# - reports/baseline_eval_my_model.png (ROC/PR curves)
# - reports/calibration_my_model.png (NEW!)
# - reports/calibration_table_my_model.csv (NEW!)
# - reports/multi_curves_my_model.png (ROC/PR at different times)
# - reports/time_metrics_my_model.png (Performance over time)
# - reports/calibration_over_time_my_model.png (NEW!)


# Option 2: Quick calibration check only
from evaluation_with_calibration import analyze_calibration_only

cal_metrics = analyze_calibration_only(data, model_name="my_model")

print(f"ECE: {cal_metrics['ece']:.4f}")
print(f"Brier Score: {cal_metrics['brier_score']:.4f}")


# Option 3: Compare calibration of multiple models
from calibration_plots import plot_calibration_comparison
from astra.evaluation.utils import prepare_model
from astra.evaluation.predictive_performance import _get_predictions
import torch

models = ["model1", "model2", "model3"]
y_true_list = []
y_pred_list = []

for model_name in models:
    cfg["model_name"] = model_name
    model, device = prepare_model(data, cfg)
    preds, targs = _get_predictions(model, holdout_mixed_dls.valid, device)
    y_pred_list.append(preds[:, 1].cpu().numpy())
    y_true_list.append(targs.cpu().numpy())

fig = plot_calibration_comparison(
    y_true_list,
    y_pred_list,
    model_names=models,
    save_path="reports/calibration_comparison.png"
)
    """)
# calibration_plots.py
"""
Comprehensive calibration analysis for binary classification models.

Provides multiple calibration visualizations:
1. Reliability diagram (calibration curve)
2. Calibration histogram (distribution of predicted probabilities)
3. Expected Calibration Error (ECE) calculation
4. Brier score
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd


def calculate_ece(y_true, y_pred, n_bins=4):
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual frequencies.
    Lower is better (0 = perfect calibration).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        
    Returns:
        ece: Expected Calibration Error
        bin_data: Dictionary with per-bin statistics
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            
            # ECE contribution from this bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'bin_center': (bin_lower + bin_upper) / 2,
                'n_samples': in_bin.sum(),
                'prop_samples': prop_in_bin,
                'avg_confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'calibration_error': avg_confidence_in_bin - accuracy_in_bin
            })
        else:
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'bin_center': (bin_lower + bin_upper) / 2,
                'n_samples': 0,
                'prop_samples': 0,
                'avg_confidence': np.nan,
                'accuracy': np.nan,
                'calibration_error': np.nan
            })
    
    return ece, bin_data


def plot_calibration_curve(
    y_true,
    y_pred,
    n_bins=4,
    strategy='uniform',
    title=None,
    save_path=None
):
    """
    Plot calibration curve (reliability diagram) with additional statistics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities for positive class
        n_bins: Number of bins
        strategy: 'uniform' or 'quantile' binning
        title: Optional title
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # =========================================================================
    # LEFT PLOT: Calibration Curve (Reliability Diagram)
    # =========================================================================
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, 
        y_pred, 
        n_bins=n_bins,
        strategy=strategy
    )
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Plot actual calibration
    ax1.plot(
        mean_predicted_value, 
        fraction_of_positives, 
        marker='o',
        linewidth=2,
        markersize=8,
        color='#2E86AB',
        label='Model Calibration'
    )
    
    # Add confidence bars (based on bin sizes)
    # Calculate bin sizes for error bars
    bin_counts = np.zeros(len(mean_predicted_value))
    bin_boundaries = np.linspace(0, 1, n_bins + 1) if strategy == 'uniform' else None
    
    if strategy == 'uniform':
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            bin_counts[i] = in_bin.sum()
    
    # Wilson score interval for confidence (approximation)
    for i, (pred_val, true_frac) in enumerate(zip(mean_predicted_value, fraction_of_positives)):
        if bin_counts[i] > 0:
            n = bin_counts[i]
            # Simple standard error
            se = np.sqrt(true_frac * (1 - true_frac) / n) if n > 0 else 0
            ax1.errorbar(
                pred_val, 
                true_frac, 
                yerr=1.96*se,  # 95% CI
                color='#2E86AB',
                alpha=0.3,
                capsize=5
            )
    
    # Calculate metrics
    ece, bin_data = calculate_ece(y_true, y_pred, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_pred)
    
    # Add text box with metrics
    textstr = f'ECE: {ece:.4f}\nBrier Score: {brier:.4f}\nBins: {n_bins}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(
        0.05, 0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=_FIG_STYLE['annotation'],
        verticalalignment='top',
        bbox=props
    )

    ax1.set_xlabel('Mean Predicted Probability', fontsize=_FIG_STYLE['axis_label'])
    ax1.set_ylabel('Fraction of Positives (Actual)', fontsize=_FIG_STYLE['axis_label'])
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax1.legend(loc='upper left', fontsize=_FIG_STYLE['legend'])
    ax1.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal', adjustable='box')
    
    # =========================================================================
    # RIGHT PLOT: Histogram of Predicted Probabilities
    # =========================================================================
    
    # Separate by true class
    y_pred_pos = y_pred[y_true == 1]
    y_pred_neg = y_pred[y_true == 0]
    
    # Plot histograms
    ax2.hist(
        y_pred_neg, 
        bins=30, 
        alpha=0.6, 
        color='#FF6B6B',
        label=f'Negative (n={len(y_pred_neg)})',
        density=True
    )
    ax2.hist(
        y_pred_pos, 
        bins=30, 
        alpha=0.6, 
        color='#4ECDC4',
        label=f'Positive (n={len(y_pred_pos)})',
        density=True
    )
    
    # Add vertical line at 0.5
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Threshold=0.5')
    
    # Add statistics
    mean_pred = y_pred.mean()
    median_pred = np.median(y_pred)
    
    ax2.axvline(mean_pred, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean={mean_pred:.3f}')
    
    ax2.set_xlabel('Predicted Probability', fontsize=_FIG_STYLE['axis_label'])
    ax2.set_ylabel('Density', fontsize=_FIG_STYLE['axis_label'])
    ax2.set_title('Distribution of Predicted Probabilities', fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax2.legend(loc='upper right', fontsize=_FIG_STYLE['legend'])
    ax2.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([0, 1])

    # Overall title
    if title:
        fig.suptitle(title, fontsize=_FIG_STYLE['suptitle'], fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration plot saved to {save_path}")
    
    return fig


def plot_calibration_comparison(
    y_true_list,
    y_pred_list,
    model_names,
    n_bins=4,
    title="Model Calibration Comparison",
    save_path=None
):
    """
    Compare calibration curves for multiple models.
    
    Args:
        y_true_list: List of true labels (one per model)
        y_pred_list: List of predicted probabilities (one per model)
        model_names: List of model names
        n_bins: Number of bins
        title: Plot title
        save_path: Optional save path
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Plot each model
    for i, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, 
            y_pred, 
            n_bins=n_bins,
            strategy='uniform'
        )
        
        ece, _ = calculate_ece(y_true, y_pred, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_pred)
        
        color = colors[i % len(colors)]
        ax.plot(
            mean_predicted_value, 
            fraction_of_positives, 
            marker='o',
            linewidth=2,
            markersize=8,
            color=color,
            label=f'{name} (ECE={ece:.3f}, Brier={brier:.3f})'
        )
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylabel('Fraction of Positives', fontsize=_FIG_STYLE['axis_label'])
    ax.set_title(title, fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax.legend(loc='upper left', fontsize=_FIG_STYLE['legend'])
    ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration comparison saved to {save_path}")
    
    return fig


def plot_calibration_over_time(
    preds_df: pd.DataFrame,
    n_bins=4,
    time_points=None,
    save_path=None
):
    """
    Plot how calibration changes over different time censoring points.
    
    Args:
        preds_df: DataFrame with columns ['PID', 'censor_step', 'time_hours', 'pred', 'true_label']
        n_bins: Number of bins for calibration
        time_points: Optional list of specific censor_steps to plot
        save_path: Optional save path
        
    Returns:
        fig: Matplotlib figure
    """
    if time_points is None:
        # Select evenly spaced time points
        unique_steps = sorted(preds_df['censor_step'].unique())
        step = max(len(unique_steps) // 8, 1)
        time_points = unique_steps[::step]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
    
    # =========================================================================
    # LEFT: Calibration curves at different time points
    # =========================================================================
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect', alpha=0.7)
    
    ece_over_time = []
    time_labels = []
    
    for i, censor_step in enumerate(time_points):
        df_step = preds_df[preds_df['censor_step'] == censor_step]
        
        if len(df_step) > 0 and 'true_label' in df_step.columns:
            y_true = df_step['true_label'].values
            y_pred = df_step['pred'].values
            
            # Skip if only one class
            if len(np.unique(y_true)) < 2:
                continue
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy='uniform'
            )
            
            ece, _ = calculate_ece(y_true, y_pred, n_bins=n_bins)
            ece_over_time.append(ece)
            
            # Format time label
            time_hours = df_step['time_hours'].iloc[0]
            if time_hours < 24:
                time_label = f'{int(time_hours)}h'
            else:
                time_label = f'{int(time_hours/24)}d'
            time_labels.append(time_label)
            
            ax1.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker='o',
                linewidth=2,
                markersize=6,
                color=colors[i],
                label=f'{time_label} (ECE={ece:.3f})',
                alpha=0.8
            )
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=_FIG_STYLE['axis_label'])
    ax1.set_ylabel('Fraction of Positives', fontsize=_FIG_STYLE['axis_label'])
    ax1.set_title('Calibration Curves Over Time', fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax1.legend(loc='upper left', fontsize=_FIG_STYLE['legend'], ncol=2)
    ax1.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal', adjustable='box')

    # =========================================================================
    # RIGHT: ECE over time
    # =========================================================================
    if len(ece_over_time) > 0:
        ax2.plot(
            range(len(ece_over_time)),
            ece_over_time,
            marker='o',
            linewidth=2,
            markersize=8,
            color='#2E86AB'
        )
        ax2.set_xticks(range(len(time_labels)))
        ax2.set_xticklabels(time_labels, rotation=45, fontsize=_FIG_STYLE['tick_label'])
        ax2.set_xlabel('Time Available', fontsize=_FIG_STYLE['axis_label'])
        ax2.set_ylabel('Expected Calibration Error (ECE)', fontsize=_FIG_STYLE['axis_label'])
        ax2.set_title('Model Calibration Over Time', fontsize=_FIG_STYLE['title'], fontweight='bold')
        ax2.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration over time plot saved to {save_path}")
    
    return fig


def calibration_summary_table(y_true, y_pred, n_bins=4):
    """
    Generate a summary table of calibration statistics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        DataFrame with calibration statistics per bin
    """
    ece, bin_data = calculate_ece(y_true, y_pred, n_bins=n_bins)
    
    df = pd.DataFrame(bin_data)
    df = df[df['n_samples'] > 0]  # Remove empty bins
    
    # Format for display
    df['bin_range'] = df.apply(
        lambda row: f"({row['bin_lower']:.2f}, {row['bin_upper']:.2f}]", 
        axis=1
    )
    
    # Reorder columns
    display_cols = [
        'bin_range', 'n_samples', 'prop_samples', 
        'avg_confidence', 'accuracy', 'calibration_error'
    ]
    
    df_display = df[display_cols].copy()
    df_display.columns = [
        'Bin Range', 'N Samples', 'Prop Samples',
        'Avg Confidence', 'Accuracy', 'Cal. Error'
    ]
    
    # Add summary row
    summary_row = pd.DataFrame([{
        'Bin Range': 'OVERALL',
        'N Samples': len(y_true),
        'Prop Samples': 1.0,
        'Avg Confidence': y_pred.mean(),
        'Accuracy': y_true.mean(),
        'Cal. Error': ece
    }])
    
    df_display = pd.concat([df_display, summary_row], ignore_index=True)
    
    return df_display


# ============================================================================
# CONVENIENCE FUNCTION FOR EVALUATION INTEGRATION
# ============================================================================

def add_calibration_to_eval(
    model,
    holdout_mixed_dls,
    device: str = 'cuda',
    model_name: str = '',
    save_dir: str = 'reports/calibration'
):
    """
    Add calibration plots to existing evaluation workflow.

    Args:
        model: Trained nn.Module (already on device)
        holdout_mixed_dls: Holdout dataloaders
        device: Device string
        model_name: Model name for saving
        save_dir: Directory to save plots

    Returns:
        dict with calibration metrics
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Generating calibration analysis...")

    from astra.evaluation.predictive_performance import _get_predictions
    preds, targets = _get_predictions(model, holdout_mixed_dls.train, device)

    y_pred = preds[:, 1].cpu().numpy()
    y_true = targets.cpu().numpy()
    
    # Calculate metrics
    ece, bin_data = calculate_ece(y_true, y_pred, n_bins=4)
    brier = brier_score_loss(y_true, y_pred)
    
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    
    # Generate calibration plot
    fig = plot_calibration_curve(
        y_true, 
        y_pred,
        n_bins=4,
        title=f"Model Calibration: {model_name}",
        save_path=f"{save_dir}/calibration_{model_name}.png"
    )
    
    # Generate summary table
    cal_table = calibration_summary_table(y_true, y_pred, n_bins=4)
    logger.debug("Calibration Summary Table:\n%s", cal_table.to_string(index=False))

    # Save table
    cal_table.to_csv(f"{save_dir}/calibration_table_{model_name}.csv", index=False)
    logger.info(f"Calibration table saved to {save_dir}/calibration_table_{model_name}.csv")
    
    return {
        'ece': ece,
        'brier_score': brier,
        'calibration_table': cal_table,
        'bin_data': bin_data
    }


def plot_survival_calibration(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    survival_probs: np.ndarray,
    eval_time: int,
    n_bins: int = 10,
    title: str = "Survival Calibration",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """D-calibration plot: predicted S(t) vs observed survival in risk bins.

    Args:
        event_times: [N] time to event or censoring (steps).
        event_indicators: [N] 1 = event, 0 = censored.
        survival_probs: [N, seq_len] survival probabilities.
        eval_time: Step index to evaluate calibration at.
        n_bins: Number of calibration bins.
        title: Plot title.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure.
    """
    from astra.evaluation.survival_metrics import dcalibration

    cal = dcalibration(event_times, event_indicators, survival_probs, eval_time, n_bins)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    predicted = cal["predicted_survival"]
    observed = cal["observed_survival"]
    counts = cal["bin_counts"]

    if predicted and observed:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        sizes = [max(20, c * 2) for c in counts]
        ax.scatter(predicted, observed, s=sizes, alpha=0.7, zorder=5)
        ax.plot(predicted, observed, 'b-', alpha=0.5)

        for p, o, c in zip(predicted, observed, counts):
            ax.annotate(f'n={c}', (p, o), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel('Predicted S(t)')
    ax.set_ylabel('Observed survival fraction')
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info(__doc__)
    logger.info("\n" + "="*80)
    logger.info("USAGE EXAMPLES")
    logger.info("="*80)
    logger.info("""
# 1. Basic calibration plot
from calibration_plots import plot_calibration_curve

fig = plot_calibration_curve(
    y_true, 
    y_pred,
    n_bins=10,
    title="My Model Calibration",
    save_path="calibration.png"
)

# 2. Add to evaluation workflow
from calibration_plots import add_calibration_to_eval

cal_metrics = add_calibration_to_eval(
    learn, 
    holdout_mixed_dls, 
    model_name="my_model",
    save_dir="reports/"
)

# 3. Compare multiple models
from calibration_plots import plot_calibration_comparison

fig = plot_calibration_comparison(
    y_true_list=[y_true1, y_true2],
    y_pred_list=[y_pred1, y_pred2],
    model_names=["Model 1", "Model 2"],
    save_path="calibration_comparison.png"
)

# 4. Calibration over time (requires preds_df with 'true_label' column)
from calibration_plots import plot_calibration_over_time

# First, add true labels to preds_df
preds_df['true_label'] = preds_df['PID'].map(
    dict(zip(holdout.base.PID, holdout.base.Deceased30d))
)

fig = plot_calibration_over_time(
    preds_df,
    save_path="calibration_over_time.png"
)
    """)