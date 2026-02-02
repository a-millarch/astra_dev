# predictive_performance_uncertainty.py
"""
Extension of predictive_performance.py with uncertainty quantification.

Adds epistemic uncertainty estimates to time-dependent predictions using:
1. Monte-Carlo Dropout for epistemic uncertainty
2. Conformal Prediction for statistically guaranteed intervals
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from astra.utils import cfg, logger
from astra.evaluation.utils import save_figure
from astra.evaluation.predictive_performance import (
    TimeDependentEvaluator,
    TimeMetricResult,
    step_to_time,
    format_step_label
)
from astra.evaluation.uncertainty import (
    UncertaintyQuantifier,
    UncertaintyResult,
    calculate_calibration_metrics,
    analyze_uncertainty_by_time
)


@dataclass
class TimeMetricResultWithUncertainty:
    """Extended TimeMetricResult with uncertainty metrics"""
    # Original metrics
    time_min: float
    time_hours: float
    time_days: float
    censor_step: int
    auroc: float
    auroc_ci: Tuple[float, float]
    auprc: float
    auprc_ci: Tuple[float, float]
    n_samples: int
    n_positive: int

    # Uncertainty metrics
    mean_epistemic_uncertainty: float  # Average std dev across patients
    mean_entropy: float               # Average entropy
    mean_bald: float                  # Average BALD score
    mean_conformal_width: Optional[float] = None  # Average interval width

    # Calibration metrics
    ece: Optional[float] = None  # Expected Calibration Error
    uncertainty_separation: Optional[float] = None  # Errors vs correct


class TimeDependentEvaluatorWithUncertainty(TimeDependentEvaluator):
    """
    Extended evaluator that adds uncertainty quantification to time-dependent evaluation.

    Provides:
    - Individual prediction uncertainty for each patient at each timepoint
    - Calibrated prediction intervals
    - Temporal analysis of uncertainty evolution
    - Support for both cohort-level and individual patient analysis
    """

    def __init__(
        self,
        data: dict,
        learn,
        cfg: dict,
        n_mc_samples: int = 30,
        alpha: float = 0.1,
        use_conformal: bool = True
    ):
        """
        Args:
            data: Output from prepare_data_and_dls()
            learn: Trained fastai Learner (already patched)
            cfg: Configuration dictionary
            n_mc_samples: Number of MC dropout samples (10-30 recommended)
            alpha: Significance level for conformal prediction (0.1 = 90% CI)
            use_conformal: Whether to use conformal prediction
        """
        super().__init__(data, learn, cfg)

        self.uq = UncertaintyQuantifier(
            n_mc_samples=n_mc_samples,
            alpha=alpha,
            use_conformal=use_conformal
        )
        self.n_mc_samples = n_mc_samples
        self.use_conformal = use_conformal

    def get_holdout_pids(self, max_samples: Optional[int] = None) -> List:
        """
        Get PIDs from holdout dataset in dataloader order.

        Args:
            max_samples: Maximum number of PIDs to return (None = all)

        Returns:
            List of PIDs in the order they appear in the holdout dataloader
        """
        pids = self.data["holdout"].tab_df['PID'].tolist()
        if max_samples is not None and len(pids) > max_samples:
            pids = pids[:max_samples]
        return pids

    def get_sample_indices_for_pids(
        self,
        pids: List,
        target_pids: List
    ) -> Tuple[List[int], List]:
        """
        Find sample indices for given PIDs.

        Args:
            pids: List of all PIDs in dataloader order
            target_pids: List of PIDs to find

        Returns:
            Tuple of (indices, found_pids) - only includes PIDs that were found
        """
        indices = []
        found_pids = []

        for target_pid in target_pids:
            try:
                idx = pids.index(target_pid)
                indices.append(idx)
                found_pids.append(target_pid)
            except ValueError:
                logger.warning(f"PID {target_pid} not found in holdout data")

        return indices, found_pids

    def calibrate_uncertainty(
        self,
        cal_fraction: float = 0.3,
        random_state: int = 42
    ):
        """
        Calibrate conformal predictor using a fraction of holdout set.

        This splits the holdout set into:
        - Calibration set (cal_fraction, e.g., 30%)
        - Test set (1 - cal_fraction, e.g., 70%)

        Args:
            cal_fraction: Fraction of holdout to use for calibration
            random_state: Random seed for reproducible splits
        """
        if not self.use_conformal:
            logger.info("Conformal prediction disabled, skipping calibration")
            return

        logger.info(f"Calibrating uncertainty quantifier on {cal_fraction*100:.0f}% of holdout...")

        # Split holdout data
        # Convert holdout_y to numpy array for indexing
        holdout_y_array = np.array(self.holdout_y)
        n_holdout = len(holdout_y_array)
        n_cal = int(n_holdout * cal_fraction)

        # Random split
        np.random.seed(random_state)
        indices = np.random.permutation(n_holdout)
        cal_indices = indices[:n_cal]

        # Create calibration dataloader (using full time series)
        from tsai.data.core import get_ts_dls
        from tsai.data.tabular import get_tabular_dls
        from tsai.data.mixed import get_mixed_dls

        # Initialize holdout_tab_normalized if not already done
        if self.holdout_tab_normalized is None:
            if self.num_cols and self.data.get("tab_scaler") is not None:
                self.holdout_tab_normalized = self.holdout.tab_df.copy()
                self.holdout_tab_normalized[self.num_cols] = self.data["tab_scaler"].transform(
                    self.holdout.tab_df[self.num_cols]
                )
            else:
                self.holdout_tab_normalized = self.holdout.tab_df

        # Calibration set continuous TS
        cal_ts_dls = get_ts_dls(
            self.holdout_X_normalized[cal_indices],
            holdout_y_array[cal_indices],
            splits=None,
            tfms=self.tfms,
            batch_tfms=None,
            bs=self.bs,
            drop_last=False,
            shuffle=False
        )

        # Calibration set tabular
        cal_tab_df = self.holdout_tab_normalized.iloc[cal_indices]
        cal_tab_dls = get_tabular_dls(
            cal_tab_df,
            procs=self.procs,
            cat_names=self.cat_cols.copy(),
            cont_names=self.num_cols.copy(),
            y_names=self.target,
            splits=None,
            drop_last=False,
            shuffle=False,
            classes=self.classes
        )

        # Calibration set categorical TS
        cal_ts_cat_dls = get_ts_dls(
            self.holdout_X_multi_hot[cal_indices].astype(np.int64),
            holdout_y_array[cal_indices],
            splits=None,
            bs=self.bs,
            shuffle=False
        )
        cal_ts_cat_dls.ts_cat_dims = self.data["ts_cat_dls"].ts_cat_dims
        cal_ts_cat_dls.X_multi_hot = self.holdout_X_multi_hot[cal_indices]

        # Combine
        cal_mixed_dls = get_mixed_dls(
            cal_ts_dls,
            cal_tab_dls,
            cal_ts_cat_dls,
            bs=self.bs,
            shuffle_valid=False
        )

        # Calibrate
        self.uq.calibrate(
            self.learn,
            cal_mixed_dls.train,
            holdout_y_array[cal_indices]
        )

        # Store calibration indices for later exclusion
        self.cal_indices = set(cal_indices)

        logger.info("✓ Uncertainty calibration complete")

    def evaluate_at_timestep_with_uncertainty(
        self,
        censor_step: int,
        exclude_cal_set: bool = True
    ) -> Optional[Tuple[TimeMetricResultWithUncertainty, UncertaintyResult]]:
        """
        Evaluate model at a time point WITH uncertainty quantification.

        Args:
            censor_step: Time step to censor at
            exclude_cal_set: Whether to exclude calibration set from evaluation

        Returns:
            Tuple of (TimeMetricResultWithUncertainty, UncertaintyResult) or None
        """
        # Create censored dataloaders
        dls = self.create_censored_dataloaders_fast(censor_step)
        if dls is None:
            return None

        # Get predictions with uncertainty
        uncertainty_result = self.uq.predict_with_uncertainty(
            self.learn,
            dls.train
        )

        # Get targets
        with torch.no_grad():
            _, targets = self.learn.get_preds(dl=dls.train)
        ys = targets.cpu().numpy()

        # Exclude calibration set if requested
        if exclude_cal_set and hasattr(self, 'cal_indices'):
            test_mask = np.array([i not in self.cal_indices for i in range(len(ys))])
            y_preds = uncertainty_result.pred_mean[test_mask]
            ys = ys[test_mask]

            # Update uncertainty result to only include test set
            uncertainty_result.pred_mean = uncertainty_result.pred_mean[test_mask]
            uncertainty_result.pred_std = uncertainty_result.pred_std[test_mask]
            uncertainty_result.entropy = uncertainty_result.entropy[test_mask]
            uncertainty_result.bald = uncertainty_result.bald[test_mask]
            if uncertainty_result.conformal_lower is not None:
                uncertainty_result.conformal_lower = uncertainty_result.conformal_lower[test_mask]
                uncertainty_result.conformal_upper = uncertainty_result.conformal_upper[test_mask]
                uncertainty_result.conformal_width = uncertainty_result.conformal_width[test_mask]
        else:
            y_preds = uncertainty_result.pred_mean

        # Check if we have both classes
        if ys.sum() == 0 or ys.sum() == len(ys):
            logger.warning(f"Skipping censor_step={censor_step}: only one class in targets")
            return None

        # Calculate performance metrics
        from astra.evaluation.utils import calculate_roc_auc_ci, calculate_average_precision_ci
        auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(ys, y_preds)
        auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(ys, y_preds)

        # Calculate uncertainty metrics
        mean_epistemic_uncertainty = uncertainty_result.pred_std.mean()
        mean_entropy = uncertainty_result.entropy.mean()
        mean_bald = uncertainty_result.bald.mean()
        mean_conformal_width = (
            uncertainty_result.conformal_width.mean()
            if uncertainty_result.conformal_width is not None
            else None
        )

        # Calculate calibration metrics
        cal_metrics = calculate_calibration_metrics(
            y_preds,
            uncertainty_result.pred_std,
            ys,
            n_bins=10
        )

        # Convert step to time
        time_min = step_to_time(censor_step)
        if time_min is None:
            logger.warning(f"Could not convert step {censor_step} to time")
            return None

        result = TimeMetricResultWithUncertainty(
            time_min=time_min,
            time_hours=time_min / 60,
            time_days=time_min / (24 * 60),
            censor_step=censor_step,
            auroc=auroc,
            auroc_ci=(auroc_lower, auroc_upper),
            auprc=auprc,
            auprc_ci=(auprc_lower, auprc_upper),
            n_samples=len(ys),
            n_positive=int(ys.sum()),
            mean_epistemic_uncertainty=mean_epistemic_uncertainty,
            mean_entropy=mean_entropy,
            mean_bald=mean_bald,
            mean_conformal_width=mean_conformal_width,
            ece=cal_metrics['ece'],
            uncertainty_separation=cal_metrics['uncertainty_separation']
        )

        return result, uncertainty_result

    def evaluate_over_time_with_uncertainty(
        self,
        censor_steps: List[int],
        save_predictions: bool = True,
        model_name: Optional[str] = None,
        exclude_cal_set: bool = True,
        pids: Optional[List] = None
    ) -> Tuple[List[TimeMetricResultWithUncertainty], Optional[pd.DataFrame]]:
        """
        Evaluate over time WITH uncertainty quantification.

        Supports both cohort-level and individual patient analysis.

        Args:
            censor_steps: List of time steps to evaluate at
            save_predictions: Whether to save predictions with uncertainties
            model_name: Model name for saving
            exclude_cal_set: Whether to exclude calibration set (ignored if pids specified)
            pids: Optional list of specific PIDs to evaluate. If None, evaluates all patients.

        Returns:
            Tuple of (results list, predictions DataFrame)
        """
        import time

        results = []
        preds_over_time = [] if save_predictions else None
        patient_ids = self.holdout.base.PID.values

        # Determine which patients to include
        if pids is not None:
            # Individual patient mode
            all_pids = self.get_holdout_pids()
            indices, found_pids = self.get_sample_indices_for_pids(all_pids, pids)

            if not found_pids:
                logger.error(f"None of the specified PIDs found in holdout data")
                logger.info(f"Available PIDs (first 10): {all_pids[:10]}")
                return [], None

            test_mask = np.zeros(len(patient_ids), dtype=bool)
            test_mask[indices] = True
            eval_patient_ids = patient_ids[test_mask]

            logger.info(f"Evaluating {len(found_pids)} specific patient(s): {found_pids}")
            if len(found_pids) < len(pids):
                missing = set(pids) - set(found_pids)
                logger.warning(f"Missing PIDs: {missing}")

        elif exclude_cal_set and hasattr(self, 'cal_indices'):
            # Cohort mode, exclude calibration set
            test_mask = np.array([i not in self.cal_indices for i in range(len(patient_ids))])
            eval_patient_ids = patient_ids[test_mask]
            logger.info(f"Evaluating on test set only: {test_mask.sum()} patients "
                       f"(excluded {len(self.cal_indices)} calibration patients)")
        else:
            # Cohort mode, all patients
            test_mask = np.ones(len(patient_ids), dtype=bool)
            eval_patient_ids = patient_ids

        logger.info(f"Time-dependent evaluation with uncertainty at {len(censor_steps)} points...")
        start_time = time.time()

        for i, censor_step in enumerate(censor_steps):
            # Progress logging
            if i % 10 == 0 or i == len(censor_steps) - 1:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_per_step = elapsed / i
                    remaining = avg_per_step * (len(censor_steps) - i)
                    logger.info(
                        f"Progress: {i+1}/{len(censor_steps)} ({100*i/len(censor_steps):.1f}%) "
                        f"- ~{remaining/60:.1f}min remaining"
                    )

            # Evaluate with uncertainty
            eval_result = self.evaluate_at_timestep_with_uncertainty(
                censor_step,
                exclude_cal_set=exclude_cal_set
            )

            if eval_result is not None:
                result, uncertainty_result = eval_result
                results.append(result)

                # Save predictions with uncertainties
                if save_predictions:
                    for j, (pid, pred, std, entropy, bald) in enumerate(zip(
                        eval_patient_ids,
                        uncertainty_result.pred_mean,
                        uncertainty_result.pred_std,
                        uncertainty_result.entropy,
                        uncertainty_result.bald
                    )):
                        pred_dict = {
                            "PID": pid,
                            "censor_step": censor_step,
                            "time_min": result.time_min,
                            "time_hours": result.time_hours,
                            "time_days": result.time_days,
                            "pred_mean": float(pred),
                            "epistemic_uncertainty": float(std),
                            "entropy": float(entropy),
                            "bald": float(bald)
                        }

                        # Add conformal intervals if available
                        if uncertainty_result.conformal_lower is not None:
                            pred_dict["conformal_lower"] = float(uncertainty_result.conformal_lower[j])
                            pred_dict["conformal_upper"] = float(uncertainty_result.conformal_upper[j])
                            pred_dict["conformal_width"] = float(uncertainty_result.conformal_width[j])

                        preds_over_time.append(pred_dict)

        total_time = time.time() - start_time
        logger.info(
            f"✓ Uncertainty evaluation complete: {len(results)}/{len(censor_steps)} successful "
            f"in {total_time/60:.1f} minutes ({total_time/len(censor_steps):.2f}s per step)"
        )

        # Save predictions
        if save_predictions and preds_over_time and model_name:
            preds_df = pd.DataFrame(preds_over_time)
            os.makedirs('models/eval', exist_ok=True)
            preds_df.to_pickle(f'models/eval/preds_uncertainty_{model_name}.pkl')
            preds_df.to_csv(f'data/processed/preds_uncertainty_{model_name}.csv', index=False)
            logger.info(f"Saved uncertainty predictions to models/eval/preds_uncertainty_{model_name}.pkl")
            return results, preds_df

        return results, None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_uncertainty_over_time(
    results: List[TimeMetricResultWithUncertainty],
    cut_hours: int = 72,
    max_days: int = 30
):
    """
    Plot how uncertainty evolves over time.

    Shows:
    - Epistemic uncertainty (std dev)
    - Predictive entropy
    - Conformal interval width (if available)
    - ECE (calibration)

    Args:
        results: List of TimeMetricResultWithUncertainty
        cut_hours: Hours cutoff for first plot
        max_days: Days range for second plot

    Returns:
        matplotlib Figure
    """
    if not results:
        raise ValueError("No results to plot")

    # Extract data
    times_h = np.array([r.time_hours for r in results])
    times_d = np.array([r.time_days for r in results])

    epistemic_unc = np.array([r.mean_epistemic_uncertainty for r in results])
    entropy = np.array([r.mean_entropy for r in results])
    ece = np.array([r.ece if r.ece is not None else np.nan for r in results])

    has_conformal = results[0].mean_conformal_width is not None
    if has_conformal:
        conformal_width = np.array([r.mean_conformal_width for r in results])

    # Create figure
    n_rows = 2 if has_conformal else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))

    # ========================================================================
    # Row 1: Epistemic Uncertainty (std dev)
    # ========================================================================

    # Hours view
    ax = axes[0, 0]
    mask_cut = times_h <= cut_hours
    ax.plot(times_h[mask_cut], epistemic_unc[mask_cut], 'o-', color='C0', label='Epistemic Uncertainty')
    ax.fill_between(times_h[mask_cut], 0, epistemic_unc[mask_cut], alpha=0.3, color='C0')
    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_xlim(0, cut_hours)
    ax.set_ylabel("Mean Std Dev", fontsize=11)
    ax.set_title("A) Epistemic Uncertainty Over Hours", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Days view
    ax = axes[0, 1]
    ax.plot(times_d, epistemic_unc, 'o-', color='C0', label='Epistemic Uncertainty')
    ax.fill_between(times_d, 0, epistemic_unc, alpha=0.3, color='C0')
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_xlim(0, max_days)
    ax.set_ylabel("Mean Std Dev", fontsize=11)
    ax.set_title("B) Epistemic Uncertainty Over Days", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 2: Calibration (ECE) and Entropy
    # ========================================================================

    # ECE over hours
    ax = axes[1, 0]
    mask_cut = times_h <= cut_hours
    ax.plot(times_h[mask_cut], ece[mask_cut], 's-', color='C3', label='ECE')
    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_xlim(0, cut_hours)
    ax.set_ylabel("Expected Calibration Error", fontsize=11)
    ax.set_title("C) Calibration Over Hours", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Poor calibration (>0.1)')
    ax.legend()

    # Entropy over days
    ax = axes[1, 1]
    ax.plot(times_d, entropy, '^-', color='C2', label='Entropy')
    ax.fill_between(times_d, 0, entropy, alpha=0.3, color='C2')
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_xlim(0, max_days)
    ax.set_ylabel("Mean Entropy", fontsize=11)
    ax.set_title("D) Predictive Entropy Over Days", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_patient_uncertainty_examples(
    preds_df: pd.DataFrame,
    n_patients: int = 6,
    max_days: int = 30
):
    """
    Plot individual patient prediction trajectories with uncertainty.

    Shows how predictions and uncertainty evolve for individual patients.

    Args:
        preds_df: DataFrame with columns [PID, time_days, pred_mean, epistemic_uncertainty, ...]
        n_patients: Number of patients to plot
        max_days: Maximum days to show

    Returns:
        matplotlib Figure
    """
    # Select diverse patients (high/medium/low final predictions)
    final_time = preds_df['time_days'].max()
    final_preds = preds_df[preds_df['time_days'] >= final_time * 0.9].groupby('PID')['pred_mean'].mean()

    # Get patients with diverse predictions
    sorted_pids = final_preds.sort_values().index
    n_total = len(sorted_pids)
    selected_pids = [
        sorted_pids[0],  # Lowest
        sorted_pids[n_total // 4],
        sorted_pids[n_total // 2],
        sorted_pids[3 * n_total // 4],
        sorted_pids[-1],  # Highest
        sorted_pids[n_total // 3]  # Extra
    ][:n_patients]

    # Create figure
    n_cols = 3
    n_rows = (n_patients + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_patients > 1 else [axes]

    for i, pid in enumerate(selected_pids):
        ax = axes[i]

        # Get patient data
        patient_data = preds_df[preds_df['PID'] == pid].sort_values('time_days')
        patient_data = patient_data[patient_data['time_days'] <= max_days]

        times = patient_data['time_days'].values
        preds = patient_data['pred_mean'].values
        unc = patient_data['epistemic_uncertainty'].values

        # Plot prediction with uncertainty
        ax.plot(times, preds, 'o-', color='C0', label='Prediction', linewidth=2)
        ax.fill_between(
            times,
            np.maximum(0, preds - 2*unc),
            np.minimum(1, preds + 2*unc),
            alpha=0.3,
            color='C0',
            label='±2σ (95% CI)'
        )

        # Add conformal intervals if available
        if 'conformal_lower' in patient_data.columns:
            lower = patient_data['conformal_lower'].values
            upper = patient_data['conformal_upper'].values
            ax.fill_between(
                times,
                lower,
                upper,
                alpha=0.2,
                color='C1',
                label='Conformal 90% CI'
            )

        ax.set_xlabel("Time (days)", fontsize=10)
        ax.set_ylabel("Predicted Risk", fontsize=10)
        ax.set_title(f"Patient {pid}", fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, max_days)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig
