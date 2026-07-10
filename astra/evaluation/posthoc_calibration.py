"""
Posthoc calibration pipeline for time-dependent mortality predictions.

Fits calibrators (isotonic regression, Platt scaling) on trainval predictions
at each evaluation timepoint, then evaluates calibrated predictions on holdout.

Usage:
    from astra.evaluation.posthoc_calibration import run_posthoc_calibration
    summary_df = run_posthoc_calibration(data, cfg)
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

from astra.utils import cfg as global_cfg, save_figure, ensure_parent_dir
from astra.data.mixed_dataloader import AstraMixedDataset, AstraMixedDataLoader
from astra.evaluation.utils import (
    prepare_model, time_to_step, step_to_time, get_total_steps,
)
from astra.evaluation.predictive_performance import (
    _get_predictions, _to_device, compute_net_benefit, format_step_label,
)
from astra.evaluation.calibration import calculate_ece

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

# Journal submission output constraints — see predictive_performance._SUBMISSION_KW
_SUBMISSION_KW = dict(
    fit_long_side_px=1200,
    max_long_side_px=1200,
    max_bytes=5_000_000,
)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TimepointPredictions:
    """Predictions at a single censoring timepoint."""
    censor_step: int
    time_hours: float
    y_true: np.ndarray
    y_prob: np.ndarray
    n_samples: int
    n_positive: int


@dataclass
class CalibratorResult:
    """Evaluation result for one calibrator at one timepoint."""
    censor_step: int
    time_hours: float
    time_label: str
    method: str
    calibrator_type: str  # 'per_timepoint' or 'global'
    ece_raw: float
    ece_cal: float
    brier_raw: float
    brier_cal: float
    auroc_raw: float
    auroc_cal: float
    auprc_raw: float
    auprc_cal: float
    n_samples: int
    n_positive: int


# ============================================================================
# TRAINVAL EVALUATORS — mirror holdout evaluators but use trainval data
# ============================================================================

class _TrainvalEvaluator:
    """
    Mirrors TimeDependentEvaluator but operates on trainval data.

    Used to get predictions on trainval for fitting calibrators.
    """

    def __init__(self, data: dict, model: torch.nn.Module, cfg: dict,
                 device: str = 'cuda'):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.eval()

        self.bs = cfg["training"]["bs"]

        # Trainval pre-normalized data
        self.X_normalized = data["X"]
        self.X_multi_hot = data["X_multi_hot"]
        self.y = data["y"]
        self.trajectory_lengths = data.get("trajectory_lengths")

        # Tabular arrays from trainval dataset
        trainval_ds = data["mixed_dls"]._train_ds
        if hasattr(trainval_ds, 'dataset'):
            trainval_ds = trainval_ds.dataset
        self.x_cat = trainval_ds.x_cat.numpy()
        self.x_cont = trainval_ds.x_cont.numpy()

        logger.info(f"_TrainvalEvaluator initialized: {len(self.y)} samples")

    def _censor_data(self, X: np.ndarray, censor_step: int) -> np.ndarray:
        X_censored = X.copy()
        if censor_step < X.shape[2] - 1:
            X_censored[:, :, censor_step + 1:] = 0.0
        return X_censored

    def get_predictions_at_step(self, censor_step: int) -> Optional[TimepointPredictions]:
        """Get trainval predictions at a censoring step (active patients only)."""
        y = np.array(self.y)

        # Active-only: exclude patients whose trajectory ended before this step
        if self.trajectory_lengths is not None:
            active_mask = self.trajectory_lengths > censor_step
            if active_mask.sum() < 2:
                return None
            y = y[active_mask]
            X_norm = self.X_normalized[active_mask]
            X_mh = self.X_multi_hot[active_mask]
            x_cat = self.x_cat[active_mask]
            x_cont = self.x_cont[active_mask]
            traj = self.trajectory_lengths[active_mask]
        else:
            X_norm = self.X_normalized
            X_mh = self.X_multi_hot
            x_cat = self.x_cat
            x_cont = self.x_cont
            traj = None

        if len(set(y)) < 2:
            return None

        X_censored = self._censor_data(X_norm, censor_step)
        X_mh_censored = self._censor_data(X_mh, censor_step)

        effective_traj = None
        if traj is not None:
            effective_traj = np.minimum(traj, censor_step + 1)

        dataset = AstraMixedDataset(
            X_ts=X_censored,
            x_cat=x_cat,
            x_cont=x_cont,
            X_ts_cat=X_mh_censored,
            y=y,
            trajectory_lengths=effective_traj,
        )
        dls = AstraMixedDataLoader(
            dataset, splits=None, bs=self.bs, shuffle_train=False,
        )

        preds, targets = _get_predictions(self.model, dls.train, self.device)
        y_prob = preds[:, 1].numpy()
        y_true = targets.numpy()

        time_min = step_to_time(censor_step)
        time_hours = time_min / 60 if time_min else 0

        return TimepointPredictions(
            censor_step=censor_step,
            time_hours=time_hours,
            y_true=y_true,
            y_prob=y_prob,
            n_samples=len(y_true),
            n_positive=int(y_true.sum()),
        )


class _TrainvalTemporalEvaluator:
    """
    Mirrors TemporalEvaluator but operates on trainval data.

    Single forward pass gives predictions at all timesteps.
    """

    def __init__(self, data: dict, model: torch.nn.Module, cfg: dict,
                 device: str = 'cuda'):
        self.data = data
        self.model = model
        self.device = device
        self.model.eval()

        self._preds = None
        self._traj_lengths = np.array(data.get("trajectory_lengths", []))
        self._y = np.array(data["y"])

        logger.info(f"_TrainvalTemporalEvaluator initialized: {len(self._y)} samples")

    def _get_all_predictions(self) -> np.ndarray:
        if self._preds is not None:
            return self._preds

        dls = self.data["mixed_dls"]
        all_preds = []
        with torch.no_grad():
            for batch in dls.train:
                inputs, targets = batch
                inputs = _to_device(inputs, self.device)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        self._preds = np.concatenate(all_preds, axis=0)
        return self._preds

    def get_predictions_at_step(self, censor_step: int) -> Optional[TimepointPredictions]:
        preds_all = self._get_all_predictions()
        y_true = self._y
        traj_lengths = self._traj_lengths

        # Active-only: exclude patients whose trajectory ended before this step
        if len(traj_lengths) > 0:
            active_mask = traj_lengths > censor_step
            if active_mask.sum() < 2:
                return None
            y_true = y_true[active_mask]
            preds_sub = preds_all[active_mask]
            traj_sub = traj_lengths[active_mask]
            effective_steps = np.minimum(censor_step, traj_sub - 1)
            effective_steps = np.maximum(effective_steps, 0).astype(int)
            y_prob = preds_sub[np.arange(len(preds_sub)), effective_steps]
        else:
            step = min(censor_step, preds_all.shape[1] - 1)
            y_prob = preds_all[:, step]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return None

        time_min = step_to_time(censor_step)
        time_hours = time_min / 60 if time_min else 0

        return TimepointPredictions(
            censor_step=censor_step,
            time_hours=time_hours,
            y_true=y_true,
            y_prob=y_prob,
            n_samples=len(y_true),
            n_positive=int(y_true.sum()),
        )


class _HoldoutEvaluator:
    """
    Lightweight holdout prediction collector (non-temporal models).

    Mirrors _TrainvalEvaluator but uses holdout data keys.
    """

    def __init__(self, data: dict, model: torch.nn.Module, cfg: dict,
                 device: str = 'cuda'):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.eval()
        self.bs = cfg["training"]["bs"]

        self.X_normalized = data["tX"]
        self.X_multi_hot = data["tX_multi_hot"]
        self.y = data["ty"]
        self.trajectory_lengths = data.get("holdout_trajectory_lengths")

        holdout_ds = data["holdout_mixed_dls"]._train_ds
        if hasattr(holdout_ds, 'dataset'):
            holdout_ds = holdout_ds.dataset
        self.x_cat = holdout_ds.x_cat.numpy()
        self.x_cont = holdout_ds.x_cont.numpy()

        logger.info(f"_HoldoutEvaluator initialized: {len(self.y)} samples")

    def _censor_data(self, X: np.ndarray, censor_step: int) -> np.ndarray:
        X_censored = X.copy()
        if censor_step < X.shape[2] - 1:
            X_censored[:, :, censor_step + 1:] = 0.0
        return X_censored

    def get_predictions_at_step(self, censor_step: int) -> Optional[TimepointPredictions]:
        y = np.array(self.y)

        # Active-only: exclude patients whose trajectory ended before this step
        if self.trajectory_lengths is not None:
            active_mask = self.trajectory_lengths > censor_step
            if active_mask.sum() < 2:
                return None
            y = y[active_mask]
            X_norm = self.X_normalized[active_mask]
            X_mh = self.X_multi_hot[active_mask]
            x_cat = self.x_cat[active_mask]
            x_cont = self.x_cont[active_mask]
            traj = self.trajectory_lengths[active_mask]
        else:
            X_norm = self.X_normalized
            X_mh = self.X_multi_hot
            x_cat = self.x_cat
            x_cont = self.x_cont
            traj = None

        if len(set(y)) < 2:
            return None

        X_censored = self._censor_data(X_norm, censor_step)
        X_mh_censored = self._censor_data(X_mh, censor_step)

        effective_traj = None
        if traj is not None:
            effective_traj = np.minimum(traj, censor_step + 1)

        dataset = AstraMixedDataset(
            X_ts=X_censored,
            x_cat=x_cat,
            x_cont=x_cont,
            X_ts_cat=X_mh_censored,
            y=y,
            trajectory_lengths=effective_traj,
        )
        dls = AstraMixedDataLoader(
            dataset, splits=None, bs=self.bs, shuffle_train=False,
        )

        preds, targets = _get_predictions(self.model, dls.train, self.device)
        y_prob = preds[:, 1].numpy()
        y_true = targets.numpy()

        time_min = step_to_time(censor_step)
        time_hours = time_min / 60 if time_min else 0

        return TimepointPredictions(
            censor_step=censor_step,
            time_hours=time_hours,
            y_true=y_true,
            y_prob=y_prob,
            n_samples=len(y_true),
            n_positive=int(y_true.sum()),
        )


class _HoldoutTemporalEvaluator:
    """Lightweight holdout prediction collector for temporal models."""

    def __init__(self, data: dict, model: torch.nn.Module, cfg: dict,
                 device: str = 'cuda'):
        self.data = data
        self.model = model
        self.device = device
        self.model.eval()

        self._preds = None
        self._traj_lengths = np.array(
            data.get("holdout_trajectory_lengths",
                      data.get("traj_lengths_holdout", []))
        )
        self._y = np.array(data["ty"])

        logger.info(f"_HoldoutTemporalEvaluator initialized: {len(self._y)} samples")

    def _get_all_predictions(self) -> np.ndarray:
        if self._preds is not None:
            return self._preds

        dls = self.data["holdout_mixed_dls"]
        all_preds = []
        with torch.no_grad():
            for batch in dls.train:
                inputs, targets = batch
                inputs = _to_device(inputs, self.device)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        self._preds = np.concatenate(all_preds, axis=0)
        return self._preds

    def get_predictions_at_step(self, censor_step: int) -> Optional[TimepointPredictions]:
        preds_all = self._get_all_predictions()
        y_true = self._y
        traj_lengths = self._traj_lengths

        # Active-only: exclude patients whose trajectory ended before this step
        if len(traj_lengths) > 0:
            active_mask = traj_lengths > censor_step
            if active_mask.sum() < 2:
                return None
            y_true = y_true[active_mask]
            preds_sub = preds_all[active_mask]
            traj_sub = traj_lengths[active_mask]
            effective_steps = np.minimum(censor_step, traj_sub - 1)
            effective_steps = np.maximum(effective_steps, 0).astype(int)
            y_prob = preds_sub[np.arange(len(preds_sub)), effective_steps]
        else:
            step = min(censor_step, preds_all.shape[1] - 1)
            y_prob = preds_all[:, step]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return None

        time_min = step_to_time(censor_step)
        time_hours = time_min / 60 if time_min else 0

        return TimepointPredictions(
            censor_step=censor_step,
            time_hours=time_hours,
            y_true=y_true,
            y_prob=y_prob,
            n_samples=len(y_true),
            n_positive=int(y_true.sum()),
        )


# ============================================================================
# CALIBRATOR FITTING
# ============================================================================

def _fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression calibrator."""
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    ir.fit(y_prob, y_true)
    return ir


def _fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling (logistic regression on logits)."""
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(logits, y_true)
    return lr


def fit_calibrators(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    methods: List[str] = ['isotonic', 'platt'],
) -> Dict[str, object]:
    """Fit multiple calibration methods. Returns {method_name: fitted_calibrator}."""
    calibrators = {}
    for method in methods:
        if method == 'isotonic':
            calibrators[method] = _fit_isotonic(y_true, y_prob)
        elif method == 'platt':
            calibrators[method] = _fit_platt(y_true, y_prob)
        else:
            logger.warning(f"Unknown calibration method: {method}")
    return calibrators


def apply_calibrator(
    calibrator: object,
    y_prob: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply a fitted calibrator to transform predictions."""
    if method == 'isotonic':
        return calibrator.transform(y_prob)
    elif method == 'platt':
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
        return calibrator.predict_proba(logits)[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")


# ============================================================================
# PREDICTION COLLECTION
# ============================================================================

def _collect_predictions(
    evaluator,
    censor_steps: List[int],
    min_positive: int = 20,
    label: str = "data",
) -> Dict[int, TimepointPredictions]:
    """Collect predictions at multiple timepoints from an evaluator."""
    predictions = {}
    for step in censor_steps:
        tp = evaluator.get_predictions_at_step(step)
        if tp is None:
            logger.warning(f"No predictions at step {step} ({label})")
            continue
        if tp.n_positive < min_positive:
            logger.warning(
                f"Skipping step {step} ({label}): only {tp.n_positive} "
                f"positives (min={min_positive})"
            )
            continue
        predictions[step] = tp
    return predictions


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def _evaluate_calibrated(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    censor_step: int,
    time_hours: float,
    method: str,
    calibrator_type: str,
    n_bins: int = 4,
) -> CalibratorResult:
    """Compare raw vs calibrated predictions at one timepoint."""
    ece_raw, _ = calculate_ece(y_true, y_prob_raw, n_bins=n_bins)
    ece_cal, _ = calculate_ece(y_true, y_prob_cal, n_bins=n_bins)

    brier_raw = brier_score_loss(y_true, y_prob_raw)
    brier_cal = brier_score_loss(y_true, y_prob_cal)

    auroc_raw = roc_auc_score(y_true, y_prob_raw)
    auroc_cal = roc_auc_score(y_true, y_prob_cal)

    auprc_raw = average_precision_score(y_true, y_prob_raw)
    auprc_cal = average_precision_score(y_true, y_prob_cal)

    # Sanity check: AUROC should not change significantly
    auroc_diff = abs(auroc_cal - auroc_raw)
    if auroc_diff > 0.001:
        logger.warning(
            f"AUROC changed by {auroc_diff:.4f} after {method} calibration "
            f"at step {censor_step} ({calibrator_type}). "
            f"Raw={auroc_raw:.4f}, Cal={auroc_cal:.4f}"
        )

    return CalibratorResult(
        censor_step=censor_step,
        time_hours=time_hours,
        time_label=format_step_label(censor_step),
        method=method,
        calibrator_type=calibrator_type,
        ece_raw=ece_raw,
        ece_cal=ece_cal,
        brier_raw=brier_raw,
        brier_cal=brier_cal,
        auroc_raw=auroc_raw,
        auroc_cal=auroc_cal,
        auprc_raw=auprc_raw,
        auprc_cal=auprc_cal,
        n_samples=len(y_true),
        n_positive=int(y_true.sum()),
    )


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_calibration_metrics_over_time(
    results: List[CalibratorResult],
    methods: List[str],
    model_name: str,
    save_dir: str,
):
    """Plot ECE and Brier score over time: raw vs calibrated per method."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    colors = {'isotonic': '#2E86AB', 'platt': '#A23B72'}

    # Filter to per-timepoint results
    per_tp = [r for r in results if r.calibrator_type == 'per_timepoint']

    for method in methods:
        method_results = sorted(
            [r for r in per_tp if r.method == method],
            key=lambda r: r.censor_step,
        )
        if not method_results:
            continue

        steps = [r.time_hours for r in method_results]
        color = colors.get(method, '#333333')

        # ECE
        axes[0].plot(steps, [r.ece_raw for r in method_results],
                     color='grey', linestyle='--', linewidth=1.5,
                     alpha=0.7, label='Raw' if method == methods[0] else None)
        axes[0].plot(steps, [r.ece_cal for r in method_results],
                     color=color, linewidth=2, marker='o', markersize=4,
                     label=f'{method.capitalize()} calibrated')

        # Brier
        axes[1].plot(steps, [r.brier_raw for r in method_results],
                     color='grey', linestyle='--', linewidth=1.5,
                     alpha=0.7, label='Raw' if method == methods[0] else None)
        axes[1].plot(steps, [r.brier_cal for r in method_results],
                     color=color, linewidth=2, marker='o', markersize=4,
                     label=f'{method.capitalize()} calibrated')

    axes[0].set_ylabel('Expected Calibration Error (ECE)', fontsize=_FIG_STYLE['axis_label'])
    axes[0].set_title('Calibration Metrics Over Time: Raw vs Calibrated', fontsize=_FIG_STYLE['title'], fontweight='bold')
    axes[0].legend(fontsize=_FIG_STYLE['legend'])
    axes[0].tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (hours)', fontsize=_FIG_STYLE['axis_label'])
    axes[1].set_ylabel('Brier Score', fontsize=_FIG_STYLE['axis_label'])
    axes[1].legend(fontsize=_FIG_STYLE['legend'])
    axes[1].tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"calibration_analysis_{model_name}", save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)
    logger.info(f"Saved calibration_analysis_{model_name}.png")


def _plot_reliability_diagrams(
    holdout_preds: Dict[int, TimepointPredictions],
    calibrated_preds: Dict[int, Dict[str, np.ndarray]],
    best_method: str,
    model_name: str,
    save_dir: str,
    n_bins: int = 4,
):
    """Grid of reliability diagrams at key timepoints (before/after)."""
    from sklearn.calibration import calibration_curve

    steps = sorted(holdout_preds.keys())
    n = len(steps)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    # Smaller per-panel size (3.5 in) so a 3x3 grid lands within the 1200 px cap.
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, step in enumerate(steps):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        tp = holdout_preds[step]
        y_true, y_prob_raw = tp.y_true, tp.y_prob

        # Raw calibration curve
        try:
            frac_raw, mean_raw = calibration_curve(
                y_true, y_prob_raw, n_bins=n_bins, strategy='uniform'
            )
            brier_raw = brier_score_loss(y_true, y_prob_raw)
            ax.plot(mean_raw, frac_raw, 'o--', color='grey', linewidth=2.0,
                    markersize=5, alpha=0.7, label=f'Raw (Brier={brier_raw:.3f})')
        except Exception:
            logger.debug(f"Raw calibration curve failed at step {step}; skipping panel curve")

        # Calibrated curve
        if step in calibrated_preds and best_method in calibrated_preds[step]:
            y_prob_cal = calibrated_preds[step][best_method]
            try:
                frac_cal, mean_cal = calibration_curve(
                    y_true, y_prob_cal, n_bins=n_bins, strategy='uniform'
                )
                brier_cal = brier_score_loss(y_true, y_prob_cal)
                ax.plot(mean_cal, frac_cal, 'o-', color='#2E86AB', linewidth=2.5,
                        markersize=7, label=f'{best_method.capitalize()} (Brier={brier_cal:.3f})')
            except Exception:
                logger.debug(f"Calibrated ({best_method}) curve failed at step {step}; skipping panel curve")

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_title(format_step_label(step), fontsize=_FIG_STYLE['title'], fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
        ax.legend(fontsize=8, handlelength=1.5, labelspacing=0.3)
        if row == nrows - 1:
            ax.set_xlabel('Predicted probability', fontsize=_FIG_STYLE['axis_label'])
        if col == 0:
            ax.set_ylabel('Observed frequency', fontsize=_FIG_STYLE['axis_label'])

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'Reliability Diagrams: Raw vs {best_method.capitalize()} Calibrated',
                 fontsize=_FIG_STYLE['suptitle'], fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, f"reliability_diagrams_{model_name}", save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)
    logger.info(f"Saved reliability_diagrams_{model_name}.png")


def _plot_dca_comparison(
    holdout_preds: Dict[int, TimepointPredictions],
    calibrated_preds: Dict[int, Dict[str, np.ndarray]],
    best_method: str,
    model_name: str,
    save_dir: str,
    max_threshold: float = 0.5,
    n_points: int = 200,
):
    """DCA at key timepoints: raw vs calibrated net benefit."""
    steps = sorted(holdout_preds.keys())
    n = len(steps)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    # Smaller per-panel size so a 3x3 grid lands within the 1200 px cap.
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    thresholds = np.linspace(0.01, max_threshold, n_points)

    for idx, step in enumerate(steps):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        tp = holdout_preds[step]
        y_true = tp.y_true.astype(float)
        y_prob_raw = tp.y_prob.astype(float)

        nb_raw, nb_all, nb_none = compute_net_benefit(y_true, y_prob_raw, thresholds)

        ymin = min(nb_raw.min(), -0.01) - 0.005
        ymax = max(nb_raw.max(), y_true.mean()) * 1.15 + 0.005

        ax.plot(thresholds, nb_raw, color='grey', linewidth=1.5, alpha=0.7, label='Raw')
        ax.plot(thresholds, np.clip(nb_all, ymin, None),
                color='black', linewidth=1, linestyle=':', alpha=0.5, label='Treat All')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        if step in calibrated_preds and best_method in calibrated_preds[step]:
            y_prob_cal = calibrated_preds[step][best_method].astype(float)
            nb_cal, _, _ = compute_net_benefit(y_true, y_prob_cal, thresholds)
            ax.plot(thresholds, nb_cal, color='#2E86AB', linewidth=2,
                    label=f'{best_method.capitalize()}')
            ymin = min(ymin, nb_cal.min() - 0.005)
            ymax = max(ymax, nb_cal.max() * 1.15 + 0.005)

        ax.set_ylim(ymin, ymax)
        ax.set_title(format_step_label(step), fontsize=_FIG_STYLE['title'])
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
        if idx == 0:
            ax.legend(fontsize=_FIG_STYLE['legend'])
        if row == nrows - 1:
            ax.set_xlabel('Threshold', fontsize=_FIG_STYLE['axis_label'])
        if col == 0:
            ax.set_ylabel('Net Benefit', fontsize=_FIG_STYLE['axis_label'])

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'Decision Curve Analysis: Raw vs {best_method.capitalize()} Calibrated',
                 fontsize=_FIG_STYLE['suptitle'], fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, f"dca_comparison_{model_name}", save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)
    logger.info(f"Saved dca_comparison_{model_name}.png")


def _plot_dca_calibrated(
    holdout_preds: Dict[int, TimepointPredictions],
    calibrated_preds: Dict[int, Dict[str, np.ndarray]],
    best_method: str,
    model_name: str,
    save_dir: str,
    max_threshold: float = 0.5,
    n_points: int = 200,
):
    """
    Standalone DCA multi-curve using calibrated predictions at each timepoint.

    Matches the style of plot_decision_curves_over_time() from
    predictive_performance.py but uses the calibrated probability output.
    """
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
              '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
    thresholds = np.linspace(0.01, max_threshold, n_points)

    fig, ax = plt.subplots(figsize=(9, 6))
    global_ymin = 0.0
    global_ymax = 0.0
    treat_all_curves = []

    steps = sorted(holdout_preds.keys())
    for i, step in enumerate(steps):
        tp = holdout_preds[step]
        y_true = tp.y_true.astype(float)

        # Use calibrated predictions if available, else raw
        if step in calibrated_preds and best_method in calibrated_preds[step]:
            y_prob = calibrated_preds[step][best_method].astype(float)
        else:
            y_prob = tp.y_prob.astype(float)

        prevalence = y_true.mean()
        nb_model, _, _ = compute_net_benefit(y_true, y_prob, thresholds)

        label = format_step_label(step)
        color = colors[i % len(colors)]
        ax.plot(thresholds, nb_model, color=color, linewidth=1.8, label=label)

        nb_treat_all = prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)
        treat_all_curves.append((nb_treat_all, color))

        global_ymin = min(global_ymin, nb_model.min())
        global_ymax = max(global_ymax, nb_model.max())

    ymin = min(global_ymin, -0.01) - 0.005
    ymax = global_ymax * 1.15 + 0.005

    first = True
    for nb_treat_all, color in treat_all_curves:
        nb_clipped = np.clip(nb_treat_all, ymin, None)
        ax.plot(thresholds, nb_clipped, color=color, linewidth=1.0,
                linestyle='--', alpha=0.4,
                label='Treat All' if first else None)
        first = False

    ax.axhline(y=0, color='black', linewidth=1, label='Treat None')

    ax.set_xlabel("Threshold Probability", fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylabel("Net Benefit", fontsize=_FIG_STYLE['axis_label'])
    ax.set_title(
        f"Decision Curves (Calibrated — {best_method.capitalize()})",
        fontsize=_FIG_STYLE['title'], fontweight='bold',
    )
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=_FIG_STYLE['legend'],
              title="Time Available", title_fontsize=_FIG_STYLE['legend'])
    ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_threshold)
    ax.set_ylim(ymin, ymax)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    save_figure(fig, f"dca_calibrated_{model_name}", save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)
    logger.info(f"Saved dca_calibrated_{model_name}.png")


def _plot_per_timepoint_vs_global(
    results: List[CalibratorResult],
    best_method: str,
    model_name: str,
    save_dir: str,
):
    """Bar chart comparing per-timepoint vs global calibrator ECE."""
    per_tp = sorted(
        [r for r in results if r.method == best_method and r.calibrator_type == 'per_timepoint'],
        key=lambda r: r.censor_step,
    )
    glob = sorted(
        [r for r in results if r.method == best_method and r.calibrator_type == 'global'],
        key=lambda r: r.censor_step,
    )

    if not per_tp or not glob:
        return

    labels = [r.time_label for r in per_tp]
    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, [r.ece_raw for r in per_tp], width,
           label='Raw', color='grey', alpha=0.6)
    ax.bar(x, [r.ece_cal for r in per_tp], width,
           label='Per-timepoint', color='#2E86AB')
    ax.bar(x + width, [r.ece_cal for r in glob], width,
           label='Global', color='#A23B72')

    ax.set_xlabel('Timepoint', fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylabel('ECE', fontsize=_FIG_STYLE['axis_label'])
    ax.set_title(f'ECE: Raw vs Per-Timepoint vs Global ({best_method.capitalize()})',
                 fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=_FIG_STYLE['tick_label'])
    ax.tick_params(axis='y', labelsize=_FIG_STYLE['tick_label'])
    ax.legend(fontsize=_FIG_STYLE['legend'])
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    save_figure(fig, f"per_timepoint_vs_global_{model_name}", save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)
    logger.info(f"Saved per_timepoint_vs_global_{model_name}.png")


# ============================================================================
# TEMPORAL CALIBRATOR — per-window calibration for temporal prediction models
# ============================================================================

class TemporalCalibrator:
    """Per-window posthoc calibration for temporal prediction models.

    A single global calibrator systematically miscalibrates early vs late
    predictions because base rate, active cohort composition, and prediction
    distribution all shift over time.  This class fits one calibrator per
    time-window (aligned to bin-interval phase boundaries by default) on
    trainval data, then transforms holdout predictions step-by-step using the
    appropriate window's calibrator.

    Typical usage::

        tc = TemporalCalibrator(method='isotonic')
        tc.fit(tv_preds_all, tv_y, tv_traj)          # [N_tv, seq_len]
        ho_calibrated = tc.transform(ho_preds_all, ho_traj)  # [N_ho, seq_len]
        # or single-step:
        cal_probs = tc.transform_at_step(raw_probs, step=42)
    """

    def __init__(self, method: str = 'isotonic', window_hours: Optional[float] = None,
                 min_samples: int = 50):
        """
        Args:
            method: 'isotonic' or 'platt'.
            window_hours: Fixed window width in hours.  *None* (default) uses
                bin-interval phase boundaries from config (recommended).
            min_samples: Minimum prediction–label pairs per window to fit a
                calibrator.  Windows below this threshold are merged with the
                next window.
        """
        self.method = method
        self.window_hours = window_hours
        self.min_samples = min_samples
        self._calibrators: Dict[int, object] = {}
        self._step_to_window: Dict[int, int] = {}
        self._window_boundaries: List[Tuple[int, int]] = []
        self._fitted = False

    # ── public API ──────────────────────────────────────────────────────

    def fit(self, preds_all: np.ndarray, y_true: np.ndarray,
            traj_lengths: Optional[np.ndarray] = None) -> 'TemporalCalibrator':
        """Fit per-window calibrators on trainval predictions.

        Args:
            preds_all: ``[N, seq_len]`` sigmoid probabilities.
            y_true: ``[N]`` binary labels.
            traj_lengths: ``[N]`` trajectory lengths (active-only filtering).
        """
        _N, seq_len = preds_all.shape

        windows = self._compute_windows(seq_len)
        self._window_boundaries = windows

        for w_idx, (start, end) in enumerate(windows):
            for step in range(start, end):
                self._step_to_window[step] = w_idx

        for w_idx, (start, end) in enumerate(windows):
            y_probs_w: List[np.ndarray] = []
            y_true_w: List[np.ndarray] = []

            for step in range(start, end):
                if traj_lengths is not None:
                    active = traj_lengths > step
                    if active.sum() < 2:
                        continue
                    y_probs_w.append(preds_all[active, step])
                    y_true_w.append(y_true[active])
                else:
                    y_probs_w.append(preds_all[:, step])
                    y_true_w.append(y_true)

            if not y_probs_w:
                continue

            y_prob_pooled = np.concatenate(y_probs_w)
            y_true_pooled = np.concatenate(y_true_w)

            if len(np.unique(y_true_pooled)) < 2:
                continue
            if len(y_prob_pooled) < self.min_samples:
                continue

            cal = fit_calibrators(y_true_pooled, y_prob_pooled, [self.method])
            self._calibrators[w_idx] = cal[self.method]

        self._fitted = True
        window_desc = ", ".join(
            f"[{s}-{e})" for s, e in self._window_boundaries
        )
        logger.info(
            f"TemporalCalibrator fitted: {len(self._calibrators)}/{len(windows)} "
            f"windows, method={self.method}, windows={window_desc}"
        )
        return self

    def transform(self, preds_all: np.ndarray,
                  traj_lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform a full ``[N, seq_len]`` prediction matrix.

        Returns a copy with each step calibrated by its window's calibrator.
        Steps without a fitted calibrator are left unchanged.
        """
        if not self._fitted:
            raise RuntimeError("TemporalCalibrator.fit() must be called first")

        calibrated = preds_all.copy()
        _N, seq_len = preds_all.shape

        for step in range(seq_len):
            w_idx = self._step_to_window.get(step)
            if w_idx is None or w_idx not in self._calibrators:
                continue

            cal = self._calibrators[w_idx]

            if traj_lengths is not None:
                active = traj_lengths > step
                if active.sum() == 0:
                    continue
                calibrated[active, step] = apply_calibrator(
                    cal, preds_all[active, step], self.method
                )
            else:
                calibrated[:, step] = apply_calibrator(
                    cal, preds_all[:, step], self.method
                )

        return calibrated

    def transform_at_step(self, y_prob: np.ndarray, step: int) -> np.ndarray:
        """Calibrate predictions at a single timestep.

        Args:
            y_prob: ``[N]`` raw probabilities at *step*.
            step: timestep index.

        Returns:
            ``[N]`` calibrated probabilities (unchanged if no calibrator).
        """
        if not self._fitted:
            raise RuntimeError("TemporalCalibrator.fit() must be called first")

        w_idx = self._step_to_window.get(step)
        if w_idx is None or w_idx not in self._calibrators:
            return y_prob
        return apply_calibrator(self._calibrators[w_idx], y_prob, self.method)

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the entire calibrator to *path*."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"TemporalCalibrator saved to {path}")

    @staticmethod
    def load(path: str) -> 'TemporalCalibrator':
        """Load a previously saved ``TemporalCalibrator``."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, TemporalCalibrator):
            raise TypeError(f"Expected TemporalCalibrator, got {type(obj)}")
        logger.info(f"TemporalCalibrator loaded from {path} "
                     f"({len(obj._calibrators)} windows, method={obj.method})")
        return obj

    # ── diagnostics ─────────────────────────────────────────────────────

    def evaluate(self, preds_all: np.ndarray, y_true: np.ndarray,
                 traj_lengths: Optional[np.ndarray] = None,
                 n_bins: int = 4) -> pd.DataFrame:
        """Evaluate calibration quality per window: raw vs calibrated ECE/Brier.

        Returns a DataFrame with one row per window.
        """
        calibrated = self.transform(preds_all, traj_lengths)
        rows = []

        for w_idx, (start, end) in enumerate(self._window_boundaries):
            y_prob_raw_w: List[np.ndarray] = []
            y_prob_cal_w: List[np.ndarray] = []
            y_true_w: List[np.ndarray] = []

            for step in range(start, end):
                if traj_lengths is not None:
                    active = traj_lengths > step
                    if active.sum() < 2:
                        continue
                    y_prob_raw_w.append(preds_all[active, step])
                    y_prob_cal_w.append(calibrated[active, step])
                    y_true_w.append(y_true[active])
                else:
                    y_prob_raw_w.append(preds_all[:, step])
                    y_prob_cal_w.append(calibrated[:, step])
                    y_true_w.append(y_true)

            if not y_true_w:
                continue

            yt = np.concatenate(y_true_w)
            yr = np.concatenate(y_prob_raw_w)
            yc = np.concatenate(y_prob_cal_w)

            if len(np.unique(yt)) < 2:
                continue

            ece_raw, _ = calculate_ece(yt, yr, n_bins=n_bins)
            ece_cal, _ = calculate_ece(yt, yc, n_bins=n_bins)
            brier_raw = brier_score_loss(yt, yr)
            brier_cal = brier_score_loss(yt, yc)

            start_h = step_to_time(start)
            end_h = step_to_time(min(end - 1, preds_all.shape[1] - 1))
            start_h = start_h / 60 if start_h else 0
            end_h = end_h / 60 if end_h else 0

            rows.append({
                'window': w_idx,
                'steps': f'{start}-{end}',
                'time_range': f'{start_h:.0f}h-{end_h:.0f}h',
                'n_pairs': len(yt),
                'prevalence': yt.mean(),
                'ece_raw': ece_raw,
                'ece_cal': ece_cal,
                'ece_reduction': ece_raw - ece_cal,
                'brier_raw': brier_raw,
                'brier_cal': brier_cal,
                'brier_reduction': brier_raw - brier_cal,
            })

        return pd.DataFrame(rows)

    # ── internals ───────────────────────────────────────────────────────

    def _compute_windows(self, seq_len: int) -> List[Tuple[int, int]]:
        """Compute window boundaries aligned to bin-interval phases."""
        from astra.evaluation.utils import _get_intervals_from_cfg

        if self.window_hours is not None:
            windows: List[Tuple[int, int]] = []
            start = 0
            while start < seq_len:
                start_time = step_to_time(start) or 0
                end_time_min = start_time + self.window_hours * 60
                end_step = start + 1
                while end_step < seq_len:
                    t = step_to_time(end_step)
                    if t is not None and t >= end_time_min:
                        break
                    end_step += 1
                windows.append((start, min(end_step, seq_len)))
                start = end_step
            return windows

        # Default: use bin-interval phase boundaries from config
        intervals = _get_intervals_from_cfg()
        windows = []
        cum_steps = 0

        for start_min, end_min, bin_min in intervals:
            if end_min is not None:
                n_steps = (end_min - start_min) // bin_min
            else:
                n_steps = seq_len - cum_steps
            if n_steps <= 0:
                continue
            w_start = cum_steps
            w_end = min(cum_steps + n_steps, seq_len)
            if w_start < w_end:
                windows.append((w_start, w_end))
            cum_steps += n_steps

        if not windows:
            windows = [(0, seq_len)]

        return windows


def fit_temporal_calibrator(
    data: dict,
    model: torch.nn.Module,
    cfg: dict,
    device: str = 'cuda',
    method: str = 'isotonic',
    save_path: Optional[str] = None,
) -> TemporalCalibrator:
    """Convenience: fit a ``TemporalCalibrator`` on trainval predictions.

    Creates a ``_TrainvalTemporalEvaluator``, runs a single forward pass, and
    fits the calibrator.  Optionally saves to *save_path*.
    """
    tv_eval = _TrainvalTemporalEvaluator(data, model, cfg, device)
    preds_all = tv_eval._get_all_predictions()
    y_true = tv_eval._y
    traj_lengths = tv_eval._traj_lengths if len(tv_eval._traj_lengths) > 0 else None

    tc = TemporalCalibrator(method=method)
    tc.fit(preds_all, y_true, traj_lengths)

    if save_path:
        tc.save(save_path)

    return tc


# ============================================================================
# CALIBRATOR PERSISTENCE
# ============================================================================

def _save_calibrators(
    per_timepoint_calibrators: Dict[int, Dict[str, object]],
    global_calibrators: Dict[str, object],
    best_method: str,
    methods: List[str],
    summary_df: pd.DataFrame,
    calibrator_dir: str,
):
    """Save fitted calibrators and metadata."""
    os.makedirs(calibrator_dir, exist_ok=True)

    # Per-timepoint
    for step, cals in per_timepoint_calibrators.items():
        for method, cal in cals.items():
            path = os.path.join(calibrator_dir, f"{method}_step{step}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(cal, f)

    # Global
    for method, cal in global_calibrators.items():
        path = os.path.join(calibrator_dir, f"{method}_global.pkl")
        with open(path, 'wb') as f:
            pickle.dump(cal, f)

    # Metadata
    meta = {
        'best_method': best_method,
        'methods': methods,
        'timepoints': list(per_timepoint_calibrators.keys()),
    }
    with open(os.path.join(calibrator_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved calibrators to {calibrator_dir}")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_posthoc_calibration(
    data: dict,
    cfg: dict,
    methods: List[str] = ['isotonic', 'platt'],
    key_timepoints: Optional[List[int]] = None,
    min_positive_samples: int = 20,
    n_bins: int = 4,
    save_dir: str = 'reports/calibration',
    calibrator_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full posthoc calibration pipeline.

    1. Get trainval predictions at key timepoints (for fitting)
    2. Fit per-timepoint and global calibrators
    3. Get holdout predictions at key timepoints
    4. Evaluate calibrated holdout predictions
    5. Generate comparison plots and save calibrators

    Args:
        data: Output from prepare_data_and_dls()
        cfg: Configuration dictionary
        methods: Calibration methods to compare
        key_timepoints: Censor steps to evaluate (default: standard 8)
        min_positive_samples: Skip timepoints with fewer positives
        n_bins: Number of bins for ECE computation
        save_dir: Directory for plots and CSV
        calibrator_dir: Directory for saved calibrators (default: models/calibrators/{model_name})

    Returns:
        summary_df: DataFrame with all calibration results
    """
    model_name = cfg["model_name"]
    is_temporal = cfg.get("model", {}).get("temporal_head", False)

    if calibrator_dir is None:
        calibrator_dir = f"models/calibrators/{model_name}"

    os.makedirs(save_dir, exist_ok=True)

    # Default key timepoints
    if key_timepoints is None:
        max_step = get_total_steps() - 2  # last step needs full-length traj; cap to N-2
        raw = [
            time_to_step(1, 'h'), time_to_step(6, 'h'),
            time_to_step(12, 'h'), time_to_step(72, 'h'),
            time_to_step(7, 'D'), time_to_step(14, 'D'),
            time_to_step(30, 'D'), time_to_step(90, 'D'),
        ]
        key_timepoints = sorted({
            min(t, max_step) for t in raw if t is not None
        })

    logger.info(f"Posthoc calibration: {len(key_timepoints)} timepoints, methods={methods}")

    # ========================================================================
    # Load model
    # ========================================================================
    model, device = prepare_model(data, cfg)

    # ========================================================================
    # Create evaluators
    # ========================================================================
    if is_temporal:
        trainval_eval = _TrainvalTemporalEvaluator(data, model, cfg, device)
        holdout_eval = _HoldoutTemporalEvaluator(data, model, cfg, device)
    else:
        trainval_eval = _TrainvalEvaluator(data, model, cfg, device)
        holdout_eval = _HoldoutEvaluator(data, model, cfg, device)

    # ========================================================================
    # Collect predictions
    # ========================================================================
    logger.info("Collecting trainval predictions...")
    t0 = time.time()
    trainval_preds = _collect_predictions(
        trainval_eval, key_timepoints, min_positive_samples, "trainval"
    )
    logger.info(f"Trainval predictions collected at {len(trainval_preds)} timepoints "
                f"in {time.time() - t0:.1f}s")

    logger.info("Collecting holdout predictions...")
    t0 = time.time()
    holdout_preds = _collect_predictions(
        holdout_eval, key_timepoints, min_positive_samples, "holdout"
    )
    logger.info(f"Holdout predictions collected at {len(holdout_preds)} timepoints "
                f"in {time.time() - t0:.1f}s")

    # Use steps present in both
    valid_steps = sorted(set(trainval_preds.keys()) & set(holdout_preds.keys()))
    if not valid_steps:
        logger.error("No valid timepoints for calibration. Aborting.")
        return pd.DataFrame()

    logger.info(f"Calibrating at {len(valid_steps)} timepoints: "
                f"{[format_step_label(s) for s in valid_steps]}")

    # ========================================================================
    # Fit per-timepoint calibrators
    # ========================================================================
    logger.info("Fitting per-timepoint calibrators...")
    per_tp_calibrators: Dict[int, Dict[str, object]] = {}
    for step in valid_steps:
        tp = trainval_preds[step]
        per_tp_calibrators[step] = fit_calibrators(tp.y_true, tp.y_prob, methods)

    # ========================================================================
    # Fit global calibrators (pool trainval across all timepoints)
    # ========================================================================
    logger.info("Fitting global calibrators...")
    global_y_true = np.concatenate([trainval_preds[s].y_true for s in valid_steps])
    global_y_prob = np.concatenate([trainval_preds[s].y_prob for s in valid_steps])
    global_calibrators = fit_calibrators(global_y_true, global_y_prob, methods)

    # ========================================================================
    # Evaluate on holdout
    # ========================================================================
    logger.info("Evaluating calibrated predictions on holdout...")
    all_results: List[CalibratorResult] = []
    # Store calibrated predictions for plotting
    calibrated_holdout: Dict[int, Dict[str, np.ndarray]] = {}

    for step in valid_steps:
        ho = holdout_preds[step]
        calibrated_holdout[step] = {}

        for method in methods:
            # Per-timepoint calibrator
            cal = per_tp_calibrators[step][method]
            y_cal = apply_calibrator(cal, ho.y_prob, method)
            calibrated_holdout[step][method] = y_cal

            result = _evaluate_calibrated(
                ho.y_true, ho.y_prob, y_cal,
                step, ho.time_hours, method, 'per_timepoint', n_bins,
            )
            all_results.append(result)

            # Global calibrator
            g_cal = global_calibrators[method]
            y_cal_g = apply_calibrator(g_cal, ho.y_prob, method)

            result_g = _evaluate_calibrated(
                ho.y_true, ho.y_prob, y_cal_g,
                step, ho.time_hours, method, 'global', n_bins,
            )
            all_results.append(result_g)

    # ========================================================================
    # Build summary DataFrame
    # ========================================================================
    summary_df = pd.DataFrame([
        {
            'censor_step': r.censor_step,
            'time_hours': r.time_hours,
            'time_label': r.time_label,
            'method': r.method,
            'calibrator_type': r.calibrator_type,
            'ece_raw': r.ece_raw,
            'ece_cal': r.ece_cal,
            'ece_reduction': r.ece_raw - r.ece_cal,
            'brier_raw': r.brier_raw,
            'brier_cal': r.brier_cal,
            'brier_reduction': r.brier_raw - r.brier_cal,
            'auroc_raw': r.auroc_raw,
            'auroc_cal': r.auroc_cal,
            'auprc_raw': r.auprc_raw,
            'auprc_cal': r.auprc_cal,
            'n_samples': r.n_samples,
            'n_positive': r.n_positive,
        }
        for r in all_results
    ])

    # ========================================================================
    # Determine best method (highest mean ECE reduction, per-timepoint)
    # ========================================================================
    per_tp_summary = summary_df[summary_df['calibrator_type'] == 'per_timepoint']
    mean_ece_reduction = per_tp_summary.groupby('method')['ece_reduction'].mean()
    best_method = mean_ece_reduction.idxmax() if len(mean_ece_reduction) > 0 else methods[0]

    logger.info(f"Best calibration method: {best_method} "
                f"(mean ECE reduction: {mean_ece_reduction.get(best_method, 0):.4f})")

    # Log summary table
    logger.info("\n=== Calibration Summary (per-timepoint) ===")
    for method in methods:
        method_rows = per_tp_summary[per_tp_summary['method'] == method]
        if len(method_rows) > 0:
            logger.info(f"\n  {method.upper()}:")
            for _, row in method_rows.iterrows():
                logger.info(
                    f"    {row['time_label']:>10s}: ECE {row['ece_raw']:.4f} -> "
                    f"{row['ece_cal']:.4f} ({row['ece_reduction']:+.4f}), "
                    f"Brier {row['brier_raw']:.4f} -> {row['brier_cal']:.4f}"
                )

    # ========================================================================
    # Save CSV
    # ========================================================================
    csv_path = os.path.join(save_dir, f"calibration_summary_{model_name}.csv")
    ensure_parent_dir(csv_path)
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved calibration summary to {csv_path}")

    # ========================================================================
    # Generate plots
    # ========================================================================
    logger.info("Generating calibration plots...")

    _plot_calibration_metrics_over_time(all_results, methods, model_name, save_dir)

    # Collect predictions at all timepoints with a lower threshold
    # so that later timepoints (7D, 14D, 30D) are not dropped from plots
    all_holdout_preds = _collect_predictions(
        holdout_eval, key_timepoints, min_positive=5, label="holdout-plots"
    )
    # Calibrate the extra steps using global calibrator
    all_calibrated = dict(calibrated_holdout)  # copy existing per-tp calibrated
    for step in all_holdout_preds:
        if step not in all_calibrated:
            all_calibrated[step] = {}
            ho = all_holdout_preds[step]
            for method in methods:
                g_cal = global_calibrators[method]
                all_calibrated[step][method] = apply_calibrator(
                    g_cal, ho.y_prob, method
                )

    _plot_reliability_diagrams(
        all_holdout_preds, all_calibrated, best_method, model_name, save_dir, n_bins
    )
    _plot_dca_calibrated(
        all_holdout_preds, all_calibrated, best_method, model_name, save_dir
    )
    _plot_dca_comparison(
        all_holdout_preds, all_calibrated, best_method, model_name, save_dir
    )
    _plot_per_timepoint_vs_global(all_results, best_method, model_name, save_dir)

    # ========================================================================
    # Save calibrators
    # ========================================================================
    _save_calibrators(
        per_tp_calibrators, global_calibrators,
        best_method, methods, summary_df, calibrator_dir,
    )

    logger.info("Posthoc calibration complete.")
    return summary_df
