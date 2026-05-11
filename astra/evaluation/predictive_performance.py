# predictive_performance.py
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from astra.utils import save_figure, ensure_parent_dir
from astra.data.dataloader import normalize_with_padding_mask
from astra.data.mixed_dataloader import (
    AstraMixedDataset,
    AstraMixedDataLoader,
)

from astra.evaluation.utils import (
    calculate_roc_auc_ci, calculate_average_precision_ci,
    bootstrap_recall_ci, find_optimal_fbeta_threshold,
    _parse_timedelta_to_minutes, _get_intervals_from_cfg,
    time_to_step, step_to_time, prepare_model, get_max_days, get_total_steps
)
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from astra.models.hybrid.training import get_backbone
from astra.visualize.evaluation import plot_evaluation, evaluate_detection_rate

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

# Journal submission output constraints.
# fit_long_side_px=1200 computes the highest DPI that keeps the tight-bbox output
# under 1200 px on the long side (≥300 DPI for the figsizes used here). Warnings
# fire if the final file exceeds 1200 px or 5 MB.
_SUBMISSION_KW = dict(
    fit_long_side_px=1200,
    max_long_side_px=1200,
    max_bytes=5_000_000,
)


@dataclass
class TimeMetricResult:
    """Container for time-dependent evaluation results"""
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
    # Survival-specific metrics (optional, None for classification mode)
    cindex: Optional[float] = None
    cindex_ci: Optional[Tuple[float, float]] = None
    brier_score: Optional[float] = None


@dataclass
class PercentileRecallResult:
    """Container for percentile-based recall at a single time point."""
    time_min: float
    time_hours: float
    time_days: float
    censor_step: int
    recalls: Dict[int, float]                    # {percentile: recall}
    recall_cis: Dict[int, Tuple[float, float]]   # {percentile: (lower, upper)}
    n_samples: int
    n_positive: int


_TARGET_DISPLAY = {"deceased_30d": "30-day Mortality"}


def _save_time_metrics_csv(results: List['TimeMetricResult'], path: str) -> None:
    """Persist a list of TimeMetricResult to CSV for downstream reporting."""
    rows = []
    for r in results:
        rows.append({
            "censor_step": r.censor_step,
            "time_min": r.time_min,
            "time_hours": r.time_hours,
            "time_days": r.time_days,
            "auroc": r.auroc,
            "auroc_ci_lower": r.auroc_ci[0],
            "auroc_ci_upper": r.auroc_ci[1],
            "auprc": r.auprc,
            "auprc_ci_lower": r.auprc_ci[0],
            "auprc_ci_upper": r.auprc_ci[1],
            "n_samples": r.n_samples,
            "n_positive": r.n_positive,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Time metrics saved to {path}")

def _display_target(name: str) -> str:
    return _TARGET_DISPLAY.get(name, name)


def _get_predictions(model, dataloader, device, temporal_head=False):
    """
    Run direct model inference on a dataloader.

    Args:
        model: nn.Module (already on device)
        dataloader: iterable yielding ((x_ts, (x_cat, x_cont), x_ts_cat), y)
        device: str
        temporal_head: if True, returns sigmoid probabilities [n, seq_len]

    Returns:
        preds: tensor of predictions
        targets: tensor of targets
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = _to_device(inputs, device)
            targets = targets.to(device)

            logits = model(inputs)

            if temporal_head:
                probs = torch.sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=-1)

            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def _to_device(obj, device):
    """Recursively move tensors to device."""
    if isinstance(obj, torch.Tensor):
        t = obj.to(device)
        if type(t) is not torch.Tensor:
            t = t.as_subclass(torch.Tensor)
        return t
    elif isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(item, device) for item in obj)
    return obj


# ============================================================================
# DECISION CURVE ANALYSIS (NET BENEFIT)
# ============================================================================

def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute net benefit for decision curve analysis.

    Args:
        y_true: Binary labels (0/1), shape [N]
        y_prob: Predicted probabilities, shape [N]
        thresholds: Threshold probabilities in (0, 1)

    Returns:
        nb_model: Net benefit of the model at each threshold
        nb_treat_all: Net benefit of "treat all" strategy
        nb_treat_none: Net benefit of "treat none" (always 0)
    """
    N = len(y_true)
    prevalence = y_true.mean()
    nb_model = np.empty_like(thresholds, dtype=float)
    nb_treat_all = np.empty_like(thresholds, dtype=float)

    for i, t in enumerate(thresholds):
        weight = t / (1.0 - t)
        predicted_positive = y_prob >= t
        tp = np.sum(predicted_positive & (y_true == 1))
        fp = np.sum(predicted_positive & (y_true == 0))
        nb_model[i] = tp / N - fp / N * weight
        nb_treat_all[i] = prevalence - (1.0 - prevalence) * weight

    nb_treat_none = np.zeros_like(thresholds)
    return nb_model, nb_treat_all, nb_treat_none


def plot_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    max_threshold: float = 0.5,
    n_points: int = 200,
) -> plt.Figure:
    """Plot decision curve analysis for a single set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(0.01, max_threshold, n_points)

    nb_model, nb_treat_all, nb_treat_none = compute_net_benefit(
        y_true, y_prob, thresholds
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    # Determine y-range from model curve, then clip Treat All to that range
    ymin = min(nb_model.min(), -0.01) - 0.005
    ymax = max(nb_model.max(), y_true.mean()) * 1.15 + 0.005
    nb_treat_all_clipped = np.clip(nb_treat_all, ymin, None)

    ax.plot(thresholds, nb_model, color='#1F77B4', linewidth=2, label=model_name)
    ax.plot(thresholds, nb_treat_all_clipped, color='grey', linewidth=1.5, linestyle='--',
            label='Treat All')
    ax.axhline(y=0, color='black', linewidth=1, label='Treat None')

    ax.set_xlabel("Threshold Probability", fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylabel("Net Benefit", fontsize=_FIG_STYLE['axis_label'])
    ax.set_title("Decision Curve Analysis", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax.legend(fontsize=_FIG_STYLE['legend'])
    ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_threshold)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    return fig

def plot_decision_curves_over_time(
    evaluator,
    censor_steps: List[int],
    labels: Optional[List[str]] = None,
    max_threshold: float = 0.5,
    n_points: int = 200,
) -> plt.Figure:
    """
    Plot decision curves at multiple timepoints.

    Follows the same pattern as plot_multiple_roc_pr_curves(): iterates
    censor_steps, extracts predictions, computes net benefit per timepoint.
    Each timepoint gets its own "Treat All" baseline computed from
    the prevalence in the active cohort at that step.
    """
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
              '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
    thresholds = np.linspace(0.01, max_threshold, n_points)

    fig, ax = plt.subplots(figsize=(9, 6))
    global_ymin = 0.0
    global_ymax = 0.0

    # Store per-timepoint treat-all data
    treat_all_curves = []

    for i, censor_step in enumerate(censor_steps):
        dls = evaluator.create_censored_dataloaders_fast(censor_step)
        if dls is None:
            logger.warning(f"DCA: skipping step {censor_step}: dataloader creation failed")
            treat_all_curves.append(None)
            continue

        preds, targets = _get_predictions(evaluator.model, dls.train, evaluator.device)
        y_preds = preds[:, 1].numpy()
        ys = targets.numpy()

        if len(set(ys)) < 2:
            logger.warning(f"DCA: skipping step {censor_step}: only one class")
            treat_all_curves.append(None)
            continue

        # Per-timepoint prevalence
        prevalence = ys.mean()

        nb_model, _, _ = compute_net_benefit(ys, y_preds, thresholds)

        label = labels[i] if labels and i < len(labels) else format_step_label(censor_step)
        color = colors[i % len(colors)]
        ax.plot(thresholds, nb_model, color=color, linewidth=1.8, label=label)

        # Compute and store per-timepoint treat-all
        nb_treat_all = prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)
        treat_all_curves.append((nb_treat_all, color, label, prevalence))

        global_ymin = min(global_ymin, nb_model.min())
        global_ymax = max(global_ymax, nb_model.max())

    # Determine y-axis range from model curves only
    ymin = min(global_ymin, -0.01) - 0.005
    ymax = global_ymax * 1.15 + 0.005

    # Plot per-timepoint "Treat All" lines, clipped to visible range
    first_treat_all = True
    for curve_data in treat_all_curves:
        if curve_data is None:
            continue
        nb_treat_all, color, label, prevalence = curve_data
        nb_treat_all_clipped = np.clip(nb_treat_all, ymin, None)
        legend_label = "Treat All" if first_treat_all else None
        ax.plot(
            thresholds, nb_treat_all_clipped,
            color=color, linewidth=1.0, linestyle='--', alpha=0.4,
            label=legend_label,
        )
        first_treat_all = False

    # "Treat None" baseline
    ax.axhline(y=0, color='black', linewidth=1, label='Treat None')

    ax.set_xlabel("Threshold Probability", fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylabel("Net Benefit", fontsize=_FIG_STYLE['axis_label'])
    ax.set_title("Decision Curves at Different Time Points", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=_FIG_STYLE['legend'],
              title="Time Available", title_fontsize=_FIG_STYLE['legend'])
    ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_threshold)
    ax.set_ylim(ymin, ymax)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    return fig

def _plot_decision_curves_temporal(
    preds_all: np.ndarray,
    y_true: np.ndarray,
    traj_lengths: np.ndarray,
    censor_steps: List[int],
    labels: Optional[List[str]] = None,
    max_threshold: float = 0.5,
    n_points: int = 200,
) -> plt.Figure:
    """
    Plot decision curves at multiple timepoints for temporal models.

    Uses pre-computed predictions matrix instead of running inference per step.
    Each timepoint gets its own "Treat All" baseline computed from
    the prevalence in the active cohort at that step.
    """
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
              '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
    thresholds = np.linspace(0.01, max_threshold, n_points)

    fig, ax = plt.subplots(figsize=(9, 6))
    global_ymin = 0.0
    global_ymax = 0.0

    max_step = preds_all.shape[1] - 1

    # Store per-timepoint prevalences for treat-all lines
    treat_all_curves = []

    for i, censor_step in enumerate(censor_steps):
        # Use effective step (min of censor_step, trajectory length - 1)
        effective_steps = np.minimum(censor_step, traj_lengths - 1).astype(int)
        effective_steps = np.clip(effective_steps, 0, max_step)

        # Only include patients whose trajectory reaches this timepoint
        active_mask = traj_lengths > censor_step
        if active_mask.sum() < 10:
            logger.warning(f"DCA temporal: skipping step {censor_step}: too few active patients")
            treat_all_curves.append(None)
            continue

        y_sub = y_true[active_mask]
        preds_sub = preds_all[np.where(active_mask)[0], effective_steps[active_mask]]

        if len(set(y_sub)) < 2:
            logger.warning(f"DCA temporal: skipping step {censor_step}: only one class")
            treat_all_curves.append(None)
            continue

        # Per-timepoint prevalence
        prevalence = y_sub.mean()

        nb_model, _, _ = compute_net_benefit(y_sub, preds_sub, thresholds)

        label = labels[i] if labels and i < len(labels) else format_step_label(censor_step)
        color = colors[i % len(colors)]
        ax.plot(thresholds, nb_model, color=color, linewidth=1.8, label=label)

        # Compute and store per-timepoint treat-all
        nb_treat_all = prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)
        treat_all_curves.append((nb_treat_all, color, label, prevalence))

        global_ymin = min(global_ymin, nb_model.min())
        global_ymax = max(global_ymax, nb_model.max())

    # Determine y-axis range from model curves only
    ymin = min(global_ymin, -0.01) - 0.005
    ymax = global_ymax * 1.15 + 0.005

    # Plot per-timepoint "Treat All" lines, clipped to visible range
    first_treat_all = True
    for curve_data in treat_all_curves:
        if curve_data is None:
            continue
        nb_treat_all, color, label, prevalence = curve_data
        nb_treat_all_clipped = np.clip(nb_treat_all, ymin, None)
        legend_label = "Treat All" if first_treat_all else None
        ax.plot(
            thresholds, nb_treat_all_clipped,
            color=color, linewidth=1.0, linestyle='--', alpha=0.4,
            label=legend_label,
        )
        first_treat_all = False

    # "Treat None" baseline
    ax.axhline(y=0, color='black', linewidth=1, label='Treat None')

    ax.set_xlabel("Threshold Probability", fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylabel("Net Benefit", fontsize=_FIG_STYLE['axis_label'])
    ax.set_title("Decision Curves at Different Time Points", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=_FIG_STYLE['legend'],
              title="Time Available", title_fontsize=_FIG_STYLE['legend'])
    ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_threshold)
    ax.set_ylim(ymin, ymax)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    return fig


class TimeDependentEvaluator:
    """
    Evaluates model performance at different time censoring points.

    Works directly with normalized data from prepare_data_and_dls().
    Only censors (masks) data, doesn't re-normalize.
    """

    def __init__(self, data: dict, model: torch.nn.Module, cfg: dict,
                 device: str = 'cuda', active_only: bool = False):
        """
        Initialize evaluator with data and model.

        Args:
            data: Output from prepare_data_and_dls()
            model: Trained model (nn.Module, already on device)
            cfg: Configuration dictionary
            device: Device string
            active_only: If True, only include patients with active trajectories
                         at each evaluation timestep
        """
        self.data = data
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.eval()

        # Cache static components
        self.cat_cols = data["cat_cols"]
        self.num_cols = data["num_cols"]
        self.classes = data["classes"]
        self.cat_encoder = data["cat_encoder"]
        self.target = cfg["target"]
        self.bs = cfg["training"]["bs"]

        # Cache pre-normalized data
        self.holdout_X_normalized = data["tX"]
        self.holdout_X_multi_hot = data["tX_multi_hot"]
        self.holdout_y = data["ty"]

        # Pre-encoded tabular arrays from holdout dataset
        holdout_ds = data["holdout_mixed_dls"]._train_ds
        if hasattr(holdout_ds, 'dataset'):
            holdout_ds = holdout_ds.dataset
        self.holdout_x_cat = holdout_ds.x_cat.numpy()
        self.holdout_x_cont = holdout_ds.x_cont.numpy()
        self.holdout_trajectory_lengths = data.get("holdout_trajectory_lengths")
        self.active_only = active_only

        self.holdout = data["holdout"]

        mode_str = " (active-only mode)" if active_only else ""
        logger.info(f"TimeDependentEvaluator initialized with pre-normalized data{mode_str}")

    def _get_active_mask(self, censor_step: int) -> np.ndarray:
        """Boolean mask: True for patients with trajectory_length > censor_step."""
        if self.holdout_trajectory_lengths is None:
            return np.ones(len(self.holdout_y), dtype=bool)
        return self.holdout_trajectory_lengths > censor_step

    def _censor_normalized_data(self, X_normalized: np.ndarray, censor_step: int) -> np.ndarray:
        X_censored = X_normalized.copy()
        if censor_step < X_normalized.shape[2] - 1:
            X_censored[:, :, censor_step+1:] = 0.0
        return X_censored

    def _censor_multihot(self, X_multi_hot: np.ndarray, censor_step: int) -> np.ndarray:
        X_censored = X_multi_hot.copy()
        if censor_step < X_multi_hot.shape[2] - 1:
            X_censored[:, :, censor_step+1:] = 0
        return X_censored

    def create_censored_dataloaders_fast(self, censor_step: int) -> Optional[AstraMixedDataLoader]:
        """
        Create dataloaders by censoring pre-normalized data.
        When active_only=True, filters to patients with active trajectories.
        """
        # Select patient subset
        if self.active_only:
            mask = self._get_active_mask(censor_step)
            if mask.sum() < 2:
                logger.warning(f"Too few active patients ({mask.sum()}) at step {censor_step}")
                return None
            X_norm = self.holdout_X_normalized[mask]
            X_mh = self.holdout_X_multi_hot[mask]
            x_cat = self.holdout_x_cat[mask]
            x_cont = self.holdout_x_cont[mask]
            y = np.array(self.holdout_y)[mask]
            traj = self.holdout_trajectory_lengths[mask] if self.holdout_trajectory_lengths is not None else None
        else:
            X_norm = self.holdout_X_normalized
            X_mh = self.holdout_X_multi_hot
            x_cat = self.holdout_x_cat
            x_cont = self.holdout_x_cont
            y = self.holdout_y
            traj = self.holdout_trajectory_lengths

        if len(set(y)) < 2:
            logger.debug(f"Only one class in dataset at step {censor_step}")
            return None

        X_censored = self._censor_normalized_data(X_norm, censor_step)
        X_multi_hot_censored = self._censor_multihot(X_mh, censor_step)

        # Effective trajectory: min(original, censor_step + 1) per sample
        effective_traj = None
        if traj is not None:
            effective_traj = np.minimum(traj, censor_step + 1)

        dataset = AstraMixedDataset(
            X_ts=X_censored,
            x_cat=x_cat,
            x_cont=x_cont,
            X_ts_cat=X_multi_hot_censored,
            y=y,
            trajectory_lengths=effective_traj,
        )
        return AstraMixedDataLoader(
            dataset,
            splits=None,
            bs=self.bs,
            shuffle_train=False,
        )

    def _get_active_counts(self, censor_step: int) -> Optional[TimeMetricResult]:
        """Return a counts-only result (NaN metrics) for single-class time points."""
        time_min = step_to_time(censor_step)
        if time_min is None:
            return None

        if self.active_only:
            mask = self._get_active_mask(censor_step)
            n_samples = int(mask.sum())
            if n_samples == 0:
                return None
            y = np.array(self.holdout_y)[mask]
        else:
            y = np.array(self.holdout_y)
            n_samples = len(y)

        return TimeMetricResult(
            time_min=time_min,
            time_hours=time_min / 60,
            time_days=time_min / (24 * 60),
            censor_step=censor_step,
            auroc=float('nan'),
            auroc_ci=(float('nan'), float('nan')),
            auprc=float('nan'),
            auprc_ci=(float('nan'), float('nan')),
            n_samples=n_samples,
            n_positive=int(y.sum()),
        )

    def evaluate_at_timestep(self, censor_step: int) -> Optional[TimeMetricResult]:
        dls = self.create_censored_dataloaders_fast(censor_step)
        if dls is None:
            # Dataloader failed (single class or <2 patients) — return counts only
            return self._get_active_counts(censor_step)

        preds, targets = _get_predictions(self.model, dls.train, self.device)
        y_preds = preds[:, 1].numpy()
        ys = targets.numpy()

        time_min = step_to_time(censor_step)
        if time_min is None:
            logger.warning(f"Could not convert step {censor_step} to time")
            return None

        if ys.sum() == 0 or ys.sum() == len(ys):
            logger.debug(f"Single class at censor_step={censor_step}, metrics undefined")
            return TimeMetricResult(
                time_min=time_min,
                time_hours=time_min / 60,
                time_days=time_min / (24 * 60),
                censor_step=censor_step,
                auroc=float('nan'),
                auroc_ci=(float('nan'), float('nan')),
                auprc=float('nan'),
                auprc_ci=(float('nan'), float('nan')),
                n_samples=len(ys),
                n_positive=int(ys.sum()),
            )

        auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(ys, y_preds)
        auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(ys, y_preds)

        return TimeMetricResult(
            time_min=time_min,
            time_hours=time_min / 60,
            time_days=time_min / (24 * 60),
            censor_step=censor_step,
            auroc=auroc,
            auroc_ci=(auroc_lower, auroc_upper),
            auprc=auprc,
            auprc_ci=(auprc_lower, auprc_upper),
            n_samples=len(ys),
            n_positive=int(ys.sum())
        )

    def evaluate_over_time_ultra_fast(
        self,
        censor_steps: List[int],
        save_predictions: bool = True,
        model_name: Optional[str] = None
    ) -> Tuple[List[TimeMetricResult], Optional[pd.DataFrame]]:
        import time

        results = []
        preds_over_time = [] if save_predictions else None
        patient_ids = self.holdout.base.PID.values

        logger.info(f"Ultra-fast evaluation at {len(censor_steps)} time points...")
        start_time = time.time()

        for i, censor_step in enumerate(censor_steps):
            if i % 10 == 0 or i == len(censor_steps) - 1:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_per_step = elapsed / i
                    remaining = avg_per_step * (len(censor_steps) - i)
                    logger.info(
                        f"Progress: {i+1}/{len(censor_steps)} ({100*i/len(censor_steps):.1f}%) "
                        f"- ~{remaining/60:.1f}min remaining"
                    )

            result = self.evaluate_at_timestep(censor_step)

            if result is not None:
                results.append(result)

                if save_predictions and not np.isnan(result.auroc):
                    dls = self.create_censored_dataloaders_fast(censor_step)
                    preds, _ = _get_predictions(self.model, dls.train, self.device)
                    y_preds = preds[:, 1].numpy()

                    if self.active_only:
                        mask = self._get_active_mask(censor_step)
                        active_pids = patient_ids[mask]
                    else:
                        active_pids = patient_ids

                    for pid, pred in zip(active_pids, y_preds):
                        preds_over_time.append({
                            "PID": pid,
                            "censor_step": censor_step,
                            "time_min": result.time_min,
                            "time_hours": result.time_hours,
                            "time_days": result.time_days,
                            "pred": float(pred)
                        })

        total_time = time.time() - start_time
        logger.info(
            f"Ultra-fast evaluation complete: {len(results)}/{len(censor_steps)} successful "
            f"in {total_time/60:.1f} minutes ({total_time/len(censor_steps):.2f}s per step)"
        )

        if save_predictions and preds_over_time and model_name:
            preds_df = pd.DataFrame(preds_over_time)
            os.makedirs(f'reports/eval/{model_name}/predictions', exist_ok=True)
            preds_df.to_pickle(f'reports/eval/{model_name}/predictions/preds_{model_name}.pkl')
            logger.info(f"Saved predictions to reports/eval/{model_name}/predictions/preds_{model_name}.pkl")
            return results, preds_df

        return results, None

    def evaluate_percentile_recall_over_time(
        self,
        censor_steps: List[int],
        percentiles: List[int] = (5, 10, 15, 20, 25),
        n_bootstraps: int = 1000,
    ) -> List[PercentileRecallResult]:
        """Compute recall at top-K% risk thresholds across time points."""
        import time as time_module

        results = []
        logger.info(f"Percentile recall evaluation at {len(censor_steps)} time points "
                     f"(percentiles={list(percentiles)})...")
        start_time = time_module.time()

        for i, censor_step in enumerate(censor_steps):
            if i % 10 == 0:
                logger.debug(f"Percentile recall progress: {i+1}/{len(censor_steps)}")

            dls = self.create_censored_dataloaders_fast(censor_step)
            if dls is None:
                continue

            preds, targets = _get_predictions(self.model, dls.train, self.device)
            y_preds = preds[:, 1].numpy()
            ys = targets.numpy()

            time_min = step_to_time(censor_step)
            if time_min is None or ys.sum() == 0:
                continue

            recalls = {}
            recall_cis = {}
            for perc in percentiles:
                r, ci_lo, ci_hi = bootstrap_recall_ci(
                    y_preds, ys, perc, n_bootstraps=n_bootstraps
                )
                recalls[perc] = r
                recall_cis[perc] = (ci_lo, ci_hi)

            results.append(PercentileRecallResult(
                time_min=time_min,
                time_hours=time_min / 60,
                time_days=time_min / (24 * 60),
                censor_step=censor_step,
                recalls=recalls,
                recall_cis=recall_cis,
                n_samples=len(ys),
                n_positive=int(ys.sum()),
            ))

        total_time = time_module.time() - start_time
        logger.info(f"Percentile recall complete: {len(results)} time points in {total_time:.1f}s")
        return results


# ============================================================================
# TEMPORAL EVALUATOR (PER-TIMESTEP MODELS)
# ============================================================================

class TemporalEvaluator:
    """
    Evaluator for per-timestep prediction models.

    Key advantage: ONE forward pass gives predictions at ALL timesteps.
    No censoring loop, no repeated dataloader creation.
    """

    def __init__(self, data: dict, model: torch.nn.Module, cfg: dict,
                 device: str = 'cuda', active_only: bool = False,
                 calibrator=None):
        self.data = data
        self.model = model
        self.cfg = cfg
        self.device = device
        self.active_only = active_only
        self.survival_mode = cfg.get("model", {}).get("survival_mode", False)
        self.calibrator = calibrator
        self.model.eval()

        self._holdout_preds = None
        self._holdout_preds_calibrated = None
        self._holdout_survival = None  # [N, seq_len] survival probs S(t)
        self._holdout_traj_lengths = np.array(
            data.get("holdout_trajectory_lengths",
                      data.get("traj_lengths_holdout", []))
        )
        self.holdout = data["holdout"]

        mode_str = " (active-only mode)" if active_only else ""
        surv_str = " [survival]" if self.survival_mode else ""
        cal_str = " [calibrated]" if calibrator is not None else ""
        logger.info(f"TemporalEvaluator initialized{mode_str}{surv_str}{cal_str}")

    def _get_all_predictions(self, calibrated: bool = True) -> np.ndarray:
        """Get per-timestep predictions.

        For classification: sigmoid probabilities [N, seq_len].
        For survival: cumulative incidence 1-S(t) [N, seq_len].

        Args:
            calibrated: If True and a calibrator is set, return calibrated
                probabilities.  Raw predictions are always cached separately.
        """
        if self._holdout_preds is None:
            holdout_dls = self.data["holdout_mixed_dls"]
            all_logits = []

            with torch.no_grad():
                for batch in holdout_dls.train:
                    inputs, targets = batch
                    inputs = _to_device(inputs, self.device)
                    logits = self.model(inputs)
                    all_logits.append(logits.cpu())

            all_logits_cat = torch.cat(all_logits, dim=0)  # [N, seq_len]

            if self.survival_mode:
                from astra.training.utils import hazards_to_survival
                survival_probs = hazards_to_survival(all_logits_cat).numpy()
                self._holdout_survival = survival_probs
                self._holdout_preds = 1.0 - survival_probs
            else:
                self._holdout_preds = torch.sigmoid(all_logits_cat).numpy()

        if calibrated and self.calibrator is not None:
            if self._holdout_preds_calibrated is None:
                traj = self._holdout_traj_lengths if len(self._holdout_traj_lengths) > 0 else None
                self._holdout_preds_calibrated = self.calibrator.transform(
                    self._holdout_preds, traj
                )
            return self._holdout_preds_calibrated

        return self._holdout_preds

    def evaluate_at_timestep(self, censor_step: int) -> Optional[TimeMetricResult]:
        preds_all = self._get_all_predictions()
        ys = np.array(self.data["ty"])
        traj_lengths = self._holdout_traj_lengths

        # Active-only filtering
        if self.active_only and len(traj_lengths) > 0:
            mask = traj_lengths > censor_step
            if mask.sum() < 2:
                logger.warning(f"Too few active patients ({mask.sum()}) at step {censor_step}")
                return None
            preds_subset = preds_all[mask]
            ys = ys[mask]
            traj_subset = traj_lengths[mask]
        else:
            preds_subset = preds_all
            traj_subset = traj_lengths

        if len(traj_subset) > 0:
            effective_steps = np.minimum(censor_step, traj_subset - 1)
            effective_steps = np.maximum(effective_steps, 0).astype(int)
        else:
            effective_steps = np.full(len(preds_subset), censor_step, dtype=int)
            effective_steps = np.minimum(effective_steps, preds_subset.shape[1] - 1)

        y_preds = preds_subset[np.arange(len(preds_subset)), effective_steps]

        time_min = step_to_time(censor_step)
        if time_min is None:
            return None

        if ys.sum() == 0 or ys.sum() == len(ys):
            logger.debug(f"Single class at censor_step={censor_step}, metrics undefined")
            return TimeMetricResult(
                time_min=time_min,
                time_hours=time_min / 60,
                time_days=time_min / (24 * 60),
                censor_step=censor_step,
                auroc=float('nan'),
                auroc_ci=(float('nan'), float('nan')),
                auprc=float('nan'),
                auprc_ci=(float('nan'), float('nan')),
                n_samples=len(ys),
                n_positive=int(ys.sum()),
            )

        auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(ys, y_preds)
        auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(ys, y_preds)

        # Survival-specific metrics (C-index, Brier)
        cindex_val = None
        cindex_ci_val = None
        brier_val = None
        if self.survival_mode:
            ho_event_times = self.data.get("holdout_event_times")
            ho_event_indicators = self.data.get("holdout_event_indicators")
            if ho_event_times is not None and ho_event_indicators is not None:
                from astra.evaluation.survival_metrics import concordance_index as _ci
                # Use the same subset if active_only filtering was applied
                if self.active_only and len(traj_lengths) > 0:
                    mask = traj_lengths > censor_step
                    et = ho_event_times[mask]
                    ei = ho_event_indicators[mask]
                else:
                    et = ho_event_times
                    ei = ho_event_indicators
                try:
                    cindex_val, cindex_ci_val = _ci(et, ei, y_preds, n_bootstrap=500)
                except Exception as e:
                    logger.debug(f"C-index failed at step {censor_step}: {e}")

        return TimeMetricResult(
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
            cindex=cindex_val,
            cindex_ci=cindex_ci_val,
            brier_score=brier_val,
        )

    def evaluate_over_time(
        self,
        censor_steps: List[int],
        save_predictions: bool = True,
        model_name: Optional[str] = None,
    ) -> Tuple[List[TimeMetricResult], Optional[pd.DataFrame]]:
        import time as time_module

        preds_all = self._get_all_predictions()
        patient_ids = self.holdout.base.PID.values

        results = []
        preds_over_time = [] if save_predictions else None

        logger.info(f"Temporal evaluation at {len(censor_steps)} time points "
                     f"(single forward pass, {preds_all.shape[0]} patients)...")
        start_time = time_module.time()

        for censor_step in censor_steps:
            result = self.evaluate_at_timestep(censor_step)
            if result is None:
                continue
            results.append(result)

            if save_predictions:
                traj_lengths = self._holdout_traj_lengths

                if self.active_only and len(traj_lengths) > 0:
                    mask = traj_lengths > censor_step
                    preds_subset = preds_all[mask]
                    traj_subset = traj_lengths[mask]
                    active_pids = patient_ids[mask]
                else:
                    preds_subset = preds_all
                    traj_subset = traj_lengths
                    active_pids = patient_ids

                if len(traj_subset) > 0:
                    effective_steps = np.minimum(
                        censor_step, traj_subset - 1
                    )
                    effective_steps = np.maximum(effective_steps, 0).astype(int)
                else:
                    effective_steps = np.minimum(
                        censor_step, preds_subset.shape[1] - 1
                    )
                y_preds = preds_subset[np.arange(len(preds_subset)), effective_steps]

                for pid, pred in zip(active_pids, y_preds):
                    preds_over_time.append({
                        "PID": pid,
                        "censor_step": censor_step,
                        "time_min": result.time_min,
                        "time_hours": result.time_hours,
                        "time_days": result.time_days,
                        "pred": float(pred),
                    })

        total_time = time_module.time() - start_time
        logger.info(
            f"Temporal evaluation complete: {len(results)}/{len(censor_steps)} "
            f"in {total_time:.1f}s"
        )

        if save_predictions and preds_over_time and model_name:
            preds_df = pd.DataFrame(preds_over_time)
            os.makedirs(f'reports/eval/{model_name}/predictions', exist_ok=True)
            preds_df.to_pickle(f'reports/eval/{model_name}/predictions/preds_{model_name}.pkl')
            logger.info(f"Saved predictions to reports/eval/{model_name}/predictions/preds_{model_name}.pkl")
            return results, preds_df

        return results, pd.DataFrame(preds_over_time) if preds_over_time else (results, None)

    def evaluate_percentile_recall_over_time(
        self,
        censor_steps: List[int],
        percentiles: List[int] = (5, 10, 15, 20, 25),
        n_bootstraps: int = 1000,
    ) -> List[PercentileRecallResult]:
        """Compute recall at top-K% risk thresholds across time points.

        Reuses cached single-forward-pass predictions — no extra inference.
        """
        preds_all = self._get_all_predictions()
        ys = np.array(self.data["ty"])
        traj_lengths = self._holdout_traj_lengths

        results = []
        logger.info(f"Temporal percentile recall at {len(censor_steps)} time points "
                     f"(percentiles={list(percentiles)})...")

        for censor_step in censor_steps:
            # Active-only filtering (same logic as evaluate_at_timestep)
            if self.active_only and len(traj_lengths) > 0:
                mask = traj_lengths > censor_step
                if mask.sum() < 2:
                    continue
                preds_subset = preds_all[mask]
                ys_subset = ys[mask]
                traj_subset = traj_lengths[mask]
            else:
                preds_subset = preds_all
                ys_subset = ys
                traj_subset = traj_lengths

            # Pick effective timestep per patient
            if len(traj_subset) > 0:
                effective_steps = np.minimum(censor_step, traj_subset - 1)
                effective_steps = np.maximum(effective_steps, 0).astype(int)
            else:
                effective_steps = np.full(len(preds_subset), censor_step, dtype=int)
                effective_steps = np.minimum(effective_steps, preds_subset.shape[1] - 1)

            y_preds = preds_subset[np.arange(len(preds_subset)), effective_steps]

            time_min = step_to_time(censor_step)
            if time_min is None or ys_subset.sum() == 0:
                continue

            recalls = {}
            recall_cis = {}
            for perc in percentiles:
                r, ci_lo, ci_hi = bootstrap_recall_ci(
                    y_preds, ys_subset, perc, n_bootstraps=n_bootstraps
                )
                recalls[perc] = r
                recall_cis[perc] = (ci_lo, ci_hi)

            results.append(PercentileRecallResult(
                time_min=time_min,
                time_hours=time_min / 60,
                time_days=time_min / (24 * 60),
                censor_step=censor_step,
                recalls=recalls,
                recall_cis=recall_cis,
                n_samples=len(ys_subset),
                n_positive=int(ys_subset.sum()),
            ))

        logger.info(f"Temporal percentile recall complete: {len(results)} time points")
        return results


# ============================================================================
# TIME THRESHOLD GENERATION
# ============================================================================

def generate_time_thresholds(max_days=None, cut_hours=72, step_hours=1, step_days=1):
    """Generate list of time steps to evaluate at.

    Args:
        max_days: Maximum days to evaluate. Defaults to config-derived horizon
                  via ``get_max_days()``.
    """
    if max_days is None:
        max_days = get_max_days()

    thresholds = []

    for h in range(step_hours, cut_hours+1, step_hours):
        step = time_to_step(h, 'h')
        if step is not None:
            thresholds.append(step)

    start_day = int(np.ceil(cut_hours/24))
    for d in range(start_day+1, max_days+1, step_days):
        step = time_to_step(d, 'D')
        if step is not None:
            thresholds.append(step)

    return sorted(list(set(thresholds)))


def format_step_label(step):
    """Convert step to human-readable time label."""
    time_min = step_to_time(step)

    if time_min is None:
        return f"Step {step}"

    if time_min < 60:
        return f"{int(time_min)} min"
    elif time_min < 24 * 60:
        hours = time_min / 60
        if hours.is_integer():
            hours = int(hours)
        return f"{hours} h"
    else:
        days = time_min / (24 * 60)
        if days.is_integer():
            days = int(days)
        return f"{days} day" + ("s" if days != 1 else "")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_time_metrics(results: List[TimeMetricResult], cut_hours=72, max_days=None):
    if max_days is None:
        max_days = get_max_days()
    if not results:
        raise ValueError("No results to plot")

    times_h = np.array([r.time_hours for r in results])
    times_d = np.array([r.time_days for r in results])
    auroc_vals = np.array([r.auroc for r in results])
    auroc_lower = np.array([r.auroc_ci[0] for r in results])
    auroc_upper = np.array([r.auroc_ci[1] for r in results])
    auprc_vals = np.array([r.auprc for r in results])
    auprc_lower = np.array([r.auprc_ci[0] for r in results])
    auprc_upper = np.array([r.auprc_ci[1] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    mask_cut = times_h <= cut_hours

    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals[mask_cut], auroc_lower[mask_cut], auroc_upper[mask_cut], 'o', "C0", "AUROC"),
        ("AUPRC", auprc_vals[mask_cut], auprc_lower[mask_cut], auprc_upper[mask_cut], 's', "C1", "AUPRC")
    ]:
        valid = ~np.isnan(vals)
        x, vals, lower, upper = times_h[mask_cut][valid], vals[valid], lower[valid], upper[valid]
        if len(x) > 0:
            if x[-1] < cut_hours:
                x_ext = np.append(x, cut_hours)
                vals_ext = np.append(vals, vals[-1])
                lower_ext = np.append(lower, lower[-1])
                upper_ext = np.append(upper, upper[-1])
            else:
                x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper

            ax1.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=4)
            ax1.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax1.set_xlabel("Time (hours)", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours+1, 6))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylabel("Score", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_title("A) Performance over Hours", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=_FIG_STYLE['legend'])
    ax1.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax1.set_ylim(0.0, 1.0)

    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals, auroc_lower, auroc_upper, 'o', "C0", "AUROC"),
        ("AUPRC", auprc_vals, auprc_lower, auprc_upper, 's', "C1", "AUPRC")
    ]:
        valid = ~np.isnan(vals)
        x, vals, lower, upper = times_d[valid], vals[valid], lower[valid], upper[valid]
        if len(x) > 0:
            if x[-1] < max_days:
                x_ext = np.append(x, max_days)
                vals_ext = np.append(vals, vals[-1])
                lower_ext = np.append(lower, lower[-1])
                upper_ext = np.append(upper, upper[-1])
            else:
                x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper

            ax2.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=4)
            ax2.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax2.set_xlabel("Time (days)", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days+1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylabel("Score", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_title("B) Performance over Days", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=_FIG_STYLE['legend'])
    ax2.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()
    return fig


def plot_prediction_distribution(
    preds_df: pd.DataFrame,
    y_true: np.ndarray,
    holdout_pids: np.ndarray,
    cut_hours: int = 72,
    max_days: float = None,
    hour_timepoints: List[float] = None,
    day_timepoints: List[float] = None,
) -> plt.Figure:
    """
    Plot distribution of predicted probabilities per outcome category across timepoints.

    Following Van Calster et al. (Lancet Digital Health 2025) recommendation for
    risk distribution plots using split violin plots at selected timepoints.

    Args:
        preds_df: DataFrame with columns PID, censor_step, time_hours, time_days, pred
        y_true: True binary labels for holdout patients
        holdout_pids: Patient IDs corresponding to y_true
        cut_hours: Hour cutoff for the hours panel
        max_days: Max days for the days panel (from config if None)
        hour_timepoints: Timepoints (hours) to show in panel A
        day_timepoints: Timepoints (days) to show in panel B

    Returns:
        matplotlib Figure
    """
    if max_days is None:
        max_days = get_max_days()

    if hour_timepoints is None:
        hour_timepoints = [1, 6, 12, 24, 48, 72]
    if day_timepoints is None:
        day_timepoints = [3, 7, 14, 30, 60, 90]
    day_timepoints = [d for d in day_timepoints if d <= max_days]

    # Map PID → true label
    pid_to_label = dict(zip(holdout_pids, y_true))
    df = preds_df.copy()
    df['true_label'] = df['PID'].map(pid_to_label)
    df = df.dropna(subset=['true_label'])

    COLOR_NEG = '#2CA02C'  # survived (green)
    COLOR_POS = '#D62728'  # deceased (red)
    MIN_SAMPLES = 5

    def _snap_timepoints(available, requested):
        """Snap requested timepoints to nearest available, deduplicated."""
        snapped = []
        seen = set()
        for t in requested:
            closest = min(available, key=lambda x: abs(x - t))
            if closest not in seen:
                snapped.append((t, closest))
                seen.add(closest)
        return snapped

    def _draw_split_violins(ax, df, time_col, timepoint_pairs, time_unit):
        positions = list(range(len(timepoint_pairs)))

        for i, (requested, actual) in enumerate(timepoint_pairs):
            subset = df[df[time_col] == actual]
            preds_neg = subset.loc[subset['true_label'] == 0, 'pred'].values
            preds_pos = subset.loc[subset['true_label'] == 1, 'pred'].values

            # Left half: survived (negative)
            if len(preds_neg) >= MIN_SAMPLES:
                parts = ax.violinplot(
                    preds_neg, positions=[i], showmedians=False,
                    showextrema=False, widths=0.8
                )
                for body in parts['bodies']:
                    verts = body.get_paths()[0].vertices
                    center = i
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, center)
                    body.set_facecolor(COLOR_NEG)
                    body.set_edgecolor('black')
                    body.set_linewidth(0.5)
                    body.set_alpha(0.7)
                med = np.median(preds_neg)
                ax.hlines(med, i - 0.35, i, colors='black', linewidth=1.5)

            # Right half: deceased (positive)
            if len(preds_pos) >= MIN_SAMPLES:
                parts = ax.violinplot(
                    preds_pos, positions=[i], showmedians=False,
                    showextrema=False, widths=0.8
                )
                for body in parts['bodies']:
                    verts = body.get_paths()[0].vertices
                    center = i
                    verts[:, 0] = np.clip(verts[:, 0], center, np.inf)
                    body.set_facecolor(COLOR_POS)
                    body.set_edgecolor('black')
                    body.set_linewidth(0.5)
                    body.set_alpha(0.7)
                med = np.median(preds_pos)
                ax.hlines(med, i, i + 0.35, colors='black', linewidth=1.5)

            # Sample count annotation below the x-axis label (outside the plot area
            # so it can't cover data). constrained_layout reserves bottom margin.
            n_neg = len(preds_neg)
            n_pos = len(preds_pos)
            ax.text(
                i, -0.22, f"n={n_neg}/{n_pos}",
                ha='center', va='top', fontsize=9, color='#555555',
                transform=ax.get_xaxis_transform(), clip_on=False,
            )

        # Format x-axis with requested timepoint labels (combine time + n= on one tick)
        ax.set_xticks(positions)
        if time_unit == 'hours':
            labels = [f"{int(t)}h" for t, _ in timepoint_pairs]
        else:
            labels = [f"{int(t)}d" for t, _ in timepoint_pairs]
        ax.set_xticklabels(labels, fontsize=_FIG_STYLE['tick_label'])

    # 3-row layout: panel A, panel B, dedicated legend row below both.
    # constrained_layout auto-reserves bottom margin for the n=X/Y annotations
    # that live below each panel's x-axis label.
    fig = plt.figure(figsize=(6, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.10])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.axis('off')

    # Panel A: Hours
    avail_hours = sorted(df['time_hours'].unique())
    hour_pairs = _snap_timepoints(avail_hours, hour_timepoints)
    if hour_pairs:
        _draw_split_violins(ax1, df, 'time_hours', hour_pairs, 'hours')

    ax1.set_title("A) Prediction Distribution over Hours", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax1.set_ylabel("Predicted Mortality Risk", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_xlabel("Time (hours)", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Days
    avail_days = sorted(df['time_days'].unique())
    day_pairs = _snap_timepoints(avail_days, day_timepoints)
    if day_pairs:
        _draw_split_violins(ax2, df, 'time_days', day_pairs, 'days')

    ax2.set_title("B) Prediction Distribution over Days", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax2.set_ylabel("Predicted Mortality Risk", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_xlabel("Time (days)", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Shared legend in the dedicated bottom row (below both panels)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEG, edgecolor='black', alpha=0.7, label='Survived'),
        Patch(facecolor=COLOR_POS, edgecolor='black', alpha=0.7, label='Deceased'),
    ]
    ax_legend.legend(
        handles=legend_elements, loc='center', ncol=2,
        fontsize=_FIG_STYLE['legend'], frameon=True, framealpha=0.9,
    )
    ax1.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax2.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

    return fig


def plot_multi_percentile_recall(
    results: List[PercentileRecallResult],
    percentiles: List[int] = (5, 10, 15, 20, 25),
    cut_hours: int = 72,
    max_days: float = None,
):
    """Plot recall over time for multiple top-percentile risk thresholds.

    1x2 layout: hours (left) | days (right), with one line per percentile.

    Args:
        results: Output from evaluate_percentile_recall_over_time().
        percentiles: Percentiles to plot (must match keys in results).
        cut_hours: Hour cutoff for the left subplot.
        max_days: Day limit for right subplot (default: from config).

    Returns:
        matplotlib Figure.
    """
    if max_days is None:
        max_days = get_max_days()
    if not results:
        raise ValueError("No percentile recall results to plot")

    times_h = np.array([r.time_hours for r in results])
    times_d = np.array([r.time_days for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    mask_cut = times_h <= cut_hours

    for i, perc in enumerate(percentiles):
        color = f"C{i}"
        recall_vals = np.array([r.recalls[perc] for r in results])
        recall_lower = np.array([r.recall_cis[perc][0] for r in results])
        recall_upper = np.array([r.recall_cis[perc][1] for r in results])
        label = f"Top {perc}%"

        # --- Left panel: hours ---
        x_h = times_h[mask_cut]
        v_h = recall_vals[mask_cut]
        lo_h = recall_lower[mask_cut]
        hi_h = recall_upper[mask_cut]

        if len(x_h) > 0:
            if x_h[-1] < cut_hours:
                x_h = np.append(x_h, cut_hours)
                v_h = np.append(v_h, v_h[-1])
                lo_h = np.append(lo_h, lo_h[-1])
                hi_h = np.append(hi_h, hi_h[-1])
            ax1.plot(x_h, v_h, color=color, marker='.', label=label,
                     markersize=4, linewidth=2, alpha=0.8)
            ax1.fill_between(x_h, lo_h, hi_h, color=color, alpha=0.15)

        # --- Right panel: days ---
        x_d = times_d
        v_d = recall_vals
        lo_d = recall_lower
        hi_d = recall_upper

        if len(x_d) > 0:
            if x_d[-1] < max_days:
                x_d = np.append(x_d, max_days)
                v_d = np.append(v_d, v_d[-1])
                lo_d = np.append(lo_d, lo_d[-1])
                hi_d = np.append(hi_d, hi_d[-1])
            ax2.plot(x_d, v_d, color=color, marker='.', label=label,
                     markersize=4, linewidth=2, alpha=0.8)
            ax2.fill_between(x_d, lo_d, hi_d, color=color, alpha=0.15)

    # Left panel formatting
    ax1.set_xlabel("Time (hours)", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours + 1, 6 if cut_hours <= 72 else 4))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Sensitivity", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_title(f"A) High-Risk Sensitivity until {cut_hours}h", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=_FIG_STYLE['legend'])
    ax1.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

    # Right panel formatting
    ax2.set_xlabel("Time (days)", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days + 1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Sensitivity", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_title(f"B) High-Risk Sensitivity up to {int(max_days)} days", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=_FIG_STYLE['legend'])
    ax2.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

    plt.tight_layout(pad=3.0)
    return fig


def plot_time_metrics_comparison(
    results_all: List[TimeMetricResult],
    results_active: List[TimeMetricResult],
    cut_hours=72, max_days=None,
    target_name: str = "deceased_30d",
    static_scores: Optional[Dict[str, dict]] = None,
):
    """Overlay all-patients vs active-only AUROC/AUPRC with population context.

    2x2 layout:
        Top row:    performance curves (AUROC/AUPRC, all vs active-only)
        Bottom row: active patient counts, prevalence, and all-patients N reference

    Args:
        static_scores: Optional dict from evaluate_static_scores(). Each entry
            maps score name to {auroc, auroc_ci, auprc, auprc_ci, n}. Drawn as
            horizontal reference lines on performance panels.
    """
    if max_days is None:
        max_days = get_max_days()
    if not results_all or not results_active:
        raise ValueError("Both result sets required for comparison plot")

    # 4-row layout with dedicated legend rows. constrained_layout auto-sizes each
    # row based on actual artist extents (panel titles, x-labels, legend heights)
    # so legends never collide with adjacent panel content.
    fig = plt.figure(figsize=(11, 8.25), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.18, 1.0, 0.18])
    ax_perf_h = fig.add_subplot(gs[0, 0])
    ax_perf_d = fig.add_subplot(gs[0, 1])
    ax_legend_top = fig.add_subplot(gs[1, :])
    ax_legend_top.axis('off')
    ax_count_h = fig.add_subplot(gs[2, 0])
    ax_count_d = fig.add_subplot(gs[2, 1])
    ax_legend_bot = fig.add_subplot(gs[3, :])
    ax_legend_bot.axis('off')

    # ── Top row: performance curves ──────────────────────────────────────
    datasets = [
        ("All patients", results_all, "-"),
        ("Active only", results_active, "--"),
    ]

    for label_prefix, results, linestyle in datasets:
        times_h = np.array([r.time_hours for r in results])
        times_d = np.array([r.time_days for r in results])
        auroc_vals = np.array([r.auroc for r in results])
        auroc_lower = np.array([r.auroc_ci[0] for r in results])
        auroc_upper = np.array([r.auroc_ci[1] for r in results])
        auprc_vals = np.array([r.auprc for r in results])
        auprc_lower = np.array([r.auprc_ci[0] for r in results])
        auprc_upper = np.array([r.auprc_ci[1] for r in results])

        mask_cut = times_h <= cut_hours

        for vals, lower, upper, color, metric_name in [
            (auroc_vals, auroc_lower, auroc_upper, "C0", "AUROC"),
            (auprc_vals, auprc_lower, auprc_upper, "C1", "AUPRC"),
        ]:
            # Hours panel
            valid_h = mask_cut & ~np.isnan(vals)
            x = times_h[valid_h]
            v, lo, hi = vals[valid_h], lower[valid_h], upper[valid_h]
            if len(x) > 0:
                if x[-1] < cut_hours:
                    x = np.append(x, cut_hours)
                    v = np.append(v, v[-1])
                    lo = np.append(lo, lo[-1])
                    hi = np.append(hi, hi[-1])
                ax_perf_h.plot(x, v, color=color, linestyle=linestyle,
                               label=f"{metric_name} ({label_prefix})", markersize=3)
                ax_perf_h.fill_between(x, lo, hi, color=color, alpha=0.1)

            # Days panel
            valid_d = ~np.isnan(vals)
            x = times_d[valid_d]
            v, lo, hi = vals[valid_d], lower[valid_d], upper[valid_d]
            if len(x) > 0:
                if x[-1] < max_days:
                    x = np.append(x, max_days)
                    v = np.append(v, v[-1])
                    lo = np.append(lo, lo[-1])
                    hi = np.append(hi, hi[-1])
                ax_perf_d.plot(x, v, color=color, linestyle=linestyle,
                               label=f"{metric_name} ({label_prefix})", markersize=3)
                ax_perf_d.fill_between(x, lo, hi, color=color, alpha=0.1)

    # ── Static score reference lines (optional) ────────────────────────
    if static_scores:
        score_colors = {"ISS": "C2", "RTS": "C3", "TRISS": "C4"}
        for score_name, metrics in static_scores.items():
            color = score_colors.get(score_name, "C5")
            auroc = metrics["auroc"]
            auprc = metrics["auprc"]
            n = metrics["n"]
            for i, ax in enumerate([ax_perf_h, ax_perf_d]):
                # Only add labels on first axis (hours) to avoid legend duplicates
                auroc_label = f"AUROC {score_name} ({auroc:.3f}, n={n})" if i == 0 else None
                auprc_label = f"AUPRC {score_name} ({auprc:.3f}, n={n})" if i == 0 else None
                ax.axhline(y=auroc, color=color, linestyle=":", linewidth=1.5,
                           label=auroc_label)
                ax.axhline(y=auprc, color=color, linestyle="-.", linewidth=1.5,
                           label=auprc_label)

    for ax, xlabel, xlim, xticks, title in [
        (ax_perf_h, "Time (hours)", cut_hours,
         np.arange(0, cut_hours + 1, 6), "A) Performance over Hours"),
        (ax_perf_d, "Time (days)", max_days,
         np.arange(0, max_days + 1, 5), "B) Performance over Days"),
    ]:
        ax.set_xlabel(xlabel, fontsize=_FIG_STYLE['axis_label'])
        ax.set_xlim(0, xlim)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_ylabel("Score", fontsize=_FIG_STYLE['axis_label'])
        ax.set_title(title, fontsize=_FIG_STYLE['title'], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
        ax.set_ylim(0.0, 1.0)

    # ── Bottom row: patient counts & prevalence ──────────────────────────
    PREV_COLOR = "#1F77B4"
    ACTIVE_COLOR = "#2CA02C"
    POSITIVE_COLOR = "#D62728"
    ALL_COLOR = "#7F7F7F"

    act_times_h = np.array([r.time_hours for r in results_active])
    act_times_d = np.array([r.time_days for r in results_active])
    act_n_samples = np.array([r.n_samples for r in results_active])
    act_n_positive = np.array([r.n_positive for r in results_active])
    act_prevalence = np.where(act_n_samples > 0, act_n_positive / act_n_samples, 0.0)

    all_n = results_all[0].n_samples if results_all else 0
    mask_cut_act = act_times_h <= cut_hours

    prev_ax_ref = None
    for ax, times, n_samp, n_pos, prev, mask, xlabel, xlim, title in [
        (ax_count_h, act_times_h, act_n_samples, act_n_positive, act_prevalence,
         mask_cut_act, "Time (hours)", cut_hours, "C) Active Patients over Hours"),
        (ax_count_d, act_times_d, act_n_samples, act_n_positive, act_prevalence,
         np.ones(len(act_times_d), dtype=bool), "Time (days)", max_days,
         "D) Active Patients over Days"),
    ]:
        ax.plot(times[mask], n_samp[mask], color=ACTIVE_COLOR, label="Active patients")
        ax.plot(times[mask], n_pos[mask], color=POSITIVE_COLOR,
                label=f"{_display_target(target_name)} (active)")
        ax.axhline(y=all_n, color=ALL_COLOR, linestyle=":", linewidth=1.2,
                    label=f"All patients (N={all_n})")
        ax.set_xlabel(xlabel, fontsize=_FIG_STYLE['axis_label'])
        ax.set_xlim(0, xlim)
        ax.set_ylabel("Count", fontsize=_FIG_STYLE['axis_label'])
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=_FIG_STYLE['title'], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

        ax_prev = ax.twinx()
        ax_prev.plot(times[mask], prev[mask] * 100, color=PREV_COLOR,
                     linestyle="--", linewidth=1.5, label="Prevalence (%)")
        ax_prev.set_ylabel("Prevalence (%)", fontsize=_FIG_STYLE['axis_label'], color=PREV_COLOR)
        ax_prev.set_ylim(0, 12)
        ax_prev.tick_params(axis='y', labelcolor=PREV_COLOR, labelsize=_FIG_STYLE['tick_label'])
        if prev_ax_ref is None:
            prev_ax_ref = ax_prev

    # ── Legends in dedicated gridspec rows (no overlap with panels) ──────
    perf_handles, perf_labels = ax_perf_h.get_legend_handles_labels()
    ax_legend_top.legend(
        perf_handles, perf_labels, loc='center', ncol=2,
        fontsize=_FIG_STYLE['legend'], frameon=True, framealpha=0.9,
    )

    count_handles, count_labels = ax_count_h.get_legend_handles_labels()
    prev_handles, prev_labels = prev_ax_ref.get_legend_handles_labels()
    ax_legend_bot.legend(
        count_handles + prev_handles, count_labels + prev_labels,
        loc='center', ncol=4, fontsize=_FIG_STYLE['legend'],
        frameon=True, framealpha=0.9,
    )

    return fig


def plot_trauma_score_comparison(
    score_name: str,
    paired: Dict[str, List[TimeMetricResult]],
    cut_hours=72, max_days=None,
    target_name: str = "deceased_30d",
    results_counts: Optional[List[TimeMetricResult]] = None,
):
    """Single score vs HNN comparison on identical patients per timestep.

    2x2 layout:
        Top row:    AUROC/AUPRC over time (HNN + score as curves with CIs)
        Bottom row: active patient counts, prevalence for this score's subset

    Args:
        score_name: Name of the score (e.g. "RTS", "TRISS").
        paired: {"score": List[TimeMetricResult], "model": List[TimeMetricResult]}.
            Model results are computed on the exact same patients as the score
            at each timestep (fair comparison).
        results_counts: Optional separate results for count panels (bottom row).

    Convention: AUROC = solid line, AUPRC = dotted line, same color per model.
    """
    if max_days is None:
        max_days = get_max_days()

    score_n = paired["score"][0].n_samples if paired["score"] else 0
    # 4-row layout with dedicated legend rows (mirrors plot_time_metrics_comparison).
    # constrained_layout auto-sizes rows to avoid overlap with titles/x-labels.
    fig = plt.figure(figsize=(11, 8.25), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.18, 1.0, 0.18])
    ax_perf_h = fig.add_subplot(gs[0, 0])
    ax_perf_d = fig.add_subplot(gs[0, 1])
    ax_legend_top = fig.add_subplot(gs[1, :])
    ax_legend_top.axis('off')
    ax_count_h = fig.add_subplot(gs[2, 0])
    ax_count_d = fig.add_subplot(gs[2, 1])
    ax_legend_bot = fig.add_subplot(gs[3, :])
    ax_legend_bot.axis('off')

    # ── Color assignments ────────────────────────────────────────────────
    score_color = "C3"
    hnn_color = "C0"

    # ── Helper to plot AUROC (solid) + AUPRC (dotted) for one model ──────
    def _plot_model(results, model_name, color, alpha_ci=0.12):
        times_h = np.array([r.time_hours for r in results])
        times_d = np.array([r.time_days for r in results])
        auroc_vals = np.array([r.auroc for r in results])
        auroc_lo = np.array([r.auroc_ci[0] for r in results])
        auroc_hi = np.array([r.auroc_ci[1] for r in results])
        auprc_vals = np.array([r.auprc for r in results])
        auprc_lo = np.array([r.auprc_ci[0] for r in results])
        auprc_hi = np.array([r.auprc_ci[1] for r in results])
        mask_cut = times_h <= cut_hours

        for vals, lo, hi, ls, metric in [
            (auroc_vals, auroc_lo, auroc_hi, "-", "AUROC"),
            (auprc_vals, auprc_lo, auprc_hi, ":", "AUPRC"),
        ]:
            # Hours panel
            valid_h = mask_cut & ~np.isnan(vals)
            x = times_h[valid_h]
            v, vlo, vhi = vals[valid_h], lo[valid_h], hi[valid_h]
            if len(x) > 0:
                if x[-1] < cut_hours:
                    x = np.append(x, cut_hours)
                    v = np.append(v, v[-1])
                    vlo = np.append(vlo, vlo[-1])
                    vhi = np.append(vhi, vhi[-1])
                ax_perf_h.plot(x, v, color=color, linestyle=ls,
                               label=f"{metric} ({model_name})", linewidth=1.8)
                ax_perf_h.fill_between(x, vlo, vhi, color=color, alpha=alpha_ci)

            # Days panel (no duplicate labels)
            valid_d = ~np.isnan(vals)
            x = times_d[valid_d]
            v, vlo, vhi = vals[valid_d], lo[valid_d], hi[valid_d]
            if len(x) > 0:
                if x[-1] < max_days:
                    x = np.append(x, max_days)
                    v = np.append(v, v[-1])
                    vlo = np.append(vlo, vlo[-1])
                    vhi = np.append(vhi, vhi[-1])
                ax_perf_d.plot(x, v, color=color, linestyle=ls, linewidth=1.8)
                ax_perf_d.fill_between(x, vlo, vhi, color=color, alpha=alpha_ci)

    # Plot HNN (on this score's patient set) then the score itself
    _plot_model(paired["model"], "HNN", hnn_color, alpha_ci=0.12)
    _plot_model(paired["score"], score_name, score_color, alpha_ci=0.08)

    subset_label = f" (N={score_n})"
    for ax, xlabel, xlim, xticks, title in [
        (ax_perf_h, "Time (hours)", cut_hours,
         np.arange(0, cut_hours + 1, 6),
         f"A) HNN vs {score_name}{subset_label} — Hours"),
        (ax_perf_d, "Time (days)", max_days,
         np.arange(0, max_days + 1, 5),
         f"B) HNN vs {score_name}{subset_label} — Days"),
    ]:
        ax.set_xlabel(xlabel, fontsize=_FIG_STYLE['axis_label'])
        ax.set_xlim(0, xlim)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_ylabel("Score", fontsize=_FIG_STYLE['axis_label'])
        ax.set_title(title, fontsize=_FIG_STYLE['title'], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
        ax.set_ylim(0.0, 1.0)

    # Performance legend — dedicated gridspec row (no overlap risk)
    perf_handles, perf_labels = ax_perf_h.get_legend_handles_labels()
    ax_legend_top.legend(
        perf_handles, perf_labels, loc='center', ncol=2,
        fontsize=_FIG_STYLE['legend'], frameon=True, framealpha=0.9,
    )

    # ── Bottom row: patient counts & prevalence ──────────────────────────
    PREV_COLOR = "#1F77B4"
    ACTIVE_COLOR = "#2CA02C"
    POSITIVE_COLOR = "#D62728"

    # Use results_counts for bottom panels if provided (covers full time range)
    count_source = results_counts if results_counts is not None else paired["model"]
    times_h = np.array([r.time_hours for r in count_source])
    times_d = np.array([r.time_days for r in count_source])
    act_n_samples = np.array([r.n_samples for r in count_source])
    act_n_positive = np.array([r.n_positive for r in count_source])
    act_prevalence = np.where(act_n_samples > 0, act_n_positive / act_n_samples, 0.0)

    # Extend to end of time range with zeros so lines go to 0 instead of clipping
    if len(times_d) > 0 and times_d[-1] < max_days:
        times_h = np.append(times_h, max_days * 24.0)
        times_d = np.append(times_d, max_days)
        act_n_samples = np.append(act_n_samples, 0)
        act_n_positive = np.append(act_n_positive, 0)
        act_prevalence = np.append(act_prevalence, 0.0)

    mask_cut_act = times_h <= cut_hours

    prev_ax_ref = None
    for ax, times, n_samp, n_pos, prev, mask, xlabel, xlim, title in [
        (ax_count_h, times_h, act_n_samples, act_n_positive, act_prevalence,
         mask_cut_act, "Time (hours)", cut_hours, "C) Active Patients over Hours"),
        (ax_count_d, times_d, act_n_samples, act_n_positive, act_prevalence,
         np.ones(len(times_d), dtype=bool), "Time (days)", max_days,
         "D) Active Patients over Days"),
    ]:
        ax.plot(times[mask], n_samp[mask], color=ACTIVE_COLOR, label="Active patients")
        ax.plot(times[mask], n_pos[mask], color=POSITIVE_COLOR,
                label=f"{_display_target(target_name)} (active)")
        ax.set_xlabel(xlabel, fontsize=_FIG_STYLE['axis_label'])
        ax.set_xlim(0, xlim)
        ax.set_ylabel("Count", fontsize=_FIG_STYLE['axis_label'])
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=_FIG_STYLE['title'], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

        ax_prev = ax.twinx()
        ax_prev.plot(times[mask], prev[mask] * 100, color=PREV_COLOR,
                     linestyle="--", linewidth=1.5, label="Prevalence (%)")
        ax_prev.set_ylabel("Prevalence (%)", fontsize=_FIG_STYLE['axis_label'], color=PREV_COLOR)
        ax_prev.set_ylim(0, 12)
        ax_prev.tick_params(axis='y', labelcolor=PREV_COLOR, labelsize=_FIG_STYLE['tick_label'])
        if prev_ax_ref is None:
            prev_ax_ref = ax_prev

    # Count legend — dedicated gridspec row
    count_handles, count_labels = ax_count_h.get_legend_handles_labels()
    prev_handles, prev_labels = prev_ax_ref.get_legend_handles_labels()
    ax_legend_bot.legend(
        count_handles + prev_handles, count_labels + prev_labels,
        loc='center', ncol=3, fontsize=_FIG_STYLE['legend'],
        frameon=True, framealpha=0.9,
    )

    return fig


def plot_delong_comparison(
    score_name: str,
    paired: Dict,
    cut_hours=72, max_days=None,
):
    """Standalone DeLong statistical comparison: HNN vs a trauma score over time.

    2x2 layout:
        Top row:    Delta AUROC (HNN - score) with 95% CI from DeLong SE
        Bottom row: -log10(FDR-adjusted p-value) trajectory

    Left column = hours (0 to cut_hours), right column = days (0 to max_days).

    Args:
        score_name: Name of the baseline score (e.g. "RTS", "TRISS").
        paired: Dict from evaluate_static_scores_over_time(delong=True).
            Must contain keys: delong_hours, delong_delta, delong_se,
            delong_p_adj, delong_significant.
    """
    if max_days is None:
        max_days = get_max_days()

    hours = np.array(paired["delong_hours"])
    days = hours / 24.0
    delta = np.array(paired["delong_delta"])
    se = np.array(paired["delong_se"])
    p_adj = np.array(paired["delong_p_adj"])
    sig = np.array(paired["delong_significant"])

    ci_lo = delta - 1.96 * se
    ci_hi = delta + 1.96 * se

    # Clamp p_adj floor for log transform (avoid -log10(0) = inf)
    p_adj_safe = np.clip(p_adj, 1e-20, 1.0)
    neg_log_p = -np.log10(p_adj_safe)

    SIG_COLOR = "#2CA02C"
    NONSIG_COLOR = "#999999"
    DELTA_COLOR = "C0"

    # 4-row layout: suptitle at top, panels A/B, dedicated legend axis,
    # panels C/D, dedicated legend axis. Matches plot_time_metrics_comparison.
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.32,
                          height_ratios=[1.0, 0.10, 1.0, 0.12],
                          top=0.86, bottom=0.05, left=0.08, right=0.97)
    ax_d_h = fig.add_subplot(gs[0, 0])
    ax_d_d = fig.add_subplot(gs[0, 1])
    ax_legend_top = fig.add_subplot(gs[1, :])
    ax_legend_top.axis('off')
    ax_p_h = fig.add_subplot(gs[2, 0])
    ax_p_d = fig.add_subplot(gs[2, 1])
    ax_legend_bot = fig.add_subplot(gs[3, :])
    ax_legend_bot.axis('off')

    n_sig = int(sig.sum())
    n_total = len(sig)

    # ── Top row: Delta AUROC with CI ─────────────────────────────────────
    for ax, times, xlim, xlabel, title_lbl in [
        (ax_d_h, hours, cut_hours, "Time (hours)", "A"),
        (ax_d_d, days, max_days, "Time (days)", "B"),
    ]:
        mask = times <= xlim
        t = times[mask]
        d, lo, hi = delta[mask], ci_lo[mask], ci_hi[mask]
        s = sig[mask]

        # CI band
        ax.fill_between(t, lo, hi, color=DELTA_COLOR, alpha=0.12)
        # Delta line
        ax.plot(t, d, color=DELTA_COLOR, linewidth=1.5)
        # Green fill where significant and HNN wins
        sig_win = s & (d > 0)
        if sig_win.any():
            ax.fill_between(t, 0, d,
                            where=sig_win, color=SIG_COLOR, alpha=0.18,
                            label="Significant (FDR<0.05)")
        # Reference line at 0
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

        ax.set_xlabel(xlabel, fontsize=_FIG_STYLE['axis_label'])
        ax.set_xlim(0, xlim)
        ax.set_ylabel(r"$\Delta$ AUROC (HNN $-$ " + score_name + ")", fontsize=_FIG_STYLE['axis_label'])
        ax.set_title(
            f"{title_lbl}) Delta AUROC: HNN vs {score_name}",
            fontsize=_FIG_STYLE['title'], fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

    # Legend for top row (dedicated axis, no overlap with titles)
    d_handles, d_labels = ax_d_h.get_legend_handles_labels()
    if d_handles:
        ax_legend_top.legend(
            d_handles, d_labels, loc='center', ncol=2,
            fontsize=_FIG_STYLE['legend'], frameon=True, framealpha=0.9,
        )

    # ── Bottom row: -log10(p_adj) trajectory ─────────────────────────────
    threshold = -np.log10(0.05)

    for ax, times, xlim, xlabel, title_lbl in [
        (ax_p_h, hours, cut_hours, "Time (hours)", "C"),
        (ax_p_d, days, max_days, "Time (days)", "D"),
    ]:
        mask = times <= xlim
        t = times[mask]
        nlp = neg_log_p[mask]
        s = sig[mask]

        # Scatter: green = significant, gray = not
        ax.scatter(t[s], nlp[s], c=SIG_COLOR, s=18, zorder=3,
                   label="Significant (FDR<0.05)")
        ax.scatter(t[~s], nlp[~s], c=NONSIG_COLOR, s=18, zorder=3,
                   label="Not significant")
        # Connect with a thin line
        ax.plot(t, nlp, color='#666666', linewidth=0.6, alpha=0.5, zorder=2)
        # Significance threshold
        ax.axhline(threshold, color='red', linewidth=1.0, linestyle='--',
                    alpha=0.6, label=r"$\alpha$ = 0.05")

        ax.set_xlabel(xlabel, fontsize=_FIG_STYLE['axis_label'])
        ax.set_xlim(0, xlim)
        ax.set_ylabel(r"$-\log_{10}(p_{adj})$", fontsize=_FIG_STYLE['axis_label'])
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"{title_lbl}) DeLong p-value (FDR-corrected)",
            fontsize=_FIG_STYLE['title'], fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

    # Legend for bottom row (dedicated axis)
    p_handles, p_labels = ax_p_h.get_legend_handles_labels()
    if p_handles:
        ax_legend_bot.legend(
            p_handles, p_labels, loc='center', ncol=3,
            fontsize=_FIG_STYLE['legend'], frameon=True, framealpha=0.9,
        )

    # Suptitle + summary inside figure coords (top=0.86 reserved by GridSpec)
    mean_delta = float(np.mean(delta))
    summary = (
        f"DeLong paired test: {n_sig}/{n_total} time points significant "
        f"(BH-FDR<0.05), mean {chr(916)}AUROC = {mean_delta:+.3f}"
    )
    fig.suptitle(
        f"HNN vs {score_name} — Statistical Comparison",
        fontsize=_FIG_STYLE['suptitle'], fontweight='bold', y=0.97,
    )
    fig.text(0.5, 0.91, summary, ha='center', va='center',
             fontsize=_FIG_STYLE['annotation'], fontstyle='italic', color='#444444')

    return fig


def plot_n_active_over_time(
    results_active: List[TimeMetricResult],
    cut_hours=72, max_days=None,
    target_name: str = "deceased_30d"
):
    """Show active patient count, outcome-positive count, and prevalence over time."""
    if max_days is None:
        max_days = get_max_days()
    if not results_active:
        raise ValueError("No active-only results to plot")

    times_h = np.array([r.time_hours for r in results_active])
    times_d = np.array([r.time_days for r in results_active])
    n_samples = np.array([r.n_samples for r in results_active])
    n_positive = np.array([r.n_positive for r in results_active])
    prevalence = np.where(n_samples > 0, n_positive / n_samples, 0.0)

    PREV_COLOR = "#1F77B4"  # blue
    ACTIVE_COLOR = "#2CA02C"  # green
    POSITIVE_COLOR = "#D62728"  # red

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax in [ax1, ax2]:
        ax.set_box_aspect(1)

    mask_cut = times_h <= cut_hours

    # Hours panel
    ax1.plot(times_h[mask_cut], n_samples[mask_cut], color=ACTIVE_COLOR, label="Active patients")
    ax1.plot(times_h[mask_cut], n_positive[mask_cut], color=POSITIVE_COLOR, label=f"{_display_target(target_name)} (active)")
    ax1.set_xlabel("Time (hours)", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_xlim(0, cut_hours)
    ax1.set_ylabel("Count", fontsize=_FIG_STYLE['axis_label'])
    ax1.set_ylim(bottom=0)
    ax1.set_title("A) Active Patients over Hours", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

    ax1_prev = ax1.twinx()
    ax1_prev.plot(times_h[mask_cut], prevalence[mask_cut] * 100, color=PREV_COLOR,
                  linestyle="--", linewidth=1.5, label="Prevalence (%)")
    ax1_prev.set_ylabel("Prevalence (%)", fontsize=_FIG_STYLE['axis_label'], color=PREV_COLOR)
    ax1_prev.set_ylim(0, 12)
    ax1_prev.tick_params(axis='y', labelcolor=PREV_COLOR, labelsize=_FIG_STYLE['tick_label'])

    # Days panel
    ax2.plot(times_d, n_samples, color=ACTIVE_COLOR, label="Active patients")
    ax2.plot(times_d, n_positive, color=POSITIVE_COLOR, label=f"{_display_target(target_name)} (active)")
    ax2.set_xlabel("Time (days)", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_xlim(0, max_days)
    ax2.set_ylabel("Count", fontsize=_FIG_STYLE['axis_label'])
    ax2.set_ylim(bottom=0)
    ax2.set_title("B) Active Patients over Days", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])

    ax2_prev = ax2.twinx()
    ax2_prev.plot(times_d, prevalence * 100, color=PREV_COLOR,
                  linestyle="--", linewidth=1.5, label="Prevalence (%)")
    ax2_prev.set_ylabel("Prevalence (%)", fontsize=_FIG_STYLE['axis_label'], color=PREV_COLOR)
    ax2_prev.set_ylim(0, 12)
    ax2_prev.tick_params(axis='y', labelcolor=PREV_COLOR, labelsize=_FIG_STYLE['tick_label'])

    # Combined legend below
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_prev.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=3, fontsize=_FIG_STYLE['legend'],
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def _plot_roc_pr_curves_from_arrays(
    censor_steps: List[int],
    preds_per_step: List[np.ndarray],
    targets_per_step: List[np.ndarray],
    labels: Optional[List[str]] = None,
):
    """Shared plotting logic for ROC/PR multi-curve plots.

    Args:
        censor_steps: List of censor step indices.
        preds_per_step: List of 1-D prediction arrays, one per step.
        targets_per_step: List of 1-D target arrays, one per step.
        labels: Optional display labels per step.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
              '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']

    baseline = None

    for i, censor_step in enumerate(censor_steps):
        y_preds = preds_per_step[i]
        ys = targets_per_step[i]

        if len(set(ys)) < 2:
            logger.warning(f"Skipping step {censor_step}: only one class")
            continue

        if baseline is None:
            baseline = ys.sum() / len(ys)

        label = labels[i] if labels and i < len(labels) else format_step_label(censor_step)
        color = colors[i % len(colors)]

        fpr, tpr, _ = roc_curve(ys, y_preds)
        roc_auc = roc_auc_score(ys, y_preds)
        ax_roc.plot(fpr, tpr, color=color, label=f"{label} (AUC={roc_auc:.3f})", linewidth=2)

        precision, recall, _ = precision_recall_curve(ys, y_preds)
        auprc = average_precision_score(ys, y_preds)
        ax_pr.plot(recall, precision, color=color, label=f"{label} (AUC={auprc:.3f})", linewidth=2)

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, c="grey", alpha=0.7, label='Chance')
    ax_roc.set_title("ROC Curves at Different Time Points", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=_FIG_STYLE['axis_label'])
    ax_roc.set_ylabel("True Positive Rate", fontsize=_FIG_STYLE['axis_label'])
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(fontsize=_FIG_STYLE['legend'], title="Time Available", title_fontsize=_FIG_STYLE['legend'])
    ax_roc.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax_roc.set_aspect('equal', adjustable='box')

    if baseline is not None:
        ax_pr.axhline(y=baseline, color='grey', linestyle='--', lw=1.5, alpha=0.7, label=f'Baseline ({baseline:.3f})')
    ax_pr.set_title("Precision-Recall Curves at Different Time Points", fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=_FIG_STYLE['axis_label'])
    ax_pr.set_ylabel("Precision", fontsize=_FIG_STYLE['axis_label'])
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=_FIG_STYLE['legend'],
                title="Time Available", title_fontsize=_FIG_STYLE['legend'])
    ax_pr.tick_params(axis='both', labelsize=_FIG_STYLE['tick_label'])
    ax_pr.set_aspect('equal', adjustable='box')

    fig.subplots_adjust(right=0.82, wspace=0.3)
    plt.tight_layout()

    return fig


def plot_multiple_roc_pr_curves(
    evaluator: TimeDependentEvaluator,
    censor_steps: List[int],
    labels: Optional[List[str]] = None
):
    preds_list = []
    targs_list = []
    valid_steps = []
    valid_labels = []

    for i, censor_step in enumerate(censor_steps):
        dls = evaluator.create_censored_dataloaders_fast(censor_step)
        if dls is None:
            logger.warning(f"Skipping step {censor_step}: dataloader creation failed")
            continue

        preds, targets = _get_predictions(evaluator.model, dls.train, evaluator.device)
        preds_list.append(preds[:, 1].numpy())
        targs_list.append(targets.numpy())
        valid_steps.append(censor_step)
        valid_labels.append(
            labels[i] if labels and i < len(labels) else format_step_label(censor_step)
        )

    return _plot_roc_pr_curves_from_arrays(valid_steps, preds_list, targs_list, valid_labels)


def plot_multiple_roc_pr_curves_temporal(
    temporal_eval: 'TemporalEvaluator',
    censor_steps: List[int],
    labels: Optional[List[str]] = None,
):
    """Create multi-curve ROC/PR plot using temporal (single forward pass) predictions."""
    preds_all = temporal_eval._get_all_predictions()  # [N, seq_len]
    ys = np.array(temporal_eval.data["ty"])
    traj_lengths = temporal_eval._holdout_traj_lengths

    preds_list = []
    targs_list = []
    valid_steps = []
    valid_labels = []

    for i, censor_step in enumerate(censor_steps):
        # Apply active-only filtering if configured
        if temporal_eval.active_only and len(traj_lengths) > 0:
            mask = traj_lengths > censor_step
            if mask.sum() < 2:
                logger.warning(f"Skipping step {censor_step}: too few active patients ({mask.sum()})")
                continue
            preds_subset = preds_all[mask]
            ys_subset = ys[mask]
            traj_subset = traj_lengths[mask]
        else:
            preds_subset = preds_all
            ys_subset = ys
            traj_subset = traj_lengths

        # Pick effective timestep per patient (min of censor_step, traj_end)
        if len(traj_subset) > 0:
            effective_steps = np.minimum(censor_step, traj_subset - 1).astype(int)
            effective_steps = np.maximum(effective_steps, 0)
        else:
            effective_steps = np.full(len(preds_subset), min(censor_step, preds_subset.shape[1] - 1), dtype=int)

        y_preds = preds_subset[np.arange(len(preds_subset)), effective_steps]

        preds_list.append(y_preds)
        targs_list.append(ys_subset)
        valid_steps.append(censor_step)
        valid_labels.append(
            labels[i] if labels and i < len(labels) else format_step_label(censor_step)
        )

    return _plot_roc_pr_curves_from_arrays(valid_steps, preds_list, targs_list, valid_labels)


def _run_trauma_score_comparison(data, cfg, results_all, results_active,
                                 preds_df_active, model_name,
                                 delong: bool = False):
    """Shared trauma score comparison logic for both temporal and non-temporal paths."""
    try:
        from astra.evaluation.trauma_scores import (
            build_trauma_score_df,
            evaluate_static_scores,
            evaluate_static_scores_over_time,
        )

        logger.info("=" * 80)
        logger.info("TRAUMA SCORE COMPARISON")
        logger.info("=" * 80)

        trauma_df = build_trauma_score_df(data, cfg)
        holdout_pids = data["holdout"].base.PID.values
        holdout_y = np.array(data["ty"])

        # B) Static reference lines on the standard comparison plot
        static_scores_all = evaluate_static_scores(
            trauma_df, holdout_y, holdout_pids
        )

        if static_scores_all and results_active and results_all:
            fig_cmp_ts = plot_time_metrics_comparison(
                results_all, results_active,
                target_name=cfg["target"],
                static_scores=static_scores_all,
            )
            save_figure(
                fig_cmp_ts,
                f"time_metrics_comparison_trauma_{model_name}",
                save_dir=f'reports/eval/{model_name}',
                **_SUBMISSION_KW,
            )
            logger.info("Comparison plot with trauma score baselines saved")

        # C) Filtered comparison: patients with RTS scores, time-varying
        rts_valid = trauma_df.dropna(subset=["RTS"])
        valid_pids = rts_valid["PID"].values
        logger.info(f"Filtered subset: {len(valid_pids)} patients with RTS scores")

        if len(valid_pids) >= 20 and preds_df_active is not None:
            # Time-varying score + model metrics on identical patients per step
            score_results = evaluate_static_scores_over_time(
                trauma_df, preds_df_active, holdout_y, holdout_pids,
                valid_pids=valid_pids,
                delong=delong,
            )

            if score_results:
                # One plot per score (each has different patient population)
                for sname, paired in score_results.items():
                    if sname == "ISS":
                        continue
                    fig_trauma = plot_trauma_score_comparison(
                        sname, paired,
                        target_name=cfg["target"],
                        results_counts=paired["counts"],
                    )
                    save_figure(
                        fig_trauma,
                        f"trauma_{sname.lower()}_comparison_{model_name}",
                        save_dir=f'reports/eval/{model_name}',
                        **_SUBMISSION_KW,
                    )
                    logger.info(f"HNN vs {sname} comparison plot saved")

                    # Standalone DeLong significance plot
                    if "delong_significant" in paired:
                        fig_dl = plot_delong_comparison(sname, paired)
                        save_figure(
                            fig_dl,
                            f"delong_{sname.lower()}_comparison_{model_name}",
                            save_dir=f'reports/eval/{model_name}',
                            **_SUBMISSION_KW,
                        )
                        logger.info(f"DeLong {sname} comparison plot saved")
        else:
            logger.warning(
                f"Too few patients with RTS ({len(valid_pids)}) or "
                f"no active predictions — skipping filtered comparison"
            )

    except Exception as e:
        logger.error(f"Trauma score comparison failed: {e}", exc_info=True)


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def run_eval(data, cfg: dict, multicurve: bool = True, comprehensive_eval: bool = True,
             active_only: bool = False, trauma_scores: bool = False,
             delong: bool = False):
    """
    Enhanced evaluation with time-dependent metrics.

    Uses direct model inference (no FastAI Learner).

    Args:
        active_only: If True, also runs active-only evaluation and generates
                     comparison plots (all patients vs active-only).
        trauma_scores: If True, compute traditional trauma risk scores (RTS, ISS,
                      TRISS) and add them as baselines to comparison plots.
        delong: If True, run paired DeLong tests between HNN and trauma scores
                at each timestep with Benjamini-Hochberg FDR correction.
                Only effective when trauma_scores=True.
    """
    model_name = cfg["model_name"]
    holdout_mixed_dls = data["holdout_mixed_dls"]

    model_cfg = cfg.get("model", {})
    is_temporal = model_cfg.get("temporal_head", False)

    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    model, device = prepare_model(data, cfg)

    # ============================================================================
    # TEMPORAL MODEL: single-forward-pass evaluation
    # ============================================================================
    if is_temporal:
        logger.info("Running temporal baseline evaluation (last timestep)...")
        temporal_eval = TemporalEvaluator(data, model, cfg, device=device)
        preds_all = temporal_eval._get_all_predictions()

        traj_lens = temporal_eval._holdout_traj_lengths
        if len(traj_lens) > 0:
            last_steps = np.minimum(preds_all.shape[1] - 1, traj_lens - 1).astype(int)
            last_steps = np.maximum(last_steps, 0)
        else:
            last_steps = np.full(len(preds_all), preds_all.shape[1] - 1, dtype=int)
        baseline_preds = preds_all[np.arange(len(preds_all)), last_steps]
        targs = np.array(data["ty"])
        evalplt = plot_evaluation(
            torch.tensor(baseline_preds), torch.tensor(targs), cfg["target"]
        )
        save_figure(evalplt, f"baseline_eval_{model_name}",
                    save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
        logger.info("Baseline temporal evaluation saved")

        # Decision Curve Analysis (baseline — full trajectory)
        fig_dca = plot_decision_curve(targs, baseline_preds, model_name=model_name)
        save_figure(fig_dca, f"dca_baseline_{model_name}",
                    save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
        logger.info("Baseline decision curve saved")

        # Confusion matrices at F-beta optimised thresholds (per-timestep calibrated)
        from astra.evaluation.posthoc_calibration import (
            fit_temporal_calibrator, TemporalCalibrator,
        )
        logger.info("Fitting per-timestep temporal calibrator on trainval...")
        temporal_calibrator = fit_temporal_calibrator(
            data, model, cfg, device=device, method='isotonic',
            save_path=f'models/calibrators/{model_name}/temporal_calibrator.pkl',
        )

        # Calibrate holdout predictions at each patient's last step
        ho_traj = temporal_eval._holdout_traj_lengths
        ho_cal_baseline = np.empty_like(baseline_preds)
        for step_val in np.unique(last_steps):
            mask = last_steps == step_val
            ho_cal_baseline[mask] = temporal_calibrator.transform_at_step(
                baseline_preds[mask], int(step_val)
            )

        # Calibrate trainval last-step predictions for threshold finding
        from astra.evaluation.posthoc_calibration import _TrainvalTemporalEvaluator
        _tv_eval = _TrainvalTemporalEvaluator(data, model, cfg, device)
        tv_preds_cat = _tv_eval._get_all_predictions()
        tv_targs = _tv_eval._y
        tv_traj = _tv_eval._traj_lengths
        if len(tv_traj) > 0:
            tv_last = np.minimum(tv_preds_cat.shape[1] - 1, tv_traj - 1).astype(int)
            tv_last = np.maximum(tv_last, 0)
        else:
            tv_last = np.full(len(tv_preds_cat), tv_preds_cat.shape[1] - 1, dtype=int)
        tv_baseline_preds = tv_preds_cat[np.arange(len(tv_preds_cat)), tv_last]
        tv_cal = np.empty_like(tv_baseline_preds)
        for step_val in np.unique(tv_last):
            mask = tv_last == step_val
            tv_cal[mask] = temporal_calibrator.transform_at_step(
                tv_baseline_preds[mask], int(step_val)
            )
        logger.info("Per-timestep calibrator fitted and applied")

        # Evaluate calibration quality per window
        cal_eval_df = temporal_calibrator.evaluate(
            preds_all, targs, ho_traj if len(ho_traj) > 0 else None
        )
        cal_csv = f'reports/eval/{model_name}/calibration/temporal_calibration_eval.csv'
        ensure_parent_dir(cal_csv)
        cal_eval_df.to_csv(cal_csv, index=False)
        logger.info(f"Per-window calibration evaluation:\n{cal_eval_df.to_string(index=False)}")

        # Find thresholds on calibrated trainval, evaluate on calibrated holdout
        for beta, label in [(1, "F1"), (5, "F5")]:
            thr, score = find_optimal_fbeta_threshold(tv_targs, tv_cal, beta=beta)
            logger.info(f"  {label} optimal threshold={thr:.4f} (score={score:.4f}) on calibrated trainval")
            fig_cm, _, _ = evaluate_detection_rate(
                ho_cal_baseline, targs, threshold=thr, label=label
            )
            save_figure(fig_cm, f"cm_{label}_{model_name}",
                        save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)

        key_timepoints = None
        if multicurve:
            max_step = get_total_steps() - 1
            key_timepoints = sorted({
                min(t, max_step) for t in [
                    time_to_step(1, 'h'), time_to_step(6, 'h'),
                    time_to_step(12, 'h'), time_to_step(72, 'h'),
                    time_to_step(7, 'D'), time_to_step(14, 'D'),
                    time_to_step(30, 'D'), time_to_step(90, 'D'),
                ] if t is not None
            })

            logger.info("Performance at key time points (temporal model):")
            for step in key_timepoints:
                result = temporal_eval.evaluate_at_timestep(step)
                if result:
                    line = (
                        f"  {format_step_label(step):>12s}: "
                        f"AUROC={result.auroc:.3f} [{result.auroc_ci[0]:.3f}-{result.auroc_ci[1]:.3f}], "
                        f"AUPRC={result.auprc:.3f} [{result.auprc_ci[0]:.3f}-{result.auprc_ci[1]:.3f}]"
                    )
                    if result.cindex is not None:
                        line += f", C-index={result.cindex:.3f}"
                        if result.cindex_ci:
                            line += f" [{result.cindex_ci[0]:.3f}-{result.cindex_ci[1]:.3f}]"
                    logger.info(line)

            # ROC/PR curves at key timepoints (temporal)
            labels = [format_step_label(step) for step in key_timepoints]
            fig_curves = plot_multiple_roc_pr_curves_temporal(
                temporal_eval, key_timepoints, labels=labels
            )
            save_figure(fig_curves, f"multi_curves_{model_name}",
                        save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
            logger.info("Multiple ROC/PR curves plot saved (temporal)")

            # Decision curves at key timepoints (temporal)
            fig_dca_time = _plot_decision_curves_temporal(
                preds_all, targs, traj_lens, key_timepoints, labels=labels
            )
            save_figure(fig_dca_time, f"dca_multicurve_{model_name}",
                        save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
            logger.info("Time-dependent decision curves saved (temporal)")

        if comprehensive_eval:
            censor_thresholds = generate_time_thresholds(
                cut_hours=72, step_hours=1, step_days=1
            )
            results, preds_df = temporal_eval.evaluate_over_time(
                censor_thresholds, save_predictions=True, model_name=model_name,
            )

            if not results:
                logger.error("No valid results from temporal evaluation!")
                return None, None

            os.makedirs(f'reports/eval/{model_name}/predictions', exist_ok=True)
            if preds_df is not None:
                preds_df.to_csv(
                    f'reports/eval/{model_name}/predictions/preds_df_{model_name}.csv', index=False
                )
            _save_time_metrics_csv(
                results, f'reports/eval/{model_name}/predictions/time_metrics_{model_name}.csv'
            )

            fig_time = plot_time_metrics(results, cut_hours=72)
            save_figure(fig_time, f"time_metrics_{model_name}",
                        save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)

            # Percentile recall plot
            percentiles = [5, 10, 15, 20, 25]
            recall_results = temporal_eval.evaluate_percentile_recall_over_time(
                censor_thresholds, percentiles=percentiles
            )
            if recall_results:
                fig_recall = plot_multi_percentile_recall(recall_results, percentiles)
                save_figure(fig_recall, f"multi_percentile_recall_{model_name}",
                            save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
                logger.info("Percentile recall plot saved")

            # Active-only evaluation and comparison
            results_active = None
            preds_df_active = None
            if active_only or trauma_scores or comprehensive_eval:
                logger.info("Running active-only temporal evaluation...")
                temporal_eval_active = TemporalEvaluator(
                    data, model, cfg, device=device, active_only=True
                )
                results_active, preds_df_active = temporal_eval_active.evaluate_over_time(
                    censor_thresholds, save_predictions=True, model_name=f"{model_name}_active",
                )
                if results_active:
                    if preds_df_active is not None:
                        preds_df_active.to_csv(
                            f'reports/eval/{model_name}/predictions/preds_df_{model_name}_active.csv', index=False
                        )
                    _save_time_metrics_csv(
                        results_active, f'reports/eval/{model_name}/predictions/time_metrics_{model_name}_active.csv'
                    )
                    fig_cmp = plot_time_metrics_comparison(
                        results, results_active, target_name=cfg["target"]
                    )
                    save_figure(fig_cmp, f"time_metrics_comparison_{model_name}",
                                save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
                    fig_n = plot_n_active_over_time(results_active, target_name=cfg["target"])
                    save_figure(fig_n, f"n_active_{model_name}",
                                save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
                    logger.info("Active-only comparison plots saved")

            # Prediction distribution plot (prefer active-only predictions)
            dist_preds = preds_df_active if preds_df_active is not None else preds_df
            if dist_preds is not None:
                fig_dist = plot_prediction_distribution(
                    dist_preds, np.array(data["ty"]),
                    data["holdout"].base.PID.values
                )
                save_figure(fig_dist, f"pred_distribution_{model_name}",
                            save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
                logger.info("Prediction distribution plot saved")

            # Trauma score comparison (temporal path)
            if trauma_scores:
                _run_trauma_score_comparison(
                    data, cfg, results, results_active, preds_df_active, model_name,
                    delong=delong,
                )

            logger.info("="*80)
            logger.info("TEMPORAL EVALUATION SUMMARY")
            logger.info("="*80)
            if key_timepoints:
                for step in key_timepoints:
                    matching = [r for r in results if r.censor_step == step]
                    if active_only and results_active:
                        matching_active = [r for r in results_active if r.censor_step == step]
                    else:
                        matching_active = []
                    if matching:
                        r = matching[0]
                        line = (f"  {format_step_label(step):>12s}: "
                                f"AUROC={r.auroc:.3f}, AUPRC={r.auprc:.3f}")
                        if r.cindex is not None:
                            line += f", C-index={r.cindex:.3f}"
                        if matching_active:
                            ra = matching_active[0]
                            line += (f"  |  Active: AUROC={ra.auroc:.3f}, "
                                     f"AUPRC={ra.auprc:.3f} (n={ra.n_samples})")
                        logger.info(line)
            return results, preds_df
        return None, None

    # ============================================================================
    # NON-TEMPORAL MODEL: existing evaluation path
    # ============================================================================
    logger.info("Running baseline evaluation with full time series...")
    preds, targs = _get_predictions(model, holdout_mixed_dls.train, device)

    evalplt = plot_evaluation(preds[:, 1], targs, cfg["target"])
    save_figure(evalplt, f"baseline_eval_{model_name}",
                save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
    logger.info("Baseline ROC/PR plot saved")

    # Decision Curve Analysis (baseline — full trajectory)
    fig_dca = plot_decision_curve(
        targs.numpy(), preds[:, 1].numpy(), model_name=model_name
    )
    save_figure(fig_dca, f"dca_baseline_{model_name}",
                save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
    logger.info("Baseline decision curve saved")

    # Confusion matrices at F-beta optimised thresholds (calibrated)
    logger.info("Computing calibrated F-beta thresholds...")
    tv_preds_raw, tv_targs_raw = _get_predictions(model, data["mixed_dls"].train, device)
    tv_y_pred = tv_preds_raw[:, 1].numpy()
    tv_y_true = tv_targs_raw.numpy()
    holdout_y_pred = preds[:, 1].numpy()
    holdout_y_true = targs.numpy()

    # Fit isotonic calibrator on trainval, apply to both
    from astra.evaluation.posthoc_calibration import fit_calibrators, apply_calibrator
    calibrators = fit_calibrators(tv_y_true, tv_y_pred, methods=['isotonic'])
    iso_cal = calibrators['isotonic']
    tv_y_cal = apply_calibrator(iso_cal, tv_y_pred, 'isotonic')
    ho_y_cal = apply_calibrator(iso_cal, holdout_y_pred, 'isotonic')
    logger.info("Isotonic calibrator fit on trainval, applied to holdout")

    for beta, label in [(1, "F1"), (5, "F5")]:
        thr, score = find_optimal_fbeta_threshold(tv_y_true, tv_y_cal, beta=beta)
        logger.info(f"  {label} optimal threshold={thr:.4f} (score={score:.4f}) on calibrated trainval")
        fig_cm, _, _ = evaluate_detection_rate(
            ho_y_cal, holdout_y_true, threshold=thr, label=label
        )
        save_figure(fig_cm, f"cm_{label}_{model_name}",
                    save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)

    # Initialize evaluator with pre-normalized data
    evaluator = TimeDependentEvaluator(data, model, cfg, device=device)

    # MULTIPLE ROC/PR CURVES AT KEY TIMEPOINTS
    if multicurve:
        logger.info("Creating multiple ROC/PR curves at key timepoints...")

        max_step = get_total_steps() - 2
        key_timepoints = sorted({
            min(t, max_step) for t in [
                time_to_step(1, 'h'), time_to_step(6, 'h'),
                time_to_step(12, 'h'), time_to_step(72, 'h'),
                time_to_step(7, 'D'), time_to_step(14, 'D'),
                time_to_step(30, 'D'), time_to_step(90, 'D'),
            ] if t is not None
        }, reverse=True)

        labels = [format_step_label(step) for step in key_timepoints]

        fig_curves = plot_multiple_roc_pr_curves(
            evaluator,
            key_timepoints,
            labels=labels
        )
        save_figure(fig_curves, f"multi_curves_{model_name}",
                    save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
        logger.info("Multiple curves plot saved")

        # Decision curves at key timepoints (active-only for correct prevalence)
        evaluator_dca = TimeDependentEvaluator(
            data, model, cfg, device=device, active_only=True
        )
        fig_dca_time = plot_decision_curves_over_time(
            evaluator_dca, key_timepoints, labels=labels
        )
        save_figure(fig_dca_time, f"dca_multicurve_{model_name}",
                    save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
        logger.info("Time-dependent decision curves saved")

    # COMPREHENSIVE TIME-DEPENDENT EVALUATION
    if comprehensive_eval:
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE TIME-DEPENDENT EVALUATION")
        logger.info("="*80)

        censor_thresholds = generate_time_thresholds(
            cut_hours=72,
            step_hours=1,
            step_days=1
        )
        logger.info(f"Generated {len(censor_thresholds)} time thresholds")
        logger.info(f"Range: {censor_thresholds[0]} to {censor_thresholds[-1]} steps")

        results, preds_df = evaluator.evaluate_over_time_ultra_fast(
            censor_thresholds,
            save_predictions=True,
            model_name=model_name
        )

        if not results:
            logger.error("No valid results from time-dependent evaluation!")
            return None, None

        logger.info(f"Evaluated at {len(results)} time points")

        os.makedirs(f'reports/eval/{model_name}/predictions', exist_ok=True)
        preds_df.to_csv(f'reports/eval/{model_name}/predictions/preds_df_{model_name}.csv', index=False)
        logger.info(f"Predictions saved to CSV")

        logger.info("Creating time-dependent metrics plot...")
        fig_time = plot_time_metrics(results, cut_hours=72)
        save_figure(fig_time, f"time_metrics_{model_name}",
                    save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
        logger.info("Time metrics plot saved")

        # Percentile recall plot
        percentiles = [5, 10, 15, 20, 25]
        recall_results = evaluator.evaluate_percentile_recall_over_time(
            censor_thresholds, percentiles=percentiles
        )
        if recall_results:
            fig_recall = plot_multi_percentile_recall(recall_results, percentiles)
            save_figure(fig_recall, f"multi_percentile_recall_{model_name}",
                        save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
            logger.info("Percentile recall plot saved")

        # Active-only evaluation and comparison
        results_active = None
        preds_df_active = None
        if active_only or trauma_scores:
            logger.info("Running active-only evaluation...")
            evaluator_active = TimeDependentEvaluator(
                data, model, cfg, device=device, active_only=True
            )
            results_active, preds_df_active = evaluator_active.evaluate_over_time_ultra_fast(
                censor_thresholds, save_predictions=True, model_name=f"{model_name}_active"
            )
            if results_active:
                if preds_df_active is not None:
                    preds_df_active.to_csv(
                        f'reports/eval/{model_name}/predictions/preds_df_{model_name}_active.csv', index=False
                    )
                fig_cmp = plot_time_metrics_comparison(
                    results, results_active, target_name=cfg["target"]
                )
                save_figure(fig_cmp, f"time_metrics_comparison_{model_name}",
                            save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
                fig_n = plot_n_active_over_time(results_active, target_name=cfg["target"])
                save_figure(fig_n, f"n_active_{model_name}",
                            save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
                logger.info("Active-only comparison plots saved")

        # Prediction distribution plot (active-only predictions)
        if preds_df_active is None and preds_df is not None:
            logger.info("Running active-only evaluation for distribution plot...")
            evaluator_active = TimeDependentEvaluator(
                data, model, cfg, device=device, active_only=True
            )
            _, preds_df_active = evaluator_active.evaluate_over_time_ultra_fast(
                censor_thresholds, save_predictions=True, model_name=f"{model_name}_active"
            )
        dist_preds = preds_df_active if preds_df_active is not None else preds_df
        if dist_preds is not None:
            fig_dist = plot_prediction_distribution(
                dist_preds, np.array(data["ty"]),
                data["holdout"].base.PID.values
            )
            save_figure(fig_dist, f"pred_distribution_{model_name}",
                        save_dir=f'reports/eval/{model_name}', **_SUBMISSION_KW)
            logger.info("Prediction distribution plot saved")

        # ================================================================
        # TRAUMA SCORE COMPARISON (optional, Azure-only)
        # ================================================================
        if trauma_scores:
            _run_trauma_score_comparison(
                data, cfg, results, results_active, preds_df_active, model_name,
                delong=delong,
            )

        logger.info("="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total time points evaluated: {len(results)}")
        logger.info(f"Predictions saved: {len(preds_df)} patient-timepoint pairs")

        logger.info("\nPerformance at key time points:")
        if multicurve:
            for step in key_timepoints[::-1]:
                matching = [r for r in results if r.censor_step == step]
                if active_only and results_active:
                    matching_active = [r for r in results_active if r.censor_step == step]
                else:
                    matching_active = []
                if matching:
                    r = matching[0]
                    line = (
                        f"  {format_step_label(step):>12s}: "
                        f"AUROC={r.auroc:.3f} [{r.auroc_ci[0]:.3f}-{r.auroc_ci[1]:.3f}], "
                        f"AUPRC={r.auprc:.3f} [{r.auprc_ci[0]:.3f}-{r.auprc_ci[1]:.3f}]"
                    )
                    if matching_active:
                        ra = matching_active[0]
                        line += (f"  |  Active: AUROC={ra.auroc:.3f}, "
                                 f"AUPRC={ra.auprc:.3f} (n={ra.n_samples})")
                    logger.info(line)

        logger.info("="*80)
        logger.info("Comprehensive evaluation complete!")
        logger.info("="*80)

        return results, preds_df

    else:
        logger.info("Skipping comprehensive evaluation (comprehensive_eval=False)")
        return None, None
