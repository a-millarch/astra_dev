# predictive_performance.py
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

from astra.utils import save_figure
from astra.data.dataloader import normalize_with_padding_mask
from astra.data.mixed_dataloader import (
    AstraMixedDataset,
    AstraMixedDataLoader,
)

from astra.evaluation.utils import (
    calculate_roc_auc_ci, calculate_average_precision_ci,
    _parse_timedelta_to_minutes, _get_intervals_from_cfg,
    time_to_step, step_to_time, prepare_model, get_max_days
)
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from astra.models.hybrid.training import get_backbone
from astra.visualize.evaluation import plot_evaluation

logger = logging.getLogger(__name__)


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

    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine y-range from model curve, then clip Treat All to that range
    ymin = min(nb_model.min(), -0.01) - 0.005
    ymax = max(nb_model.max(), y_true.mean()) * 1.15 + 0.005
    nb_treat_all_clipped = np.clip(nb_treat_all, ymin, None)

    ax.plot(thresholds, nb_model, color='#1F77B4', linewidth=2, label=model_name)
    ax.plot(thresholds, nb_treat_all_clipped, color='grey', linewidth=1.5, linestyle='--',
            label='Treat All')
    ax.axhline(y=0, color='black', linewidth=1, label='Treat None')

    ax.set_xlabel("Threshold Probability", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title("Decision Curve Analysis", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
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

    ax.set_xlabel("Threshold Probability", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title("Decision Curves at Different Time Points", fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9,
              title="Time Available", title_fontsize=10)
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

    ax.set_xlabel("Threshold Probability", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title("Decision Curves at Different Time Points", fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9,
              title="Time Available", title_fontsize=10)
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
            os.makedirs('reports/predictions', exist_ok=True)
            preds_df.to_pickle(f'reports/predictions/preds_{model_name}.pkl')
            logger.info(f"Saved predictions to reports/predictions/preds_{model_name}.pkl")
            return results, preds_df

        return results, None


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
                 device: str = 'cuda', active_only: bool = False):
        self.data = data
        self.model = model
        self.cfg = cfg
        self.device = device
        self.active_only = active_only
        self.model.eval()

        self._holdout_preds = None
        self._holdout_traj_lengths = np.array(
            data.get("holdout_trajectory_lengths",
                      data.get("traj_lengths_holdout", []))
        )
        self.holdout = data["holdout"]

        mode_str = " (active-only mode)" if active_only else ""
        logger.info(f"TemporalEvaluator initialized{mode_str}")

    def _get_all_predictions(self) -> np.ndarray:
        if self._holdout_preds is not None:
            return self._holdout_preds

        holdout_dls = self.data["holdout_mixed_dls"]
        all_preds = []

        with torch.no_grad():
            for batch in holdout_dls.train:
                inputs, targets = batch
                inputs = _to_device(inputs, self.device)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        self._holdout_preds = np.concatenate(all_preds, axis=0)
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
            os.makedirs('reports/predictions', exist_ok=True)
            preds_df.to_pickle(f'reports/predictions/preds_{model_name}.pkl')
            logger.info(f"Saved predictions to reports/predictions/preds_{model_name}.pkl")
            return results, preds_df

        return results, pd.DataFrame(preds_over_time) if preds_over_time else (results, None)


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours+1, 6))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("A) Performance over Hours", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
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

    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days+1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("B) Performance over Days", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()
    return fig


def plot_time_metrics_comparison(
    results_all: List[TimeMetricResult],
    results_active: List[TimeMetricResult],
    cut_hours=72, max_days=None
):
    """Overlay all-patients vs active-only AUROC/AUPRC curves."""
    if max_days is None:
        max_days = get_max_days()
    if not results_all or not results_active:
        raise ValueError("Both result sets required for comparison plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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
            # Hours panel — filter NaN metrics
            valid_h = mask_cut & ~np.isnan(vals)
            x = times_h[valid_h]
            v, lo, hi = vals[valid_h], lower[valid_h], upper[valid_h]
            if len(x) > 0:
                if x[-1] < cut_hours:
                    x = np.append(x, cut_hours)
                    v = np.append(v, v[-1])
                    lo = np.append(lo, lo[-1])
                    hi = np.append(hi, hi[-1])
                ax1.plot(x, v, color=color, linestyle=linestyle,
                         label=f"{metric_name} ({label_prefix})", markersize=3)
                ax1.fill_between(x, lo, hi, color=color, alpha=0.1)

            # Days panel — filter NaN metrics
            valid_d = ~np.isnan(vals)
            x = times_d[valid_d]
            v, lo, hi = vals[valid_d], lower[valid_d], upper[valid_d]
            if len(x) > 0:
                if x[-1] < max_days:
                    x = np.append(x, max_days)
                    v = np.append(v, v[-1])
                    lo = np.append(lo, lo[-1])
                    hi = np.append(hi, hi[-1])
                ax2.plot(x, v, color=color, linestyle=linestyle,
                         label=f"{metric_name} ({label_prefix})", markersize=3)
                ax2.fill_between(x, lo, hi, color=color, alpha=0.1)

    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours+1, 6))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("A) Performance over Hours", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.0)

    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days+1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("B) Performance over Days", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.0)

    # Shared legend below both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.18)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    mask_cut = times_h <= cut_hours

    # Hours panel
    ax1.plot(times_h[mask_cut], n_samples[mask_cut], color=ACTIVE_COLOR, label="Active patients")
    ax1.plot(times_h[mask_cut], n_positive[mask_cut], color=POSITIVE_COLOR, label=f"{target_name} = 1 (active)")
    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_xlim(0, cut_hours)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("A) Active Patients over Hours", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax1_prev = ax1.twinx()
    ax1_prev.plot(times_h[mask_cut], prevalence[mask_cut] * 100, color=PREV_COLOR,
                  linestyle="--", linewidth=1.5, label="Prevalence (%)")
    ax1_prev.set_ylabel("Prevalence (%)", fontsize=10, color=PREV_COLOR)
    ax1_prev.set_ylim(0, 12)
    ax1_prev.tick_params(axis='y', labelcolor=PREV_COLOR)

    # Days panel
    ax2.plot(times_d, n_samples, color=ACTIVE_COLOR, label="Active patients")
    ax2.plot(times_d, n_positive, color=POSITIVE_COLOR, label=f"{target_name} = 1 (active)")
    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_xlim(0, max_days)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("B) Active Patients over Days", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax2_prev = ax2.twinx()
    ax2_prev.plot(times_d, prevalence * 100, color=PREV_COLOR,
                  linestyle="--", linewidth=1.5, label="Prevalence (%)")
    ax2_prev.set_ylabel("Prevalence (%)", fontsize=10, color=PREV_COLOR)
    ax2_prev.set_ylim(0, 12)
    ax2_prev.tick_params(axis='y', labelcolor=PREV_COLOR)

    # Combined legend below
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_prev.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.18)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def plot_multiple_roc_pr_curves(
    evaluator: TimeDependentEvaluator,
    censor_steps: List[int],
    labels: Optional[List[str]] = None
):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
              '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']

    baseline = None

    for i, censor_step in enumerate(censor_steps):
        dls = evaluator.create_censored_dataloaders_fast(censor_step)
        if dls is None:
            logger.warning(f"Skipping step {censor_step}: dataloader creation failed")
            continue

        preds, targets = _get_predictions(evaluator.model, dls.train, evaluator.device)
        y_preds = preds[:, 1].numpy()
        ys = targets.numpy()

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
    ax_roc.set_title("ROC Curves at Different Time Points", fontsize=13, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(fontsize=9, title="Time Available", title_fontsize=10)
    ax_roc.set_aspect('equal', adjustable='box')

    if baseline is not None:
        ax_pr.axhline(y=baseline, color='grey', linestyle='--', lw=1.5, alpha=0.7, label=f'Baseline ({baseline:.3f})')
    ax_pr.set_title("Precision-Recall Curves at Different Time Points", fontsize=13, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9,
                title="Time Available", title_fontsize=10)
    ax_pr.set_aspect('equal', adjustable='box')

    fig.subplots_adjust(right=0.82, wspace=0.3)
    plt.tight_layout()

    return fig


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def run_eval(data, cfg: dict, multicurve: bool = True, comprehensive_eval: bool = True,
             active_only: bool = False):
    """
    Enhanced evaluation with time-dependent metrics.

    Uses direct model inference (no FastAI Learner).

    Args:
        active_only: If True, also runs active-only evaluation and generates
                     comparison plots (all patients vs active-only).
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
        save_figure(evalplt, f"baseline_eval_{model_name}", save_dir='reports/eval')
        logger.info("Baseline temporal evaluation saved")

        # Decision Curve Analysis (baseline — full trajectory)
        fig_dca = plot_decision_curve(targs, baseline_preds, model_name=model_name)
        save_figure(fig_dca, f"dca_baseline_{model_name}", save_dir='reports/eval')
        plt.close(fig_dca)
        logger.info("Baseline decision curve saved")

        key_timepoints = None
        if multicurve:
            key_timepoints = [
                time_to_step(1, 'h'), time_to_step(6, 'h'),
                time_to_step(12, 'h'), time_to_step(72, 'h'),
                time_to_step(7, 'D'), time_to_step(13, 'D'),
                time_to_step(30, 'D'), time_to_step(90, 'D'),
            ]
            key_timepoints = [t for t in key_timepoints if t is not None]

            logger.info("Performance at key time points (temporal model):")
            for step in key_timepoints:
                result = temporal_eval.evaluate_at_timestep(step)
                if result:
                    logger.info(
                        f"  {format_step_label(step):>12s}: "
                        f"AUROC={result.auroc:.3f} [{result.auroc_ci[0]:.3f}-{result.auroc_ci[1]:.3f}], "
                        f"AUPRC={result.auprc:.3f} [{result.auprc_ci[0]:.3f}-{result.auprc_ci[1]:.3f}]"
                    )

            # Decision curves at key timepoints (temporal)
            labels = [format_step_label(step) for step in key_timepoints]
            fig_dca_time = _plot_decision_curves_temporal(
                preds_all, targs, traj_lens, key_timepoints, labels=labels
            )
            save_figure(fig_dca_time, f"dca_multicurve_{model_name}", save_dir='reports/eval')
            plt.close(fig_dca_time)
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

            os.makedirs('reports/predictions', exist_ok=True)
            if preds_df is not None:
                preds_df.to_csv(
                    f'reports/predictions/preds_df_{model_name}.csv', index=False
                )

            fig_time = plot_time_metrics(results, cut_hours=72)
            save_figure(fig_time, f"time_metrics_{model_name}", save_dir='reports/eval')

            # Active-only evaluation and comparison
            if active_only:
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
                            f'reports/predictions/preds_df_{model_name}_active.csv', index=False
                        )
                    fig_cmp = plot_time_metrics_comparison(results, results_active)
                    save_figure(fig_cmp, f"time_metrics_comparison_{model_name}", save_dir='reports/eval')
                    fig_n = plot_n_active_over_time(results_active, target_name=cfg["target"])
                    save_figure(fig_n, f"n_active_{model_name}", save_dir='reports/eval')
                    logger.info("Active-only comparison plots saved")

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
    save_figure(evalplt, f"baseline_eval_{model_name}", save_dir='reports/eval')
    logger.info("Baseline ROC/PR plot saved")

    # Decision Curve Analysis (baseline — full trajectory)
    fig_dca = plot_decision_curve(
        targs.numpy(), preds[:, 1].numpy(), model_name=model_name
    )
    save_figure(fig_dca, f"dca_baseline_{model_name}", save_dir='reports/eval')
    plt.close(fig_dca)
    logger.info("Baseline decision curve saved")

    # Initialize evaluator with pre-normalized data
    evaluator = TimeDependentEvaluator(data, model, cfg, device=device)

    # MULTIPLE ROC/PR CURVES AT KEY TIMEPOINTS
    if multicurve:
        logger.info("Creating multiple ROC/PR curves at key timepoints...")

        key_timepoints = [
            time_to_step(1, 'h'),
            time_to_step(6, 'h'),
            time_to_step(12, 'h'),
            time_to_step(72, 'h'),
            time_to_step(7, 'D'),
            time_to_step(13, 'D'),
            time_to_step(30, 'D'),
            time_to_step(90, 'D'),
        ]

        key_timepoints = [t for t in key_timepoints if t is not None]
        key_timepoints.reverse()

        labels = [format_step_label(step) for step in key_timepoints]

        fig_curves = plot_multiple_roc_pr_curves(
            evaluator,
            key_timepoints,
            labels=labels
        )
        save_figure(fig_curves, f"multi_curves_{model_name}", save_dir='reports/eval')
        logger.info("Multiple curves plot saved")

        # Decision curves at key timepoints
        fig_dca_time = plot_decision_curves_over_time(
            evaluator, key_timepoints, labels=labels
        )
        save_figure(fig_dca_time, f"dca_multicurve_{model_name}", save_dir='reports/eval')
        plt.close(fig_dca_time)
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

        os.makedirs('reports/predictions', exist_ok=True)
        preds_df.to_csv(f'reports/predictions/preds_df_{model_name}.csv', index=False)
        logger.info(f"Predictions saved to CSV")

        logger.info("Creating time-dependent metrics plot...")
        fig_time = plot_time_metrics(results, cut_hours=72)
        save_figure(fig_time, f"time_metrics_{model_name}", save_dir='reports/eval')
        logger.info("Time metrics plot saved")

        # Active-only evaluation and comparison
        results_active = None
        if active_only:
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
                        f'reports/predictions/preds_df_{model_name}_active.csv', index=False
                    )
                fig_cmp = plot_time_metrics_comparison(results, results_active)
                save_figure(fig_cmp, f"time_metrics_comparison_{model_name}", save_dir='reports/eval')
                fig_n = plot_n_active_over_time(results_active, target_name=cfg["target"])
                save_figure(fig_n, f"n_active_{model_name}", save_dir='reports/eval')
                logger.info("Active-only comparison plots saved")

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
