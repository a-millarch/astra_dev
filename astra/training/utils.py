"""
Training utilities: early stopping, metrics, checkpoint management.
"""

import logging
import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def hazards_to_survival(logits: torch.Tensor) -> torch.Tensor:
    """Convert discrete hazard logits to survival probabilities.

    Args:
        logits: [batch, seq_len] raw logits from temporal/survival head.

    Returns:
        S: [batch, seq_len] where S[i, t] = P(T > t | data).
           S[i, 0] = 1 - h(0), S[i, t] = prod_{k=0}^{t} (1 - h(k)).
    """
    hazards = torch.sigmoid(logits)
    log_survival = torch.cumsum(torch.log1p(-hazards + 1e-7), dim=1)
    return torch.exp(log_survival)


class EarlyStopping:
    """
    Early stopping with support for both min and max mode.

    Tracks two levels of "best":
    - **Global best**: the best score and model state across ALL phases.
      Never reset — this is what gets restored at the end of training.
    - **Phase-local best**: used for patience counting within a single phase.
      Reset between phases so each phase gets a fresh patience budget.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change to count as improvement.
        mode: 'min' for loss, 'max' for AUROC.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        # Phase-local tracking (reset between phases)
        self.best_score: Optional[float] = None
        # Global tracking (never reset — preserved across all phases)
        self.global_best_score: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.early_stop = False

    def _is_improvement(self, score: float, reference: float) -> bool:
        if self.mode == "max":
            return score > reference + self.min_delta
        return score < reference - self.min_delta

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        # Phase-local comparison for patience
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # Global best tracking — only save state when globally best
        if self.global_best_score is None or self._is_improvement(score, self.global_best_score):
            self.global_best_score = score
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
            # Also reset patience counter on global improvement
            self.counter = 0

        return self.early_stop

    def reset_patience(self) -> None:
        """
        Reset patience counter and phase-local comparison score for a new phase.

        Keeps global_best_score and best_state intact so the globally best
        model is preserved across all phases.
        """
        self.counter = 0
        self.best_score = None  # forces re-baseline on first epoch of new phase
        self.early_stop = False

    def restore_best(self, model: nn.Module) -> None:
        """Restore model to the best checkpoint seen across all phases."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored best model (global_best_score={self.global_best_score:.4f})")


@torch.no_grad()
def compute_auroc(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    temporal_head: bool = False,
) -> float:
    """
    Compute AUROC on a dataloader.

    Handles the TSAI mixed dataloader batch format: ((x_ts, x_tab, x_ts_cat), y).
    For temporal models, uses prediction at the last valid timestep per sample.
    """
    model.eval()
    all_probs = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        inputs, targets = batch
        inputs = _to_device(inputs, device)
        targets = _to_device(targets, device)

        # Unpack survival targets if present
        if isinstance(targets, (tuple, list)):
            y_binary = targets[0]
        else:
            y_binary = targets

        logits = model(inputs)

        if temporal_head:
            # logits: [batch, seq_len] — use prediction at last valid timestep
            x_ts = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            has_data = (x_ts.abs() > 1e-6).any(dim=1)  # [batch, seq_len]
            seq_len = x_ts.shape[2]
            positions = torch.arange(seq_len, device=x_ts.device).unsqueeze(0)
            masked_pos = torch.where(has_data, positions,
                                     torch.tensor(-1, device=x_ts.device))
            last_step = masked_pos.max(dim=1).values.clamp(min=0).long()
            logits_last = logits[torch.arange(logits.size(0), device=device), last_step]
            probs = torch.sigmoid(logits_last)
        else:
            probs = F.softmax(logits, dim=-1)[:, 1]  # probability of class 1

        all_probs.append(probs.cpu().numpy())
        all_targets.append(y_binary.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    try:
        return roc_auc_score(all_targets, all_probs)
    except ValueError:
        logger.warning("AUROC undefined (only one class in targets)")
        return 0.0


@torch.no_grad()
def _safe_auroc(targets: np.ndarray, probs: np.ndarray) -> float:
    """Compute AUROC, returning 0.0 if undefined (single class)."""
    try:
        return float(roc_auc_score(targets, probs))
    except ValueError:
        return 0.0


def _safe_auprc(targets: np.ndarray, probs: np.ndarray) -> float:
    """Compute AUPRC, returning 0.0 if undefined (single class)."""
    try:
        return float(average_precision_score(targets, probs))
    except ValueError:
        return 0.0


# Default timepoints (hours) for multi-timepoint active-only validation
# Covers early (6h, 24h), mid (72h, 7d), and late (14d, 30d) horizons
VAL_TIMEPOINTS_HOURS = [6, 24, 72, 168, 336, 720]


def _safe_cindex(event_times: np.ndarray, event_indicators: np.ndarray,
                  risk_scores: np.ndarray) -> float:
    """Compute Harrell's concordance index, returning 0.5 if undefined."""
    try:
        from lifelines.utils import concordance_index
        return float(concordance_index(event_times, -risk_scores, event_indicators))
    except Exception:
        try:
            # Fallback: manual implementation
            n = len(event_times)
            concordant = 0
            discordant = 0
            for i in range(n):
                if event_indicators[i] == 0:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if event_times[j] > event_times[i]:
                        if risk_scores[i] > risk_scores[j]:
                            concordant += 1
                        elif risk_scores[i] < risk_scores[j]:
                            discordant += 1
            total = concordant + discordant
            return concordant / total if total > 0 else 0.5
        except Exception:
            return 0.5


def compute_val_metrics(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    temporal_head: bool = False,
    survival_mode: bool = False,
) -> Dict[str, float]:
    """
    Compute validation metrics on a dataloader.

    For classification: AUROC and AUPRC.
    For survival: C-index (concordance) as primary metric, plus AUROC at timepoints.

    Returns dict with 'auroc' and 'auprc' keys (classification) or
    'auroc', 'auprc', 'cindex' keys (survival).
    """
    model.eval()

    if not temporal_head:
        # Standard head: unchanged behavior
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                inputs, targets = batch
                inputs = _to_device(inputs, device)
                targets = _to_device(targets, device)
                logits = model(inputs)
                probs = F.softmax(logits, dim=-1)[:, 1]
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        return {
            "auroc": _safe_auroc(all_targets, all_probs),
            "auprc": _safe_auprc(all_targets, all_probs),
        }

    # --- Temporal head: collect logits, targets, traj_lengths ---
    all_logits = []  # [batch, seq_len]
    all_targets_binary = []
    all_traj_lengths = []
    all_event_times = []
    all_event_indicators = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            inputs, targets = batch
            inputs = _to_device(inputs, device)
            targets = _to_device(targets, device)
            logits = model(inputs)  # [batch, seq_len]

            # Get trajectory lengths from batch (element 3) or infer
            if isinstance(inputs, (tuple, list)) and len(inputs) >= 4:
                traj_lens = inputs[3]
            else:
                x_ts = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
                has_data = (x_ts.abs() > 1e-6).any(dim=1)
                seq_len = x_ts.shape[2]
                positions = torch.arange(seq_len, device=x_ts.device).unsqueeze(0)
                masked_pos = torch.where(has_data, positions,
                                         torch.tensor(-1, device=x_ts.device))
                traj_lens = (masked_pos.max(dim=1).values + 1).clamp(min=1)

            all_logits.append(logits.cpu())
            all_traj_lengths.append(traj_lens.cpu().numpy())

            # Unpack survival targets if present
            if survival_mode and isinstance(targets, (tuple, list)):
                y_bin, ev_times, ev_inds = targets
                all_targets_binary.append(y_bin.cpu().numpy())
                all_event_times.append(ev_times.cpu().numpy())
                all_event_indicators.append(ev_inds.cpu().numpy())
            else:
                all_targets_binary.append(targets.cpu().numpy())

    all_logits = torch.cat(all_logits, dim=0)       # [N, seq_len]
    all_targets_binary = np.concatenate(all_targets_binary)  # [N]
    all_traj_lengths = np.concatenate(all_traj_lengths)  # [N]

    # --- Survival mode: compute C-index using cumulative incidence ---
    if survival_mode and all_event_times:
        all_event_times_arr = np.concatenate(all_event_times)
        all_event_indicators_arr = np.concatenate(all_event_indicators)

        # Compute survival probabilities and derive risk scores
        survival_probs = hazards_to_survival(all_logits)  # [N, seq_len]

        # Risk at last valid timestep per patient: 1 - S(traj_len - 1)
        last_steps = np.minimum(all_traj_lengths - 1, all_logits.shape[1] - 1).astype(int)
        surv_at_last = survival_probs[
            torch.arange(len(survival_probs)),
            torch.from_numpy(last_steps),
        ].numpy()
        risk_at_last = 1.0 - surv_at_last

        cindex = _safe_cindex(all_event_times_arr, all_event_indicators_arr, risk_at_last)

        # Also compute AUROC on binary labels for comparability
        last_auroc = _safe_auroc(all_targets_binary, risk_at_last)
        last_auprc = _safe_auprc(all_targets_binary, risk_at_last)

        logger.info(
            f"  Survival val: C-index={cindex:.4f}, "
            f"AUROC={last_auroc:.4f}, AUPRC={last_auprc:.4f}"
        )
        return {"auroc": last_auroc, "auprc": last_auprc, "cindex": cindex}

    # --- Classification temporal head: multi-timepoint active-only evaluation ---
    # Convert evaluation timepoints (hours) to step indices
    from astra.evaluation.utils import time_to_step
    eval_steps = [time_to_step(h, 'h') for h in VAL_TIMEPOINTS_HOURS]

    # Multi-timepoint active-only metrics
    tp_aurocs = []
    tp_auprcs = []
    for step in eval_steps:
        active_mask = all_traj_lengths > step
        n_active = active_mask.sum()
        if n_active < 10 or len(np.unique(all_targets_binary[active_mask])) < 2:
            continue  # skip timepoints with too few samples or single class
        probs_at_step = torch.sigmoid(all_logits[active_mask, step]).numpy()
        targets_at_step = all_targets_binary[active_mask]
        tp_aurocs.append(_safe_auroc(targets_at_step, probs_at_step))
        tp_auprcs.append(_safe_auprc(targets_at_step, probs_at_step))

    # Last-step metrics (for logging / backward compat)
    last_steps = np.minimum(all_traj_lengths - 1, all_logits.shape[1] - 1).astype(int)
    last_probs = torch.sigmoid(
        all_logits[torch.arange(len(all_logits)), torch.from_numpy(last_steps)]
    ).numpy()
    last_auroc = _safe_auroc(all_targets_binary, last_probs)
    last_auprc = _safe_auprc(all_targets_binary, last_probs)

    # Use multi-timepoint sum if available, else fall back to last-step.
    # Sum (not mean) so that each well-performing timepoint adds to the score,
    # incentivizing models that maintain performance across the full horizon.
    if tp_auprcs:
        auroc = float(np.sum(tp_aurocs))
        auprc = float(np.sum(tp_auprcs))
        logger.info(
            f"  Multi-timepoint val (active-only at {VAL_TIMEPOINTS_HOURS}h, "
            f"{len(tp_aurocs)} valid): "
            f"sum_AUROC={auroc:.4f}, sum_AUPRC={auprc:.4f} | "
            f"last-step: AUROC={last_auroc:.4f}, AUPRC={last_auprc:.4f}"
        )
    else:
        auroc = last_auroc
        auprc = last_auprc
        logger.info(f"  Val (last-step only): AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    return {"auroc": auroc, "auprc": auprc}


def _to_device(obj, device: str):
    """Recursively move tensors to device, ensuring plain torch.Tensor type."""
    if isinstance(obj, torch.Tensor):
        t = obj.to(device)
        if type(t) is not torch.Tensor:
            t = t.as_subclass(torch.Tensor)
        return t
    elif isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(item, device) for item in obj)
    return obj


class MetricTracker:
    """Track training/validation metrics across epochs and phases."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}

    def update(self, phase: str, epoch: int, **metrics) -> None:
        for key, value in metrics.items():
            full_key = f"{phase}/{key}"
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)

    def get(self, key: str) -> List[float]:
        return self.history.get(key, [])

    def best(self, metric_suffix: str, mode: str = "max") -> float:
        """Return the best value for a metric across all phases."""
        values = []
        for key, vals in self.history.items():
            if key.endswith(f"/{metric_suffix}") and vals:
                values.extend(vals)
        if not values:
            return 0.0
        return max(values) if mode == "max" else min(values)

    def summary(self) -> Dict[str, float]:
        """Return the last value for each tracked metric."""
        return {k: v[-1] for k, v in self.history.items() if v}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    score: float,
    path: str,
    **extra,
) -> None:
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
            **extra,
        },
        path,
    )


def save_model_checkpoint(
    model: nn.Module,
    model_name: str,
    save_dir: str = "models",
) -> None:
    """
    Save a pure-PyTorch model state dict.

    Saves as ``{'model': state_dict}`` to ``{save_dir}/{model_name}.pth``.
    """
    from astra.data.mixed_dataloader import save_model
    save_model(model, model_name, save_dir=save_dir)
