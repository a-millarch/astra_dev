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
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping with support for both min and max mode.

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
        self.best_score: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.early_stop = False

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
            return False

        improved = False
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset_patience(self) -> None:
        """
        Reset patience counter and phase-local comparison score for a new phase.

        Keeps best_state intact so the globally best model is preserved across
        all phases. Only resets the patience mechanism so the new phase gets a
        fresh budget of epochs before early stopping triggers.
        """
        self.counter = 0
        self.best_score = None  # forces re-baseline on first epoch of new phase
        self.early_stop = False

    def restore_best(self, model: nn.Module) -> None:
        """Restore model to the best checkpoint seen so far."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored best model (score={self.best_score:.4f})")


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
        all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    try:
        return roc_auc_score(all_targets, all_probs)
    except ValueError:
        logger.warning("AUROC undefined (only one class in targets)")
        return 0.0


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
