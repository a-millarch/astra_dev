"""
Training utilities: early stopping, metrics, checkpoint management.
"""

import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from astra.utils import logger


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
) -> float:
    """
    Compute AUROC on a dataloader.

    Handles the TSAI mixed dataloader batch format: ((x_ts, x_tab, x_ts_cat), y).
    """
    model.eval()
    all_probs = []
    all_targets = []

    for batch in dataloader:
        inputs, targets = batch
        # Move inputs to device (nested tuple)
        inputs = _to_device(inputs, device)
        targets = targets.to(device)

        logits = model(inputs)
        probs = F.softmax(logits, dim=-1)[:, 1]  # probability of class 1

        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    try:
        return roc_auc_score(all_targets, all_probs)
    except ValueError:
        # Only one class present in targets
        logger.warning("AUROC undefined (only one class in targets)")
        return 0.0


def _to_device(obj, device: str):
    """Recursively move tensors in nested tuples/lists to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
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


def save_model_fastai_compatible(
    model: nn.Module,
    data: dict,
    model_name: str,
    cfg: dict,
) -> None:
    """
    Save a pure-PyTorch model in FastAI-compatible format so run_eval() works.

    Creates a temporary Learner, attaches the model state dict, and uses
    learn.save() which writes to the models/ directory.
    """
    from fastai.learner import Learner as FastAILearner
    from astra.models.hybrid.training import get_backbone

    # Build a fresh backbone with identical architecture
    backbone = get_backbone(data, cfg)
    backbone.load_state_dict(model.state_dict())

    mixed_dls = data["mixed_dls"]
    learn = FastAILearner(mixed_dls, backbone, metrics=None)
    learn.save(model_name)
    logger.info(f"Model saved in FastAI format: models/{model_name}.pth")
