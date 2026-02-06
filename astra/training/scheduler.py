"""
Learning rate schedulers for pure-PyTorch training.

Supports per-parameter-group learning rates via LambdaLR (each group's
base_lr is multiplied by the lambda, preserving discriminative ratios).
"""

import math
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def get_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Cosine annealing with linear warmup. Compatible with per-group LRs.

    Args:
        optimizer: Optimizer with (possibly multiple) param groups.
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of initial LR (0.0 = decay to zero).

    Returns:
        LambdaLR scheduler that preserves per-group LR ratios.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def get_one_cycle_scheduler(
    optimizer: Optimizer,
    max_lrs: List[float],
    total_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> OneCycleLR:
    """
    PyTorch-native one-cycle schedule (replaces FastAI's fit_one_cycle).

    Args:
        optimizer: Optimizer with param groups.
        max_lrs: List of max learning rates, one per param group.
        total_steps: Total number of training steps.
        pct_start: Fraction of steps spent increasing LR.
        div_factor: Initial LR = max_lr / div_factor.
        final_div_factor: Final LR = max_lr / (div_factor * final_div_factor).

    Returns:
        OneCycleLR scheduler.
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
