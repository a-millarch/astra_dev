"""
Pure-PyTorch finetuning with 4-phase transfer learning.

Phase 1: Head-only (warm up randomly initialized classification head)
Phase 2: Partial unfreeze (upper transformer layers + head)
Phase 3: Full finetune (all layers with discriminative LRs)
Phase 4: Early prediction hardening (optional progressive time masking)
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from astra.utils import cfg, clear_mem
from astra.models.hybrid.model import TSTabFusionTransformerMultiHot
from astra.models.hybrid.mlm import TSTabFusionMLM, MLMConfig
from astra.models.hybrid.training import get_backbone
from astra.data.mixed_dataloader import (
    AstraMixedDataset,
    AstraMixedDataLoader,
    get_stratified_splits,
    save_model,
)

from astra.training.param_groups import (
    get_layer_groups,
    get_optimizer_param_groups,
    freeze_to,
    unfreeze_from,
    unfreeze_all,
    set_dropout_rates,
)
from astra.training.scheduler import get_cosine_warmup_scheduler
from astra.training.utils import (
    EarlyStopping,
    MetricTracker,
    compute_auroc,
    compute_val_metrics,
    _to_device,
)
from astra.data.dataloader import save_deployment_bundle

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """All finetuning hyperparameters."""

    # Phase 1: Head-only
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3

    # Phase 2: Partial unfreeze (upper transformer layers + head)
    phase2_epochs: int = 12
    phase2_lr: float = 3e-4
    phase2_unfreeze_from: str = "transformer_4_5"

    # Phase 3: Full finetune
    phase3_epochs: int = 8
    phase3_lr: float = 1e-4

    # Phase 4: Early prediction hardening (optional)
    enable_early_prediction: bool = False
    phase4_epochs: int = 8
    phase4_lr: float = 5e-5
    masking_prob: float = 0.5
    early_weight: float = 2.0
    min_timesteps: int = 2

    # Discriminative LR
    lr_decay_factor: float = 0.1

    # Optimization
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    grad_clip: float = 1.0

    # Regularization
    label_smoothing: float = 0.1

    # Dropout overrides (None = use model defaults from config)
    fc_dropout: Optional[float] = None
    res_dropout: Optional[float] = None

    # Validation / early stopping
    patience: int = 7
    valid_size: float = 0.2

    # Checkpointing
    save_dir: str = "./models"
    model_name: str = ""

    # Use pretrained weights
    use_pretrained: bool = True
    pretrain_checkpoint_dir: Optional[str] = None

    # Class imbalance weighting for standard (non-temporal) head
    # Multiplier applied to computed n_neg/n_pos ratio. 1.0 = full correction,
    # 0.0 = no correction. Values around 0.5-0.8 often work best.
    pos_weight_factor: float = 0.7

    # Time weighting for temporal head (training-specific, not model arch)
    time_weighting: str = "uniform"     # 'uniform', 'early', or 'late'
    early_weight_factor: float = 2.0

    # Validation objective weights (must sum to 1.0)
    val_auroc_weight: float = 0.3
    val_auprc_weight: float = 0.7


def create_split_dataloaders(data: dict, splits, cfg_dict: dict):
    """
    Create train/valid mixed dataloaders from existing data arrays with split indices.

    Args:
        data: Output from prepare_data_and_dls().
        splits: Tuple of (train_indices, valid_indices).
        cfg_dict: Global config dict.

    Returns:
        AstraMixedDataLoader with train and valid splits.
    """
    # Get pre-encoded tabular arrays from the existing dataset
    trainval_ds = data["mixed_dls"]._train_ds
    if hasattr(trainval_ds, 'dataset'):
        trainval_ds = trainval_ds.dataset

    dataset = AstraMixedDataset(
        X_ts=trainval_ds.X_ts.numpy(),
        x_cat=trainval_ds.x_cat.numpy(),
        x_cont=trainval_ds.x_cont.numpy(),
        X_ts_cat=trainval_ds.X_ts_cat.numpy(),
        y=trainval_ds.y.numpy(),
        trajectory_lengths=trainval_ds.traj_lengths.numpy(),
    )

    return AstraMixedDataLoader(
        dataset,
        splits=splits,
        bs=cfg_dict["training"]["bs"],
        shuffle_train=True,
    )


def load_pretrained_backbone(
    data: dict,
    cfg_dict: dict,
    pretrain_cfg: Optional[MLMConfig] = None,
    checkpoint_dir: Optional[str] = None,
) -> nn.Module:
    """
    Create a backbone and load pretrained weights from MLM checkpoint.

    Follows the same pattern as run_finetune() in training.py:288-297.

    When EBM feature is enabled, the current backbone has c_in+1 channels
    while the pretrained checkpoint has c_in. Handles this by loading all
    weights except W_P, then expanding W_P with Xavier-initialized weights
    for the new EBM channel.

    When temporal_head=True, the checkpoint won't have temporal_pred_head
    weights (pretraining doesn't use classification head). We load with
    strict=False so the new head gets random initialization.
    """
    model_cfg = cfg_dict.get("model", {})
    temporal_head = model_cfg.get("temporal_head", False)
    causal = model_cfg.get("causal", False)
    temporal_head_dropout = model_cfg.get("temporal_head_dropout", 0.3)

    backbone = get_backbone(
        data, cfg_dict,
        temporal_head=temporal_head,
        causal=causal,
        temporal_head_dropout=temporal_head_dropout,
        temporal_channel_idx=data.get('temporal_channel_idx'),
        exclude_channel_indices=data.get('exclude_channel_indices', []),
    )

    if checkpoint_dir is None:
        checkpoint_dir = f'./pretrain_checkpoints/{cfg_dict["model_name"]}'

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Need the same MLM model structure to load, then extract backbone
    if pretrain_cfg is None:
        pc = cfg_dict["pretrain"]
        pretrain_cfg = MLMConfig(
            mask_prob_ts=pc["mask_prob_ts"],
            mask_prob_cat_ts=pc["mask_prob_cat_ts"],
            mask_prob_cat=pc["mask_prob_cat"],
            mask_prob_cont=pc["mask_prob_cont"],
        )

    # Check for c_in mismatch (EBM feature or temporal features add channels)
    full_c_in = backbone.W_P.in_channels
    pretrain_W_P_weight = checkpoint["model_state_dict"]["backbone.W_P.weight"]
    pretrain_c_in = pretrain_W_P_weight.shape[1]

    if pretrain_c_in != full_c_in:
        logger.info(
            f"W_P channel mismatch: checkpoint has {pretrain_c_in}, "
            f"current model has {full_c_in} — expanding W_P"
        )
        # Identify which channel indices are NEW (not in the pretrained model).
        # df2xy sorts channels by FEATURE name alphabetically — same order as W_P.
        features_sorted = sorted(data["trainval"].complete["FEATURE"].unique())
        temporal_names = set(
            cfg_dict.get("temporal_features", {}).get("features", [])
        )
        new_channel_indices = sorted(
            {i for i, f in enumerate(features_sorted)
             if f in temporal_names or f == "_ebm_pred"}
        )
        if not new_channel_indices:
            # Fallback: treat the last N channels as new (best-effort)
            n_new = full_c_in - pretrain_c_in
            new_channel_indices = list(range(full_c_in - n_new, full_c_in))
            logger.warning(
                f"Could not identify new channel names; Xavier-initializing "
                f"last {n_new} channel(s): {new_channel_indices}"
            )
        logger.info(f"New channel indices to Xavier-init: {new_channel_indices} "
                    f"({[features_sorted[i] for i in new_channel_indices]})")

        # Load only backbone weights (except W_P) via MLM wrapper.
        # MLM heads (ts_head, cat_heads, etc.) may also have c_in-dependent
        # shapes, so we skip all non-backbone keys.
        mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
        filtered_state = {
            k: v for k, v in checkpoint["model_state_dict"].items()
            if k.startswith("backbone.") and "W_P" not in k
        }
        mlm_model.load_state_dict(filtered_state, strict=False)
        backbone = mlm_model.backbone

        # Expand W_P: copy pretrained weights, Xavier-init new channel(s)
        _expand_w_p(backbone, pretrain_W_P_weight,
                     checkpoint["model_state_dict"]["backbone.W_P.bias"],
                     new_channel_indices)
        logger.info(f"Pretrained weights loaded with W_P expansion from {checkpoint_path}")
    else:
        mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
        # Use strict=False when temporal head or causal masking is enabled —
        # the checkpoint won't have temporal_pred_head or causal_mask
        needs_strict_false = temporal_head or causal
        mlm_model.load_state_dict(checkpoint["model_state_dict"],
                                  strict=not needs_strict_false)
        backbone = mlm_model.backbone
        logger.info(f"Pretrained weights loaded from {checkpoint_path}")

    return backbone


def _expand_w_p(
    backbone: nn.Module,
    old_weight: torch.Tensor,
    old_bias: torch.Tensor,
    new_channel_indices: Optional[list] = None,
):
    """
    Expand W_P Conv1d to accommodate additional input channel(s).

    Copies pretrained weights for existing channels to their correct
    positions and Xavier-initializes any new channels (e.g. EBM,
    elapsed_hours, bin_width_hours).

    W_P is nn.Conv1d(c_in, continuous_dim, kernel_size=1):
        weight shape: [out_channels, in_channels, 1]
        bias shape: [out_channels]

    Args:
        new_channel_indices: Sorted list of indices (into the new W_P) that
            correspond to newly added channels and should be Xavier-initialized.
            All other indices are filled from the pretrained weights in order.
    """
    new_c_in = backbone.W_P.in_channels
    old_c_in = old_weight.shape[1]
    new_set = set(new_channel_indices or [])

    with torch.no_grad():
        backbone.W_P.bias.data.copy_(old_bias)

        old_idx = 0
        for new_idx in range(new_c_in):
            if new_idx in new_set:
                nn.init.xavier_uniform_(
                    backbone.W_P.weight.data[:, new_idx : new_idx + 1, :]
                )
            else:
                backbone.W_P.weight.data[:, new_idx : new_idx + 1, :] = (
                    old_weight[:, old_idx : old_idx + 1, :]
                )
                old_idx += 1

    logger.info(
        f"W_P expanded: {old_c_in} → {new_c_in} channels "
        f"(Xavier-initialized at indices {sorted(new_set)})"
    )


def _apply_progressive_time_masking(
    x_ts: torch.Tensor,
    min_timesteps: int = 2,
    max_timesteps: Optional[int] = None,
) -> torch.Tensor:
    """
    Randomly truncate time series by zeroing out future timesteps.

    Ported from ProgressiveTimeMaskingCallback in callbacks.py.

    Args:
        x_ts: Continuous TS tensor [batch, c_in, seq_len] (TSAI format).
        min_timesteps: Minimum timesteps to keep.
        max_timesteps: Maximum timesteps (None = use full sequence length).

    Returns:
        Masked tensor with same shape.
    """
    # Detect shape: model expects [batch, c_in, seq_len] from TSAI
    # but callbacks used [batch, seq_len, features]
    # The TSAI mixed_dls outputs [batch, c_in, seq_len]
    batch_size, c_in, seq_len = x_ts.shape
    if max_timesteps is None:
        max_timesteps = seq_len

    cutoffs = torch.randint(
        min_timesteps,
        min(seq_len, max_timesteps) + 1,
        (batch_size,),
        device=x_ts.device,
    )

    # Mask: [batch, 1, seq_len] — broadcast over channels
    timestep_indices = torch.arange(seq_len, device=x_ts.device).expand(batch_size, -1)
    mask = (timestep_indices < cutoffs.unsqueeze(1)).unsqueeze(1).float()

    return x_ts * mask


def _compute_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    x_ts: torch.Tensor,
    label_smoothing: float = 0.0,
    early_weight: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute loss weighted by data availability.

    Ported from WeightedLossCallback in callbacks.py.
    Samples with less data (sparser time series) get higher weight.
    """
    loss_per_sample = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing,
        weight=class_weights,
    )

    # Data availability: fraction of non-zero timesteps
    # x_ts shape: [batch, c_in, seq_len]
    data_present = (x_ts.abs().sum(dim=1) > 1e-6).float()  # [batch, seq_len]
    availability_ratio = data_present.sum(dim=1) / x_ts.shape[2]  # [batch]

    # Weight: high for sparse data, low for dense data
    weights = early_weight - (early_weight - 1.0) * availability_ratio

    return (loss_per_sample * weights).mean()


def _infer_trajectory_lengths_from_batch(x_ts: torch.Tensor) -> torch.Tensor:
    """
    Infer trajectory lengths from a batch of time series data.

    A timestep is padding if ALL channels are zero (or near-zero).
    After normalization, measured values are ~N(0,1) and padding/missing = 0.0.

    Args:
        x_ts: [batch, c_in, seq_len] -- already normalized

    Returns:
        traj_lengths: [batch] -- last non-padding timestep index + 1
    """
    has_data = (x_ts.abs() > 1e-6).any(dim=1)  # [batch, seq_len]
    seq_len = x_ts.shape[2]
    positions = torch.arange(seq_len, device=x_ts.device).unsqueeze(0)  # [1, seq_len]
    masked_positions = torch.where(has_data, positions, torch.tensor(-1, device=x_ts.device))
    traj_lengths = masked_positions.max(dim=1).values + 1  # [batch]
    return traj_lengths.clamp(min=1)


def compute_temporal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    traj_lengths: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    time_weighting: str = "uniform",
    early_weight_factor: float = 2.0,
) -> torch.Tensor:
    """
    Per-timestep BCE loss with padding mask and optional time weighting.

    Args:
        logits: [batch, seq_len] -- raw logits from temporal head
        targets: [batch] -- binary labels (0/1)
        traj_lengths: [batch] -- number of valid timesteps per sample
        pos_weight: Scalar tensor for class imbalance (ratio of neg/pos)
        time_weighting: 'uniform', 'early', or 'late'
        early_weight_factor: Maximum weight factor (early: applied to first steps,
            late: applied to last steps)

    Returns:
        Scalar loss
    """
    batch_size, seq_len = logits.shape
    device = logits.device

    # Expand targets to [batch, seq_len] — same label at every timestep
    targets_expanded = targets.float().unsqueeze(1).expand_as(logits)

    # Padding mask: True = valid position
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    valid_mask = positions < traj_lengths.unsqueeze(1)  # [batch, seq_len]

    # Per-element BCE loss
    loss_per_element = F.binary_cross_entropy_with_logits(
        logits, targets_expanded,
        pos_weight=pos_weight,
        reduction="none",
    )  # [batch, seq_len]

    # Apply padding mask
    loss_per_element = loss_per_element * valid_mask.float()

    # Optional time weighting
    if time_weighting == "early":
        time_weights = torch.linspace(
            early_weight_factor, 1.0, seq_len, device=device
        )
        loss_per_element = loss_per_element * time_weights.unsqueeze(0)
    elif time_weighting == "late":
        time_weights = torch.linspace(
            1.0, early_weight_factor, seq_len, device=device
        )
        loss_per_element = loss_per_element * time_weights.unsqueeze(0)

    # Average over valid positions
    n_valid = valid_mask.float().sum().clamp(min=1.0)
    return loss_per_element.sum() / n_valid


def compute_survival_loss(
    logits: torch.Tensor,
    event_times: torch.Tensor,
    event_indicators: torch.Tensor,
    traj_lengths: torch.Tensor,
    time_weighting: str = "uniform",
    early_weight_factor: float = 2.0,
) -> torch.Tensor:
    """
    Discrete-time survival negative log-likelihood (Nnet-survival).

    Each timestep k contributes a binary cross-entropy term:
      - k < event_time: target = 0 (survived this interval)
      - k == event_time AND event_indicator == 1: target = 1 (event here)
      - k > event_time OR k >= traj_length: masked out

    Args:
        logits: [batch, seq_len] hazard logits from temporal head.
        event_times: [batch] discrete timestep of event or censoring.
        event_indicators: [batch] 1 = event observed, 0 = censored.
        traj_lengths: [batch] valid timesteps (padding boundary).
        time_weighting: 'uniform', 'early', or 'late'.
        early_weight_factor: max weight factor for time weighting.

    Returns:
        Scalar loss.
    """
    batch_size, seq_len = logits.shape
    device = logits.device

    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    event_times_col = event_times.unsqueeze(1)  # [batch, 1]
    event_ind_col = event_indicators.unsqueeze(1).float()  # [batch, 1]

    # Per-timestep targets: 1 only at event timestep for patients with events
    targets = torch.zeros_like(logits)
    is_event_step = (positions == event_times_col) & (event_ind_col == 1.0)
    targets[is_event_step] = 1.0

    # Valid mask: positions <= event_time for events, < event_time for censored
    # Also respect padding boundary
    max_valid_step = torch.where(
        event_indicators.bool(),
        event_times + 1,  # include event timestep
        event_times,       # exclude censoring timestep (survived up to but not including)
    ).unsqueeze(1)  # [batch, 1]
    valid_mask = (positions < max_valid_step) & (positions < traj_lengths.unsqueeze(1))

    # Per-element BCE
    loss_per_element = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )
    loss_per_element = loss_per_element * valid_mask.float()

    # Optional time weighting
    if time_weighting == "early":
        time_weights = torch.linspace(
            early_weight_factor, 1.0, seq_len, device=device
        )
        loss_per_element = loss_per_element * time_weights.unsqueeze(0)
    elif time_weighting == "late":
        time_weights = torch.linspace(
            1.0, early_weight_factor, seq_len, device=device
        )
        loss_per_element = loss_per_element * time_weights.unsqueeze(0)

    # Average over valid positions
    n_valid = valid_mask.float().sum().clamp(min=1.0)
    return loss_per_element.sum() / n_valid


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
    grad_clip: float = 1.0,
    label_smoothing: float = 0.1,
    # Phase 4 options
    enable_masking: bool = False,
    masking_prob: float = 0.5,
    min_timesteps: int = 2,
    enable_weighting: bool = False,
    early_weight: float = 2.0,
    # Per-timestep prediction options
    temporal_head: bool = False,
    pos_weight: Optional[torch.Tensor] = None,
    time_weighting: str = "uniform",
    early_weight_factor: float = 2.0,
    # Class weighting for standard (non-temporal) head
    class_weights: Optional[torch.Tensor] = None,
    # Survival mode
    survival_mode: bool = False,
    desc: str = "Training",
) -> float:
    """
    Single epoch training loop for the TSAI mixed dataloader format.

    Args:
        model: The backbone model.
        dataloader: TSAI mixed dataloader (train split).
        optimizer: Optimizer with param groups.
        scheduler: LR scheduler (stepped per batch).
        device: Device.
        grad_clip: Max gradient norm.
        label_smoothing: Label smoothing for cross-entropy.
        enable_masking: Whether to apply progressive time masking.
        masking_prob: Probability of masking a batch.
        min_timesteps: Minimum timesteps to keep when masking.
        enable_weighting: Whether to weight loss by data availability.
        early_weight: Weight multiplier for sparse-data samples.
        temporal_head: Whether model has per-timestep prediction head.
        pos_weight: Class imbalance weight for BCE (temporal mode).
        time_weighting: 'uniform' or 'early' (temporal mode).
        early_weight_factor: Max weight for early timesteps (temporal mode).
        class_weights: Per-class weights for cross-entropy (standard mode).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        inputs, targets = batch
        inputs = _to_device(inputs, device)
        targets = _to_device(targets, device)

        # Unpack survival targets: (y_binary, event_times, event_indicators)
        if survival_mode and isinstance(targets, (tuple, list)):
            y_binary, event_times, event_indicators = targets
        else:
            y_binary = targets
            event_times = None
            event_indicators = None

        # Optionally apply progressive time masking (Phase 4)
        if enable_masking and torch.rand(1).item() < masking_prob:
            x_ts = inputs[0]
            x_ts = _apply_progressive_time_masking(x_ts, min_timesteps=min_timesteps)
            inputs = [x_ts] + inputs[1:]

        optimizer.zero_grad()
        logits = model(inputs)

        # Compute loss — use real trajectory lengths from dataloader when available
        x_ts = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        if len(inputs) >= 4:
            traj_lengths = inputs[3]
        else:
            traj_lengths = _infer_trajectory_lengths_from_batch(x_ts)

        if survival_mode and temporal_head and event_times is not None:
            loss = compute_survival_loss(
                logits, event_times, event_indicators, traj_lengths,
                time_weighting=time_weighting,
                early_weight_factor=early_weight_factor,
            )
        elif temporal_head:
            loss = compute_temporal_loss(
                logits, y_binary, traj_lengths,
                pos_weight=pos_weight,
                time_weighting=time_weighting,
                early_weight_factor=early_weight_factor,
            )
        elif enable_weighting:
            loss = _compute_weighted_loss(
                logits, y_binary, x_ts,
                label_smoothing=label_smoothing,
                early_weight=early_weight,
                class_weights=class_weights,
            )
        else:
            loss = F.cross_entropy(
                logits, y_binary, label_smoothing=label_smoothing,
                weight=class_weights,
            )

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        postfix = {'Loss': f"{total_loss / n_batches:.4f}"}
        if scheduler is not None:
            postfix['LR'] = f"{scheduler.get_last_lr()[0]:.2e}"
        pbar.set_postfix(postfix)

    return total_loss / max(n_batches, 1)


def _run_phase(
    phase_name: str,
    model: nn.Module,
    train_dl,
    valid_dl,
    n_epochs: int,
    base_lr: float,
    finetune_cfg: FinetuneConfig,
    device: str,
    tracker: MetricTracker,
    early_stopper: Optional[EarlyStopping] = None,
    trial=None,
    global_epoch: int = 0,
    enable_masking: bool = False,
    enable_weighting: bool = False,
    # Per-timestep prediction options (passed through from run_finetune_v2)
    temporal_head: bool = False,
    pos_weight: Optional[torch.Tensor] = None,
    time_weighting: str = "uniform",
    early_weight_factor: float = 2.0,
    # Class weighting for standard (non-temporal) head
    class_weights: Optional[torch.Tensor] = None,
    # Survival mode
    survival_mode: bool = False,
) -> int:
    """
    Run a single training phase.

    Args:
        valid_dl: Validation dataloader. None = no validation (full trainval mode).
        early_stopper: Early stopping tracker. None = train for full epoch count.

    Returns:
        Updated global_epoch counter.
    """
    if n_epochs <= 0:
        return global_epoch

    # Build optimizer for currently trainable parameters
    param_groups = get_optimizer_param_groups(
        model, base_lr, finetune_cfg.lr_decay_factor, finetune_cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler: cosine warmup
    steps_per_epoch = len(train_dl)
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = int(finetune_cfg.warmup_fraction * total_steps)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    logger.info(f"--- {phase_name} ({n_epochs} epochs, base_lr={base_lr:.2e}) ---")

    for epoch in range(n_epochs):
        desc = f"{phase_name} [{epoch+1}/{n_epochs}]"
        train_loss = train_one_epoch(
            model, train_dl, optimizer, scheduler,
            device=device,
            grad_clip=finetune_cfg.grad_clip,
            label_smoothing=finetune_cfg.label_smoothing,
            enable_masking=enable_masking,
            masking_prob=finetune_cfg.masking_prob,
            min_timesteps=finetune_cfg.min_timesteps,
            enable_weighting=enable_weighting,
            early_weight=finetune_cfg.early_weight,
            temporal_head=temporal_head,
            pos_weight=pos_weight,
            time_weighting=time_weighting,
            early_weight_factor=early_weight_factor,
            class_weights=class_weights,
            survival_mode=survival_mode,
            desc=desc,
        )

        if valid_dl is not None:
            metrics = compute_val_metrics(model, valid_dl, device=device,
                                          temporal_head=temporal_head,
                                          survival_mode=survival_mode)
            val_auroc = metrics["auroc"]
            val_auprc = metrics["auprc"]

            if survival_mode and "cindex" in metrics:
                val_cindex = metrics["cindex"]
                # Survival mode: C-index is the primary metric
                val_score = val_cindex
                tracker.update(phase_name, global_epoch,
                               train_loss=train_loss, val_auroc=val_auroc,
                               val_auprc=val_auprc, val_cindex=val_cindex,
                               val_score=val_score)
                logger.info(
                    f"  Epoch {global_epoch + 1}: loss={train_loss:.4f}, "
                    f"val_cindex={val_cindex:.4f}, val_auroc={val_auroc:.4f}"
                )
            else:
                # Classification mode: combined AUROC + AUPRC
                val_score = (finetune_cfg.val_auroc_weight * val_auroc
                             + finetune_cfg.val_auprc_weight * val_auprc)
                tracker.update(phase_name, global_epoch,
                               train_loss=train_loss, val_auroc=val_auroc,
                               val_auprc=val_auprc, val_score=val_score)
                logger.info(
                    f"  Epoch {global_epoch + 1}: loss={train_loss:.4f}, "
                    f"val_auroc={val_auroc:.4f}, val_auprc={val_auprc:.4f}"
                )

            # Optuna reporting + pruning (report combined score)
            if trial is not None:
                import optuna
                trial.report(val_score, global_epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Early stopping on combined metric
            if early_stopper is not None and early_stopper(val_score, model):
                logger.info(f"  Early stopping at epoch {global_epoch + 1}")
                break
        else:
            tracker.update(phase_name, global_epoch, train_loss=train_loss)
            logger.info(
                f"  Epoch {global_epoch + 1}: loss={train_loss:.4f} (no validation)"
            )

        global_epoch += 1

    return global_epoch


def run_finetune_v2(
    data: dict,
    finetune_cfg: FinetuneConfig,
    pretrain_cfg: Optional[MLMConfig] = None,
    device: str = "cuda",
    trial=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Four-phase finetuning with transfer learning.

    Args:
        data: Output from prepare_data_and_dls().
        finetune_cfg: Finetuning configuration.
        pretrain_cfg: MLM config (needed to reconstruct model for weight loading).
        device: cuda or cpu.
        trial: Optuna trial for HP search integration (None = no HP search).
        verbose: Whether to log progress.

    Returns:
        Dict with 'model', 'tracker', 'best_auroc'.
    """
    # ========================================================================
    # 1. Load backbone (pretrained or fresh)
    # ========================================================================
    # Read temporal model settings from global config (single source of truth)
    model_cfg = cfg.get("model", {})
    temporal_head = model_cfg.get("temporal_head", False)
    causal = model_cfg.get("causal", False)
    temporal_head_dropout = model_cfg.get("temporal_head_dropout", 0.3)
    survival_mode = model_cfg.get("survival_mode", False)

    # Auto-enable causal masking when temporal head is on (prevents silent leakage)
    if temporal_head and not causal:
        logger.warning("temporal_head=True but causal=False! Auto-enabling causal masking. "
                       "Pass causal=False explicitly only if you intend to allow future info leakage.")
        cfg["model"]["causal"] = True
        causal = True

    if finetune_cfg.use_pretrained:
        backbone = load_pretrained_backbone(
            data, cfg,
            pretrain_cfg=pretrain_cfg,
            checkpoint_dir=finetune_cfg.pretrain_checkpoint_dir,
        )
    else:
        backbone = get_backbone(
            data, cfg,
            temporal_head=temporal_head,
            causal=causal,
            temporal_head_dropout=temporal_head_dropout,
            temporal_channel_idx=data.get('temporal_channel_idx'),
            exclude_channel_indices=data.get('exclude_channel_indices', []),
        )
        logger.info("Using randomly initialized backbone (no pretraining)")

    # Apply dropout overrides if specified
    if finetune_cfg.fc_dropout is not None or finetune_cfg.res_dropout is not None:
        set_dropout_rates(backbone, finetune_cfg.fc_dropout, finetune_cfg.res_dropout)

    backbone = backbone.to(device)

    # ========================================================================
    # 2. Create dataloaders
    # ========================================================================
    no_validation = finetune_cfg.valid_size <= 0.0

    if trial is not None and no_validation:
        raise ValueError(
            "Cannot use Optuna HPO (trial != None) with valid_size=0.0. "
            "Validation data is required for trial evaluation."
        )

    if no_validation:
        logger.info("Training on full trainval (no validation split)")
        train_dl = data["mixed_dls"].train
        valid_dl = None
    else:
        logger.info("Creating train/valid split for finetuning...")
        y = data["y"]
        splits = get_stratified_splits(
            y,
            valid_size=finetune_cfg.valid_size,
            random_state=42,
        )
        logger.info(f"  Train: {len(splits[0])} samples, Valid: {len(splits[1])} samples")

        mixed_dls = create_split_dataloaders(data, splits, cfg)
        train_dl = mixed_dls.train
        valid_dl = mixed_dls.valid

    # ========================================================================
    # 3. Setup tracking
    # ========================================================================
    tracker = MetricTracker()
    early_stopper = None if no_validation else EarlyStopping(
        patience=finetune_cfg.patience, mode="max",
    )
    global_epoch = 0

    # Log layer groups
    layer_groups = get_layer_groups(backbone)
    logger.info(f"Model layer groups ({len(layer_groups)}):")
    for name, params in layer_groups.items():
        n_params = sum(p.numel() for _, p in params)
        logger.info(f"  {name}: {n_params:,} params")

    # Class imbalance handling
    y_arr = np.array(data["y"])
    n_pos = y_arr.sum()
    n_neg = len(y_arr) - n_pos
    imbalance_ratio = float(n_neg / max(n_pos, 1))
    logger.info(f"Class distribution: {int(n_neg)} neg / {int(n_pos)} pos "
                f"(ratio={imbalance_ratio:.1f}:1)")

    phase_kwargs = {}
    if temporal_head:
        pw = torch.tensor([imbalance_ratio], device=device)
        phase_kwargs = dict(
            temporal_head=True,
            pos_weight=pw,
            time_weighting=finetune_cfg.time_weighting,
            early_weight_factor=finetune_cfg.early_weight_factor,
            survival_mode=survival_mode,
        )
        if survival_mode:
            logger.info(f"Survival mode: discrete-time hazard with temporal head, "
                         f"time_weighting={finetune_cfg.time_weighting}")
        else:
            logger.info(f"Temporal head: pos_weight={pw.item():.2f}, "
                         f"time_weighting={finetune_cfg.time_weighting}")
    else:
        # Standard head: weighted cross-entropy for class imbalance
        factor = finetune_cfg.pos_weight_factor
        if factor > 0:
            cw = torch.tensor([1.0, imbalance_ratio * factor], device=device)
            phase_kwargs = dict(class_weights=cw)
            logger.info(f"Standard head: class_weights={cw.tolist()} "
                         f"(pos_weight_factor={factor})")
        else:
            phase_kwargs = dict(class_weights=None)
            logger.info("Standard head: no class weighting (pos_weight_factor=0)")

    # ========================================================================
    # 4. Phase 1: Head-only training
    # ========================================================================
    # Freeze everything except head
    freeze_to(backbone, list(layer_groups.keys())[-2])  # freeze up to last transformer group
    logger.info("Phase 1: Training head only")

    global_epoch = _run_phase(
        "phase1_head", backbone, train_dl, valid_dl,
        n_epochs=finetune_cfg.phase1_epochs,
        base_lr=finetune_cfg.phase1_lr,
        finetune_cfg=finetune_cfg,
        device=device,
        tracker=tracker,
        early_stopper=early_stopper,
        trial=trial,
        global_epoch=global_epoch,
        **phase_kwargs,
    )

    # ========================================================================
    # 5. Phase 2: Partial unfreeze (upper transformer + head)
    # ========================================================================
    if early_stopper is not None:
        early_stopper.reset_patience()  # fresh patience budget; best_state preserved
    unfreeze_from(backbone, finetune_cfg.phase2_unfreeze_from)
    logger.info(f"Phase 2: Unfreezing from {finetune_cfg.phase2_unfreeze_from}")

    global_epoch = _run_phase(
        "phase2_partial", backbone, train_dl, valid_dl,
        n_epochs=finetune_cfg.phase2_epochs,
        base_lr=finetune_cfg.phase2_lr,
        finetune_cfg=finetune_cfg,
        device=device,
        tracker=tracker,
        early_stopper=early_stopper,
        trial=trial,
        global_epoch=global_epoch,
        **phase_kwargs,
    )

    # ========================================================================
    # 6. Phase 3: Full finetune
    # ========================================================================
    if early_stopper is not None:
        early_stopper.reset_patience()
    unfreeze_all(backbone)
    logger.info("Phase 3: Full finetune (all layers)")

    global_epoch = _run_phase(
        "phase3_full", backbone, train_dl, valid_dl,
        n_epochs=finetune_cfg.phase3_epochs,
        base_lr=finetune_cfg.phase3_lr,
        finetune_cfg=finetune_cfg,
        device=device,
        tracker=tracker,
        early_stopper=early_stopper,
        trial=trial,
        global_epoch=global_epoch,
        **phase_kwargs,
    )

    # ========================================================================
    # 7. Phase 4: Early prediction hardening (optional)
    # ========================================================================
    if finetune_cfg.enable_early_prediction and finetune_cfg.phase4_epochs > 0:
        if early_stopper is not None:
            early_stopper.reset_patience()
        logger.info("Phase 4: Early prediction hardening (progressive masking + weighted loss)")

        global_epoch = _run_phase(
            "phase4_early", backbone, train_dl, valid_dl,
            n_epochs=finetune_cfg.phase4_epochs,
            base_lr=finetune_cfg.phase4_lr,
            finetune_cfg=finetune_cfg,
            device=device,
            tracker=tracker,
            early_stopper=early_stopper,
            trial=trial,
            global_epoch=global_epoch,
            enable_masking=True,
            enable_weighting=True,
            **phase_kwargs,
        )

    # ========================================================================
    # 8. Restore best model (if validation was used) and save
    # ========================================================================
    if early_stopper is not None:
        early_stopper.restore_best(backbone)
        best_score = early_stopper.best_score or 0.0
        # Extract individual metrics from tracker at best epoch
        best_auroc = tracker.best("val_auroc")
        best_auprc = tracker.best("val_auprc")
        logger.info(f"Best validation: score={best_score:.4f}, "
                     f"auroc={best_auroc:.4f}, auprc={best_auprc:.4f}")
    else:
        best_auroc = None
        best_auprc = None
        best_score = None
        logger.info("Full trainval training complete (no validation metrics available)")

    # Save model + deployment bundle
    if finetune_cfg.model_name:
        save_model(backbone, finetune_cfg.model_name)
        save_deployment_bundle(data, cfg, finetune_cfg.model_name)

    clear_mem()

    return {
        "model": backbone,
        "tracker": tracker,
        "best_auroc": best_auroc,
        "best_auprc": best_auprc,
        "best_score": best_score,
    }
