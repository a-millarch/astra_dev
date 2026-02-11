"""
Pure-PyTorch finetuning with 4-phase transfer learning.

Phase 1: Head-only (warm up randomly initialized classification head)
Phase 2: Partial unfreeze (upper transformer layers + head)
Phase 3: Full finetune (all layers with discriminative LRs)
Phase 4: Early prediction hardening (optional progressive time masking)

Replaces the FastAI-based run_finetune() while keeping TSAI data loading.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from tsai.data.core import get_ts_dls
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.validation import get_splits

from astra.utils import cfg, logger, clear_mem
from astra.models.hybrid.model import TSTabFusionTransformerMultiHot
from astra.models.hybrid.mlm import TSTabFusionMLM, MLMConfig
from astra.models.hybrid.training import get_backbone
from astra.data.dataloader import dfwide2ts_dls

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
    save_model_fastai_compatible,
    _to_device,
)


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


def create_split_dataloaders(data: dict, splits, cfg_dict: dict):
    """
    Create train/valid mixed dataloaders from existing data arrays with split indices.

    Follows the same pattern as run_pretrain() in training.py for creating
    split-aware TSAI dataloaders from pre-normalized data.

    Args:
        data: Output from prepare_data_and_dls().
        splits: Tuple of (train_indices, valid_indices).
        cfg_dict: Global config dict.

    Returns:
        Mixed dataloader with train and valid splits.
    """
    X = data["X"]
    y = data["y"]
    num_cols = data["num_cols"]
    cat_cols = data["cat_cols"]
    tfms = data["tfms"]
    procs = data["procs"]
    bs = cfg_dict["training"]["bs"]

    # Reconstruct normalized tabular DataFrame
    tab_scaler = data.get("tab_scaler", None)
    if tab_scaler is not None and num_cols:
        trainval_tab_normalized = data["trainval"].tab_df.copy()
        trainval_tab_normalized[num_cols] = tab_scaler.transform(
            data["trainval"].tab_df[num_cols]
        )
    else:
        trainval_tab_normalized = data["trainval"].tab_df

    # 1. Continuous TS
    ts_dls = get_ts_dls(
        X, y,
        splits=splits,
        tfms=tfms,
        batch_tfms=None,
        bs=bs,
        drop_last=False,
    )

    # 2. Tabular
    tab_dls = get_tabular_dls(
        trainval_tab_normalized,
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg_dict["target"],
        splits=splits,
        bs=bs,
        drop_last=False,
    )

    # 3. Categorical TS
    ts_cat_dls = get_ts_dls(
        data["ts_cat_dls"].X_multi_hot.astype(np.int64),
        y,
        splits=splits,
        bs=bs,
        drop_last=False,
    )

    # 4. Combine
    mixed_dls = get_mixed_dls(ts_dls, tab_dls, ts_cat_dls, bs=bs)
    return mixed_dls


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
    """
    backbone = get_backbone(data, cfg_dict)

    if checkpoint_dir is None:
        checkpoint_dir = f'./pretrain_checkpoints/{cfg_dict["model_name"]}'

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Need the same MLM model structure to load, then extract backbone
    if pretrain_cfg is None:
        pc = cfg_dict["pretrain"]
        pretrain_cfg = MLMConfig(
            mask_prob_ts=pc["mask_prob_ts"],
            mask_prob_cat_ts=pc["mask_prob_cat_ts"],
            mask_prob_cat=pc["mask_prob_cat"],
            mask_prob_cont=pc["mask_prob_cont"],
        )

    # Check for c_in mismatch (EBM feature adds a channel)
    full_c_in = backbone.W_P.in_channels
    pretrain_W_P_weight = checkpoint["model_state_dict"]["backbone.W_P.weight"]
    pretrain_c_in = pretrain_W_P_weight.shape[1]

    if pretrain_c_in != full_c_in:
        logger.info(
            f"W_P channel mismatch: checkpoint has {pretrain_c_in}, "
            f"current model has {full_c_in} — expanding W_P"
        )
        # Load all weights except W_P via MLM wrapper
        mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
        filtered_state = {
            k: v for k, v in checkpoint["model_state_dict"].items()
            if "W_P" not in k
        }
        mlm_model.load_state_dict(filtered_state, strict=False)
        backbone = mlm_model.backbone

        # Expand W_P: copy pretrained weights, Xavier-init new channel(s)
        _expand_w_p(backbone, pretrain_W_P_weight,
                     checkpoint["model_state_dict"]["backbone.W_P.bias"],
                     data.get("ebm_channel_idx"))
        logger.info(f"Pretrained weights loaded with W_P expansion from {checkpoint_path}")
    else:
        mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
        mlm_model.load_state_dict(checkpoint["model_state_dict"])
        backbone = mlm_model.backbone
        logger.info(f"Pretrained weights loaded from {checkpoint_path}")

    return backbone


def _expand_w_p(
    backbone: nn.Module,
    old_weight: torch.Tensor,
    old_bias: torch.Tensor,
    ebm_channel_idx: Optional[int] = None,
):
    """
    Expand W_P Conv1d to accommodate additional input channel(s).

    Copies pretrained weights for existing channels to their correct
    positions and Xavier-initializes the new EBM channel.

    W_P is nn.Conv1d(c_in, continuous_dim, kernel_size=1):
        weight shape: [out_channels, in_channels, 1]
        bias shape: [out_channels]
    """
    new_c_in = backbone.W_P.in_channels
    old_c_in = old_weight.shape[1]

    with torch.no_grad():
        backbone.W_P.bias.data.copy_(old_bias)

        old_idx = 0
        for new_idx in range(new_c_in):
            if new_idx == ebm_channel_idx:
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
        f"(EBM at idx {ebm_channel_idx}, Xavier-initialized)"
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
) -> torch.Tensor:
    """
    Compute loss weighted by data availability.

    Ported from WeightedLossCallback in callbacks.py.
    Samples with less data (sparser time series) get higher weight.
    """
    loss_per_sample = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing,
    )

    # Data availability: fraction of non-zero timesteps
    # x_ts shape: [batch, c_in, seq_len]
    data_present = (x_ts.abs().sum(dim=1) > 1e-6).float()  # [batch, seq_len]
    availability_ratio = data_present.sum(dim=1) / x_ts.shape[2]  # [batch]

    # Weight: high for sparse data, low for dense data
    weights = early_weight - (early_weight - 1.0) * availability_ratio

    return (loss_per_sample * weights).mean()


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

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        inputs, targets = batch
        inputs = _to_device(inputs, device)
        targets = _to_device(targets, device)

        # Optionally apply progressive time masking (Phase 4)
        if enable_masking and torch.rand(1).item() < masking_prob:
            x_ts, x_tab, x_ts_cat = inputs
            x_ts = _apply_progressive_time_masking(x_ts, min_timesteps=min_timesteps)
            inputs = (x_ts, x_tab, x_ts_cat)

        optimizer.zero_grad()
        logits = model(inputs)

        # Compute loss
        if enable_weighting:
            x_ts = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            loss = _compute_weighted_loss(
                logits, targets, x_ts,
                label_smoothing=label_smoothing,
                early_weight=early_weight,
            )
        else:
            loss = F.cross_entropy(
                logits, targets, label_smoothing=label_smoothing,
            )

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

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
    early_stopper: EarlyStopping,
    trial=None,
    global_epoch: int = 0,
    enable_masking: bool = False,
    enable_weighting: bool = False,
) -> int:
    """
    Run a single training phase.

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
        )
        val_auroc = compute_auroc(model, valid_dl, device=device)

        tracker.update(phase_name, global_epoch, train_loss=train_loss, val_auroc=val_auroc)
        logger.info(
            f"  Epoch {global_epoch + 1}: loss={train_loss:.4f}, val_auroc={val_auroc:.4f}"
        )

        # Optuna reporting + pruning
        if trial is not None:
            import optuna
            trial.report(val_auroc, global_epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping (tracks best model internally)
        if early_stopper(val_auroc, model):
            logger.info(f"  Early stopping at epoch {global_epoch + 1}")
            break

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
    if finetune_cfg.use_pretrained:
        backbone = load_pretrained_backbone(
            data, cfg,
            pretrain_cfg=pretrain_cfg,
            checkpoint_dir=finetune_cfg.pretrain_checkpoint_dir,
        )
    else:
        backbone = get_backbone(data, cfg)
        logger.info("Using randomly initialized backbone (no pretraining)")

    # Apply dropout overrides if specified
    if finetune_cfg.fc_dropout is not None or finetune_cfg.res_dropout is not None:
        set_dropout_rates(backbone, finetune_cfg.fc_dropout, finetune_cfg.res_dropout)

    backbone = backbone.to(device)

    # ========================================================================
    # 2. Create train/valid split dataloaders
    # ========================================================================
    logger.info("Creating train/valid split for finetuning...")
    y = data["y"]
    splits = get_splits(
        y,
        valid_size=finetune_cfg.valid_size,
        stratify=True,
        random_state=42,
        shuffle=True,
    )
    splits = (splits[0], splits[1])
    logger.info(f"  Train: {len(splits[0])} samples, Valid: {len(splits[1])} samples")

    mixed_dls = create_split_dataloaders(data, splits, cfg)
    train_dl = mixed_dls.train
    valid_dl = mixed_dls.valid

    # ========================================================================
    # 3. Setup tracking
    # ========================================================================
    tracker = MetricTracker()
    early_stopper = EarlyStopping(
        patience=finetune_cfg.patience, mode="max",
    )
    global_epoch = 0

    # Log layer groups
    layer_groups = get_layer_groups(backbone)
    logger.info(f"Model layer groups ({len(layer_groups)}):")
    for name, params in layer_groups.items():
        n_params = sum(p.numel() for _, p in params)
        logger.info(f"  {name}: {n_params:,} params")

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
    )

    # ========================================================================
    # 5. Phase 2: Partial unfreeze (upper transformer + head)
    # ========================================================================
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
    )

    # ========================================================================
    # 6. Phase 3: Full finetune
    # ========================================================================
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
    )

    # ========================================================================
    # 7. Phase 4: Early prediction hardening (optional)
    # ========================================================================
    if finetune_cfg.enable_early_prediction and finetune_cfg.phase4_epochs > 0:
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
        )

    # ========================================================================
    # 8. Restore best model and save
    # ========================================================================
    early_stopper.restore_best(backbone)
    best_auroc = early_stopper.best_score or 0.0
    logger.info(f"Best validation AUROC: {best_auroc:.4f}")

    # Save in FastAI-compatible format
    if finetune_cfg.model_name:
        save_model_fastai_compatible(backbone, data, finetune_cfg.model_name, cfg)

    clear_mem()

    return {
        "model": backbone,
        "tracker": tracker,
        "best_auroc": best_auroc,
    }
