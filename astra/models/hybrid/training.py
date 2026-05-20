import logging
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch

from astra.utils import cfg, clear_mem, PROJECT_ROOT
from astra.models.hybrid.mlm import TSTabFusionMLM, MLMConfig, pretrain_mlm_enhanced
from astra.models.hybrid.model import TSTabFusionTransformerMultiHot
from astra.data.mixed_dataloader import (
    AstraMixedDataset,
    AstraMixedDataLoader,
    get_stratified_splits,
)

logger = logging.getLogger(__name__)


def get_backbone(
    data, cfg,
    temporal_head=False,
    causal=False,
    temporal_head_dropout=0.3,
    temporal_head_mult=0.5,
    temporal_channel_idx=None,
    exclude_channel_indices=None,
    bin_width_channel_idx=None,
):
    model_cfg = cfg["model"]
    backbone = TSTabFusionTransformerMultiHot(
        c_in=data["c_in"],
        c_out=2,
        seq_len=data["seq_len"],
        classes=data["classes"],
        cont_names=data["num_cols"],
        ts_cat_dims=data["ts_cat_dims"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        fc_dropout=model_cfg["fc_dropout"],
        res_dropout=model_cfg["res_dropout"],
        fc_mults=(model_cfg["fc_mults_1"], model_cfg["fc_mults_2"]),
        cat_ts_combine='add',
        use_count_normalization=False,
        temporal_head=temporal_head,
        causal=causal,
        temporal_head_dropout=temporal_head_dropout,
        temporal_head_mult=temporal_head_mult,
        temporal_channel_idx=temporal_channel_idx,
        exclude_channel_indices=exclude_channel_indices or [],
        head_pool=model_cfg.get("head_pool", "flatten"),
        per_feature_cont_proj=model_cfg.get("per_feature_cont_proj", False),
        cat_ts_gate=model_cfg.get("cat_ts_gate", False),
        local_temporal_kernel=model_cfg.get("local_temporal_kernel", 1),
        bin_width_channel_idx=bin_width_channel_idx,
        bin_width_modulation=model_cfg.get("bin_width_modulation", False),
        ts_cat_profile_dims=data.get("ts_cat_profile_dims"),
    )
    return backbone

def run_pretrain(data, pretrain_cfg=None, device='cuda'):
    """
    Run pretraining with NORMALIZED continuous features.

    Args:
        data: dict returned by prepare_data_and_dls() with fixed normalization
        pretrain_cfg: MLMConfig or None (then a default is used)
        device: 'cuda' or 'cpu'

    Returns:
        pretrain_cfg, mlm_model, mixed_dls_ul
    """
    if pretrain_cfg is None:
        pc = cfg["pretrain"]
        pretrain_cfg = MLMConfig(
            mask_prob_ts=pc["mask_prob_ts"],
            mask_prob_cat_ts=pc["mask_prob_cat_ts"],
            mask_prob_cat=pc["mask_prob_cat"],
            mask_prob_cont=pc["mask_prob_cont"],
            epochs=pc["epochs"],
            lr=pc["lr"],
            warmup_epochs=pc["warmup_epochs"],
            ts_loss_weight=pc["ts_loss_weight"],
            cat_ts_loss_weight=pc["cat_ts_loss_weight"],
            cat_loss_weight=pc["cat_loss_weight"],
            cont_loss_weight=pc["cont_loss_weight"],
            contrastive_weight=pc["contrastive_weight"],
            temperature=pc["temperature"],
            patience=pc["patience"],
            save_best=pc["save_best"],
            checkpoint_dir=str(PROJECT_ROOT / pc["checkpoint_dir"] / cfg["model_name"]),
            sparse_aware_masking=pc.get("sparse_aware_masking", False),
            mask_prob_ts_data=pc.get("mask_prob_ts_data", 0.30),
            mask_prob_ts_empty=pc.get("mask_prob_ts_empty", 0.05),
            mask_prob_cat_ts_data=pc.get("mask_prob_cat_ts_data", 0.40),
            mask_prob_cat_ts_empty=pc.get("mask_prob_cat_ts_empty", 0.03),
        )

    # ============================================================================
    # EXTRACT NORMALIZED DATA (already normalized in prepare_data_and_dls)
    # ============================================================================
    X = data["X"]  # Already normalized!
    y = data["y"]
    num_cols = data["num_cols"]
    cat_cols = data["cat_cols"]
    classes = data["classes"]

    # ============================================================================
    # VALIDATE NORMALIZATION
    # ============================================================================
    logger.info("Validating data normalization for pretraining...")
    non_pad_flat = X[~np.isclose(X, 0.0, atol=1e-8)]
    logger.info(f"  X overall mean: {X.mean():.4f}, std: {X.std():.4f} (includes {np.isclose(X, 0.0, atol=1e-8).mean()*100:.0f}% padding)")
    logger.info(f"  X non-padding mean: {non_pad_flat.mean():.4f}, std: {non_pad_flat.std():.4f}")

    if abs(non_pad_flat.mean()) > 1.0 or not (0.5 < non_pad_flat.std() < 2.0):
        logger.warning(f"X normalization looks suspicious! non-padding mean={non_pad_flat.mean():.4f}, std={non_pad_flat.std():.4f}")
        logger.warning("    Expected: mean ~ 0, std ~ 1")
    else:
        logger.info("  Time series normalization looks good")

    # ============================================================================
    # CREATE NORMALIZED TABULAR DATAFRAME
    # ============================================================================
    tab_scaler = data.get("tab_scaler", None)
    tab_encoder = data.get("tab_encoder", None)

    if tab_scaler is not None and num_cols:
        logger.info("Creating normalized tabular DataFrame for pretraining...")
        trainval_tab_normalized = data["trainval"].tab_df.copy()
        trainval_tab_normalized[num_cols] = tab_scaler.transform(
            data["trainval"].tab_df[num_cols]
        )
        tab_mean = trainval_tab_normalized[num_cols].mean().mean()
        tab_std = trainval_tab_normalized[num_cols].std().mean()
        logger.info(f"  Tabular mean: {tab_mean:.4f}, std: {tab_std:.4f}")

        if abs(tab_mean) > 1.0 or not (0.5 < tab_std < 2.0):
            logger.warning(f"Tabular normalization looks suspicious!")
        else:
            logger.info("  Tabular normalization looks good")
    else:
        logger.warning("No tab_scaler found in data dict - using raw tabular data")
        trainval_tab_normalized = data["trainval"].tab_df

    # ============================================================================
    # CREATE TRAIN/VALID SPLITS FOR PRETRAINING
    # ============================================================================
    logger.info("Creating train/valid splits for unsupervised pretraining...")

    splits = get_stratified_splits(y, valid_size=0.2, random_state=42)

    logger.info(f"  Train samples: {len(splits[0])}")
    logger.info(f"  Valid samples: {len(splits[1])}")

    # ============================================================================
    # CREATE UNLABELED DATALOADERS (no targets needed for pretraining)
    # ============================================================================
    logger.info("Creating unlabeled dataloaders for pretraining...")

    # Encode tabular data
    trainval_tab_encoded = tab_encoder.transform(trainval_tab_normalized, cat_cols, num_cols)
    x_cat, x_cont = tab_encoder.get_cat_cont_arrays(trainval_tab_encoded, cat_cols, num_cols)

    X_multi_hot = data["X_multi_hot"]

    pretrain_dataset = AstraMixedDataset(
        X_ts=X,
        x_cat=x_cat,
        x_cont=x_cont,
        X_ts_cat=X_multi_hot,
        y=y,
        X_ts_cat_profiles=data.get("X_ts_cat_profiles"),
    )
    mixed_dls_ul = AstraMixedDataLoader(
        pretrain_dataset,
        splits=splits,
        bs=cfg["training"]["bs"],
        shuffle_train=True,
    )
    logger.info("  Created mixed dataloader")

    # ============================================================================
    # VALIDATE DATALOADER OUTPUTS
    # ============================================================================
    logger.info("Validating dataloader batch...")

    for batch in mixed_dls_ul.train:
        inputs, targets = batch

        if isinstance(inputs, (tuple, list)):
            ts_batch = inputs[0]
            ts_mean = float(ts_batch.mean())
            ts_std = float(ts_batch.std())
            logger.info(f"  TS batch mean: {ts_mean:.4f}, std: {ts_std:.4f}")

            if abs(ts_mean) > 2.0:
                logger.warning("  TS batch not normalized!")

        break

    # ============================================================================
    # CREATE BACKBONE AND MLM MODEL
    # ============================================================================
    logger.info("Creating backbone and MLM model...")

    backbone = get_backbone(
        data, cfg,
        temporal_channel_idx=data.get('temporal_channel_idx'),
        exclude_channel_indices=data.get('exclude_channel_indices', []),
        bin_width_channel_idx=data.get('bin_width_channel_idx'),
    )
    logger.info(f"  Backbone: {type(backbone).__name__}")

    mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
    logger.info(f"  MLM model created")

    # ============================================================================
    # RUN PRETRAINING
    # ============================================================================
    logger.info("="*80)
    logger.info("STARTING PRETRAINING")
    logger.info("="*80)
    logger.info(f"  Epochs: {pretrain_cfg.epochs}")
    logger.info(f"  Learning rate: {pretrain_cfg.lr}")
    logger.info(f"  Batch size: {cfg['training']['bs']}")
    logger.info(f"  Device: {device}")
    logger.info("="*80)

    history = pretrain_mlm_enhanced(
        mlm_model,
        train_loader=mixed_dls_ul.train,
        val_loader=mixed_dls_ul.valid,
        config=pretrain_cfg,
        device=device
    )

    logger.info("="*80)
    logger.info("PRETRAINING COMPLETE")
    logger.info("="*80)
    checkpoint_dir = Path(pretrain_cfg.checkpoint_dir)
    if (checkpoint_dir / 'best_model.pt').exists():
        logger.info(f"Pretrained model saved to {checkpoint_dir / 'best_model.pt'}")
    else:
        logger.error(f"WARNING: best_model.pt not found in {checkpoint_dir}!")

    logger.info("\nExpected loss ranges (with normalization):")
    logger.info("  Total loss: 5-50 (not 1000s!)")
    logger.info("  TS loss: 0.5-5")
    logger.info("  Cat TS loss: 0.3-2")
    logger.info("  Cat loss: 0.5-3")
    logger.info("  Cont loss: 0.5-10 (not 9000!)")
    logger.info("  Contrastive: 1-10")

    return pretrain_cfg, mlm_model, mixed_dls_ul
