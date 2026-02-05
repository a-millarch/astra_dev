import os
import numpy as np
from tqdm.auto import tqdm

import torch
from fastai.learner import Learner

from tsai.data.core import get_ts_dls
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.validation import get_splits
from tsai.all import LabelSmoothingCrossEntropyFlat, Learner

from astra.utils import cfg, logger, clear_mem
from astra.models.hybrid.mlm import TSTabFusionMLM, MLMConfig, pretrain_mlm_enhanced

from astra.models.callbacks import SkipValidationCallback, ProgressiveTimeMaskingCallback, WeightedLossCallback
from astra.models.hybrid.model import TSTabFusionTransformerMultiHot
from astra.data.dataloader import dfwide2ts_dls, tscatdfwide2x

def get_backbone(data, cfg): #TODO: use cfg model parameters
    backbone = TSTabFusionTransformerMultiHot(
    c_in=data["ts_dls"].vars,
    c_out=2,
    seq_len=data["mixed_dls"].len,
    classes= data["classes"],
    cont_names=data["num_cols"],
    ts_cat_dims=data["ts_cat_dls"].ts_cat_dims,
    d_model=cfg["model"]["d_model"],
    n_layers=cfg["model"]["n_layers"],
    n_heads=cfg["model"]["n_heads"],
    fc_dropout=cfg["model"]["fc_dropout"] ,
    res_dropout=cfg["model"]["res_dropout"] ,
    fc_mults=(cfg["model"]["fc_mults_1"],cfg["model"]["fc_mults_2"]),
   # d_ff= cfg["model"]["d_ff"],

    cat_ts_combine='add',
    use_count_normalization=False
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
            checkpoint_dir=pc["checkpoint_dir"],
        )

    # ============================================================================
    # EXTRACT NORMALIZED DATA (already normalized in prepare_data_and_dls)
    # ============================================================================
    X = data["X"]  # ← Already normalized!
    y = data["y"]
    num_cols = data["num_cols"]
    cat_cols = data["cat_cols"]
    classes = data["classes"]
    tfms = data["tfms"]
    batch_tfms = data.get("batch_tfms", None)  # Should be None with fixed normalization
    procs = data["procs"]  # Should NOT include Normalize
    
    # ============================================================================
    # VALIDATE NORMALIZATION
    # ============================================================================
    logger.info("Validating data normalization for pretraining...")
    logger.info(f"  X mean: {X.mean():.4f}, std: {X.std():.4f}")
    
    if abs(X.mean()) > 1.0 or not (0.5 < X.std() < 2.0):
        logger.warning(f"⚠️  X normalization looks suspicious! mean={X.mean():.4f}, std={X.std():.4f}")
        logger.warning("    Expected: mean ≈ 0, std ≈ 1")
    else:
        logger.info("  ✓ Time series normalization looks good")
    
    # ============================================================================
    # CREATE NORMALIZED TABULAR DATAFRAME
    # ============================================================================
    # The raw trainval.tab_df is NOT normalized
    # We need to apply the tab_scaler to get normalized version
    
    tab_scaler = data.get("tab_scaler", None)
    
    if tab_scaler is not None and num_cols:
        logger.info("Creating normalized tabular DataFrame for pretraining...")
        
        # Create normalized copy
        trainval_tab_normalized = data["trainval"].tab_df.copy()
        trainval_tab_normalized[num_cols] = tab_scaler.transform(
            data["trainval"].tab_df[num_cols]
        )
        
        # Validate
        tab_mean = trainval_tab_normalized[num_cols].mean().mean()
        tab_std = trainval_tab_normalized[num_cols].std().mean()
        logger.info(f"  Tabular mean: {tab_mean:.4f}, std: {tab_std:.4f}")
        
        if abs(tab_mean) > 1.0 or not (0.5 < tab_std < 2.0):
            logger.warning(f"⚠️  Tabular normalization looks suspicious!")
        else:
            logger.info("  ✓ Tabular normalization looks good")
    else:
        # No scaler (shouldn't happen with new code) or no continuous cols
        logger.warning("No tab_scaler found in data dict - using raw tabular data")
        trainval_tab_normalized = data["trainval"].tab_df
    
    # ============================================================================
    # CREATE TRAIN/VALID SPLITS FOR PRETRAINING
    # ============================================================================
    logger.info("Creating train/valid splits for unsupervised pretraining...")
    
    ts_splits = get_splits(
        y,
        valid_size=0.2,
        stratify=True,
        random_state=42,
        shuffle=True,
        check_splits=True,
        show_plot=True
    )
    splits = (ts_splits[0], ts_splits[1])
    
    logger.info(f"  Train samples: {len(splits[0])}")
    logger.info(f"  Valid samples: {len(splits[1])}")
    
    # ============================================================================
    # CREATE UNLABELED DATALOADERS (no targets needed for pretraining)
    # ============================================================================
    logger.info("Creating unlabeled dataloaders for pretraining...")
    
    # 1. Continuous time series (already normalized)
    ts_dls_ul = get_ts_dls(
        X,  # ← Already normalized
        splits=splits,
        tfms=tfms,
        batch_tfms=None,  # ← NO batch transforms! Already normalized
        bs=cfg["training"]["bs"],
        drop_last=False
    )
    logger.info("  ✓ Created continuous TS dataloader")
    
    # 2. Tabular features (now normalized!)
    tab_dls_ul = get_tabular_dls(
        trainval_tab_normalized,  # ← NOW NORMALIZED!
        procs=procs,  # Should NOT include Normalize
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        splits=splits,
        bs=cfg["training"]["bs"],
        drop_last=False
    )
    logger.info("  ✓ Created tabular dataloader")
    
    # 3. Categorical time series (no normalization needed)
    ts_cat_dls_ul = get_ts_dls(
        data["ts_cat_dls"].X_multi_hot.astype(np.int64),
        splits=splits,
        bs=cfg["training"]["bs"],
        drop_last=False
    )
    logger.info("  ✓ Created categorical TS dataloader")
    
    # 4. Combine into mixed dataloader
    mixed_dls_ul = get_mixed_dls(
        ts_dls_ul,
        tab_dls_ul,
        ts_cat_dls_ul,
        bs=cfg["training"]["bs"]
    )
    logger.info("  ✓ Created mixed dataloader")
    
    # ============================================================================
    # VALIDATE DATALOADER OUTPUTS
    # ============================================================================
    logger.info("Validating dataloader batch...")
    
    for batch in mixed_dls_ul.train:
        inputs, targets = batch
        
        # Check continuous TS
        if isinstance(inputs, (tuple, list)):
            ts_batch = inputs[0]
            # Convert to float for formatting
            ts_mean = float(ts_batch.mean())
            ts_std = float(ts_batch.std())
            logger.info(f"  TS batch mean: {ts_mean:.4f}, std: {ts_std:.4f}")
            
            if abs(ts_mean) > 2.0:
                logger.warning("  ⚠️  TS batch not normalized!")
        
        break  # Just check first batch
    
    # ============================================================================
    # CREATE BACKBONE AND MLM MODEL
    # ============================================================================
    logger.info("Creating backbone and MLM model...")
    
    backbone = get_backbone(data, cfg)
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
    logger.info('Pretrained model saved')
    
    # Expected loss ranges with normalized data:
    logger.info("\n📊 Expected loss ranges (with normalization):")
    logger.info("  Total loss: 5-50 (not 1000s!)")
    logger.info("  TS loss: 0.5-5")
    logger.info("  Cat TS loss: 0.3-2")
    logger.info("  Cat loss: 0.5-3")
    logger.info("  Cont loss: 0.5-10 (not 9000!)")
    logger.info("  Contrastive: 1-10")
    
    return pretrain_cfg, mlm_model, mixed_dls_ul




def run_finetune(
    data,
    model_name: str,
    use_pretrained: bool = True,
    pretrain_cfg: MLMConfig = None,
    skip_valid: bool = True,
    lr: float = 4.7863e-4,
    n_epochs: int = 22
):
    """
    Fine-tune classifier.
    If use_pretrained=True, loads backbone from pretrain checkpoint.
    """
    mixed_dls = data["mixed_dls"]
    num_cols = data["num_cols"]
    classes = data["classes"]

    backbone = get_backbone(data, cfg)
    
    if use_pretrained:
        if pretrain_cfg is None:
            checkpoint_dir = './pretrain_checkpoints'
        else:
            checkpoint_dir = pretrain_cfg.checkpoint_dir
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
        # need the same MLM model structure to load, then extract backbone
        mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
        mlm_model.load_state_dict(checkpoint['model_state_dict'])
        backbone = mlm_model.backbone
        logger.info("Pretrained model loaded")

    loss_func = LabelSmoothingCrossEntropyFlat()
    cbs = [SkipValidationCallback()] if skip_valid else None

    learn = Learner(
        mixed_dls,
        backbone,
        loss_func=loss_func,
        metrics=None,
        cbs=cbs
    )
    
    learn.fit_one_cycle(n_epochs, lr)
    learn.save(model_name)
    clear_mem()
    return learn


def get_preds_mixed(learner, dl, with_input=False, with_decoded=False, 
                     with_loss=False, act=None, save_preds=None, save_targs=None,
                     concat_dim=0, debug=False):
    """
    Custom get_preds that works with TSAI mixed dataloaders.
    
    Drop-in replacement for learner.get_preds() that bypasses the
    problematic callback system.
    
    Args:
        learner: fastai Learner
        dl: DataLoader (e.g., mixed_dls.train)
        with_input: Return inputs
        with_decoded: Return decoded predictions
        with_loss: Return losses
        act: Activation function to apply (default: from loss_func)
        save_preds: Path to save predictions
        save_targs: Path to save targets
        concat_dim: Dimension to concatenate results
    
    Returns:
        Tuple of (predictions, targets) or more depending on flags
    """
    learner.model.eval()
    
    all_preds = []
    all_targets = []
    all_inputs = [] if with_input else None
    all_losses = [] if with_loss else None
    
    # Get activation function
    if act is None:
        act = getattr(learner.loss_func, 'activation', None)
        if act is None:
            act = torch.nn.Identity()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dl, desc="Getting predictions", leave=False)):
            if debug and batch_idx == 0:
                print(f"\n=== DEBUG: First Batch ===")
                print(f"Batch type: {type(batch)}")
                print(f"Batch length: {len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}")
                if isinstance(batch, (tuple, list)):
                    for i, elem in enumerate(batch):
                        print(f"  batch[{i}]: type={type(elem)}, ", end="")
                        if torch.is_tensor(elem):
                            print(f"shape={elem.shape}, dtype={elem.dtype}")
                        elif isinstance(elem, (tuple, list)):
                            print(f"length={len(elem)}")
                            for j, sub in enumerate(elem):
                                print(f"    batch[{i}][{j}]: type={type(sub)}, ", end="")
                                if torch.is_tensor(sub):
                                    print(f"shape={sub.shape}")
                                else:
                                    print(f"value={sub}")
                        else:
                            print(f"value={elem}")
                print("=========================\n")
            
            # Unpack batch - handle different formats
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 1:
                    # Only inputs, no targets
                    inputs = batch[0]
                    targets = None
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            else:
                # Single element batch
                inputs = batch
                targets = None
            
            # Move to device
            device = learner.dls.device if hasattr(learner.dls, 'device') else 'cuda'
            
            if isinstance(inputs, (tuple, list)):
                # Mixed dataloader: (x_ts, x_tab, x_ts_cat)
                inputs_device = []
                for inp in inputs:
                    if isinstance(inp, (tuple, list)):
                        # Nested tuple (x_tab)
                        inputs_device.append(tuple(i.to(device) if torch.is_tensor(i) else i for i in inp))
                    elif torch.is_tensor(inp):
                        inputs_device.append(inp.to(device))
                    else:
                        inputs_device.append(inp)
                inputs_device = tuple(inputs_device)
            else:
                inputs_device = inputs.to(device) if torch.is_tensor(inputs) else inputs
            
            # Handle targets
            if targets is not None:
                # Handle case where targets might be a tuple or list
                if isinstance(targets, (tuple, list)):
                    targets = targets[0] if len(targets) > 0 else targets
                
                # Ensure targets is a tensor
                if not torch.is_tensor(targets):
                    targets = torch.tensor(targets, device=device)
                elif targets.device != torch.device(device):
                    targets = targets.to(device)
            else:
                # No targets in batch - create dummy targets
                # This shouldn't happen with get_preds, but handle it gracefully
                batch_size = inputs_device[0].shape[0] if isinstance(inputs_device, tuple) else inputs_device.shape[0]
                targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Forward pass
            preds = learner.model(inputs_device)
            
            # Apply activation
            if act is not None:
                preds = act(preds)
            
            # Calculate loss if requested
            if with_loss:
                loss = learner.loss_func(preds, targets)
                all_losses.append(loss.detach().cpu())
            
            # Store results
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            if with_input:
                # Store inputs (this can be memory intensive)
                if isinstance(inputs_device, (tuple, list)):
                    inputs_cpu = []
                    for inp in inputs_device:
                        if isinstance(inp, (tuple, list)):
                            inputs_cpu.append(tuple(i.cpu() if torch.is_tensor(i) else i for i in inp))
                        elif torch.is_tensor(inp):
                            inputs_cpu.append(inp.cpu())
                        else:
                            inputs_cpu.append(inp)
                    all_inputs.append(tuple(inputs_cpu))
                else:
                    all_inputs.append(inputs_device.cpu())
    
    # Concatenate results
    preds = torch.cat(all_preds, dim=concat_dim)
    targets = torch.cat(all_targets, dim=concat_dim)
    
    # Save if requested
    if save_preds is not None:
        torch.save(preds, save_preds)
    if save_targs is not None:
        torch.save(targets, save_targs)
    
    # Build return tuple
    result = [preds, targets]
    
    if with_input:
        # Inputs are harder to concatenate due to nested structure
        result = [all_inputs] + result
    
    if with_loss:
        losses = torch.stack(all_losses)
        result.append(losses)
    
    if with_decoded:
        # For classification, decoded = argmax
        if preds.ndim > 1 and preds.shape[1] > 1:
            decoded = torch.argmax(preds, dim=1)
        else:
            decoded = preds
        result.insert(1, decoded)
    
    return tuple(result) if len(result) > 2 else (result[0], result[1])


def patch_learner_get_preds(learner):
    """
    Patch a Learner instance to use the custom get_preds.
    
    Args:
        learner: fastai Learner instance
    
    Returns:
        Same learner (modified in-place)
    
    Usage:
        learn = Learner(mixed_dls, backbone, ...)
        learn = patch_learner_get_preds(learn)
        
        # Now this works!
        preds, targs = learn.get_preds(dl=mixed_dls.train)
    """
    import types
    
    def custom_get_preds(self, ds_idx=1, dl=None, with_input=False, with_decoded=False,
                         with_loss=False, act=None, inner=False, reorder=True, 
                         cbs=None, save_preds=None, save_targs=None, concat_dim=0):
        """Custom get_preds that bypasses callbacks."""
        if dl is None:
            dl = self.dls[ds_idx]
        
        return get_preds_mixed(
            self, dl, 
            with_input=with_input,
            with_decoded=with_decoded,
            with_loss=with_loss,
            act=act,
            save_preds=save_preds,
            save_targs=save_targs,
            concat_dim=concat_dim
        )
    
    # Replace get_preds method
    learner.get_preds = types.MethodType(custom_get_preds, learner)
    
    return learner


# ============================================================================
# Convenience Function
# ============================================================================

def create_learner_with_working_get_preds(dls, model, loss_func=None, opt_func=None, 
                                          lr=0.001, splitter=None, cbs=None, 
                                          metrics=None, path=None, model_dir='models', 
                                          wd=None, wd_bn_bias=False, train_bn=True, 
                                          moms=(0.95, 0.85, 0.95)):
    """
    Create a Learner with working get_preds() for mixed dataloaders.
    
    Drop-in replacement for Learner() that automatically patches get_preds.
    
    Usage:
        learn = create_learner_with_working_get_preds(
            mixed_dls, backbone,
            loss_func=nn.CrossEntropyLoss(),
            metrics=[accuracy]
        )
        
        # Everything works!
        learn.fit_one_cycle(22, lr=4.7863e-4)
        preds, targs = learn.get_preds(dl=mixed_dls.train)
    """
    learn = Learner(
        dls, model, 
        loss_func=loss_func,
        opt_func=opt_func,
        lr=lr,
        splitter=splitter,
        cbs=cbs,
        metrics=metrics,
        path=path,
        model_dir=model_dir,
        wd=wd,
        wd_bn_bias=wd_bn_bias,
        train_bn=train_bn,
        moms=moms
    )
    
    # Patch get_preds
    learn = patch_learner_get_preds(learn)
    
    return learn


def run_finetune_early_prediction_optimized(
    data,
    model_name: str,
    use_pretrained: bool = True,
    pretrain_cfg = None,
    skip_valid: bool = True,
    lr: float = 1e-4,
    n_epochs: int = 40,
    # Early prediction parameters
    enable_time_masking: bool = True,
    enable_sample_weighting: bool = True,
    masking_prob: float = 0.4,
    early_weight: float = 2.0,
    min_timesteps: int = 6
):
    """
    Fine-tune classifier with early prediction optimization.
    
    Args:
        enable_time_masking: Apply progressive masking during training
        enable_sample_weighting: Weight loss by data availability
        masking_prob: Fraction of batches to mask (0.5 = 50%)
        early_weight: Max weight for early samples (1.5-3.0)
        min_timesteps: Minimum timesteps to keep when masking
    """
    from astra.models.hybrid.training import get_backbone, SkipValidationCallback
    from astra.models.hybrid.mlm import TSTabFusionMLM
    from astra.utils import logger, cfg, clear_mem
    import os
    
    mixed_dls = data["mixed_dls"]
    backbone = get_backbone(data, cfg)
    
    # Load pretrained weights
    if use_pretrained:
        checkpoint_dir = pretrain_cfg.checkpoint_dir if pretrain_cfg else './pretrain_checkpoints'
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
            mlm_model.load_state_dict(checkpoint['model_state_dict'])
            backbone = mlm_model.backbone
            logger.info("✓ Pretrained model loaded")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    # Use standard loss (weighting applied via callback)
    loss_func = LabelSmoothingCrossEntropyFlat()
    
    # Setup callbacks
    cbs = []
    
    if skip_valid:
        cbs.append(SkipValidationCallback())
    
    if enable_time_masking:
        time_mask_cb = ProgressiveTimeMaskingCallback(
            min_timesteps=min_timesteps,
            max_timesteps=114,
            prob=masking_prob
        )
        cbs.append(time_mask_cb)
        logger.info(f"✓ Progressive masking: prob={masking_prob}, min={min_timesteps} steps")
    
    if enable_sample_weighting:
        # Use the weighted loss callback approach
        weight_cb = WeightedLossCallback(loss_func, early_weight=early_weight)
        cbs.append(weight_cb)
        logger.info(f"✓ Sample weighting: early_weight={early_weight}")
    
    # Create learner
    learn = Learner(
        mixed_dls,
        backbone,
        loss_func=loss_func,
        metrics=None,
        cbs=cbs
    )
    
    # Log configuration
    logger.info("="*80)
    logger.info("TRAINING WITH EARLY PREDICTION OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {n_epochs}, LR: {lr}")
    logger.info(f"  Time masking: {enable_time_masking}")
    logger.info(f"  Sample weighting: {enable_sample_weighting}")
    logger.info("="*80)
    
    # Train
    learn.fit_one_cycle(n_epochs, lr)
    
    # Save
    learn.save(model_name)
    logger.info(f"✓ Model saved: {model_name}")
    
    clear_mem()
    return learn