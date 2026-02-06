"""
Two-stage Optuna hyperparameter search.

Stage 1 (Architecture): Search model shape (d_model, n_layers, etc.)
    without pretraining. Faster per trial.
Stage 2 (Training HPs): Fix architecture, pretrain once, sweep finetuning
    hyperparameters on top of pretrained checkpoint.
"""

import os
from typing import Dict, Any, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from astra.utils import cfg, logger, clear_mem
from astra.models.hybrid.model import TSTabFusionTransformerMultiHot
from astra.models.hybrid.training import get_backbone
from astra.models.hybrid.mlm import MLMConfig
from astra.training.finetune import (
    FinetuneConfig,
    run_finetune_v2,
    create_split_dataloaders,
)
from astra.training.param_groups import set_dropout_rates


# ============================================================================
# STAGE 1: Architecture Search (no pretraining)
# ============================================================================


def _build_backbone_from_trial(data: dict, cfg_dict: dict, trial: optuna.Trial):
    """
    Build a backbone with trial-suggested architecture parameters.

    Returns:
        Backbone model and a dict of the suggested architecture params.
    """
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    n_layers = trial.suggest_categorical("n_layers", [4, 6, 8, 10])
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    fc_mults_1 = trial.suggest_float("fc_mults_1", 0.1, 0.5)
    fc_mults_2 = trial.suggest_float("fc_mults_2", 0.05, 0.3)
    fc_dropout = trial.suggest_float("fc_dropout", 0.1, 0.9)
    res_dropout = trial.suggest_float("res_dropout", 0.0, 0.4)

    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        n_heads = min(n_heads, d_model)
        while d_model % n_heads != 0:
            n_heads -= 1

    backbone = TSTabFusionTransformerMultiHot(
        c_in=data["ts_dls"].vars,
        c_out=2,
        seq_len=data["mixed_dls"].len,
        classes=data["classes"],
        cont_names=data["num_cols"],
        ts_cat_dims=data["ts_cat_dls"].ts_cat_dims,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        fc_dropout=fc_dropout,
        res_dropout=res_dropout,
        fc_mults=(fc_mults_1, fc_mults_2),
        cat_ts_combine="add",
        use_count_normalization=False,
    )

    arch_params = {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "fc_mults_1": fc_mults_1,
        "fc_mults_2": fc_mults_2,
        "fc_dropout": fc_dropout,
        "res_dropout": res_dropout,
    }

    return backbone, arch_params


def arch_objective(trial: optuna.Trial, data: dict, cfg_dict: dict, device: str = "cuda"):
    """
    Stage 1 objective: evaluate an architecture without pretraining.

    Uses a fixed, moderate training schedule (short 3-phase finetune from
    random init) to evaluate the architecture quality.
    """
    # Build model with trial-suggested architecture
    backbone, arch_params = _build_backbone_from_trial(data, cfg_dict, trial)

    # Use a fixed short training schedule for architecture evaluation
    finetune_cfg = FinetuneConfig(
        phase1_epochs=3,
        phase1_lr=1e-3,
        phase2_epochs=8,
        phase2_lr=3e-4,
        phase3_epochs=5,
        phase3_lr=1e-4,
        lr_decay_factor=0.1,
        weight_decay=0.01,
        label_smoothing=0.1,
        patience=5,
        use_pretrained=False,  # No pretraining for arch search
        fc_dropout=arch_params["fc_dropout"],
        res_dropout=arch_params["res_dropout"],
    )

    result = run_finetune_v2(
        data, finetune_cfg,
        device=device,
        trial=trial,
        verbose=False,
    )

    clear_mem()
    return result["best_auroc"]


def run_arch_sweep(
    data: dict,
    cfg_dict: dict,
    n_trials: int = 30,
    device: str = "cuda",
    study_name: str = "astra_arch_search",
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 1: Search for optimal model architecture.

    Args:
        data: Output from prepare_data_and_dls().
        cfg_dict: Global config.
        n_trials: Number of trials.
        device: cuda or cpu.
        study_name: Optuna study name.
        storage: Optuna storage URL (None = in-memory).

    Returns:
        Dict with 'best_params', 'study'.
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Architecture Search")
    logger.info(f"  Trials: {n_trials}")
    logger.info("=" * 80)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: arch_objective(trial, data, cfg_dict, device),
        n_trials=n_trials,
    )

    best = study.best_params
    logger.info("=" * 80)
    logger.info(f"Stage 1 complete. Best AUROC: {study.best_value:.4f}")
    logger.info(f"Best architecture: {best}")
    logger.info("=" * 80)

    return {"best_params": best, "study": study}


# ============================================================================
# STAGE 2: Training HP Search (with pretraining)
# ============================================================================


def training_objective(
    trial: optuna.Trial,
    data: dict,
    cfg_dict: dict,
    device: str = "cuda",
    pretrain_cfg: Optional[MLMConfig] = None,
    pretrain_checkpoint_dir: Optional[str] = None,
):
    """
    Stage 2 objective: sweep finetuning HPs on top of pretrained checkpoint.
    """
    # Suggest finetuning hyperparameters
    finetune_lr = trial.suggest_float("finetune_lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    lr_decay_factor = trial.suggest_float("lr_decay_factor", 0.01, 0.5, log=True)

    phase1_epochs = trial.suggest_int("phase1_epochs", 2, 8)
    phase2_epochs = trial.suggest_int("phase2_epochs", 5, 20)
    phase3_epochs = trial.suggest_int("phase3_epochs", 3, 15)
    phase4_epochs = trial.suggest_int("phase4_epochs", 0, 10)

    # Early prediction params (only relevant if phase4_epochs > 0)
    masking_prob = trial.suggest_float("masking_prob", 0.3, 0.7)
    early_weight = trial.suggest_float("early_weight", 1.0, 3.0)

    finetune_cfg = FinetuneConfig(
        phase1_epochs=phase1_epochs,
        phase1_lr=finetune_lr * 10,  # Head LR is typically higher
        phase2_epochs=phase2_epochs,
        phase2_lr=finetune_lr * 3,
        phase3_epochs=phase3_epochs,
        phase3_lr=finetune_lr,
        enable_early_prediction=phase4_epochs > 0,
        phase4_epochs=phase4_epochs,
        phase4_lr=finetune_lr * 0.5,
        masking_prob=masking_prob,
        early_weight=early_weight,
        lr_decay_factor=lr_decay_factor,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        use_pretrained=True,
        pretrain_checkpoint_dir=pretrain_checkpoint_dir,
        patience=7,
    )

    result = run_finetune_v2(
        data, finetune_cfg,
        pretrain_cfg=pretrain_cfg,
        device=device,
        trial=trial,
        verbose=False,
    )

    clear_mem()
    return result["best_auroc"]


def run_training_sweep(
    data: dict,
    cfg_dict: dict,
    n_trials: int = 50,
    device: str = "cuda",
    pretrain_cfg: Optional[MLMConfig] = None,
    pretrain_checkpoint_dir: Optional[str] = None,
    study_name: str = "astra_training_search",
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 2: Search for optimal finetuning hyperparameters.

    Assumes pretraining has already been done and checkpoint exists.

    Args:
        data: Output from prepare_data_and_dls().
        cfg_dict: Global config.
        n_trials: Number of trials.
        device: cuda or cpu.
        pretrain_cfg: MLM config for weight loading.
        pretrain_checkpoint_dir: Path to pretrained checkpoint.
        study_name: Optuna study name.
        storage: Optuna storage URL (None = in-memory).

    Returns:
        Dict with 'best_params', 'study', 'best_finetune_cfg'.
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: Training HP Search")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Pretrain checkpoint: {pretrain_checkpoint_dir}")
    logger.info("=" * 80)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=8),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: training_objective(
            trial, data, cfg_dict, device, pretrain_cfg, pretrain_checkpoint_dir,
        ),
        n_trials=n_trials,
    )

    best = study.best_params
    logger.info("=" * 80)
    logger.info(f"Stage 2 complete. Best AUROC: {study.best_value:.4f}")
    logger.info(f"Best training HPs: {best}")
    logger.info("=" * 80)

    # Reconstruct best FinetuneConfig
    best_cfg = FinetuneConfig(
        phase1_epochs=best["phase1_epochs"],
        phase1_lr=best["finetune_lr"] * 10,
        phase2_epochs=best["phase2_epochs"],
        phase2_lr=best["finetune_lr"] * 3,
        phase3_epochs=best["phase3_epochs"],
        phase3_lr=best["finetune_lr"],
        enable_early_prediction=best["phase4_epochs"] > 0,
        phase4_epochs=best["phase4_epochs"],
        phase4_lr=best["finetune_lr"] * 0.5,
        masking_prob=best["masking_prob"],
        early_weight=best["early_weight"],
        lr_decay_factor=best["lr_decay_factor"],
        weight_decay=best["weight_decay"],
        label_smoothing=best["label_smoothing"],
        use_pretrained=True,
        pretrain_checkpoint_dir=pretrain_checkpoint_dir,
    )

    return {"best_params": best, "study": study, "best_finetune_cfg": best_cfg}


def report_sweep_results(study: optuna.Study) -> None:
    """Print a summary of the sweep results."""
    logger.info(f"\nStudy: {study.study_name}")
    logger.info(f"  Completed trials: {len(study.trials)}")
    logger.info(f"  Best value: {study.best_value:.4f}")
    logger.info(f"  Best params:")
    for k, v in study.best_params.items():
        logger.info(f"    {k}: {v}")

    # Parameter importance (if enough trials)
    if len(study.trials) >= 5:
        try:
            importances = optuna.importance.get_param_importances(study)
            logger.info(f"  Parameter importance:")
            for k, v in importances.items():
                logger.info(f"    {k}: {v:.3f}")
        except Exception:
            pass
