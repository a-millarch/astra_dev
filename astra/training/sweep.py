"""
Two-stage Optuna hyperparameter search.

Stage 1 (Architecture): Search model shape (d_model, n_layers, etc.)
    without pretraining. Faster per trial.
Stage 2 (Training HPs): Fix architecture, pretrain once, sweep finetuning
    hyperparameters on top of pretrained checkpoint.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from astra.utils import cfg, clear_mem
from astra.models.hybrid.model import TSTabFusionTransformerMultiHot
from astra.models.hybrid.training import get_backbone
from astra.models.hybrid.mlm import MLMConfig
from astra.training.finetune import (
    FinetuneConfig,
    run_finetune_v2,
    create_split_dataloaders,
)
from astra.training.param_groups import set_dropout_rates

logger = logging.getLogger(__name__)


SWEEP_RESULTS_DIR = Path("configs/sweep_results")


def _save_best_callback(study_name: str, save_path: Path):
    """Return an Optuna callback that saves best params to YAML after each trial."""

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if trial.value is None or trial.value < study.best_value:
            return  # Not a new best

        save_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "study_name": study_name,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "completed_trials": len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]),
        }
        with open(save_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False)
        logger.info(f"Saved best params (trial {study.best_trial.number}, "
                     f"score {study.best_value:.4f}) → {save_path}")

    return callback


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
    n_layers = trial.suggest_categorical("n_layers", [2, 4, 6, 8])
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    fc_mults_1 = trial.suggest_float("fc_mults_1", 0.1, 0.5)
    fc_mults_2 = trial.suggest_float("fc_mults_2", 0.05, 0.3)
    fc_dropout = trial.suggest_float("fc_dropout", 0.1, 0.9)
    res_dropout = trial.suggest_float("res_dropout", 0.0, 0.4)
    head_pool = trial.suggest_categorical("head_pool", ["flatten", "mean_cat"])

    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        n_heads = min(n_heads, d_model)
        while d_model % n_heads != 0:
            n_heads -= 1

    backbone = TSTabFusionTransformerMultiHot(
        c_in=data["c_in"],
        c_out=2,
        seq_len=data["seq_len"],
        classes=data["classes"],
        cont_names=data["num_cols"],
        ts_cat_dims=data["ts_cat_dims"],
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        fc_dropout=fc_dropout,
        res_dropout=res_dropout,
        fc_mults=(fc_mults_1, fc_mults_2),
        cat_ts_combine="add",
        use_count_normalization=False,
        head_pool=head_pool,
    )

    arch_params = {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "fc_mults_1": fc_mults_1,
        "fc_mults_2": fc_mults_2,
        "fc_dropout": fc_dropout,
        "res_dropout": res_dropout,
        "head_pool": head_pool,
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
    return result["best_score"]


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

    save_path = SWEEP_RESULTS_DIR / f"{study_name}_best.yaml"
    study.optimize(
        lambda trial: arch_objective(trial, data, cfg_dict, device),
        n_trials=n_trials,
        callbacks=[_save_best_callback(study_name, save_path)],
    )

    best = study.best_params
    logger.info("=" * 80)
    logger.info(f"Stage 1 complete. Best score (AUROC+AUPRC): {study.best_value:.4f}")
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

    # Class imbalance handling
    pos_weight_factor = trial.suggest_float("pos_weight_factor", 0.3, 1.0, log=True)

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
        pos_weight_factor=pos_weight_factor,
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

    # Record actual epoch counts per phase (for post-HPO full trainval retrain)
    tracker = result["tracker"]
    for phase_key, attr_name, cfg_field in [
        ("phase1_head", "phase1_actual_epochs", "phase1_epochs"),
        ("phase2_partial", "phase2_actual_epochs", "phase2_epochs"),
        ("phase3_full", "phase3_actual_epochs", "phase3_epochs"),
        ("phase4_early", "phase4_actual_epochs", "phase4_epochs"),
    ]:
        actual = len(tracker.get(f"{phase_key}/train_loss"))
        trial.set_user_attr(attr_name, actual)

    clear_mem()
    return result["best_score"]


def run_training_sweep(
    data: dict,
    cfg_dict: dict,
    n_trials: int = 50,
    device: str = "cuda",
    pretrain_cfg: Optional[MLMConfig] = None,
    pretrain_checkpoint_dir: Optional[str] = None,
    study_name: str = "astra_training_search",
    storage: Optional[str] = None,
    retrain_full: bool = False,
    model_name: Optional[str] = None,
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
        retrain_full: If True, retrain on full trainval with best HPs after sweep.
        model_name: Model name for the retrained model (required if retrain_full=True).

    Returns:
        Dict with 'best_params', 'study', 'best_finetune_cfg', and
        optionally 'retrain_result' if retrain_full=True.
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

    save_path = SWEEP_RESULTS_DIR / f"{study_name}_best.yaml"
    study.optimize(
        lambda trial: training_objective(
            trial, data, cfg_dict, device, pretrain_cfg, pretrain_checkpoint_dir,
        ),
        n_trials=n_trials,
        callbacks=[_save_best_callback(study_name, save_path)],
    )

    best = study.best_params
    logger.info("=" * 80)
    logger.info(f"Stage 2 complete. Best score (AUROC+AUPRC): {study.best_value:.4f}")
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
        pos_weight_factor=best["pos_weight_factor"],
        use_pretrained=True,
        pretrain_checkpoint_dir=pretrain_checkpoint_dir,
    )

    retrain_result = None
    if retrain_full:
        logger.info("=" * 80)
        logger.info("FINAL RETRAIN: Training on full trainval with best HPs")
        logger.info("=" * 80)

        # Use actual epoch counts from the best trial (accounts for early stopping)
        best_attrs = study.best_trial.user_attrs
        best_cfg.phase1_epochs = best_attrs.get("phase1_actual_epochs", best_cfg.phase1_epochs)
        best_cfg.phase2_epochs = best_attrs.get("phase2_actual_epochs", best_cfg.phase2_epochs)
        best_cfg.phase3_epochs = best_attrs.get("phase3_actual_epochs", best_cfg.phase3_epochs)
        best_cfg.phase4_epochs = best_attrs.get("phase4_actual_epochs", best_cfg.phase4_epochs)
        logger.info(f"  Using epoch counts from best trial: "
                     f"P1={best_cfg.phase1_epochs}, P2={best_cfg.phase2_epochs}, "
                     f"P3={best_cfg.phase3_epochs}, P4={best_cfg.phase4_epochs}")

        best_cfg.valid_size = 0.0
        best_cfg.model_name = model_name or cfg_dict.get("model_name", "")

        retrain_result = run_finetune_v2(
            data, best_cfg,
            pretrain_cfg=pretrain_cfg,
            device=device,
            trial=None,
        )
        logger.info("Full trainval retrain complete")

    return {
        "best_params": best,
        "study": study,
        "best_finetune_cfg": best_cfg,
        "retrain_result": retrain_result,
    }


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
