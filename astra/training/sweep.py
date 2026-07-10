"""
Joint Optuna hyperparameter search.

Searches architecture AND training HPs together in a single study.
Sweep trials use random init (no pretraining) for speed.
After sweep: pretrain with best architecture → finetune on full trainval
with best training HPs.
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


def _best_to_config_sections(best: dict, best_attrs: dict = None) -> dict:
    """
    Convert best trial params into model: and finetune: sections
    that match defaults.yaml structure for easy copy-paste.
    """
    model_section = {
        "d_model": best["d_model"],
        "n_layers": best["n_layers"],
        "n_heads": best["n_heads"],
        "res_dropout": best["res_dropout"],
        "head_pool": best.get("head_pool", "mean_cat"),
    }
    # fc_mults / fc_dropout only present when non-temporal head was searched
    if "fc_mults_1" in best:
        model_section["fc_mults_1"] = best["fc_mults_1"]
        model_section["fc_mults_2"] = best["fc_mults_2"]
    if "fc_dropout" in best:
        model_section["fc_dropout"] = best["fc_dropout"]
    if "temporal_head_dropout" in best:
        model_section["temporal_head_dropout"] = best["temporal_head_dropout"]
    if "temporal_head_mult" in best:
        model_section["temporal_head_mult"] = best["temporal_head_mult"]

    # Compute phase2_unfreeze_from dynamically from n_layers
    n_layers = best["n_layers"]
    n_groups = (n_layers + 1) // 2
    unfreeze_idx = (n_groups // 2) * 2
    unfreeze_end = min(unfreeze_idx + 1, n_layers - 1)

    # Expand base_lr + lr_phase_decay back to per-phase LRs for config output
    _base_lr = best["base_lr"]
    _lr_decay = best["lr_phase_decay"]

    finetune_section = {
        "phase1_epochs": best["phase1_epochs"],
        "phase1_lr": _base_lr,
        "phase2_epochs": best["phase2_epochs"],
        "phase2_lr": round(_base_lr * _lr_decay, 7),
        "phase2_unfreeze_from": f"transformer_{unfreeze_idx}_{unfreeze_end}",
        "phase3_epochs": best["phase3_epochs"],
        "phase3_lr": round(_base_lr * _lr_decay ** 2, 7),
        "phase4_epochs": best["phase4_epochs"],
        "phase4_lr": round(_base_lr * _lr_decay ** 3, 8),
        "enable_early_prediction": best["phase4_epochs"] > 0,
        "masking_prob": best["masking_prob"],
        "early_weight": best["early_weight"],
        "lr_decay_factor": best["lr_decay_factor"],
        "weight_decay": best["weight_decay"],
        "label_smoothing": best["label_smoothing"],
        "pos_weight_factor": best["pos_weight_factor"],
        "time_weighting": best.get("time_weighting", "uniform"),
        "temporal_loss_averaging": best.get("temporal_loss_averaging", "per_sample"),
        "eval_timeframe_weighting": best.get("eval_timeframe_weighting", False),
        "eval_timeframe_weight": best.get("eval_timeframe_weight", 3.0),
        "ranking_loss_weight": best.get("ranking_loss_weight", 0.0),
        "early_weight_factor": best.get("early_weight_factor", 2.0),
        "temporal_crop_prob": best.get("temporal_crop_prob", 0.0),
        "temporal_crop_all_phases": best.get("temporal_crop_prob", 0.0) > 0,
        "val_auroc_weight": best.get("val_auroc_weight", 0.3),
        "val_auprc_weight": best.get("val_auprc_weight", 0.7),
    }

    # Override epoch counts with actual (early-stopped) values if available
    if best_attrs:
        for key, attr in [
            ("phase1_epochs", "phase1_actual_epochs"),
            ("phase2_epochs", "phase2_actual_epochs"),
            ("phase3_epochs", "phase3_actual_epochs"),
            ("phase4_epochs", "phase4_actual_epochs"),
        ]:
            if attr in best_attrs:
                finetune_section[key] = best_attrs[attr]

    return {"model": model_section, "finetune": finetune_section}


def _save_best_callback(study_name: str, save_path: Path):
    """Return an Optuna callback that saves best params to YAML after each trial."""

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if trial.value is None or trial.value < study.best_value:
            return  # Not a new best

        save_path.parent.mkdir(parents=True, exist_ok=True)
        config_sections = _best_to_config_sections(
            study.best_params, study.best_trial.user_attrs)
        result = {
            "study_name": study_name,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "completed_trials": len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]),
            **config_sections,
        }
        with open(save_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved best params (trial {study.best_trial.number}, "
                     f"score {study.best_value:.4f}) → {save_path}")

    return callback


def _study_early_stopping_callback(patience: int, min_delta: float = 0.005):
    """Stop the study if the best score hasn't meaningfully improved.

    Args:
        patience: Number of completed trials without meaningful improvement before stopping.
        min_delta: Minimum improvement over the reference score to reset patience.
    """
    # Track the reference score separately — only updated on meaningful improvements
    state = {"ref_score": None, "ref_trial": None, "trials_since": 0}

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            return

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < patience:
            return

        # Initialize reference from first completed trial
        if state["ref_score"] is None:
            state["ref_score"] = study.best_value
            state["ref_trial"] = study.best_trial.number
            state["trials_since"] = 0
            return

        # Check if current best is a meaningful improvement over reference
        if study.best_value >= state["ref_score"] + min_delta:
            state["ref_score"] = study.best_value
            state["ref_trial"] = study.best_trial.number
            state["trials_since"] = 0
        else:
            state["trials_since"] += 1

        if state["trials_since"] >= patience:
            logger.info(
                f"Study early stopping: no meaningful improvement (>{min_delta}) for "
                f"{state['trials_since']} trials after reference trial "
                f"#{state['ref_trial']} (score {state['ref_score']:.4f}). Stopping."
            )
            study.stop()

    return callback


# ============================================================================
# Joint objective: architecture + training HPs
# ============================================================================


def joint_objective(
    trial: optuna.Trial,
    data: dict,
    cfg_dict: dict,
    device: str = "cuda",
):
    """
    Joint objective: search architecture AND training HPs together.

    No pretraining — trains from random init with a validation split to
    evaluate each configuration. Pretraining is done once for the final
    winner after the sweep completes.

    All search ranges are read from cfg_dict["sweep"]["search_space"].
    """
    ss = cfg_dict["sweep"]["search_space"]

    # --- Architecture parameters ---
    d_model = trial.suggest_categorical("d_model", ss["d_model"])
    n_layers = trial.suggest_categorical("n_layers", ss["n_layers"])
    n_heads = trial.suggest_categorical("n_heads", ss["n_heads"])
    res_dropout = trial.suggest_float("res_dropout", *ss["res_dropout"], step=0.05)
    temporal_head = cfg_dict["model"].get("temporal_head", False)
    if temporal_head:
        # Temporal head replaces the standard MLP head — fc_mults and fc_dropout are unused
        fc_mults_1 = cfg_dict["model"].get("fc_mults_1", 0.3)
        fc_mults_2 = cfg_dict["model"].get("fc_mults_2", 0.1)
        fc_dropout = cfg_dict["model"].get("fc_dropout", 0.5)
        head_pool = "mean_cat"
        temporal_head_dropout = trial.suggest_float(
            "temporal_head_dropout", *ss.get("temporal_head_dropout", [0.0, 0.6]),
            step=0.05,
        )
        temporal_head_mult = trial.suggest_float(
            "temporal_head_mult", *ss.get("temporal_head_mult", [0.25, 2.0]),
            step=0.05,
        )
    else:
        fc_mults_1 = trial.suggest_float("fc_mults_1", *ss["fc_mults_1"], step=0.05)
        fc_mults_2 = trial.suggest_float("fc_mults_2", *ss["fc_mults_2"], step=0.05)
        fc_dropout = trial.suggest_float("fc_dropout", *ss["fc_dropout"], step=0.05)
        head_pool = trial.suggest_categorical("head_pool", ss["head_pool"])
        temporal_head_dropout = cfg_dict["model"].get("temporal_head_dropout", 0.3)
        temporal_head_mult = cfg_dict["model"].get("temporal_head_mult", 0.5)

    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        n_heads = min(n_heads, d_model)
        while d_model % n_heads != 0:
            n_heads -= 1

    # --- Training parameters ---
    # LR parameterized as base_lr × lr_phase_decay^phase (monotonically decreasing)
    base_lr = round(trial.suggest_float("base_lr", *ss["base_lr"], log=True), 6)
    lr_phase_decay = round(trial.suggest_float("lr_phase_decay", *ss["lr_phase_decay"], log=True), 3)
    phase1_lr = base_lr
    phase2_lr = round(base_lr * lr_phase_decay, 7)
    phase3_lr = round(base_lr * lr_phase_decay ** 2, 7)
    phase4_lr = round(base_lr * lr_phase_decay ** 3, 8)

    weight_decay = round(trial.suggest_float("weight_decay", *ss["weight_decay"], log=True), 5)
    label_smoothing = trial.suggest_float("label_smoothing", *ss["label_smoothing"], step=0.01)
    lr_decay_factor = round(trial.suggest_float("lr_decay_factor", *ss["lr_decay_factor"], log=True), 3)

    phase1_epochs = trial.suggest_int("phase1_epochs", *ss["phase1_epochs"])
    phase2_epochs = trial.suggest_int("phase2_epochs", *ss["phase2_epochs"])
    phase3_epochs = trial.suggest_int("phase3_epochs", *ss["phase3_epochs"])
    phase4_epochs = trial.suggest_int("phase4_epochs", *ss["phase4_epochs"])

    pos_weight_factor = trial.suggest_float("pos_weight_factor", *ss["pos_weight_factor"], step=0.05)
    masking_prob = trial.suggest_float("masking_prob", *ss["masking_prob"], step=0.05)
    early_weight = trial.suggest_float("early_weight", *ss["early_weight"], step=0.1)

    # Temporal head time weighting and loss improvements
    if temporal_head:
        time_weighting = trial.suggest_categorical(
            "time_weighting", ["uniform", "early", "late"]
        )
        temporal_loss_averaging = trial.suggest_categorical(
            "temporal_loss_averaging",
            ss.get("temporal_loss_averaging", ["per_sample", "global"]),
        )
        eval_timeframe_weighting = trial.suggest_categorical(
            "eval_timeframe_weighting",
            [str(v).lower() for v in ss.get("eval_timeframe_weighting", [True, False])],
        )
        eval_timeframe_weighting = eval_timeframe_weighting in ("true", "True", True)
        eval_timeframe_weight = (
            trial.suggest_float(
                "eval_timeframe_weight", *ss.get("eval_timeframe_weight", [1.5, 5.0]),
                step=0.5,
            ) if eval_timeframe_weighting else 3.0
        )
        ranking_loss_weight = trial.suggest_float(
            "ranking_loss_weight", *ss.get("ranking_loss_weight", [0.0, 0.3]),
            step=0.05,
        )
        early_weight_factor = trial.suggest_float(
            "early_weight_factor", *ss.get("early_weight_factor", [1.0, 5.0]),
            step=0.5,
        )
        val_auroc_weight = trial.suggest_float(
            "val_auroc_weight", *ss.get("val_auroc_weight", [0.1, 0.9]),
            step=0.1,
        )
        val_auprc_weight = round(1.0 - val_auroc_weight, 2)
    else:
        time_weighting = "uniform"
        temporal_loss_averaging = "per_sample"
        eval_timeframe_weighting = False
        eval_timeframe_weight = 3.0
        ranking_loss_weight = 0.0
        early_weight_factor = 2.0
        val_auroc_weight = 0.3
        val_auprc_weight = 0.7

    # Temporal cropping augmentation
    temporal_crop_prob = trial.suggest_float("temporal_crop_prob", *ss["temporal_crop_prob"], step=0.05)

    # --- Temporarily override global cfg with trial architecture ---
    orig_model_cfg = {k: cfg_dict["model"][k] for k in [
        "d_model", "n_layers", "n_heads", "fc_mults_1", "fc_mults_2",
        "fc_dropout", "res_dropout", "temporal_head_dropout", "temporal_head_mult",
    ] if k in cfg_dict["model"]}
    orig_head_pool = cfg_dict["model"].get("head_pool")

    try:
        cfg_dict["model"]["d_model"] = d_model
        cfg_dict["model"]["n_layers"] = n_layers
        cfg_dict["model"]["n_heads"] = n_heads
        if not temporal_head:
            cfg_dict["model"]["fc_mults_1"] = fc_mults_1
            cfg_dict["model"]["fc_mults_2"] = fc_mults_2
            cfg_dict["model"]["fc_dropout"] = fc_dropout
        cfg_dict["model"]["res_dropout"] = res_dropout
        cfg_dict["model"]["head_pool"] = head_pool
        cfg_dict["model"]["temporal_head_dropout"] = temporal_head_dropout
        cfg_dict["model"]["temporal_head_mult"] = temporal_head_mult

        # Compute phase2_unfreeze_from dynamically: unfreeze top ~half of layers
        # Layer groups are pairs: transformer_0_1, transformer_2_3, ...
        # For n_layers=8: 4 groups, unfreeze from transformer_4_5 (top 2 groups)
        # For n_layers=12: 6 groups, unfreeze from transformer_6_7 (top 3 groups)
        n_groups = (n_layers + 1) // 2  # number of transformer pair-groups
        unfreeze_idx = (n_groups // 2) * 2  # start of top-half pair
        unfreeze_end = min(unfreeze_idx + 1, n_layers - 1)
        phase2_unfreeze = f"transformer_{unfreeze_idx}_{unfreeze_end}"

        finetune_cfg = FinetuneConfig(
            phase1_epochs=phase1_epochs,
            phase1_lr=phase1_lr,
            phase2_epochs=phase2_epochs,
            phase2_lr=phase2_lr,
            phase2_unfreeze_from=phase2_unfreeze,
            phase3_epochs=phase3_epochs,
            phase3_lr=phase3_lr,
            enable_early_prediction=phase4_epochs > 0,
            phase4_epochs=phase4_epochs,
            phase4_lr=phase4_lr,
            masking_prob=masking_prob,
            early_weight=early_weight,
            lr_decay_factor=lr_decay_factor,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            pos_weight_factor=pos_weight_factor,
            fc_dropout=fc_dropout,
            res_dropout=res_dropout,
            use_pretrained=False,  # No pretraining during sweep
            patience=7,
            time_weighting=time_weighting,
            temporal_loss_averaging=temporal_loss_averaging,
            eval_timeframe_weighting=eval_timeframe_weighting,
            eval_timeframe_weight=eval_timeframe_weight,
            ranking_loss_weight=ranking_loss_weight,
            early_weight_factor=early_weight_factor,
            temporal_crop_prob=temporal_crop_prob,
            temporal_crop_all_phases=True,
            val_auroc_weight=val_auroc_weight,
            val_auprc_weight=val_auprc_weight,
        )

        result = run_finetune_v2(
            data, finetune_cfg,
            device=device,
            trial=trial,
            verbose=False,
        )

        # Record actual epoch counts per phase (for post-sweep full trainval retrain)
        tracker = result["tracker"]
        for phase_key, attr_name in [
            ("phase1_head", "phase1_actual_epochs"),
            ("phase2_partial", "phase2_actual_epochs"),
            ("phase3_full", "phase3_actual_epochs"),
            ("phase4_early", "phase4_actual_epochs"),
        ]:
            actual = len(tracker.get(f"{phase_key}/train_loss"))
            trial.set_user_attr(attr_name, actual)

    finally:
        # Restore original cfg
        for k, v in orig_model_cfg.items():
            cfg_dict["model"][k] = v
        if orig_head_pool is not None:
            cfg_dict["model"]["head_pool"] = orig_head_pool

    clear_mem()
    return result["best_score"]


def _build_best_finetune_cfg(best: dict, pretrain_checkpoint_dir: str = None) -> FinetuneConfig:
    """Reconstruct FinetuneConfig from best trial params."""
    # Compute phase2_unfreeze_from based on best n_layers
    n_layers = best["n_layers"]
    n_groups = (n_layers + 1) // 2
    unfreeze_idx = (n_groups // 2) * 2
    unfreeze_end = min(unfreeze_idx + 1, n_layers - 1)
    phase2_unfreeze = f"transformer_{unfreeze_idx}_{unfreeze_end}"

    # Expand base_lr + lr_phase_decay back to per-phase LRs
    _base_lr = best["base_lr"]
    _lr_decay = best["lr_phase_decay"]

    return FinetuneConfig(
        phase1_epochs=best["phase1_epochs"],
        phase1_lr=_base_lr,
        phase2_epochs=best["phase2_epochs"],
        phase2_lr=round(_base_lr * _lr_decay, 7),
        phase2_unfreeze_from=phase2_unfreeze,
        phase3_epochs=best["phase3_epochs"],
        phase3_lr=round(_base_lr * _lr_decay ** 2, 7),
        enable_early_prediction=best["phase4_epochs"] > 0,
        phase4_epochs=best["phase4_epochs"],
        phase4_lr=round(_base_lr * _lr_decay ** 3, 8),
        masking_prob=best["masking_prob"],
        early_weight=best["early_weight"],
        lr_decay_factor=best["lr_decay_factor"],
        weight_decay=best["weight_decay"],
        label_smoothing=best["label_smoothing"],
        pos_weight_factor=best["pos_weight_factor"],
        fc_dropout=best.get("fc_dropout"),
        res_dropout=best["res_dropout"],
        use_pretrained=True,  # Final retrain uses pretrained weights
        pretrain_checkpoint_dir=pretrain_checkpoint_dir,
        time_weighting=best.get("time_weighting", "uniform"),
        early_weight_factor=best.get("early_weight_factor", 2.0),
        temporal_loss_averaging=best.get("temporal_loss_averaging", "per_sample"),
        eval_timeframe_weighting=best.get("eval_timeframe_weighting", False),
        eval_timeframe_weight=best.get("eval_timeframe_weight", 3.0),
        ranking_loss_weight=best.get("ranking_loss_weight", 0.0),
        temporal_crop_prob=best.get("temporal_crop_prob", 0.0),
        temporal_crop_all_phases=best.get("temporal_crop_prob", 0.0) > 0,
        val_auroc_weight=best.get("val_auroc_weight", 0.3),
        val_auprc_weight=best.get("val_auprc_weight", 0.7),
    )


def run_sweep(
    data: dict,
    cfg_dict: dict,
    n_trials: int = 80,
    device: str = "cuda",
    study_name: str = "astra_joint_sweep",
    storage: Optional[str] = None,
    retrain_full: bool = False,
    model_name: Optional[str] = None,
    pretrain_cfg: Optional[MLMConfig] = None,
) -> Dict[str, Any]:
    """
    Joint architecture + training HP search.

    Sweep trials train from random init (no pretraining) for speed.
    If retrain_full=True: pretrain with best architecture, then finetune
    on full trainval with best training HPs.

    Args:
        data: Output from prepare_data_and_dls().
        cfg_dict: Global config.
        n_trials: Number of trials.
        device: cuda or cpu.
        study_name: Optuna study name.
        storage: Optuna storage URL (None = in-memory).
        retrain_full: If True, pretrain + retrain on full trainval with best HPs.
        model_name: Model name for the retrained model (required if retrain_full).
        pretrain_cfg: MLM config for pretraining the final model.

    Returns:
        Dict with 'best_params', 'study', 'best_finetune_cfg', and
        optionally 'retrain_result' if retrain_full=True.
    """
    logger.info("=" * 80)
    logger.info("JOINT HP SWEEP: Architecture + Training")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  No pretraining during sweep (random init)")
    logger.info("=" * 80)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=8),
        load_if_exists=True,
    )

    # Seed first trial with current defaults.yaml values as a known-good baseline.
    # This ensures the sweep result is at least as good as the manual config.
    if len(study.trials) == 0:
        model_cfg = cfg_dict.get("model", {})
        ft_cfg = cfg_dict.get("finetune", {})
        seed_params = {
            # Architecture
            "d_model": model_cfg.get("d_model", 64),
            "n_layers": model_cfg.get("n_layers", 8),
            "n_heads": model_cfg.get("n_heads", 8),
            "res_dropout": model_cfg.get("res_dropout", 0.22),
            # Training (LR parameterized as base_lr × decay^phase)
            "base_lr": ft_cfg.get("phase1_lr", 1e-3),
            "lr_phase_decay": round(
                (ft_cfg.get("phase4_lr", 5e-5) / ft_cfg.get("phase1_lr", 1e-3)) ** (1/3),
                3,
            ),  # geometric mean decay from phase1→phase4
            "phase1_epochs": ft_cfg.get("phase1_epochs", 5),
            "phase2_epochs": ft_cfg.get("phase2_epochs", 12),
            "phase3_epochs": ft_cfg.get("phase3_epochs", 16),
            "phase4_epochs": ft_cfg.get("phase4_epochs", 8),
            "weight_decay": ft_cfg.get("weight_decay", 0.01),
            "label_smoothing": ft_cfg.get("label_smoothing", 0.1),
            "lr_decay_factor": ft_cfg.get("lr_decay_factor", 0.1),
            "masking_prob": ft_cfg.get("masking_prob", 0.8),
            "early_weight": ft_cfg.get("early_weight", 2.0),
            "pos_weight_factor": ft_cfg.get("pos_weight_factor", 0.0),
            "temporal_crop_prob": ft_cfg.get("temporal_crop_prob", 0.0),
        }
        # head_pool, fc_mults, fc_dropout only searched for non-temporal head
        if not model_cfg.get("temporal_head", False):
            seed_params["fc_mults_1"] = model_cfg.get("fc_mults_1", 0.3)
            seed_params["fc_mults_2"] = model_cfg.get("fc_mults_2", 0.1)
            seed_params["fc_dropout"] = model_cfg.get("fc_dropout", 0.75)
            seed_params["head_pool"] = model_cfg.get("head_pool", "mean_cat")
        else:
            seed_params["time_weighting"] = ft_cfg.get("time_weighting", "uniform")
            seed_params["temporal_head_dropout"] = model_cfg.get("temporal_head_dropout", 0.3)
            seed_params["temporal_head_mult"] = model_cfg.get("temporal_head_mult", 0.5)
            seed_params["temporal_loss_averaging"] = ft_cfg.get("temporal_loss_averaging", "per_sample")
            seed_params["eval_timeframe_weighting"] = str(ft_cfg.get("eval_timeframe_weighting", False)).lower()
            if ft_cfg.get("eval_timeframe_weighting", False):
                seed_params["eval_timeframe_weight"] = ft_cfg.get("eval_timeframe_weight", 3.0)
            seed_params["ranking_loss_weight"] = ft_cfg.get("ranking_loss_weight", 0.0)
            seed_params["early_weight_factor"] = ft_cfg.get("early_weight_factor", 2.0)
            seed_params["val_auroc_weight"] = ft_cfg.get("val_auroc_weight", 0.3)
        study.enqueue_trial(seed_params)
        logger.info("Enqueued seed trial with defaults.yaml hyperparameters")

    study_patience = cfg_dict.get("sweep", {}).get("study_patience", 0)
    study_min_delta = cfg_dict.get("sweep", {}).get("study_min_delta", 0.005)
    save_path = SWEEP_RESULTS_DIR / f"{study_name}_best.yaml"
    callbacks = [_save_best_callback(study_name, save_path)]
    if study_patience > 0:
        callbacks.append(_study_early_stopping_callback(study_patience, min_delta=study_min_delta))
        logger.info(f"  Study early stopping: patience={study_patience} trials, min_delta={study_min_delta}")
    study.optimize(
        lambda trial: joint_objective(trial, data, cfg_dict, device),
        n_trials=n_trials,
        callbacks=callbacks,
        catch=(Exception,),  # Don't abort sweep on individual trial failures
    )

    best = study.best_params
    logger.info("=" * 80)
    logger.info(f"Sweep complete. Best score (AUROC+AUPRC): {study.best_value:.4f}")
    logger.info(f"Best params: {best}")
    logger.info("=" * 80)

    # Apply best architecture to global config
    cfg_dict["model"]["d_model"] = best["d_model"]
    cfg_dict["model"]["n_layers"] = best["n_layers"]
    cfg_dict["model"]["n_heads"] = best["n_heads"]
    if "fc_mults_1" in best:
        cfg_dict["model"]["fc_mults_1"] = best["fc_mults_1"]
        cfg_dict["model"]["fc_mults_2"] = best["fc_mults_2"]
    if "fc_dropout" in best:
        cfg_dict["model"]["fc_dropout"] = best["fc_dropout"]
    cfg_dict["model"]["res_dropout"] = best["res_dropout"]
    if "head_pool" in best:
        cfg_dict["model"]["head_pool"] = best["head_pool"]
    if "temporal_head_dropout" in best:
        cfg_dict["model"]["temporal_head_dropout"] = best["temporal_head_dropout"]
    if "temporal_head_mult" in best:
        cfg_dict["model"]["temporal_head_mult"] = best["temporal_head_mult"]
    logger.info(f"Updated global model config with best architecture")

    best_cfg = _build_best_finetune_cfg(best)

    retrain_result = None
    if retrain_full:
        # --- Pretrain with best architecture ---
        if pretrain_cfg is not None:
            from astra.models.hybrid.training import run_pretrain
            logger.info("=" * 80)
            logger.info("PRETRAINING with best architecture")
            logger.info("=" * 80)
            pretrain_cfg, _, _ = run_pretrain(
                data, pretrain_cfg=pretrain_cfg, device=device,
            )
            best_cfg.pretrain_checkpoint_dir = pretrain_cfg.checkpoint_dir

        # --- Retrain on full trainval ---
        logger.info("=" * 80)
        logger.info("FINAL RETRAIN: Full trainval with best HPs")
        logger.info("=" * 80)

        # Use the sweep's suggested epoch counts as budgets (not early-stopped
        # counts, which were calibrated for random init on 80% data and would
        # likely undertrain when retraining with pretrained weights on 100% data)
        logger.info(f"  Using suggested epoch budgets from best trial: "
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

    # Save final config-ready YAML (with actual epoch counts if retrained)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_attrs = study.best_trial.user_attrs if retrain_full else {}
    config_sections = _best_to_config_sections(best, best_attrs)
    final_result = {
        "study_name": study_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "completed_trials": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        **config_sections,
    }
    with open(save_path, "w") as f:
        yaml.dump(final_result, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Final sweep results saved → {save_path}")

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
            logger.debug("Could not compute Optuna parameter importances; skipping")
