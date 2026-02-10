"""
CLI entry point for the revised training pipeline.

Usage:
    # Standard finetune (with pretrained weights)
    python -m astra.training.train --pretrain --finetune --eval

    # Finetune with early prediction hardening
    python -m astra.training.train --finetune --early-prediction --eval

    # Architecture sweep (Stage 1)
    python -m astra.training.train --sweep-arch --n-arch-trials 30

    # Training HP sweep (Stage 2, requires pretrained checkpoint)
    python -m astra.training.train --sweep-train --n-train-trials 50 --eval

    # Full two-stage sweep
    python -m astra.training.train --sweep-arch --sweep-train --eval

    # Quick test
    python -m astra.training.train --finetune --no-use-pretrained --eval
"""

import argparse
import yaml
from pathlib import Path

from astra.utils import logger, cfg
from astra.data.dataloader import prepare_data_and_dls
from astra.models.hybrid.training import run_pretrain
from astra.models.hybrid.mlm import MLMConfig
from astra.evaluation.predictive_performance import run_eval

from astra.training.finetune import FinetuneConfig, run_finetune_v2
from astra.training.sweep import (
    run_arch_sweep,
    run_training_sweep,
    report_sweep_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASTRA Training Pipeline v2 (transfer learning + HP search)"
    )

    # Pipeline stages
    parser.add_argument("--pretrain", action="store_true", default=False,
                        help="Run MLM pretraining")
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, default=True,
                        help="Run finetuning")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True,
                        help="Run evaluation")

    # Sweep stages
    parser.add_argument("--sweep-arch", action="store_true", default=False,
                        help="Stage 1: Architecture search (no pretraining)")
    parser.add_argument("--sweep-train", action="store_true", default=False,
                        help="Stage 2: Training HP search (with pretraining)")
    parser.add_argument("--n-arch-trials", type=int, default=30,
                        help="Number of architecture search trials")
    parser.add_argument("--n-train-trials", type=int, default=50,
                        help="Number of training HP search trials")
    parser.add_argument("--study-storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")

    # Finetuning options
    parser.add_argument("--use-pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="Load pretrained weights before finetuning")
    parser.add_argument("--early-prediction", action="store_true", default=False,
                        help="Enable Phase 4: progressive time masking + weighted loss")

    # Eval options
    parser.add_argument("--comprehensive-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multicurve", action="store_true", default=False)

    # Config override
    parser.add_argument("--finetune-config", type=str, default=None,
                        help="Path to finetune YAML config (overrides defaults)")

    return parser.parse_args()


def load_finetune_config(config_path: str = None) -> FinetuneConfig:
    """Load FinetuneConfig from YAML file, or return defaults."""
    if config_path is None:
        default_path = Path("configs/finetune.yaml")
        if default_path.exists():
            config_path = str(default_path)
        else:
            return FinetuneConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    ft = raw.get("finetune", {})
    return FinetuneConfig(**{k: v for k, v in ft.items() if hasattr(FinetuneConfig, k)})


def _get_pretrain_cfg() -> MLMConfig:
    """Build MLMConfig from global cfg (matching existing pattern)."""
    pc = cfg["pretrain"]
    return MLMConfig(
        mask_prob_ts=pc["mask_prob_ts"],
        mask_prob_cat_ts=pc["mask_prob_cat_ts"],
        mask_prob_cat=pc["mask_prob_cat"],
        mask_prob_cont=pc["mask_prob_cont"],
        epochs=pc["epochs"],
        lr=pc["lr"],
        warmup_epochs=pc["warmup_epochs"],
        ts_loss_weight=pc["ts_loss_weight"],
        cat_ts_loss_weight=pc.get("cat_ts_loss_weight", 1.0),
        cat_loss_weight=pc["cat_loss_weight"],
        cont_loss_weight=pc["cont_loss_weight"],
        contrastive_weight=pc["contrastive_weight"],
        temperature=pc["temperature"],
        patience=pc["patience"],
        save_best=pc["save_best"],
        checkpoint_dir=f'{pc["checkpoint_dir"]}/{cfg["model_name"]}',
    )


def main():
    args = parse_args()

    # ========================================================================
    # Load data (shared across all stages)
    # ========================================================================
    logger.info("Loading data...")
    data = prepare_data_and_dls(cfg)
    model_name = cfg["model_name"]

    # ========================================================================
    # Stage 0: Pretraining (if no arch sweep — otherwise pretrain after sweep)
    # ========================================================================
    if args.pretrain and not args.sweep_arch:
        pretrain_cfg = _get_pretrain_cfg()
        logger.info("=== Running Pretraining ===")
        pretrain_cfg, _, _ = run_pretrain(data, pretrain_cfg=pretrain_cfg, device="cuda")

    # ========================================================================
    # Stage 1: Architecture sweep (optional)
    # ========================================================================
    if args.sweep_arch:
        logger.info("=== Running Architecture Sweep (Stage 1) ===")
        arch_result = run_arch_sweep(
            data, cfg,
            n_trials=args.n_arch_trials,
            device="cuda",
            storage=args.study_storage,
        )
        report_sweep_results(arch_result["study"])

        # Update cfg with best architecture
        best_arch = arch_result["best_params"]
        cfg["model"]["d_model"] = best_arch["d_model"]
        cfg["model"]["n_layers"] = best_arch["n_layers"]
        cfg["model"]["n_heads"] = best_arch["n_heads"]
        cfg["model"]["fc_mults_1"] = best_arch["fc_mults_1"]
        cfg["model"]["fc_mults_2"] = best_arch["fc_mults_2"]
        cfg["model"]["fc_dropout"] = best_arch["fc_dropout"]
        cfg["model"]["res_dropout"] = best_arch["res_dropout"]

        logger.info(f"Updated model config with best architecture: {best_arch}")

        # Pretrain with the best architecture
        if args.pretrain:
            pretrain_cfg = _get_pretrain_cfg()
            logger.info("=== Pretraining with best architecture ===")
            pretrain_cfg, _, _ = run_pretrain(data, pretrain_cfg=pretrain_cfg, device="cuda")

    # ========================================================================
    # Stage 2: Training HP sweep (optional)
    # ========================================================================
    best_finetune_cfg = None
    if args.sweep_train:
        pretrain_cfg = _get_pretrain_cfg()
        logger.info("=== Running Training HP Sweep (Stage 2) ===")
        train_result = run_training_sweep(
            data, cfg,
            n_trials=args.n_train_trials,
            device="cuda",
            pretrain_cfg=pretrain_cfg,
            pretrain_checkpoint_dir=pretrain_cfg.checkpoint_dir,
            storage=args.study_storage,
        )
        report_sweep_results(train_result["study"])
        best_finetune_cfg = train_result["best_finetune_cfg"]

    # ========================================================================
    # Finetuning
    # ========================================================================
    if args.finetune:
        pretrain_cfg = _get_pretrain_cfg()
        logger.info("=== Running Finetuning (v2) ===")

        if best_finetune_cfg is not None:
            finetune_cfg = best_finetune_cfg
            logger.info("Using best HPs from sweep")
        else:
            finetune_cfg = load_finetune_config(args.finetune_config)

        # Apply CLI overrides
        finetune_cfg.use_pretrained = args.use_pretrained
        finetune_cfg.model_name = model_name
        finetune_cfg.pretrain_checkpoint_dir = pretrain_cfg.checkpoint_dir

        if args.early_prediction:
            finetune_cfg.enable_early_prediction = True

        result = run_finetune_v2(
            data, finetune_cfg,
            pretrain_cfg=pretrain_cfg,
            device="cuda",
        )
        logger.info(f"Finetuning complete. Best AUROC: {result['best_auroc']:.4f}")

    # ========================================================================
    # Evaluation
    # ========================================================================
    if args.eval:
        logger.info("=== Running Evaluation ===")
        results, preds_df = run_eval(
            data, model_name, args.multicurve, args.comprehensive_eval,
        )


if __name__ == "__main__":
    main()
