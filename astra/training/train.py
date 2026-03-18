"""
CLI entry point for the ASTRA training pipeline.

Usage:
    # Full pipeline: pretrain → finetune on full trainval → eval
    python -m astra.training.train --pretrain --finetune --eval

    # Finetune only (using existing pretrained checkpoint, full trainval)
    python -m astra.training.train --finetune --eval

    # Finetune with 80/20 validation split + early stopping
    python -m astra.training.train --finetune --no-skip-valid --eval

    # Finetune with early prediction hardening
    python -m astra.training.train --finetune --early-prediction --eval

    # Full pipeline with HP sweep: pretrain → HPO → retrain on full trainval → eval
    python -m astra.training.train --pretrain --sweep-train --finetune --eval

    # Architecture sweep (Stage 1)
    python -m astra.training.train --sweep-arch --n-arch-trials 30

    # Training HP sweep only (Stage 2, without final retrain)
    python -m astra.training.train --sweep-train --no-finetune --n-train-trials 50

    # Quick test (no pretraining, no eval)
    python -m astra.training.train --finetune --no-use-pretrained --no-eval
"""

import argparse
import logging
from pathlib import Path

from astra.utils import cfg, setup_logging

logger = logging.getLogger(__name__)
from astra.data.caching import prepare_data_and_dls_cached
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
    parser.add_argument("--skip-valid", action=argparse.BooleanOptionalAction, default=True,
                        help="Train on full trainval without validation split (use --no-skip-valid for 80/20 split with early stopping)")
    parser.add_argument("--early-prediction", action="store_true", default=False,
                        help="Enable Phase 4: progressive time masking + weighted loss")

    # Eval options
    parser.add_argument("--comprehensive-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multicurve", action="store_true", default=False)
    parser.add_argument("--active-only", action="store_true", default=False,
                        help="Also run active-only evaluation (patients still in hospital) and generate comparison plots")

    # Calibration
    parser.add_argument("--calibrate", action="store_true", default=False,
                        help="Run posthoc calibration analysis (isotonic/Platt at each timepoint)")

    # Temporal validation
    parser.add_argument("--validate-temporal", action="store_true", default=False,
                        help="Cross-validate temporal eval vs censored-dataloader eval")

    # Logging
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging")

    return parser.parse_args()


def load_finetune_config() -> FinetuneConfig:
    """Build FinetuneConfig from the global cfg['finetune'] section."""
    ft = cfg.get("finetune", {})
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
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # ========================================================================
    # Load data (shared across all stages)
    # ========================================================================
    logger.info("Loading data...")
    data = prepare_data_and_dls_cached(cfg)
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
    sweep_retrained = False
    if args.sweep_train:
        pretrain_cfg = _get_pretrain_cfg()
        logger.info("=== Running Training HP Sweep (Stage 2) ===")

        # When sweep + finetune + skip-valid: retrain on full trainval inside
        # the sweep using the best trial's actual epoch counts
        do_retrain = args.finetune and args.skip_valid
        train_result = run_training_sweep(
            data, cfg,
            n_trials=args.n_train_trials,
            device="cuda",
            pretrain_cfg=pretrain_cfg,
            pretrain_checkpoint_dir=pretrain_cfg.checkpoint_dir,
            storage=args.study_storage,
            retrain_full=do_retrain,
            model_name=model_name if do_retrain else None,
        )
        report_sweep_results(train_result["study"])
        best_finetune_cfg = train_result["best_finetune_cfg"]
        sweep_retrained = do_retrain and train_result.get("retrain_result") is not None

    # ========================================================================
    # Finetuning
    # ========================================================================
    if args.finetune and not sweep_retrained:
        pretrain_cfg = _get_pretrain_cfg()
        logger.info("=== Running Finetuning (v2) ===")

        if best_finetune_cfg is not None:
            finetune_cfg = best_finetune_cfg
            logger.info("Using best HPs from sweep")
        else:
            finetune_cfg = load_finetune_config()

        # Apply CLI overrides
        finetune_cfg.use_pretrained = args.use_pretrained
        finetune_cfg.model_name = model_name
        finetune_cfg.pretrain_checkpoint_dir = pretrain_cfg.checkpoint_dir

        if args.skip_valid:
            finetune_cfg.valid_size = 0.0
            logger.info("--skip-valid: training on full trainval data (valid_size=0.0)")

        if args.early_prediction:
            finetune_cfg.enable_early_prediction = True

        result = run_finetune_v2(
            data, finetune_cfg,
            pretrain_cfg=pretrain_cfg,
            device="cuda",
        )
        if result["best_auroc"] is not None:
            logger.info(f"Finetuning complete. Best AUROC: {result['best_auroc']:.4f}")
        else:
            logger.info("Finetuning complete (full trainval, no validation AUROC)")
    elif sweep_retrained:
        logger.info("Finetuning already completed during sweep (retrain on full trainval)")

    # ========================================================================
    # Evaluation
    # ========================================================================
    if args.eval:
        logger.info("=== Running Evaluation ===")
        results, preds_df = run_eval(
            data, cfg, args.multicurve, args.comprehensive_eval,
            active_only=args.active_only,
        )

    # ========================================================================
    # Posthoc calibration
    # ========================================================================
    if args.calibrate:
        from astra.evaluation.posthoc_calibration import run_posthoc_calibration
        logger.info("=== Running Posthoc Calibration ===")
        cal_summary = run_posthoc_calibration(data, cfg)
        if len(cal_summary) > 0:
            logger.info(f"Calibration summary: {len(cal_summary)} results saved")

    # ========================================================================
    # Temporal validation (cross-check eval methods)
    # ========================================================================
    if args.validate_temporal:
        from astra.evaluation.validate_temporal import run_validation
        logger.info("=== Running Temporal Validation ===")
        run_validation(data, model_name, device="cuda")


if __name__ == "__main__":
    main()
