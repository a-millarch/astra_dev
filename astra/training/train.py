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

    # Joint HP sweep → pretrain best arch → retrain full trainval → eval
    python -m astra.training.train --sweep --finetune --eval

    # Sweep only (no final retrain/eval)
    python -m astra.training.train --sweep --no-finetune --no-eval --n-trials 80

    # Quick test (no pretraining, no eval)
    python -m astra.training.train --finetune --no-use-pretrained --no-eval
"""

import argparse
import logging
from pathlib import Path

from astra.utils import cfg, get_cfg, setup_logging, PROJECT_ROOT

logger = logging.getLogger(__name__)
from astra.data.caching import prepare_data_and_dls_cached
from astra.models.hybrid.training import run_pretrain
from astra.models.hybrid.mlm import MLMConfig
from astra.evaluation.predictive_performance import run_eval

from astra.training.finetune import FinetuneConfig, run_finetune_v2
from astra.training.sweep import run_sweep, report_sweep_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASTRA Training Pipeline v2 (transfer learning + HP search)"
    )

    # Config
    parser.add_argument("--config", type=str, default="defaults.yaml",
                        help="Config YAML filename in configs/ dir (default: defaults.yaml)")

    # Pipeline stages
    parser.add_argument("--pretrain", action="store_true", default=False,
                        help="Run MLM pretraining")
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, default=True,
                        help="Run finetuning")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True,
                        help="Run evaluation")

    # Sweep
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="Joint architecture + training HP sweep")
    parser.add_argument("--n-trials", type=int, default=80,
                        help="Number of sweep trials")
    parser.add_argument("--study-storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")

    # Finetuning options
    parser.add_argument("--use-pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="Load pretrained weights before finetuning")
    parser.add_argument("--skip-valid", action=argparse.BooleanOptionalAction, default=True,
                        help="Train on full trainval without validation split (use --no-skip-valid for 80/20 split with early stopping)")
    parser.add_argument("--valid-size", type=float, default=None,
                        help="Validation split fraction (e.g. 0.1). Implies --no-skip-valid.")
    parser.add_argument("--early-prediction", action="store_true", default=False,
                        help="Enable Phase 4: progressive time masking + weighted loss")

    # Eval options
    parser.add_argument("--comprehensive-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multicurve", action="store_true", default=False)
    parser.add_argument("--active-only", action="store_true", default=False,
                        help="Also run active-only evaluation (patients still in hospital) and generate comparison plots")
    parser.add_argument("--trauma-scores", action="store_true", default=False,
                        help="Compute traditional trauma risk scores (RTS, ISS, TRISS) and add as baselines (Azure-only)")
    parser.add_argument("--delong", action="store_true", default=False,
                        help="Run paired DeLong tests between HNN and trauma scores with FDR correction (requires --trauma-scores)")

    # Calibration
    parser.add_argument("--calibrate", action="store_true", default=False,
                        help="Run posthoc calibration analysis (isotonic/Platt at each timepoint)")

    # SHAP
    parser.add_argument("--shap", action="store_true", default=False,
                        help="Run cohort temporal SHAP analysis with visualizations")
    parser.add_argument("--shap-max-patients", type=int, default=20,
                        help="Max holdout patients for temporal SHAP (default: 20)")
    parser.add_argument("--shap-representative", action="store_true", default=False,
                        help="Use stratified representative sampling for temporal SHAP")

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
        checkpoint_dir=str(PROJECT_ROOT / pc["checkpoint_dir"] / cfg["model_name"]),
    )


def main():
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Load config from configs/ dir (always applied, defaults to defaults.yaml)
    import astra.utils as _utils
    _cfg = get_cfg(_utils.PROJECT_ROOT / "configs" / args.config)
    _utils.cfg.clear()
    _utils.cfg.update(_cfg)

    # ========================================================================
    # Load data (shared across all stages)
    # ========================================================================
    logger.info("Loading data...")
    data = prepare_data_and_dls_cached(cfg)
    model_name = cfg["model_name"]

    # ========================================================================
    # Pretraining (standalone, when no sweep)
    # ========================================================================
    if args.pretrain and not args.sweep:
        pretrain_cfg = _get_pretrain_cfg()
        logger.info("=== Running Pretraining ===")
        pretrain_cfg, _, _ = run_pretrain(data, pretrain_cfg=pretrain_cfg, device="cuda")

    # ========================================================================
    # Joint HP sweep (architecture + training HPs, no pretraining during sweep)
    # ========================================================================
    best_finetune_cfg = None
    sweep_retrained = False
    if args.sweep:
        pretrain_cfg = _get_pretrain_cfg()
        logger.info("=== Running Joint HP Sweep ===")

        # When sweep + finetune + skip-valid: pretrain best arch + retrain
        # on full trainval inside the sweep
        do_retrain = args.finetune and args.skip_valid
        sweep_result = run_sweep(
            data, cfg,
            n_trials=args.n_trials,
            device="cuda",
            storage=args.study_storage,
            retrain_full=do_retrain,
            model_name=model_name if do_retrain else None,
            pretrain_cfg=pretrain_cfg if do_retrain else None,
        )
        report_sweep_results(sweep_result["study"])
        best_finetune_cfg = sweep_result["best_finetune_cfg"]
        sweep_retrained = do_retrain and sweep_result.get("retrain_result") is not None

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

        if args.valid_size is not None:
            finetune_cfg.valid_size = args.valid_size
            logger.info(f"--valid-size={args.valid_size}: using {args.valid_size:.0%} validation split")
        elif args.skip_valid:
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
            trauma_scores=args.trauma_scores,
            delong=args.delong,
        )

    # ========================================================================
    # Posthoc calibration
    # ========================================================================
    if args.calibrate:
        from astra.evaluation.posthoc_calibration import run_posthoc_calibration
        logger.info("=== Running Posthoc Calibration ===")
        cal_summary = run_posthoc_calibration(
            data, cfg, save_dir=f'reports/eval/{model_name}/calibration',
        )
        if len(cal_summary) > 0:
            logger.info(f"Calibration summary: {len(cal_summary)} results saved")

    # ========================================================================
    # SHAP analysis
    # ========================================================================
    if args.shap:
        from astra.evaluation.behavior import (
            shap_analysis, visualize_shap_summary,
            run_cohort_temporal_shap_analysis,
        )
        from astra.evaluation.utils import prepare_model
        logger.info("=== Running SHAP Analysis ===")
        model, device = prepare_model(data, cfg)
        model_name = cfg["model_name"]

        # Aggregate SHAP (auto-uses 'mean' for temporal head)
        shap_results = shap_analysis(
            data, model, model_name=model_name,
            visualize=False, max_test_samples=500,
            max_background_samples=1000, density_normalize=True,
        )
        visualize_shap_summary(
            shap_results["shap_results"],
            channel2feature=shap_results["channel2feature"],
            feature_names_cat=shap_results["static_cat_names"],
            feature_names_cont=cfg["dataset"]["num_cols"],
            class_idx=1, max_display=20,
            save_path=f'reports/eval/{model_name}/shap_class1.png',
            density_normalize=True,
        )

        # Per-timeframe cohort temporal SHAP (active-only background)
        run_cohort_temporal_shap_analysis(
            data, model,
            max_patients=args.shap_max_patients,
            max_background_samples=1000,
            save_dir=f'reports/eval/{model_name}/temporal_shap',
            density_normalize=True,
            active_only=True,
            representative=args.shap_representative,
        )

        # Paper-quality summary panel from saved CSV
        from astra.evaluation.shap_paper_figures import figure_shap_summary_panel
        shap_dir = f'reports/eval/{model_name}/temporal_shap'
        csv_path = f'{shap_dir}/cohort_shap_all_features_active_dn.csv'
        pkl_path = f'{shap_dir}/cohort_temporal_shap_results_active_dn.pkl'
        if Path(csv_path).exists():
            figure_shap_summary_panel(
                csv_path=csv_path, save_dir=shap_dir, pickle_path=pkl_path,
            )
        logger.info("SHAP analysis complete")

    # ========================================================================
    # Temporal validation (cross-check eval methods)
    # ========================================================================
    if args.validate_temporal:
        from astra.evaluation.validate_temporal import run_validation
        logger.info("=== Running Temporal Validation ===")
        run_validation(data, model_name, device="cuda")


if __name__ == "__main__":
    main()
