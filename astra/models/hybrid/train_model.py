import argparse

from astra.utils import logger, cfg
from astra.evaluation.predictive_performance import run_eval

from astra.data.caching import prepare_data_and_dls_cached
from astra.data.dataloader import save_normalization_artifacts, save_deployment_bundle
from astra.models.hybrid.training import run_pretrain
from astra.training.finetune import run_finetune_v2, FinetuneConfig
from astra.models.hybrid.mlm import MLMConfig


def parse_args():
    parser = argparse.ArgumentParser(description='MLM Pretraining + Fine-tuning Pipeline')

    # 1. Pipeline Stages (Defaults: Only Finetune and Eval are on)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--finetune', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=True)

    # 2. Logic Flags (Default: Load pretrained model if finetuning)
    parser.add_argument('--use-pretrained', action=argparse.BooleanOptionalAction, default=True,
                       help='Load pretrained weights before finetuning')
    parser.add_argument('--skip-valid',  action=argparse.BooleanOptionalAction, default=True)

    # 3. Eval Flags (Default: Comprehensive is OFF, Multicurve is OFF)
    parser.add_argument('--comprehensive-eval', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--multicurve', action='store_true', default=False)

    # 4. Per-timestep prediction
    parser.add_argument('--temporal-head', action='store_true', default=False)

    return parser.parse_args()

def main():
    args = parse_args()
    data = prepare_data_and_dls_cached(cfg)
    pretrain_params = dict(cfg["pretrain"])
    pretrain_params["checkpoint_dir"] = f'{pretrain_params.get("checkpoint_dir", "./pretrain_checkpoints")}/{cfg["model_name"]}'
    pretrain_cfg = MLMConfig(**pretrain_params)

    if args.pretrain:
        logger.info("=== Running Pretraining ===")
        pretrain_cfg, _, _ = run_pretrain(data, pretrain_cfg=pretrain_cfg, device='cuda')

    if args.finetune:
        logger.info("=== Running Fine-tuning ===")

        finetune_cfg = FinetuneConfig(
            model_name=cfg["model_name"],
            use_pretrained=args.use_pretrained,
            temporal_head=args.temporal_head,
        )

        result = run_finetune_v2(
            data,
            finetune_cfg=finetune_cfg,
            pretrain_cfg=pretrain_cfg,
            device='cuda',
        )
        logger.info(f"Best validation AUROC: {result['best_auroc']:.4f}")

    if args.finetune:
        logger.info("=== Saving deployment bundle ===")
        save_deployment_bundle(data, cfg, cfg["model_name"])

    if args.eval:
        logger.info("=== Running Evaluation ===")
        results, preds_df = run_eval(data, cfg["model_name"], args.multicurve, args.comprehensive_eval)

if __name__ == "__main__":
    main()
