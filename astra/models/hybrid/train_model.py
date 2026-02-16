import argparse

from astra.utils import logger, cfg, logger
from astra.evaluation.predictive_performance import run_eval

from astra.data.caching import prepare_data_and_dls_cached
from astra.data.dataloader import save_normalization_artifacts
from astra.models.hybrid.training import run_pretrain, run_finetune, run_finetune_early_prediction_optimized
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
    parser.add_argument('--alternative-fine-tune', action='store_true', default=False)
    parser.add_argument('--skip-valid',  action=argparse.BooleanOptionalAction, default=True)
    
    # 3. Eval Flags (Default: Comprehensive is OFF, Multicurve is OFF)
    parser.add_argument('--comprehensive-eval', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--multicurve', action='store_true', default=False)
    
    return parser.parse_args()

def main():
    args = parse_args()
    data = prepare_data_and_dls_cached(cfg)
    pretrain_params = dict(cfg["pretrain"])
    pretrain_params["checkpoint_dir"] = f'{pretrain_params.get("checkpoint_dir", "./pretrain_checkpoints")}/{cfg["model_name"]}'
    pretrain_cfg = MLMConfig(**pretrain_params)
    
    if args.pretrain:
        logger.info("=== Running Pretraining ===")
        pretrain_cfg, _, _ = run_pretrain(data, pretrain_cfg = pretrain_cfg, device='cuda')
    
    if args.finetune:
        logger.info("=== Running Fine-tuning ===")
        
        if args.alternative_fine_tune:
            logger.info(">> Alternative method")
            learn = run_finetune_early_prediction_optimized(
                                                data,
                                                model_name=cfg["model_name"],
                                                use_pretrained=args.use_pretrained,
                                                pretrain_cfg=pretrain_cfg,
                                                lr=4.7863e-4,
                                                n_epochs=22,
                                                # New parameters:
                                                enable_time_masking=True,      # ← Progressive masking
                                                enable_sample_weighting=True,  # ← Weight samples by data
                                                masking_prob=0.5,              # ← 50% of batches get masked
                                                early_weight=2.0,              # ← 2x penalty for early errors
                                                min_timesteps=2                # ← Keep at least 1 hour
                                                )        
        else: 
            run_finetune(data, cfg["model_name"], args.use_pretrained, pretrain_cfg,
                    args.skip_valid, args.lr, args.finetune_epochs)
    

    if args.eval:
        logger.info("=== Running Evaluation ===")
        results, preds_df = run_eval(data, cfg["model_name"], args.multicurve,args.comprehensive_eval)
       # if args.comprehensive_eval:
        #    logger.info(f"Generated {len(results[0])} time points with CIs")

if __name__ == "__main__":
    main()
