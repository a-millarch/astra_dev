#!/usr/bin/env python
"""
DIAGNOSTIC: Inference-time ablation — zero out temporal channels on a trained model.

If zeroing temporal channels at inference IMPROVES performance on a model
trained WITH temporal features, it confirms they are actively harmful.

Run on Azure:
    python -m scripts.diagnostics.diag_ablation
"""
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from astra.utils import cfg
from astra.data.dataloader import prepare_data_and_dls

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (tuple, list)):
        return type(obj)(to_device(item, device) for item in obj)
    return obj


def evaluate(model, holdout_dls, device, temporal_indices=None, zero_temporal=False):
    """Run inference, optionally zeroing temporal channels."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in holdout_dls.train:
            inputs, targets = batch
            inputs = to_device(inputs, device)

            if zero_temporal and temporal_indices:
                # Clone and zero temporal channels
                x_ts = inputs[0].clone()
                for idx in temporal_indices:
                    x_ts[:, idx, :] = 0.0
                if len(inputs) == 3:
                    inputs = (x_ts, inputs[1], inputs[2])
                elif len(inputs) == 2:
                    inputs = (x_ts, inputs[1])

            logits = model(inputs)

            # Handle different output formats
            if logits.dim() == 2 and logits.shape[1] == 2:
                # Dense head: [batch, 2] → softmax
                probs = torch.softmax(logits, dim=1)[:, 1]
            elif logits.dim() == 2 and logits.shape[1] > 2:
                # Temporal head: [batch, seq_len] → use last valid position
                probs = torch.sigmoid(logits[:, -1])
            else:
                probs = torch.sigmoid(logits.squeeze())

            all_preds.append(probs.cpu().numpy())
            all_targets.append(
                targets.cpu().numpy() if isinstance(targets, torch.Tensor)
                else np.array(targets)
            )

    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model checkpoint")
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info("DIAGNOSTIC: Temporal Channel Ablation at Inference")
    logger.info("=" * 72)

    data = prepare_data_and_dls(cfg)
    ts_channel_names = data["ts_channel_names"]
    tf_features = cfg.get("temporal_features", {}).get("features", [])
    temporal_indices = [i for i, n in enumerate(ts_channel_names) if n in tf_features]

    if not temporal_indices:
        logger.info("No temporal channels found — nothing to ablate.")
        return

    logger.info(f"Temporal channels to ablate: "
                f"{[(i, ts_channel_names[i]) for i in temporal_indices]}")

    # Build model
    from astra.models.hybrid.training import get_backbone
    model_cfg = cfg.get("model", {})
    model = get_backbone(
        data, cfg,
        temporal_head=model_cfg.get("temporal_head", False),
        causal=model_cfg.get("causal", False),
        temporal_channel_idx=data.get("temporal_channel_idx"),
        exclude_channel_indices=data.get("exclude_channel_indices", []),
    )

    # Load weights
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded model from: {args.model_path}")
    else:
        # Try to find latest finetuned model
        model_name = cfg.get("model_name", "model")
        search_dirs = [
            Path(f"models/{model_name}"),
            Path(f"checkpoints/{model_name}"),
            Path("models"),
        ]
        for d in search_dirs:
            if d.exists():
                candidates = list(d.glob("*finetune*.pt")) + list(d.glob("*best*.pt"))
                if candidates:
                    checkpoint_path = sorted(candidates)[-1]
                    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                            weights_only=False)
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    logger.info(f"Loaded model from: {checkpoint_path}")
                    break
        else:
            logger.warning("No checkpoint found — using random weights (results not meaningful)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    holdout_dls = data["holdout_mixed_dls"]

    # --- Evaluate with temporal features ---
    preds_with, targets = evaluate(
        model, holdout_dls, device,
        temporal_indices=temporal_indices, zero_temporal=False
    )
    auroc_with = roc_auc_score(targets, preds_with)
    auprc_with = average_precision_score(targets, preds_with)

    # --- Evaluate WITHOUT temporal features ---
    preds_without, _ = evaluate(
        model, holdout_dls, device,
        temporal_indices=temporal_indices, zero_temporal=True
    )
    auroc_without = roc_auc_score(targets, preds_without)
    auprc_without = average_precision_score(targets, preds_without)

    # --- Results ---
    logger.info(f"\n{'':=<72}")
    logger.info(f"RESULTS (holdout set)")
    logger.info(f"{'':=<72}")
    logger.info(f"  {'Condition':<30s} {'AUROC':>10s} {'AUPRC':>10s}")
    logger.info(f"  {'-'*50}")
    logger.info(f"  {'With temporal features':<30s} {auroc_with:>10.4f} {auprc_with:>10.4f}")
    logger.info(f"  {'Temporal channels zeroed':<30s} {auroc_without:>10.4f} {auprc_without:>10.4f}")
    logger.info(f"  {'Difference':<30s} {auroc_without - auroc_with:>+10.4f} "
                f"{auprc_without - auprc_with:>+10.4f}")

    if auroc_without > auroc_with + 0.005:
        logger.info(f"\n  ** CONFIRMED: Temporal features are ACTIVELY HARMFUL at inference.")
        logger.info(f"     The model improves by +{auroc_without-auroc_with:.4f} AUROC "
                    f"when they are zeroed.")
    elif abs(auroc_without - auroc_with) <= 0.005:
        logger.info(f"\n  Temporal features have NEGLIGIBLE effect at inference.")
        logger.info(f"  The damage may be during TRAINING (optimization landscape).")
    else:
        logger.info(f"\n  Temporal features provide some signal at inference.")
        logger.info(f"  Consider: the model learned to depend on them during training.")

    # --- Prediction distribution comparison ---
    logger.info(f"\n  Prediction distribution:")
    logger.info(f"    With temporal — mean: {preds_with.mean():.4f}, "
                f"std: {preds_with.std():.4f}")
    logger.info(f"    Without temporal — mean: {preds_without.mean():.4f}, "
                f"std: {preds_without.std():.4f}")
    logger.info(f"    Correlation: {np.corrcoef(preds_with, preds_without)[0,1]:.4f}")

    logger.info("=" * 72)


if __name__ == "__main__":
    main()
