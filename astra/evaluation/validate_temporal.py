"""
Validate temporal (per-timestep) prediction model by cross-checking evaluation methods.

Compares:
  Method A  "TemporalEvaluator"  — full uncensored data, trusts causal mask, picks position
  Method B  "Censored input"     — zeroes out future input data, forward pass, picks position
  Method C  "No causal mask"     — full data with causal mask disabled, picks position

If A ≈ B: causal mask works correctly.
If A >> B: information is leaking from future positions.
If A ≈ C: causal mask has no effect (model is using future info freely).

Usage:
    python -m astra.evaluation.validate_temporal
    python -m astra.evaluation.validate_temporal --temporal-head  # explicit flag
"""

import argparse
import logging
import numpy as np
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

from astra.utils import cfg
from astra.data.caching import prepare_data_and_dls_cached
from astra.models.hybrid.training import get_backbone
from astra.evaluation.utils import prepare_model
from astra.evaluation.predictive_performance import (
    TemporalEvaluator,
    time_to_step,
    step_to_time,
    format_step_label,
    generate_time_thresholds,
)
from astra.evaluation.utils import calculate_roc_auc_ci, calculate_average_precision_ci

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    censor_step: int
    label: str
    # Method A: TemporalEvaluator (full data + causal mask)
    auroc_temporal: Optional[float] = None
    auprc_temporal: Optional[float] = None
    # Method B: Censored input (gold standard)
    auroc_censored: Optional[float] = None
    auprc_censored: Optional[float] = None
    # Method C: No causal mask (leak detector)
    auroc_no_causal: Optional[float] = None
    auprc_no_causal: Optional[float] = None


def _run_temporal_model_on_censored_data(
    model: torch.nn.Module,
    data: dict,
    censor_step: int,
    device: str = "cuda",
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Run temporal model on censored (future-zeroed) input data.

    Iterates over holdout_mixed_dls (same dataloaders as Methods A and C),
    manually zeros future positions in each batch, runs the forward pass,
    and clips predictions per-patient by trajectory_length.

    Returns:
        (y_preds, ys) or (None, ys) if failed
    """
    from astra.training.finetune import _infer_trajectory_lengths_from_batch

    holdout_dls = data["holdout_mixed_dls"]
    all_preds = []
    all_targets = []
    all_traj_lengths = []

    model.eval()
    with torch.no_grad():
        for batch in holdout_dls.train:
            inputs, targets = batch
            inputs = _to_device(inputs, device)

            # Unpack: (x_ts, x_tab, x_ts_cat) or (x_ts, x_tab)
            if isinstance(inputs, (tuple, list)):
                inputs = list(inputs)
                x_ts = inputs[0]

                # Infer trajectory lengths from UNCENSORED data
                traj_lens = _infer_trajectory_lengths_from_batch(x_ts)
                all_traj_lengths.append(traj_lens.cpu().numpy())

                # Censor: zero out future positions in x_ts
                if censor_step < x_ts.shape[2]:
                    x_ts_censored = x_ts.clone()
                    x_ts_censored[:, :, censor_step:] = 0.0
                    inputs[0] = x_ts_censored

                # Censor: zero out future positions in x_ts_cat (if present)
                if len(inputs) >= 3 and inputs[2] is not None:
                    x_ts_cat = inputs[2]
                    if censor_step < x_ts_cat.shape[2]:
                        x_ts_cat_censored = x_ts_cat.clone()
                        x_ts_cat_censored[:, :, censor_step:] = 0.0
                        inputs[2] = x_ts_cat_censored

                inputs = tuple(inputs)

            logits = model(inputs)  # [batch, seq_len]
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds_all = np.concatenate(all_preds, axis=0)  # [n, seq_len]
    ys = np.concatenate(all_targets, axis=0)
    traj_lengths = np.concatenate(all_traj_lengths, axis=0)

    # Extract prediction at min(censor_step, trajectory_length - 1) per patient
    effective_steps = np.minimum(censor_step, traj_lengths - 1)
    effective_steps = np.maximum(effective_steps, 0).astype(int)
    y_preds = preds_all[np.arange(len(preds_all)), effective_steps]

    return y_preds, ys


def _run_temporal_model_no_causal(
    model: torch.nn.Module,
    data: dict,
    censor_step: int,
    traj_lengths: np.ndarray,
    device: str = "cuda",
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Run temporal model with causal mask temporarily disabled.

    This tests whether the causal mask is actually doing anything.
    """
    # Temporarily disable causal masking
    original_causal = model.causal
    original_mask = model.causal_mask

    model.causal = False
    model.causal_mask = None

    holdout_dls = data["holdout_mixed_dls"]
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch in holdout_dls.train:
            inputs, targets = batch
            inputs = _to_device(inputs, device)
            logits = model(inputs)  # [batch, seq_len]
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Restore causal masking
    model.causal = original_causal
    model.causal_mask = original_mask

    preds_all = np.concatenate(all_preds, axis=0)
    ys = np.concatenate(all_targets, axis=0)

    # Extract at censor_step
    if len(traj_lengths) > 0:
        effective_steps = np.minimum(censor_step, traj_lengths - 1)
        effective_steps = np.maximum(effective_steps, 0).astype(int)
    else:
        effective_steps = min(censor_step, preds_all.shape[1] - 1)

    y_preds = preds_all[np.arange(len(preds_all)), effective_steps]
    return y_preds, ys


def _to_device(obj, device):
    """Recursively move tensors to device."""
    if isinstance(obj, torch.Tensor):
        t = obj.to(device)
        if type(t) is not torch.Tensor:
            t = t.as_subclass(torch.Tensor)
        return t
    elif isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(item, device) for item in obj)
    return obj


def diagnose_causal_mask(model, data, device="cuda"):
    """
    Definitive test: does the causal mask actually change model output?

    Takes one batch, runs with mask on vs off, compares predictions
    at early positions. Also tests censored input vs full input with mask.
    """
    holdout_dls = data["holdout_mixed_dls"]
    batch = next(iter(holdout_dls.train))
    inputs, targets = batch
    inputs = _to_device(inputs, device)

    model.eval()
    with torch.no_grad():
        # Run 1: WITH causal mask
        model.causal = True
        if not hasattr(model, '_orig_causal_mask') and model.causal_mask is not None:
            model._orig_causal_mask = model.causal_mask
        from astra.models.hybrid.model import _build_causal_mask
        n_cat = len(data["classes"])
        n_cont = len(data["num_cols"])
        seq_len = model.seq_len
        model.causal_mask = _build_causal_mask(seq_len, n_cat + n_cont, device)
        model.causal = True
        logits_causal = model(inputs)  # [batch, seq_len]

        # Run 2: WITHOUT causal mask
        model.causal = False
        saved_mask = model.causal_mask
        model.causal_mask = None
        logits_no_causal = model(inputs)  # [batch, seq_len]

        # Restore
        model.causal = True
        model.causal_mask = saved_mask

    # Compare at key positions
    logger.info("=" * 70)
    logger.info("CAUSAL MASK DIAGNOSTIC (single batch)")
    logger.info("=" * 70)
    logger.info(f"  Batch size: {logits_causal.shape[0]}, Seq len: {logits_causal.shape[1]}")
    logger.info(f"  Causal mask shape: {model.causal_mask.shape}")
    logger.info(f"  Causal mask True count: {model.causal_mask.sum().item()} "
                f"(of {model.causal_mask.numel()})")

    probs_causal = torch.sigmoid(logits_causal).cpu().numpy()
    probs_no_causal = torch.sigmoid(logits_no_causal).cpu().numpy()

    for pos in [0, 5, 11, 30, 50, 80, 90]:
        if pos >= probs_causal.shape[1]:
            continue
        diff = np.abs(probs_causal[:, pos] - probs_no_causal[:, pos])
        logger.info(
            f"  Position {pos:>3d}: "
            f"causal={probs_causal[:, pos].mean():.4f}, "
            f"no_causal={probs_no_causal[:, pos].mean():.4f}, "
            f"mean_abs_diff={diff.mean():.6f}, "
            f"max_abs_diff={diff.max():.6f}"
        )

    total_diff = np.abs(probs_causal - probs_no_causal)
    logger.info(f"  Overall: mean_diff={total_diff.mean():.6f}, max_diff={total_diff.max():.6f}")

    if total_diff.max() < 1e-5:
        logger.info("  VERDICT: Causal mask has NO effect on outputs!")
        logger.info("  Possible causes:")
        logger.info("    1. Mask not reaching _ScaledDotProductAttention")
        logger.info("    2. attn_mask parameter name mismatch in forward chain")
        logger.info("    3. Model was saved without mask and load overwrites it")
    elif total_diff.mean() < 0.01:
        logger.info("  VERDICT: Causal mask has minimal effect")
    else:
        logger.info("  VERDICT: Causal mask IS changing outputs")

    # Also check: does position 0 differ? It should NOT (it's causal either way)
    pos0_diff = np.abs(probs_causal[:, 0] - probs_no_causal[:, 0]).mean()
    logger.info(f"  Position 0 diff (should be ~0): {pos0_diff:.8f}")

    logger.info("=" * 70)
    return total_diff.mean()


def run_validation(
    data: dict,
    model_name: str,
    key_steps: Optional[List[int]] = None,
    device: str = "cuda",
) -> List[ValidationResult]:
    """
    Cross-validate temporal model evaluation methods.

    Args:
        data: Output from prepare_data_and_dls()
        model_name: Saved model name
        key_steps: Censor steps to validate at (defaults to key timepoints)
        device: Compute device

    Returns:
        List of ValidationResult for each timepoint
    """
    model_cfg = cfg.get("model", {})

    # Default key timepoints
    if key_steps is None:
        key_steps = [
            time_to_step(1, "h"),
            time_to_step(3, "h"),
            time_to_step(6, "h"),
            time_to_step(12, "h"),
            time_to_step(24, "h"),
            time_to_step(48, "h"),
            time_to_step(72, "h"),
            time_to_step(7, "D"),
            time_to_step(14, "D"),
            time_to_step(30, "D"),
        ]
        key_steps = [s for s in key_steps if s is not None]

    # ========================================================================
    # Load temporal model (FORCE causal=True for validation)
    # ========================================================================
    logger.info(f"Loading temporal model: {model_name}")
    logger.info(f"  cfg model.causal = {model_cfg.get('causal', 'NOT SET')}")
    backbone = get_backbone(
        data, cfg,
        temporal_head=True,
        causal=True,  # Force True regardless of config
        temporal_head_dropout=model_cfg.get("temporal_head_dropout", 0.3),
        temporal_head_mult=model_cfg.get("temporal_head_mult", 0.5),
        temporal_channel_idx=data.get('temporal_channel_idx'),
        exclude_channel_indices=data.get('exclude_channel_indices', []),
        bin_width_channel_idx=data.get('bin_width_channel_idx'),
    )

    from astra.data.mixed_dataloader import load_model_state
    state_dict = load_model_state(model_name)
    backbone.load_state_dict(state_dict, strict=False)
    backbone = backbone.to(device)

    model = backbone
    model.eval()

    logger.info(f"Model loaded. causal={model.causal}, "
                f"temporal_head_enabled={model.temporal_head_enabled}")
    logger.info(f"Causal mask: shape={model.causal_mask.shape if model.causal_mask is not None else 'None'}, "
                f"device={model.causal_mask.device if model.causal_mask is not None else 'N/A'}")

    # ========================================================================
    # DIAGNOSTIC: verify causal mask actually changes model output
    # ========================================================================
    diagnose_causal_mask(model, data, device)

    # ========================================================================
    # Setup evaluators
    # ========================================================================
    traj_lengths = np.array(data.get("holdout_trajectory_lengths", []))

    # Method A: TemporalEvaluator (single forward pass)
    temporal_eval = TemporalEvaluator(data, model, cfg, device=device)

    # ========================================================================
    # Run validation at each timepoint
    # ========================================================================
    results = []

    logger.info("=" * 90)
    logger.info("TEMPORAL MODEL VALIDATION")
    logger.info("=" * 90)
    logger.info(f"{'Time':>10s}  {'Step':>4s}  "
                f"{'A:AUROC':>7s} {'A:AUPRC':>7s}  "
                f"{'B:AUROC':>7s} {'B:AUPRC':>7s}  "
                f"{'C:AUROC':>7s} {'C:AUPRC':>7s}  "
                f"{'A-B AUROC':>9s} {'A-B AUPRC':>9s}")
    logger.info("-" * 90)

    for censor_step in key_steps:
        label = format_step_label(censor_step)
        vr = ValidationResult(censor_step=censor_step, label=label)

        # --- Method A: TemporalEvaluator ---
        result_a = temporal_eval.evaluate_at_timestep(censor_step)
        if result_a is not None:
            vr.auroc_temporal = result_a.auroc
            vr.auprc_temporal = result_a.auprc

        # --- Method B: Censored input (gold standard) ---
        y_preds_b, ys_b = _run_temporal_model_on_censored_data(
            model, data, censor_step, device=device,
        )
        if y_preds_b is not None and len(ys_b) > 0 and 0 < ys_b.sum() < len(ys_b):
            auroc_b, _, _ = calculate_roc_auc_ci(ys_b, y_preds_b)
            auprc_b, _, _ = calculate_average_precision_ci(ys_b, y_preds_b)
            vr.auroc_censored = auroc_b
            vr.auprc_censored = auprc_b

        # --- Method C: No causal mask ---
        y_preds_c, ys_c = _run_temporal_model_no_causal(
            model, data, censor_step, traj_lengths, device=device,
        )
        if y_preds_c is not None and len(ys_c) > 0 and 0 < ys_c.sum() < len(ys_c):
            auroc_c, _, _ = calculate_roc_auc_ci(ys_c, y_preds_c)
            auprc_c, _, _ = calculate_average_precision_ci(ys_c, y_preds_c)
            vr.auroc_no_causal = auroc_c
            vr.auprc_no_causal = auprc_c

        results.append(vr)

        # Print row
        def _fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A "

        diff_auroc = ""
        diff_auprc = ""
        if vr.auroc_temporal is not None and vr.auroc_censored is not None:
            diff_auroc = f"{vr.auroc_temporal - vr.auroc_censored:+.4f}"
        if vr.auprc_temporal is not None and vr.auprc_censored is not None:
            diff_auprc = f"{vr.auprc_temporal - vr.auprc_censored:+.4f}"

        logger.info(
            f"{label:>10s}  {censor_step:>4d}  "
            f"{_fmt(vr.auroc_temporal)} {_fmt(vr.auprc_temporal)}  "
            f"{_fmt(vr.auroc_censored)} {_fmt(vr.auprc_censored)}  "
            f"{_fmt(vr.auroc_no_causal)} {_fmt(vr.auprc_no_causal)}  "
            f"{diff_auroc:>9s} {diff_auprc:>9s}"
        )

    # ========================================================================
    # Summary interpretation
    # ========================================================================
    logger.info("=" * 90)
    logger.info("INTERPRETATION GUIDE:")
    logger.info("  A = TemporalEvaluator (full data + causal mask, pick position)")
    logger.info("  B = Censored input    (future data zeroed, pick position) [GOLD STANDARD]")
    logger.info("  C = No causal mask    (full data, no mask, pick position) [LEAK DETECTOR]")
    logger.info("")
    logger.info("  A ≈ B          → Causal mask works correctly")
    logger.info("  A >> B         → LEAK: causal mask not blocking future info")
    logger.info("  A ≈ C          → Causal mask has no effect")
    logger.info("  A ≈ B ≈ C      → Model doesn't use future data (naturally causal)")
    logger.info("  B high at 1H   → Model is genuinely good (or data itself leaks)")
    logger.info("=" * 90)

    # Compute average diffs
    diffs_auroc = []
    diffs_auprc = []
    for vr in results:
        if vr.auroc_temporal is not None and vr.auroc_censored is not None:
            diffs_auroc.append(vr.auroc_temporal - vr.auroc_censored)
        if vr.auprc_temporal is not None and vr.auprc_censored is not None:
            diffs_auprc.append(vr.auprc_temporal - vr.auprc_censored)

    if diffs_auroc:
        mean_diff_auroc = np.mean(diffs_auroc)
        mean_diff_auprc = np.mean(diffs_auprc)
        logger.info(f"Mean A-B diff:  AUROC={mean_diff_auroc:+.4f}, AUPRC={mean_diff_auprc:+.4f}")

        if abs(mean_diff_auroc) < 0.01 and abs(mean_diff_auprc) < 0.01:
            logger.info("VERDICT: Causal mask appears to work correctly (A ≈ B)")
        elif mean_diff_auroc > 0.02 or mean_diff_auprc > 0.02:
            logger.info("VERDICT: POSSIBLE LEAK — TemporalEvaluator gives inflated metrics (A > B)")
        else:
            logger.info("VERDICT: Small differences — inspect per-timepoint results")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate temporal model evaluation")
    parser.add_argument("--temporal-head", action="store_true", default=True)
    parser.add_argument("--no-causal", action="store_true", default=False)
    args = parser.parse_args()

    # Set config
    cfg.setdefault("model", {})["temporal_head"] = True
    cfg["model"]["causal"] = not args.no_causal

    logger.info("Loading data...")
    data = prepare_data_and_dls_cached(cfg)
    model_name = cfg["model_name"]

    results = run_validation(data, model_name, device="cuda")
    return results


if __name__ == "__main__":
    main()
