"""
Layer group management for discriminative learning rates and selective freezing.

Splits TSTabFusionTransformerMultiHot into named parameter groups:
  - embeddings: W_P, embeds, conv, ts_cat_embeds, pos_enc, res_drop
  - transformer_0_1 through transformer_{N-2}_{N-1}: transformer layer pairs
  - head: classification head (Flatten + MLP)
"""

import logging
import math
from collections import OrderedDict
from typing import Dict, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def get_layer_groups(model: nn.Module) -> OrderedDict:
    """
    Extract named parameter groups from TSTabFusionTransformerMultiHot.

    Returns:
        OrderedDict mapping group name -> list of (param_name, param) tuples.
        Groups are ordered from lowest (embeddings) to highest (head).
    """
    groups = OrderedDict()

    # --- Embedding group: all input projections ---
    embedding_names = {"W_P", "embeds", "conv", "ts_cat_embeds", "pos_enc", "res_drop"}
    emb_params = []
    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        if top_level in embedding_names:
            emb_params.append((name, param))
    groups["embeddings"] = emb_params

    # --- Transformer layer groups (pairs of 2) ---
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        n_layers = len(model.transformer.layers)
        # Group in pairs of 2 (or 1 if odd remainder)
        i = 0
        while i < n_layers:
            end = min(i + 2, n_layers)
            group_name = f"transformer_{i}_{end - 1}"
            group_params = []
            for name, param in model.named_parameters():
                for layer_idx in range(i, end):
                    prefix = f"transformer.layers.{layer_idx}."
                    if name.startswith(prefix):
                        group_params.append((name, param))
                        break
            groups[group_name] = group_params
            i = end

    # --- Head group ---
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("head.") or name.startswith("temporal_pred_head."):
            head_params.append((name, param))
    groups["head"] = head_params

    return groups


def get_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    lr_decay_factor: float = 0.1,
    weight_decay: float = 0.01,
) -> List[Dict]:
    """
    Build optimizer param groups with exponentially decaying learning rates.

    The head gets `base_lr`. Each group below gets a progressively smaller LR,
    with the embeddings getting `base_lr * lr_decay_factor`.

    Args:
        model: The backbone model.
        base_lr: Learning rate for the head (highest group).
        lr_decay_factor: Ratio between the lowest group LR and base_lr.
        weight_decay: Weight decay for all groups.

    Returns:
        List of dicts suitable for torch.optim.AdamW.
    """
    layer_groups = get_layer_groups(model)
    n_groups = len(layer_groups)

    if n_groups <= 1:
        return [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]

    # Exponential decay: group 0 gets base_lr * lr_decay_factor, last gets base_lr
    # lr_i = base_lr * lr_decay_factor^((n_groups - 1 - i) / (n_groups - 1))
    param_groups = []
    for i, (group_name, params_list) in enumerate(layer_groups.items()):
        params = [p for _, p in params_list if p.requires_grad]
        if not params:
            continue
        exponent = (n_groups - 1 - i) / (n_groups - 1)
        group_lr = base_lr * (lr_decay_factor ** exponent)
        param_groups.append({
            "params": params,
            "lr": group_lr,
            "weight_decay": weight_decay,
            "name": group_name,
        })

    return param_groups


def freeze_to(model: nn.Module, group_name: str) -> None:
    """
    Freeze all parameter groups up to and including `group_name`.
    Everything after `group_name` remains trainable.

    Args:
        model: The backbone model.
        group_name: Name of the last group to freeze (inclusive).
    """
    layer_groups = get_layer_groups(model)
    freeze = True
    for name, params_list in layer_groups.items():
        # Flip the flag AFTER processing the matched group (inclusive freeze)
        for _, param in params_list:
            param.requires_grad = not freeze
        if name == group_name:
            freeze = False  # groups after this one become trainable

    _log_trainable_summary(model, layer_groups)


def unfreeze_from(model: nn.Module, group_name: str) -> None:
    """
    Freeze everything below `group_name`, unfreeze `group_name` and above.

    If `group_name` doesn't exist (e.g. model has fewer layers than expected),
    falls back to the last transformer group so at least something is unfrozen.

    Args:
        model: The backbone model.
        group_name: Name of the first group to unfreeze (inclusive).
    """
    layer_groups = get_layer_groups(model)

    if group_name not in layer_groups:
        # Fall back to last transformer group (the one just before "head")
        transformer_groups = [n for n in layer_groups if n.startswith("transformer_")]
        fallback = transformer_groups[-1] if transformer_groups else "head"
        logger.warning(f"Layer group '{group_name}' not found in model "
                       f"(available: {list(layer_groups.keys())}). "
                       f"Falling back to '{fallback}'")
        group_name = fallback

    reached = False
    for name, params_list in layer_groups.items():
        if name == group_name:
            reached = True
        for _, param in params_list:
            param.requires_grad = reached

    _log_trainable_summary(model, layer_groups)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def set_dropout_rates(
    model: nn.Module,
    fc_dropout: Optional[float] = None,
    res_dropout: Optional[float] = None,
) -> None:
    """
    Modify dropout rates in-place. Pretrained weights are unaffected by
    dropout rate changes, so this is safe to call after loading a checkpoint.

    Args:
        model: The backbone model.
        fc_dropout: New dropout rate for the classification head MLP.
        res_dropout: New dropout rate for transformer residual connections.
    """
    if fc_dropout is not None:
        # Head MLP dropout layers (None when temporal head replaces standard head)
        if hasattr(model, "head") and model.head is not None:
            for module in model.head.modules():
                if isinstance(module, nn.Dropout):
                    module.p = fc_dropout
        # Temporal prediction head dropout (when temporal_head is enabled)
        if hasattr(model, "temporal_pred_head") and model.temporal_pred_head is not None:
            for module in model.temporal_pred_head.modules():
                if isinstance(module, nn.Dropout):
                    module.p = fc_dropout

    if res_dropout is not None:
        # Transformer residual dropout + top-level res_drop
        if hasattr(model, "res_drop") and model.res_drop is not None:
            model.res_drop.p = res_dropout
        if hasattr(model, "transformer"):
            for layer in model.transformer.layers:
                if hasattr(layer, "dropout_attn"):
                    layer.dropout_attn.p = res_dropout
                if hasattr(layer, "dropout_ffn"):
                    layer.dropout_ffn.p = res_dropout


def _log_trainable_summary(model: nn.Module, layer_groups: OrderedDict) -> None:
    """Log a summary of trainable vs frozen parameters per group."""
    total_trainable = 0
    total_frozen = 0
    for group_name, params_list in layer_groups.items():
        trainable = sum(p.numel() for _, p in params_list if p.requires_grad)
        frozen = sum(p.numel() for _, p in params_list if not p.requires_grad)
        total_trainable += trainable
        total_frozen += frozen
        status = "trainable" if trainable > 0 else "frozen"
        logger.info(f"  {group_name}: {status} ({trainable:,} params)")

    logger.info(
        f"  Total: {total_trainable:,} trainable, {total_frozen:,} frozen "
        f"({total_trainable / max(total_trainable + total_frozen, 1) * 100:.1f}% trainable)"
    )
