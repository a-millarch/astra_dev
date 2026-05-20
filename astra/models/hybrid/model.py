import logging
import math

import numpy as np
import torch.nn.functional as F

from torch import Tensor
import torch
import torch.nn as nn

from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def ifnone(a, b):
    """Returns b if a is None else a"""
    return b if a is None else a


def _build_causal_mask(seq_len: int, n_static: int, device: torch.device) -> torch.Tensor:
    """
    Build causal attention mask for temporal positions with static tokens as read-only context.

    Temporal position t can only attend to positions 0..t (causal) and all statics.
    Static positions can only attend to other statics (NOT to temporal positions).

    This prevents an information bridge: without blocking static→temporal attention,
    statics absorb future temporal info through attention, then early temporal positions
    read those contaminated statics — bypassing the causal constraint entirely.

    Args:
        seq_len: Number of temporal positions.
        n_static: Number of static positions (n_cat + n_cont).
        device: Target device.

    Returns:
        Boolean mask [total_len, total_len] where True = blocked.
    """
    total_len = seq_len + n_static
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    # Temporal→temporal: causal (position t attends only to ≤t)
    mask[:seq_len, :seq_len] = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    # Temporal→static: unmasked (temporal tokens CAN read statics)
    # mask[:seq_len, seq_len:] remains False
    # Static→temporal: BLOCKED (statics must NOT absorb temporal info)
    mask[seq_len:, :seq_len] = True
    # Static→static: unmasked (statics can attend to each other)
    # mask[seq_len:, seq_len:] remains False
    return mask


class TemporalPredictionHead(nn.Module):
    """
    Per-timestep prediction head: shared MLP applied to each temporal position.

    Takes transformer output [batch, total_len, d_model], slices the first seq_len
    positions (temporal only), applies a shared MLP to each, outputs [batch, seq_len].
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.3,
                 head_mult: float = 0.5):
        super().__init__()
        self.seq_len = seq_len
        hidden = max(1, int(d_model * head_mult))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, total_len, d_model] -- transformer output
        Returns:
            logits: [batch, seq_len] -- one logit per timestep
        """
        x_temporal = x[:, :self.seq_len, :]  # [batch, seq_len, d_model]
        logits = self.mlp(x_temporal).squeeze(-1)  # [batch, seq_len]
        return logits


class _Flatten(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.full = full
    
    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class Sequential(nn.Sequential):
    """Class that allows you to pass one or multiple inputs"""
    def forward(self, *x):
        for i, module in enumerate(self._modules.values()):
            x = module(*x) if isinstance(x, (list, tuple)) else module(x)
        return x

class _MLP(nn.Module):
    def __init__(self, dims, bn=False, act=None, skip=False, dropout=0., bn_final=False):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for i, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = i >= (len(dims) - 2)
            if bn and (not is_last or bn_final): 
                layers.append(nn.BatchNorm1d(dim_in))
            if dropout and not is_last:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_in, dim_out))
            if is_last: 
                break
            layers.append(ifnone(act, nn.ReLU()))
        self.mlp = nn.Sequential(*layers)
        self.shortcut = nn.Linear(dims[0], dims[-1]) if skip else None

    def forward(self, x):
        if self.shortcut is not None: 
            return self.mlp(x) + self.shortcut(x)
        else:
            return self.mlp(x)

    
class _TabFusionEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 res_dropout=0.1, activation='gelu', res_attention=False, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            _TabFusionEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, 
                                   d_ff=d_ff, res_dropout=res_dropout, 
                                   activation=activation, res_attention=res_attention) 
            for i in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src, attn_mask=None, key_padding_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: 
                output, scores = mod(output, prev=scores, attn_mask=attn_mask, 
                                    key_padding_mask=key_padding_mask)
            return output
        else:
            for mod in self.layers: 
                output = mod(output, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            return output



class _TabFusionEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 res_dropout=0.1, activation="gelu", res_attention=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)
        d_ff = ifnone(d_ff, d_model * 4)
        
        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, res_attention=res_attention)
        
        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.layernorm_attn = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            self._get_activation_fn(activation), 
            nn.Linear(d_ff, d_model)
        )
        
        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.layernorm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None, attn_mask=None, key_padding_mask: Optional[Tensor] = None):
        # Multi-Head attention sublayer
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, 
                                                 key_padding_mask=key_padding_mask, 
                                                 attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, 
                                        key_padding_mask=key_padding_mask, 
                                        attn_mask=attn_mask)
        self.attn = attn
        
        # Add & Norm
        src = src + self.dropout_attn(src2)
        src = self.layernorm_attn(src)
        
        # Feed-forward sublayer
        src2 = self.ff(src)
        
        # Add & Norm
        src = src + self.dropout_ffn(src2)
        src = self.layernorm_ffn(src)
        
        if self.res_attention:
            return src, scores
        else:
            return src

    def _get_activation_fn(self, activation):
        if callable(activation): 
            return activation()
        elif activation.lower() == "relu": 
            return nn.ReLU()
        elif activation.lower() == "gelu": 
            return nn.GELU()
        else:
            return nn.GELU()


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, res_attention: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)
        
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(self.d_k, self.res_attention)
        
    def forward(self, Q, K, V, prev=None, attn_mask=None, key_padding_mask: Optional[Tensor] = None):
        bs = Q.size(0)
        
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        if self.res_attention:
            context, attn, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, 
                                                   key_padding_mask=key_padding_mask, 
                                                   attn_mask=attn_mask)
        else:
            context, attn = self.sdp_attn(q_s, k_s, v_s, 
                                          key_padding_mask=key_padding_mask, 
                                          attn_mask=attn_mask)
        
        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        
        # Linear
        output = self.W_O(context)
        
        if self.res_attention: 
            return output, attn, scores
        else: 
            return output, attn
    
################################################




class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, res_attention: bool = False): 
        super().__init__()
        self.d_k = d_k
        self.res_attention = res_attention
        
    def forward(self, q, k, v, prev=None, attn_mask=None, key_padding_mask: Optional[Tensor] = None):
        # MatMul (q, k) - similarity scores
        scores = torch.matmul(q, k)
        
        # Scale
        scores = scores / (self.d_k ** 0.5)
        
        # Attention mask (optional)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(attn_mask, float('-inf'))
            else:
                scores += attn_mask
        
        # Key padding mask (optional)
        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        
        # SoftMax
        if prev is not None: 
            scores = scores + prev
        
        attn = F.softmax(scores, dim=-1)
        
        # MatMul (attn, v)
        context = torch.matmul(attn, v)
        
        if self.res_attention: 
            return context, attn, scores
        else: 
            return context, attn



class MultiHotEmbedding(nn.Module):
    """
    Embedding layer for multi-hot encoded categorical features.
    Takes multi-hot vectors and produces weighted sum of embeddings.
    """
    
    def __init__(self, n_classes: int, embedding_dim: int, use_counts: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.use_counts = use_counts
    
    def forward(self, x_multi_hot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_multi_hot: [batch_size, seq_len, n_classes]
        
        Returns:
            embedded: [batch_size, seq_len, embedding_dim]
        """
        all_embeddings = self.embedding.weight
        
        if self.use_counts:
            # Normalize counts to probabilities
            count_sum = x_multi_hot.sum(dim=-1, keepdim=True).clamp(min=1.0)
            weights = x_multi_hot / count_sum
            embedded = torch.matmul(weights, all_embeddings)
        else:
            # Simple weighted sum for binary multi-hot
            embedded = torch.matmul(x_multi_hot, all_embeddings)
        
        return embedded


class TimeAwarePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal tokens; learned for static tokens.

    Temporal positions receive sinusoidal embeddings computed from raw elapsed_hours
    (admission-relative hours at each bin).  Static token positions (categorical +
    continuous features) receive a learned embedding.

    When elapsed_hours is None (temporal_features mode != 'sinusoidal'), temporal
    tokens receive zero PE and static tokens use the learned embedding — identical
    to the previous nn.Parameter(zeros) behaviour so there is no regression.

    A single learnable time_scale parameter allows the model to calibrate how
    "spread out" the sinusoidal bands are relative to the raw hour values.
    """

    def __init__(self, d_model: int, n_static_tokens: int):
        super().__init__()
        self.d_model = d_model
        self.n_static_tokens = n_static_tokens
        # Learned embeddings for static tokens (replaces old pos_enc static slice)
        self.static_pos = nn.Parameter(torch.zeros(1, n_static_tokens, d_model))
        # Learnable global time scale (initialised to 1.0)
        self.time_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        elapsed_hours: Optional[torch.Tensor] = None,
        ts_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len + n_static_tokens, d_model]
            elapsed_hours: [batch, seq_len] raw hours, or None for zero-PE fallback.
            ts_padding_mask: [batch, seq_len] bool, True = padding.  When provided,
                padding positions receive zero PE instead of cos(0)=1 contamination.
        Returns:
            x + positional encoding, same shape as x.
        """
        bs, total_len, d = x.shape
        n_ts = total_len - self.n_static_tokens

        if elapsed_hours is not None:
            t = elapsed_hours.unsqueeze(-1).float() * self.time_scale  # [B, T, 1]
            half_d = d // 2
            freq = torch.exp(
                torch.arange(half_d, device=x.device, dtype=torch.float32)
                * -(math.log(10000.0) / half_d)
            )
            # [B, T, d]  (truncate last dim in case d is odd)
            ts_pos = torch.cat([torch.sin(t * freq), torch.cos(t * freq)], dim=-1)[:, :, :d]
            # Zero out PE at padding positions to prevent cos(0)=1 contamination
            if ts_padding_mask is not None:
                ts_pos = ts_pos.masked_fill(ts_padding_mask.unsqueeze(-1), 0.0)
        else:
            ts_pos = torch.zeros(bs, n_ts, d, device=x.device)

        static = self.static_pos.expand(bs, -1, -1)           # [B, n_static, d]
        return x + torch.cat([ts_pos, static], dim=1)         # [B, total_len, d]


class TSTabFusionTransformerMultiHot(nn.Module):
    """
    TSTabFusionTransformer with multi-hot categorical time series support.
    
    FIXED to work with TSAI's get_mixed_dls format.
    
    Key differences from standard version:
    - Accepts multi-hot encoded categorical TS (multiple labels per timestep)
    - Uses MultiHotEmbedding instead of standard Embedding
    - Handles variable number of active categories per timestep
    
    Use this when your categorical features can have multiple values at once,
    e.g., multiple medications, multiple diagnoses, multiple events.
    """
    
    def __init__(
        self,
        c_in: int,                              # Continuous TS channels
        c_out: int,                             # Output classes
        seq_len: int,                           # Sequence length
        classes: Dict,                          # Static categorical features
        cont_names: List[str],                  # Static continuous features
        ts_cat_dims: Optional[Dict[str, int]] = None,  # NEW: {feature_name: n_classes}
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: Optional[int] = None,
        res_attention: bool = True,
        attention_act: str = 'gelu',
        res_dropout: float = 0.,
        fc_mults: tuple = (1, .5),
        fc_dropout: float = 0.,
        fc_act = None,
        fc_skip: bool = False,
        fc_bn: bool = False,
        bn_final: bool = False,
        init: bool = True,
        key_padding_mask: str = 'auto',
        cat_ts_combine: str = 'add',           # 'add' or 'concat'
        use_count_normalization: bool = False,  # NEW: Normalize counts
        temporal_head: bool = False,            # Per-timestep prediction head
        causal: bool = False,                   # Causal attention masking
        temporal_head_dropout: float = 0.3,     # Dropout for temporal head MLP
        temporal_head_mult: float = 0.5,        # Hidden dim = int(d_model * mult)
        temporal_channel_idx: Optional[int] = None,          # Index of elapsed_hours in x_ts
        exclude_channel_indices: Optional[List[int]] = None, # Aux channels to skip in W_P
        head_pool: str = 'flatten',             # 'flatten' (legacy) or 'mean_cat' (pooled)
        per_feature_cont_proj: bool = False,    # Per-feature static continuous projection
        cat_ts_gate: bool = False,              # Learned sigmoid gate per categorical group
        local_temporal_kernel: int = 1,         # Depthwise conv kernel before W_P (1=disabled)
        bin_width_channel_idx: Optional[int] = None,  # Index of bin_width_hours in x_ts
        bin_width_modulation: bool = False,     # Modulate W_P output by bin width
        ts_cat_profile_dims: Optional[Dict[str, int]] = None,  # {category: n_levels} for profiled categories
    ):
        """
        Args:
            ts_cat_dims: Dictionary mapping feature names to their dimensions
                        For multi-hot: dimension = number of possible categories
                        Example: {'medication': 50, 'diagnosis': 100}
        """
        super().__init__()
        self.key_padding_mask = key_padding_mask
        self.cat_ts_combine = cat_ts_combine
        self.use_count_normalization = use_count_normalization
        
        # === MULTI-HOT CATEGORICAL TIME SERIES (NEW) ===
        # Initialize this FIRST to determine W_P size
        if ts_cat_dims is not None:
            zero_feats = [k for k, v in ts_cat_dims.items() if v == 0]
            if zero_feats:
                logger.warning(
                    f"Filtering {len(zero_feats)} categorical TS features "
                    f"with 0 classes: {zero_feats}"
                )
                ts_cat_dims = {k: v for k, v in ts_cat_dims.items() if v > 0}
            if not ts_cat_dims:
                ts_cat_dims = None

        if ts_cat_dims is not None:
            self.ts_cat_names = list(ts_cat_dims.keys())
            self.n_ts_cat = len(ts_cat_dims)
            
            if cat_ts_combine == 'concat':
                # Split embedding dimension among features
                emb_dim = max(1, d_model // (self.n_ts_cat + 1))
                self.ts_cat_embeds = nn.ModuleList([
                    MultiHotEmbedding(n_classes, emb_dim, use_count_normalization)
                    for n_classes in ts_cat_dims.values()
                ])
                # Continuous gets remaining dimension
                continuous_dim = d_model - (emb_dim * self.n_ts_cat)
            else:  # 'add'
                # Each feature gets full d_model dimensions
                self.ts_cat_embeds = nn.ModuleList([
                    MultiHotEmbedding(n_classes, d_model, use_count_normalization)
                    for n_classes in ts_cat_dims.values()
                ])
                continuous_dim = d_model
            
            self.ts_cat_dims = ts_cat_dims
            # Fix 3: Learned gate per categorical group
            if cat_ts_gate and cat_ts_combine == 'add':
                self.cat_ts_gate_params = nn.Parameter(torch.zeros(self.n_ts_cat))
            else:
                self.cat_ts_gate_params = None
        else:
            self.n_ts_cat = 0
            self.ts_cat_embeds = None
            self.ts_cat_dims = {}
            self.cat_ts_gate_params = None
            continuous_dim = d_model

        # === PROFILED CATEGORICAL TIME SERIES ===
        # Per-category ordinal embeddings for categories with clinician-defined profiles.
        # These are separate from multi-hot binary categories.
        if ts_cat_profile_dims:
            from astra.data.preprocessing import ProfileEmbedding
            profile_emb_dim = d_model  # always 'add' mode for profiles
            self.profile_embedding = ProfileEmbedding(
                profile_dims=ts_cat_profile_dims,
                embedding_dim=profile_emb_dim,
                zero_absent=True,
            )
            self.ts_cat_profile_dims = ts_cat_profile_dims
        else:
            self.profile_embedding = None
            self.ts_cat_profile_dims = {}

        # === AUXILIARY CHANNEL EXCLUSION ===
        # Channels listed in exclude_channel_indices (e.g. elapsed_hours, bin_width_hours)
        # remain in x_ts for extraction but are NOT projected through W_P.
        # This keeps W_P operating on ~N(0,1) normalized clinical data only.
        self.temporal_channel_idx = temporal_channel_idx
        self.exclude_channel_indices = sorted(exclude_channel_indices or [])
        self._signal_indices = [i for i in range(c_in) if i not in set(self.exclude_channel_indices)]
        n_signal = len(self._signal_indices)

        # === CONTINUOUS TIME SERIES ===
        # Fix 9: Optional depthwise conv for local temporal context before W_P
        if local_temporal_kernel > 1:
            if causal:
                pad = nn.ConstantPad1d((local_temporal_kernel - 1, 0), 0.0)
            else:
                pad = nn.ConstantPad1d((local_temporal_kernel // 2, local_temporal_kernel // 2), 0.0)
            self.local_temporal_conv = nn.Sequential(
                pad,
                nn.Conv1d(n_signal, n_signal, kernel_size=local_temporal_kernel,
                          padding=0, groups=n_signal),
                nn.GELU(),
            )
        else:
            self.local_temporal_conv = None

        # Initialize W_P AFTER determining the correct output dimension
        self.W_P = nn.Conv1d(n_signal, continuous_dim, 1)

        # Fix 5: Bin-width modulation
        self.bin_width_channel_idx = bin_width_channel_idx
        if bin_width_modulation and bin_width_channel_idx is not None:
            self.bin_width_mod = nn.Sequential(
                nn.Linear(1, d_model),
                nn.Sigmoid(),
            )
        else:
            self.bin_width_mod = None

        # === STATIC CATEGORICAL FEATURES ===
        n_cat = len(classes)
        n_classes = [len(v) for v in classes.values()]
        self.n_emb = sum(n_classes)
        self.embeds = nn.ModuleList([nn.Embedding(ni, d_model) for ni in n_classes])

        # === STATIC CONTINUOUS FEATURES ===
        n_cont = len(cont_names)
        self.n_cont = n_cont
        # Fix 1: Per-feature projections for static continuous features
        self.per_feature_cont = per_feature_cont_proj
        if per_feature_cont_proj and n_cont > 0:
            self.cont_projections = nn.ModuleList([
                nn.Linear(1, d_model) for _ in range(n_cont)
            ])
            if init:
                for proj in self.cont_projections:
                    nn.init.kaiming_normal_(proj.weight)
            self.conv = None
        else:
            self.conv = nn.Conv1d(1, d_model, 1)
            if init:
                nn.init.kaiming_normal_(self.conv.weight)
            self.cont_projections = None
        
        # === TRANSFORMER ===
        self.res_drop = nn.Dropout(res_dropout) if res_dropout else None
        self.pos_enc = TimeAwarePositionalEncoding(d_model, n_static_tokens=n_cat + n_cont)
        self.transformer = _TabFusionEncoder(
            n_cat + n_cont, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
            res_dropout=res_dropout, activation=attention_act,
            res_attention=res_attention, n_layers=n_layers
        )
        
        # === HEAD ===
        self.temporal_head_enabled = temporal_head
        self.causal = causal
        self.seq_len = seq_len

        self.head_pool = head_pool
        self._n_static = n_cat + n_cont

        if temporal_head:
            # Per-timestep prediction head (~2K params)
            self.temporal_pred_head = TemporalPredictionHead(
                d_model, seq_len, dropout=temporal_head_dropout,
                head_mult=temporal_head_mult,
            )
            self.head = None  # Skip large flatten+MLP
            self.head_nf = d_model
        elif head_pool == 'mean_cat':
            # Pooled head: mean-pool temporal tokens, concat static tokens
            # Input: d_model (temporal mean) + d_model * n_static (flattened statics)
            mlp_input_size = d_model * (1 + n_cat + n_cont)
            hidden_dimensions = list(map(lambda t: int(mlp_input_size * t), fc_mults))
            all_dimensions = [mlp_input_size, *hidden_dimensions, c_out]
            self.head_nf = mlp_input_size
            self.head = _MLP(all_dimensions, act=fc_act, skip=fc_skip, bn=fc_bn,
                             dropout=fc_dropout, bn_final=bn_final)
            self.temporal_pred_head = None
            logger.info(f"Pooled head (mean_cat): input={mlp_input_size}, "
                        f"dims={all_dimensions}, "
                        f"params={sum(p.numel() for p in self.head.parameters()):,}")
        else:
            # Original flatten + MLP head
            mlp_input_size = (d_model * (n_cat + n_cont + seq_len))
            hidden_dimensions = list(map(lambda t: int(mlp_input_size * t), fc_mults))
            all_dimensions = [mlp_input_size, *hidden_dimensions, c_out]
            self.head_nf = mlp_input_size
            self.head = nn.Sequential(
                _Flatten(),
                _MLP(all_dimensions, act=fc_act, skip=fc_skip, bn=fc_bn,
                     dropout=fc_dropout, bn_final=bn_final)
            )
            self.temporal_pred_head = None

        # Causal mask (registered as buffer for device tracking)
        if causal:
            n_static = n_cat + n_cont
            self.register_buffer(
                'causal_mask',
                _build_causal_mask(seq_len, n_static, torch.device('cpu'))
            )
        else:
            self.causal_mask = None
    
    def forward(self, *x):
        """
        FIXED: Forward pass compatible with fastai's get_preds() and training.
        
        Handles both:
        - Packed input: forward((x_ts, x_tab, x_ts_cat)) - from dataloader
        - Unpacked input: forward(x_ts, x_tab, x_ts_cat) - from get_preds()
        
        TSAI's get_mixed_dls outputs batches in this format:
            ((x_ts, x_tab, x_ts_cat), y)
        
        Where:
            x[0] = x_ts: [batch, c_in, seq_len] - continuous time series
            x[1] = x_tab: (x_cat, x_cont) - static tabular features (tuple)
            x[2] = x_ts_cat: [batch, total_cat_dim, seq_len] - categorical time series
        
        Args:
            *x: Variable arguments to handle both packed and unpacked formats
        """
        
        # === HANDLE PACKED vs UNPACKED INPUTS ===
        if len(x) >= 2:
            # Unpacked: forward(x_ts, x_tab, x_ts_cat) or forward(x_ts, x_tab)
            # This happens with get_preds() and some callbacks
            x_input = x
        elif len(x) == 1:
            # Packed: forward((x_ts, x_tab, x_ts_cat))
            # This happens during normal training
            x_input = x[0]
            if not isinstance(x_input, (tuple, list)):
                x_input = (x_input,)
        else:
            raise ValueError(f"Unexpected number of inputs: {len(x)}")
        
        # === PARSE INPUT (FIXED FOR TSAI FORMAT) ===
        traj_lengths = None
        if isinstance(x_input, (tuple, list)):
            if len(x_input) >= 3:
                # TSAI format: (x_ts, x_tab, x_ts_cat, traj_lengths[, profiles])
                x_ts = x_input[0]                    # Continuous TS
                x_tab = x_input[1]                   # Tabular (tuple)
                x_ts_cat_multi_hot = x_input[2]      # Categorical TS
                if len(x_input) >= 4:
                    traj_lengths = x_input[3]         # Trajectory lengths

                # Unpack tabular
                if isinstance(x_tab, (tuple, list)) and len(x_tab) == 2:
                    x_cat, x_cont = x_tab
                else:
                    # Fallback: treat entire x_tab as categorical
                    x_cat = x_tab
                    x_cont = torch.tensor([], device=x_ts.device)

            elif len(x_input) == 2:
                # Format without categorical TS: (x_ts, x_tab)
                x_ts = x_input[0]
                x_tab = x_input[1]
                x_ts_cat_multi_hot = None
                
                # Unpack tabular
                if isinstance(x_tab, (tuple, list)) and len(x_tab) == 2:
                    x_cat, x_cont = x_tab
                else:
                    x_cat = x_tab
                    x_cont = torch.tensor([], device=x_ts.device)
            else:
                raise ValueError(
                    f"Expected input with 2-4 elements, got {len(x_input)}. "
                    f"Format: (x_ts, x_tab, x_ts_cat[, traj_lengths]) or (x_ts, x_tab)"
                )
        else:
            # Single tensor input (backward compatibility)
            x_ts = x_input
            x_ts_cat_multi_hot = None
            x_cat = torch.tensor([], device=x_input.device)
            x_cont = torch.tensor([], device=x_input.device)
        
        # === EXTRACT PROFILE TENSOR ===
        x_ts_cat_profiles = None
        if isinstance(x_input, (tuple, list)) and len(x_input) >= 5:
            x_ts_cat_profiles = x_input[4]  # [bs, n_profiled, seq_len] or None

        # === HANDLE KEY PADDING MASK ===
        if traj_lengths is not None:
            # Proper padding mask from trajectory lengths (preferred)
            key_padding_mask = self._build_traj_padding_mask(x_ts, traj_lengths)
        elif self.key_padding_mask == "auto":
            x_ts, key_padding_mask = self._key_padding_mask(x_ts)
        else:
            key_padding_mask = None
        
        # === PROCESS CONTINUOUS TIME SERIES ===
        # Extract raw elapsed_hours for sinusoidal positional encoding (before stripping)
        if self.temporal_channel_idx is not None:
            elapsed_hours = x_ts[:, self.temporal_channel_idx, :]   # [bs, seq_len]
        else:
            elapsed_hours = None

        # Extract bin_width_hours for modulation (Fix 5)
        if self.bin_width_channel_idx is not None:
            bin_width_hours = x_ts[:, self.bin_width_channel_idx, :]  # [bs, seq_len]
        else:
            bin_width_hours = None

        # Strip auxiliary channels (elapsed_hours, bin_width_hours) before W_P so that
        # only ~N(0,1) normalized clinical features are projected.
        x_ts_signal = x_ts[:, self._signal_indices, :] if self.exclude_channel_indices else x_ts

        # Fix 9: Local temporal context before pointwise projection
        if self.local_temporal_conv is not None:
            x_ts_signal = self.local_temporal_conv(x_ts_signal)

        x = self.W_P(x_ts_signal).transpose(1, 2)  # [bs, seq_len, d_model]

        # Fix 5: Modulate by bin width (wider bins → different representation scaling)
        if self.bin_width_mod is not None and bin_width_hours is not None:
            bw_scale = self.bin_width_mod(bin_width_hours.unsqueeze(-1))  # [bs, T, d_model]
            x = x * bw_scale
        
        # === PROCESS MULTI-HOT CATEGORICAL TIME SERIES ===
        if self.n_ts_cat > 0 and x_ts_cat_multi_hot is not None:
            # x_ts_cat_multi_hot: [bs, total_cat_dim, seq_len] from TSAI
            # Could be TSTensor (TSAI's custom class) or regular tensor
            
            # Convert TSTensor to regular tensor if needed
            if hasattr(x_ts_cat_multi_hot, 'data'):
                # TSTensor has .data attribute
                x_ts_cat_multi_hot = x_ts_cat_multi_hot.data
            
            # Ensure it's a float tensor for embedding
            x_ts_cat_multi_hot = x_ts_cat_multi_hot.float()
            
            # Need to transpose to [bs, seq_len, total_cat_dim] for embedding
            x_ts_cat_multi_hot = x_ts_cat_multi_hot.transpose(1, 2)
            
            x_ts_cat_embedded_list = []
            dim_offset = 0
            
            for embed_layer, (feat_name, n_classes) in zip(
                self.ts_cat_embeds, self.ts_cat_dims.items()
            ):
                # Extract multi-hot vector for this feature
                feat_multi_hot = x_ts_cat_multi_hot[
                    :, :, dim_offset:dim_offset + n_classes
                ]
                
                # Embed
                feat_embedded = embed_layer(feat_multi_hot)
                x_ts_cat_embedded_list.append(feat_embedded)
                
                dim_offset += n_classes
            
            # Combine embeddings
            if self.cat_ts_combine == 'add':
                stacked = torch.stack(x_ts_cat_embedded_list, dim=0)  # [n_groups, B, T, d]
                if self.cat_ts_gate_params is not None:
                    # Fix 3: Learned sigmoid gate per categorical group
                    gates = torch.sigmoid(self.cat_ts_gate_params)  # [n_groups]
                    stacked = stacked * gates[:, None, None, None]
                x = x + stacked.sum(dim=0)
            else:  # 'concat'
                # Concatenate all categorical embeddings
                x_ts_cat_concat = torch.cat(x_ts_cat_embedded_list, dim=-1)
                x = torch.cat([x, x_ts_cat_concat], dim=-1)
        
        # === PROCESS PROFILED CATEGORICAL TIME SERIES ===
        if self.profile_embedding is not None and x_ts_cat_profiles is not None:
            # x_ts_cat_profiles: [bs, n_profiled, seq_len] → [bs, seq_len, n_profiled]
            x_profiles = x_ts_cat_profiles.transpose(1, 2)
            profile_embedded = self.profile_embedding(x_profiles)  # [bs, seq_len, d_model]
            x = x + profile_embedded

        # === PROCESS STATIC CATEGORICAL FEATURES ===
        if self.n_emb != 0 and x_cat.numel() > 0:
            x_cat_list = [e(x_cat[:, i]).unsqueeze(1) for i, e in enumerate(self.embeds)]
            x_cat_embedded = torch.cat(x_cat_list, 1)
            x = torch.cat([x, x_cat_embedded], 1)
        
        # === PROCESS STATIC CONTINUOUS FEATURES ===
        if self.n_cont != 0 and x_cont.numel() > 0:
            if self.cont_projections is not None:
                # Fix 1: Per-feature projection — each feature gets its own linear map
                x_cont_proj = torch.stack([
                    proj(x_cont[:, i:i+1]) for i, proj in enumerate(self.cont_projections)
                ], dim=1)  # [batch, n_cont, d_model]
            else:
                x_cont_proj = self.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_proj], 1)
        
        # === TRANSFORMER ===
        # Extract temporal padding mask for sinusoidal PE (prevents cos(0)=1 contamination)
        ts_padding_mask = None
        if key_padding_mask is not None:
            ts_padding_mask = key_padding_mask[:, :self.seq_len]
        x = self.pos_enc(x, elapsed_hours=elapsed_hours, ts_padding_mask=ts_padding_mask)

        if self.res_drop is not None:
            x = self.res_drop(x)

        attn_mask = self.causal_mask if self.causal else None
        x = self.transformer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        if key_padding_mask is not None:
            # x: [batch, total_len, d_model], mask: [batch, total_len] (True=padding)
            x = x * (~key_padding_mask).unsqueeze(-1)

        # === HEAD ===
        if self.temporal_head_enabled and self.temporal_pred_head is not None:
            x = self.temporal_pred_head(x)  # [batch, seq_len]
        elif self.head_pool == 'mean_cat':
            # Pool temporal tokens, concat static tokens
            x_temporal = x[:, :self.seq_len, :]    # [B, seq_len, d_model]
            x_static = x[:, self.seq_len:, :]      # [B, n_static, d_model]
            if key_padding_mask is not None:
                # Masked mean: exclude padding positions
                ts_mask = ~key_padding_mask[:, :self.seq_len]  # [B, seq_len], True=valid
                ts_mask_f = ts_mask.unsqueeze(-1).float()      # [B, seq_len, 1]
                x_pooled = (x_temporal * ts_mask_f).sum(dim=1) / ts_mask_f.sum(dim=1).clamp(min=1)
            else:
                x_pooled = x_temporal.mean(dim=1)  # [B, d_model]
            x = torch.cat([x_pooled, x_static.reshape(x.shape[0], -1)], dim=1)
            x = self.head(x)  # [batch, c_out]
        else:
            x = self.head(x)  # [batch, c_out]
        return x
    
    def _build_traj_padding_mask(self, x_ts, traj_lengths):
        """Build key_padding_mask from trajectory lengths.

        Args:
            x_ts: [batch, c_in, seq_len]
            traj_lengths: [batch] — number of real timesteps per sample

        Returns:
            mask: [batch, total_len] where True = padding (masked out).
                  total_len = seq_len + n_static_tokens.
        """
        bs, _, seq_len = x_ts.shape
        device = x_ts.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)     # [1, seq_len]
        tl = traj_lengths.to(device).unsqueeze(1)                         # [batch, 1]
        ts_mask = positions >= tl                                         # [batch, seq_len]

        # Static tokens (categorical + continuous) are never masked
        n_static = self.pos_enc.n_static_tokens
        if n_static > 0:
            static_mask = torch.zeros(bs, n_static, dtype=torch.bool, device=device)
            return torch.cat([ts_mask, static_mask], dim=1)               # [batch, total_len]
        return ts_mask

    def _key_padding_mask(self, x):
        """Handle NaN values in time series (legacy fallback)."""
        mask = torch.isnan(x)
        x[mask] = 0
        if mask.any():
            mask = (mask.float().mean(1) == 1).bool()
            return x, mask
        else:
            return x, None

