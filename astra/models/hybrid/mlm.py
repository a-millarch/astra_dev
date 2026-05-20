import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
# ============================================================================
# Configuration Management
# ============================================================================

# ============================================================================
# Enhanced MLM Model with Contrastive Learning
# ============================================================================

@dataclass
class MLMConfig:
    """Configuration for MLM pre-training with multi-hot categorical TS"""
    # Masking strategy
    mask_prob_ts: float = 0.15       # Continuous time series
    mask_prob_cat_ts: float = 0.15   # NEW: Multi-hot categorical TS
    mask_prob_cat: float = 0.15      # Static categorical
    mask_prob_cont: float = 0.15     # Static continuous
    replace_prob: float = 0.8
    random_prob: float = 0.1

    # Sparse-aware masking (Fix 6): preferentially mask non-empty timesteps
    sparse_aware_masking: bool = False
    mask_prob_ts_data: float = 0.30       # Mask prob for non-empty continuous TS timesteps
    mask_prob_ts_empty: float = 0.05      # Mask prob for empty continuous TS timesteps
    mask_prob_cat_ts_data: float = 0.40   # Mask prob for non-empty categorical TS timesteps
    mask_prob_cat_ts_empty: float = 0.03  # Mask prob for empty categorical TS timesteps

    # Training hyperparameters
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Loss weighting
    ts_loss_weight: float = 1.0
    cat_ts_loss_weight: float = 1.0      # NEW: Multi-hot cat TS loss
    cat_loss_weight: float = 1.0         # Static cat loss
    cont_loss_weight: float = 1.0
    contrastive_weight: float = 0.0
    
    # Contrastive learning
    temperature: float = 0.07
    projection_dim: int = 128
    
    # Checkpointing
    save_best: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    val_frequency: int = 1


class TSTabFusionMLM(nn.Module):
    """
    MLM wrapper for TSTabFusionTransformerMultiHot.
    
    Handles:
    - Continuous time series masking & reconstruction
    - Multi-hot categorical time series masking & reconstruction (NEW)
    - Static categorical masking & reconstruction
    - Static continuous masking & reconstruction
    - Contrastive learning across all modalities
    """
    
    def __init__(self, backbone, config: MLMConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        
        d_model = backbone.W_P.out_channels
        # W_P only projects signal channels; reconstruction target must match.
        n_signal = backbone.W_P.in_channels

        # === RECONSTRUCTION HEADS ===

        # 1. Continuous time series reconstruction (signal channels only)
        self.ts_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, n_signal)
        )
        
        # 2. Multi-hot categorical TS reconstruction (NEW)
        if backbone.n_ts_cat > 0:
            self.cat_ts_heads = nn.ModuleDict()
            for feat_name, n_classes in backbone.ts_cat_dims.items():
                if n_classes == 0:
                    continue
                self.cat_ts_heads[feat_name] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model, n_classes)
                )
        else:
            self.cat_ts_heads = None
        
        # 2b. Profile categorical TS reconstruction (per-category cross-entropy)
        if backbone.ts_cat_profile_dims:
            self.profile_heads = nn.ModuleDict()
            for cat_name, n_levels in backbone.ts_cat_profile_dims.items():
                self.profile_heads[cat_name] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model, n_levels + 1)  # +1 for absent level 0
                )
        else:
            self.profile_heads = None

        # 3. Static categorical reconstruction
        if backbone.n_emb != 0:
            n_classes = [emb.num_embeddings for emb in backbone.embeds]
            self.cat_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model, n_class)
                ) for n_class in n_classes
            ])
        else:
            self.cat_heads = None
        
        # 4. Static continuous reconstruction
        if backbone.n_cont != 0:
            self.cont_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 1)
            )
        else:
            self.cont_head = None
        
        # 5. Contrastive learning projection
        if config.contrastive_weight > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, config.projection_dim)
            )
        else:
            self.projection_head = None
    
    def create_mlm_mask(self, shape, device, mask_prob):
        """Create random mask for MLM."""
        return torch.rand(shape, device=device) < mask_prob
    
    def mask_time_series(self, x_ts):
        """Mask continuous time series data."""
        bs, c_in, seq_len = x_ts.shape
        device = x_ts.device

        # Create mask per timestep (same mask for all channels)
        if self.config.sparse_aware_masking:
            # Fix 6: Preferentially mask non-empty timesteps
            has_data = (x_ts.abs().sum(dim=1) > 0)  # [bs, seq_len]
            probs = torch.where(
                has_data,
                torch.tensor(self.config.mask_prob_ts_data, device=device),
                torch.tensor(self.config.mask_prob_ts_empty, device=device),
            )
            mask = torch.rand((bs, seq_len), device=device) < probs
        else:
            mask = self.create_mlm_mask((bs, seq_len), device, self.config.mask_prob_ts)
        
        original_x_ts = x_ts.clone()
        masked_x_ts = x_ts.clone()
        
        for i in range(bs):
            for t in range(seq_len):
                if mask[i, t]:
                    rand_val = torch.rand(1).item()
                    if rand_val < self.config.replace_prob:
                        # Replace with zeros
                        masked_x_ts[i, :, t] = 0
                    elif rand_val < self.config.replace_prob + self.config.random_prob:
                        # Replace with random timestep
                        random_t = torch.randint(0, seq_len, (1,)).item()
                        masked_x_ts[i, :, t] = x_ts[i, :, random_t]
        
        return masked_x_ts, mask, original_x_ts
    
    def mask_categorical_ts(self, x_ts_cat):
        """
        Mask multi-hot categorical time series (NEW).
        
        Args:
            x_ts_cat: [bs, n_categories, seq_len] - multi-hot encoded
        
        Returns:
            masked, mask, original
        """
        if x_ts_cat is None:
            return None, None, None
        
        # Convert TSTensor to regular tensor if needed
        if hasattr(x_ts_cat, 'data'):
            x_ts_cat = x_ts_cat.data
        
        x_ts_cat = x_ts_cat.float()
        
        bs, n_categories, seq_len = x_ts_cat.shape
        device = x_ts_cat.device
        
        # Create mask per timestep (mask all categories at once)
        if self.config.sparse_aware_masking:
            # Fix 6: Preferentially mask non-empty timesteps
            has_data = (x_ts_cat.sum(dim=1) > 0)  # [bs, seq_len]
            probs = torch.where(
                has_data,
                torch.tensor(self.config.mask_prob_cat_ts_data, device=device),
                torch.tensor(self.config.mask_prob_cat_ts_empty, device=device),
            )
            mask = torch.rand((bs, seq_len), device=device) < probs
        else:
            mask = self.create_mlm_mask((bs, seq_len), device, self.config.mask_prob_cat_ts)
        
        original_x_ts_cat = x_ts_cat.clone()
        masked_x_ts_cat = x_ts_cat.clone()
        
        for i in range(bs):
            for t in range(seq_len):
                if mask[i, t]:
                    rand_val = torch.rand(1).item()
                    if rand_val < self.config.replace_prob:
                        # Replace with all zeros (no categories active)
                        masked_x_ts_cat[i, :, t] = 0
                    elif rand_val < self.config.replace_prob + self.config.random_prob:
                        # Replace with random timestep
                        random_t = torch.randint(0, seq_len, (1,)).item()
                        masked_x_ts_cat[i, :, t] = x_ts_cat[i, :, random_t]
        
        return masked_x_ts_cat, mask, original_x_ts_cat
    
    def mask_categorical(self, x_cat):
        """Mask static categorical features."""
        if x_cat is None or x_cat.numel() == 0:
            return x_cat, None, x_cat
        
        bs, n_cat = x_cat.shape
        device = x_cat.device
        
        mask = self.create_mlm_mask((bs, n_cat), device, self.config.mask_prob_cat)
        
        original_x_cat = x_cat.clone()
        masked_x_cat = x_cat.clone()
        
        for i in range(bs):
            for j in range(n_cat):
                if mask[i, j]:
                    rand_val = torch.rand(1).item()
                    n_classes = self.backbone.embeds[j].num_embeddings
                    if rand_val < self.config.replace_prob:
                        masked_x_cat[i, j] = 0
                    elif rand_val < self.config.replace_prob + self.config.random_prob:
                        masked_x_cat[i, j] = torch.randint(0, n_classes, (1,)).item()
        
        return masked_x_cat, mask, original_x_cat
    
    def mask_continuous(self, x_cont):
        """Mask static continuous features."""
        if x_cont is None or x_cont.numel() == 0:
            return x_cont, None, x_cont
        
        bs, n_cont = x_cont.shape
        device = x_cont.device
        
        mask = self.create_mlm_mask((bs, n_cont), device, self.config.mask_prob_cont)
        
        original_x_cont = x_cont.clone()
        masked_x_cont = x_cont.clone()
        
        # Use feature means for better masking
        feature_means = x_cont.mean(dim=0, keepdim=True)
        
        for i in range(bs):
            for j in range(n_cont):
                if mask[i, j]:
                    rand_val = torch.rand(1).item()
                    if rand_val < self.config.replace_prob:
                        masked_x_cont[i, j] = feature_means[0, j]
                    elif rand_val < self.config.replace_prob + self.config.random_prob:
                        random_idx = torch.randint(0, bs, (1,)).item()
                        masked_x_cont[i, j] = x_cont[random_idx, j]
        
        return masked_x_cont, mask, original_x_cont
    
    def forward_encoder(self, x_ts, x_ts_cat, x_cat, x_cont, x_ts_cat_profiles=None):
        """
        Forward pass through encoder (UPDATED for TSAI format).

        Args:
            x_ts: Continuous TS
            x_ts_cat: Multi-hot categorical TS (NEW)
            x_cat: Static categorical
            x_cont: Static continuous
            x_ts_cat_profiles: Profile categorical TS [bs, n_profiled, seq_len] (optional)
        """
        # Pack into TSAI format: (x_ts, x_tab, x_ts_cat)
        x_tab = (x_cat, x_cont)
        x = (x_ts, x_tab, x_ts_cat)
        
        # Use backbone's forward (handles everything internally)
        # Get intermediate representations before the head
        # We need to access transformer output, not final classification
        
        # Handle NaN
        if self.backbone.key_padding_mask == "auto":
            x_ts, key_padding_mask = self.backbone._key_padding_mask(x_ts)
        else:
            key_padding_mask = None
        
        # Extract raw elapsed_hours before stripping (needed for sinusoidal PE)
        if self.backbone.temporal_channel_idx is not None:
            elapsed_hours = x_ts[:, self.backbone.temporal_channel_idx, :]
        else:
            elapsed_hours = None

        # Extract bin_width_hours for modulation (Fix 5 mirror)
        if self.backbone.bin_width_channel_idx is not None:
            bin_width_hours = x_ts[:, self.backbone.bin_width_channel_idx, :]
        else:
            bin_width_hours = None

        # Strip auxiliary channels before W_P (mirrors backbone.forward)
        x_ts_signal = (
            x_ts[:, self.backbone._signal_indices, :]
            if self.backbone.exclude_channel_indices else x_ts
        )

        # Fix 9 mirror: Local temporal context before W_P
        if self.backbone.local_temporal_conv is not None:
            x_ts_signal = self.backbone.local_temporal_conv(x_ts_signal)

        # Continuous TS encoding (signal channels only)
        x_encoded = self.backbone.W_P(x_ts_signal).transpose(1, 2)  # [bs, seq_len, d_model]

        # Fix 5 mirror: Bin-width modulation
        if self.backbone.bin_width_mod is not None and bin_width_hours is not None:
            bw_scale = self.backbone.bin_width_mod(bin_width_hours.unsqueeze(-1))
            x_encoded = x_encoded * bw_scale
        
        # Multi-hot categorical TS encoding (if present)
        if self.backbone.n_ts_cat > 0 and x_ts_cat is not None:
            # Convert TSTensor if needed
            if hasattr(x_ts_cat, 'data'):
                x_ts_cat = x_ts_cat.data
            x_ts_cat = x_ts_cat.float()
            
            # Transpose to [bs, seq_len, n_categories]
            x_ts_cat = x_ts_cat.transpose(1, 2)
            
            x_ts_cat_embedded_list = []
            dim_offset = 0
            
            for embed_layer, (feat_name, n_classes) in zip(
                self.backbone.ts_cat_embeds, self.backbone.ts_cat_dims.items()
            ):
                feat_multi_hot = x_ts_cat[:, :, dim_offset:dim_offset + n_classes]
                feat_embedded = embed_layer(feat_multi_hot)
                x_ts_cat_embedded_list.append(feat_embedded)
                dim_offset += n_classes
            
            # Combine embeddings
            if self.backbone.cat_ts_combine == 'add':
                stacked = torch.stack(x_ts_cat_embedded_list, dim=0)  # [n_groups, B, T, d]
                if self.backbone.cat_ts_gate_params is not None:
                    # Fix 3 mirror: Learned sigmoid gate per categorical group
                    gates = torch.sigmoid(self.backbone.cat_ts_gate_params)
                    stacked = stacked * gates[:, None, None, None]
                x_encoded = x_encoded + stacked.sum(dim=0)
            else:  # 'concat'
                x_ts_cat_concat = torch.cat(x_ts_cat_embedded_list, dim=-1)
                x_encoded = torch.cat([x_encoded, x_ts_cat_concat], dim=-1)
        
        # Profile categorical TS encoding (if present)
        if self.backbone.profile_embedding is not None and x_ts_cat_profiles is not None:
            x_profiles = x_ts_cat_profiles.transpose(1, 2)  # [bs, seq_len, n_profiled]
            profile_embedded = self.backbone.profile_embedding(x_profiles)
            x_encoded = x_encoded + profile_embedded

        # Static categorical encoding
        if self.backbone.n_emb != 0 and x_cat is not None and x_cat.numel() > 0:
            x_cat_list = [e(x_cat[:, i]).unsqueeze(1) for i, e in enumerate(self.backbone.embeds)]
            x_cat_embedded = torch.cat(x_cat_list, 1)
            x_encoded = torch.cat([x_encoded, x_cat_embedded], 1)

        # Static continuous encoding
        if self.backbone.n_cont != 0 and x_cont is not None and x_cont.numel() > 0:
            if self.backbone.cont_projections is not None:
                # Fix 1 mirror: Per-feature projection
                x_cont_proj = torch.stack([
                    proj(x_cont[:, i:i+1]) for i, proj in enumerate(self.backbone.cont_projections)
                ], dim=1)  # [batch, n_cont, d_model]
            else:
                x_cont_proj = self.backbone.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x_encoded = torch.cat([x_encoded, x_cont_proj], 1)
        
        # Positional encoding (time-aware if elapsed_hours available, else zero TS PE)
        x_encoded = self.backbone.pos_enc(x_encoded, elapsed_hours=elapsed_hours)
        
        if self.backbone.res_drop is not None:
            x_encoded = self.backbone.res_drop(x_encoded)
        
        # Transformer
        x_encoded = self.backbone.transformer(x_encoded, key_padding_mask=key_padding_mask)
        
        if key_padding_mask is not None:
            x_encoded = x_encoded * torch.logical_not(key_padding_mask.unsqueeze(1))
        
        return x_encoded
    
    def contrastive_loss(self, z1, z2):
        """Compute NT-Xent contrastive loss."""
        batch_size = z1.shape[0]
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2
        )
        
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        similarity_matrix = similarity_matrix / self.config.temperature
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def forward(self, x_ts, x_ts_cat, x_cat, x_cont,
                return_contrastive=False, x_ts_cat_profiles=None):
        """
        Forward pass with MLM (UPDATED for multi-hot categorical TS).

        Args:
            x_ts: [bs, c_in, seq_len] - continuous time series
            x_ts_cat: [bs, n_categories, seq_len] - multi-hot categorical TS
            x_cat: [bs, n_cat] - static categorical
            x_cont: [bs, n_cont] - static continuous
            return_contrastive: Whether to compute contrastive loss
            x_ts_cat_profiles: [bs, n_profiled, seq_len] - profile levels (optional)
        """
        # Mask profiles (same timestep mask as categorical TS, set to 0 = absent)
        original_profiles = None
        masked_profiles = x_ts_cat_profiles
        profile_mask = None
        if x_ts_cat_profiles is not None and self.profile_heads is not None:
            original_profiles = x_ts_cat_profiles.clone()
            bs, n_prof, sl = x_ts_cat_profiles.shape
            profile_mask = self.create_mlm_mask(
                (bs, sl), x_ts_cat_profiles.device,
                self.config.mask_prob_cat_ts,
            )
            masked_profiles = x_ts_cat_profiles.clone()
            # Set masked timesteps to 0 (absent) across all profiled categories
            masked_profiles[:, :, profile_mask] = 0

        # Create augmented views for contrastive learning
        if return_contrastive and self.config.contrastive_weight > 0:
            # First view
            masked_ts1, ts_mask1, original_ts = self.mask_time_series(x_ts)
            masked_ts_cat1, ts_cat_mask1, original_ts_cat = self.mask_categorical_ts(x_ts_cat)
            masked_cat1, cat_mask1, original_cat = self.mask_categorical(x_cat)
            masked_cont1, cont_mask1, original_cont = self.mask_continuous(x_cont)

            # Second view
            masked_ts2, _, _ = self.mask_time_series(x_ts)
            masked_ts_cat2, _, _ = self.mask_categorical_ts(x_ts_cat)
            masked_cat2, _, _ = self.mask_categorical(x_cat)
            masked_cont2, _, _ = self.mask_continuous(x_cont)

            encoder_output1 = self.forward_encoder(
                masked_ts1, masked_ts_cat1, masked_cat1, masked_cont1,
                x_ts_cat_profiles=masked_profiles,
            )
            encoder_output2 = self.forward_encoder(
                masked_ts2, masked_ts_cat2, masked_cat2, masked_cont2,
                x_ts_cat_profiles=masked_profiles,
            )

            encoder_output = encoder_output1
            ts_mask, ts_cat_mask = ts_mask1, ts_cat_mask1
            cat_mask, cont_mask = cat_mask1, cont_mask1

            # Global pooling for contrastive
            z1 = encoder_output1.mean(dim=1)
            z2 = encoder_output2.mean(dim=1)
            z1 = self.projection_head(z1)
            z2 = self.projection_head(z2)
            
            contrastive_loss = self.contrastive_loss(z1, z2)
        else:
            masked_ts, ts_mask, original_ts = self.mask_time_series(x_ts)
            masked_ts_cat, ts_cat_mask, original_ts_cat = self.mask_categorical_ts(x_ts_cat)
            masked_cat, cat_mask, original_cat = self.mask_categorical(x_cat)
            masked_cont, cont_mask, original_cont = self.mask_continuous(x_cont)

            encoder_output = self.forward_encoder(
                masked_ts, masked_ts_cat, masked_cat, masked_cont,
                x_ts_cat_profiles=masked_profiles,
            )
            contrastive_loss = None
        
        seq_len = x_ts.shape[2]
        n_cat = x_cat.shape[1] if x_cat is not None and x_cat.numel() > 0 else 0
        n_cont = x_cont.shape[1] if x_cont is not None and x_cont.numel() > 0 else 0
        
        # Split encoder output
        ts_output = encoder_output[:, :seq_len, :]
        cat_output = encoder_output[:, seq_len:seq_len+n_cat, :] if n_cat > 0 else None
        cont_output = encoder_output[:, seq_len+n_cat:, :] if n_cont > 0 else None
        
        losses = {}
        
        # 1. Continuous TS reconstruction (signal channels only — aux channels excluded)
        if ts_mask is not None and ts_mask.any():
            ts_pred = self.ts_head(ts_output).transpose(1, 2)  # [bs, n_signal, seq_len]
            # Compare against signal channels only; aux channels are not reconstructed
            if self.backbone.exclude_channel_indices:
                original_ts_signal = original_ts[:, self.backbone._signal_indices, :]
            else:
                original_ts_signal = original_ts
            expand = ts_mask.unsqueeze(1).expand_as(ts_pred)
            ts_loss = F.mse_loss(ts_pred[expand], original_ts_signal[expand])
            losses['ts_loss'] = ts_loss * self.config.ts_loss_weight
        
        # 2. Multi-hot categorical TS reconstruction (NEW)
        if ts_cat_mask is not None and ts_cat_mask.any() and self.cat_ts_heads is not None:
            cat_ts_losses = []
            dim_offset = 0

            for feat_name, n_classes in self.backbone.ts_cat_dims.items():
                if n_classes == 0:
                    continue

                feat_pred = self.cat_ts_heads[feat_name](ts_output)  # [bs, seq_len, n_classes]
                feat_pred = feat_pred.transpose(1, 2)  # [bs, n_classes, seq_len]

                feat_target = original_ts_cat[:, dim_offset:dim_offset+n_classes, :]

                mask_expanded = ts_cat_mask.unsqueeze(1).expand_as(feat_pred)
                masked_pred = feat_pred[mask_expanded]
                masked_target = feat_target[mask_expanded]

                if masked_pred.numel() > 0:
                    feat_loss = F.binary_cross_entropy_with_logits(masked_pred, masked_target)
                    cat_ts_losses.append(feat_loss)

                dim_offset += n_classes

            if cat_ts_losses:
                losses['cat_ts_loss'] = torch.stack(cat_ts_losses).mean() * self.config.cat_ts_loss_weight
        
        # 2b. Profile categorical TS reconstruction (per-category cross-entropy)
        if (profile_mask is not None and profile_mask.any()
                and self.profile_heads is not None and original_profiles is not None):
            profile_losses = []
            for i, (cat_name, n_levels) in enumerate(self.backbone.ts_cat_profile_dims.items()):
                # Predict profile level from transformer output at masked timesteps
                feat_pred = self.profile_heads[cat_name](ts_output)  # [bs, seq_len, n_levels+1]
                feat_target = original_profiles[:, i, :]              # [bs, seq_len]

                # Apply mask: only compute loss at masked positions
                masked_pred = feat_pred[profile_mask]                 # [n_masked, n_levels+1]
                masked_target = feat_target[profile_mask].long()      # [n_masked]

                if masked_pred.numel() > 0:
                    profile_losses.append(F.cross_entropy(masked_pred, masked_target))

            if profile_losses:
                losses['profile_loss'] = (
                    torch.stack(profile_losses).mean() * self.config.cat_ts_loss_weight
                )

        # 3. Static categorical reconstruction
        if cat_mask is not None and cat_mask.any() and cat_output is not None:
            cat_losses = []
            for i in range(n_cat):
                if cat_mask[:, i].any():
                    cat_pred = self.cat_heads[i](cat_output[:, i, :])
                    cat_loss = F.cross_entropy(
                        cat_pred[cat_mask[:, i]],
                        original_cat[cat_mask[:, i], i]
                    )
                    cat_losses.append(cat_loss)
            if cat_losses:
                losses['cat_loss'] = torch.stack(cat_losses).mean() * self.config.cat_loss_weight
        
        # 4. Static continuous reconstruction
        if cont_mask is not None and cont_mask.any() and cont_output is not None:
            cont_pred = self.cont_head(cont_output).squeeze(-1)
            cont_loss = F.mse_loss(
                cont_pred[cont_mask],
                original_cont[cont_mask]
            )
            losses['cont_loss'] = cont_loss * self.config.cont_loss_weight
        
        # 5. Contrastive loss
        if contrastive_loss is not None:
            losses['contrastive_loss'] = contrastive_loss * self.config.contrastive_weight
        
        # Total loss (skip NaN components to prevent poisoning the sum)
        finite_losses = [v for v in losses.values() if torch.isfinite(v)]
        total_loss = sum(finite_losses) if finite_losses else torch.tensor(0.0, device=x_ts.device)
        losses['total_loss'] = total_loss

        return losses


# ============================================================================
# Training with All Enhancements
# ============================================================================

class EarlyStopping:
    """Early stopping helper."""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_mlm_enhanced(
    model,
    train_loader,
    config,
    val_loader=None,
    device='cuda'
):
    """
    Enhanced pre-training for multi-hot categorical TS.
    
    UPDATED to handle TSAI's get_mixed_dls format:
        Batch format: ((x_ts, x_tab, x_ts_cat), y)
        where x_tab = (x_cat, x_cont)
    
    Args:
        model: TSTabFusionMLM instance
        train_loader: TSAI DataLoader from get_mixed_dls
        config: MLMConfig
        val_loader: Optional validation DataLoader
        device: Device to use
    
    Returns:
        history: Training history dict
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    num_training_steps = config.epochs * len(train_loader)
    num_warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta
    )
    
    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'ts_loss': [],
        'cat_ts_loss': [],
        'cat_loss': [],
        'cont_loss': [],
        'contrastive_loss': [],
        'profile_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # === TRAINING ===
        model.train()
        meters = defaultdict(lambda: {'sum': 0, 'count': 0})
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in pbar:
            # Unpack TSAI format: ((x_ts, x_tab, x_ts_cat), y)
            inputs, targets = batch
            
            # inputs is a tuple: (x_ts, x_tab, x_ts_cat, traj_lengths[, profiles])
            x_ts = inputs[0]           # Continuous TS
            x_tab = inputs[1]          # Tabular (tuple)
            x_ts_cat = inputs[2]       # Categorical TS (multi-hot)
            x_profiles = inputs[4] if len(inputs) >= 5 else None  # Profile TS (optional)

            # Unpack tabular
            x_cat, x_cont = x_tab

            # Move to device
            x_ts = x_ts.to(device)
            x_ts_cat = x_ts_cat.to(device) if x_ts_cat is not None else None
            x_cat = x_cat.to(device) if x_cat is not None and x_cat.numel() > 0 else None
            x_cont = x_cont.to(device) if x_cont is not None and x_cont.numel() > 0 else None
            if x_profiles is not None:
                x_profiles = x_profiles.to(device)

            # Forward pass
            optimizer.zero_grad()

            losses = model(
                x_ts, x_ts_cat, x_cat, x_cont,
                return_contrastive=(config.contrastive_weight > 0),
                x_ts_cat_profiles=x_profiles,
            )
            
            loss = losses['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update meters
            total_val = loss.item()
            if np.isfinite(total_val):
                meters['total']['sum'] += total_val
                meters['total']['count'] += 1
            else:
                logger.warning(f"NaN/Inf total_loss at batch — check component losses: "
                               f"{', '.join(f'{k}={v.item():.4f}' for k, v in losses.items() if k != 'total_loss')}")
            
            for loss_name in ['ts_loss', 'cat_ts_loss', 'cat_loss', 'cont_loss', 'contrastive_loss', 'profile_loss']:
                if loss_name in losses:
                    val = losses[loss_name].item()
                    if np.isfinite(val):
                        meters[loss_name]['sum'] += val
                        meters[loss_name]['count'] += 1
                    else:
                        logger.warning(f"NaN/Inf in {loss_name} at batch — skipping")
            
            # Update progress bar
            postfix = {
                'Loss': f"{meters['total']['sum']/meters['total']['count']:.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            }
            
            if meters['ts_loss']['count'] > 0:
                postfix['TS'] = f"{meters['ts_loss']['sum']/meters['ts_loss']['count']:.4f}"
            if meters['cat_ts_loss']['count'] > 0:
                postfix['CatTS'] = f"{meters['cat_ts_loss']['sum']/meters['cat_ts_loss']['count']:.4f}"
            if meters['cat_loss']['count'] > 0:
                postfix['Cat'] = f"{meters['cat_loss']['sum']/meters['cat_loss']['count']:.4f}"
            if meters['cont_loss']['count'] > 0:
                postfix['Cont'] = f"{meters['cont_loss']['sum']/meters['cont_loss']['count']:.4f}"
            if meters['contrastive_loss']['count'] > 0:
                postfix['Contr'] = f"{meters['contrastive_loss']['sum']/meters['contrastive_loss']['count']:.4f}"
            
            pbar.set_postfix(postfix)
        
        # Record training metrics
        avg_train_loss = (meters['total']['sum'] / meters['total']['count']
                          if meters['total']['count'] > 0 else float('nan'))
        history['train_loss'].append(avg_train_loss)
        for loss_name in ['ts_loss', 'cat_ts_loss', 'cat_loss', 'cont_loss', 'contrastive_loss', 'profile_loss']:
            history[loss_name].append(
                meters[loss_name]['sum'] / meters[loss_name]['count']
                if meters[loss_name]['count'] > 0 else 0
            )
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # === VALIDATION ===
        val_loss = None
        if val_loader is not None and (epoch + 1) % config.val_frequency == 0:
            model.eval()
            val_meters = {'total': {'sum': 0, 'count': 0}}
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    # Unpack TSAI format
                    inputs, targets = batch
                    x_ts = inputs[0]
                    x_tab = inputs[1]
                    x_ts_cat = inputs[2]
                    x_profiles = inputs[4] if len(inputs) >= 5 else None
                    x_cat, x_cont = x_tab

                    # Move to device
                    x_ts = x_ts.to(device)
                    x_ts_cat = x_ts_cat.to(device) if x_ts_cat is not None else None
                    x_cat = x_cat.to(device) if x_cat is not None and x_cat.numel() > 0 else None
                    x_cont = x_cont.to(device) if x_cont is not None and x_cont.numel() > 0 else None
                    if x_profiles is not None:
                        x_profiles = x_profiles.to(device)

                    losses = model(
                        x_ts, x_ts_cat, x_cat, x_cont,
                        x_ts_cat_profiles=x_profiles,
                    )
                    
                    val_total = losses['total_loss'].item()
                    if np.isfinite(val_total):
                        val_meters['total']['sum'] += val_total
                        val_meters['total']['count'] += 1

            val_loss = (val_meters['total']['sum'] / val_meters['total']['count']
                        if val_meters['total']['count'] > 0 else float('nan'))
            history['val_loss'].append(val_loss)

            # Save best model
            if config.save_best and np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_dir / 'best_model.pt')
                logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}")
        if val_loss is not None:
            logger.info(f"  Val Loss: {val_loss:.4f}")
    
    # Always save final model state (fallback if best_model.pt was never written)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, checkpoint_dir / 'last_model.pt')

    if not (checkpoint_dir / 'best_model.pt').exists():
        logger.warning("No best_model.pt was saved during training (val_loss may have been NaN). "
                        "Copying last_model.pt as best_model.pt.")
        torch.save(torch.load(checkpoint_dir / 'last_model.pt', weights_only=False),
                    checkpoint_dir / 'best_model.pt')

    # Plot training curves
    plot_training_curves(history, save_path=checkpoint_dir / 'training_curves.png')

    return history


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves with multi-hot categorical TS loss."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    def _finite(data):
        """Filter NaN/Inf for safe plotting."""
        arr = np.array(data, dtype=float)
        return np.where(np.isfinite(arr), arr, np.nan)

    # 1. Total loss
    train_loss = _finite(history['train_loss'])
    n_epochs = len(train_loss)
    if n_epochs > 0:
        axes[0, 0].plot(range(n_epochs), train_loss, label='Train Loss', linewidth=2)
    if history['val_loss']:
        val_loss = _finite(history['val_loss'])
        val_epochs = np.linspace(0, max(n_epochs - 1, 0), len(val_loss))
        axes[0, 0].plot(val_epochs, val_loss, label='Val Loss', linewidth=2, linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Component losses
    component_labels = [
        ('ts_loss', 'TS Loss'), ('cat_ts_loss', 'Cat TS Loss'),
        ('cat_loss', 'Static Cat Loss'), ('cont_loss', 'Static Cont Loss'),
        ('contrastive_loss', 'Contrastive Loss'), ('profile_loss', 'Profile Loss'),
    ]
    for key, label in component_labels:
        data = history.get(key, [])
        if data and any(v != 0 for v in data):
            axes[0, 1].plot(data, label=label)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Learning rate
    axes[1, 0].plot(history['lr'], linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Train vs Val loss detail (more useful than ratios that require cat_ts > 0)
    if n_epochs > 0:
        axes[1, 1].plot(range(n_epochs), train_loss, label='Train Loss', linewidth=2)
    if history['val_loss']:
        val_loss = _finite(history['val_loss'])
        val_epochs = np.linspace(0, max(n_epochs - 1, 0), len(val_loss))
        axes[1, 1].plot(val_epochs, val_loss, label='Val Loss', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Train vs Val Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

