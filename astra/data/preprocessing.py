#preprocessing.py
import pandas as pd
import numpy as np  
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict


# ============================================================================
# Multi-Hot Encoding
# ============================================================================

class MultiHotCategoricalEncoder:
    """
    Converts multi-label categorical time series to multi-hot vectors.
    
    Input format (wide with lists):
        TIMESTEP  PID  FEATURE     0                    1                           2
        0         1    medication  aspirin              [aspirin, ibuprofen]        NaN
        1         2    medication  [insulin, aspirin]   NaN                         insulin
    
    Output shape: [n_samples, n_categories, seq_len]
        Example: [17390, 12, 114]
        - 17390 patients
        - 12 medication types (channels)
        - 114 timesteps
    
    UPDATED: Now stores category_labels in encoding_info for SHAP visualization.
    """
    
    def __init__(self):
        self.encoders_ = {}
        self.n_classes_ = {}
    
    def fit(self, df: pd.DataFrame, sample_col: str, timestep_cols: List[str], 
            cat_col: str, feature_names: Optional[List[str]] = None):
        """
        Fit encoder on multi-label categorical data in WIDE format.
        
        Args:
            df: DataFrame with columns [sample_col, cat_col, timestep_0, timestep_1, ...]
                Cells can contain single values, lists, or NaN
            sample_col: Column identifying each sample (e.g., 'PID')
            timestep_cols: List of timestep column names (e.g., ['0', '1', '2', ...])
            cat_col: Column containing categorical feature name (e.g., 'FEATURE')
            feature_names: Optional list of categorical feature names to process
        """
        if feature_names is None:
            feature_names = df[cat_col].unique().tolist()
        
        for feat_name in feature_names:
            # Get all unique values for this categorical feature across all timesteps
            feat_df = df[df[cat_col] == feat_name]
            
            all_values = set()
            for ts_col in timestep_cols:
                # Handle cells that contain lists, single values, or NaN
                for cell in feat_df[ts_col].dropna():
                    if isinstance(cell, list):
                        all_values.update(cell)
                    else:
                        all_values.add(cell)
            
            # Create mapping: value -> index
            sorted_values = sorted(list(all_values))
            value_to_idx = {val: idx for idx, val in enumerate(sorted_values)}
            
            self.encoders_[feat_name] = {
                'value_to_idx': value_to_idx,
                'idx_to_value': {idx: val for val, idx in value_to_idx.items()},
                'n_classes': len(sorted_values),
                'category_labels': sorted_values  # ADDED: Store actual labels
            }
            self.n_classes_[feat_name] = len(sorted_values)
        
        return self
    
    def transform(self, df: pd.DataFrame, sample_col: str, timestep_cols: List[str], 
                  cat_col: str) -> Tuple[np.ndarray, Dict]:
        """
        Transform multi-label categorical data to multi-hot encoding.
        
        Args:
            df: DataFrame in wide format with timestep columns
            sample_col: Column identifying samples
            timestep_cols: List of timestep column names
            cat_col: Column containing feature names
        
        Returns:
            X_multi_hot: Array of shape [n_samples, n_categories, seq_len]
            encoding_info: Dictionary with encoding details including category_labels
        """
        samples = sorted(df[sample_col].unique())
        n_samples = len(samples)
        seq_len = len(timestep_cols)
        
        # Get all categorical features
        cat_features = list(self.encoders_.keys())
        
        # Calculate total dimension
        total_dim = sum(self.n_classes_[feat] for feat in cat_features)
        
        # Initialize output array: [n_samples, n_categories, seq_len]
        X_multi_hot = np.zeros((n_samples, total_dim, seq_len), dtype=np.float32)
        
        # Create sample index mapping
        sample_to_idx = {sample: idx for idx, sample in enumerate(samples)}
        
        # Process each categorical feature
        dim_offset = 0
        encoding_info = {
            'feature_ranges': {}, 
            'feature_names': cat_features,
            'category_labels': {}  # ADDED: Store labels per feature
        }
        
        for feat_name in cat_features:
            encoder = self.encoders_[feat_name]
            n_classes = encoder['n_classes']
            
            # Store dimension range for this feature
            encoding_info['feature_ranges'][feat_name] = (dim_offset, dim_offset + n_classes)
            
            # ADDED: Store category labels for this feature
            encoding_info['category_labels'][feat_name] = encoder['category_labels']
            
            # Get data for this feature
            feat_df = df[df[cat_col] == feat_name].set_index(sample_col)
            
            for sample_id in samples:
                if sample_id not in feat_df.index:
                    continue
                
                sample_idx = sample_to_idx[sample_id]
                sample_data = feat_df.loc[sample_id]
                
                # Handle case where sample appears multiple times (multiple rows)
                if isinstance(sample_data, pd.DataFrame):
                    # Multiple rows for this sample/feature
                    for ts_idx, ts_col in enumerate(timestep_cols):
                        # Collect all values from all rows for this timestep
                        values = set()
                        for cell in sample_data[ts_col].dropna():
                            if isinstance(cell, list):
                                values.update(cell)
                            else:
                                values.add(cell)
                        
                        # Set multi-hot encoding
                        for val in values:
                            if val in encoder['value_to_idx']:
                                val_idx = encoder['value_to_idx'][val]
                                # KEY: [sample, category, time]
                                X_multi_hot[sample_idx, dim_offset + val_idx, ts_idx] = 1.0
                else:
                    # Single row for this sample/feature
                    for ts_idx, ts_col in enumerate(timestep_cols):
                        cell = sample_data[ts_col]
                        
                        # Normalize to iterable of values
                        if isinstance(cell, (list, tuple, np.ndarray)):
                            vals = cell
                        elif pd.isna(cell):
                            vals = []
                        else:
                            vals = [cell]
                        
                        # Set multi-hot encoding
                        for v in vals:
                            if v in encoder['value_to_idx']:
                                val_idx = encoder['value_to_idx'][v]
                                # KEY: [sample, category, time]
                                X_multi_hot[sample_idx, dim_offset + val_idx, ts_idx] = 1.0
            
            dim_offset += n_classes
        
        return X_multi_hot, encoding_info
    
    def fit_transform(self, df: pd.DataFrame, sample_col: str, timestep_cols: List[str], 
                      cat_col: str, feature_names: Optional[List[str]] = None):
        """Fit and transform in one step."""
        self.fit(df, sample_col, timestep_cols, cat_col, feature_names)
        return self.transform(df, sample_col, timestep_cols, cat_col)
    
    def get_category_labels(self) -> Dict[str, List[str]]:
        """
        Get category labels for all features.
        
        Returns:
            Dict mapping feature names to list of category labels
            e.g., {'medication': ['Aspirin', 'Ibuprofen', ...], 'procedures': ['X-ray', ...]}
        """
        return {
            feat_name: encoder['category_labels'] 
            for feat_name, encoder in self.encoders_.items()
        }


# ============================================================================
# NEURAL NETWORK LAYER: Multi-Hot Embedding
# ============================================================================

class MultiHotEmbedding(nn.Module):
    """
    Embedding layer for multi-hot encoded categorical features.
    
    Instead of standard embedding (one index → one vector),
    this takes multi-hot vectors and produces weighted sum of embeddings.
    """
    
    def __init__(self, n_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
    
    def forward(self, x_multi_hot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_multi_hot: [batch_size, seq_len, n_classes] - multi-hot vectors
        
        Returns:
            embedded: [batch_size, seq_len, embedding_dim]
        """
        # Get all embeddings
        all_embeddings = self.embedding.weight  # [n_classes, embedding_dim]
        
        # Weighted sum: multi-hot weights × embeddings
        # [bs, seq, n_classes] @ [n_classes, emb_dim] → [bs, seq, emb_dim]
        embedded = torch.matmul(x_multi_hot, all_embeddings)
        
        return embedded


class MultiHotEmbeddingWithCount(nn.Module):
    """
    Embedding layer that handles count-based encoding.
    Normalizes by sum to create weighted average.
    """
    
    def __init__(self, n_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
    
    def forward(self, x_count: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_count: [batch_size, seq_len, n_classes] - count vectors
        
        Returns:
            embedded: [batch_size, seq_len, embedding_dim]
        """
        all_embeddings = self.embedding.weight
        
        # Normalize counts to get weights
        count_sum = x_count.sum(dim=-1, keepdim=True).clamp(min=1.0)  # Avoid div by 0
        weights = x_count / count_sum
        
        # Weighted sum
        embedded = torch.matmul(weights, all_embeddings)

        return embedded


class ProfileEmbedding(nn.Module):
    """Per-category ordinal embedding for profiled categorical TS.

    Each category has its own ``nn.Embedding`` table so the model can learn
    that levels within a category are related.  Level 0 = absent (maps to a
    zero vector by default, contributing nothing when the category is inactive).

    Args:
        profile_dims: ``{category_name: n_levels}``
            e.g. ``{'antibiotics': 3, 'opioids': 2}``.
            ``n_levels`` is the max profile level (1-based); the absent level 0
            is added automatically.
        embedding_dim: Output embedding dimension per category.
        zero_absent: If True, level-0 embedding is initialised to zeros so
            absent categories contribute nothing (gradient still flows).
    """

    def __init__(
        self,
        profile_dims: dict,
        embedding_dim: int,
        zero_absent: bool = True,
    ):
        super().__init__()
        self.category_order = list(profile_dims.keys())
        self.n_categories = len(self.category_order)
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleDict()
        for cat_name, n_levels in profile_dims.items():
            # +1 for the absent level at index 0
            emb = nn.Embedding(n_levels + 1, embedding_dim)
            if zero_absent:
                emb.weight.data[0].zero_()
            self.embeddings[cat_name] = emb

    def forward(self, x_profiles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_profiles: ``[batch_size, seq_len, n_profiled_categories]``
                Integer profile levels (0 = absent, 1..N = profile levels).

        Returns:
            embedded: ``[batch_size, seq_len, embedding_dim]``
                Sum of per-category embeddings.
        """
        parts = []
        for i, cat_name in enumerate(self.category_order):
            level = x_profiles[:, :, i].long()                 # [bs, seq]
            cat_embed = self.embeddings[cat_name](level)        # [bs, seq, emb_dim]
            parts.append(cat_embed)

        if not parts:
            bs, seq = x_profiles.shape[:2]
            return torch.zeros(bs, seq, self.embedding_dim, device=x_profiles.device)

        return sum(parts)