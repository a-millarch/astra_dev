"""
Pure-PyTorch replacements for TSAI/FastAI data loading.

Provides:
  - df2xy_pure:           long-format DF → [n_samples, n_features, seq_len] array
  - TabularEncoder:       replaces Categorify + FillMissing
  - AstraMixedDataset:    torch Dataset yielding ((x_ts, (x_cat, x_cont), x_ts_cat), y)
  - AstraMixedDataLoader: wraps dataset into .train / .valid DataLoaders
  - get_stratified_splits: replaces tsai.data.validation.get_splits
  - save_model / load_model_state: plain torch checkpoint helpers
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


# ============================================================================
# df2xy: long-format DataFrame → 3D numpy array
# ============================================================================

def df2xy_pure(
    df: pd.DataFrame,
    sample_col: str = 'PID',
    feat_col: str = 'FEATURE',
    data_cols=None,
    target_col: Optional[str] = None,
):
    """
    Convert long-format DataFrame to 3D array [n_samples, n_features, seq_len].

    Replicates tsai.data.preparation.df2xy behaviour:
      - Sorts by [sample_col, feat_col] ascending
      - Channel order = sorted unique FEATURE values

    Args:
        df:          DataFrame with columns [sample_col, feat_col, *data_cols, target_col]
        sample_col:  Column identifying the sample (patient).
        feat_col:    Column identifying the feature (channel).
        data_cols:   Timestep columns.  If None, inferred as all columns
                     after the first 3 (matching TSAI default).
        target_col:  Target column name (one value per sample).  May be None.

    Returns:
        X: ndarray [n_samples, n_features, seq_len]
        y: ndarray [n_samples, 1] or None
    """
    if data_cols is None:
        data_cols = df.columns[3:]

    # Convert Index to list for .loc indexing
    data_cols = list(data_cols)

    # Deterministic ordering: same as TSAI
    df_sorted = df.sort_values([sample_col, feat_col]).reset_index(drop=True)

    samples = df_sorted[sample_col].unique()       # preserves sorted order
    features = sorted(df_sorted[feat_col].unique()) # alphabetical

    n_samples = len(samples)
    n_features = len(features)
    seq_len = len(data_cols)

    # Build index mappings
    sample_to_idx = {s: i for i, s in enumerate(samples)}
    feat_to_idx = {f: i for i, f in enumerate(features)}

    # Pre-extract values as numpy for speed
    X = np.full((n_samples, n_features, seq_len), np.nan, dtype=np.float64)

    # Vectorised: group by (sample, feature), fill X
    grouped = df_sorted.groupby([sample_col, feat_col])
    for (sample, feat), group in grouped:
        si = sample_to_idx[sample]
        fi = feat_to_idx[feat]
        vals = group[data_cols].values
        if len(vals) == 1:
            X[si, fi, :] = vals[0].astype(np.float64)
        else:
            # Multiple rows for same (sample, feat) — take last
            X[si, fi, :] = vals[-1].astype(np.float64)

    # Target extraction
    y = None
    if target_col and target_col in df_sorted.columns:
        y_df = df_sorted.drop_duplicates(subset=[sample_col])
        y_df = y_df.set_index(sample_col).loc[samples]
        y = y_df[[target_col]].values  # [n_samples, 1]

    return X, y


# ============================================================================
# TabularEncoder: replaces Categorify + FillMissing
# ============================================================================

class TabularEncoder:
    """
    Pure-Python replacement for FastAI's Categorify + FillMissing processors.

    Behaviour:
      - FillMissing: for each numerical column with NaN, creates ``{col}_na``
        boolean indicator and fills NaN with 0.
      - Categorify: integer-encodes categorical columns.  Builds ``.classes``
        dict: ``{col: ['#na#', val1, val2, ...]}`` where index 0 is always
        the unknown/missing sentinel.

    Usage:
        encoder = TabularEncoder()
        encoder.fit(combined_df, cat_cols, num_cols)
        df_encoded = encoder.transform(trainval_df, cat_cols, num_cols)
        classes = encoder.classes   # {col: ['#na#', ...]}
    """

    def __init__(self):
        self.classes: Dict[str, list] = {}
        self.cat_mappings: Dict[str, dict] = {}
        self.na_cols: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame, cat_cols: list, num_cols: list):
        """Fit on combined trainval+holdout to capture all possible categories."""
        # 1. FillMissing: identify numerical columns with NaN
        self.na_cols = [col for col in num_cols if df[col].isna().any()]

        # 2. Build category lists for every categorical column
        all_cat_cols = list(cat_cols)
        for col in self.na_cols:
            all_cat_cols.append(f'{col}_na')

        for col in all_cat_cols:
            if col.endswith('_na') and col not in df.columns:
                # Will be created during transform
                unique_vals = [False, True]
            else:
                unique_vals = sorted(
                    df[col].dropna().unique().tolist(),
                    key=lambda v: str(v),
                )
                # Prevent duplicate '#na#' when fillna('#na#') makes it a data value
                unique_vals = [v for v in unique_vals if v != '#na#']
            self.classes[col] = ['#na#'] + unique_vals
            self.cat_mappings[col] = {v: i for i, v in enumerate(self.classes[col])}

        self._fitted = True
        logger.info(f"TabularEncoder fitted: {len(self.classes)} categorical columns, "
                     f"{len(self.na_cols)} NA indicators")

    def transform(self, df: pd.DataFrame, cat_cols: list, num_cols: list) -> pd.DataFrame:
        """Apply encoding: create _na indicators, fill NaN, integer-code categoricals."""
        assert self._fitted, "Must call fit() before transform()"
        df = df.copy()

        # 1. FillMissing: create {col}_na indicators, fill NaN with 0
        for col in self.na_cols:
            na_name = f'{col}_na'
            df[na_name] = df[col].isna()
            df[col] = df[col].fillna(0)

        # 2. Categorify: integer-encode all categorical columns
        all_cat_cols = list(cat_cols) + [f'{c}_na' for c in self.na_cols]
        for col in all_cat_cols:
            mapping = self.cat_mappings.get(col, {})
            df[col] = df[col].map(lambda v, m=mapping: m.get(v, 0))

        return df

    def get_cat_cont_arrays(
        self, df: pd.DataFrame, cat_cols: list, num_cols: list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (x_cat, x_cont) numpy arrays from a *transformed* DataFrame.

        Returns:
            x_cat:  int64 array [n_samples, n_cat_features]
            x_cont: float32 array [n_samples, n_cont_features]
        """
        all_cat_cols = list(cat_cols) + [f'{c}_na' for c in self.na_cols]
        x_cat = df[all_cat_cols].values.astype(np.int64)
        x_cont = df[num_cols].values.astype(np.float32) if num_cols else np.zeros((len(df), 0), dtype=np.float32)
        return x_cat, x_cont


# ============================================================================
# AstraMixedDataset: PyTorch Dataset
# ============================================================================

class AstraMixedDataset(Dataset):
    """
    Dataset holding all three modalities.

    Each sample returns:
        ((x_ts, (x_cat, x_cont), x_ts_cat), y)

    matching the batch format expected by TSTabFusionTransformerMultiHot.forward().
    """

    def __init__(
        self,
        X_ts: np.ndarray,
        x_cat: np.ndarray,
        x_cont: np.ndarray,
        X_ts_cat: np.ndarray,
        y,
        trajectory_lengths: Optional[np.ndarray] = None,
        event_times: Optional[np.ndarray] = None,
        event_indicators: Optional[np.ndarray] = None,
        X_ts_cat_profiles: Optional[np.ndarray] = None,
    ):
        self.X_ts = torch.from_numpy(np.asarray(X_ts, dtype=np.float32))
        self.x_cat = torch.from_numpy(np.asarray(x_cat, dtype=np.int64))
        self.x_cont = torch.from_numpy(np.asarray(x_cont, dtype=np.float32))
        self.X_ts_cat = torch.from_numpy(np.asarray(X_ts_cat, dtype=np.int64))
        self.y = torch.tensor(np.asarray(y, dtype=np.int64)).squeeze()
        if trajectory_lengths is not None:
            self.traj_lengths = torch.from_numpy(
                np.asarray(trajectory_lengths, dtype=np.int64)
            )
        else:
            # Fallback: assume full sequence length (no masking)
            self.traj_lengths = torch.full(
                (len(self.y),), self.X_ts.shape[-1], dtype=torch.int64
            )

        # Survival labels (optional — None when survival_mode is disabled)
        if event_times is not None:
            self.event_times = torch.from_numpy(
                np.asarray(event_times, dtype=np.int64)
            )
            self.event_indicators = torch.from_numpy(
                np.asarray(event_indicators, dtype=np.int64)
            )
        else:
            self.event_times = None
            self.event_indicators = None

        # Profile-based categorical TS (optional — None when profiles disabled)
        if X_ts_cat_profiles is not None:
            self.X_ts_cat_profiles = torch.from_numpy(
                np.asarray(X_ts_cat_profiles, dtype=np.int64)
            )
        else:
            self.X_ts_cat_profiles = None

    def __len__(self):
        return len(self.y)

    @property
    def has_survival_labels(self) -> bool:
        return self.event_times is not None

    @property
    def has_profiles(self) -> bool:
        return self.X_ts_cat_profiles is not None

    def __getitem__(self, idx):
        inputs = (
            self.X_ts[idx], (self.x_cat[idx], self.x_cont[idx]),
            self.X_ts_cat[idx], self.traj_lengths[idx],
        )
        if self.has_profiles:
            inputs = inputs + (self.X_ts_cat_profiles[idx],)
        if self.has_survival_labels:
            targets = (self.y[idx], self.event_times[idx], self.event_indicators[idx])
        else:
            targets = self.y[idx]
        return inputs, targets


# ============================================================================
# AstraMixedDataLoader: .train / .valid DataLoader wrapper
# ============================================================================

class AstraMixedDataLoader:
    """
    Thin wrapper that provides ``.train`` and ``.valid`` DataLoader properties.

    Replaces TSAI's ``get_mixed_dls()`` return object.

    Args:
        dataset:       AstraMixedDataset (full dataset).
        splits:        Optional (train_indices, valid_indices).  If None the
                       entire dataset is used for both train and valid.
        bs:            Batch size.
        shuffle_train: Whether to shuffle the train DataLoader.
        drop_last:     Whether to drop the last incomplete batch.
        num_workers:   DataLoader workers (0 = main process).
    """

    def __init__(
        self,
        dataset: AstraMixedDataset,
        splits: Optional[Tuple[list, list]] = None,
        bs: int = 64,
        shuffle_train: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
    ):
        if splits is not None:
            train_idx, valid_idx = splits
            self._train_ds = Subset(dataset, train_idx)
            self._valid_ds = Subset(dataset, valid_idx)
        else:
            # No split — full dataset used for both (evaluation / no-split DLs)
            self._train_ds = dataset
            self._valid_ds = dataset

        self._bs = bs
        self._shuffle_train = shuffle_train
        self._drop_last = drop_last
        self._num_workers = num_workers

    @property
    def train(self):
        return DataLoader(
            self._train_ds,
            batch_size=self._bs,
            shuffle=self._shuffle_train,
            drop_last=self._drop_last,
            num_workers=self._num_workers,
        )

    @property
    def valid(self):
        return DataLoader(
            self._valid_ds,
            batch_size=self._bs,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
        )


# ============================================================================
# Stratified splits
# ============================================================================

def get_stratified_splits(
    y,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[list, list]:
    """
    Stratified train/valid split.  Replaces ``tsai.data.validation.get_splits``.

    Returns:
        (train_indices, valid_indices)
    """
    y_arr = np.asarray(y)
    if valid_size <= 0.0:
        return list(range(len(y_arr))), []
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=valid_size, random_state=random_state,
    )
    train_idx, valid_idx = next(sss.split(np.zeros(len(y_arr)), y_arr))
    return list(train_idx), list(valid_idx)


# ============================================================================
# Model save / load
# ============================================================================

def save_model(model: torch.nn.Module, model_name: str, save_dir: str = 'models'):
    """Save model state dict in plain PyTorch format."""
    from astra.utils import PROJECT_ROOT
    save_dir = str(PROJECT_ROOT / save_dir) if not os.path.isabs(save_dir) else save_dir
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save({'model': model.state_dict()}, path)
    logger.info(f"Model saved: {path}")


def load_model_state(model_name: str, save_dir: str = 'models') -> dict:
    """
    Load model state dict from a checkpoint.

    Supports:
      - ``{'model': state_dict}`` (new format)
      - Plain state dict

    Returns:
        state_dict: OrderedDict of parameter tensors
    """
    from astra.utils import PROJECT_ROOT
    save_dir = str(PROJECT_ROOT / save_dir) if not os.path.isabs(save_dir) else save_dir
    path = os.path.join(save_dir, f'{model_name}.pth')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    # Assume it's a plain state dict
    return checkpoint
