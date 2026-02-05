# train_ebm_over_time.py
"""
Train multiple EBM models at different time masking points for temporal comparison with hybrid models.

This script:
1. Generates time thresholds matching the hybrid model evaluation
2. For each time point, creates an AggregatedDS with corresponding masking point
3. Trains an EBM model on that aggregated dataset
4. Saves the model and evaluation metrics
5. Creates a summary of performance over time

Usage:
    python train_ebm_over_time.py --max_days 30 --step_hours 6 --step_days 1
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from interpret.glassbox import ExplainableBoostingClassifier

from astra.utils import get_base_df, get_train_test_split, cfg, logger
from astra.data.datasets import AggregatedDS


def step_to_timedelta(step: int) -> pd.Timedelta:
    """
    Convert step index to pandas Timedelta for masking point.

    Uses the same time binning scheme as the hybrid model evaluation.
    """
    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]

    bins_cum = [0]
    for interval in intervals[:-1]:
        duration_min = (interval['end_h'] - interval['start_h']) * 60
        bins = duration_min // interval['bin_min']
        bins_cum.append(bins_cum[-1] + bins)

    for i in range(len(bins_cum) - 1):
        if bins_cum[i] <= step < bins_cum[i+1]:
            interval = intervals[i]
            step_offset = step - bins_cum[i]
            start_min = interval['start_h'] * 60
            time_min = start_min + (step_offset + 1) * interval['bin_min']
            return pd.Timedelta(minutes=time_min)

    # Last interval
    interval = intervals[-1]
    step_offset = step - bins_cum[-1]
    start_min = interval['start_h'] * 60
    time_min = start_min + (step_offset + 1) * interval['bin_min']
    return pd.Timedelta(minutes=time_min)


def time_to_step(time_value, time_unit='min'):
    """Convert time value to time step index (matching hybrid model evaluation)."""
    if time_unit == 'min':
        time_min = time_value
    elif time_unit == 'h':
        time_min = time_value * 60
    elif time_unit == 'D':
        time_min = time_value * 24 * 60
    else:
        raise ValueError("Unsupported time unit. Use 'min', 'h' or 'D'.")

    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]

    for i, interval in enumerate(intervals):
        start_min = interval['start_h'] * 60
        end_min = interval['end_h'] * 60 if interval['end_h'] is not None else float('inf')
        if start_min < time_min <= end_min:
            offset_min = time_min - start_min
            step_offset = int(np.ceil(offset_min / interval['bin_min'])) - 1
            bins_cum = 0
            for j in range(i):
                duration_min = (intervals[j]['end_h'] - intervals[j]['start_h']) * 60
                bins_cum += duration_min // intervals[j]['bin_min']
            return bins_cum + step_offset
    return None


def generate_time_thresholds(max_days=30, cut_hours=72, step_hours=1, step_days=1):
    """Generate list of time steps to evaluate at (matching hybrid model evaluation)."""
    thresholds = []

    # Hourly steps up to cut_hours
    for h in range(step_hours, cut_hours+1, step_hours):
        step = time_to_step(h, 'h')
        if step is not None:
            thresholds.append(step)

    # Daily steps after cut_hours
    start_day = int(np.ceil(cut_hours/24))
    for d in range(start_day+1, max_days+1, step_days):
        step = time_to_step(d, 'D')
        if step is not None:
            thresholds.append(step)

    return sorted(list(set(thresholds)))


def format_step_label(step):
    """Convert step to human-readable time label."""
    timedelta = step_to_timedelta(step)
    time_min = timedelta.total_seconds() / 60

    if time_min < 60:
        return f"{int(time_min)}min"
    elif time_min < 24 * 60:
        hours = time_min / 60
        if hours.is_integer():
            hours = int(hours)
        return f"{hours}h"
    else:
        days = time_min / (24 * 60)
        if days.is_integer():
            days = int(days)
        return f"{days}D"


def temporal_train_val_split(X, y, val_frac=0.25):
    """
    Temporal train/validation split (preserves time ordering).

    Note: This is for train/val split within the training set.
    The train/test split (base_df level) is handled by get_train_test_split().
    """
    n = len(X)
    split_idx = int((1 - val_frac) * n)

    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]

    y_train = np.asarray(y[:split_idx])
    y_val = np.asarray(y[split_idx:])

    return X_train, X_val, y_train, y_val


def train_ebm_at_timepoint(
    base_df: pd.DataFrame,
    cfg: dict,
    masking_point: pd.Timedelta,
    step: int,
    val_frac: float = 0.20,
    ebm_params: dict = None,
    save_dir: str = "models/ebm",
    no_preprocess: bool = True
) -> Dict:
    """
    Train an EBM model at a specific time masking point.

    Args:
        base_df: Base dataframe with patient data
        cfg: Configuration dictionary
        masking_point: Time to mask data at (pd.Timedelta)
        step: Time step index (for naming/tracking)
        val_frac: Validation fraction for temporal split
        ebm_params: EBM hyperparameters (dict)
        save_dir: Directory to save models
        no_preprocess: Whether to skip preprocessing (passthrough)

    Returns:
        Dictionary with training results
    """
    time_label = format_step_label(step)
    logger.info(f"\n{'='*80}")
    logger.info(f"Training EBM at time point: {time_label} (step={step}, masking={masking_point})")
    logger.info(f"{'='*80}")

    # ============================================================================
    # CREATE AGGREGATED DATASET
    # ============================================================================
    logger.info("Creating aggregated dataset...")
    agg_ds = AggregatedDS(
        cfg=cfg,
        base_df=base_df,
        masking_point=masking_point,
        agg_funcs=['first', 'last', 'min', 'max', 'mean', 'std'],
        concepts=cfg["concepts"],
        default_mode=True,
    )

    X, y = agg_ds.get_X_y()
    model_X = X.copy(deep=True)

    logger.info(f"Dataset shape: {model_X.shape}")
    logger.info(f"Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")

    categorical_features = agg_ds.categorical_features
    continuous_features = agg_ds.continuous_features

    logger.info(f"Features: {len(categorical_features)} categorical + {len(continuous_features)} continuous")

    # ============================================================================
    # PREPROCESSING PIPELINE
    # ============================================================================
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ]
    )

    continuous_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    if no_preprocess:
        logger.info("Using passthrough preprocessing")
        categorical_pipeline = 'passthrough'
        continuous_pipeline = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("cont", continuous_pipeline, continuous_features),
        ],
        remainder="drop"
    )

    # ============================================================================
    # TRAIN/VAL SPLIT (TEMPORAL)
    # ============================================================================
    X_train, X_val, y_train, y_val = temporal_train_val_split(
        model_X, y, val_frac=val_frac
    )

    logger.info(f"Train: {len(X_train)} samples ({y_train.sum()} positive)")
    logger.info(f"Val:   {len(X_val)} samples ({y_val.sum()} positive)")

    # Check if we have both classes
    if len(set(y_train)) < 2 or len(set(y_val)) < 2:
        logger.warning(f"Skipping {time_label}: insufficient class diversity")
        return None

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # ============================================================================
    # TRAIN EBM
    # ============================================================================
    logger.info("Training EBM model...")

    # Default EBM parameters
    if ebm_params is None:
        ebm_params = {
            'random_state': 42,
            'interactions': 3,
            'validation_size': 0.2,
            'early_stopping_rounds': 100,
            'max_leaves': 2,
            'inner_bags': 0,
        }

    feature_names = categorical_features + continuous_features

    ebm = ExplainableBoostingClassifier(
        feature_names=feature_names,
        **ebm_params
    )

    ebm.fit(X_train_proc, y_train)
    logger.info("✓ EBM training complete")

    # ============================================================================
    # EVALUATE
    # ============================================================================
    logger.info("Evaluating on validation set...")

    y_proba = ebm.predict_proba(X_val_proc)[:, 1]
    y_pred = ebm.predict(X_val_proc)
    y_val_bin = np.array(y_val).round().astype(int)

    # Calculate metrics
    auroc = roc_auc_score(y_val_bin, y_proba)
    auprc = average_precision_score(y_val_bin, y_proba)

    logger.info(f"Results: AUROC={auroc:.3f}, AUPRC={auprc:.3f}")

    # ============================================================================
    # SAVE MODEL
    # ============================================================================
    os.makedirs(save_dir, exist_ok=True)

    model_filename = f"ebm_step{step:04d}_{time_label.replace(' ', '')}.pkl"
    model_path = os.path.join(save_dir, model_filename)

    # Save using pickle
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': ebm,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'categorical_features': categorical_features,
            'continuous_features': continuous_features,
            'step': step,
            'masking_point': masking_point,
            'time_label': time_label,
        }, f)

    logger.info(f"✓ Model saved: {model_path}")

    # ============================================================================
    # RETURN RESULTS
    # ============================================================================
    return {
        'step': step,
        'time_label': time_label,
        'masking_point': str(masking_point),
        'auroc': auroc,
        'auprc': auprc,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_positive_train': int(y_train.sum()),
        'n_positive_val': int(y_val.sum()),
        'model_path': model_path,
    }


def train_multiple_ebms(
    max_days: int = 30,
    cut_hours: int = 72,
    step_hours: int = 1,
    step_days: int = 1,
    val_frac: float = 0.20,
    ebm_params: dict = None,
    save_dir: str = "models/ebm",
    subsample_steps: int = None,
):
    """
    Train multiple EBM models at different time points.

    Args:
        max_days: Maximum days for evaluation
        cut_hours: Hours cutoff for hourly steps
        step_hours: Step size for hourly evaluations
        step_days: Step size for daily evaluations
        val_frac: Validation fraction
        ebm_params: EBM hyperparameters
        save_dir: Directory to save models
        subsample_steps: If provided, only train at every Nth step (for faster testing)

    Returns:
        DataFrame with results for all trained models
    """
    logger.info("="*80)
    logger.info("TRAINING EBM MODELS OVER TIME")
    logger.info("="*80)

    # ============================================================================
    # LOAD BASE DATA AND SPLIT (matching hybrid model)
    # ============================================================================
    logger.info("Loading base dataframe and applying train/test split...")
    base_df_full = get_base_df()

    # Use the same split as hybrid model (from config)
    train_df, test_df = get_train_test_split(cfg, base_df_full)

    logger.info(f"✓ Total patients: {len(base_df_full)}")
    logger.info(f"  Training set: {len(train_df)} patients")
    logger.info(f"  Test set:     {len(test_df)} patients (will be used in evaluate_ebm_over_time.py)")
    logger.info(f"Split strategy: {cfg.get('holdout_type', 'temporal')}")

    # Use only training data for EBM training
    base_df = train_df

    # ============================================================================
    # GENERATE TIME THRESHOLDS
    # ============================================================================
    logger.info(f"\nGenerating time thresholds (max_days={max_days}, cut_hours={cut_hours})...")
    time_steps = generate_time_thresholds(
        max_days=max_days,
        cut_hours=cut_hours,
        step_hours=step_hours,
        step_days=step_days
    )

    # Subsample if requested (for faster testing)
    if subsample_steps is not None and subsample_steps > 1:
        time_steps = time_steps[::subsample_steps]
        logger.info(f"Subsampled to every {subsample_steps}th step")

    logger.info(f"✓ Will train {len(time_steps)} EBM models")
    logger.info(f"Time range: {format_step_label(time_steps[0])} to {format_step_label(time_steps[-1])}")

    # ============================================================================
    # TRAIN MODELS
    # ============================================================================
    results = []
    failed_steps = []

    for i, step in enumerate(time_steps):
        logger.info(f"\n[{i+1}/{len(time_steps)}] Processing step {step}...")

        # Convert step to masking point
        masking_point = step_to_timedelta(step)

        try:
            result = train_ebm_at_timepoint(
                base_df=base_df,
                cfg=cfg,
                masking_point=masking_point,
                step=step,
                val_frac=val_frac,
                ebm_params=ebm_params,
                save_dir=save_dir,
            )

            if result is not None:
                results.append(result)
            else:
                failed_steps.append(step)

        except Exception as e:
            logger.error(f"Failed to train at step {step}: {e}")
            failed_steps.append(step)
            continue

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Successfully trained: {len(results)}/{len(time_steps)} models")
    if failed_steps:
        logger.info(f"Failed steps: {failed_steps}")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n✓ Training results saved: {results_path}")

    # Print summary
    logger.info("\nPerformance Summary:")
    logger.info("-"*80)
    logger.info(f"{'Time':>12} | {'AUROC':>6} | {'AUPRC':>6} | {'Train N':>8} | {'Val N':>8}")
    logger.info("-"*80)
    for _, row in results_df.iterrows():
        logger.info(
            f"{row['time_label']:>12} | {row['auroc']:>6.3f} | {row['auprc']:>6.3f} | "
            f"{row['n_train']:>8} | {row['n_val']:>8}"
        )
    logger.info("-"*80)

    # Summary statistics
    logger.info(f"\nMean AUROC: {results_df['auroc'].mean():.3f} ± {results_df['auroc'].std():.3f}")
    logger.info(f"Mean AUPRC: {results_df['auprc'].mean():.3f} ± {results_df['auprc'].std():.3f}")

    return results_df


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Train EBM models at multiple time points')
    parser.add_argument('--max_days', type=int, default=30, help='Maximum days for evaluation')
    parser.add_argument('--cut_hours', type=int, default=72, help='Hours cutoff for hourly steps')
    parser.add_argument('--step_hours', type=int, default=1, help='Step size for hourly evaluations')
    parser.add_argument('--step_days', type=int, default=1, help='Step size for daily evaluations')
    parser.add_argument('--val_frac', type=float, default=0.20, help='Validation fraction')
    parser.add_argument('--save_dir', type=str, default='models/ebm', help='Directory to save models')
    parser.add_argument('--subsample_steps', type=int, default=None, help='Train at every Nth step (for testing)')

    # EBM hyperparameters
    parser.add_argument('--interactions', type=int, default=3, help='EBM interactions depth')
    parser.add_argument('--max_leaves', type=int, default=2, help='EBM max leaves')
    parser.add_argument('--early_stopping_rounds', type=int, default=100, help='EBM early stopping')

    args = parser.parse_args()

    # Build EBM params
    ebm_params = {
        'random_state': 42,
        'interactions': args.interactions,
        'validation_size': 0.2,
        'early_stopping_rounds': args.early_stopping_rounds,
        'max_leaves': args.max_leaves,
        'inner_bags': 0,
    }

    # Run training
    results_df = train_multiple_ebms(
        max_days=args.max_days,
        cut_hours=args.cut_hours,
        step_hours=args.step_hours,
        step_days=args.step_days,
        val_frac=args.val_frac,
        ebm_params=ebm_params,
        save_dir=args.save_dir,
        subsample_steps=args.subsample_steps,
    )

    logger.info("\n" + "="*80)
    logger.info("ALL DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
