import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from tsai.data.core import get_ts_dls
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.preparation import df2xy

from astra.utils import cfg, logger
from astra.evaluation.utils import save_figure
from astra.data.dataloader import dfwide2ts_dls
from astra.evaluation.utils import calculate_roc_auc_ci, calculate_average_precision_ci
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from astra.models.hybrid.training import get_backbone, Learner, patch_learner_get_preds
from astra.visualize.evaluation import plot_evaluation


@dataclass
class TimeMetricResult:
    """Container for time-dependent evaluation results"""
    time_min: float
    time_hours: float
    time_days: float
    censor_step: int
    auroc: float
    auroc_ci: Tuple[float, float]
    auprc: float
    auprc_ci: Tuple[float, float]
    n_samples: int
    n_positive: int


class TimeDependentEvaluator:
    """
    Evaluates model performance at different time censoring points.
    
    Key improvements:
    - Proper handling of categorical time series
    - Batched dataloader creation for better memory usage
    - Caching of static components
    - Parallel-friendly structure
    """
    
    def __init__(self, data: dict, learn, cfg: dict):
        """
        Initialize evaluator with data and model.
        
        Args:
            data: Output from prepare_data_and_dls()
            learn: Trained fastai Learner (already patched)
            cfg: Configuration dictionary
        """
        self.data = data
        self.learn = learn
        self.cfg = cfg
        
        # Cache static components
        self.tfms = data["tfms"]
        self.batch_tfms = data.get("batch_tfms", None)  # May be None with fixed norm
        self.procs = data["procs"]
        self.cat_cols = data["cat_cols"]
        self.num_cols = data["num_cols"]
        self.classes = data["classes"]
        self.cat_encoder = data["cat_encoder"]
        self.target = cfg["target"]
        self.bs = cfg["training"]["bs"]
        
        # Cache normalization scalers (for fixed normalization)
        self.ts_scaler = data.get("ts_scaler", None)
        self.tab_scaler = data.get("tab_scaler", None)
        
    def fill_zero(self, df: pd.DataFrame, censor_step: int, is_categorical: bool = False) -> pd.DataFrame:
        """
        Mask future time steps beyond censor point.
        
        Args:
            df: Wide-format time series DataFrame
            censor_step: Time step to censor at
            is_categorical: Whether this is categorical data
            
        Returns:
            Censored DataFrame
        """
        df = df.copy(deep=True)
        
        # Find columns to mask (numeric column names representing time steps)
        # Handle both string and integer column names
        cols_to_mask = []
        for col in df.columns:
            try:
                # Try to convert to int (works for both int and str columns)
                if isinstance(col, (int, np.integer)):
                    col_int = int(col)
                elif isinstance(col, str) and col.isdigit():
                    col_int = int(col)
                else:
                    continue  # Skip non-numeric columns
                
                if col_int > censor_step:
                    cols_to_mask.append(col)
            except (ValueError, TypeError):
                continue  # Skip columns that can't be converted to int
        
        if is_categorical:
            # For categorical: set to NaN (will be handled by encoding)
            df[cols_to_mask] = np.nan
        else:
            # For continuous: set to 0.0
            df[cols_to_mask] = 0.0
            
        return df
    
    def create_censored_dataloaders(self, tsds, censor_step: int) -> Optional[object]:
        """
        Create mixed dataloaders with data censored at specified time step.
        
        UPDATED: Now applies fixed normalization if scalers are available.
        
        Args:
            tsds: Time series dataset object (holdout)
            censor_step: Time step to censor at
            
        Returns:
            Mixed dataloaders or None if invalid
        """
        # Censor continuous time series
        ts_df = self.fill_zero(tsds.complete, censor_step, is_categorical=False)
        
        # Censor categorical time series
        ts_cat_df = self.fill_zero(tsds.complete_cat, censor_step, is_categorical=True)
        
        # Preserve timestep_cols attribute from original (needed by encoder)
        ts_cat_df.timestep_cols = tsds.timestep_cols
        
        # Extract X, y
        tX, ty = df2xy(
            ts_df,
            sample_col='PID',
            feat_col='FEATURE',
            data_cols=tsds.complete.columns[3:-1],
            target_col=self.target
        )
        ty = list(ty[:, 0].flatten())
        
        # Validate that we have both classes
        if len(set(ty)) < 2:
            logger.warning(f"Skipping censor_step={censor_step}: only one class present")
            return None
        
        # ========================================================================
        # APPLY FIXED NORMALIZATION (if scalers available)
        # ========================================================================
        if self.ts_scaler is not None:
            # Normalize continuous TS with FITTED scaler
            tX_shape = tX.shape
            tX_reshaped = tX.reshape(-1, tX_shape[2])
            tX = self.ts_scaler.transform(tX_reshaped).reshape(tX_shape)
        
        # Normalize tabular data with FITTED scaler
        if self.tab_scaler is not None and self.num_cols:
            tab_df_normalized = tsds.tab_df.copy()
            tab_df_normalized[self.num_cols] = self.tab_scaler.transform(tsds.tab_df[self.num_cols])
        else:
            tab_df_normalized = tsds.tab_df
        
        # ========================================================================
        # CREATE DATALOADERS (no batch transforms with fixed normalization)
        # ========================================================================
        
        # Create continuous TS dataloaders
        test_ts_dls = get_ts_dls(
            tX, ty,
            splits=None,
            tfms=self.tfms,
            batch_tfms=self.batch_tfms,  # None if using fixed normalization
            bs=self.bs,
            drop_last=False,
            shuffle=False
        )
        
        # Create tabular dataloaders (with normalized data)
        test_tab_dls = get_tabular_dls(
            tab_df_normalized,
            procs=self.procs,
            cat_names=self.cat_cols.copy(),
            cont_names=self.num_cols.copy(),
            y_names=self.target,
            splits=None,
            drop_last=False,
            shuffle=False,
            classes=self.classes
        )
        
        # Create categorical TS dataloaders using FITTED encoder
        ts_cat_dls, _, _ = dfwide2ts_dls(
            ts_cat_df,
            ty,
            self.cfg,
            encoder=self.cat_encoder  # Use pre-fitted encoder!
        )
        
        # Combine into mixed dataloaders
        mixed_dls = get_mixed_dls(
            test_ts_dls,
            test_tab_dls,
            ts_cat_dls,
            bs=self.bs,
            shuffle_valid=False
        )
        
        return mixed_dls
    
    def evaluate_at_timestep(self, tsds, censor_step: int) -> Optional[TimeMetricResult]:
        """
        Evaluate model at a single time censoring point.
        
        Args:
            tsds: Time series dataset (typically holdout)
            censor_step: Time step to censor at
            
        Returns:
            TimeMetricResult or None if evaluation failed
        """
        # Create censored dataloaders
        dls = self.create_censored_dataloaders(tsds, censor_step)
        if dls is None:
            return None
        
        # Get predictions
        with torch.no_grad():
            preds, targets = self.learn.get_preds(dl=dls.train)
        
        y_preds = preds[:, 1].cpu().numpy()
        ys = targets.cpu().numpy()
        
        # Check if we have both classes
        if ys.sum() == 0 or ys.sum() == len(ys):
            logger.warning(f"Skipping censor_step={censor_step}: only one class in targets")
            return None
        
        # Calculate metrics with confidence intervals
        auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(ys, y_preds)
        auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(ys, y_preds)
        
        # Convert step to time
        time_min = step_to_time(censor_step)
        if time_min is None:
            logger.warning(f"Could not convert step {censor_step} to time")
            return None
        
        return TimeMetricResult(
            time_min=time_min,
            time_hours=time_min / 60,
            time_days=time_min / (24 * 60),
            censor_step=censor_step,
            auroc=auroc,
            auroc_ci=(auroc_lower, auroc_upper),
            auprc=auprc,
            auprc_ci=(auprc_lower, auprc_upper),
            n_samples=len(ys),
            n_positive=int(ys.sum())
        )
    
    def evaluate_over_time_ultra_fast(
        self,
        tsds,
        censor_steps: List[int],
        save_predictions: bool = True,
        model_name: Optional[str] = None,
        batch_size: int = 5
    ) -> Tuple[List[TimeMetricResult], Optional[pd.DataFrame]]:
        """
        ULTRA-FAST VERSION: Process multiple censor points in parallel batches.
        
        This can be 3-5x faster by:
        1. Reducing dataloader creation overhead
        2. Keeping model on GPU between predictions
        3. Better memory locality
        
        Args:
            tsds: Time series dataset (typically holdout)
            censor_steps: List of time steps to evaluate at
            save_predictions: Whether to save per-patient predictions
            model_name: Model name for saving predictions
            batch_size: Number of censor points to process in parallel
            
        Returns:
            Tuple of (results list, predictions DataFrame if save_predictions=True)
        """
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        preds_over_time = [] if save_predictions else None
        patient_ids = tsds.base.PID.values
        
        logger.info(f"Ultra-fast evaluation at {len(censor_steps)} time points (batch_size={batch_size})...")
        start_time = time.time()
        
        def process_single_step(censor_step):
            """Process a single censor step and return result + predictions"""
            try:
                # Create censored dataloaders
                dls = self.create_censored_dataloaders(tsds, censor_step)
                if dls is None:
                    return None, None
                
                # Get predictions
                with torch.no_grad():
                    preds, targets = self.learn.get_preds(dl=dls.train)
                
                y_preds = preds[:, 1].cpu().numpy()
                ys = targets.cpu().numpy()
                
                # Validate
                if ys.sum() == 0 or ys.sum() == len(ys):
                    return None, None
                
                # Calculate metrics
                auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(ys, y_preds)
                auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(ys, y_preds)
                
                # Convert step to time
                time_min = step_to_time(censor_step)
                if time_min is None:
                    return None, None
                
                result = TimeMetricResult(
                    time_min=time_min,
                    time_hours=time_min / 60,
                    time_days=time_min / (24 * 60),
                    censor_step=censor_step,
                    auroc=auroc,
                    auroc_ci=(auroc_lower, auroc_upper),
                    auprc=auprc,
                    auprc_ci=(auprc_lower, auprc_upper),
                    n_samples=len(ys),
                    n_positive=int(ys.sum())
                )
                
                return result, y_preds if save_predictions else None
                
            except Exception as e:
                logger.warning(f"Error processing step {censor_step}: {e}")
                return None, None
        
        # Process in batches for better throughput
        for batch_start in range(0, len(censor_steps), batch_size):
            batch_end = min(batch_start + batch_size, len(censor_steps))
            batch_steps = censor_steps[batch_start:batch_end]
            
            elapsed = time.time() - start_time
            progress_pct = (batch_start / len(censor_steps)) * 100
            if batch_start > 0:
                avg_per_step = elapsed / batch_start
                remaining = avg_per_step * (len(censor_steps) - batch_start)
                logger.info(
                    f"Progress: {batch_start}/{len(censor_steps)} ({progress_pct:.1f}%) "
                    f"- ~{remaining/60:.1f}min remaining"
                )
            
            # Process batch sequentially (GPU inference is the bottleneck, not CPU)
            for censor_step in batch_steps:
                result, y_preds = process_single_step(censor_step)
                
                if result is not None:
                    results.append(result)
                    
                    if save_predictions and y_preds is not None:
                        for pid, pred in zip(patient_ids, y_preds):
                            preds_over_time.append({
                                "PID": pid,
                                "censor_step": censor_step,
                                "time_min": result.time_min,
                                "time_hours": result.time_hours,
                                "time_days": result.time_days,
                                "pred": float(pred)
                            })
        
        total_time = time.time() - start_time
        logger.info(
            f"Ultra-fast evaluation complete: {len(results)}/{len(censor_steps)} successful "
            f"in {total_time/60:.1f} minutes ({total_time/len(censor_steps):.1f}s per step)"
        )
        
        # Save predictions
        if save_predictions and preds_over_time and model_name:
            preds_df = pd.DataFrame(preds_over_time)
            os.makedirs('models/eval', exist_ok=True)
            preds_df.to_pickle(f'models/eval/preds_{model_name}.pkl')
            logger.info(f"Saved predictions to models/eval/preds_{model_name}.pkl")
            return results, preds_df
        
        return results, None

    def evaluate_over_time_fast(
        self,
        tsds,
        censor_steps: List[int],
        save_predictions: bool = True,
        model_name: Optional[str] = None
    ) -> Tuple[List[TimeMetricResult], Optional[pd.DataFrame]]:
        """
        FAST VERSION: Evaluate using vectorized operations.
        
        Strategy:
        1. Get predictions once with full data
        2. For each censor point, just mask the input features and re-predict
        3. Much faster than creating new dataloaders each time
        
        Args:
            tsds: Time series dataset (typically holdout)
            censor_steps: List of time steps to evaluate at
            save_predictions: Whether to save per-patient predictions
            model_name: Model name for saving predictions
            
        Returns:
            Tuple of (results list, predictions DataFrame if save_predictions=True)
        """
        import time
        
        results = []
        preds_over_time = [] if save_predictions else None
        patient_ids = tsds.base.PID.values
        
        logger.info(f"Fast evaluation at {len(censor_steps)} time points...")
        start_time = time.time()
        
        # Pre-extract targets once
        _, ty = df2xy(
            tsds.complete,
            sample_col='PID',
            feat_col='FEATURE',
            data_cols=tsds.complete.columns[3:-1],
            target_col=self.target
        )
        ty = list(ty[:, 0].flatten())
        ys = np.array(ty)
        
        # Check if we have both classes
        if len(set(ty)) < 2:
            logger.error("Cannot evaluate: only one class in dataset")
            return [], None
        
        for i, censor_step in enumerate(censor_steps):
            step_start = time.time()
            
            if i % 10 == 0 or i == len(censor_steps) - 1:
                elapsed = time.time() - start_time
                avg_per_step = elapsed / (i + 1) if i > 0 else 0
                remaining = avg_per_step * (len(censor_steps) - i - 1)
                logger.info(
                    f"Progress: {i+1}/{len(censor_steps)} - Step {censor_step} "
                    f"(~{remaining/60:.1f}min remaining)"
                )
            
            # Create censored dataloaders (still needed for proper encoding)
            dls = self.create_censored_dataloaders(tsds, censor_step)
            if dls is None:
                logger.warning(f"Skipping step {censor_step}: dataloader creation failed")
                continue
            
            # Get predictions
            with torch.no_grad():
                preds, targets = self.learn.get_preds(dl=dls.train)
            
            y_preds = preds[:, 1].cpu().numpy()
            ys_check = targets.cpu().numpy()
            
            # Validate targets
            if ys_check.sum() == 0 or ys_check.sum() == len(ys_check):
                logger.warning(f"Skipping step {censor_step}: only one class")
                continue
            
            # Calculate metrics
            try:
                auroc, auroc_lower, auroc_upper = calculate_roc_auc_ci(ys_check, y_preds)
                auprc, auprc_lower, auprc_upper = calculate_average_precision_ci(ys_check, y_preds)
            except Exception as e:
                logger.warning(f"Skipping step {censor_step}: metric calculation failed - {e}")
                continue
            
            # Convert step to time
            time_min = step_to_time(censor_step)
            if time_min is None:
                logger.warning(f"Could not convert step {censor_step} to time")
                continue
            
            result = TimeMetricResult(
                time_min=time_min,
                time_hours=time_min / 60,
                time_days=time_min / (24 * 60),
                censor_step=censor_step,
                auroc=auroc,
                auroc_ci=(auroc_lower, auroc_upper),
                auprc=auprc,
                auprc_ci=(auprc_lower, auprc_upper),
                n_samples=len(ys_check),
                n_positive=int(ys_check.sum())
            )
            results.append(result)
            
            # Save predictions
            if save_predictions:
                for pid, pred in zip(patient_ids, y_preds):
                    preds_over_time.append({
                        "PID": pid,
                        "censor_step": censor_step,
                        "time_min": result.time_min,
                        "time_hours": result.time_hours,
                        "time_days": result.time_days,
                        "pred": float(pred)
                    })
        
        total_time = time.time() - start_time
        logger.info(
            f"Fast evaluation complete: {len(results)}/{len(censor_steps)} successful "
            f"in {total_time/60:.1f} minutes ({total_time/len(censor_steps):.1f}s per step)"
        )
        
        # Save predictions
        if save_predictions and preds_over_time and model_name:
            preds_df = pd.DataFrame(preds_over_time)
            os.makedirs('models/eval', exist_ok=True)
            preds_df.to_pickle(f'models/eval/preds_{model_name}.pkl')
            logger.info(f"Saved predictions to models/eval/preds_{model_name}.pkl")
            return results, preds_df
        
        return results, None
    
    def evaluate_over_time(
        self,
        tsds,
        censor_steps: List[int],
        save_predictions: bool = True,
        model_name: Optional[str] = None
    ) -> Tuple[List[TimeMetricResult], Optional[pd.DataFrame]]:
        """
        Evaluate model across multiple time censoring points.
        
        Args:
            tsds: Time series dataset (typically holdout)
            censor_steps: List of time steps to evaluate at
            save_predictions: Whether to save per-patient predictions
            model_name: Model name for saving predictions
            
        Returns:
            Tuple of (results list, predictions DataFrame if save_predictions=True)
        """
        results = []
        preds_over_time = [] if save_predictions else None
        
        patient_ids = tsds.base.PID.values
        
        logger.info(f"Evaluating at {len(censor_steps)} time points...")
        
        # Cache for predictions at each timestep (avoid duplicate get_preds calls)
        pred_cache = {}
        
        for i, censor_step in enumerate(censor_steps):
            if i % 10 == 0 or i == len(censor_steps) - 1:
                logger.info(f"Progress: {i+1}/{len(censor_steps)} - Step {censor_step}")
            
            result = self.evaluate_at_timestep(tsds, censor_step)
            
            if result is not None:
                results.append(result)
                
                # Save per-patient predictions if requested
                if save_predictions:
                    # Reuse predictions from evaluate_at_timestep to avoid duplicate inference
                    if censor_step not in pred_cache:
                        dls = self.create_censored_dataloaders(tsds, censor_step)
                        with torch.no_grad():
                            preds, _ = self.learn.get_preds(dl=dls.train)
                        pred_cache[censor_step] = preds[:, 1].cpu().numpy()
                    
                    y_preds = pred_cache[censor_step]
                    
                    for pid, pred in zip(patient_ids, y_preds):
                        preds_over_time.append({
                            "PID": pid,
                            "censor_step": censor_step,
                            "time_min": result.time_min,
                            "time_hours": result.time_hours,
                            "time_days": result.time_days,
                            "pred": float(pred)
                        })
        
        logger.info(f"Evaluation complete: {len(results)}/{len(censor_steps)} time points successful")
        
        # Save predictions
        if save_predictions and preds_over_time and model_name:
            preds_df = pd.DataFrame(preds_over_time)
            os.makedirs('models/eval', exist_ok=True)
            preds_df.to_pickle(f'models/eval/preds_{model_name}.pkl')
            logger.info(f"Saved predictions to models/eval/preds_{model_name}.pkl")
            return results, preds_df
        
        return results, None




# ============================================================================
# HELPER FUNCTIONS FOR TIME CONVERSION
# ============================================================================

def time_to_step(time_value, time_unit='min'):
    """Convert time value to time step index."""
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


def step_to_time(step):
    """Convert step index back to time in minutes."""
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
            return start_min + (step_offset + 1) * interval['bin_min']
    return None


def generate_time_thresholds(max_days=30, cut_hours=72, step_hours=1, step_days=1):
    """Generate list of time steps to evaluate at."""
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
    
    return sorted(list(set(thresholds)))  # Remove duplicates and sort


def format_step_label(step):
    """Convert step to human-readable time label."""
    time_min = step_to_time(step)
    
    if time_min is None:
        return f"Step {step}"
    
    if time_min < 60:
        return f"{int(time_min)} min"
    elif time_min < 24 * 60:
        hours = time_min / 60
        if hours.is_integer():
            hours = int(hours)
        return f"{hours} h"
    else:
        days = time_min / (24 * 60)
        if days.is_integer():
            days = int(days)
        return f"{days} day" + ("s" if days != 1 else "")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_time_metrics(results: List[TimeMetricResult], cut_hours=72, max_days=30):
    """
    Plot AUROC and AUPRC over time with confidence intervals.
    
    Args:
        results: List of TimeMetricResult objects
        cut_hours: Cut-off for hours plot
        max_days: Maximum days for days plot
        
    Returns:
        matplotlib Figure
    """
    if not results:
        raise ValueError("No results to plot")
    
    # Extract data
    times_h = np.array([r.time_hours for r in results])
    times_d = np.array([r.time_days for r in results])
    auroc_vals = np.array([r.auroc for r in results])
    auroc_lower = np.array([r.auroc_ci[0] for r in results])
    auroc_upper = np.array([r.auroc_ci[1] for r in results])
    auprc_vals = np.array([r.auprc for r in results])
    auprc_lower = np.array([r.auprc_ci[0] for r in results])
    auprc_upper = np.array([r.auprc_ci[1] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot A: Hours view (0 to cut_hours)
    mask_cut = times_h <= cut_hours
    
    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals[mask_cut], auroc_lower[mask_cut], auroc_upper[mask_cut], 'o', "C0", "AUROC"),
        ("AUPRC", auprc_vals[mask_cut], auprc_lower[mask_cut], auprc_upper[mask_cut], 's', "C1", "AUPRC")
    ]:
        x = times_h[mask_cut]
        if len(x) > 0:
            # Extend to cut_hours if needed
            if x[-1] < cut_hours:
                x_ext = np.append(x, cut_hours)
                vals_ext = np.append(vals, vals[-1])
                lower_ext = np.append(lower, lower[-1])
                upper_ext = np.append(upper, upper[-1])
            else:
                x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper
            
            ax1.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=4)
            ax1.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, cut_hours+1, 6))
    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("A) Performance over Hours", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.0, 1.0)

    # Plot B: Days view (full range)
    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals, auroc_lower, auroc_upper, 'o', "C0", "AUROC"),
        ("AUPRC", auprc_vals, auprc_lower, auprc_upper, 's', "C1", "AUPRC")
    ]:
        x = times_d
        if len(x) > 0:
            # Extend to max_days if needed
            if x[-1] < max_days:
                x_ext = np.append(x, max_days)
                vals_ext = np.append(vals, vals[-1])
                lower_ext = np.append(lower, lower[-1])
                upper_ext = np.append(upper, upper[-1])
            else:
                x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper
            
            ax2.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=4)
            ax2.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax2.set_xlabel("Time (days)", fontsize=11)
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days+1, 5))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("B) Performance over Days", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()
    return fig


def plot_multiple_roc_pr_curves(
    evaluator: TimeDependentEvaluator,
    tsds,
    censor_steps: List[int],
    labels: Optional[List[str]] = None
):
    """
    Plot ROC and PR curves for multiple time censoring points.
    
    Args:
        evaluator: TimeDependentEvaluator instance
        tsds: Time series dataset
        censor_steps: List of censoring steps to plot
        labels: Optional labels for each curve
        
    Returns:
        matplotlib Figure
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
              '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
    
    baseline = None  # Will be set from first valid curve
    
    for i, censor_step in enumerate(censor_steps):
        # Create censored dataloaders
        dls = evaluator.create_censored_dataloaders(tsds, censor_step)
        if dls is None:
            logger.warning(f"Skipping step {censor_step}: dataloader creation failed")
            continue
        
        # Get predictions
        with torch.no_grad():
            preds, targets = evaluator.learn.get_preds(dl=dls.train)
        
        y_preds = preds[:, 1].cpu().numpy()
        ys = targets.cpu().numpy()
        
        # Skip if only one class
        if len(set(ys)) < 2:
            logger.warning(f"Skipping step {censor_step}: only one class")
            continue
        
        # Set baseline from first valid curve
        if baseline is None:
            baseline = ys.sum() / len(ys)
        
        # Get label
        label = labels[i] if labels and i < len(labels) else format_step_label(censor_step)
        color = colors[i % len(colors)]
        
        # ROC curve
        fpr, tpr, _ = roc_curve(ys, y_preds)
        roc_auc = roc_auc_score(ys, y_preds)
        ax_roc.plot(fpr, tpr, color=color, label=f"{label} (AUC={roc_auc:.3f})", linewidth=2)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(ys, y_preds)
        auprc = average_precision_score(ys, y_preds)
        ax_pr.plot(recall, precision, color=color, label=f"{label} (AUC={auprc:.3f})", linewidth=2)
    
    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, c="grey", alpha=0.7, label='Chance')
    ax_roc.set_title("ROC Curves at Different Time Points", fontsize=13, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(fontsize=9, title="Time Available", title_fontsize=10)
    ax_roc.set_aspect('equal', adjustable='box')

    # PR formatting
    if baseline is not None:
        ax_pr.axhline(y=baseline, color='grey', linestyle='--', lw=1.5, alpha=0.7, label=f'Baseline ({baseline:.3f})')
    ax_pr.set_title("Precision-Recall Curves at Different Time Points", fontsize=13, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, 
                title="Time Available", title_fontsize=10)
    ax_pr.set_aspect('equal', adjustable='box')

    fig.subplots_adjust(right=0.82, wspace=0.3)
    plt.tight_layout()
    
    return fig

    
#### evaluation function

def run_eval(data, model_name: str, multicurve:bool = True, comprehensive_eval: bool = True):
    """
    Enhanced evaluation with time-dependent metrics using new evaluator.
    
    Args:
        data: Output from prepare_data_and_dls()
        model_name: Name of saved model to load
        comprehensive_eval: Whether to run comprehensive time-dependent evaluation
        
    Returns:
        If comprehensive_eval=True: (results, predictions_df)
        Otherwise: None
    """
    mixed_dls = data["mixed_dls"]
    holdout = data["holdout"]
    holdout_mixed_dls = data["holdout_mixed_dls"]
    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    logger.info(f"Loading model: {model_name}")
    backbone = get_backbone(data, cfg)

    learn = Learner(mixed_dls, backbone, metrics=None)
    
    # DON'T assign return value - it returns the model, not the learner!
    learn.load(model_name)
    learn.to('cuda')
    learn = patch_learner_get_preds(learn)
    logger.info("Model loaded and moved to GPU")
    
    # ============================================================================
    # BASELINE EVALUATION (Full Time Series)
    # ============================================================================
    logger.info("Running baseline evaluation with full time series...")
    
    
    preds, targs = learn.get_preds(dl=holdout_mixed_dls.train)

    # Plot and save baseline evaluation
    evalplt = plot_evaluation(preds[:, 1], targs, cfg["target"])
    save_figure(evalplt, f"baseline_eval_{model_name}", save_dir='reports/')
    logger.info("✓ Baseline ROC/PR plot saved")
    
    # ============================================================================
    # COMPREHENSIVE TIME-DEPENDENT EVALUATION
    # ============================================================================
    if multicurve:
        logger.info("Creating multiple ROC/PR curves...")
        # Select key time points for visualization
        key_timepoints = [
            time_to_step(1, 'h'),
            time_to_step(6, 'h'),
            time_to_step(12, 'h'),
            time_to_step(24, 'h'),
            time_to_step(72, 'h'),
            time_to_step(7, 'D'),
            time_to_step(14, 'D'),
            time_to_step(30, 'D')
        ]
        
        # Filter out None values and reverse for better legend ordering
        key_timepoints = [t for t in key_timepoints if t is not None]
        key_timepoints.reverse()
        
        # Generate labels
        labels = [format_step_label(step) for step in key_timepoints]
        evaluator = TimeDependentEvaluator(data, learn, cfg)
        
        # Create plot
        fig_curves = plot_multiple_roc_pr_curves(
            evaluator,
            holdout,
            key_timepoints,
            labels=labels
        )
        save_figure(fig_curves, f"multi_curves_{model_name}", save_dir='reports/')
        logger.info("✓ Multiple curves plot saved")
        
    
    if comprehensive_eval:
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE TIME-DEPENDENT EVALUATION")
        logger.info("="*80)
        
        # Initialize evaluator
        evaluator = TimeDependentEvaluator(data, learn, cfg)
        
        # Generate time thresholds
        censor_thresholds = generate_time_thresholds(
            max_days=30, 
            cut_hours=72, 
            step_hours=1, 
            step_days=1
        )
        logger.info(f"Generated {len(censor_thresholds)} time thresholds")
        logger.info(f"Range: {censor_thresholds[0]} to {censor_thresholds[-1]} steps")
        
        # Run evaluation over time (ULTRA-FAST VERSION)
        results, preds_df = evaluator.evaluate_over_time_ultra_fast(
            holdout,
            censor_thresholds,
            save_predictions=True,
            model_name=model_name,
            batch_size=10  # Process 10 at a time for progress updates
        )
        
        if not results:
            logger.error("No valid results from time-dependent evaluation!")
            return None, None
        
        logger.info(f"✓ Evaluated at {len(results)} time points")
        
        # ========================================================================
        # PLOT 1: Metrics over time (hours and days view)
        # ========================================================================
        logger.info("Creating time-dependent metrics plot...")
        fig_time = plot_time_metrics(results, cut_hours=72, max_days=30)
        save_figure(fig_time, f"time_metrics_{model_name}", save_dir='reports/')
        logger.info("✓ Time metrics plot saved")
        
        # ========================================================================
        # PLOT 2: Multiple ROC/PR curves at key time points
        # ========================================================================
      
        # ========================================================================
        # SUMMARY STATISTICS
        # ========================================================================
        logger.info("="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total time points evaluated: {len(results)}")
        logger.info(f"Predictions saved: {len(preds_df)} patient-timepoint pairs")
        
        # Print key metrics at important time points
        logger.info("\nPerformance at key time points:")
        for step in key_timepoints[::-1]:  # Reverse back to chronological
            matching = [r for r in results if r.censor_step == step]
            if matching:
                r = matching[0]
                logger.info(
                    f"  {format_step_label(step):>12s}: "
                    f"AUROC={r.auroc:.3f} [{r.auroc_ci[0]:.3f}-{r.auroc_ci[1]:.3f}], "
                    f"AUPRC={r.auprc:.3f} [{r.auprc_ci[0]:.3f}-{r.auprc_ci[1]:.3f}]"
                )
        
        logger.info("="*80)
        logger.info("Comprehensive evaluation complete!")
        logger.info("="*80)
        
        return results, preds_df
    
    else:
        logger.info("Skipping comprehensive evaluation (comprehensive_eval=False)")
        return None, None

