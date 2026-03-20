"""
Survival-specific evaluation metrics for discrete-time hazard models.

Provides:
  - concordance_index: Harrell's C-index with bootstrap CI
  - time_dependent_auc: Cumulative/dynamic AUC at multiple timepoints
  - brier_score_survival: IPC-weighted time-dependent Brier score
  - dcalibration: D-calibration for survival models
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def concordance_index(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Compute Harrell's concordance index with bootstrap 95% CI.

    Args:
        event_times: [N] time to event or censoring.
        event_indicators: [N] 1 = event observed, 0 = censored.
        risk_scores: [N] predicted risk (higher = more risk).
        n_bootstrap: Number of bootstrap iterations for CI.
        seed: Random seed.

    Returns:
        (cindex, (ci_lower, ci_upper))
    """
    cindex = _compute_cindex(event_times, event_indicators, risk_scores)

    # Bootstrap CI
    rng = np.random.RandomState(seed)
    n = len(event_times)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            c = _compute_cindex(event_times[idx], event_indicators[idx], risk_scores[idx])
            boot_vals.append(c)
        except Exception:
            continue

    if boot_vals:
        ci_lower = float(np.percentile(boot_vals, 2.5))
        ci_upper = float(np.percentile(boot_vals, 97.5))
    else:
        ci_lower = ci_upper = cindex

    return float(cindex), (ci_lower, ci_upper)


def _compute_cindex(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Compute Harrell's C-index.

    Tries lifelines first, falls back to manual computation.
    """
    try:
        from lifelines.utils import concordance_index as _lf_cindex
        return float(_lf_cindex(event_times, -risk_scores, event_indicators))
    except ImportError:
        pass

    # Manual implementation
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied_risk = 0
    for i in range(n):
        if event_indicators[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[j] > event_times[i]:
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied_risk += 1
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def time_dependent_auc(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    survival_probs: np.ndarray,
    eval_times: np.ndarray,
) -> List[Tuple[float, float]]:
    """Compute cumulative/dynamic AUC at multiple evaluation times.

    Uses scikit-survival if available, otherwise falls back to binary AUROC
    at each timepoint (treating event_by_time_t as binary).

    Args:
        event_times: [N] time to event or censoring (in steps).
        event_indicators: [N] 1 = event, 0 = censored.
        survival_probs: [N, seq_len] survival probabilities S(t).
        eval_times: [K] evaluation time points (step indices).

    Returns:
        List of (eval_time, auc) tuples.
    """
    results = []

    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv

        # Convert to structured array for sksurv
        y_struct = Surv.from_arrays(event_indicators.astype(bool), event_times.astype(float))

        # Risk scores: 1 - S(t) at each evaluation time
        for t in eval_times:
            t_int = int(t)
            if t_int >= survival_probs.shape[1]:
                continue
            risk_at_t = 1.0 - survival_probs[:, t_int]

            try:
                auc_vals, mean_auc = cumulative_dynamic_auc(
                    y_struct, y_struct, risk_at_t, times=[float(t)]
                )
                results.append((float(t), float(auc_vals[0])))
            except Exception:
                # Fallback to binary AUROC
                _auc = _binary_auroc_at_time(event_times, event_indicators, risk_at_t, t_int)
                results.append((float(t), _auc))

    except ImportError:
        logger.info("scikit-survival not installed, using binary AUROC fallback for td-AUC")
        for t in eval_times:
            t_int = int(t)
            if t_int >= survival_probs.shape[1]:
                continue
            risk_at_t = 1.0 - survival_probs[:, t_int]
            _auc = _binary_auroc_at_time(event_times, event_indicators, risk_at_t, t_int)
            results.append((float(t), _auc))

    return results


def _binary_auroc_at_time(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
    eval_time: int,
) -> float:
    """Fallback: binary AUROC treating 'event by time t' as positive class."""
    from sklearn.metrics import roc_auc_score

    # Binary outcome: event occurred by eval_time
    event_by_t = ((event_times <= eval_time) & (event_indicators == 1)).astype(int)
    # Exclude patients censored before eval_time (ambiguous)
    include = (event_indicators == 1) | (event_times > eval_time)
    if include.sum() < 10 or len(np.unique(event_by_t[include])) < 2:
        return 0.5
    try:
        return float(roc_auc_score(event_by_t[include], risk_scores[include]))
    except ValueError:
        return 0.5


def brier_score_survival(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    survival_probs: np.ndarray,
    eval_times: np.ndarray,
) -> List[Tuple[float, float]]:
    """Compute IPC-weighted Brier score at multiple evaluation times.

    Args:
        event_times: [N] time to event or censoring (in steps).
        event_indicators: [N] 1 = event, 0 = censored.
        survival_probs: [N, seq_len] survival probabilities S(t).
        eval_times: [K] evaluation time points (step indices).

    Returns:
        List of (eval_time, brier_score) tuples.
    """
    results = []

    try:
        from sksurv.metrics import brier_score as sksurv_brier
        from sksurv.util import Surv

        y_struct = Surv.from_arrays(event_indicators.astype(bool), event_times.astype(float))

        for t in eval_times:
            t_int = int(t)
            if t_int >= survival_probs.shape[1]:
                continue
            surv_at_t = survival_probs[:, t_int]
            try:
                _, bs = sksurv_brier(y_struct, y_struct, surv_at_t, times=[float(t)])
                results.append((float(t), float(bs[0])))
            except Exception as e:
                logger.debug(f"Brier score at t={t} failed: {e}")

    except ImportError:
        logger.info("scikit-survival not installed, skipping Brier score computation")

    return results


def dcalibration(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    survival_probs: np.ndarray,
    eval_time: int,
    n_bins: int = 10,
) -> Dict[str, float]:
    """D-calibration: compare predicted S(t) vs observed survival in risk bins.

    Groups patients into bins by predicted S(eval_time), then computes
    observed survival (Kaplan-Meier) within each bin.

    Args:
        event_times: [N] time to event or censoring.
        event_indicators: [N] 1 = event, 0 = censored.
        survival_probs: [N, seq_len] survival probabilities S(t).
        eval_time: Step at which to evaluate calibration.
        n_bins: Number of risk stratification bins.

    Returns:
        Dict with 'predicted_survival', 'observed_survival', 'bin_counts'.
    """
    if eval_time >= survival_probs.shape[1]:
        return {"predicted_survival": [], "observed_survival": [], "bin_counts": []}

    pred_surv = survival_probs[:, eval_time]
    bin_edges = np.linspace(0, 1, n_bins + 1)

    predicted = []
    observed = []
    counts = []

    for i in range(n_bins):
        mask = (pred_surv >= bin_edges[i]) & (pred_surv < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (pred_surv == bin_edges[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin < 5:
            continue

        predicted.append(float(pred_surv[mask].mean()))
        counts.append(int(n_in_bin))

        # Observed survival: fraction with event_time > eval_time or censored after eval_time
        bin_et = event_times[mask]
        bin_ei = event_indicators[mask]
        # Simple estimate: fraction not having event by eval_time
        n_events_by_t = ((bin_et <= eval_time) & (bin_ei == 1)).sum()
        observed.append(1.0 - n_events_by_t / n_in_bin)

    return {
        "predicted_survival": predicted,
        "observed_survival": observed,
        "bin_counts": counts,
    }
