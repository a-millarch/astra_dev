from astra.utils import logger, cfg
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================================
# Time ↔ step utilities (reads bin grid from config)
# ============================================================================

def _parse_timedelta_to_minutes(s):
    """Parse a time string like '3h', '5min', '14D' to minutes."""
    s = s.strip()
    if s.endswith('min'):
        return int(s[:-3])
    elif s.endswith('h'):
        return int(s[:-1]) * 60
    elif s.endswith('D'):
        return int(s[:-1]) * 24 * 60
    else:
        raise ValueError(f"Cannot parse time string: {s}")


def _get_intervals(bin_intervals, bin_freq_include=None):
    """
    Parse bin_intervals into a list of (start_min, end_min, bin_min) tuples,
    filtered by bin_freq_include.

    Args:
        bin_intervals: OrderedDict mapping interval endpoints (e.g. '3h', '14D', 'end')
            to bin frequencies (e.g. '5min', '10min', '1h').
        bin_freq_include: Optional list of frequency strings to keep.  When set,
            intervals whose frequency is not in the list are skipped (but
            their time span still advances ``start_min`` so that later
            intervals get the correct offset).
    """
    intervals = []
    start_min = 0

    for end_str, freq_str in bin_intervals.items():
        end_min = None if end_str == "end" else _parse_timedelta_to_minutes(end_str)
        bin_min = _parse_timedelta_to_minutes(freq_str)
        if bin_freq_include is None or freq_str in bin_freq_include:
            intervals.append((start_min, end_min, bin_min))
        if end_min is not None:
            start_min = end_min

    return intervals


def _get_intervals_from_cfg():
    """
    Parse cfg['bin_intervals'] into a list of (start_min, end_min, bin_min) tuples,
    respecting cfg['bin_freq_include'] filter.
    """
    return _get_intervals(
        cfg["bin_intervals"],
        cfg.get("bin_freq_include"),
    )


def time_to_step(time_value, time_unit='min', data_config=None):
    """Convert time value to time step index using bin intervals.

    Args:
        time_value: Numeric time offset from admission start.
        time_unit: ``'min'``, ``'h'`` or ``'D'``.
        data_config: Optional dict with ``'bin_intervals'`` and
            ``'bin_freq_include'`` keys (e.g. from a deployment bundle).
            When *None*, reads from the global ``cfg``.
    """
    if time_unit == 'min':
        time_min = time_value
    elif time_unit == 'h':
        time_min = time_value * 60
    elif time_unit == 'D':
        time_min = time_value * 24 * 60
    else:
        raise ValueError("Unsupported time unit. Use 'min', 'h' or 'D'.")

    if time_min <= 0:
        return 0

    if data_config is not None:
        intervals = _get_intervals(
            data_config['bin_intervals'],
            data_config.get('bin_freq_include'),
        )
    else:
        intervals = _get_intervals_from_cfg()

    for i, (start_min, end_min, bin_min) in enumerate(intervals):
        eff_end = end_min if end_min is not None else float('inf')
        if start_min < time_min <= eff_end:
            offset_min = time_min - start_min
            step_offset = int(np.ceil(offset_min / bin_min)) - 1
            bins_cum = 0
            for j in range(i):
                s, e, b = intervals[j]
                if e is not None:
                    bins_cum += (e - s) // b
            return bins_cum + step_offset
    return None


def step_to_time(step, data_config=None):
    """Convert step index back to time in minutes using bin intervals.

    Args:
        step: 0-based step index.
        data_config: Optional dict with ``'bin_intervals'`` and
            ``'bin_freq_include'`` keys.  When *None*, reads from
            the global ``cfg``.
    """
    if data_config is not None:
        intervals = _get_intervals(
            data_config['bin_intervals'],
            data_config.get('bin_freq_include'),
        )
    else:
        intervals = _get_intervals_from_cfg()

    bins_cum = [0]
    for start_min, end_min, bin_min in intervals[:-1]:
        if end_min is not None:
            duration = end_min - start_min
            bins_cum.append(bins_cum[-1] + duration // bin_min)

    for i in range(len(bins_cum) - 1):
        if bins_cum[i] <= step < bins_cum[i + 1]:
            start_min, end_min, bin_min = intervals[i]
            step_offset = step - bins_cum[i]
            return start_min + (step_offset + 1) * bin_min
    return None


def time_to_hours(minutes):
    """Format a time in minutes to a human-readable string (e.g. '6.0h' or '2.5d')."""
    if minutes is None:
        return "N/A"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    else:
        return f"{hours/24:.1f}d"

def prepare_learner(data, cfg):
    import torch
    from astra.models.hybrid.training import get_backbone, Learner, patch_learner_get_preds
    model_name = cfg["model_name"]
    # Detect temporal head config
    model_cfg = cfg.get("model", {})
    is_temporal = model_cfg.get("temporal_head", False)

    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    logger.info(f"Loading model: {model_name}")
    backbone = get_backbone(
        data, cfg,
        temporal_head=is_temporal,
        causal=model_cfg.get("causal", False),
        temporal_head_dropout=model_cfg.get("temporal_head_dropout", 0.3),
    )
    learn = Learner(data["holdout_mixed_dls"], backbone, metrics=None)
    learn.load(model_name, strict=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learn.to(device)
    learn = patch_learner_get_preds(learn)
    logger.info(f"Model loaded (temporal_head={is_temporal})")
    return learn

def delong_roc_variance(ground_truth, predictions):
    order = np.argsort(predictions)
    ground_truth = ground_truth[order]
    predictions = predictions[order]
    n_pos = np.sum(ground_truth)
    n_neg = len(ground_truth) - n_pos
    pos_ranks = np.where(ground_truth == 1)[0] + 1
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    v01 = (auc / (2 - auc) - auc ** 2) / n_neg
    v10 = (2 * auc ** 2 / (1 + auc) - auc ** 2) / n_pos
    return v01 + v10

def calculate_roc_auc_ci(y_true, y_pred, alpha=0.95):
    auc = roc_auc_score(y_true, y_pred)
    auc_var = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_var)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    ci[ci < 0] = 0
    return auc, ci[0], ci[1]

def calculate_average_precision_ci(y_true, y_pred, alpha=0.95, n_bootstraps=1000):
    ap = average_precision_score(y_true, y_pred)
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = average_precision_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(np.array(bootstrapped_scores))
    ci_lower = sorted_scores[int((1.0-alpha)/2 * len(sorted_scores))]
    ci_upper = sorted_scores[int((1.0+alpha)/2 * len(sorted_scores))]
    return ap, float(ci_lower), float(ci_upper)