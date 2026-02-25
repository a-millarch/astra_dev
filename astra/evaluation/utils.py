import logging
import math

from astra.utils import cfg
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


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


def check_bin_alignment(bin_intervals=None, bin_freq_include=None):
    """Check that all bin intervals divide evenly (no partial last bins).

    Prints a table of all active intervals and flags any that are not exact.
    Returns True if all intervals are aligned, False otherwise.

    Args:
        bin_intervals: OrderedDict of interval config (defaults to cfg['bin_intervals']).
        bin_freq_include: List of frequency strings to include (defaults to cfg value).

    Example::

        from astra.evaluation.utils import check_bin_alignment
        check_bin_alignment()   # uses current cfg
    """
    if bin_intervals is None:
        bin_intervals = cfg["bin_intervals"]
    if bin_freq_include is None:
        bin_freq_include = cfg.get("bin_freq_include")

    intervals = _get_intervals(bin_intervals, bin_freq_include)

    all_ok = True
    total_steps = 0
    rows = []
    for start_min, end_min, bin_min in intervals:
        if end_min is None:
            rows.append((start_min, "open", bin_min, "?", "open-ended"))
            continue
        duration = end_min - start_min
        n_exact = duration / bin_min
        n_bins = math.ceil(n_exact)
        status = "OK" if duration % bin_min == 0 else f"PARTIAL ({n_exact:.3g} bins)"
        if duration % bin_min != 0:
            all_ok = False
        total_steps += n_bins
        rows.append((start_min, end_min, bin_min, n_bins, status))

    # Pretty-print
    def fmt(minutes):
        if minutes == "open":
            return "open"
        h = minutes / 60
        if h < 24:
            return f"{h:.4g}h"
        return f"{h/24:.4g}D"

    header = f"{'Start':>8}  {'End':>8}  {'Bin':>6}  {'Steps':>6}  Status"
    print(header)
    print("-" * len(header))
    for start_min, end_min, bin_min, n_bins, status in rows:
        print(f"{fmt(start_min):>8}  {fmt(end_min):>8}  {fmt(bin_min):>6}  {str(n_bins):>6}  {status}")
    print("-" * len(header))
    print(f"{'Total steps:':>{len(header) - 7}} {total_steps}")
    print()
    if all_ok:
        print("All intervals aligned.")
    else:
        print("WARNING: partial bins detected — fix the interval boundaries in bin_intervals config.")
    return all_ok


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
            step_offset = math.ceil(offset_min / bin_min) - 1
            bins_cum = 0
            for j in range(i):
                s, e, b = intervals[j]
                if e is not None:
                    bins_cum += math.ceil((e - s) / b)
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
    for start_min, end_min, bin_min in intervals:
        if end_min is not None:
            bins_cum.append(bins_cum[-1] + math.ceil((end_min - start_min) / bin_min))
        else:
            bins_cum.append(float('inf'))

    for i in range(len(intervals)):
        if bins_cum[i] <= step < bins_cum[i + 1]:
            start_min, end_min, bin_min = intervals[i]
            step_offset = step - bins_cum[i]
            t = start_min + (step_offset + 1) * bin_min
            # Clamp to interval end so partial last bins don't overshoot
            if end_min is not None:
                t = min(t, end_min)
            return t
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

def prepare_model(data, cfg):
    """
    Load a trained model and return (model, device).

    Replaces the old ``prepare_learner()`` which returned a FastAI Learner.
    """
    import torch
    from astra.models.hybrid.training import get_backbone
    from astra.data.mixed_dataloader import load_model_state

    model_name = cfg["model_name"]
    model_cfg = cfg.get("model", {})
    is_temporal = model_cfg.get("temporal_head", False)

    logger.info(f"Loading model: {model_name}")
    backbone = get_backbone(
        data, cfg,
        temporal_head=is_temporal,
        causal=model_cfg.get("causal", False),
        temporal_head_dropout=model_cfg.get("temporal_head_dropout", 0.3),
        temporal_channel_idx=data.get('temporal_channel_idx'),
        exclude_channel_indices=data.get('exclude_channel_indices', []),
    )

    state_dict = load_model_state(model_name)
    backbone.load_state_dict(state_dict, strict=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = backbone.to(device)
    backbone.eval()
    logger.info(f"Model loaded (temporal_head={is_temporal})")
    return backbone, device

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