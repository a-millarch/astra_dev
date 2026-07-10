import logging
import math

from astra.utils import cfg
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

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

    Logs a table of all active intervals and flags any that are not exact.
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
        n_bins = duration // bin_min
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
    logger.info(header)
    logger.info("-" * len(header))
    for start_min, end_min, bin_min, n_bins, status in rows:
        logger.info(f"{fmt(start_min):>8}  {fmt(end_min):>8}  {fmt(bin_min):>6}  {str(n_bins):>6}  {status}")
    logger.info("-" * len(header))
    logger.info(f"{'Total steps:':>{len(header) - 7}} {total_steps}")
    if all_ok:
        logger.info("All intervals aligned.")
    else:
        logger.warning("Partial bins detected — fix the interval boundaries in bin_intervals config.")
    return all_ok


def get_max_days(data_config=None):
    """Return the maximum time horizon in days from bin interval config.

    This is derived from the last interval's end time and should be used
    instead of hardcoding ``max_days`` in evaluation/plotting functions.
    """
    if data_config is not None:
        intervals = _get_intervals(
            data_config['bin_intervals'],
            data_config.get('bin_freq_include'),
        )
    else:
        intervals = _get_intervals_from_cfg()

    last_end_min = max(
        end for _, end, _ in intervals if end is not None
    )
    return int(last_end_min / (24 * 60))


def get_total_steps(data_config=None):
    """Compute total number of bin steps from config.

    This is the canonical source of truth for sequence length.
    All code that needs the number of time steps should call this
    rather than hardcoding a value.

    Args:
        data_config: Optional dict with ``'bin_intervals'`` and
            ``'bin_freq_include'`` keys.  When *None*, reads from
            the global ``cfg``.

    Returns:
        int: Total number of time steps.
    """
    if data_config is not None:
        intervals = _get_intervals(
            data_config['bin_intervals'],
            data_config.get('bin_freq_include'),
        )
    else:
        intervals = _get_intervals_from_cfg()

    total = 0
    for start_min, end_min, bin_min in intervals:
        if end_min is None:
            continue
        total += (end_min - start_min) // bin_min
    return total


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
    for start_min, end_min, bin_min in intervals:
        if end_min is not None:
            bins_cum.append(bins_cum[-1] + (end_min - start_min) // bin_min)
        else:
            bins_cum.append(float('inf'))

    for i in range(len(intervals)):
        # Use <= on the last interval so the boundary step (== total_steps)
        # still resolves to the interval endpoint time.
        is_last = (i == len(intervals) - 1)
        upper_ok = step <= bins_cum[i + 1] if is_last else step < bins_cum[i + 1]
        if bins_cum[i] <= step and upper_ok:
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
        temporal_head_mult=model_cfg.get("temporal_head_mult", 0.5),
        temporal_channel_idx=data.get('temporal_channel_idx'),
        exclude_channel_indices=data.get('exclude_channel_indices', []),
        bin_width_channel_idx=data.get('bin_width_channel_idx'),
    )

    state_dict = load_model_state(model_name)
    backbone.load_state_dict(state_dict, strict=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = backbone.to(device)
    backbone.eval()
    logger.info(f"Model loaded (temporal_head={is_temporal})")
    return backbone, device


def mc_dropout_predict(model, inputs, n_samples=30):
    """
    MC Dropout: run N forward passes with dropout active for uncertainty estimation.

    Args:
        model: trained backbone model
        inputs: tuple of input tensors (same format as model.forward)
        n_samples: number of stochastic forward passes

    Returns:
        mean_probs: [batch, ...] mean predicted probabilities
        std_probs: [batch, ...] standard deviation of predicted probabilities
    """
    import torch.nn as nn

    # Enable dropout but keep normalization layers in eval mode
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds.append(probs)

    model.eval()  # Restore
    preds = torch.stack(preds)  # [n_samples, batch, ...]
    return preds.mean(dim=0), preds.std(dim=0)


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


def _compute_placement_values(y_true, y_score):
    """Compute DeLong placement values (structural components) for one predictor.

    For each positive sample, V_10 = fraction of negatives scored below it.
    For each negative sample, V_01 = fraction of positives scored above it.
    These are the building blocks of the DeLong covariance matrix.

    Reference: Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm".
    """
    order = np.argsort(-y_score)  # descending
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = int(pos_mask.sum())  # number of positives
    n = int(neg_mask.sum())  # number of negatives

    # For each positive: fraction of negatives with strictly lower score
    # Handle ties via midranks
    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]

    # V10[i] = P(X_neg < X_pos_i) + 0.5 * P(X_neg == X_pos_i)
    v10 = np.zeros(m)
    for i, ps in enumerate(pos_scores):
        v10[i] = (np.sum(neg_scores < ps) + 0.5 * np.sum(neg_scores == ps)) / n

    # V01[j] = P(X_pos > X_neg_j) + 0.5 * P(X_pos == X_neg_j)
    v01 = np.zeros(n)
    for j, ns in enumerate(neg_scores):
        v01[j] = (np.sum(pos_scores > ns) + 0.5 * np.sum(pos_scores == ns)) / m

    return v10, v01


def delong_test_paired(y_true, y_pred_a, y_pred_b):
    """Two-sided DeLong test for two correlated AUROCs on the same samples.

    Tests H0: AUC_A == AUC_B for two models evaluated on the same ground truth.
    Accounts for correlation between the two AUCs through shared samples.

    Args:
        y_true: Binary ground truth labels, shape (n,).
        y_pred_a: Predicted scores from model A, shape (n,).
        y_pred_b: Predicted scores from model B, shape (n,).

    Returns:
        (z_stat, p_value, se_diff): z-statistic, two-sided p-value, and
            standard error of the AUC difference. se_diff can be used to
            compute a 95% CI for delta AUC: delta +/- 1.96 * se_diff.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())

    if m < 2 or n < 2:
        return 0.0, 1.0, 0.0

    v10_a, v01_a = _compute_placement_values(y_true, y_pred_a)
    v10_b, v01_b = _compute_placement_values(y_true, y_pred_b)

    # AUC = mean of placement values
    auc_a = np.mean(v10_a)
    auc_b = np.mean(v10_b)

    # Covariance matrix of (AUC_A, AUC_B) via DeLong decomposition:
    # S = S10/m + S01/n  where S10, S01 are 2x2 covariance matrices
    # of the placement value vectors for positives and negatives
    s10 = np.cov(np.column_stack([v10_a, v10_b]), rowvar=False, ddof=1)
    s01 = np.cov(np.column_stack([v01_a, v01_b]), rowvar=False, ddof=1)
    S = s10 / m + s01 / n

    # Variance of the difference AUC_A - AUC_B
    # Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B) = S[0,0] + S[1,1] - 2*S[0,1]
    var_diff = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]

    if var_diff <= 0:
        return 0.0, 1.0, 0.0

    se_diff = np.sqrt(var_diff)
    z = (auc_a - auc_b) / se_diff
    p = 2.0 * stats.norm.sf(abs(z))
    return float(z), float(p), float(se_diff)


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of raw p-values.
        alpha: FDR level (default 0.05).

    Returns:
        (rejected, adjusted_p): Boolean mask of rejected hypotheses and
            adjusted p-values.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    rank = np.arange(1, n + 1)

    # Adjusted p-values: p_adj[i] = min(p[i] * n / rank[i], 1.0)
    # enforced to be monotonically non-decreasing from the right
    adjusted = np.minimum(p[order] * n / rank, 1.0)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Map back to original order
    adjusted_out = np.empty(n)
    adjusted_out[order] = adjusted

    rejected = adjusted_out <= alpha
    return rejected, adjusted_out

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


def _recall_at_percentile(y_preds, y_true, percentile):
    """Calculate recall when selecting the top-percentile highest-risk patients.

    Args:
        y_preds: Prediction probabilities.
        y_true: Binary ground-truth labels.
        percentile: Top percentage to select (e.g. 10 for top 10%).

    Returns:
        Recall (float) within the selected group.
    """
    n_total_positive = np.sum(y_true == 1)
    if n_total_positive == 0 or len(y_preds) == 0:
        return 0.0

    n_select = max(1, int(np.ceil(len(y_preds) * percentile / 100)))
    top_indices = np.argsort(y_preds)[-n_select:]
    return float(np.sum(y_true[top_indices] == 1)) / n_total_positive


def bootstrap_recall_ci(y_preds, y_true, percentile, n_bootstraps=1000, alpha=0.95):
    """Bootstrap confidence interval for recall at a given top-percentile threshold.

    Mirrors ``calculate_average_precision_ci`` in structure.

    Args:
        y_preds: Prediction probabilities (1-D array).
        y_true: Binary labels (1-D array).
        percentile: Top percentage to select (e.g. 10 for top 10%).
        n_bootstraps: Number of bootstrap resamples.
        alpha: Confidence level.

    Returns:
        (recall, ci_lower, ci_upper)
    """
    recall = _recall_at_percentile(y_preds, y_true, percentile)

    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if np.sum(y_true[indices]) == 0:
            continue
        score = _recall_at_percentile(y_preds[indices], y_true[indices], percentile)
        bootstrapped_scores.append(score)

    if len(bootstrapped_scores) == 0:
        return recall, 0.0, 1.0

    sorted_scores = np.sort(np.array(bootstrapped_scores))
    ci_lower = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    ci_upper = sorted_scores[int((1.0 + alpha) / 2 * len(sorted_scores))]
    return recall, float(ci_lower), float(ci_upper)


def find_optimal_fbeta_threshold(y_true, y_pred, beta=1.0):
    """Find the probability threshold that maximises F-beta score.

    Uses the precision-recall curve to evaluate all unique thresholds
    in a single vectorised pass (no grid search needed).

    Args:
        y_true: Binary labels (1-D array).
        y_pred: Predicted probabilities (1-D array).
        beta: Beta parameter (1 = F1, 5 = F5, etc.).

    Returns:
        (best_threshold, best_fbeta)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    precision = precision[:-1]
    recall = recall[:-1]

    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    fbeta = np.where(denom > 0, (1 + beta_sq) * precision * recall / denom, 0.0)

    best_idx = np.argmax(fbeta)
    return float(thresholds[best_idx]), float(fbeta[best_idx])