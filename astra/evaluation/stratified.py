​"""
Cluster-stratified evaluation of ASTRA model predictions.

Produces per-cluster and comparative evaluation plots from pre-computed
predictions. No model re-inference or data pipeline required — only needs
the saved predictions CSV and a cluster mapping CSV.

Required input files:
    preds CSV:    PID, censor_step, time_hours, time_days, pred
    cluster CSV:  PID, dec_cluster, deceased_30d

Output directory: reports/eval/{model_name}/cluster/

Usage:
    python -m astra.evaluation.stratified \\
        --preds reports/eval/24042026_astra/predictions/preds_df_24042026_astra_active.csv \\
        --clusters data/interim/holdout_cluster.csv \\
        --model-name 24042026_astra
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    sns = None
    _HAS_SEABORN = False
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from astra.evaluation.utils import (
    calculate_average_precision_ci,
    calculate_roc_auc_ci,
    find_optimal_fbeta_threshold,
)

try:
    from astra.utils import save_figure
except Exception:
    def save_figure(fig, filename, save_dir='.', **kw):
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{filename}.png'),
                    dpi=300, bbox_inches='tight')

logger = logging.getLogger(__name__)

# ── Style constants (matching main evaluation) ──────────────────────────
_FIG_STYLE = dict(
    title=16, axis_label=14, tick_label=12,
    legend=12, annotation=11, suptitle=18,
)
_SUBMISSION_KW = dict(
    fit_long_side_px=1200, max_long_side_px=1200, max_bytes=5_000_000,
)

CLUSTER_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
MIN_POSITIVE = 5


def _get_max_days():
    try:
        from astra.evaluation.utils import get_max_days
        return get_max_days()
    except Exception:
        return 90


# ════════════════════════════════════════════════════════════════════════
# INLINED UTILITIES (avoid heavy transitive imports from
# predictive_performance / calibration)
# ════════════════════════════════════════════════════════════════════════

def _compute_net_benefit(y_true, y_prob, thresholds):
    """Net benefit for decision curve analysis."""
    N = len(y_true)
    prevalence = y_true.mean()
    nb_model = np.empty_like(thresholds, dtype=float)
    nb_treat_all = np.empty_like(thresholds, dtype=float)
    for i, t in enumerate(thresholds):
        weight = t / (1.0 - t)
        pos = y_prob >= t
        tp = np.sum(pos & (y_true == 1))
        fp = np.sum(pos & (y_true == 0))
        nb_model[i] = tp / N - fp / N * weight
        nb_treat_all[i] = prevalence - (1.0 - prevalence) * weight
    return nb_model, nb_treat_all


def _calculate_ece(y_true, y_pred, n_bins=4):
    """Expected Calibration Error."""
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (y_pred > lo) & (y_pred <= hi)
        prop = mask.mean()
        if prop > 0:
            ece += abs(y_pred[mask].mean() - y_true[mask].mean()) * prop
    return ece


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════

def load_data(
    preds_csv: str,
    cluster_csv: str,
    preds_extra_csv: str = None,
) -> pd.DataFrame:
    """Load and merge predictions with cluster assignments.

    Args:
        preds_csv:       Path to primary predictions CSV
                         (PID, censor_step, time_hours, time_days, pred).
        cluster_csv:     Path to cluster CSV
                         (PID, dec_cluster, deceased_30d).
        preds_extra_csv: Optional path to a second predictions CSV
                         (same columns) to concatenate before merging —
                         use this to supply trainval predictions when
                         evaluating over the full population.
    """
    preds = pd.read_csv(preds_csv)

    if preds_extra_csv is not None:
        extra = pd.read_csv(preds_extra_csv)
        overlap = set(preds['PID'].unique()) & set(extra['PID'].unique())
        if overlap:
            logger.warning(
                f"{len(overlap)} PIDs appear in both preds files — "
                "keeping primary preds for duplicates"
            )
            extra = extra[~extra['PID'].isin(overlap)]
        preds = pd.concat([preds, extra], ignore_index=True)
        logger.info(
            f"Combined predictions: {preds['PID'].nunique()} patients "
            f"({pd.read_csv(preds_csv)['PID'].nunique()} primary + "
            f"{extra['PID'].nunique()} extra)"
        )
        logger.warning(
            "Extra predictions are in-sample (trainval) — AUROC/AUPRC "
            "for those patients will be optimistic. Calibration metrics "
            "are less affected."
        )

    clusters = pd.read_csv(cluster_csv)

    for name, df, cols in [
        ('preds', preds, ['PID', 'censor_step', 'pred']),
        ('cluster', clusters, ['PID', 'dec_cluster', 'deceased_30d']),
    ]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    df = preds.merge(
        clusters[['PID', 'dec_cluster', 'deceased_30d']],
        on='PID', how='inner',
    )
    n_preds = preds['PID'].nunique()
    n_matched = df['PID'].nunique()
    if n_matched < n_preds:
        logger.warning(
            f"{n_preds - n_matched} patients in preds not found in cluster CSV"
        )
    logger.info(
        f"Loaded {len(df):,} rows — {n_matched} patients, "
        f"{df['dec_cluster'].nunique()} clusters"
    )
    return df


# ════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ════════════════════════════════════════════════════════════════════════

def _safe_metrics(y_true, y_pred):
    """AUROC + CI, AUPRC + CI.  Returns None if insufficient samples."""
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos < MIN_POSITIVE or n_neg < MIN_POSITIVE:
        return None
    try:
        auroc, auc_lo, auc_hi = calculate_roc_auc_ci(y_true, y_pred)
        auprc, ap_lo, ap_hi = calculate_average_precision_ci(y_true, y_pred)
    except Exception:
        return None
    return dict(
        auroc=auroc, auroc_ci=(auc_lo, auc_hi),
        auprc=auprc, auprc_ci=(ap_lo, ap_hi),
        n_samples=len(y_true), n_positive=n_pos,
    )


def compute_cluster_time_metrics(df: pd.DataFrame) -> dict:
    """AUROC/AUPRC at every (censor_step x cluster).

    Returns ``{cluster_id_or_'all': [dict per step]}``.
    """
    clusters = sorted(df['dec_cluster'].unique())
    steps = sorted(df['censor_step'].unique())
    out = {c: [] for c in clusters}
    out['all'] = []

    for step in steps:
        sdf = df[df['censor_step'] == step]
        if sdf.empty:
            continue
        row0 = sdf.iloc[0]
        info = dict(
            censor_step=step,
            time_hours=float(row0.get('time_hours', np.nan)),
            time_days=float(row0.get('time_days', np.nan)),
        )

        # overall
        m = _safe_metrics(
            sdf['deceased_30d'].values.astype(int), sdf['pred'].values
        )
        if m:
            out['all'].append({**info, **m})

        # per cluster
        for c in clusters:
            cdf = sdf[sdf['dec_cluster'] == c]
            if len(cdf) < MIN_POSITIVE * 2:
                continue
            m = _safe_metrics(
                cdf['deceased_30d'].values.astype(int), cdf['pred'].values
            )
            if m:
                out[c].append({**info, **m})

    for k, v in out.items():
        logger.info(f"  cluster={k}: {len(v)} valid timepoints")
    return out


def _baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Last available prediction per patient (full trajectory)."""
    return df.loc[df.groupby('PID')['censor_step'].idxmax()].copy()


# ════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════

def _cluster_label(c, n):
    return f'Cluster {c} (n={n})'


# ── 1) AUROC / AUPRC over time ─────────────────────────────────────────

def plot_time_metrics(cluster_results, cut_hours=72, max_days=None):
    """2x2 overlay: AUROC/AUPRC x hours/days, one line per cluster."""
    if max_days is None:
        max_days = _get_max_days()
    clusters = [k for k in sorted((k for k in cluster_results if k != 'all'), key=str)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (metric, ylabel) in enumerate(
        [('auroc', 'AUROC'), ('auprc', 'AUPRC')]
    ):
        for col, (tkey, xlabel, xlim) in enumerate([
            ('time_hours', 'Time (hours)', cut_hours),
            ('time_days', 'Time (days)', max_days),
        ]):
            ax = axes[row, col]
            panel = chr(65 + row * 2 + col)

            # Overall (dashed black)
            r_all = cluster_results.get('all', [])
            if r_all:
                x = np.array([d[tkey] for d in r_all])
                y = np.array([d[metric] for d in r_all])
                lo = np.array([d[f'{metric}_ci'][0] for d in r_all])
                hi = np.array([d[f'{metric}_ci'][1] for d in r_all])
                mask = (x <= xlim) if col == 0 else np.ones(len(x), bool)
                if mask.any():
                    ax.plot(x[mask], y[mask], 'k--', lw=2, alpha=.6,
                            label='Overall')
                    ax.fill_between(x[mask], lo[mask], hi[mask],
                                    color='k', alpha=.06)

            for i, c in enumerate(clusters):
                r = cluster_results.get(c, [])
                if not r:
                    continue
                x = np.array([d[tkey] for d in r])
                y = np.array([d[metric] for d in r])
                lo = np.array([d[f'{metric}_ci'][0] for d in r])
                hi = np.array([d[f'{metric}_ci'][1] for d in r])
                mask = (x <= xlim) if col == 0 else np.ones(len(x), bool)
                clr = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                if mask.any():
                    ax.plot(x[mask], y[mask], color=clr, lw=1.5, alpha=.85,
                            label=_cluster_label(c, r[0]['n_samples']))
                    ax.fill_between(x[mask], lo[mask], hi[mask],
                                    color=clr, alpha=.10)

            ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, xlim), ylim=(0, 1))
            ax.set_yticks(np.arange(0, 1.1, .1))
            ax.set_title(
                f'{panel}) {ylabel} over {"Hours" if col == 0 else "Days"}',
                fontsize=_FIG_STYLE['title'], fontweight='bold',
            )
            ax.grid(True, alpha=.3)
            ax.legend(fontsize=_FIG_STYLE['legend'] - 1, loc='lower right')
            ax.tick_params(labelsize=_FIG_STYLE['tick_label'])

    plt.tight_layout()
    return fig


# ── 2) Baseline ROC + PR ───────────────────────────────────────────────

def plot_baseline_roc_pr(baseline_df):
    """Overlaid ROC and PR curves — one per cluster + overall baseline."""
    clusters = sorted(baseline_df['dec_cluster'].unique())
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))

    for i, c in enumerate(clusters):
        cdf = baseline_df[baseline_df['dec_cluster'] == c]
        yt = cdf['deceased_30d'].values.astype(int)
        yp = cdf['pred'].values
        if yt.sum() < MIN_POSITIVE or (len(yt) - yt.sum()) < MIN_POSITIVE:
            logger.warning(f"Cluster {c}: skipping ROC/PR (pos={yt.sum()})")
            continue

        clr = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        auroc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)
        fpr, tpr, _ = roc_curve(yt, yp)
        prec, rec, _ = precision_recall_curve(yt, yp)

        ax_roc.plot(fpr, tpr, color=clr, lw=2,
                    label=f'C{c} (AUC={auroc:.3f}, n={len(yt)})')
        ax_pr.plot(rec, prec, color=clr, lw=2,
                   label=f'C{c} (AP={auprc:.3f}, n={len(yt)})')

    # Overall baseline (dashed black)
    yt_all = baseline_df['deceased_30d'].values.astype(int)
    yp_all = baseline_df['pred'].values
    if yt_all.sum() >= MIN_POSITIVE:
        auroc_all = roc_auc_score(yt_all, yp_all)
        auprc_all = average_precision_score(yt_all, yp_all)
        fpr_all, tpr_all, _ = roc_curve(yt_all, yp_all)
        prec_all, rec_all, _ = precision_recall_curve(yt_all, yp_all)
        ax_roc.plot(fpr_all, tpr_all, 'k--', lw=2, alpha=0.6,
                    label=f'Overall (AUC={auroc_all:.3f}, n={len(yt_all)})')
        ax_pr.plot(rec_all, prec_all, 'k--', lw=2, alpha=0.6,
                   label=f'Overall (AP={auprc_all:.3f}, n={len(yt_all)})')

    ax_roc.plot([0, 1], [0, 1], color='lightgrey', lw=1, ls=':')
    ax_roc.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax_roc.set_title('A) ROC by Cluster',
                      fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax_roc.legend(fontsize=_FIG_STYLE['legend'] - 1,
                  loc='lower right', bbox_to_anchor=(1.0, 0.0),
                  framealpha=0.9)
    ax_roc.grid(True, alpha=.3)

    prev = baseline_df['deceased_30d'].mean()
    ax_pr.axhline(prev, color='lightgrey', ls=':', lw=1,
                  label=f'Prevalence ({prev:.1%})')
    ax_pr.set(xlabel='Recall', ylabel='Precision')
    ax_pr.set_title('B) Precision-Recall by Cluster',
                     fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax_pr.legend(fontsize=_FIG_STYLE['legend'] - 1,
                 loc='upper right', bbox_to_anchor=(1.0, 1.0),
                 framealpha=0.9)
    ax_pr.grid(True, alpha=.3)

    for ax in (ax_roc, ax_pr):
        ax.xaxis.label.set_fontsize(_FIG_STYLE['axis_label'])
        ax.yaxis.label.set_fontsize(_FIG_STYLE['axis_label'])
        ax.tick_params(labelsize=_FIG_STYLE['tick_label'])

    plt.tight_layout()
    return fig


# ── 3) Confusion matrices ──────────────────────────────────────────────

def plot_confusion_matrices(baseline_df, threshold, label):
    """1-row grid: one normalized CM per cluster, with absolute counts."""
    clusters = sorted(baseline_df['dec_cluster'].unique())
    n = len(clusters)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, c in enumerate(clusters):
        cdf = baseline_df[baseline_df['dec_cluster'] == c]
        yt = cdf['deceased_30d'].values.astype(int)
        yp_bin = (cdf['pred'].values >= threshold).astype(int)
        n_pos = int(yt.sum())

        cm_norm = confusion_matrix(yt, yp_bin, labels=[0, 1], normalize='true')
        cm_abs = confusion_matrix(yt, yp_bin, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm_norm, display_labels=['Neg', 'Pos'])
        disp.plot(ax=axes[i], values_format='.2f', colorbar=False)

        # Overlay absolute counts
        for r in range(2):
            for c_ in range(2):
                txt = axes[i].texts[r * 2 + c_]
                txt.set_text(f'{cm_norm[r, c_]:.2f}\n({cm_abs[r, c_]})')
                txt.set_fontsize(10)

        axes[i].set_title(
            f'Cluster {c}\nn={len(cdf)}, pos={n_pos}',
            fontsize=_FIG_STYLE['tick_label'],
        )

    fig.suptitle(
        f'Confusion Matrices — {label} (threshold={threshold:.3f})',
        fontsize=_FIG_STYLE['title'], fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


# ── 4) Calibration ─────────────────────────────────────────────────────

def plot_calibration(baseline_df, n_bins=4):
    """Overlaid reliability diagram + ECE/Brier bar chart."""
    clusters = sorted(baseline_df['dec_cluster'].unique())
    fig, (ax_cal, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    eces, briers, labels = [], [], []
    for i, c in enumerate(clusters):
        cdf = baseline_df[baseline_df['dec_cluster'] == c]
        yt = cdf['deceased_30d'].values.astype(int)
        yp = cdf['pred'].values
        if yt.sum() < MIN_POSITIVE:
            continue

        clr = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        frac_pos, mean_pred = calibration_curve(
            yt, yp, n_bins=n_bins, strategy='uniform'
        )
        ece = _calculate_ece(yt, yp, n_bins=n_bins)
        brier = brier_score_loss(yt, yp)

        ax_cal.plot(mean_pred, frac_pos, 'o-', color=clr, lw=2,
                    markersize=6, label=f'C{c} (ECE={ece:.3f})')
        eces.append(ece)
        briers.append(brier)
        labels.append(f'C{c}')

    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=.5, label='Perfect')
    ax_cal.set(xlabel='Mean Predicted Probability',
               ylabel='Fraction of Positives', xlim=(0, 1), ylim=(0, 1))
    ax_cal.set_aspect('equal', adjustable='box')
    ax_cal.set_title('A) Reliability Diagram',
                      fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax_cal.legend(fontsize=_FIG_STYLE['legend'] - 1)
    ax_cal.grid(True, alpha=.3)

    x_pos = np.arange(len(labels))
    w = 0.35
    colors = [CLUSTER_COLORS[i] for i in range(len(labels))]
    ax_bar.bar(x_pos - w / 2, eces, w, color=colors, alpha=.8, label='ECE')
    ax_bar.bar(x_pos + w / 2, briers, w, color=colors, alpha=.45,
               label='Brier Score')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title('B) ECE & Brier Score',
                      fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax_bar.legend(fontsize=_FIG_STYLE['legend'])
    ax_bar.grid(True, alpha=.3, axis='y')

    for ax in (ax_cal, ax_bar):
        ax.xaxis.label.set_fontsize(_FIG_STYLE['axis_label'])
        ax.yaxis.label.set_fontsize(_FIG_STYLE['axis_label'])
        ax.tick_params(labelsize=_FIG_STYLE['tick_label'])

    plt.tight_layout()
    return fig


# ── 5) Decision curve analysis ─────────────────────────────────────────

def plot_dca(df, baseline_df, max_threshold=0.5):
    """Per-cluster DCA at key timepoints (temporal split-violin style).

    One subplot per cluster (+ overall), each showing net benefit curves
    at selected timepoints (1h, 6h, 24h, 72h).  Mirrors the style of
    _plot_decision_curves_temporal from predictive_performance.py.
    """
    clusters = sorted(baseline_df['dec_cluster'].unique())
    groups = list(clusters) + ['all']
    n_groups = len(groups)
    ncols = 3
    nrows = (n_groups + ncols - 1) // ncols

    thresholds = np.linspace(0.01, max_threshold, 200)
    target_hours = [1, 6, 24, 72]
    tp_colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728',
                 '#9467BD', '#8C564B']

    # Snap requested hours to nearest available in df
    avail_hours = sorted(df['time_hours'].unique())
    selected_hours = []
    seen = set()
    for t in target_hours:
        closest = min(avail_hours, key=lambda x: abs(x - t))
        if closest not in seen:
            selected_hours.append((t, closest))
            seen.add(closest)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, c in enumerate(groups):
        ax = axes[idx]

        if c == 'all':
            pids = set(baseline_df['PID'].values)
            title = f'Overall (n={len(pids)})'
        else:
            pids = set(baseline_df[baseline_df['dec_cluster'] == c]['PID'].values)
            n_pos = int(baseline_df[baseline_df['dec_cluster'] == c]['deceased_30d'].sum())
            title = f'Cluster {c} (n={len(pids)}, pos={n_pos})'

        sub_df = df[df['PID'].isin(pids)]

        global_ymin, global_ymax = 0.0, 0.0
        treat_all_curves = []

        for i, (req_h, actual_h) in enumerate(selected_hours):
            step_df = sub_df[sub_df['time_hours'] == actual_h]
            if step_df.empty:
                treat_all_curves.append(None)
                continue

            yt = step_df['deceased_30d'].values.astype(float)
            yp = step_df['pred'].values

            if yt.sum() < MIN_POSITIVE or (len(yt) - yt.sum()) < MIN_POSITIVE:
                treat_all_curves.append(None)
                continue

            prevalence = yt.mean()
            nb_model, _ = _compute_net_benefit(yt, yp, thresholds)
            clr = tp_colors[i % len(tp_colors)]
            ax.plot(thresholds, nb_model, color=clr, lw=1.8,
                    label=f'{int(req_h)}h')

            nb_ta = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
            treat_all_curves.append((nb_ta, clr))

            global_ymin = min(global_ymin, nb_model.min())
            global_ymax = max(global_ymax, nb_model.max())

        ymin = min(global_ymin, -0.01) - 0.005
        ymax = global_ymax * 1.15 + 0.005

        first_ta = True
        for curve_data in treat_all_curves:
            if curve_data is None:
                continue
            nb_ta, clr = curve_data
            ax.plot(thresholds, np.clip(nb_ta, ymin, None),
                    color=clr, lw=1.0, ls='--', alpha=0.35,
                    label='Treat All' if first_ta else None)
            first_ta = False

        ax.axhline(0, color='black', lw=1, label='Treat None')
        ax.set_title(title, fontsize=_FIG_STYLE['tick_label'] + 1,
                     fontweight='bold')
        ax.set_xlim(0, max_threshold)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Threshold Probability',
                      fontsize=_FIG_STYLE['tick_label'])
        ax.set_ylabel('Net Benefit', fontsize=_FIG_STYLE['tick_label'])
        ax.legend(fontsize=9, loc='upper right',
                  bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=_FIG_STYLE['tick_label'])

    for j in range(n_groups, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Decision Curve Analysis — Per Cluster × Timepoint',
                 fontsize=_FIG_STYLE['suptitle'], fontweight='bold')
    plt.tight_layout()
    return fig


# ── 6) Prediction distribution ─────────────────────────────────────────

def plot_pred_distribution(baseline_df):
    """Violin plot of baseline predictions per cluster, split by outcome."""
    clusters = sorted(baseline_df['dec_cluster'].unique())

    # Long-form DataFrame: one block per cluster + overall
    parts = []
    for c in clusters:
        sub = baseline_df[baseline_df['dec_cluster'] == c][['pred', 'deceased_30d']].copy()
        sub['group'] = f'C{c}'
        parts.append(sub)
    overall = baseline_df[['pred', 'deceased_30d']].copy()
    overall['group'] = 'All'
    parts.append(overall)

    plot_df = pd.concat(parts, ignore_index=True)
    plot_df['Outcome'] = plot_df['deceased_30d'].map({0: 'Survived', 1: 'Deceased'})

    group_order = [f'C{c}' for c in clusters] + ['All']
    counts = {
        g: (
            int((plot_df['group'] == g).sum()),
            int(((plot_df['group'] == g) & (plot_df['deceased_30d'] == 1)).sum()),
        )
        for g in group_order
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    if _HAS_SEABORN:
        sns.violinplot(
            data=plot_df, x='group', y='pred', hue='Outcome',
            order=group_order, hue_order=['Survived', 'Deceased'],
            palette={'Survived': '#2CA02C', 'Deceased': '#D62728'},
            inner='box', cut=0, linewidth=1.0,
            ax=ax,
        )
    else:
        # Matplotlib fallback: side-by-side violins per outcome
        palette = {'Survived': '#2CA02C', 'Deceased': '#D62728'}
        outcomes = ['Survived', 'Deceased']
        offsets = [-0.2, 0.2]
        for outcome, offset, clr in zip(outcomes, offsets, ['#2CA02C', '#D62728']):
            for i, g in enumerate(group_order):
                sub = plot_df[(plot_df['group'] == g) & (plot_df['Outcome'] == outcome)]['pred'].values
                if len(sub) >= 2:
                    vp = ax.violinplot(sub, positions=[i + offset], widths=0.35,
                                       showmedians=True, showextrema=False)
                    for pc in vp['bodies']:
                        pc.set_facecolor(clr)
                        pc.set_alpha(0.7)
                    vp['cmedians'].set_color(clr)
        # Dummy legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#2CA02C', alpha=0.7, label='Survived'),
            Patch(facecolor='#D62728', alpha=0.7, label='Deceased'),
        ], fontsize=_FIG_STYLE['legend'])

    # Overlay individual points for very small positive groups
    rng = np.random.default_rng(42)
    for i, g in enumerate(group_order):
        pos_sub = plot_df[(plot_df['group'] == g) & (plot_df['Outcome'] == 'Deceased')]
        if 0 < len(pos_sub) < MIN_POSITIVE * 2:
            jitter = rng.uniform(-0.06, 0.06, len(pos_sub))
            ax.scatter(
                i + 0.2 + jitter, pos_sub['pred'].values,
                color='#D62728', zorder=5, s=40, alpha=0.9,
                edgecolors='white', linewidths=0.5,
            )

    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(
        [f'{g}\n(n={counts[g][0]}, pos={counts[g][1]})' for g in group_order],
        fontsize=_FIG_STYLE['tick_label'],
    )
    ax.set_xlabel('Cluster', fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylabel('Predicted Probability', fontsize=_FIG_STYLE['axis_label'])
    ax.set_ylim(0, 1)
    ax.set_title(
        'Prediction Distribution by Cluster and Outcome',
        fontsize=_FIG_STYLE['title'], fontweight='bold',
    )
    ax.grid(True, alpha=.3, axis='y')
    ax.tick_params(axis='y', labelsize=_FIG_STYLE['tick_label'])

    plt.tight_layout()
    return fig


# ── 7) Time-series prediction distribution per cluster ─────────────────

def plot_cluster_pred_distributions(df, baseline_df, model_name, save_dir):
    """Split-violin prediction distribution (hours + days) for each cluster.

    Uses the exact plot_prediction_distribution implementation from
    predictive_performance.py (Van Calster et al. style split violins).
    Saves one figure per cluster plus one for the overall population.
    """
    try:
        from astra.evaluation.predictive_performance import plot_prediction_distribution
    except Exception as e:
        logger.warning(f"plot_prediction_distribution not available — skipping: {e}")
        return

    clusters = sorted(baseline_df['dec_cluster'].unique())
    groups = [(c, f'cluster{c}') for c in clusters] + [('all', 'all')]

    for c, label in groups:
        if c == 'all':
            bl = baseline_df
            preds_df = df
        else:
            pids = set(baseline_df[baseline_df['dec_cluster'] == c]['PID'].values)
            bl = baseline_df[baseline_df['dec_cluster'] == c]
            preds_df = df[df['PID'].isin(pids)]

        holdout_pids = bl['PID'].values
        y_true = bl['deceased_30d'].values.astype(int)
        n_pos = int(y_true.sum())

        logger.info(
            f"  Pred dist [{label}]: {len(holdout_pids)} patients, {n_pos} positive"
        )

        try:
            fig = plot_prediction_distribution(
                preds_df=preds_df,
                y_true=y_true,
                holdout_pids=holdout_pids,
            )
            save_figure(
                fig,
                f'cluster_pred_dist_time_{label}_{model_name}',
                save_dir=save_dir,
                **_SUBMISSION_KW,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"  Pred dist [{label}]: failed — {e}")


# ── 8) Cluster prevalence overview ─────────────────────────────────────

def plot_prevalence(baseline_df):
    """Bar chart of cluster sizes and mortality prevalence."""
    clusters = sorted(baseline_df['dec_cluster'].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sizes, prevs = [], []
    for c in clusters:
        cdf = baseline_df[baseline_df['dec_cluster'] == c]
        sizes.append(len(cdf))
        prevs.append(cdf['deceased_30d'].mean() * 100)

    x = np.arange(len(clusters))
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]

    ax1.bar(x, sizes, color=colors, alpha=.8)
    for xi, s in zip(x, sizes):
        ax1.text(xi, s + max(sizes) * 0.02, str(s), ha='center',
                 fontsize=_FIG_STYLE['annotation'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{c}' for c in clusters])
    ax1.set_ylabel('Number of Patients', fontsize=_FIG_STYLE['axis_label'])
    ax1.set_title('A) Cluster Size', fontsize=_FIG_STYLE['title'],
                   fontweight='bold')
    ax1.grid(True, alpha=.3, axis='y')
    ax1.tick_params(labelsize=_FIG_STYLE['tick_label'])

    bars = ax2.bar(x, prevs, color=colors, alpha=.8)
    for xi, p in zip(x, prevs):
        ax2.text(xi, p + max(prevs) * 0.03, f'{p:.1f}%', ha='center',
                 fontsize=_FIG_STYLE['annotation'])
    overall_prev = baseline_df['deceased_30d'].mean() * 100
    ax2.axhline(overall_prev, color='black', ls='--', lw=1.5, alpha=.6,
                label=f'Overall ({overall_prev:.1f}%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{c}' for c in clusters])
    ax2.set_ylabel('30-day Mortality (%)', fontsize=_FIG_STYLE['axis_label'])
    ax2.set_title('B) Prevalence per Cluster',
                   fontsize=_FIG_STYLE['title'], fontweight='bold')
    ax2.legend(fontsize=_FIG_STYLE['legend'])
    ax2.grid(True, alpha=.3, axis='y')
    ax2.tick_params(labelsize=_FIG_STYLE['tick_label'])

    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════

def create_summary_csv(cluster_results, baseline_df, save_path):
    """Write a CSV with per-cluster metrics at key timepoints + baseline."""
    rows = []

    # Baseline metrics per cluster
    clusters = sorted(baseline_df['dec_cluster'].unique())
    for c in list(clusters) + ['all']:
        if c == 'all':
            cdf = baseline_df
        else:
            cdf = baseline_df[baseline_df['dec_cluster'] == c]

        yt = cdf['deceased_30d'].values.astype(int)
        yp = cdf['pred'].values
        n_pos = int(yt.sum())

        if n_pos >= MIN_POSITIVE and (len(yt) - n_pos) >= MIN_POSITIVE:
            auroc = roc_auc_score(yt, yp)
            auprc = average_precision_score(yt, yp)
            ece = _calculate_ece(yt, yp)
            brier = brier_score_loss(yt, yp)
        else:
            auroc = auprc = ece = brier = np.nan

        rows.append(dict(
            cluster=c, timepoint='baseline',
            n_patients=len(cdf), n_positive=n_pos,
            prevalence=cdf['deceased_30d'].mean(),
            auroc=auroc, auprc=auprc, ece=ece, brier=brier,
        ))

    # Time-dependent metrics at key hours
    key_hours = [1, 6, 12, 24, 48, 72]
    key_days = [7, 14, 30, 60, 90]

    for key in list(cluster_results.keys()):
        for entry in cluster_results[key]:
            h = entry['time_hours']
            d = entry['time_days']
            # Check if this is near a key timepoint
            near_hour = any(abs(h - kh) < 0.5 for kh in key_hours)
            near_day = any(abs(d - kd) < 0.5 for kd in key_days) and h > 72
            if near_hour or near_day:
                if h <= 72:
                    tp_label = f'{h:.0f}h'
                else:
                    tp_label = f'{d:.0f}d'
                rows.append(dict(
                    cluster=key, timepoint=tp_label,
                    n_patients=entry['n_samples'],
                    n_positive=entry['n_positive'],
                    prevalence=entry['n_positive'] / entry['n_samples'],
                    auroc=entry['auroc'], auprc=entry['auprc'],
                    ece=np.nan, brier=np.nan,
                ))

    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary.to_csv(save_path, index=False, float_format='%.4f')
    logger.info(f"Summary saved to {save_path}")
    return summary


# ════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════

def run_stratified_evaluation(
    preds_csv: str,
    cluster_csv: str,
    model_name: str,
    save_dir: str = None,
    preds_extra_csv: str = None,
):
    """Run full cluster-stratified evaluation from saved predictions.

    Args:
        preds_csv:       Path to predictions CSV.
        cluster_csv:     Path to cluster mapping CSV (PID, dec_cluster, deceased_30d).
        model_name:      Model name (for titles and file paths).
        save_dir:        Output directory. Defaults to
                         ``reports/eval/{model_name}/cluster/``.
        preds_extra_csv: Optional second predictions CSV (e.g. trainval) to
                         concatenate with preds_csv before evaluation.
    """
    if save_dir is None:
        save_dir = f'reports/eval/{model_name}/cluster'
    os.makedirs(save_dir, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────
    logger.info("Loading data...")
    df = load_data(preds_csv, cluster_csv, preds_extra_csv=preds_extra_csv)

    # ── Baseline predictions (last step per patient) ────────────────────
    logger.info("Computing baseline predictions (full trajectory)...")
    baseline_df = _baseline(df)
    logger.info(
        f"Baseline: {len(baseline_df)} patients, "
        f"{int(baseline_df['deceased_30d'].sum())} positive"
    )

    # ── Prevalence overview ─────────────────────────────────────────────
    logger.info("Plotting cluster prevalence overview...")
    fig = plot_prevalence(baseline_df)
    save_figure(fig, f'cluster_prevalence_{model_name}',
                save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)

    # ── Baseline ROC/PR ─────────────────────────────────────────────────
    logger.info("Plotting baseline ROC/PR curves...")
    fig = plot_baseline_roc_pr(baseline_df)
    save_figure(fig, f'cluster_baseline_eval_{model_name}',
                save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)

    # ── Confusion matrices at F1 and F5 thresholds ──────────────────────
    logger.info("Computing F-beta thresholds on evaluation population...")
    yt_all = baseline_df['deceased_30d'].values.astype(int)
    yp_all = baseline_df['pred'].values

    for beta, label in [(1, 'F1'), (5, 'F5')]:
        thr, score = find_optimal_fbeta_threshold(yt_all, yp_all, beta=beta)
        logger.info(f"  {label} threshold={thr:.4f} (score={score:.4f})")
        fig = plot_confusion_matrices(baseline_df, thr, label)
        save_figure(fig, f'cluster_cm_{label}_{model_name}',
                    save_dir=save_dir, **_SUBMISSION_KW)
        plt.close(fig)

    # ── DCA ─────────────────────────────────────────────────────────────
    logger.info("Plotting decision curve analysis...")
    fig = plot_dca(df, baseline_df)
    save_figure(fig, f'cluster_dca_{model_name}',
                save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)

    # ── Calibration ─────────────────────────────────────────────────────
    logger.info("Plotting calibration...")
    fig = plot_calibration(baseline_df)
    save_figure(fig, f'cluster_calibration_{model_name}',
                save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)

    # ── Prediction distribution (baseline violin) ──────────────────────
    logger.info("Plotting prediction distributions...")
    fig = plot_pred_distribution(baseline_df)
    save_figure(fig, f'cluster_pred_distribution_{model_name}',
                save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)

    # ── Time-series prediction distributions per cluster ────────────────
    logger.info("Plotting time-series prediction distributions per cluster...")
    plot_cluster_pred_distributions(df, baseline_df, model_name, save_dir)

    # ── Time-dependent metrics ──────────────────────────────────────────
    logger.info("Computing time-dependent metrics per cluster...")
    cluster_results = compute_cluster_time_metrics(df)

    logger.info("Plotting time-dependent metrics overlay...")
    fig = plot_time_metrics(cluster_results)
    save_figure(fig, f'cluster_time_metrics_{model_name}',
                save_dir=save_dir, **_SUBMISSION_KW)
    plt.close(fig)

    # ── Summary CSV ─────────────────────────────────────────────────────
    summary_path = os.path.join(save_dir, f'cluster_summary_{model_name}.csv')
    summary = create_summary_csv(cluster_results, baseline_df, summary_path)

    # ── Print summary ───────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CLUSTER-STRATIFIED EVALUATION SUMMARY")
    logger.info("=" * 70)
    bl = summary[summary['timepoint'] == 'baseline']
    for _, r in bl.iterrows():
        c = r['cluster']
        logger.info(
            f"  Cluster {c:>3}: n={int(r['n_patients']):>5}, "
            f"pos={int(r['n_positive']):>4} ({r['prevalence']:.1%}), "
            f"AUROC={r['auroc']:.3f}, AUPRC={r['auprc']:.3f}, "
            f"ECE={r['ece']:.4f}"
        )
    logger.info("=" * 70)
    logger.info(f"All outputs saved to {save_dir}/")

    return cluster_results, summary


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Cluster-stratified evaluation of ASTRA predictions',
    )
    parser.add_argument(
        '--preds', required=True,
        help='Path to predictions CSV '
             '(e.g. reports/eval/.../preds_df_..._active.csv)',
    )
    parser.add_argument(
        '--clusters', required=True,
        help='Path to cluster CSV with PID, dec_cluster, deceased_30d',
    )
    parser.add_argument(
        '--model-name', required=True,
        help='Model name (e.g. 24042026_astra)',
    )
    parser.add_argument(
        '--save-dir', default=None,
        help='Output directory (default: reports/eval/{model-name}/cluster/)',
    )
    parser.add_argument(
        '--preds-extra', default=None,
        help='Optional second predictions CSV to concatenate (e.g. trainval preds) '
             'for full-population evaluation. NOTE: these predictions are in-sample '
             'so discrimination metrics (AUROC/AUPRC) will be optimistic for those patients.',
    )

    args = parser.parse_args()
    run_stratified_evaluation(
        preds_csv=args.preds,
        cluster_csv=args.clusters,
        model_name=args.model_name,
        save_dir=args.save_dir,
        preds_extra_csv=args.preds_extra,
    )
