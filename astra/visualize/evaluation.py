import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
    
    precision_recall_curve, 
    average_precision_score
)

from sklearn.calibration import calibration_curve





def plot_patient_preds(preds_df, patient_id, xlim=60*24*30, timeunit='min', xstep=60):
    # Filter predictions for the given patient
    patient_preds = preds_df[preds_df["PID"] == patient_id]

    # Sort by time to ensure lineplot is ordered correctly
    patient_preds = patient_preds.sort_values(by="time_min")

    # Plot prediction over time
    plt.figure(figsize=(12, 2), dpi=1200)
    x = patient_preds[f"time_{timeunit}"]
    plt.plot(x, patient_preds["pred"], marker='o', linestyle='-', markersize=2)
    plt.title(f"Mortality Risk Prediction Timeline for PID: {patient_id}", fontweight='bold')
    plt.xlabel(f"Time ({timeunit})")
    plt.ylabel("Prediction")
    plt.xlim(0, xlim)
    plt.ylim(0, 1)
    plt.grid(True)

    # xticks for every step of x
    plt.xticks(np.arange(0, xlim + xstep, xstep))

    plt.show()
    return plt


def plot_box_kde(df, dep, y):

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    pal = {1: sns.color_palette()[1], 0: sns.color_palette()[0]}
    sns.boxenplot(data=df, x=dep, y=y, ax=axs[0], palette=pal, hue=dep).legend(
        [], [], frameon=False
    )
    sns.kdeplot(data=df[df[dep] == 0], x=y, ax=axs[1], label="0", clip=(0, 1))
    sns.kdeplot(data=df[df[dep] == 1], x=y, ax=axs[1], label="1", clip=(0, 1))
    fig.legend()

    mask_0 = df[dep] == 0
    mask_1 = df[dep] == 1

    print(
        "is null in total\t\t", "%.2f" % float(df[y].isna().sum() / len(df) * 100), "%"
    )
    print(
        "is null in",
        dep,
        "== 0\t",
        "%.2f" % float(df[mask_0][y].isna().sum() / len(df[mask_0]) * 100),
        "%",
    )
    print(
        "is null in",
        dep,
        "== 1\t",
        "%.2f" % float(df[mask_1][y].isna().sum() / len(df[mask_1]) * 100),
        "%",
    )
    return fig


def plot_hist_kde(df, dep, y):

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    pal = {1: sns.color_palette()[1], 0: sns.color_palette()[0]}

    # Replace boxenplot with histplot
    sns.histplot(
        data=df, x=y, hue=dep, ax=axs[0], palette=pal, kde=True, stat="density"
    )
    axs[0].legend(title=dep)

    sns.kdeplot(data=df[df[dep] == 0], x=y, ax=axs[1], label="0", clip=(0, 1))
    sns.kdeplot(data=df[df[dep] == 1], x=y, ax=axs[1], label="1", clip=(0, 1))
    fig.legend()

    mask_0 = df[dep] == 0
    mask_1 = df[dep] == 1

    print(
        "is null in total\t\t", "%.2f" % float(df[y].isna().sum() / len(df) * 100), "%"
    )
    print(
        "is null in",
        dep,
        "== 0\t",
        "%.2f" % float(df[mask_0][y].isna().sum() / len(df[mask_0]) * 100),
        "%",
    )
    print(
        "is null in",
        dep,
        "== 1\t",
        "%.2f" % float(df[mask_1][y].isna().sum() / len(df[mask_1]) * 100),
        "%",
    )
    return fig


def plot_evaluation(y_preds, ys, target):

    fpr, tpr, thresholds = roc_curve(ys, y_preds)
    roc_auc = roc_auc_score(ys, y_preds)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    display = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{target}"
    ).plot(ax=ax[0], c="r")

    display2 = PrecisionRecallDisplay.from_predictions(
        ys, y_preds, ax=ax[1], figure=fig, name=f"{target}"
    )  # ,
    # legend=f'{target} within 12H post trauma reception')
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, c="grey")
    ax[0].set_title("Holdout:\nReceiver operator characteristics (ROC)")
    ax[0].grid()
    ax[1].plot([0, 1], [ys.sum() / len(ys), ys.sum() / len(ys)], "k--", lw=1, c="grey")
    ax[1].set_title("Holdout:\nPrecision Recall (PR)")
    ax[1].grid()
    os.makedirs("reports/eval", exist_ok=True)
    plt.savefig("reports/eval/holdout_evaluation.png", dpi=1200)
    plt.plot()
    return fig


def plot_loss(train_losses, val_losses=None, fold=None):
    """Plot training and validation loss curves.

    Args:
        train_losses: List of per-batch training losses.
        val_losses: Optional list of per-epoch validation losses.
        fold: Optional fold number for title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot training losses
    ax.plot(range(len(train_losses)), train_losses, label="Train", color="blue")

    # Plot validation losses (one per epoch, spread evenly across training iterations)
    if val_losses is not None and len(val_losses) > 0:
        n_train = len(train_losses)
        n_val = len(val_losses)
        if n_val > 1:
            val_iterations = [int(i * n_train / n_val) for i in range(n_val)]
        else:
            val_iterations = [n_train - 1]
        ax.plot(val_iterations, val_losses, label="Validation", color="orange")

    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    os.makedirs("reports/training", exist_ok=True)
    if fold is None:
        ax.set_title("Loss Plot")
        plt.savefig("reports/training/loss_plot.png")
    else:
        ax.set_title(f"Loss Plot for Fold {fold}")
        plt.savefig(f"reports/training/{fold}_fold_loss_plot.png")

    # Show the plot
    plt.show()

    # Create DataFrame
    iters_train = list(range(len(train_losses)))
    losses_all = list(train_losses)
    types_all = ["Train"] * len(train_losses)
    if val_losses is not None and len(val_losses) > 0:
        n_train = len(train_losses)
        n_val = len(val_losses)
        iters_val = [int(i * n_train / n_val) for i in range(n_val)] if n_val > 1 else [n_train - 1]
        iters_train += iters_val
        losses_all += list(val_losses)
        types_all += ["Validation"] * len(val_losses)
    df = pd.DataFrame({"Iteration": iters_train, "Loss": losses_all, "Type": types_all})

    return fig, df


def plot_fold_evaluation(metrics, target):
    fprs = metrics["fpr"]["scores"]
    tprs = metrics["tpr"]["scores"]
    precisions = metrics["precision"]["scores"]
    recalls = metrics["recall"]["scores"]
    roc_aucs = metrics["roc_auc"]["scores"]
    aps = metrics["avg_precision"]["scores"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot ROC curves
    for i in range(len(fprs)):
        RocCurveDisplay(
            fpr=fprs[i], tpr=tprs[i], roc_auc=roc_aucs[i], estimator_name=f"Fold {i+1}"
        ).plot(ax=ax[0])
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, c="grey")
    ax[0].set_title(f"Receiver Operating Characteristic (ROC) - {target}")
    ax[0].grid(True)

    # Plot PR curves
    for i in range(len(precisions)):
        PrecisionRecallDisplay(
            precision=precisions[i],
            recall=recalls[i],
            average_precision=aps[i],
            estimator_name=f"Fold {i+1}",
        ).plot(ax=ax[1])
    ax[1].plot([0, 1], [0.5, 0.5], "k--", lw=1, c="grey")
    ax[1].set_title(f"Precision-Recall (PR) - {target}")
    ax[1].grid(True)

    plt.tight_layout()
    os.makedirs("reports/eval", exist_ok=True)
    plt.savefig(f"reports/eval/{target}_evaluation_plots.png")
    plt.show()
    return fig


def evaluate_detection_rate(y_preds, y_true, threshold=0.5, label=""):
    """
    Evaluate the detection rate by calculating sensitivity and specificity.

    Parameters:
    - y_preds: Predicted probabilities for the positive class.
    - y_true: True binary labels.
    - threshold: Probability threshold for classifying as positive (default: 0.5).
    - label: Optional label shown in the figure title (e.g. "F1" or "F5").

    Returns:
        (fig, sensitivity, specificity)
    """
    import logging
    logger = logging.getLogger(__name__)

    y_true = np.asarray(y_true)
    y_preds = np.asarray(y_preds)

    positive_percentage = np.mean(y_true) * 100
    logger.info(f"Prevalence: {positive_percentage:.2f}%")

    y_pred_risk = (y_preds >= threshold).astype(int)

    cm_norm = confusion_matrix(y_true, y_pred_risk, normalize="true")
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred_risk).ravel()

    sensitivity = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0

    logger.info(f"Threshold={threshold:.4f} ({label}): "
                f"Sensitivity={sensitivity:.1f}%, Specificity={specificity:.1f}%")

    title_suffix = f" ({label}, t={threshold:.3f})" if label else f" (t={threshold:.3f})"

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    cm_absolute = confusion_matrix(y_true, y_pred_risk)
    ConfusionMatrixDisplay(
        confusion_matrix=cm_absolute, display_labels=["Negative", "Positive"]
    ).plot(ax=ax[0], values_format="d")
    ax[0].set_title(f"Confusion Matrix (Absolute){title_suffix}")

    ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=["Negative", "Positive"]
    ).plot(ax=ax[1], values_format=".2f")
    ax[1].set_title(f"Confusion Matrix (Normalized){title_suffix}")

    plt.tight_layout()
    return fig, sensitivity, specificity


def create_calibration_plot(y_true, y_pred, n_bins=4):
    # Create a Figure and Axes object
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    # Plot model calibration curve
    ax.plot(prob_pred, prob_true, marker="o", label="Model")

    # Set labels and title
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Plot")
    ax.legend(loc="lower right")
    ax.grid(True)

    # Calculate and display Brier score
    brier_score = np.mean((y_pred - y_true) ** 2)
    ax.text(0.1, 0.9, f"Brier Score: {brier_score:.4f}", transform=ax.transAxes)

    # Save the figure
    os.makedirs("reports/calibration", exist_ok=True)
    plt.savefig("reports/calibration/calibration.png")

    # Show the plot
    plt.show()

    return fig  # Return the Figure object


def plot_multiple_evaluations(predictions_by_time, labels=None):
    """Plot ROC and PR curves for multiple censoring time points.

    Args:
        predictions_by_time: List of (y_pred, y_true) tuples, one per time point.
        labels: Optional list of labels for each time point.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    colors =['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#8C564B','#E377C2','#7F7F7F','#BCBD22','#17BECF']

    for i, (y_preds, ys) in enumerate(predictions_by_time):
        
        # --- ROC ---
        fpr, tpr, _ = roc_curve(ys, y_preds)
        roc_auc = roc_auc_score(ys, y_preds)
        label = labels[i] if labels else f"t={i}"
        ax_roc.plot(fpr, tpr, color=colors[i % len(colors)], label=f"{label} (AUC={roc_auc:.3f})")
        
        # --- PR ---
       # PrecisionRecallDisplay.from_predictions(
       #     ys, y_preds, ax=ax_pr, name=label, color=colors[i % len(colors)]
       # )
        
        
        # --- PR ---
        precision, recall, _ = precision_recall_curve(ys, y_preds)
        auprc = average_precision_score(ys, y_preds)
        label = labels[i] if labels else f"t={i}"
        ax_pr.plot(recall, precision, color=colors[i % len(colors)], label=f"{label} (AUC={auprc:.3f})")
    
    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, c="grey")
    ax_roc.set_title("Holdout:\nReceiver Operator Characteristics (ROC)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid()
    ax_roc.legend(fontsize=9, title="Included time")  
    ax_roc.get_legend().get_title().set_fontsize(10)
    ax_roc.set_aspect('equal', adjustable='box')  # keep ROC plot square

    # PR formatting
    baseline = ys.sum() / len(ys)
    ax_pr.plot([0, 1], [baseline, baseline], 'k--', lw=1, c="grey")
    ax_pr.set_title("Holdout:\nPrecision Recall (PR)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid()
    ax_pr.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, title="Included time")  # outside right
    ax_pr.get_legend().get_title().set_fontsize(10)
    ax_pr.set_aspect('equal', adjustable='box') 

    # Adjust spacing to prevent overlap with PR legend
    fig.subplots_adjust(right=0.75)
    
    os.makedirs("reports/eval", exist_ok=True)
    plt.savefig("reports/eval/holdout_evaluation_multiple.png", dpi=1200)
    plt.show()
    return fig

def plot_time_metrics(results, cut_hours=72, max_days=None):
    if max_days is None:
        from astra.evaluation.utils import get_max_days
        max_days = get_max_days()
    times_h = np.array([r["time_hours"] for r in results])
    times_d = np.array([r["time_days"] for r in results])
    auroc_vals = np.array([r["AUROC"] for r in results])
    auroc_lower = np.array([r["AUROC_CI"][0] for r in results])
    auroc_upper = np.array([r["AUROC_CI"][1] for r in results])
    ap_vals = np.array([r["AP"] for r in results])
    ap_lower = np.array([r["AP_CI"][0] for r in results])
    ap_upper = np.array([r["AP_CI"][1] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    mask_cut = times_h <= cut_hours
    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals[mask_cut], auroc_lower[mask_cut], auroc_upper[mask_cut], 'o', "C0", "AUROC"),
        ("AP", ap_vals[mask_cut], ap_lower[mask_cut], ap_upper[mask_cut], 's', "C1", "AUPRC")
    ]:
        x = times_h[mask_cut]
        if len(x) > 0 and x[-1] < cut_hours:
            x_ext = np.append(x, cut_hours)
            vals_ext = np.append(vals, vals[-1])
            lower_ext = np.append(lower, lower[-1])
            upper_ext = np.append(upper, upper[-1])
        else:
            x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper
        ax1.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=3)
        ax1.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax1.set_xlabel("Time (hours)")
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, 73, 4))
    ax1.set_yticks(np.arange(0.0,1.1,0.1))
    ax1.set_ylabel("Score")
    ax1.set_title(f"A)")
    ax1.grid(True)
    ax1.legend()

    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals, auroc_lower, auroc_upper, 'o', "C0", "AUROC"),
        ("AP", ap_vals, ap_lower, ap_upper, 's', "C1", "AUPRC")
    ]:
        x = times_d
        if len(x) > 0 and x[-1] < max_days:
            x_ext = np.append(x, max_days)
            vals_ext = np.append(vals, vals[-1])
            lower_ext = np.append(lower, lower[-1])
            upper_ext = np.append(upper, upper[-1])
        else:
            x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper
        ax2.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=3)
        ax2.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax2.set_xlabel("Time (days)")
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days+1, 5))
    ax2.set_yticks(np.arange(0.0,1.1,0.1))
    ax2.set_ylabel("Score")
    ax2.set_title("B)")
    ax2.grid(True)
    ax2.legend(loc='lower right')

    plt.tight_layout(pad=3.0)
    return fig

