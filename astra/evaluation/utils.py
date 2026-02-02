from astra.utils import logger, cfg
from astra.models.hybrid.training import get_backbone, Learner, patch_learner_get_preds

import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def prepare_learner(data, model_name=None):
    if model_name is None:
        logger.info(f"Using default model name from cfg: {cfg['model_name']}")
        model_name = cfg["model_name"]

    logger.info(f"Loading model: {model_name}")
    backbone = get_backbone(data, cfg)
    learn = Learner(data["mixed_dls"], backbone, metrics=None)
    learn.load(model_name)
    learn.to('cuda')
    learn = patch_learner_get_preds(learn)
    logger.info("Model loaded")
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