# uncertainty.py
"""
Uncertainty quantification for time-dependent predictions.

Combines Monte-Carlo Dropout and Conformal Prediction to provide:
1. Epistemic uncertainty estimates (model uncertainty)
2. Statistically guaranteed prediction intervals
3. Patient-specific and time-specific uncertainty metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from astra.utils import logger


@dataclass
class UncertaintyResult:
    """Container for uncertainty quantification results"""
    # Predictions
    pred_mean: np.ndarray          # Mean prediction across MC samples
    pred_std: np.ndarray           # Standard deviation (epistemic uncertainty)
    pred_samples: np.ndarray       # All MC dropout samples [n_mc, n_patients]

    # Uncertainty metrics
    entropy: np.ndarray            # Predictive entropy
    bald: np.ndarray              # Bayesian Active Learning by Disagreement

    # Conformal prediction intervals
    conformal_lower: Optional[np.ndarray] = None
    conformal_upper: Optional[np.ndarray] = None
    conformal_width: Optional[np.ndarray] = None

    # Metadata
    n_mc_samples: int = 30
    confidence_level: float = 0.9


class MCDropoutPredictor:
    """
    Monte-Carlo Dropout for epistemic uncertainty estimation.

    Performs multiple forward passes with dropout enabled to estimate
    model uncertainty. The variance in predictions indicates how uncertain
    the model is about each prediction.
    """

    def __init__(self, n_samples: int = 30):
        """
        Args:
            n_samples: Number of MC dropout samples (10-30 is reasonable)
        """
        self.n_samples = n_samples

    def predict_with_uncertainty(
        self,
        learner,
        dataloader,
        return_all_samples: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions with MC dropout uncertainty estimates.

        Args:
            learner: Trained fastai Learner
            dataloader: DataLoader to predict on
            return_all_samples: Whether to return all MC samples

        Returns:
            Tuple of (mean_pred, std_pred, entropy, all_samples)
            - mean_pred: [n_patients] mean prediction
            - std_pred: [n_patients] epistemic uncertainty (std dev)
            - entropy: [n_patients] predictive entropy
            - all_samples: [n_mc, n_patients] all MC samples (if return_all_samples=True)
        """
        # Enable dropout for MC sampling
        learner.model.train()

        all_predictions = []

        logger.info(f"Running MC Dropout with {self.n_samples} samples...")

        for i in range(self.n_samples):
            with torch.no_grad():
                preds, _ = learner.get_preds(dl=dataloader)

            # Get positive class probability
            pred_probs = preds[:, 1].cpu().numpy()
            all_predictions.append(pred_probs)

            if (i + 1) % 10 == 0:
                logger.info(f"  MC sample {i+1}/{self.n_samples}")

        # Stack predictions: [n_samples, n_patients]
        predictions = np.stack(all_predictions, axis=0)

        # Calculate statistics
        pred_mean = predictions.mean(axis=0)
        pred_std = predictions.std(axis=0)

        # Predictive entropy: E[-log p(y|x)]
        # Higher entropy = more uncertain
        entropy = self._calculate_entropy(predictions)

        logger.info("✓ MC Dropout complete")

        if return_all_samples:
            return pred_mean, pred_std, entropy, predictions
        else:
            return pred_mean, pred_std, entropy, None

    def _calculate_entropy(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate predictive entropy from MC samples.

        For binary classification:
        H = -[p*log(p) + (1-p)*log(1-p)]

        Args:
            predictions: [n_samples, n_patients] array of probabilities

        Returns:
            entropy: [n_patients] array of entropy values
        """
        # Average probability across MC samples
        p_mean = predictions.mean(axis=0)

        # Avoid log(0)
        p_mean = np.clip(p_mean, 1e-10, 1 - 1e-10)

        # Binary entropy
        entropy = -(p_mean * np.log(p_mean) + (1 - p_mean) * np.log(1 - p_mean))

        return entropy

    def calculate_bald(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate BALD (Bayesian Active Learning by Disagreement) score.

        BALD measures the mutual information between predictions and model parameters.
        It's the difference between predictive entropy and expected entropy:
        BALD = H(y|x) - E[H(y|x,θ)]

        Higher BALD = more epistemic uncertainty (useful for active learning)

        Args:
            predictions: [n_samples, n_patients] array of probabilities

        Returns:
            bald: [n_patients] array of BALD scores
        """
        # Predictive entropy: H(y|x)
        p_mean = predictions.mean(axis=0)
        p_mean = np.clip(p_mean, 1e-10, 1 - 1e-10)
        predictive_entropy = -(p_mean * np.log(p_mean) +
                               (1 - p_mean) * np.log(1 - p_mean))

        # Expected entropy: E[H(y|x,θ)]
        predictions_clipped = np.clip(predictions, 1e-10, 1 - 1e-10)
        sample_entropies = -(predictions_clipped * np.log(predictions_clipped) +
                            (1 - predictions_clipped) * np.log(1 - predictions_clipped))
        expected_entropy = sample_entropies.mean(axis=0)

        # BALD = difference
        bald = predictive_entropy - expected_entropy

        return bald


class ConformalPredictor:
    """
    Conformal Prediction for statistically guaranteed prediction intervals.

    Provides prediction intervals with guaranteed coverage probability,
    e.g., 90% of true labels will fall within the predicted intervals.

    This is especially important for clinical applications where you need
    reliable uncertainty bounds with statistical guarantees.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level (0.1 = 90% confidence intervals)
        """
        self.alpha = alpha
        self.q_hat = None  # Calibrated threshold
        self.confidence_level = 1 - alpha

    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Calibrate conformal predictor on a calibration set.

        This should be called on a held-out calibration set (NOT the test set!)
        Typically: split your validation set into cal_set + test_set

        Args:
            predictions: [n_cal] predicted probabilities
            targets: [n_cal] true labels (0 or 1)

        Returns:
            q_hat: Calibrated quantile threshold
        """
        # Nonconformity scores: how "wrong" is each prediction?
        # For binary classification: use 1 - p(correct class)
        scores = np.zeros(len(predictions))
        for i in range(len(predictions)):
            if targets[i] == 1:
                scores[i] = 1 - predictions[i]
            else:
                scores[i] = predictions[i]

        # Calculate quantile with finite-sample correction
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, q_level)

        logger.info(f"Conformal calibration: q_hat={self.q_hat:.4f} "
                   f"(confidence={self.confidence_level*100:.0f}%)")

        return self.q_hat

    def predict_intervals(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute prediction intervals for test set.

        Args:
            predictions: [n_test] predicted probabilities

        Returns:
            Tuple of (lower_bound, upper_bound, interval_width)
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before predict_intervals()")

        # Prediction intervals
        lower = np.maximum(0, predictions - self.q_hat)
        upper = np.minimum(1, predictions + self.q_hat)
        width = upper - lower

        return lower, upper, width

    def get_prediction_sets(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction sets (which classes are predicted).

        For binary classification, returns whether each class is included
        in the prediction set.

        Args:
            predictions: [n_test] predicted probabilities
            threshold: Decision threshold

        Returns:
            Tuple of (include_class_0, include_class_1)
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before get_prediction_sets()")

        # Class 0 included if p(y=0) > threshold - q_hat
        # Class 1 included if p(y=1) > threshold - q_hat
        include_class_1 = predictions > (threshold - self.q_hat)
        include_class_0 = (1 - predictions) > (threshold - self.q_hat)

        return include_class_0, include_class_1


class UncertaintyQuantifier:
    """
    Combined uncertainty quantification using MC Dropout + Conformal Prediction.

    Usage:
        # Initialize
        uq = UncertaintyQuantifier(n_mc_samples=30, alpha=0.1)

        # Calibrate on calibration set
        uq.calibrate(learner, cal_dataloader, cal_targets)

        # Get uncertainties on test set
        results = uq.predict_with_uncertainty(learner, test_dataloader)
    """

    def __init__(
        self,
        n_mc_samples: int = 30,
        alpha: float = 0.1,
        use_conformal: bool = True
    ):
        """
        Args:
            n_mc_samples: Number of MC dropout samples
            alpha: Significance level for conformal prediction
            use_conformal: Whether to use conformal prediction
        """
        self.mc_predictor = MCDropoutPredictor(n_samples=n_mc_samples)
        self.conformal_predictor = ConformalPredictor(alpha=alpha) if use_conformal else None
        self.n_mc_samples = n_mc_samples
        self.alpha = alpha
        self.is_calibrated = False

    def calibrate(
        self,
        learner,
        cal_dataloader,
        cal_targets: np.ndarray
    ):
        """
        Calibrate conformal predictor on calibration set.

        Args:
            learner: Trained fastai Learner
            cal_dataloader: Calibration data loader
            cal_targets: True labels for calibration set
        """
        if self.conformal_predictor is None:
            logger.warning("Conformal prediction disabled, skipping calibration")
            return

        logger.info("Calibrating conformal predictor...")

        # Get MC predictions on calibration set
        pred_mean, _, _, _ = self.mc_predictor.predict_with_uncertainty(
            learner, cal_dataloader, return_all_samples=False
        )

        # Calibrate
        self.conformal_predictor.calibrate(pred_mean, cal_targets)
        self.is_calibrated = True

        logger.info("✓ Calibration complete")

    def predict_with_uncertainty(
        self,
        learner,
        dataloader
    ) -> UncertaintyResult:
        """
        Get predictions with full uncertainty quantification.

        Args:
            learner: Trained fastai Learner
            dataloader: Test data loader

        Returns:
            UncertaintyResult with all uncertainty metrics
        """
        # MC Dropout predictions
        pred_mean, pred_std, entropy, predictions = \
            self.mc_predictor.predict_with_uncertainty(
                learner, dataloader, return_all_samples=True
            )

        # BALD score
        bald = self.mc_predictor.calculate_bald(predictions)

        # Conformal intervals
        if self.conformal_predictor is not None and self.is_calibrated:
            lower, upper, width = self.conformal_predictor.predict_intervals(pred_mean)
        else:
            lower = upper = width = None

        return UncertaintyResult(
            pred_mean=pred_mean,
            pred_std=pred_std,
            pred_samples=predictions,
            entropy=entropy,
            bald=bald,
            conformal_lower=lower,
            conformal_upper=upper,
            conformal_width=width,
            n_mc_samples=self.n_mc_samples,
            confidence_level=self.conformal_predictor.confidence_level if self.conformal_predictor else None
        )


# ============================================================================
# UNCERTAINTY METRICS FOR EVALUATION
# ============================================================================

def calculate_calibration_metrics(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate calibration metrics for uncertainty estimates.

    Good uncertainty estimates should:
    - Have high uncertainty for incorrect predictions
    - Have low uncertainty for correct predictions
    - Be well-calibrated (predicted probabilities match true frequencies)

    Args:
        predictions: [n] predicted probabilities
        uncertainties: [n] uncertainty estimates (e.g., std dev)
        targets: [n] true labels
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_conf_in_bin = predictions[in_bin].mean()

            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_conf_in_bin)

    # Uncertainty-accuracy correlation
    # High uncertainty should correlate with low accuracy
    correct = (predictions > 0.5) == targets
    uncertainty_acc_corr = np.corrcoef(uncertainties, correct.astype(float))[0, 1]

    # Separation: uncertainty for errors vs correct
    uncertainty_correct = uncertainties[correct].mean()
    uncertainty_errors = uncertainties[~correct].mean()

    return {
        'ece': ece,
        'uncertainty_acc_correlation': uncertainty_acc_corr,
        'uncertainty_correct_mean': uncertainty_correct,
        'uncertainty_errors_mean': uncertainty_errors,
        'uncertainty_separation': uncertainty_errors - uncertainty_correct
    }


def analyze_uncertainty_by_time(
    uncertainty_results: List[UncertaintyResult],
    time_points: List[float]
) -> Dict[str, np.ndarray]:
    """
    Analyze how uncertainty changes over time.

    Args:
        uncertainty_results: List of UncertaintyResult for each time point
        time_points: List of time values (e.g., hours)

    Returns:
        Dictionary with temporal uncertainty analysis
    """
    # Extract metrics over time
    mean_std_over_time = [ur.pred_std.mean() for ur in uncertainty_results]
    mean_entropy_over_time = [ur.entropy.mean() for ur in uncertainty_results]

    if uncertainty_results[0].conformal_width is not None:
        mean_width_over_time = [ur.conformal_width.mean() for ur in uncertainty_results]
    else:
        mean_width_over_time = None

    return {
        'time_points': np.array(time_points),
        'mean_epistemic_uncertainty': np.array(mean_std_over_time),
        'mean_entropy': np.array(mean_entropy_over_time),
        'mean_conformal_width': np.array(mean_width_over_time) if mean_width_over_time else None
    }
