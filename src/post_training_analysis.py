"""Post-training analysis: threshold optimization, bootstrap, calibration."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from src.config import get_config
from src.utils import setup_logger, save_metadata

logger = setup_logger(__name__)


class ThresholdOptimizer:
    """Optimize classification threshold for best performance."""

    def __init__(self, config=None):
        """Initialize optimizer.

        Args:
            config: ConfigManager instance
        """
        self.config = config or get_config()

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, Any]]:
        """Find optimal threshold for given metric.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'youden_j', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metrics_dict)
        """
        logger.info(f"\n Optimizing threshold for metric: {metric}")

        thresholds = np.arange(0.0, 1.01, 0.01)
        metrics_by_threshold = {}

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            metrics = {
                'threshold': threshold,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
            }

            # Specificity
            tn = ((1 - y_true) * (1 - y_pred)).sum()
            fp = ((1 - y_true) * y_pred).sum()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['specificity'] = specificity

            # Youden's J statistic
            metrics['youden_j'] = metrics['recall'] + specificity - 1

            metrics_by_threshold[threshold] = metrics

        # Find optimal threshold
        if metric == 'youden_j':
            optimal_threshold = max(metrics_by_threshold, key=lambda x: metrics_by_threshold[x]['youden_j'])
        else:
            optimal_threshold = max(metrics_by_threshold, key=lambda x: metrics_by_threshold[x].get(metric, 0))

        optimal_metrics = metrics_by_threshold[optimal_threshold]

        logger.info(f"   Optimal threshold: {optimal_threshold:.2f}")
        logger.info(f"   Metrics at optimal threshold:")
        for metric_name, value in optimal_metrics.items():
            if metric_name != 'threshold':
                logger.info(f"      {metric_name}: {value:.4f}")

        return optimal_threshold, metrics_by_threshold

    def apply_threshold(self, y_pred_proba: np.ndarray, threshold: float) -> np.ndarray:
        """Apply threshold to probabilities.

        Args:
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        return (y_pred_proba >= threshold).astype(int)


class BootstrapAnalyzer:
    """Compute confidence intervals using bootstrap resampling."""

    def __init__(self, config=None, random_state=42):
        """Initialize analyzer.

        Args:
            config: ConfigManager instance
            random_state: Random seed
        """
        self.config = config or get_config()
        self.random_state = random_state

    def compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_iterations: int = 1000,
        ci: float = 95
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_iterations: Number of bootstrap iterations
            ci: Confidence interval (default 95%)

        Returns:
            Dictionary with CI for each metric
        """
        logger.info(f"\n Computing bootstrap CI (n={n_iterations})...")

        n = len(y_true)
        metrics_bootstrap = {
            'roc_auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'brier_score': []
        }

        np.random.seed(self.random_state)

        for iteration in range(n_iterations):
            # Resample with replacement
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_proba_boot = y_pred_proba[indices]

            # Compute metrics
            try:
                metrics_bootstrap['roc_auc'].append(roc_auc_score(y_true_boot, y_pred_proba_boot))
            except:
                pass

            y_pred_boot = (y_pred_proba_boot >= 0.5).astype(int)

            try:
                metrics_bootstrap['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
                metrics_bootstrap['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
                metrics_bootstrap['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
                metrics_bootstrap['brier_score'].append(brier_score_loss(y_true_boot, y_pred_proba_boot))
            except:
                pass

            # Specificity
            tn = ((1 - y_true_boot) * (1 - y_pred_boot)).sum()
            fp = ((1 - y_true_boot) * y_pred_boot).sum()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics_bootstrap['specificity'].append(specificity)

            if (iteration + 1) % 250 == 0:
                logger.info(f"  Completed {iteration + 1}/{n_iterations} iterations")

        # Compute confidence intervals
        ci_lower = (100 - ci) / 2
        ci_upper = 100 - ci_lower

        results = {}
        for metric_name, values in metrics_bootstrap.items():
            if values:
                results[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'ci_lower': float(np.percentile(values, ci_lower)),
                    'ci_upper': float(np.percentile(values, ci_upper))
                }

        logger.info(f"   Bootstrap CI computed for {len(results)} metrics")

        return results

    def save_bootstrap_results(self, results: Dict[str, Any], output_path: Path):
        """Save bootstrap results to JSON.

        Args:
            results: Bootstrap results dictionary
            output_path: Output file path
        """
        save_metadata(results, output_path, "Bootstrap confidence intervals (95%)")


class CalibrationAnalyzer:
    """Analyze and improve model calibration."""

    def __init__(self, config=None):
        """Initialize analyzer.

        Args:
            config: ConfigManager instance
        """
        self.config = config or get_config()

    def analyze_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Analyze model calibration using calibration curves.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with calibration metrics
        """
        logger.info("\n Analyzing model calibration...")

        # Expected calibration error
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

        # Brier score (measure of calibration)
        brier = brier_score_loss(y_true, y_pred_proba)

        results = {
            'brier_score': float(brier),
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'n_bins': n_bins
        }

        logger.info(f"   Brier Score: {brier:.4f}")
        logger.info(f"    (Lower is better; 0=perfect, 1=worst)")

        return results

    def calibrate_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = 'sigmoid'
    ):
        """Calibrate model using CalibratedClassifierCV.

        Args:
            model: Base model to calibrate
            X_train: Training features
            y_train: Training labels
            method: Calibration method ('sigmoid' or 'isotonic')

        Returns:
            Calibrated model
        """
        logger.info(f"\n Calibrating model using {method}...")

        calibrated_model = CalibratedClassifierCV(model, method=method, cv=5)
        calibrated_model.fit(X_train, y_train)

        logger.info(f"   Model calibrated with {method} method")

        return calibrated_model
