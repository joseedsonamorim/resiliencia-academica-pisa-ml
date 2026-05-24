import numpy as np
import logging
from sklearn.metrics import f1_score, recall_score, precision_score

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.logger = logging.getLogger(__name__)

    def optimize_threshold(self, metric: str = 'f1_recall') -> tuple:
        """
        Optimize threshold based on metric.
        metric options: 'f1', 'f1_recall', 'balanced'
        """
        try:
            thresholds = np.arange(0.1, 0.95, 0.05)
            best_threshold = 0.5
            best_score = 0

            results = []

            for threshold in thresholds:
                y_pred = (self.y_pred_proba >= threshold).astype(int)

                f1 = f1_score(self.y_true, y_pred, zero_division=0)
                recall = recall_score(self.y_true, y_pred, zero_division=0)
                precision = precision_score(self.y_true, y_pred, zero_division=0)

                if metric == 'f1':
                    score = f1
                elif metric == 'f1_recall':
                    score = f1 + recall
                elif metric == 'balanced':
                    score = (f1 + recall) / 2

                results.append({
                    'threshold': threshold,
                    'f1': f1,
                    'recall': recall,
                    'precision': precision,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            self.logger.info(f"Optimal threshold: {best_threshold:.2f} (metric={metric})")
            return best_threshold, results

        except Exception as e:
            self.logger.error(f"Error optimizing threshold: {str(e)}")
            raise

    def get_detailed_results(self, threshold: float) -> dict:
        """Get detailed results for specific threshold."""
        y_pred = (self.y_pred_proba >= threshold).astype(int)

        return {
            'threshold': threshold,
            'f1': f1_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'n_positive': (y_pred == 1).sum(),
            'pct_positive': 100 * (y_pred == 1).sum() / len(y_pred)
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ThresholdOptimizer module loaded successfully")
