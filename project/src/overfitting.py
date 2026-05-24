import logging

logger = logging.getLogger(__name__)

class OverfittingAnalyzer:
    def __init__(self, results: dict):
        self.results = results
        self.logger = logging.getLogger(__name__)
        self.overfitting_threshold = 0.20

    def analyze_overfitting(self) -> dict:
        """Analyze overfitting by comparing train and test F1 scores."""
        try:
            analysis = {}

            for model_name, metrics in self.results.items():
                train_f1 = metrics.get('train_f1', None)
                test_f1 = metrics.get('f1', None)

                if train_f1 is not None and test_f1 is not None:
                    gap = train_f1 - test_f1
                    risk = 'HIGH' if gap > self.overfitting_threshold else 'LOW'

                    analysis[model_name] = {
                        'train_f1': float(train_f1),
                        'test_f1': float(test_f1),
                        'gap': float(gap),
                        'risk': risk,
                        'recommendation': self._get_recommendation(gap)
                    }

                    self.logger.info(f"{model_name}: gap={gap:.4f}, risk={risk}")

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing overfitting: {str(e)}")
            raise

    def _get_recommendation(self, gap: float) -> str:
        """Get recommendation based on overfitting gap."""
        if gap > 0.30:
            return "Very high overfitting. Consider simpler model or regularization."
        elif gap > 0.20:
            return "High overfitting. Increase regularization or reduce model complexity."
        elif gap > 0.10:
            return "Moderate overfitting. Monitor closely."
        else:
            return "Model generalizes well."

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("OverfittingAnalyzer module loaded successfully")
