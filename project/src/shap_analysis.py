import numpy as np
import pandas as pd
import logging
import gc
import shap
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    def __init__(self, model, X_train, X_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.logger = logging.getLogger(__name__)

    def create_explainer(self, sample_size: int = 500):
        """Create SHAP explainer with sample for memory efficiency."""
        try:
            # Sample for memory efficiency
            if len(self.X_train) > sample_size:
                sample_indices = np.random.choice(len(self.X_train), sample_size, replace=False)
                X_sample = self.X_train.iloc[sample_indices]
            else:
                X_sample = self.X_train

            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            self.logger.info(f"SHAP explainer created with {len(X_sample)} samples")
            gc.collect()
            return explainer, X_sample

        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer: {str(e)}")
            raise

    def calculate_shap_values(self, explainer, X_sample: int = 500):
        """Calculate SHAP values for test set."""
        try:
            if len(self.X_test) > X_sample:
                test_indices = np.random.choice(len(self.X_test), X_sample, replace=False)
                X_test_sample = self.X_test.iloc[test_indices]
            else:
                X_test_sample = self.X_test

            shap_values = explainer.shap_values(X_test_sample)

            # For binary classification, take positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            self.logger.info(f"SHAP values calculated for {len(X_test_sample)} samples")
            gc.collect()
            return shap_values, X_test_sample

        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {str(e)}")
            raise

    def plot_summary(self, explainer, shap_values, X_sample, output_path: str = "project/outputs/figures/shap_summary.png"):
        """Plot SHAP summary plot."""
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"SHAP summary plot saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error plotting SHAP summary: {str(e)}")

    def get_feature_importance(self, shap_values) -> pd.DataFrame:
        """Get feature importance from SHAP values."""
        try:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': mean_abs_shap
            }).sort_values('shap_importance', ascending=False)

            return importance_df

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SHAPAnalyzer module loaded successfully")
