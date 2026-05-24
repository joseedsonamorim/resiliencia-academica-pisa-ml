import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score, recall_score, precision_score

logger = logging.getLogger(__name__)

class FairnessAnalyzer:
    def __init__(self, df: pd.DataFrame, y_pred, y_pred_proba, y_true):
        self.df = df
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.y_true = y_true
        self.logger = logging.getLogger(__name__)

    def analyze_by_gender(self) -> dict:
        """Analyze fairness by gender (ST004D01T)."""
        try:
            if 'ST004D01T' not in self.df.columns:
                self.logger.warning("ST004D01T (gender) not found in dataset")
                return {}

            analysis = {}
            genders = self.df['ST004D01T'].unique()

            for gender in genders:
                mask = self.df['ST004D01T'] == gender
                if mask.sum() > 0:
                    analysis[f'Gender_{int(gender)}'] = {
                        'n_samples': int(mask.sum()),
                        'f1': f1_score(self.y_true[mask], self.y_pred[mask], zero_division=0),
                        'recall': recall_score(self.y_true[mask], self.y_pred[mask], zero_division=0),
                        'precision': precision_score(self.y_true[mask], self.y_pred[mask], zero_division=0),
                        'positive_rate': float((self.y_pred[mask] == 1).sum() / mask.sum())
                    }

            self.logger.info(f"Gender fairness analysis completed for {len(analysis)} groups")
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing gender fairness: {str(e)}")
            raise

    def analyze_by_escs_quartile(self) -> dict:
        """Analyze fairness by ESCS quartiles."""
        try:
            if 'ESCS' not in self.df.columns:
                self.logger.warning("ESCS not found in dataset")
                return {}

            analysis = {}
            quartiles = pd.qcut(self.df['ESCS'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

            for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = quartiles == quartile
                if mask.sum() > 0:
                    analysis[f'ESCS_{quartile}'] = {
                        'n_samples': int(mask.sum()),
                        'f1': f1_score(self.y_true[mask], self.y_pred[mask], zero_division=0),
                        'recall': recall_score(self.y_true[mask], self.y_pred[mask], zero_division=0),
                        'precision': precision_score(self.y_true[mask], self.y_pred[mask], zero_division=0),
                        'positive_rate': float((self.y_pred[mask] == 1).sum() / mask.sum())
                    }

            self.logger.info(f"ESCS fairness analysis completed for {len(analysis)} quartiles")
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing ESCS fairness: {str(e)}")
            raise

    def get_fairness_report(self) -> dict:
        """Get complete fairness report."""
        try:
            report = {
                'gender_analysis': self.analyze_by_gender(),
                'escs_analysis': self.analyze_by_escs_quartile()
            }

            # Calculate disparities
            gender_data = report['gender_analysis']
            if len(gender_data) > 1:
                f1_scores = [v['f1'] for v in gender_data.values()]
                report['gender_f1_disparity'] = float(max(f1_scores) - min(f1_scores))

            escs_data = report['escs_analysis']
            if len(escs_data) > 1:
                f1_scores = [v['f1'] for v in escs_data.values()]
                report['escs_f1_disparity'] = float(max(f1_scores) - min(f1_scores))

            return report

        except Exception as e:
            self.logger.error(f"Error generating fairness report: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("FairnessAnalyzer module loaded successfully")
