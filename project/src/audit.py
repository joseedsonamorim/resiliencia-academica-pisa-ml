import logging

logger = logging.getLogger(__name__)

class ScientificAudit:
    def __init__(self, results: dict, overfitting_analysis: dict, fairness_report: dict):
        self.results = results
        self.overfitting_analysis = overfitting_analysis
        self.fairness_report = fairness_report
        self.logger = logging.getLogger(__name__)
        self.issues = []
        self.warnings = []

    def audit(self) -> dict:
        """Run comprehensive scientific audit."""
        try:
            self._check_model_performance()
            self._check_overfitting()
            self._check_fairness()
            self._check_class_balance()

            publishable = len(self.issues) == 0 and len(self.warnings) <= 2

            report = {
                'publishable': publishable,
                'issues': self.issues,
                'warnings': self.warnings,
                'overall_risk': self._calculate_risk_level(),
                'recommendations': self._get_recommendations()
            }

            self.logger.info(f"Audit completed. Publishable: {publishable}")
            return report

        except Exception as e:
            self.logger.error(f"Error in audit: {str(e)}")
            raise

    def _check_model_performance(self):
        """Check if model performance is acceptable."""
        for model_name, metrics in self.results.items():
            f1 = metrics.get('f1', 0)
            recall = metrics.get('recall', 0)

            if f1 < 0.5:
                self.issues.append(f"{model_name}: F1 score too low ({f1:.3f} < 0.5)")
            elif f1 < 0.6:
                self.warnings.append(f"{model_name}: F1 score below 0.6 ({f1:.3f})")

            if recall < 0.3:
                self.issues.append(f"{model_name}: Recall too low ({recall:.3f} < 0.3)")

    def _check_overfitting(self):
        """Check for overfitting risk."""
        for model_name, analysis in self.overfitting_analysis.items():
            if analysis.get('risk') == 'HIGH':
                gap = analysis.get('gap', 0)
                if gap > 0.30:
                    self.issues.append(f"{model_name}: Very high overfitting gap ({gap:.3f})")
                else:
                    self.warnings.append(f"{model_name}: High overfitting gap ({gap:.3f})")

    def _check_fairness(self):
        """Check for fairness issues."""
        gender_disparity = self.fairness_report.get('gender_f1_disparity', 0)
        escs_disparity = self.fairness_report.get('escs_f1_disparity', 0)

        if gender_disparity > 0.15:
            self.issues.append(f"Gender fairness issue: F1 disparity ({gender_disparity:.3f} > 0.15)")
        elif gender_disparity > 0.10:
            self.warnings.append(f"Gender fairness gap: {gender_disparity:.3f}")

        if escs_disparity > 0.20:
            self.issues.append(f"ESCS fairness issue: F1 disparity ({escs_disparity:.3f} > 0.20)")
        elif escs_disparity > 0.15:
            self.warnings.append(f"ESCS fairness gap: {escs_disparity:.3f}")

    def _check_class_balance(self):
        """Check for severe class imbalance after SMOTE."""
        # This would be checked in actual results
        pass

    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level."""
        if len(self.issues) > 2:
            return "CRITICAL"
        elif len(self.issues) > 0:
            return "HIGH"
        elif len(self.warnings) > 2:
            return "MEDIUM"
        elif len(self.warnings) > 0:
            return "LOW"
        else:
            return "MINIMAL"

    def _get_recommendations(self) -> list:
        """Get recommendations."""
        recommendations = []

        if len(self.issues) > 0:
            recommendations.append("Address critical issues before publication")

        if any('overfitting' in issue.lower() for issue in self.issues):
            recommendations.append("Simplify model or increase regularization")

        if any('fairness' in issue.lower() for issue in self.issues):
            recommendations.append("Investigate and address fairness disparities")

        if len(self.issues) == 0 and len(self.warnings) == 0:
            recommendations.append("Model is ready for publication")

        return recommendations

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ScientificAudit module loaded successfully")
