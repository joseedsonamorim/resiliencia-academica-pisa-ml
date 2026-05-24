import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_path: str = "project/outputs/reports"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_model_report(self, results: dict, best_model: str, output_file: str = "model_results.md"):
        """Generate model results report."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report = f"""# Model Results Report
Generated: {timestamp}

## Best Model: {best_model}

"""
            for model_name, metrics in results.items():
                is_best = "✓ BEST" if model_name == best_model else ""
                report += f"""### {model_name} {is_best}

| Metric | Value |
|--------|-------|
| F1 Score | {metrics['f1']:.4f} |
| Recall | {metrics['recall']:.4f} |
| Precision | {metrics['precision']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Train F1 | {metrics.get('train_f1', 'N/A')} |

"""

            with open(self.output_path / output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Model report saved to {self.output_path / output_file}")

        except Exception as e:
            self.logger.error(f"Error generating model report: {str(e)}")

    def generate_audit_report(self, audit_results: dict, output_file: str = "audit_report.md"):
        """Generate audit report."""
        try:
            publishable = "✓ YES" if audit_results['publishable'] else "✗ NO"
            report = f"""# Scientific Audit Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Publishable: {publishable}

## Risk Level: {audit_results.get('overall_risk', 'UNKNOWN')}

## Issues ({len(audit_results.get('issues', []))})

"""
            for issue in audit_results.get('issues', []):
                report += f"- ❌ {issue}\n"

            report += f"""
## Warnings ({len(audit_results.get('warnings', []))})

"""
            for warning in audit_results.get('warnings', []):
                report += f"- ⚠️  {warning}\n"

            report += """
## Recommendations

"""
            for rec in audit_results.get('recommendations', []):
                report += f"- 📋 {rec}\n"

            with open(self.output_path / output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Audit report saved to {self.output_path / output_file}")

        except Exception as e:
            self.logger.error(f"Error generating audit report: {str(e)}")

    def generate_summary_report(self, all_results: dict, output_file: str = "summary_report.md"):
        """Generate comprehensive summary report."""
        try:
            report = f"""# PISA Creative Resilience - Comprehensive Summary Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Project Overview

This project analyzes creative resilience in Brazilian students using PISA 2022 microdata.

**Creative Resilience Definition:**
Students with low socioeconomic status (ESCS ≤ Q1) AND high creative thinking (CRT_SCORE ≥ Q3).

## Key Findings

### Dataset
- Total Students: {all_results.get('n_samples', 'N/A')}
- Resilient Students: {all_results.get('n_resilient', 'N/A')} ({all_results.get('pct_resilient', 0):.2f}%)
- Features Used: {all_results.get('n_features', 'N/A')}

### Best Model
Model: {all_results.get('best_model', 'N/A')}
- F1 Score: {all_results.get('best_f1', 'N/A')}
- Recall: {all_results.get('best_recall', 'N/A')}
- ROC-AUC: {all_results.get('best_roc_auc', 'N/A')}

### Scientific Audit
Publishable: {all_results.get('publishable', 'N/A')}
Risk Level: {all_results.get('audit_risk', 'N/A')}

## Fairness Analysis

Gender Fairness Disparity: {all_results.get('gender_disparity', 'N/A')}
ESCS Fairness Disparity: {all_results.get('escs_disparity', 'N/A')}

## Clustering Results

Silhouette Score: {all_results.get('silhouette', 'N/A')}
Davies-Bouldin Index: {all_results.get('davies_bouldin', 'N/A')}

## Recommendations

{all_results.get('recommendations_text', 'See audit report for details.')}

---
For detailed analysis, see individual reports in outputs/reports/
"""

            with open(self.output_path / output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Summary report saved to {self.output_path / output_file}")

        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")

    def save_json_results(self, data: dict, output_file: str = "results.json"):
        """Save results as JSON for dashboard."""
        try:
            with open(self.output_path / output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"JSON results saved to {self.output_path / output_file}")

        except Exception as e:
            self.logger.error(f"Error saving JSON results: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ReportGenerator module loaded successfully")
