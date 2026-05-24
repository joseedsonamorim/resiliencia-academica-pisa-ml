import pandas as pd
import numpy as np
import logging
import gc
from pathlib import Path

logger = logging.getLogger(__name__)

class TargetBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)

    def calculate_crt_score(self) -> pd.Series:
        """Calculate CRT_SCORE as mean of PV1CRT to PV10CRT."""
        crt_cols = [f"PV{i}CRT" for i in range(1, 11)]
        available_cols = [col for col in crt_cols if col in self.df.columns]

        if not available_cols:
            raise ValueError("No CRT columns found in dataset")

        crt_score = self.df[available_cols].mean(axis=1, skipna=True)
        self.logger.info(f"Calculated CRT_SCORE from {len(available_cols)} PV columns")
        return crt_score

    def get_q1_escs(self) -> float:
        """Get Q1 (25th percentile) of ESCS."""
        if 'ESCS' not in self.df.columns:
            raise ValueError("ESCS column not found in dataset")

        q1_escs = self.df['ESCS'].quantile(0.25)
        self.logger.info(f"Q1_ESCS (25th percentile): {q1_escs:.4f}")
        return q1_escs

    def get_q3_crt(self, crt_score: pd.Series) -> float:
        """Get Q3 (75th percentile) of CRT_SCORE."""
        q3_crt = crt_score.quantile(0.75)
        self.logger.info(f"Q3_CRT (75th percentile): {q3_crt:.4f}")
        return q3_crt

    def build_target(self) -> pd.DataFrame:
        """Build Creative_Resilience target variable."""
        try:
            crt_score = self.calculate_crt_score()
            q1_escs = self.get_q1_escs()
            q3_crt = self.get_q3_crt(crt_score)

            # Create target variable
            self.df['CRT_SCORE'] = crt_score
            self.df['Creative_Resilience'] = (
                ((self.df['ESCS'] <= q1_escs) & (crt_score >= q3_crt)).astype(int)
            )

            # Statistics
            n_resilient = self.df['Creative_Resilience'].sum()
            pct_resilient = 100 * n_resilient / len(self.df)

            stats = {
                'total_students': len(self.df),
                'q1_escs': q1_escs,
                'q3_crt': q3_crt,
                'n_resilient': int(n_resilient),
                'pct_resilient': float(pct_resilient),
                'crt_score_mean': float(crt_score.mean()),
                'crt_score_std': float(crt_score.std()),
                'escs_mean': float(self.df['ESCS'].mean()),
                'escs_std': float(self.df['ESCS'].std())
            }

            self.logger.info(f"Target built: {n_resilient} resilient students ({pct_resilient:.2f}%)")
            gc.collect()
            return self.df, stats

        except Exception as e:
            self.logger.error(f"Error building target: {str(e)}")
            raise

    def save_target_report(self, stats: dict, output_path: str = "project/outputs/reports/target_report.md"):
        """Save target construction report."""
        try:
            report = f"""# Creative Resilience Target Construction Report

## Definition

A student is **creatively resilient** when:
- ESCS (socioeconomic status) ≤ Q1 (25th percentile)
- CRT_SCORE (creativity) ≥ Q3 (75th percentile)

## Statistics

| Metric | Value |
|--------|-------|
| Total Students | {stats['total_students']:,} |
| Q1 ESCS (25th percentile) | {stats['q1_escs']:.4f} |
| Q3 CRT_SCORE (75th percentile) | {stats['q3_crt']:.4f} |
| Resilient Students | {stats['n_resilient']:,} |
| Percentage Resilient | {stats['pct_resilient']:.2f}% |
| CRT_SCORE Mean | {stats['crt_score_mean']:.4f} |
| CRT_SCORE Std Dev | {stats['crt_score_std']:.4f} |
| ESCS Mean | {stats['escs_mean']:.4f} |
| ESCS Std Dev | {stats['escs_std']:.4f} |

## Interpretation

Students with low socioeconomic status (bottom 25%) who demonstrate high creative thinking (top 25%) represent creative resilience - the ability to think creatively despite socioeconomic constraints.

"""
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Target report saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving target report: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("TargetBuilder module loaded successfully")
