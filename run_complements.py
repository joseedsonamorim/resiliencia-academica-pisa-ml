#!/usr/bin/env python3
"""Run complementary scientific audits without modifying the main pipeline.

This script is OPTIONAL and does not change existing pipeline behavior.
It calls the new complementary modules:
- permutation_audit
- external_holdout
- resilient_profile
- cluster_interpretation
- scientific_robustness

Outputs are written to outputs/reports/, outputs/tables/, outputs/figures/.
"""

from src.permutation_audit import run_permutation_audit
from src.external_holdout import run_external_holdout
from src.resilient_profile import run_resilient_profile
from src.cluster_interpretation import run_cluster_interpretation
from src.scientific_robustness import run_scientific_robustness_report


def main():
    print('Running complement 1/5: permutation audit')
    run_permutation_audit()

    print('Running complement 2/5: external holdout')
    run_external_holdout()

    print('Running complement 3/5: resilient profile')
    run_resilient_profile()

    print('Running complement 4/5: cluster interpretation')
    run_cluster_interpretation()

    print('Running complement 5/5: scientific robustness report')
    run_scientific_robustness_report()

    print('All complements completed.')


if __name__ == '__main__':
    main()

