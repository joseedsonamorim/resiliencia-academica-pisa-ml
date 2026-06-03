from __future__ import annotations

import argparse

from src.pipeline_stages import run_stage

STAGES = [
    "data_audit",
    "data_dictionary",
    "variable_discovery",
    "leakage_audit",
    "target_comparison",
    "eda",
    "resilient_profile",
    "clusterer",
    "modeling",
    "shap",
    "fairness",
    "robustness",
    "dashboard",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PISA 2022 Creative Resilience pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGES,
        help="Pipeline stage",
    )
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
