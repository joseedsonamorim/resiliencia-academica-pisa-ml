#!/bin/zsh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

STAGES=(
  data_audit
  data_dictionary
  variable_discovery
  leakage_audit
  target_comparison
  eda
  resilient_profile
  clusterer
  modeling
  shap
  fairness
  robustness
  dashboard
)

for stage in "${STAGES[@]}"; do
  echo "=== $stage ==="
  "$DIR/run_stage.sh" "$stage"
done

echo "\nPipeline completo. Relatórios em outputs/reports/"
