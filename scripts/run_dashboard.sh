#!/bin/zsh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

"$DIR/run_stage.sh" dashboard

cd "$ROOT"
source .venv/bin/activate
exec python3 -m streamlit run dashboard/app.py
