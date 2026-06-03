#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAGE="${1:?Uso: ./scripts/run_stage.sh <stage>}"

if [ ! -d .venv ]; then
  echo ".venv não existe. Rode ./venv_setup.sh primeiro."
  exit 1
fi

source .venv/bin/activate

mkdir -p data outputs models dashboard src tests docs config project \
  outputs/figures outputs/tables outputs/reports

if [ ! -f data/pisa_brasil_estudo_limpo.csv ]; then
  if [ -f data/archive/pisa_brasil_estudo_limpo.csv ]; then
    cp data/archive/pisa_brasil_estudo_limpo.csv data/pisa_brasil_estudo_limpo.csv
  fi
fi

python3 -m src.main --stage "$STAGE"
echo "\nOK: stage '$STAGE' concluído."
