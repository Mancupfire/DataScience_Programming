#!/usr/bin/env bash
set -euo pipefail

# One-command executor for Training Model/lightgbm.ipynb

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

NOTEBOOK_DIR="$ROOT_DIR/Training Model"
NOTEBOOK_FILE="lightgbm.ipynb"
OUTPUT_NOTEBOOK="lightgbm.executed.ipynb"

cd "$NOTEBOOK_DIR"
echo "Executing notebook ($NOTEBOOK_FILE) with nbconvert..."
jupyter nbconvert --to notebook --execute "$NOTEBOOK_FILE" \
  --output "$OUTPUT_NOTEBOOK" \
  --output-dir "." \
  --ExecutePreprocessor.timeout=900 \
  --ExecutePreprocessor.kernel_name=python3

echo "Notebook run complete."
echo "Artifacts expected at Training Model/lightgbm_model.txt and Training Model/lightgbm_features.json."
