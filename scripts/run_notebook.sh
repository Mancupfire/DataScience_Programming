#!/usr/bin/env bash
set -euo pipefail

# Execute any notebook headlessly with the project virtualenv.
# Usage: ./scripts/run_notebook.sh path/to/notebook.ipynb

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/notebook.ipynb" >&2
  exit 1
fi

NOTEBOOK_PATH="$1"
if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "Notebook not found: $NOTEBOOK_PATH" >&2
  exit 1
fi

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

NOTEBOOK_ABS="$(cd "$(dirname "$NOTEBOOK_PATH")" && pwd)/$(basename "$NOTEBOOK_PATH")"
NOTEBOOK_DIR="$(dirname "$NOTEBOOK_ABS")"
NOTEBOOK_FILE="$(basename "$NOTEBOOK_ABS")"
OUTPUT_NOTEBOOK="$(basename "${NOTEBOOK_FILE%.ipynb}").executed.ipynb"

cd "$NOTEBOOK_DIR"
echo "Executing notebook ($NOTEBOOK_FILE) with nbconvert..."
jupyter nbconvert --to notebook --execute "$NOTEBOOK_FILE" \
  --output "$OUTPUT_NOTEBOOK" \
  --output-dir "." \
  --ExecutePreprocessor.timeout=900 \
  --ExecutePreprocessor.kernel_name=python3

echo "Notebook run complete. Executed copy: $NOTEBOOK_DIR/$OUTPUT_NOTEBOOK"
