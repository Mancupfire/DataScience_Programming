#!/usr/bin/env bash
set -euo pipefail

# Simple one-command launcher for the Streamlit app.

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

DATA_FILE="cleaned_dataset.csv"
if [[ ! -f "$DATA_FILE" ]]; then
  echo "WARNING: $DATA_FILE not found in the project root. You can upload a CSV inside the app."
fi

echo "Launching Streamlit..."
exec streamlit run app.py
