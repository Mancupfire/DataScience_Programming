# V-IndoorCARE Prototype

Indoor air-quality dashboard with forecasting and quick-start automation.

## Quick start (macOS/Linux)
From the project root:
```bash
./scripts/run.sh
```
The script will create `.venv`, install `requirements.txt`, check for the bundled CSV (`cleaned_dataset.csv`), and launch Streamlit.

## Manual steps (if you prefer)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows (PowerShell)
```pwsh
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Data
- Default run uses `cleaned_dataset.csv` in the repo root.
- You can also upload your own CSV inside the app (needs a `ts` timestamp column plus metric columns like PM2.5/PM10/CO2/etc.).

## Optional: AI insights
Set `OPENAI_API_KEY` in your shell or `.env` (not committed) to enable the AI summary button; otherwise the app falls back to rule-based tips.
