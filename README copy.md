# V-IndoorCARE Prototype

Streamlit dashboard that reads `IAQ Baqubah Teaching Hospital .csv`, shows current indoor air-quality metrics, renders historical charts, provides a simple forecast, and reuses the BiLSTM logic from `Spatial_BiLSTM.ipynb` as an optional predictor. It also includes an LLM insights panel behind an explicit "explain / recommend" button (OpenAI, optional).

## Quick start
1. Create a virtual env and install deps:
   ```bash
   pip install -r requirements.txt
   # Optional for BiLSTM: pip install tensorflow==2.15.0
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open the URL Streamlit prints (usually http://localhost:8501).

## Features
- **Dashboard cards:** PM2.5, PM10, CO2, Temp, Humidity, TVOC using the latest row from the CSV.
- **Trend charts:** Choose last 24h, last 7 days, or all data.
- **Forecast:** Moving-average predictor (fast) or BiLSTM lifted from the notebook (needs TensorFlow). Horizon is set in hours (e.g., 1h/6h) and respects the CSV cadence.
- **LLM insights:** Button-driven “Explain today’s air quality / Give recommendations” via OpenAI. If `OPENAI_API_KEY` is missing, a rule-based summary is shown.
- **Data preview:** Last 20 rows for quick inspection.

## Notes on the BiLSTM path
- The code in `bilstm_adapter.py` mirrors the helper functions from `Spatial_BiLSTM.ipynb` but trims epochs/windows for speed.
- TensorFlow/Keras is imported lazily; if unavailable, the app falls back to the moving-average predictor.
- The adapter expects the CSV columns `ts` and the selected metric (e.g., `PM2.5`). Large horizons will increase training time; keep the selector modest if running on CPU.

## Extending
- Swap in real sensor feeds or an API endpoint instead of the CSV.
- Attach alerting (email/SMS) when PM2.5 or CO2 cross thresholds.
- Add more charts (7-day heatmap, device-level comparisons) or export endpoints for a mobile client.
