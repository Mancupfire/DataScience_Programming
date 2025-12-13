from __future__ import annotations

import json
import os
from dotenv import load_dotenv

load_dotenv()
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from plotly.subplots import make_subplots
import streamlit as st
try:
    from bilstm_adapter import run_bilstm_prediction

    _bilstm_available = True
    _bilstm_error = ""
except Exception as exc:
    run_bilstm_prediction = None
    _bilstm_available = False
    _bilstm_error = str(exc)

try:
    from lightgbm_adapter import load_lightgbm_model, recursive_forecast, create_temporal_features
    _lightgbm_available = True
    _lightgbm_error = ""
    # Load model once at startup if possible, or lazy load
    LGBM_MODEL_PATH = Path(__file__).parent / "Training Model" / "lightgbm_model.txt"
    lgbm_model = load_lightgbm_model(LGBM_MODEL_PATH) if LGBM_MODEL_PATH.exists() else None
except Exception as exc:
    _lightgbm_available = False
    _lightgbm_error = str(exc)
    lgbm_model = None

# --- Constants & Config ---
DATA_PATH = Path(__file__).parent / "Training Model" / "cleaned_dataset.csv"
DEFAULT_METRICS = ["PM2.5", "PM10", "CO2", "Temp", "Hum", "TVOC"]
FORECAST_HOURS_CHOICES = [1, 3, 6, 12]

st.set_page_config(page_title="Smart-IndoorCARE Prototype", layout="wide", page_icon="🫧")

# --- LLM Helpers ---
def get_deepseek_llm() -> Tuple[Optional[ChatDeepSeek], str]:
    """Instantiate DeepSeek chat model if credentials are present."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None, "Set DEEPSEEK_API_KEY in your environment to enable the assistant."

    try:
        llm = ChatDeepSeek(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            temperature=0.2,
        )
        return llm, ""
    except Exception as exc:  # pragma: no cover - defensive for runtime issues
        return None, str(exc)


def pick_one_hour_point(
    future_ts: List[pd.Timestamp],
    predictions: List[float],
    freq_minutes: float,
) -> Optional[Dict[str, object]]:
    """Pick the forecast point closest to +1 hour."""
    if not predictions:
        return None

    steps_for_hour = max(1, int(np.ceil(60 / max(freq_minutes, 1))))
    idx = min(len(predictions) - 1, steps_for_hour - 1)

    ts_value = None
    if future_ts:
        ts_value = future_ts[idx] if idx < len(future_ts) else future_ts[-1]

    return {"timestamp": ts_value, "value": float(predictions[idx])}


def format_forecast_series(
    future_ts: List[pd.Timestamp],
    predictions: List[float],
    limit: int = 5,
) -> List[Dict[str, object]]:
    """Trim and serialize forecast points for prompts."""
    points = []
    for ts, val in list(zip(future_ts, predictions))[:limit]:
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        points.append({"timestamp": ts_str, "value": float(val)})
    return points

def closest_projection_value(
    future_ts: List[pd.Timestamp],
    predictions: List[float],
    target_time: pd.Timestamp,
) -> Optional[float]:
    """Return the prediction closest to the requested timestamp."""
    if not future_ts or not predictions:
        return None

    limit = min(len(future_ts), len(predictions))
    if target_time > future_ts[limit - 1]:
        return None

    idx = min(
        range(limit),
        key=lambda i: abs((future_ts[i] - target_time).total_seconds()),
    )
    return float(predictions[idx])

def describe_change_direction(current_value: float, future_value: Optional[float]) -> str:
    """Translate a numeric delta into a professional, qualitative direction."""
    if future_value is None or pd.isna(future_value) or pd.isna(current_value):
        return "Insufficient data"

    delta = future_value - current_value
    pct_change = delta / max(abs(current_value), 1e-6)

    if pct_change >= 0.25:
        return "Sharp increase"
    if pct_change >= 0.08:
        return "Gradual increase"
    if pct_change <= -0.25:
        return "Sharp decrease"
    if pct_change <= -0.08:
        return "Gradual decrease"
    return "Holding steady"

# --- Theme Injection ---
def _inject_theme() -> None:
    """Minimal custom theming."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Manrope:wght@500;700&display=swap');
        html, body, [class*="stApp"] {
            font-family: 'Space Grotesk', 'Manrope', sans-serif;
            background: radial-gradient(120% 120% at 10% 20%, #0f172a 0%, #0b1021 40%, #050912 100%);
            color: #e5e7eb;
        }
        h1, h2, h3, h4 {font-family: 'Manrope', sans-serif; color: #e5e7eb;}
        .stMetric {
            background: linear-gradient(135deg, rgba(45,212,191,0.08), rgba(59,130,246,0.08));
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 14px;
            padding: 10px 12px;
            color: #e5e7eb;
        }
        .stMarkdown, .stCaption {color: #d1d5db;}
        a {color: #22d3ee;}
        
        /* Details Button Styling - Dark background by default */
        button[kind="secondary"] {
            background: rgba(15, 23, 42, 0.9) !important;
            border: 1px solid rgba(148,163,184,0.3) !important;
            color: #e5e7eb !important;
        }
        button[kind="secondary"]:hover {
            background: rgba(45,212,191,0.2) !important;
            border-color: #2dd4bf !important;
        }
        
        /* Dialog/Modal Dark Theme Styling */
        [data-testid="stModal"] > div {
            background: radial-gradient(120% 120% at 10% 20%, #0f172a 0%, #0b1021 40%, #050912 100%) !important;
            border: 1px solid rgba(148,163,184,0.3);
            border-radius: 16px;
        }
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] {
            color: #e5e7eb !important;
        }
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stModal"] [data-testid="stMarkdownContainer"] strong {
            color: #e5e7eb !important;
        }
        [data-testid="stModal"] h1, 
        [data-testid="stModal"] h2, 
        [data-testid="stModal"] h3 {
            color: #e5e7eb !important;
        }
        [data-testid="stModal"] p, 
        [data-testid="stModal"] span,
        [data-testid="stModal"] li {
            color: #e5e7eb !important;
        }
        [data-testid="stModal"] [data-testid="stMetricValue"] {
            color: #e5e7eb !important;
        }
        [data-testid="stModal"] [data-testid="stMetricLabel"] {
            color: #9ca3af !important;
        }
        /* Dialog tabs styling */
        [data-testid="stModal"] .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        [data-testid="stModal"] .stTabs [data-baseweb="tab"] {
            color: #9ca3af;
        }
        [data-testid="stModal"] .stTabs [aria-selected="true"] {
            color: #e5e7eb !important;
            background: rgba(45,212,191,0.1);
        }
        /* Dialog tab content text */
        [data-testid="stModal"] .stTabs [data-baseweb="tab-panel"] p,
        [data-testid="stModal"] .stTabs [data-baseweb="tab-panel"] span,
        [data-testid="stModal"] .stTabs [data-baseweb="tab-panel"] li {
            color: #e5e7eb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Data Loading ---
@st.cache_data
def load_data(source: Path | st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Loads data from a local path or an uploaded file object."""
    try:
        df = pd.read_csv(source)
        df.columns = [c.strip() for c in df.columns]
        
        # Normalize column names slightly for robustness
        col_map = {c: c for c in df.columns}
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower() or c.lower() == "ts":
                col_map[c] = "ts"
        df = df.rename(columns=col_map)

        if "ts" not in df.columns:
            st.error("CSV missing required 'ts' (timestamp) column.")
            return pd.DataFrame()

        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")
        
        # Coerce numerics
        numeric_cols = [c for c in df.columns if c != "ts"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        return df.dropna(subset=["ts"])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def infer_frequency(df: pd.DataFrame) -> timedelta:
    if len(df) < 2:
        return timedelta(minutes=5)
    diffs = df["ts"].diff().dropna()
    if len(diffs) == 0:
        return timedelta(minutes=5)
    return diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()

def current_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    return df.iloc[-1].to_dict()

# --- Prediction Logic ---
def moving_average_forecast(series: pd.Series, horizon: int = 6, window: int | None = None) -> List[float]:
    if series.empty:
        return []
    series = series.dropna()
    if series.empty:
        return [0.0] * horizon
        
    window = window or max(12, horizon)
    window = min(window, len(series))
    recent = series.tail(window)
    
    baseline = recent.mean()
    # Simple linear slope
    y = recent.values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
    
    return [float(baseline + slope * (i + 1)) for i in range(horizon)]

@st.cache_data(show_spinner="Running BiLSTM model...", ttl=3600)
def cached_bilstm_wrapper(metric: str, horizon: int) -> Optional[List[float]]:
    """Wraps the external BiLSTM call to prevent re-running on every redraw."""
    if _bilstm_available and run_bilstm_prediction:
        try:
            # Note: This assumes DATA_PATH is valid. BiLSTM is disabled for uploaded files below.
            return run_bilstm_prediction(DATA_PATH, variable=metric, horizon=horizon)
        except Exception:
            return None
    return None

def compute_forecast(
    df: pd.DataFrame, 
    metric: str, 
    horizon: int, 
    method: str, 
    freq: timedelta,
    is_custom_upload: bool
) -> Tuple[List[pd.Timestamp], List[float], str]:
    
    last_ts = df["ts"].iloc[-1]
    future_ts = [last_ts + freq * (i + 1) for i in range(horizon)]
    note = ""

    if method == "BiLSTM (from notebook)":
        if is_custom_upload:
            note = "BiLSTM disabled for custom uploads (requires specific file path). Using Moving Avg."
            values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
        elif not _bilstm_available:
            note = f"BiLSTM unavailable ({_bilstm_error}). Falling back to moving average."
            values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
        else:
            # Use cached wrapper
            bilstm_preds = cached_bilstm_wrapper(metric, horizon)
            if bilstm_preds:
                values = bilstm_preds
                note = "BiLSTM predictions (Spatial_BiLSTM.ipynb)."
            else:
                note = "BiLSTM failed or returned None. Using Moving Average."
                values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
    elif method == "LightGBM (Recursive)":
        if not _lightgbm_available or lgbm_model is None:
             note = f"LightGBM unavailable ({_lightgbm_error} or model not found). Falling back to Moving Avg."
             values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
        elif metric != "PM2.5":
             note = "LightGBM model is trained for PM2.5 only. Using Moving Avg for other metrics."
             values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
        else:
            try:
                # Prepare data for LightGBM
                # We need enough history for lags (max 24h)
                # Ensure we have the necessary columns
                # The adapter handles feature creation, but we need to pass a dataframe with 'ts' index or similar
                
                # Make sure df has datetime index for the adapter
                df_lgbm = df.copy()
                if 'ts' in df_lgbm.columns:
                    df_lgbm = df_lgbm.set_index('ts')
                
                # We need at least 24 hours of data
                if len(df_lgbm) < 24:
                     note = "Not enough data for LightGBM lags (need 24h). Using Moving Avg."
                     values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
                else:
                    # Run recursive forecast (returns predictions at 5-min intervals)
                    values = recursive_forecast(lgbm_model, df_lgbm, horizon, target_col=metric)
                    # Fix: Generate timestamps at 5-min intervals to match prediction count
                    steps_per_hour = 12  # 60 / 5 = 12 steps per hour
                    total_steps = horizon * steps_per_hour
                    future_ts = [last_ts + timedelta(minutes=5) * (i + 1) for i in range(total_steps)]
                    note = "LightGBM recursive forecast (lightgbm.ipynb)."
            except Exception as e:
                note = f"LightGBM failed: {e}. Using Moving Avg."
                values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))

    else:
        values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
        note = "Moving-average regression (fast prototype)."

    return future_ts, values, note

# --- Insights ---
def generate_insights(
    current: Dict[str, float],
    forecast: List[float],
    metric: str,
    horizon_hours: int,
    freq_minutes: float,
    future_ts: List[pd.Timestamp],
    llm: Optional[ChatDeepSeek] = None,
) -> str:
    """Generate a short summary of current conditions and +1 hour outlook."""
    one_hour_point = pick_one_hour_point(future_ts, forecast, freq_minutes)
    forecast_points = format_forecast_series(future_ts, forecast)

    if llm:
        try:
            system = SystemMessage(
                content="You are an indoor air-quality co-pilot. Be concise and avoid speculation."
            )
            human_prompt = (
                "Summarize indoor air conditions now and expected state 1 hour from now.\n"
                f"Current snapshot: {json.dumps(current, default=str)}\n"
                f"Primary metric focus: {metric}\n"
                f"One-hour forecast point: {json.dumps(one_hour_point, default=str)}\n"
                f"Short forecast trajectory (~{freq_minutes:.1f} min steps): {json.dumps(forecast_points)}\n"
                "Keep to 2-3 sentences and finish with one clear action tip if needed."
            )
            response = llm.invoke([system, HumanMessage(content=human_prompt)])
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            pass

    # (Existing OpenAI logic kept as fallback)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "You are an indoor air-quality expert. Summarize today's conditions and actionable tips.\n"
                f"Current snapshot: {current}\n"
                f"Forecast for {metric} over the next {horizon_hours} hours: {forecast}\n"
                "Keep it under 120 words. Use concise bullet-like sentences."
            )
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Expert indoor air-quality assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            pass

    # Rule-based fallback
    pm = current.get("PM2.5", 0) or 0
    co2 = current.get("CO2", 0) or 0
    temp = current.get("Temp", 0) or 0
    hum = current.get("Hum", 0) or 0

    tips = []
    if pm > 35:
        tips.append("PM2.5 elevated; increase ventilation.")
    elif pm < 15:
        tips.append("Air is clean (PM2.5 low).")

    if co2 > 1000:
        tips.append("CO2 high (>1000 ppm); open windows immediately.")

    if hum > 70:
        tips.append("High humidity; run dehumidifier.")
    elif hum < 30:
        tips.append("Air is dry; usage of humidifier recommended.")

    if not tips:
        tips.append("Conditions are within optimal ranges.")

    return " • ".join(tips)


def deepseek_chat_reply(
    llm: ChatDeepSeek,
    user_input: str,
    snapshot: Dict[str, float],
    metric: str,
    one_hour_point: Optional[Dict[str, object]],
    history: List[Dict[str, str]],
    freq_minutes: float,
) -> str:
    """Chat helper to answer follow-ups using the latest context."""
    if one_hour_point and one_hour_point.get("value") is not None:
        timestamp_txt = one_hour_point.get("timestamp")
        horizon_text = (
            f"{metric} at +1h: {float(one_hour_point['value']):.2f} "
            f"(target time {timestamp_txt})"
        )
    else:
        horizon_text = "No 1-hour forecast available"

    system_prompt = (
        "You are a concise indoor air-quality assistant. Keep answers under 120 words, avoid speculation, "
        "and always rely on the provided context."
        f"\nCurrent metrics: {json.dumps(snapshot, default=str)}"
        f"\n{horizon_text}"
        f"\nData cadence is ~{freq_minutes:.1f} minutes. If data is missing, say so briefly."
    )

    messages: List[SystemMessage | HumanMessage | AIMessage] = [SystemMessage(content=system_prompt)]
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)

# --- Visualization Components ---

def get_health_status(metric: str, value: float) -> Tuple[str, str, str]:
    """
    Returns (status_label, border_color, recommendation) based on WHO/EPA standards.
    Colors: Green, Yellow, Orange, Red, Purple, Maroon.
    """
    # Defaults
    status = "Good"
    color = "#10b981" # Green
    rec = "Air quality is satisfactory."

    if metric == "PM2.5":
        if value <= 12:
            status, color, rec = "Good", "#10b981", "Air quality is satisfactory."
        elif value <= 35.4:
            status, color, rec = "Moderate", "#eab308", "Sensitive individuals should limit outdoor exertion."
        elif value <= 55.4:
            status, color, rec = "Unhealthy for SG", "#f97316", "Sensitive groups: reduce outdoor exertion."
        elif value <= 150.4:
            status, color, rec = "Unhealthy", "#ef4444", "Everyone: reduce prolonged outdoor exertion."
        elif value <= 250.4:
            status, color, rec = "Very Unhealthy", "#a855f7", "Avoid outdoor exertion. Wear a mask."
        else:
            status, color, rec = "Hazardous", "#881337", "Emergency conditions. Stay indoors."
            
    elif metric == "PM10":
        if value <= 54:
            status, color, rec = "Good", "#10b981", "Air quality is satisfactory."
        elif value <= 154:
            status, color, rec = "Moderate", "#eab308", "Sensitive individuals should limit outdoor exertion."
        elif value <= 254:
            status, color, rec = "Unhealthy for SG", "#f97316", "Sensitive groups: reduce outdoor exertion."
        elif value <= 354:
            status, color, rec = "Unhealthy", "#ef4444", "Everyone: reduce prolonged outdoor exertion."
        elif value <= 424:
            status, color, rec = "Very Unhealthy", "#a855f7", "Avoid outdoor exertion."
        else:
            status, color, rec = "Hazardous", "#881337", "Emergency conditions."

    elif metric == "CO2":
        if value <= 1000:
            status, color, rec = "Good", "#10b981", "Air is fresh."
        elif value <= 1500:
            status, color, rec = "Fair", "#eab308", "Ventilation recommended."
        elif value <= 2000:
            status, color, rec = "Poor", "#f97316", "Open windows. Drowsiness likely."
        else:
            status, color, rec = "Bad", "#ef4444", "High CO2. Maximize ventilation immediately."

    elif metric == "TVOC":
        if value <= 300:
            status, color, rec = "Good", "#10b981", "Clean air."
        elif value <= 500:
            status, color, rec = "Moderate", "#eab308", "Acceptable."
        elif value <= 1000:
            status, color, rec = "Marginal", "#f97316", "Ventilate room."
        else:
            status, color, rec = "High", "#ef4444", "Identify sources (chemicals, etc)."
            
    elif metric == "Temp":
        if 18 <= value <= 26:
            status, color, rec = "Comfortable", "#10b981", "Optimal temperature."
        elif value < 18:
            status, color, rec = "Cool", "#3b82f6", "Maybe turn up heat."
        else:
            status, color, rec = "Warm", "#f97316", "Maybe turn on AC/Fan."

    elif metric == "Hum":
        if 30 <= value <= 60:
            status, color, rec = "Optimal", "#10b981", "Comfortable humidity."
        elif value < 30:
            status, color, rec = "Dry", "#eab308", "Use humidifier."
        else:
            status, color, rec = "Humid", "#eab308", "Use dehumidifier."

    return status, color, rec

INDEX_LEVELS = {
    "PM2.5": [
        {"label": "Good", "range": "0 - 12 µg/m³", "color": "#10b981", "note": "Indoor air is clean."},
        {"label": "Moderate", "range": "12.1 - 35.4 µg/m³", "color": "#eab308", "note": "Slight particles; keep light ventilation."},
        {"label": "Unhealthy for SG", "range": "35.5 - 55.4 µg/m³", "color": "#f97316", "note": "Sensitive occupants: ventilate and run purifier."},
        {"label": "Unhealthy", "range": "55.5 - 150.4 µg/m³", "color": "#ef4444", "note": "High particulates; run purifier and reduce indoor sources."},
        {"label": "Very Unhealthy", "range": "150.5 - 250.4 µg/m³", "color": "#a855f7", "note": "Very high; keep windows closed if outdoor is worse, purifier on high."},
        {"label": "Hazardous", "range": "> 250.4 µg/m³", "color": "#881337", "note": "Severe indoor pollution; ventilate with filtration and avoid the area."},
    ],
    "PM10": [
        {"label": "Good", "range": "0 - 54 µg/m³", "color": "#10b981", "note": "Indoor air is clean."},
        {"label": "Moderate", "range": "55 - 154 µg/m³", "color": "#eab308", "note": "Some dust; light ventilation/cleaning."},
        {"label": "Unhealthy for SG", "range": "155 - 254 µg/m³", "color": "#f97316", "note": "Sensitive occupants: ventilate, reduce dust sources."},
        {"label": "Unhealthy", "range": "255 - 354 µg/m³", "color": "#ef4444", "note": "High coarse particles; clean surfaces and filter air."},
        {"label": "Very Unhealthy", "range": "355 - 424 µg/m³", "color": "#a855f7", "note": "Very high; avoid stirring dust, run purifier on high."},
        {"label": "Hazardous", "range": "> 424 µg/m³", "color": "#881337", "note": "Severe dust; isolate area and ventilate with filtration."},
    ],
    "CO2": [
        {"label": "Good", "range": "≤ 1000 ppm", "color": "#10b981", "note": "Indoor air feels fresh; ventilation is adequate."},
        {"label": "Fair", "range": "1001 - 1500 ppm", "color": "#eab308", "note": "Air is getting stale; crack a window or start exhaust."},
        {"label": "Poor", "range": "1501 - 2000 ppm", "color": "#f97316", "note": "Drowsiness likely; ventilate now or reduce occupancy."},
        {"label": "Bad", "range": "> 2000 ppm", "color": "#ef4444", "note": "Ventilation insufficient; clear the room and maximize airflow."},
    ],
    "Temp": [
        {"label": "Cool", "range": "< 18°C", "color": "#3b82f6", "note": "Below indoor comfort; adjust heating."},
        {"label": "Comfortable", "range": "18 - 26°C", "color": "#10b981", "note": "Comfortable indoor temperature."},
        {"label": "Warm", "range": "> 26°C", "color": "#f97316", "note": "Warm indoors; increase airflow or cooling."},
    ],
    "Hum": [
        {"label": "Dry", "range": "< 30%", "color": "#eab308", "note": "Dry indoors; use a humidifier and hydrate."},
        {"label": "Optimal", "range": "30 - 60%", "color": "#10b981", "note": "Comfortable indoor humidity."},
        {"label": "Humid", "range": "> 60%", "color": "#eab308", "note": "Feels damp; dehumidify and improve airflow."},
    ],
    "TVOC": [
        {"label": "Good", "range": "≤ 300 ppb", "color": "#10b981", "note": "No noticeable indoor chemical build-up."},
        {"label": "Moderate", "range": "301 - 500 ppb", "color": "#eab308", "note": "Light indoor odors; keep airflow steady."},
        {"label": "Marginal", "range": "501 - 1000 ppb", "color": "#f97316", "note": "Noticeable VOCs; ventilate and limit sprays/solvents."},
        {"label": "High", "range": "> 1000 ppb", "color": "#ef4444", "note": "High VOCs; remove sources, ventilate, and use filtration/charcoal."},
    ],
}

UNIVERSAL_LEVELS = ["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"]
LEVEL_COLORS = {
    "Good": "#10b981",
    "Moderate": "#eab308",
    "Unhealthy": "#ef4444",
    "Very Unhealthy": "#a855f7",
    "Hazardous": "#881337",
}

def check_alerts(df: pd.DataFrame, snapshot: Dict[str, float]) -> List[str]:
    """Generates alerts for critical thresholds and sudden spikes."""
    alerts = []
    
    # 1. Threshold Alerts (Critical levels)
    # PM2.5 > 150 (Unhealthy)
    if snapshot.get("PM2.5", 0) > 150.4:
        alerts.append("⚠️ **Critical Alert:** PM2.5 exceeds 150 µg/m³ (Unhealthy).")
    
    # CO2 > 1500 (Poor Ventilation)
    if snapshot.get("CO2", 0) > 1500:
        alerts.append("⚠️ **Comfort Warning:** CO2 > 1500 ppm. Poor ventilation detected.")
        
    # TVOC > 1000 (High)
    if snapshot.get("TVOC", 0) > 1000:
        alerts.append("⚠️ **Chemical Alert:** TVOC levels are very high (>1000 ppb).")

    # 2. Sudden Spike Detection (e.g., >25% increase in last 10 mins)
    # We need at least 10 mins of data
    if len(df) > 2:
        last_ts = df["ts"].max()
        ten_mins_ago = last_ts - timedelta(minutes=10)
        # Get data closest to 10 mins ago
        # We look for a point within a small window around 10 mins ago
        past_window = df[(df["ts"] >= ten_mins_ago - timedelta(minutes=2)) & 
                         (df["ts"] <= ten_mins_ago + timedelta(minutes=2))]
        
        if not past_window.empty:
            # Compare current vs average of that past window
            # Check PM2.5 spike
            current_pm = snapshot.get("PM2.5", 0)
            past_pm = past_window["PM2.5"].mean()
            
            if past_pm > 5 and current_pm > past_pm * 1.25: # >25% increase and not noise (base > 5)
                alerts.append(f"📈 **Sudden Spike:** PM2.5 increased by >25% in the last 10 minutes.")
                
            # Check CO2 spike
            current_co2 = snapshot.get("CO2", 0)
            past_co2 = past_window["CO2"].mean()
             
            if past_co2 > 400 and current_co2 > past_co2 * 1.25:
                 alerts.append(f"📈 **Sudden Spike:** CO2 levels rising rapidly.")

    return alerts

def detect_events(df: pd.DataFrame, snapshot: Dict[str, float]) -> List[str]:
    """
    Detects events based on heuristics over the last 10 minutes.
    - Cooking: PM2.5 spike (>10%) + CO2 spike (>5%)
    - Crowding: CO2 spike (>10%) only
    - Window Opening: CO2 drop (>10%) + Humidity drop (>5%)
    - Cleaning: TVOC spike (>10%)
    """
    events = []
    if len(df) < 3:
        return events

    last_ts = df["ts"].max()
    ten_mins_ago = last_ts - timedelta(minutes=10)
    
    # Get baseline (approx 10 mins ago)
    # We use a small window around 10 mins ago to be robust
    past_window = df[(df["ts"] >= ten_mins_ago - timedelta(minutes=2)) & 
                     (df["ts"] <= ten_mins_ago + timedelta(minutes=2))]
    
    if past_window.empty:
        return events

    # Current values
    curr_pm = snapshot.get("PM2.5", 0)
    curr_co2 = snapshot.get("CO2", 0)
    curr_hum = snapshot.get("Hum", 0)
    curr_tvoc = snapshot.get("TVOC", 0)

    # Past averages
    past_pm = past_window["PM2.5"].mean()
    past_co2 = past_window["CO2"].mean()
    past_hum = past_window["Hum"].mean()
    past_tvoc = past_window["TVOC"].mean()

    # Thresholds for "significant" change (avoid division by zero)
    def pct_change(curr, past):
        return (curr - past) / past if past > 1 else 0

    pm_change = pct_change(curr_pm, past_pm)
    co2_change = pct_change(curr_co2, past_co2)
    hum_change = pct_change(curr_hum, past_hum)
    tvoc_change = pct_change(curr_tvoc, past_tvoc)

    # Heuristics
    # 1. Cooking: PM2.5 > 10% AND CO2 > 5%
    if pm_change > 0.10 and co2_change > 0.05:
        events.append("🍳 **Cooking Detected** (PM2.5 & CO2 rising)")
    
    # 2. Crowding: CO2 > 10% (and not cooking)
    elif co2_change > 0.10:
        events.append("👥 **Crowding / Occupancy Increase** (CO2 rising)")

    # 3. Window Opening: CO2 drop > 10% AND Hum drop > 5%
    if co2_change < -0.10 and hum_change < -0.05:
        events.append("🪟 **Window Likely Opened** (Fresh air influx)")

    # 4. Cleaning: TVOC > 10%
    if tvoc_change > 0.10:
        events.append("🧹 **Cleaning / Chemical Use** (TVOC spike)")

    return events

def metric_cards(snapshot: Dict[str, float], df: pd.DataFrame, cols: List[str]) -> None:
    """Displays metrics with delta indicators and health categories using clickable cards that open detail dialogs."""
    
    # Calculate 1-hour average for comparison
    if not df.empty:
        last_ts = df["ts"].max()
        one_hour_ago = last_ts - timedelta(hours=1)
        recent_df = df[df["ts"] >= one_hour_ago]
    else:
        recent_df = pd.DataFrame()

    # Detailed info for each metric
    metric_details = {
        "PM2.5": {
            "full_name": "Fine Particulate Matter (PM2.5)",
            "unit": "µg/m³",
            "description": "PM2.5 or fine particles are 2.5 micrometers in diameter or smaller. They can be made of different chemicals like black carbon, ammonia, sulphur dioxide, etc. They also include pollen, mold, fungal spores, soot, etc. As PM2.5 can be airborne allergens, including molds and fungi, dander, dust mites, and cockroach antigens and can induce an immune response, they are capable of causing illness at very low exposure levels.",
            "sources": "PM2.5 can be generated indoors as a result of cooking (frying and broiling especially), running wood stoves, furnaces, tobacco smoke, renovations and other activities. It also can have outdoor origins when polluted air from outside travels inside a building via ventilation, infiltration, or air conditioning.",
            "health_effects": "Short-term: eye/nose/throat irritation, coughing. Long-term: respiratory disease, heart disease, lung cancer.",
            "tips": ["Use air purifiers with HEPA filters", "Improve ventilation", "Avoid indoor smoking", "Keep windows closed during high outdoor pollution"],
        },
        "PM10": {
            "full_name": "Coarse Particulate Matter (PM10)",
            "unit": "µg/m³",
            "description": "PM10 refers to particulate matter with a diameter of 10 micrometers or smaller. These particles are larger than PM2.5 and include dust, pollen, mold spores, and fragments from construction or road traffic. Due to their size, PM10 particles can be inhaled into the nose and throat and may irritate the respiratory system, eyes, and skin. Long-term exposure can worsen asthma and other lung conditions.",
            "sources": "PM10 can be generated indoors from activities such as cleaning, vacuuming, moving furniture, or indoor dust resuspension. Outdoor sources include road dust, construction activities, industrial emissions, and wind-blown soil that can enter indoor spaces through doors, windows, and ventilation systems.",
            "health_effects": "Aggravates asthma, causes coughing and difficulty breathing. Less dangerous than PM2.5 but still harmful.",
            "tips": ["Regular dusting and vacuuming", "Use doormats to reduce tracked-in dust", "Change HVAC filters regularly"],
        },
        "CO2": {
            "full_name": "Carbon Dioxide",
            "unit": "ppm",
            "description": "Carbon dioxide, CO2, is an odorless non-toxic gas. People exhale CO2 when they breathe, so if there are several people in poorly ventilated rooms, it can build up and affect occupants' wellbeing. This is the reason why this parameter is important in terms of indoor air quality.In indoor spaces, safe standards for CO2 concentrations are 400–1000 ppm (parts per million).",
            "sources": "Human activities such as burning oil, gas and coal as well as cement manufacturing are sources of CO2 emissions. Another source is respiration and decay.",
            "health_effects": "1000-2000 ppm: drowsiness, poor concentration. >2000 ppm: headaches, sleepiness. >5000 ppm: serious health effects.",
            "tips": ["Open windows regularly", "Use mechanical ventilation", "Add plants (minor effect)", "Limit occupancy in small rooms"],
        },
        "TVOC": {
            "full_name": "Total Volatile Organic Compounds",
            "unit": "ppb",
            "description": "Total Volatile Organic Compounds (TVOCs) are a group of carbon-based chemicals that easily evaporate at room temperature. Many VOCs have noticeable odors and can cause short-term effects such as headaches, dizziness, eye irritation, and nausea. Long-term exposure to elevated TVOC levels may affect the liver, kidneys, or nervous system and can degrade overall indoor air quality.",
            "sources": "TVOCs are commonly released from paints, varnishes, cleaning products, air fresheners, adhesives, furniture, carpets, and building materials. Other sources include tobacco smoke, cooking, personal care products, and office equipment such as printers.",
            "health_effects": "Eye/nose/throat irritation, headaches, nausea. Some VOCs are carcinogenic with long-term exposure.",
            "tips": ["Choose low-VOC products", "Ventilate when using chemicals", "Let new furniture off-gas outdoors", "Avoid air fresheners"],
        },
        "Temp": {
            "full_name": "Temperature",
            "unit": "°C",
            "description": "Temperature represents the level of heat in an indoor environment and plays a critical role in comfort, productivity, and health. Temperatures that are too high or too low can cause discomfort, fatigue, reduced concentration, and stress on the body. Maintaining an appropriate indoor temperature supports thermal comfort and overall well-being.",
            "sources": "Indoor temperature is influenced by outdoor weather conditions, heating and cooling systems, insulation quality, sunlight exposure, and the number of occupants and electronic devices in a room.",
            "health_effects": "Too cold: increased blood pressure, respiratory issues. Too hot: heat stress, dehydration, reduced concentration.",
            "tips": ["Use programmable thermostats", "Ensure proper insulation", "Use fans for air circulation", "Dress appropriately for conditions"],
        },
        "Hum": {
            "full_name": "Relative Humidity",
            "unit": "%",
            "description": "Humidity refers to the amount of water vapor present in the air. Proper indoor humidity levels help maintain comfort and reduce health risks. Low humidity can cause dry skin, eye irritation, and respiratory discomfort, while high humidity can promote mold growth, dust mites, and bacterial growth, negatively affecting indoor air quality. Recommended indoor relative humidity levels are typically 30–60%.",
            "sources": "Indoor humidity levels are affected by activities such as cooking, showering, drying clothes indoors, and breathing. Poor ventilation, leaks, and outdoor weather conditions can also significantly impact humidity levels.",
            "health_effects": "Too dry (<30%): dry skin, irritated airways, increased virus transmission. Too humid (>60%): mold growth, dust mites, discomfort.",
            "tips": ["Use humidifier in dry conditions", "Use dehumidifier/AC when humid", "Ensure bathroom ventilation", "Fix any water leaks"],
        },
    }

    # Create rows of 3 columns
    for i in range(0, len(cols), 3):
        batch_cols = cols[i:i+3]
        ui_cols = st.columns(len(batch_cols))
        
        for col_ui, name in zip(ui_cols, batch_cols):
            value = snapshot.get(name, float("nan"))
            status, color, rec = get_health_status(name, value)
            
            # Determine Delta
            delta_text = ""
            if not recent_df.empty and name in recent_df.columns:
                avg_last_hour = recent_df[name].mean()
                diff = value - avg_last_hour
                arrow = "↑" if diff > 0 else "↓"
                delta_text = f"{arrow} {abs(diff):.1f} (1h)"
            
            with col_ui:
                # Create a container for the card
                with st.container():
                    # Display the card info
                    st.markdown(
                        f'<div style="border: 2px solid {color}; border-radius: 12px; padding: 16px; background: rgba(255,255,255,0.03); margin-bottom: 8px;">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
                        f'<span style="color: #9ca3af; font-weight: 600; font-size: 0.9em;">{name}</span>'
                        f'<span style="background: {color}20; color: {color}; padding: 2px 8px; border-radius: 99px; font-size: 0.75em; font-weight: 700; border: 1px solid {color}40;">{status}</span>'
                        f'</div>'
                        f'<div style="font-size: 2em; font-weight: 700; margin-bottom: 4px;">{value:.1f}</div>'
                        f'<div style="color: #9ca3af; margin-bottom: 8px;">{delta_text}</div>'
                        f'<div style="font-size: 0.85em; color: #d1d5db; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">{rec}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Button to open detail dialog
                    if st.button(f"📊 Details", key=f"detail_{name}", use_container_width=True):
                        show_metric_detail_dialog(name, value, status, color, rec, delta_text, df, metric_details.get(name, {}))


@st.dialog("Metric Details", width="large")
def show_metric_detail_dialog(name: str, value: float, status: str, color: str, rec: str, delta_text: str, df: pd.DataFrame, details: dict):
    """Modal dialog showing detailed information about a metric."""
    
    # Header with current status - use dark text for readability on colored background
    st.markdown(
        f'<div style="text-align: center; padding: 20px; background: {color}20; border-radius: 12px; border: 2px solid {color}; margin-bottom: 20px;">'
        f'<div style="font-size: 0.9em; color: #4b5563; margin-bottom: 4px;">{details.get("full_name", name)}</div>'
        f'<div style="font-size: 3em; font-weight: 700; color: #1f2937;">{value:.1f} <span style="font-size: 0.4em; color: #6b7280;">{details.get("unit", "")}</span></div>'
        f'<div style="margin-top: 8px;">'
        f'<span style="background: {color}; color: {"#0f172a" if status == "Good" else "#ffffff"}; padding: 4px 16px; border-radius: 99px; font-weight: 700;">{status}</span>'
        f'</div>'
        f'<div style="color: #4b5563; margin-top: 8px;">{delta_text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Tabs for different info sections
    tab1, tab2, tab3 = st.tabs(["📈 Trend", "ℹ️ About", "💡 Tips"])
    
    with tab1:
        # Show recent trend for this metric
        if not df.empty and name in df.columns:
            last_24h = df[df["ts"] >= df["ts"].max() - timedelta(hours=24)]
            if not last_24h.empty:
                fig = px.line(
                    last_24h, x="ts", y=name,
                    title=f"{name} - Last 24 Hours",
                    line_shape="spline"
                )
                fig.update_traces(line_color=color)
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#374151")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current", f"{value:.1f}")
                col2.metric("24h Avg", f"{last_24h[name].mean():.1f}")
                col3.metric("24h Min", f"{last_24h[name].min():.1f}")
                col4.metric("24h Max", f"{last_24h[name].max():.1f}")
    
    with tab2:
        if details:
            st.markdown(f'<p style="color: #1f2937; font-weight: 600;">What is {name}?</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #374151;">{details.get("description", "No description available.")}</p>', unsafe_allow_html=True)
            
            st.markdown('<p style="color: #1f2937; font-weight: 600;">Common Sources:</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #374151;">{details.get("sources", "Various indoor and outdoor sources.")}</p>', unsafe_allow_html=True)
            
            st.markdown('<p style="color: #1f2937; font-weight: 600;">Health Effects:</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #374151;">{details.get("health_effects", "Varies by concentration and exposure time.")}</p>', unsafe_allow_html=True)
        else:
            st.info("Detailed information not available for this metric.")
    
    with tab3:
        if details and "tips" in details:
            st.markdown('<p style="color: #1f2937; font-weight: 600;">Recommendations:</p>', unsafe_allow_html=True)
            for tip in details["tips"]:
                st.markdown(f'<p style="color: #374151;">• {tip}</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f'<p style="color: #1f2937;"><strong>Current Recommendation:</strong> {rec}</p>', unsafe_allow_html=True)



def plot_dual_axis_trend(df: pd.DataFrame, metrics: List[str], timeframe_label: str) -> None:
    """Plots trends with CO2 on a secondary Y-axis to fix scaling issues."""
    
    # Downsample for performance if dataset is huge (>2000 points)
    if len(df) > 2000:
        step = len(df) // 2000
        chart_df = df.iloc[::step, :]
    else:
        chart_df = df

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = px.colors.qualitative.Plotly
    
    for i, col in enumerate(metrics):
        if col not in chart_df.columns:
            continue
            
        # Put CO2 on secondary axis
        is_secondary = (col == "CO2")
        
        fig.add_trace(
            go.Scatter(
                x=chart_df["ts"], 
                y=chart_df[col], 
                name=col,
                mode='lines',
                line=dict(width=2, color=colors[i % len(colors)])
            ),
            secondary_y=is_secondary,
        )

    fig.update_layout(
        height=380,
        margin=dict(t=50, l=20, r=20, b=20),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d1d5db")
    )
    
    fig.update_yaxes(title_text="Standard Units", secondary_y=False, showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="CO2 (ppm)", secondary_y=True, showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_spider_chart(snapshot: Dict[str, float], metrics: List[str]) -> None:
    """Plots a spider/radar chart of current metrics normalized to a 0-100 scale based on typical ranges."""
    if not snapshot:
        return

    # Define typical max values for normalization (approximate "high" levels)
    # This ensures the chart shape is meaningful even with different units
    limits = {
        "PM2.5": 100,   # ug/m3
        "PM10": 150,    # ug/m3
        "CO2": 2000,    # ppm
        "Temp": 40,     # C
        "Hum": 100,     # %
        "TVOC": 500,    # ppb (assumed)
    }

    r_values = []
    theta_values = []
    
    for m in metrics:
        if m in snapshot:
            val = snapshot[m]
            limit = limits.get(m, max(val * 1.2, 1.0)) # Fallback to 1.2x value if unknown
            normalized = min((val / limit) * 100, 100) # Cap at 100%
            r_values.append(normalized)
            theta_values.append(m)
            
    # Close the loop
    if r_values:
        r_values.append(r_values[0])
        theta_values.append(theta_values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_values,
        theta=theta_values,
        fill='toself',
        name='Current Status',
        line_color='#2dd4bf'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            )
        ),
        showlegend=False,
        title="Current Status Fingerprint",
        height=350,
        margin=dict(t=40, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d1d5db")
    )

    return fig, r_values, theta_values, limits


def calculate_overall_iaq_status(snapshot: Dict[str, float]) -> Tuple[str, str, int]:
    """
    Calculate an overall IAQ status based on the worst metric.
    Returns (status_label, color, score 0-100).
    """
    status_priority = {
        "Hazardous": (0, "#881337"),
        "Very Unhealthy": (1, "#a855f7"),
        "Unhealthy": (2, "#ef4444"),
        "Unhealthy for SG": (2, "#f97316"),
        "Poor": (2, "#f97316"),
        "Bad": (2, "#ef4444"),
        "High": (2, "#ef4444"),
        "Marginal": (3, "#f97316"),
        "Moderate": (4, "#eab308"),
        "Fair": (4, "#eab308"),
        "Warm": (4, "#f97316"),
        "Cool": (4, "#3b82f6"),
        "Dry": (4, "#eab308"),
        "Humid": (4, "#eab308"),
        "Good": (5, "#10b981"),
        "Optimal": (5, "#10b981"),
        "Comfortable": (5, "#10b981"),
    }
    
    worst_priority = 5
    worst_status = "Good"
    worst_color = "#10b981"
    
    for metric_name, value in snapshot.items():
        if metric_name == "ts" or pd.isna(value):
            continue
        status, color, _ = get_health_status(metric_name, value)
        priority, _ = status_priority.get(status, (5, "#10b981"))
        if priority < worst_priority:
            worst_priority = priority
            worst_status = status
            worst_color = color
    
    # Convert priority to score (0-100)
    score = int((worst_priority / 5) * 100)
    
    # Normalize status for display
    if worst_status in ["Optimal", "Comfortable"]:
        worst_status = "Good"
    elif worst_status in ["Fair", "Warm", "Cool", "Dry", "Humid"]:
        worst_status = "Moderate"
    elif worst_status in ["Poor", "Bad", "High", "Marginal", "Unhealthy for SG"]:
        worst_status = "Unhealthy"
    
    return worst_status, worst_color, score


def plot_radial_dashboard(snapshot: Dict[str, float], metrics: List[str]) -> go.Figure:
    """
    Creates a circular radial dashboard with:
    - Central status indicator showing overall IAQ status
    - Metrics positioned radially around the center with color-coded dots
    - Concentric ring background for visual depth
    """
    overall_status, status_color, score = calculate_overall_iaq_status(snapshot)
    
    # Create figure
    fig = go.Figure()
    
    # Add concentric background rings (decorative)
    for i, radius in enumerate([0.95, 0.75, 0.55]):
        opacity = 0.08 - (i * 0.02)
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-radius, y0=-radius, x1=radius, y1=radius,
            line=dict(color="rgba(148, 163, 184, 0.3)", width=1),
            fillcolor=f"rgba(148, 163, 184, {opacity})",
            layer="below"
        )
    
    # Central status circle
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-0.35, y0=-0.35, x1=0.35, y1=0.35,
        line=dict(color=status_color, width=3),
        fillcolor=status_color,
        layer="below"
    )
    
    # Central status text
    fig.add_annotation(
        x=0, y=0,
        text=f"<b>{overall_status}</b>",
        showarrow=False,
        font=dict(size=20, color="#0f172a" if overall_status == "Good" else "#ffffff"),
    )
    
    # Position metrics radially around the center
    n_metrics = len(metrics)
    available_metrics = [m for m in metrics if m in snapshot and not pd.isna(snapshot.get(m))]
    n_available = len(available_metrics)
    
    if n_available > 0:
        angle_step = 360 / n_available
        radius = 0.7  # Distance from center
        
        for i, metric_name in enumerate(available_metrics):
            value = snapshot.get(metric_name, 0)
            status, color, _ = get_health_status(metric_name, value)
            
            # Calculate position (start from top, go clockwise)
            angle_deg = -90 + (i * angle_step)  # -90 to start from top
            angle_rad = np.radians(angle_deg)
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            
            # Add colored dot for the metric
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=16, color=color, line=dict(color='white', width=2)),
                text=[""],
                hoverinfo='text',
                hovertext=f"{metric_name}: {value:.1f} ({status})",
                showlegend=False
            ))
            
            # Add metric label and value
            # Position text slightly further out
            label_radius = 0.92
            label_x = label_radius * np.cos(angle_rad)
            label_y = label_radius * np.sin(angle_rad)
            
            # Unit suffixes
            units = {
                "PM2.5": "µg/m³",
                "PM10": "µg/m³", 
                "CO2": "ppm",
                "Temp": "°C",
                "Hum": "%",
                "TVOC": "ppb"
            }
            unit = units.get(metric_name, "")
            
            fig.add_annotation(
                x=label_x, y=label_y,
                text=f"<b>{value:.0f}</b><span style='font-size:10px'>{unit}</span><br><span style='color:#9ca3af;font-size:11px'>{metric_name}</span>",
                showarrow=False,
                font=dict(size=14, color="#e5e7eb"),
                align="center"
            )
    
    # Layout
    fig.update_layout(
        xaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True
        ),
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(
            text="Indoor Air Quality Overview",
            font=dict(color="#e5e7eb", size=16),
            x=0.5
        )
    )
    
    return fig


def render_index_level_cards() -> None:
    """
    Display compact colored cards for each index level with pollutant ranges.
    Matches the reference design with 5 levels in a grid layout.
    """
    # Define ranges for each level and metric (based on existing INDEX_LEVELS thresholds)
    level_ranges = {
        "Good": {
            "color": "#10b981",
            "text_color": "#0f172a",
            "PM2.5": "0 - 12",
            "PM10": "0 - 54",
            "CO2": "≤ 1000",
            "TVOC": "≤ 300",
            "Temp": "18 - 26",
            "Hum": "30 - 60",
        },
        "Moderate": {
            "color": "#eab308",
            "text_color": "#0f172a",
            "PM2.5": "12.1 - 35.4",
            "PM10": "55 - 154",
            "CO2": "1001 - 1500",
            "TVOC": "301 - 500",
            "Temp": "< 18 or > 26",
            "Hum": "< 30 or > 60",
        },
        "Unhealthy": {
            "color": "#f97316",
            "text_color": "#ffffff",
            "PM2.5": "35.5 - 150.4",
            "PM10": "155 - 354",
            "CO2": "1501 - 2000",
            "TVOC": "501 - 1000",
            "Temp": "—",
            "Hum": "—",
        },
        "Very Unhealthy": {
            "color": "#a855f7",
            "text_color": "#ffffff",
            "PM2.5": "150.5 - 250.4",
            "PM10": "355 - 424",
            "CO2": "> 2000",
            "TVOC": "> 1000",
            "Temp": "—",
            "Hum": "—",
        },
        "Hazardous": {
            "color": "#881337",
            "text_color": "#ffffff",
            "PM2.5": "> 250.4",
            "PM10": "> 424",
            "CO2": "—",
            "TVOC": "—",
            "Temp": "—",
            "Hum": "—",
        },
    }
    
    # Build HTML for cards in a grid
    cards_html = ""
    for level_name, level_data in level_ranges.items():
        bg_color = level_data["color"]
        text_color = level_data["text_color"]
        
        # Build metric rows for this card
        metrics_html = ""
        for metric in ["PM2.5", "PM10", "CO2", "TVOC", "Temp", "Hum"]:
            range_val = level_data.get(metric, "—")
            if range_val != "—":
                metrics_html += (
                    f'<div style="display:flex;justify-content:space-between;padding:3px 0;font-size:0.85em;">'
                    f'<span style="color:{text_color};opacity:0.85;">{metric}</span>'
                    f'<span style="color:{text_color};font-weight:600;">{range_val}</span>'
                    f'</div>'
                )
        
        cards_html += (
            f'<div style="background:{bg_color};border-radius:12px;padding:12px 16px;min-width:140px;flex:1;">'
            f'<div style="font-weight:700;font-size:1.1em;color:{text_color};margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.2);padding-bottom:6px;">'
            f'{level_name}'
            f'</div>'
            f'{metrics_html}'
            f'</div>'
        )
    
    # Wrap in a flex container
    full_html = (
        '<div style="margin-top:8px;margin-bottom:24px;">'
        '<div style="color:#9ca3af;font-size:0.85em;margin-bottom:8px;">'
        '<b>Pollutant</b> | <span>Index level (concentration ranges)</span>'
        '</div>'
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
        f'{cards_html}'
        '</div>'
        '</div>'
    )
    
    st.markdown(full_html, unsafe_allow_html=True)


# --- Main App ---

def main() -> None:
    _inject_theme()
    st.title("Smart-IndoorCARE Prototype")
    st.caption("Monitoring dashboard with dual-axis trends, smart alerts, and hybrid forecasting.")

    # Sidebar Configuration
    with st.sidebar:
        st.header("Data Source")
        uploaded_file = st.file_uploader("Upload CSV (Optional)", type=["csv"])
        
        st.divider()
        st.header("Controls")
        timeframe = st.radio("History window", ["Last 24h", "Last 7 days"], index=0)
        metric = st.selectbox("Forecast metric", DEFAULT_METRICS, index=0)
        horizon_hours = st.select_slider("Forecast span (hours)", options=FORECAST_HOURS_CHOICES, value=6)
        
        # Logic for prediction method
        method_options = ["Moving average"]
        if _bilstm_available and uploaded_file is None:
            # Only allow BiLSTM if using the default file (due to path dependency)
            method_options.append("BiLSTM (from notebook)")
        
        if _lightgbm_available and lgbm_model is not None:
            method_options.append("LightGBM (Recursive)")
            
        method = st.radio(
            "Prediction model",
            method_options,
            index=0,
            help="BiLSTM available only for default dataset."
        )

    # Data Loading Strategy
    if uploaded_file:
        df = load_data(uploaded_file)
        is_custom_upload = True
    elif DATA_PATH.exists():
        df = load_data(DATA_PATH)
        is_custom_upload = False
    else:
        st.warning("No data found. Please upload a CSV file.")
        return

    if df.empty:
        return

    # Data Prep
    freq = infer_frequency(df)
    freq_minutes = max(freq.total_seconds() / 60, 1)
    snapshot = current_snapshot(df)
    available_metrics = [m for m in DEFAULT_METRICS if m in df.columns]

    with st.sidebar:
        st.header("IAQ Trends")
        trend_metrics = st.multiselect(
            "Metrics to plot",
            options=available_metrics,
            default=available_metrics[:2] if len(available_metrics) >= 2 else available_metrics,
            help="Select only the signals you want to view in the trend chart.",
        )
        st.caption(f"Data cadence: ~{freq_minutes:.1f} min | {len(df):,} rows")

    if not trend_metrics:
        trend_metrics = [metric] if metric in available_metrics else available_metrics[:1]

    # Alerts & Notifications
    alerts = check_alerts(df, snapshot)
    if alerts:
        for alert in alerts:
            st.error(alert, icon="🚨")

    # Event Detection
    events = detect_events(df, snapshot)
    if events:
        with st.container():
            st.markdown("### 🧠 AI Event Detection")
            for event in events:
                st.info(event, icon="ℹ️")

    # Top Metric Cards
    st.subheader("Current Status")
    
    # Radial Dashboard and Index Level Cards side by side
    col_radial, col_index = st.columns(2)
    with col_radial:
        radial_fig = plot_radial_dashboard(snapshot, available_metrics)
        st.plotly_chart(radial_fig, use_container_width=True)
    with col_index:
        render_index_level_cards()
    
    metric_cards(snapshot, df, available_metrics)

    # Filtering for Charts based on selected timeframe
    if timeframe == "Last 24h":
        start_time = df["ts"].max() - timedelta(hours=24)
        filtered = df[df["ts"] >= start_time]
    else:  # Last 7 days
        start_time = df["ts"].max() - timedelta(days=7)
        filtered = df[df["ts"] >= start_time]

    # Trend Charts (Dual Axis)
    st.subheader(f"IAQ Trends ({timeframe.lower()})")
    plot_dual_axis_trend(filtered, trend_metrics, timeframe.lower())

    # Forecasting
    st.subheader(f"Forecast for {metric} (next {horizon_hours}h)")
    
    horizon_steps = max(1, int(np.ceil((horizon_hours * 60) / freq_minutes)))
    current_ts = filtered["ts"].iloc[-1]
    future_ts, predictions, note = compute_forecast(
        df=filtered, # Pass recent history for context
        metric=metric, 
        horizon=horizon_steps, 
        method=method, 
        freq=freq,
        is_custom_upload=is_custom_upload
    )
    
    forecast_df = pd.DataFrame({"ts": future_ts, "Predicted": predictions})
    
    # Limit forecast to exactly the requested horizon
    horizon_end_time = current_ts + timedelta(hours=horizon_hours)
    forecast_df = forecast_df[forecast_df["ts"] <= horizon_end_time]

    # Side-by-side Forecast Charts - use filtered data matching selected timeframe
    hist_fig = px.line(
        filtered,
        x="ts",
        y=metric,
        title=f"{metric} ({timeframe})",
        color_discrete_sequence=["#3b82f6"],
        line_shape="spline",
    )
    hist_fig.update_traces(mode="lines")
    hist_fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d1d5db"),
    )

    pred_fig = px.line(forecast_df, x="ts", y="Predicted", title="Forecasted Trajectory", line_shape="spline")
    pred_fig.update_traces(line_color="#10b981", mode="lines")
    # Add the last known point to connect the lines visually
    pred_fig.add_scatter(
        x=[current_ts], 
        y=[filtered[metric].iloc[-1]], 
        mode="markers", 
        name="Now",
        marker=dict(color="#f43f5e", size=8)
    )
    pred_fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d1d5db"),
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(hist_fig, use_container_width=True)
    col2.plotly_chart(pred_fig, use_container_width=True)

    st.markdown(
        """
        **What this outlook shows:** Directional cues (Sharp/Gradual increase or decrease, or Holding steady) forecast how the selected metric is expected to move. Values in parentheses are the projected readings at each time mark so you can decide whether to ventilate, filter, or wait.
        """,
        help="Uses the current snapshot and the chosen forecasting method to summarize expected movement at near-term checkpoints.",
    )

    # Directional summary for key horizons
    outlook_targets = [
        (30, "Next 30 minutes"),
        (60, "Next 1 hour"),
        (180, "Next 3 hours"),
    ]
    current_value = snapshot.get(metric, float("nan"))
    outlook_rows = []

    for minutes, label in outlook_targets:
        target_time = current_ts + timedelta(minutes=minutes)
        projected_val = closest_projection_value(future_ts, predictions, target_time)
        direction = describe_change_direction(current_value, projected_val)
        value_text = f"{projected_val:.2f}" if projected_val is not None else "N/A"
        outlook_rows.append((label, direction, value_text))

    outlook_html = "<div style='background:#ecfdf3; border:1px solid #bbf7d0; border-radius:12px; padding:12px 16px; margin-top:8px;'>"
    outlook_html += "<div style='font-weight:700; color:#065f46; margin-bottom:8px;'>Trajectory outlook</div>"
    for label, direction, value_text in outlook_rows:
        outlook_html += (
            f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-top:1px solid #d1fae5;'>"
            f"<span style='color:#065f46;'>{label}</span>"
            f"<span style='font-weight:600; color:#0f172a;'>{direction} "
            f"<span style='color:#047857; font-weight:500;'>({value_text})</span></span>"
            f"</div>"
        )
    outlook_html += "</div>"
    st.markdown(outlook_html, unsafe_allow_html=True)

    st.caption(f"ℹ️ {note}")

    one_hour_point = pick_one_hour_point(future_ts, predictions, freq_minutes)
    llm, llm_error = get_deepseek_llm()

    # 8. Insights Section
    st.subheader("AI Analysis (DeepSeek)")
    if "insight_text" not in st.session_state:
        st.session_state["insight_text"] = ""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if llm_error:
        st.caption(f"⚠️ {llm_error}")

    col_btn, col_txt = st.columns([1, 4])
    with col_btn:
        if st.button("Summarize now + next 1h", type="primary", use_container_width=True):
            st.session_state["insight_text"] = generate_insights(
                current=snapshot,
                forecast=predictions,
                metric=metric,
                horizon_hours=horizon_hours,
                freq_minutes=freq_minutes,
                future_ts=future_ts,
                llm=llm,
            )
    
    with col_txt:
        if st.session_state["insight_text"]:
            st.info(st.session_state["insight_text"])
        else:
            st.markdown("*Click 'Summarize now + next 1h' for an automated summary of air quality conditions.*")

    st.markdown("---")
    st.subheader("Chat with your IAQ co-pilot")

    if st.session_state["chat_history"]:
        for turn in st.session_state["chat_history"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

    user_question = st.chat_input("Ask about the current air quality or the next hour forecast")
    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        if llm:
            try:
                answer = deepseek_chat_reply(
                    llm=llm,
                    user_input=user_question,
                    snapshot=snapshot,
                    metric=metric,
                    one_hour_point=one_hour_point,
                    history=st.session_state["chat_history"],
                    freq_minutes=freq_minutes,
                )
            except Exception as exc:
                answer = f"DeepSeek chat failed: {exc}"
        else:
            answer = "DeepSeek not configured. Add DEEPSEEK_API_KEY to .env to enable chat."

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(100), use_container_width=True)

if __name__ == "__main__":
    main()
