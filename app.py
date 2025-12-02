from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# --- Constants & Config ---
DATA_PATH = Path(__file__).parent / "IAQ Baqubah Teaching Hospital .csv"
DEFAULT_METRICS = ["PM2.5", "PM10", "CO2", "Temp", "Hum", "TVOC"]
FORECAST_HOURS_CHOICES = [1, 3, 6, 12]

st.set_page_config(page_title="V-IndoorCARE Prototype", layout="wide", page_icon="🫧")

# --- Theme Injection ---
def _inject_theme() -> None:
    """Minimal custom theming."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Manrope:wght@500;700&display=swap');
        html, body, [class*="stApp"] {font-family: 'Space Grotesk', 'Manrope', sans-serif; background: radial-gradient(120% 120% at 10% 20%, #0f172a 0%, #0b1021 40%, #050912 100%); color: #e5e7eb;}
        h1, h2, h3, h4 {font-family: 'Manrope', sans-serif;}
        .stMetric {background: linear-gradient(135deg, rgba(45,212,191,0.08), rgba(59,130,246,0.08)); border: 1px solid rgba(148,163,184,0.2); border-radius: 14px; padding: 10px 12px;}
        .stMarkdown, .stCaption {color: #d1d5db;}
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
    else:
        values = moving_average_forecast(df[metric], horizon, window=max(12, horizon // 2))
        note = "Moving-average regression (fast prototype)."

    return future_ts, values, note

# --- Insights ---
def generate_insights(current: Dict[str, float], forecast: List[float], metric: str, horizon_hours: int) -> str:
    # (Existing OpenAI logic kept as placeholder)
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

# --- Visualization Components ---

def metric_cards(snapshot: Dict[str, float], df: pd.DataFrame, cols: List[str]) -> None:
    """Displays metrics with delta indicators vs the last 1 hour average."""
    
    # Calculate 1-hour average for comparison
    if not df.empty:
        last_ts = df["ts"].max()
        one_hour_ago = last_ts - timedelta(hours=1)
        recent_df = df[df["ts"] >= one_hour_ago]
    else:
        recent_df = pd.DataFrame()

    card_cols = st.columns(len(cols))
    
    for col_ui, name in zip(card_cols, cols):
        value = snapshot.get(name, float("nan"))
        
        # Determine Delta
        delta_msg = None
        delta_color = "normal"
        
        if not recent_df.empty and name in recent_df.columns:
            avg_last_hour = recent_df[name].mean()
            diff = value - avg_last_hour
            delta_msg = f"{diff:+.1f} (1h trend)"
            
            # Logic: For pollutants, Positive delta = Bad (Inverse)
            if name in ["PM2.5", "PM10", "CO2", "TVOC"]:
                delta_color = "inverse" 
            # For Temp/Hum, depends on context, but let's keep normal for now
            else:
                delta_color = "normal"

        col_ui.metric(
            label=name, 
            value=f"{value:.1f}", 
            delta=delta_msg,
            delta_color=delta_color
        )

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
        title=f"IAQ Trends ({timeframe_label})",
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

# --- Main App ---

def main() -> None:
    _inject_theme()
    st.title("V-IndoorCARE Prototype")
    st.caption("Monitoring dashboard with dual-axis trends, smart alerts, and hybrid forecasting.")

    # Sidebar Configuration
    with st.sidebar:
        st.header("Data Source")
        uploaded_file = st.file_uploader("Upload CSV (Optional)", type=["csv"])
        
        st.divider()
        st.header("Controls")
        timeframe = st.radio("History window", ["Last 24h", "Last 7 days", "All data"], index=0)
        metric = st.selectbox("Forecast metric", DEFAULT_METRICS, index=0)
        horizon_hours = st.select_slider("Forecast span (hours)", options=FORECAST_HOURS_CHOICES, value=6)
        
        # Logic for prediction method
        method_options = ["Moving average"]
        if _bilstm_available and uploaded_file is None:
            # Only allow BiLSTM if using the default file (due to path dependency)
            method_options.append("BiLSTM (from notebook)")
            
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
        st.caption(f"Data cadence: ~{freq_minutes:.1f} min | {len(df):,} rows")

    # Top Metric Cards
    st.subheader("Current Status")
    metric_cards(snapshot, df, available_metrics)

    # Filtering for Charts
    if timeframe == "Last 24h":
        start_time = df["ts"].max() - timedelta(hours=24)
        filtered = df[df["ts"] >= start_time]
    elif timeframe == "Last 7 days":
        start_time = df["ts"].max() - timedelta(days=7)
        filtered = df[df["ts"] >= start_time]
    else:
        filtered = df

    # Trend Charts (Dual Axis)
    plot_dual_axis_trend(filtered, available_metrics, timeframe.lower())

    # Forecasting
    st.subheader(f"Forecast for {metric} (next {horizon_hours}h)")
    
    horizon_steps = max(1, int(np.ceil((horizon_hours * 60) / freq_minutes)))
    future_ts, predictions, note = compute_forecast(
        df=filtered, # Pass recent history for context
        metric=metric, 
        horizon=horizon_steps, 
        method=method, 
        freq=freq,
        is_custom_upload=is_custom_upload
    )
    
    forecast_df = pd.DataFrame({"ts": future_ts, "Predicted": predictions})

    # Side-by-side Forecast Charts
    hist_fig = px.line(
        filtered.tail(120),
        x="ts",
        y=metric,
        markers=True,
        title=f"{metric} (Recent History)",
        color_discrete_sequence=["#3b82f6"]
    )
    hist_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")

    pred_fig = px.line(forecast_df, x="ts", y="Predicted", markers=True, title="Forecasted Trajectory")
    pred_fig.update_traces(line_color="#10b981")
    # Add the last known point to connect the lines visually
    pred_fig.add_scatter(
        x=[filtered["ts"].iloc[-1]], 
        y=[filtered[metric].iloc[-1]], 
        mode="markers", 
        name="Now",
        marker=dict(color="#f43f5e", size=8)
    )
    pred_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")

    col1, col2 = st.columns(2)
    col1.plotly_chart(hist_fig, use_container_width=True)
    col2.plotly_chart(pred_fig, use_container_width=True)
    
    st.caption(f"ℹ️ {note}")

    # 8. Insights Section
    st.subheader("AI Analysis")
    if "insight_text" not in st.session_state:
        st.session_state["insight_text"] = ""
        
    col_btn, col_txt = st.columns([1, 4])
    with col_btn:
        if st.button("Generate Report", type="primary", use_container_width=True):
            st.session_state["insight_text"] = generate_insights(snapshot, predictions, metric, horizon_hours)
    
    with col_txt:
        if st.session_state["insight_text"]:
            st.info(st.session_state["insight_text"])
        else:
            st.markdown("*Click 'Generate Report' for an automated summary of air quality conditions.*")

    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(100), use_container_width=True)

if __name__ == "__main__":
    main()
