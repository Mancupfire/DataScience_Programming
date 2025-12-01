from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from bilstm_adapter import run_bilstm_prediction

    _bilstm_available = True
    _bilstm_error = ""
except Exception as exc:  # BilSTM remains optional
    run_bilstm_prediction = None
    _bilstm_available = False
    _bilstm_error = str(exc)


DATA_PATH = Path(__file__).parent / "IAQ Baqubah Teaching Hospital .csv"
DEFAULT_METRICS = ["PM2.5", "PM10", "CO2", "Temp", "Hum", "TVOC"]
FORECAST_HOURS_CHOICES = [1, 3, 6, 12]

st.set_page_config(page_title="V-IndoorCARE Prototype", layout="wide", page_icon="🫧")


def _inject_theme() -> None:
    """Minimal custom theming to avoid the default Streamlit look."""
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


@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "ts" not in df.columns:
        raise ValueError("CSV missing required 'ts' column.")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    numeric_cols = [c for c in df.columns if c != "ts"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts"])
    return df


def infer_frequency(df: pd.DataFrame) -> timedelta:
    if len(df) < 2:
        return timedelta(minutes=5)
    diffs = df["ts"].diff().dropna()
    if len(diffs) == 0:
        return timedelta(minutes=5)
    return diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()


def current_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    return df.iloc[-1].to_dict()


def moving_average_forecast(series: pd.Series, horizon: int = 6, window: int | None = None) -> List[float]:
    if series.empty:
        return []
    window = window or max(12, horizon)
    window = min(window, len(series))
    recent = series.tail(window)
    baseline = recent.mean()
    slope = np.polyfit(range(window), recent, 1)[0] if window > 1 else 0
    return [float(baseline + slope * (i + 1)) for i in range(horizon)]


def compute_forecast(
    df: pd.DataFrame, metric: str, horizon: int, method: str, freq: timedelta
) -> Tuple[List[pd.Timestamp], List[float], str]:
    last_ts = df["ts"].iloc[-1]
    future_ts = [last_ts + freq * (i + 1) for i in range(horizon)]
    note = ""

    if method == "BiLSTM (from notebook)":
        if not _bilstm_available or run_bilstm_prediction is None:
            note = f"BiLSTM unavailable ({_bilstm_error}). Falling back to moving average."
            values = moving_average_forecast(df[metric].dropna(), horizon, window=max(12, horizon // 2))
        else:
            try:
                values = run_bilstm_prediction(DATA_PATH, variable=metric, horizon=horizon)
                note = "BiLSTM predictions from Spatial_BiLSTM.ipynb (epochs trimmed for speed)."
            except Exception as exc:
                note = f"BiLSTM error: {exc}. Falling back to moving average."
                values = moving_average_forecast(df[metric].dropna(), horizon, window=max(12, horizon // 2))
    else:
        values = moving_average_forecast(df[metric].dropna(), horizon, window=max(12, horizon // 2))
        note = "Moving-average regression (fast prototype)."

    return future_ts, values, note


def generate_insights(current: Dict[str, float], forecast: List[float], metric: str, horizon_hours: int) -> str:
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

    pm = current.get("PM2.5", 0) or 0
    co2 = current.get("CO2", 0) or 0
    temp = current.get("Temp", 0) or 0
    hum = current.get("Hum", 0) or 0
    forecast_peek = forecast[0] if forecast else pm

    tips = []
    if pm > 35 or forecast_peek > 35:
        tips.append("PM2.5 elevated; increase ventilation and consider a portable purifier.")
    else:
        tips.append("PM2.5 is low; maintain current ventilation.")
    if co2 > 900:
        tips.append("CO2 above 900 ppm; open windows or reduce occupancy.")
    if hum > 70:
        tips.append("Humidity high; run dehumidifier to prevent mold.")
    elif hum < 30:
        tips.append("Humidity low; add moisture to avoid dryness.")
    if 22 <= temp <= 26:
        tips.append("Temperature is comfortable.")

    return " • ".join(tips)


def metric_cards(snapshot: Dict[str, float], cols: List[str]) -> None:
    card_cols = st.columns(len(cols))
    for col, name in zip(card_cols, cols):
        value = snapshot.get(name, float("nan"))
        col.metric(label=name, value=f"{value:.2f}")


def main() -> None:
    _inject_theme()
    st.title("V-IndoorCARE Prototype")
    st.caption("Quick IAQ dashboard with cards, trends, forecast, and optional BiLSTM reuse from the notebook.")

    if not DATA_PATH.exists():
        st.error(f"CSV not found at {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    freq = infer_frequency(df)
    freq_minutes = max(freq.total_seconds() / 60, 1)
    snapshot = current_snapshot(df)
    available_metrics = [m for m in DEFAULT_METRICS if m in df.columns]

    with st.sidebar:
        st.header("Controls")
        st.caption(f"Data cadence: ~{freq_minutes:.1f} min | {len(df):,} rows")
        timeframe = st.radio("History window", ["Last 24h", "Last 7 days", "All data"], index=0)
        metric = st.selectbox("Forecast metric", available_metrics, index=0)
        horizon_hours = st.select_slider("Forecast span (hours)", options=FORECAST_HOURS_CHOICES, value=6)
        horizon_steps = max(1, int(np.ceil((horizon_hours * 60) / freq_minutes)))
        method = st.radio(
            "Prediction model",
            ["Moving average", "BiLSTM (from notebook)"],
            index=0,
            help="BiLSTM uses the architecture from Spatial_BiLSTM.ipynb (requires TensorFlow).",
        )
        st.caption("BiLSTM path trains on the fly; keep smaller horizons if you want it to stay snappy on CPU.")

    st.subheader("Current snapshot")
    metric_cards(snapshot, available_metrics)

    if timeframe == "Last 24h":
        start_time = df["ts"].max() - timedelta(hours=24)
        filtered = df[df["ts"] >= start_time]
    elif timeframe == "Last 7 days":
        start_time = df["ts"].max() - timedelta(days=7)
        filtered = df[df["ts"] >= start_time]
    else:
        filtered = df

    st.subheader("Trend charts")
    fig = px.line(
        filtered,
        x="ts",
        y=[col for col in available_metrics if col in filtered.columns],
        markers=True,
        title=f"IAQ trends ({timeframe.lower()})",
    )
    fig.update_layout(height=360, legend_orientation="h", margin=dict(t=60, l=20, r=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Forecast for {metric} (next {horizon_hours}h)")
    future_ts, predictions, note = compute_forecast(filtered, metric, horizon_steps, method, freq)
    forecast_df = pd.DataFrame({"ts": future_ts, "Predicted": predictions})

    hist_fig = px.line(
        filtered.tail(120),
        x="ts",
        y=metric,
        markers=True,
        title=f"{metric} recent history",
    )
    pred_fig = px.line(forecast_df, x="ts", y="Predicted", markers=True, title="Forecasted trajectory")
    pred_fig.add_scatter(x=[filtered["ts"].iloc[-1]], y=[filtered[metric].iloc[-1]], mode="markers", name="Now")
    col1, col2 = st.columns(2)
    col1.plotly_chart(hist_fig, use_container_width=True)
    col2.plotly_chart(pred_fig, use_container_width=True)
    st.caption(f"{note} Horizon spans {horizon_steps} steps at ~{freq_minutes:.1f} min cadence.")

    st.subheader("LLM insights & recommendations")
    if "insight_text" not in st.session_state:
        st.session_state["insight_text"] = ""
    if st.button("Explain today's air quality / Give recommendations", type="primary"):
        st.session_state["insight_text"] = generate_insights(snapshot, predictions, metric, horizon_hours)
    if st.session_state["insight_text"]:
        st.write(st.session_state["insight_text"])
    else:
        st.info("Click the button to summarize conditions. If OPENAI_API_KEY is unset, a rule-based summary is used.")

    st.subheader("Data preview")
    st.dataframe(df.tail(20))

    st.caption("Prototype: metrics + charts + forecast + insights. Extend with device management and alerts as needed.")


if __name__ == "__main__":
    main()
