````markdown
# 🫧 Indoor Air Quality Dashboard

**V-IndoorCARE** is an intelligent dashboard for monitoring, visualizing, and forecasting Indoor Air Quality (IAQ).

This prototype leverages hybrid machine learning models (**LightGBM**, **BiLSTM**) to predict air quality trends (PM2.5, CO2, etc.) and integrates a Generative AI assistant (**DeepSeek/OpenAI**) to provide real-time, actionable health recommendations based on current environmental conditions.

---

## ✨ Features

* **Real-time Monitoring:** Visualizes key metrics: PM2.5, PM10, CO2, TVOC, Temperature, and Humidity.
* **Smart Alerts:** Detects critical thresholds and sudden spikes (e.g., Cooking events, Crowding).
* **Hybrid Forecasting:**
    * **LightGBM (Recursive):** Trained specifically for PM2.5 short-term forecasting.
    * **BiLSTM:** (Optional) Deep learning integration for complex temporal patterns.
    * **Moving Average:** Robust fallback for general trend lines.
* **AI Co-Pilot:** Integrated chat interface (DeepSeek) to summarize conditions and answer user queries about air quality.
* **Dual-Axis Visualization:** Compare CO2 levels against other metrics effectively.

---

## 📂 Project Structure

⚠️ **Important:** Ensure the file structure matches the tree below. The application relies on the specific folder name `Training Model` to load the dataset and models.

```text
V-IndoorCARE/
├── app.py                     # Main Streamlit application
├── lightgbm_adapter.py        # Forecasting logic for LightGBM
├── bilstm_adapter.py          # (Optional) Forecasting logic for BiLSTM
├── requirements.txt           # Python dependencies
├── .env                       # API Keys configuration (You create this)
├── scripts/
│   └── run.sh                 # Auto-setup script for Mac/Linux
└── Training Model/            # ⚠️ DO NOT RENAME THIS FOLDER
    ├── cleaned_dataset.csv    # Default historical data
    └── lightgbm_model.txt     # Pre-trained LightGBM model
````

-----

## 🚀 Quick Start

### Option A: Automated Script (macOS / Linux)

We provide a helper script that creates the environment and launches the app.

1.  **Grant permission** (first time only):
    ```bash
    chmod +x scripts/run.sh
    ```
2.  **Run the app:**
    ```bash
    ./scripts/run.sh
    ```

### Option B: Manual Installation (Windows / Mac / Linux)

**1. Create a Virtual Environment**

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Note: LightGBM is required for the advanced forecasting model
pip install lightgbm
```

> **Mac Users:** If `pip install lightgbm` fails, install the OpenMP library first: `brew install libomp`

**3. Run the Application**

```bash
streamlit run app.py
```

-----

## ⚙️ Configuration (.env)

To enable the **AI Assistant** features (Chat & Summaries), creates a file named `.env` in the root directory.

```ini
# .env file content

# Required for the AI Chat Assistant
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
DEEPSEEK_MODEL=deepseek-chat

# Optional: For legacy fallback summary features
OPENAI_API_KEY=sk-your-openai-key-here
```

*Note: If no keys are provided, the app will run in "Offline Mode" using rule-based algorithms for alerts.*

-----

## 📊 How to Use

1.  **Data Source:**

      * **Default:** The app automatically loads the dataset found in `Training Model/cleaned_dataset.csv`.
      * **Upload:** You can upload your own CSV via the sidebar. It must contain a timestamp column (e.g., `ts`, `Date`, `Time`) and metric columns.

2.  **Forecasting Controls:**

      * **Metric:** Select the target variable (e.g., PM2.5).
      * **Horizon:** Choose how far to predict (1 to 12 hours).
      * **Model:**
          * *LightGBM (Recursive):* Best for PM2.5. Requires `lightgbm_model.txt`.
          * *Moving Average:* Available for all metrics (fallback).

3.  **Interpreting the Spider Chart:**

      * **Small Shape:** Good air quality (values near center).
      * **Large Shape:** High pollution levels (values near outer edge).
      * The chart normalizes all metrics (e.g., CO2 ppm and Temp C) to a 0-100% scale based on health limits.

-----

## ⚠️ Troubleshooting

**Q: "FileNotFoundError: .../Training Model/cleaned\_dataset.csv"**
**A:** The application looks for a folder specifically named `Training Model` (with a space). Ensure this folder exists and contains your CSV file.

**Q: "LightGBM unavailable"**
**A:** This means the `lightgbm` library is not installed or the model file is missing.

1.  Run `pip install lightgbm`.
2.  Ensure `lightgbm_model.txt` is inside the `Training Model` folder.

**Q: BiLSTM is disabled**
**A:** The BiLSTM model is computationally heavy and path-dependent. It is disabled by default for custom uploaded files to prevent errors. It works only with the default dataset structure.

-----

## 📜 License

This project is for research and prototyping purposes.

```
```
