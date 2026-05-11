# ✈️ SkyRisk — AI-Powered Flight Risk Prediction System

> **A Big Data project built for DATA 603 at UMBC.**  
> SkyRisk analyzes 10 years of U.S. domestic flight data (2015–2024) across 20 major airports to predict flight delays, assess travel risk, forecast fare trends, and deliver AI-powered booking recommendations — all through an interactive Streamlit web application.

---

## 📌 What This Project Does

When a user enters a flight (origin → destination, airline, travel month), SkyRisk:

1. **Predicts delay probability** using an XGBoost model trained on 30 million flights
2. **Scores travel risk** using a weighted formula combining delay, weather, congestion, and airline reliability
3. **Forecasts fare trends** and advises whether to book now or wait
4. **Recommends alternatives** — better airlines, airports, or travel times for the same route
5. **Explains everything** through an AI chat interface powered by NVIDIA LLaMA

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  BTS On-Time Performance  ·  NOAA ISD Weather  ·  US DOT Fares │
│         (2015–2024, 20 airports, ~48.5M raw flight rows)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     DATA PIPELINE  (Apache Spark)               │
│                                                                  │
│  Phase 2: BTS download & filter → data/raw/bts/                 │
│  Phase 3: NOAA weather download → data/raw/weather/             │
│  Phase 4: DuckDB join (BTS + weather) → bts_weather_joined.parquet │
│  Phase 5: Feature engineering (40+ features) → features_final.parquet │
│  Phase 6: XGBoost model training → models/xgboost.json          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    EXPORTED CSV DATASETS                         │
│                                                                  │
│  delay_predictions.csv      · route × airline × month delay prob │
│  airline_reliability.csv    · per-carrier reliability score      │
│  airport_congestion.csv     · per-airport congestion index       │
│  airport_weather_severity.csv · weather severity by airport×month│
│  route_fare_predictions.csv · fare trend forecasts per route     │
│  time_recommendation.csv    · best/worst hours, days, months     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    STREAMLIT APPLICATION                         │
│                                                                  │
│  app.py          ← UI, tabs, session state, rendering            │
│  data_loader.py  ← Loads & normalizes all 6 CSVs                │
│  risk_engine.py  ← Travel Risk Score computation + alternatives  │
│  llm_handler.py  ← NVIDIA LLaMA chat + screenshot OCR           │
│  style_loader.py ← Adaptive light/dark theme injection           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Data Pipeline — Phase by Phase

| Phase | Script | What it does | Output |
|---|---|---|---|
| **2** | `phase2_bts_download.py` | Downloads 120 monthly BTS zip files (2015–2024), filters to 20 airports in-memory | `data/raw/bts/bts_filtered_YYYY_MM.csv` × 120 |
| **3** | `phase3_weather_download.py` | Downloads NOAA ISD hourly weather for 20 airport stations | `data/raw/weather/weather_AIRPORT_YYYY.csv` × 200 |
| **4** | `phase4_eda_duckdb.ipynb` | SQL joins BTS + weather via DuckDB (streams, no RAM spike), exports Parquet | `data/processed/bts_weather_joined.parquet` |
| **5** | `phase5_rebuild.py` | Engineers 40+ features in 2M-row chunks: time, weather severity, COVID flags, airport stats | `data/processed/features_final.parquet` |
| **6** | `phase6_model_training.py` | Trains 3 models on full 30M rows; exports CSVs for app | `models/xgboost.json`, `models/random_forest.pkl`, CSV exports |

> **Why DuckDB instead of Spark for EDA?** Spark requires cluster infrastructure to run reliably. On a 16GB Windows machine the JVM alone consumes 4GB before touching data. DuckDB streams CSV/Parquet in-process with near-zero overhead, completing the full 48.5M-row join in under 5 minutes. Spark is used in the pipeline architecture and is fully compatible with the Parquet outputs.

---

## 🤖 Machine Learning Models

### Models Trained
| Model | Method | Training Data | AUC-ROC |
|---|---|---|---|
| SGD Logistic Regression | `partial_fit` incremental | All 30M rows | 0.6591 |
| Random Forest Ensemble | Mini-forests merged | All 30M rows | 0.6831 |
| **XGBoost** ✅ Best | DMatrix, 500 rounds | All 30M rows | **0.6890** |

### Train / Test Split
- **Train:** 2015–2021 (~22M flights) — temporal split, no data leakage
- **Test:** 2022–2024 (~8M flights)

### Features (40+)
| Group | Examples |
|---|---|
| Time | `dep_hour`, `day_of_week`, `month`, `season`, `is_holiday_period` |
| Weather | `is_low_visibility`, `max_wind_ms`, `weather_severity_score`, `is_freezing` |
| Airport | `is_hub`, `airport_hist_delay_rate`, `congestion_index` |
| Carrier | `carrier_hist_delay_rate`, `route_total_flights` |
| Anomaly | `is_covid_year`, `flight_volume_index` |

### Top Features by XGBoost Gain
1. `is_low_visibility` — dominant predictor (7,700+ gain)
2. `is_covid_year` — validates 2020 anomaly flag
3. `dep_hour` — departure time is highly predictive
4. `visibility_category` — IFR/VFR flight rules
5. `season` — summer and winter peaks confirmed

### Note on AUC
AUC of 0.689 is consistent with published benchmarks for BTS + historical weather datasets (0.70–0.82 range). The ceiling is a **data limitation, not a modeling limitation** — real-time ATC feeds, inbound aircraft tail history, and gate-level congestion data would push this to 0.85+. This is documented as future work.

---

## 🎯 Travel Risk Score Formula

```
risk_score = 0.40 × delay_probability
           + 0.20 × weather_severity
           + 0.20 × congestion_score
           + 0.20 × (1 − reliability_score)
```

| Score Range | Risk Level | Recommendation |
|---|---|---|
| < 0.30 | 🟢 Low | Go Ahead |
| 0.30 – 0.60 | 🟡 Medium | Book with Caution |
| ≥ 0.60 | 🔴 High | Consider Alternatives |

---

## 🖥️ Application Features

### Tab 1 — Flight Analysis
- Select origin, destination, airline, and travel month
- Upload a flight screenshot — AI reads it automatically via OCR
- Displays risk score, delay probability, fare forecast
- Shows 3 recommendation cards: Best Option, Cheapest, Most Reliable
- Lists alternative airlines ranked by risk score

### Tab 2 — AI Chat
- Ask anything: delays, best times to fly, airline reliability, fare trends
- Maintains flight context from Tab 1 for personalized answers
- Upload screenshots mid-conversation for instant analysis
- Powered by NVIDIA LLaMA via NVIDIA NIM API

### Tab 3 — Explore Data
- Airline Reliability bar chart (color-coded green → red)
- Airport Congestion rankings
- Weather Severity heatmap (airport × month)
- Time Recommendations by hour, day, month, and season

---

## 🗃️ Dataset Overview

| Dataset | Source | Size | Period |
|---|---|---|---|
| BTS On-Time Performance | Bureau of Transportation Statistics | ~48.5M rows | 2015–2024 |
| NOAA Hourly Weather | NOAA Integrated Surface Database | ~1.7M hourly obs | 2015–2024 |
| Fare Data | US DOT Air Fare Data | Per-route quarterly | 2015–2024 |

**20 Airports Covered:**
ATL · LAX · ORD · DFW · DEN · JFK · SFO · CLT · LAS · PHX · MIA · IAH · SEA · EWR · BOS · SLC · SAN · TPA · PDX · AUS

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Data Pipeline** | Apache Spark (PySpark), DuckDB, PyArrow, Pandas |
| **Storage** | Parquet (ZSTD compressed), CSV |
| **ML Models** | XGBoost, scikit-learn (Random Forest, SGD), joblib |
| **AI / LLM** | NVIDIA NIM API — LLaMA 3.1 Vision + Chat |
| **Web App** | Streamlit, Plotly |
| **Styling** | Custom CSS — adaptive light/dark theme |
| **Languages** | Python 3.10 |

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/nikhilpatil1104/SkyRisk.git
cd SkyRisk
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your NVIDIA API key
In `llm_handler.py`:
```python
NVIDIA_API_KEY = "nvapi-YOUR_KEY_HERE"
```
Get your free key at: https://integrate.api.nvidia.com

### 4. Place CSV data files
```
SkyRisk/
├── app.py
├── data_loader.py
├── risk_engine.py
├── llm_handler.py
├── style_loader.py
├── style.css
├── requirements.txt
└── data/
    ├── delay_predictions.csv
    ├── airline_reliability.csv
    ├── airport_congestion.csv
    ├── airport_weather_severity.csv
    ├── route_fare_predictions.csv
    └── time_recommendation.csv
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Full Project Structure

```
SkyRisk/
├── app.py                          ← Main Streamlit application
├── data_loader.py                  ← CSV loader and normalizer
├── risk_engine.py                  ← Risk score engine + alternatives
├── llm_handler.py                  ← NVIDIA LLaMA integration
├── style_loader.py                 ← Theme injection
├── style.css                       ← Adaptive light/dark theme CSS
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── bts/                    ← 120 filtered BTS monthly CSVs
│   │   └── weather/                ← 200 NOAA hourly weather CSVs
│   └── processed/
│       ├── bts_weather_joined.parquet
│       └── features_final.parquet
│
├── models/
│   ├── xgboost.json                ← Best model (AUC 0.689)
│   ├── random_forest.pkl
│   ├── sgd_logistic.pkl
│   ├── encoders.pkl
│   ├── imputer.pkl
│   ├── scaler.pkl
│   └── model_results.csv
│
├── phase2_bts_download.py          ← BTS data pipeline
├── phase3_weather_download.py      ← NOAA weather pipeline
├── phase4_eda_duckdb.ipynb         ← EDA and data joining
├── phase5_rebuild.py               ← Feature engineering
├── phase6_model_training.py        ← Model training
│
└── data/notebooks/
    ├── airline_reliability.py       ← Reliability score computation
    ├── time_recommendation.py       ← Time analysis
    └── phase6_export_delay_predictions.py
```

---

## 👥 Team

**SkyRisk Analytics — DATA 603, University of Maryland Baltimore County**

| Name | Role |
|---|---|
| Nikhil Patil | Flight Delay Prediction Model · Data Pipeline · ML Engineering |
| Sarika Thunipura | Fare Trend Prediction · Data Analysis |
| Suhani Shah | Weather Data Analysis · Visualizations |
| Madhurima Mukhopadhyay | Travel Risk Score Engine · Recommendation System |

---

## 📄 License

This project was developed for academic purposes as part of the DATA 603 Big Data Technologies course at UMBC.
