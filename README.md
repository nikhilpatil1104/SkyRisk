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
5. **Searches the web in real time** via Tavily when the user asks about live prices, current schedules, or future dates
6. **Reads booking screenshots** via GPT-4o Vision — upload any screenshot from Google Flights, Kayak, Expedia, etc. and the app extracts the flight details automatically
7. **Explains everything** through Sky, an AI chat advisor powered by GPT-4o

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
│                  DATA PIPELINE  (Apache Spark + DuckDB)          │
│                                                                  │
│  Phase 2: BTS download & filter    → data/raw/bts/              │
│  Phase 3: NOAA weather download    → data/raw/weather/          │
│  Phase 4: DuckDB join (BTS+weather)→ bts_weather_joined.parquet │
│  Phase 5: Feature engineering      → features_final.parquet     │
│  Phase 6: XGBoost model training   → models/xgboost.json        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    EXPORTED CSV DATASETS                         │
│                                                                  │
│  delay_predictions.csv        · route × airline × month probs   │
│  airline_reliability.csv      · per-carrier reliability score   │
│  airport_congestion.csv       · per-airport congestion index    │
│  airport_weather_severity.csv · weather severity by airport×month│
│  route_fare_predictions.csv   · fare trend forecasts per route  │
│  time_recommendation.csv      · best/worst hours, days, months  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    STREAMLIT APPLICATION                         │
│                                                                  │
│  app.py          ← UI, 3 tabs, session state, Plotly charts     │
│  data_loader.py  ← Loads & normalizes all 6 CSVs               │
│  risk_engine.py  ← Travel Risk Score + alternatives engine      │
│  llm_handler.py  ← GPT-4o Vision + GPT-4o chat + Tavily search │
│  style_loader.py ← Adaptive light/dark theme injection          │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         GPT-4o         GPT-4o-mini     Tavily API
     (screenshot OCR,  (text extraction, (real-time web
      chat responses)   flight parsing)   search for live
                                         prices & schedules)
```

---

## 🤖 AI Layer — How llm_handler.py Works

This is the intelligence core of SkyRisk. It uses three external services:

### 1. GPT-4o Vision — Screenshot OCR
When a user uploads a booking screenshot (Google Flights, Kayak, Expedia, MakeMyTrip etc.), GPT-4o reads it and extracts:
- Origin and destination as IATA codes (e.g. "Chicago" → `ORD`)
- Airline as IATA code (e.g. "Delta" → `DL`)
- Month of travel

This extracted data is then passed directly to the risk engine for analysis — no manual input needed.

### 2. Tavily — Real-Time Web Search
Tavily is integrated to solve a key limitation of historical-only data. When a user's chat message contains live-data signals like:

```
"price", "fare", "how much", "today", "right now", "current",
"2025", "2026", "flight number", "schedule", "weather forecast",
"visa", "airline news", "strike", "cancel"...
```

The app automatically triggers `_tavily_search()` which:
1. Builds a focused search query (enriched with route context if available)
2. Calls Tavily's API with `search_depth="basic"` and up to 4 results
3. Gets back Tavily's own AI summary + source snippets
4. Injects the live results into GPT-4o's system prompt

This means Sky can answer "What's the cheapest flight from ATL to LAX next Tuesday?" with actual current prices instead of refusing or giving outdated data. **Tavily is what makes Sky never say "I don't have real-time data."**

### 3. GPT-4o / GPT-4o-mini — Chat Advisor "Sky"
The main chat model is GPT-4o with GPT-4o-mini as fallback. The system prompt:
- Injects SkyRisk's historical flight context (risk score, delay %, weather, congestion, reliability, fare trend)
- Injects Tavily's live web results when relevant
- Enforces Sky's personality — warm, direct, never refuses, explains numbers in plain English
- Generates structured JSON for recommendation cards (verdict, best option, cheapest, most reliable, fare advice, timing tip, risk factors)

---

## 🔬 Data Pipeline — Phase by Phase

| Phase | Script | What it does | Output |
|---|---|---|---|
| **2** | `phase2_bts_download.py` | Downloads 120 monthly BTS zip files, filters to 20 airports in-memory | `data/raw/bts/bts_filtered_YYYY_MM.csv` × 120 |
| **3** | `phase3_weather_download.py` | Downloads NOAA ISD hourly weather for 20 airport stations | `data/raw/weather/weather_AIRPORT_YYYY.csv` × 200 |
| **4** | `phase4_eda_duckdb.ipynb` | DuckDB joins BTS + weather, exports Parquet | `data/processed/bts_weather_joined.parquet` |
| **5** | `phase5_rebuild.py` | Engineers 40+ features in 2M-row chunks | `data/processed/features_final.parquet` |
| **6** | `phase6_model_training.py` | Trains 3 models on full 30M rows, exports CSVs | `models/xgboost.json` + CSV exports |

> **Why DuckDB instead of Spark for EDA?** Spark requires cluster infrastructure to run reliably. On a 16GB Windows machine the JVM alone consumes 4GB before touching data. DuckDB streams CSV/Parquet in-process with near-zero overhead, completing the full 48.5M-row join in under 5 minutes. Spark is used throughout the pipeline and is fully compatible with the Parquet outputs produced.

---

## 🤖 Machine Learning Models

| Model | Method | Training Data | AUC-ROC |
|---|---|---|---|
| SGD Logistic Regression | `partial_fit` incremental | All 30M rows | 0.6591 |
| Random Forest Ensemble | Mini-forests merged per chunk | All 30M rows | 0.6831 |
| **XGBoost** ✅ Best | DMatrix, 500 boosting rounds | All 30M rows | **0.6890** |

**Train / Test Split:** Train 2015–2021 · Test 2022–2024 (temporal split — no data leakage)

**Top 5 XGBoost Features by Gain:**
1. `is_low_visibility` — dominant predictor (7,700+ gain)
2. `is_covid_year` — validates 2020 anomaly flag
3. `dep_hour` — departure time is highly predictive
4. `visibility_category` — IFR/VFR flight rule category
5. `season` — summer and winter peaks confirmed

> **Note on AUC:** 0.689 is consistent with published benchmarks for BTS + historical weather (0.70–0.82 range). The ceiling is a data limitation — real-time ATC feeds and inbound aircraft tail history would push this to 0.85+. Documented as future work.

---

## 🎯 Travel Risk Score Formula

```
risk_score = 0.40 × delay_probability
           + 0.20 × weather_severity
           + 0.20 × congestion_score
           + 0.20 × (1 − reliability_score)
```

| Score | Risk Level | Verdict |
|---|---|---|
| < 0.30 | 🟢 Low | Go Ahead |
| 0.30 – 0.60 | 🟡 Medium | Book with Caution |
| ≥ 0.60 | 🔴 High | Consider Alternatives |

---

## 🖥️ Application Features

### Tab 1 — Flight Analysis
- Select origin, destination, airline, travel month — or upload a screenshot
- GPT-4o Vision reads screenshots from any booking site automatically
- Risk score gauge + breakdown chart (Plotly)
- 5 metric cards: Risk Score, Delay Probability, Weather, Congestion, Reliability
- Fare metrics: Current fare, Predicted fare, Booking advice
- AI recommendation cards: Best Option, Cheapest, Most Reliable
- Alternative airlines ranked by risk score on the same route

### Tab 2 — AI Chat (Sky)
- Conversational advisor powered by GPT-4o
- **Tavily live search** fires automatically when you ask about prices, schedules, or future dates
- Maintains full flight context from Tab 1 for personalized answers
- Upload screenshots mid-conversation for instant analysis
- Never refuses — always gives a real answer

### Tab 3 — Explore Data
- Airline Reliability bar chart (color-coded green → red)
- Airport Congestion rankings
- Weather Severity heatmap (airport × month)
- Time Recommendations tables by hour, day, month, and season

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Data Pipeline** | Apache Spark (PySpark), DuckDB, PyArrow, Pandas |
| **Storage** | Parquet (ZSTD compressed), CSV |
| **ML Models** | XGBoost, scikit-learn (Random Forest, SGD), joblib |
| **LLM — Chat & Recommendations** | OpenAI GPT-4o |
| **LLM — Screenshot OCR** | OpenAI GPT-4o Vision |
| **LLM — Text Extraction** | OpenAI GPT-4o-mini |
| **Real-Time Web Search** | Tavily API (live prices, schedules, travel news) |
| **Web App** | Streamlit |
| **Charts** | Plotly (gauge, bar, heatmap) |
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

### 3. Set environment variables
```bash
# Required
export OPENAI_API_KEY="sk-YOUR_OPENAI_KEY"

# Optional — enables live web search in AI Chat
export TAVILY_API_KEY="tvly-YOUR_TAVILY_KEY"
```

Get your keys:
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://app.tavily.com (free tier available)

> The app runs without Tavily — Sky will still answer from historical data. With Tavily, Sky can fetch live prices and current schedules.

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
├── app.py                              ← Main Streamlit application
├── data_loader.py                      ← CSV loader and normalizer
├── risk_engine.py                      ← Risk score engine + alternatives
├── llm_handler.py                      ← GPT-4o + Tavily integration
├── style_loader.py                     ← Theme injection
├── style.css                           ← Adaptive light/dark theme
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── bts/                        ← 120 filtered BTS monthly CSVs
│   │   └── weather/                    ← 200 NOAA hourly weather CSVs
│   └── processed/
│       ├── bts_weather_joined.parquet
│       └── features_final.parquet
│
├── models/
│   ├── xgboost.json                    ← Best model (AUC 0.689)
│   ├── random_forest.pkl
│   ├── sgd_logistic.pkl
│   ├── encoders.pkl
│   ├── imputer.pkl
│   ├── scaler.pkl
│   └── model_results.csv
│
├── phase2_bts_download.py
├── phase3_weather_download.py
├── phase4_eda_duckdb.ipynb
├── phase5_rebuild.py
├── phase6_model_training.py
│
└── data/notebooks/
    ├── airline_reliability.py
    ├── time_recommendation.py
    └── phase6_export_delay_predictions.py
```

---

## 👥 Team

**SkyRisk Analytics — DATA 603, University of Maryland Baltimore County**

| Name | Role |
|---|---|
| Nikhil Patil | Flight Delay Prediction · Data Pipeline · ML Engineering · App Integration |
| Sarika Thunipura | Fare Trend Prediction · Data Analysis |
| Suhani Shah | Weather Data Analysis · Visualizations |
| Madhurima Mukhopadhyay | Travel Risk Score Engine · Recommendation System |

---

## 📄 License

Developed for academic purposes — DATA 603 Big Data Technologies, UMBC.
