# SkyRisk — AI Flight Risk Advisor

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Add your NVIDIA API key in `llm_handler.py`:
```python
NVIDIA_API_KEY = "nvapi-YOUR_KEY_HERE"
```
Get your key at: https://integrate.api.nvidia.com

3. Place all CSV files in a `data/` folder next to `app.py`:
```
skyrisk/
├── app.py
├── data_loader.py
├── risk_engine.py
├── llm_handler.py
├── requirements.txt
└── data/
    ├── delay_predictions.csv
    ├── airport_congestion.csv
    ├── airline_reliability.csv
    ├── airport_weather_severity.csv
    ├── route_fare_predictions.csv
    └── time_recommendation.csv
```

4. Run:
```
streamlit run app.py
```

## Architecture

```
User (image or text)
        ↓
  llm_handler.py  ← NVIDIA LLaMA vision/chat
        ↓
  risk_engine.py  ← Pandas joins + weighted risk formula
        ↓
  data_loader.py  ← Loads & normalizes all 6 CSVs
        ↓
   app.py (Streamlit UI)
```

## Risk Score Formula
```
risk = 0.4 × delay_probability
     + 0.2 × weather_severity
     + 0.2 × congestion_score
     + 0.2 × (1 - reliability_score)
```
- Low Risk:    score < 0.3
- Medium Risk: 0.3 ≤ score < 0.6
- High Risk:   score ≥ 0.6
