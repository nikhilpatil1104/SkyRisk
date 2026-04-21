"""
Phase 6 — delay_predictions.csv Export
Reads features_final.parquet + saved XGBoost model
Outputs delay_predictions.csv in exact required format:
  origin, destination, airline, month, delay_probability
Run AFTER phase6_model_training.py completes.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
import xgboost as xgb
import os, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

PARQUET_IN   = "D:/project/data/processed/features_final.parquet"
MODELS_DIR   = "D:/project/models"
OUTPUT_DIR   = "D:/project/outputs"
CSV_OUT      = f"{OUTPUT_DIR}/delay_predictions.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AIRPORTS_20 = {
    "ATL","LAX","ORD","DFW","DEN","JFK","SFO","CLT","LAS","PHX",
    "MIA","IAH","SEA","EWR","BOS","SLC","SAN","TPA","PDX","AUS"
}

CATEGORICAL_COLS = [
    "AIRPORT_CODE", "Dest", "Reporting_Airline",
    "season", "time_of_day", "wind_category",
    "visibility_category", "precip_category", "distance_bucket"
]
NUMERIC_COLS = [
    "month", "day_of_week", "dep_hour", "quarter",
    "is_weekend", "is_holiday_period", "week_of_year",
    "is_covid_year", "flight_volume_index",
    "avg_temp_c", "min_temp_c", "max_temp_c",
    "avg_visibility_km", "min_visibility_km",
    "avg_wind_ms", "max_wind_ms",
    "total_precip_mm", "avg_pressure_hpa",
    "avg_ceiling_m", "min_ceiling_m",
    "is_freezing", "is_extreme_heat", "temp_range_c",
    "is_high_wind", "is_low_visibility",
    "is_precipitation", "is_low_ceiling",
    "weather_severity_score",
    "is_hub", "airport_hist_delay_rate",
    "daily_flight_count", "congestion_index",
    "Distance", "route_total_flights",
    "carrier_hist_delay_rate",
]

print("[load] Loading saved models and preprocessors...")
encoders     = joblib.load(f"{MODELS_DIR}/encoders.pkl")
imputer      = joblib.load(f"{MODELS_DIR}/imputer.pkl")
ALL_FEATURES = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
xgb_model    = xgb.Booster()
xgb_model.load_model(f"{MODELS_DIR}/xgboost.json")
print("[load] Done.")

# ── Collect predictions in chunks ─────────────────────────
print("[predict] Scoring all flights from features_final.parquet...")
rows = []
pf   = pq.ParquetFile(PARQUET_IN)

for n, batch in enumerate(pf.iter_batches(batch_size=2_000_000), 1):
    chunk = batch.to_pandas()
    chunk = chunk[chunk["AIRPORT_CODE"].isin(AIRPORTS_20)].copy()
    if len(chunk) == 0:
        continue

    # Save identifiers before encoding
    origins   = chunk["AIRPORT_CODE"].values
    dests     = chunk["Dest"].values
    airlines  = chunk["Reporting_Airline"].values
    months    = pd.to_datetime(chunk["FL_DATE"]).dt.month.values

    # Encode
    for col in CATEGORICAL_COLS:
        if col not in chunk.columns:
            chunk[col] = "Unknown"
        vals  = chunk[col].astype(str).fillna("Unknown")
        known = set(encoders[col].classes_)
        chunk[col] = encoders[col].transform(
            vals.map(lambda x: x if x in known else "Unknown")
        )

    X    = imputer.transform(chunk[ALL_FEATURES].values.astype(float))
    dmat = xgb.DMatrix(X, feature_names=ALL_FEATURES)
    prob = xgb_model.predict(dmat)

    chunk_df = pd.DataFrame({
        "origin":            origins,
        "destination":       dests,
        "airline":           airlines,
        "month":             months,
        "delay_probability": prob.round(4)
    })
    rows.append(chunk_df)
    print(f"  Chunk {n}: {len(chunk):,} flights scored")

print("[aggregate] Combining and aggregating by route + airline + month...")
all_preds = pd.concat(rows, ignore_index=True)

# Aggregate — mean probability per unique origin/dest/airline/month combination
delay_predictions = (
    all_preds
    .groupby(["origin", "destination", "airline", "month"])
    .agg(delay_probability=("delay_probability", "mean"))
    .reset_index()
)
delay_predictions["delay_probability"] = delay_predictions["delay_probability"].round(4)

# Sort for readability
delay_predictions = delay_predictions.sort_values(
    ["origin", "destination", "airline", "month"]
).reset_index(drop=True)

# Save
delay_predictions.to_csv(CSV_OUT, index=False)

print(f"\n[done] delay_predictions.csv")
print(f"       Rows      : {len(delay_predictions):,}")
print(f"       Columns   : origin, destination, airline, month, delay_probability")
print(f"       Location  : {CSV_OUT}")
print(f"       Size      : {os.path.getsize(CSV_OUT)/1e6:.1f} MB")
print()
print("[sample] First 10 rows:")
print(delay_predictions.head(10).to_string(index=False))
print()
print("[sample] Highest delay probability routes:")
print(delay_predictions.nlargest(5, "delay_probability")[
    ["origin","destination","airline","month","delay_probability"]
].to_string(index=False))
print()
print("[sample] Lowest delay probability routes:")
print(delay_predictions.nsmallest(5, "delay_probability")[
    ["origin","destination","airline","month","delay_probability"]
].to_string(index=False))
