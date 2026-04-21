import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_all():
    """Load and preprocess all CSVs. Returns a dict of DataFrames."""

    # --- Delay Predictions ---
    delay = pd.read_csv(os.path.join(DATA_DIR, "delay_predictions.csv"))
    delay.columns = delay.columns.str.strip().str.lower()
    # delay_probability is already 0-1

    # --- Airport Congestion ---
    congestion = pd.read_csv(os.path.join(DATA_DIR, "airport_congestion.csv"))
    congestion.columns = congestion.columns.str.strip().str.lower()
    # congestion_score already 0-1

    # --- Airline Reliability ---
    reliability = pd.read_csv(os.path.join(DATA_DIR, "airline_reliability.csv"))
    reliability.columns = reliability.columns.str.strip().str.lower()
    # reliability_score already 0-1

    # --- Airport Weather Severity ---
    # Has 10 years × 20 airports × 12 months — average across years per airport/month
    weather = pd.read_csv(os.path.join(DATA_DIR, "airport_weather_severity.csv"))
    weather.columns = weather.columns.str.strip().str.lower()
    weather = (
        weather.groupby(["airport", "month"])["severity_score_normalized"]
        .mean()
        .reset_index()
        .rename(columns={"severity_score_normalized": "weather_severity"})
    )
    # Normalize to 0-1
    weather["weather_severity"] = weather["weather_severity"] / 100.0

    # --- Route Fare Predictions ---
    fare = pd.read_csv(os.path.join(DATA_DIR, "route_fare_predictions.csv"))
    fare.columns = fare.columns.str.strip().str.lower().str.replace(" ", "_")
    # Columns: origin, destination, current_fare, predicted_fare, trend, recommendation

    # --- Time Recommendations ---
    time_rec = pd.read_csv(os.path.join(DATA_DIR, "time_recommendation.csv"))
    time_rec.columns = time_rec.columns.str.strip().str.lower()

    return {
        "delay": delay,
        "congestion": congestion,
        "reliability": reliability,
        "weather": weather,
        "fare": fare,
        "time_rec": time_rec,
    }


def get_airline_map(dfs):
    """Return dict mapping airline code -> full name."""
    rel = dfs["reliability"]
    return dict(zip(rel["airline"], rel["airline_name"]))
