import pandas as pd

# These are imported by llm_handler for the smart fallback
__all__ = ["compute_risk_score", "risk_category", "get_flight_risk", "get_alternatives", "get_best_time_to_fly"]

# ---------------------------------------------------------------------------
# Weights for risk score formula
# risk = 0.4*delay_prob + 0.2*weather_severity + 0.2*congestion + 0.2*(1-reliability)
# ---------------------------------------------------------------------------
W_DELAY       = 0.4
W_WEATHER     = 0.2
W_CONGESTION  = 0.2
W_RELIABILITY = 0.2


def compute_risk_score(delay_prob, weather_severity, congestion_score, reliability_score):
    score = (
        W_DELAY      * delay_prob
      + W_WEATHER    * weather_severity
      + W_CONGESTION * congestion_score
      + W_RELIABILITY * (1 - reliability_score)
    )
    return round(score, 4)


def risk_category(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    else:
        return "High"


def get_flight_risk(dfs, origin, destination, airline, month):
    """
    Compute risk score for a specific flight query.
    Returns a dict with all components, or None if data not found.
    """
    origin      = origin.upper().strip()
    destination = destination.upper().strip()
    airline     = airline.upper().strip()

    # --- delay probability ---
    delay_row = dfs["delay"][
        (dfs["delay"]["origin"]      == origin) &
        (dfs["delay"]["destination"] == destination) &
        (dfs["delay"]["airline"]     == airline) &
        (dfs["delay"]["month"]       == month)
    ]
    if delay_row.empty:
        return None
    delay_prob = float(delay_row["delay_probability"].values[0])

    # --- weather severity for origin airport ---
    weather_row = dfs["weather"][
        (dfs["weather"]["airport"] == origin) &
        (dfs["weather"]["month"]   == month)
    ]
    weather_severity = float(weather_row["weather_severity"].values[0]) if not weather_row.empty else 0.3

    # --- congestion for origin airport ---
    cong_row = dfs["congestion"][dfs["congestion"]["airport"] == origin]
    congestion_score = float(cong_row["congestion_score"].values[0]) if not cong_row.empty else 0.5

    # --- reliability for airline ---
    rel_row = dfs["reliability"][dfs["reliability"]["airline"] == airline]
    reliability_score = float(rel_row["reliability_score"].values[0]) if not rel_row.empty else 0.5
    airline_name      = rel_row["airline_name"].values[0] if not rel_row.empty else airline

    # --- fare info ---
    fare_row = dfs["fare"][
        (dfs["fare"]["origin"]      == origin) &
        (dfs["fare"]["destination"] == destination)
    ]
    current_fare    = float(fare_row["current_fare"].values[0])    if not fare_row.empty else None
    predicted_fare  = float(fare_row["predicted_fare"].values[0])  if not fare_row.empty else None
    fare_trend      = fare_row["trend"].values[0]                  if not fare_row.empty else "Unknown"
    fare_rec        = fare_row["recommendation"].values[0]         if not fare_row.empty else "N/A"

    score    = compute_risk_score(delay_prob, weather_severity, congestion_score, reliability_score)
    category = risk_category(score)

    return {
        "origin":              origin,
        "destination":         destination,
        "airline":             airline,
        "airline_name":        airline_name,
        "month":               month,
        "delay_probability":   delay_prob,
        "weather_severity":    weather_severity,
        "congestion_score":    congestion_score,
        "reliability_score":   reliability_score,
        "risk_score":          score,
        "risk_category":       category,
        "current_fare":        current_fare,
        "predicted_fare":      predicted_fare,
        "fare_trend":          fare_trend,
        "fare_recommendation": fare_rec,
    }


def get_alternatives(dfs, origin, destination, month, current_risk_score, top_n=3):
    """
    Find top N alternative airlines on the same route with lower risk score.
    Returns a list of dicts sorted by risk_score ASC, predicted_fare ASC.
    """
    route_delays = dfs["delay"][
        (dfs["delay"]["origin"]      == origin) &
        (dfs["delay"]["destination"] == destination) &
        (dfs["delay"]["month"]       == month)
    ].copy()

    if route_delays.empty:
        return []

    results = []
    for _, row in route_delays.iterrows():
        alt_airline = row["airline"]
        alt_delay   = row["delay_probability"]

        weather_row = dfs["weather"][
            (dfs["weather"]["airport"] == origin) &
            (dfs["weather"]["month"]   == month)
        ]
        weather_sev = float(weather_row["weather_severity"].values[0]) if not weather_row.empty else 0.3

        cong_row = dfs["congestion"][dfs["congestion"]["airport"] == origin]
        cong_score = float(cong_row["congestion_score"].values[0]) if not cong_row.empty else 0.5

        rel_row = dfs["reliability"][dfs["reliability"]["airline"] == alt_airline]
        if rel_row.empty:
            continue
        rel_score    = float(rel_row["reliability_score"].values[0])
        airline_name = rel_row["airline_name"].values[0]

        score = compute_risk_score(alt_delay, weather_sev, cong_score, rel_score)

        fare_row = dfs["fare"][
            (dfs["fare"]["origin"]      == origin) &
            (dfs["fare"]["destination"] == destination)
        ]
        pred_fare = float(fare_row["predicted_fare"].values[0]) if not fare_row.empty else None

        results.append({
            "airline":           alt_airline,
            "airline_name":      airline_name,
            "delay_probability": round(alt_delay, 4),
            "reliability_score": round(rel_score, 4),
            "risk_score":        round(score, 4),
            "risk_category":     risk_category(score),
            "predicted_fare":    pred_fare,
        })

    better = [r for r in results if r["risk_score"] < current_risk_score]
    better.sort(key=lambda x: (x["risk_score"], x["predicted_fare"] or 9999))
    return better[:top_n]


def get_best_time_to_fly(dfs):
    """Return best hours and days to fly from time_recommendation data."""
    time_rec = dfs["time_rec"]

    hours  = time_rec[time_rec["dimension"] == "hour"]
    days   = time_rec[time_rec["dimension"] == "day_of_week"]
    months = time_rec[time_rec["dimension"] == "month"]

    best_hours = hours[hours["recommendation"] == "Best"]["label"].tolist()
    best_days  = days[days["recommendation"]  == "Best"]["label"].tolist()

    return {
        "best_hours":  best_hours,
        "best_days":   best_days,
        "months_data": months[["label", "delay_rate_pct", "recommendation"]].to_dict(orient="records"),
    }
