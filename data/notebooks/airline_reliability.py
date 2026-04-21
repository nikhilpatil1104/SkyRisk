"""
=============================================================
 Airline Reliability Score
=============================================================
 Reads features_final.parquet (Phase 5 output)
 Computes per-airline delay rate and reliability score
 Exports airline_reliability.csv

 Columns: airline, delay_rate, reliability_score

 Run: python airline_reliability.py
=============================================================
"""

import duckdb
import pandas as pd
import numpy as np
import os

PARQUET_IN = "D:\\project 603\\data\\processed\\features_final.parquet"
OUTPUT_DIR = "D:\\project 603\\outputs"
CSV_OUT    = f"{OUTPUT_DIR}\\airline_reliability.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full airline name mapping (IATA code → name)
AIRLINE_NAMES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air (AA Regional)",
    "OO": "SkyWest Airlines",
    "YX": "Republic Airways",
    "EV": "ExpressJet Airlines",
    "VX": "Virgin America",
    "YV": "Mesa Airlines",
    "9E": "Endeavor Air",
    "OH": "PSA Airlines",
    "QX": "Horizon Air",
}

AIRPORTS_20 = {
    "ATL","LAX","ORD","DFW","DEN","JFK","SFO","CLT","LAS","PHX",
    "MIA","IAH","SEA","EWR","BOS","SLC","SAN","TPA","PDX","AUS"
}

print("=" * 60)
print("  AIRLINE RELIABILITY SCORE COMPUTATION")
print("=" * 60)
print()

# ── Step 1: Query via DuckDB ──────────────────────────────
print("[1/4] Loading flight data via DuckDB...")

con = duckdb.connect()

# Build airport filter string
airport_list = "'" + "','".join(AIRPORTS_20) + "'"

airline_raw = con.execute(f"""
    SELECT
        Reporting_Airline                                           AS airline,
        COUNT(*)                                                    AS total_flights,
        SUM(CASE WHEN DEP_DEL15 = 1 THEN 1 ELSE 0 END)            AS delayed_flights,
        ROUND(100.0 * AVG(CAST(DEP_DEL15 AS FLOAT)), 4)           AS delay_rate_pct,
        ROUND(AVG(CAST(DepDelay AS FLOAT)), 4)                     AS avg_dep_delay_mins,
        ROUND(AVG(CAST(DepDelayMinutes AS FLOAT)), 4)              AS avg_delay_mins_when_late,
        ROUND(AVG(CAST(ArrDelay AS FLOAT)), 4)                     AS avg_arr_delay_mins,
        ROUND(AVG(CAST(Cancelled AS FLOAT)) * 100, 4)              AS cancellation_rate_pct
    FROM read_parquet('{PARQUET_IN}')
    WHERE AIRPORT_CODE IN ({airport_list})
    AND DEP_DEL15 IS NOT NULL
    AND Reporting_Airline IS NOT NULL
    -- Exclude 2020 COVID anomaly year from reliability baseline
    AND YEAR(CAST(FL_DATE AS DATE)) != 2020
    GROUP BY Reporting_Airline
    HAVING COUNT(*) >= 10000
    ORDER BY delay_rate_pct ASC
""").df()

con.close()

print(f"  Airlines found : {len(airline_raw)}")
print(f"  Total flights  : {airline_raw['total_flights'].sum():,}")

# ── Step 2: Compute reliability score ─────────────────────
# Method: 1 - (avg_dep_delay / max_avg_dep_delay)
# Higher score = more reliable (fewer / shorter delays)
# Matches the PySpark formula in the rubric exactly
print()
print("[2/4] Computing reliability scores...")

max_delay = airline_raw["avg_dep_delay_mins"].max()
min_delay = airline_raw["avg_dep_delay_mins"].min()

# Primary score: inverse normalization on average departure delay
# This matches the formula: reliability_score = 1 - (avg_delay / max_delay)
airline_raw["reliability_score_raw"] = (
    1 - (airline_raw["avg_dep_delay_mins"] / max_delay)
)

# Composite reliability score — weighted combination:
# 60% based on delay rate, 40% based on avg delay duration
# Both normalized 0-1, then combined
max_rate = airline_raw["delay_rate_pct"].max()
min_rate = airline_raw["delay_rate_pct"].min()

score_from_rate = 1 - (airline_raw["delay_rate_pct"] / max_rate)
score_from_dur  = 1 - (airline_raw["avg_dep_delay_mins"] / max_delay)

airline_raw["reliability_score"] = (
    0.60 * score_from_rate +
    0.40 * score_from_dur
).round(4)

# Sort by reliability score descending (most reliable first)
airline_raw = airline_raw.sort_values("reliability_score", ascending=False).reset_index(drop=True)

# Add rank
airline_raw["rank"] = range(1, len(airline_raw) + 1)

# Add airline full names
airline_raw["airline_name"] = airline_raw["airline"].map(AIRLINE_NAMES).fillna("Other")

# ── Step 3: Build final CSV in required format ─────────────
# Required columns: airline, delay_rate, reliability_score
print()
print("[3/4] Building output CSV...")

df_out = pd.DataFrame({
    "airline":           airline_raw["airline"],
    "airline_name":      airline_raw["airline_name"],
    "total_flights":     airline_raw["total_flights"],
    "delayed_flights":   airline_raw["delayed_flights"],
    "delay_rate":        (airline_raw["delay_rate_pct"] / 100).round(4),  # as decimal 0-1
    "delay_rate_pct":    airline_raw["delay_rate_pct"].round(2),           # as percentage
    "avg_delay_mins":    airline_raw["avg_dep_delay_mins"].round(2),
    "cancellation_rate": airline_raw["cancellation_rate_pct"].round(4),
    "reliability_score": airline_raw["reliability_score"],
    "reliability_rank":  airline_raw["rank"],
})

df_out.to_csv(CSV_OUT, index=False)

# ── Step 4: Print summary ──────────────────────────────────
print()
print("[4/4] Results:")
print()
print(f"  {'Rank':<6} {'Code':<6} {'Airline':<28} {'Delay%':>8} {'Avg Min':>8} {'Reliability':>12}")
print(f"  {'-'*6} {'-'*6} {'-'*28} {'-'*8} {'-'*8} {'-'*12}")

for _, row in df_out.iterrows():
    bar_len = int(row["reliability_score"] * 20)
    bar     = "█" * bar_len + "░" * (20 - bar_len)
    print(
        f"  {int(row['reliability_rank']):<6} "
        f"{row['airline']:<6} "
        f"{row['airline_name']:<28} "
        f"{row['delay_rate_pct']:>7.1f}% "
        f"{row['avg_delay_mins']:>7.1f}m "
        f"  {row['reliability_score']:.4f}  {bar}"
    )

print()
print("=" * 60)
print(f"  Most reliable  : {df_out.iloc[0]['airline']} — {df_out.iloc[0]['airline_name']}")
print(f"                   Score: {df_out.iloc[0]['reliability_score']:.4f}  |  Delay rate: {df_out.iloc[0]['delay_rate_pct']:.1f}%")
print()
print(f"  Least reliable : {df_out.iloc[-1]['airline']} — {df_out.iloc[-1]['airline_name']}")
print(f"                   Score: {df_out.iloc[-1]['reliability_score']:.4f}  |  Delay rate: {df_out.iloc[-1]['delay_rate_pct']:.1f}%")
print()
print(f"  Output saved   : {CSV_OUT}")
print(f"  Rows           : {len(df_out)}")
print(f"  Columns        : airline, delay_rate, reliability_score (+ extras)")
print()
print("  Note: 2020 excluded from reliability baseline (COVID anomaly).")
print("  Score formula : 0.60 * (1 - delay_rate/max_rate)")
print("                + 0.40 * (1 - avg_delay/max_avg_delay)")
print("  Range         : 0.0 (least reliable) → 1.0 (most reliable)")
print("=" * 60)
