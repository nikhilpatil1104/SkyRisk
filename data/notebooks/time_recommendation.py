"""
=============================================================
 Time Recommendation Analysis
=============================================================
 Reads features_final.parquet (Phase 5 output)
 Computes delay patterns by departure hour, day of week,
 month, and season to generate time-based recommendations
 Exports time_recommendation.csv

 Run: python time_recommendation.py
=============================================================
"""

import duckdb
import pandas as pd
import numpy as np
import os

PARQUET_IN = "D:\\project 603\\data\\processed\\features_final.parquet"
OUTPUT_DIR = "D:\\project 603\\outputs"
CSV_OUT    = f"{OUTPUT_DIR}\\time_recommendation.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AIRPORTS_20 = {
    "ATL","LAX","ORD","DFW","DEN","JFK","SFO","CLT","LAS","PHX",
    "MIA","IAH","SEA","EWR","BOS","SLC","SAN","TPA","PDX","AUS"
}
airport_list = "'" + "','".join(AIRPORTS_20) + "'"

print("=" * 60)
print("  TIME RECOMMENDATION ANALYSIS")
print("=" * 60)
print()

con = duckdb.connect()

# ── 1. By Departure Hour ──────────────────────────────────
print("[1/5] Analysing by departure hour...")

df_hour = con.execute(f"""
    SELECT
        dep_hour                                                     AS hour,
        COUNT(*)                                                     AS total_flights,
        ROUND(AVG(CAST(DepDelay AS FLOAT)), 2)                      AS avg_delay_mins,
        ROUND(AVG(CAST(DepDelayMinutes AS FLOAT)), 2)               AS avg_delay_when_late,
        ROUND(100.0 * AVG(CAST(DEP_DEL15 AS FLOAT)), 2)            AS delay_rate_pct,
        ROUND(AVG(CAST(Cancelled AS FLOAT)) * 100, 2)               AS cancellation_rate_pct
    FROM read_parquet('{PARQUET_IN}')
    WHERE AIRPORT_CODE IN ({airport_list})
    AND DEP_DEL15 IS NOT NULL
    AND dep_hour IS NOT NULL
    AND dep_hour BETWEEN 0 AND 23
    AND YEAR(CAST(FL_DATE AS DATE)) != 2020
    GROUP BY dep_hour
    ORDER BY dep_hour
""").df()

# Normalize delay score 0-1 for recommendation strength
max_delay = df_hour["avg_delay_mins"].max()
min_delay = df_hour["avg_delay_mins"].min()
df_hour["delay_score"] = (
    (df_hour["avg_delay_mins"] - min_delay) / (max_delay - min_delay)
).round(4)

# Hour label e.g. "6 AM", "3 PM"
def hour_label(h):
    if h == 0:   return "12 AM"
    if h < 12:   return f"{h} AM"
    if h == 12:  return "12 PM"
    return f"{h-12} PM"

df_hour["hour_label"] = df_hour["hour"].apply(hour_label)

# Recommendation tier
def hour_rec(score):
    if score <= 0.20:  return "Best"
    if score <= 0.45:  return "Good"
    if score <= 0.70:  return "Fair"
    return "Avoid"

df_hour["recommendation"] = df_hour["delay_score"].apply(hour_rec)

# ── 2. By Day of Week ─────────────────────────────────────
print("[2/5] Analysing by day of week...")

df_dow = con.execute(f"""
    SELECT
        day_of_week                                                  AS dow_num,
        COUNT(*)                                                     AS total_flights,
        ROUND(AVG(CAST(DepDelay AS FLOAT)), 2)                      AS avg_delay_mins,
        ROUND(100.0 * AVG(CAST(DEP_DEL15 AS FLOAT)), 2)            AS delay_rate_pct
    FROM read_parquet('{PARQUET_IN}')
    WHERE AIRPORT_CODE IN ({airport_list})
    AND DEP_DEL15 IS NOT NULL
    AND day_of_week IS NOT NULL
    AND YEAR(CAST(FL_DATE AS DATE)) != 2020
    GROUP BY day_of_week
    ORDER BY day_of_week
""").df()

DAY_NAMES = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
df_dow["day_name"] = df_dow["dow_num"].map(DAY_NAMES)

max_dow = df_dow["avg_delay_mins"].max()
min_dow = df_dow["avg_delay_mins"].min()
df_dow["delay_score"] = (
    (df_dow["avg_delay_mins"] - min_dow) / (max_dow - min_dow)
).round(4)
df_dow["recommendation"] = df_dow["delay_score"].apply(hour_rec)

# ── 3. By Month ───────────────────────────────────────────
print("[3/5] Analysing by month...")

df_month = con.execute(f"""
    SELECT
        month                                                        AS month_num,
        COUNT(*)                                                     AS total_flights,
        ROUND(AVG(CAST(DepDelay AS FLOAT)), 2)                      AS avg_delay_mins,
        ROUND(100.0 * AVG(CAST(DEP_DEL15 AS FLOAT)), 2)            AS delay_rate_pct
    FROM read_parquet('{PARQUET_IN}')
    WHERE AIRPORT_CODE IN ({airport_list})
    AND DEP_DEL15 IS NOT NULL
    AND month IS NOT NULL
    AND YEAR(CAST(FL_DATE AS DATE)) != 2020
    GROUP BY month
    ORDER BY month
""").df()

MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
df_month["month_name"] = df_month["month_num"].map(MONTH_NAMES)

max_month = df_month["avg_delay_mins"].max()
min_month = df_month["avg_delay_mins"].min()
df_month["delay_score"] = (
    (df_month["avg_delay_mins"] - min_month) / (max_month - min_month)
).round(4)
df_month["recommendation"] = df_month["delay_score"].apply(hour_rec)

# ── 4. By Season ──────────────────────────────────────────
print("[4/5] Analysing by season...")

df_season = con.execute(f"""
    SELECT
        season,
        COUNT(*)                                                     AS total_flights,
        ROUND(AVG(CAST(DepDelay AS FLOAT)), 2)                      AS avg_delay_mins,
        ROUND(100.0 * AVG(CAST(DEP_DEL15 AS FLOAT)), 2)            AS delay_rate_pct
    FROM read_parquet('{PARQUET_IN}')
    WHERE AIRPORT_CODE IN ({airport_list})
    AND DEP_DEL15 IS NOT NULL
    AND season IS NOT NULL
    AND season != 'nan'
    AND YEAR(CAST(FL_DATE AS DATE)) != 2020
    GROUP BY season
    ORDER BY avg_delay_mins ASC
""").df()

max_s = df_season["avg_delay_mins"].max()
min_s = df_season["avg_delay_mins"].min()
df_season["delay_score"] = (
    (df_season["avg_delay_mins"] - min_s) / (max_s - min_s)
).round(4)
df_season["recommendation"] = df_season["delay_score"].apply(hour_rec)

# ── 5. Build master time_recommendation.csv ───────────────
print("[5/5] Building time_recommendation.csv...")

# Primary output — hourly (matches the example format exactly)
# Plus combined rows for day, month, season with dimension tag
rows = []

# --- Hourly rows
for _, r in df_hour.iterrows():
    rows.append({
        "dimension":         "hour",
        "value":             int(r["hour"]),
        "label":             r["hour_label"],
        "total_flights":     int(r["total_flights"]),
        "avg_delay_mins":    r["avg_delay_mins"],
        "delay_rate_pct":    r["delay_rate_pct"],
        "cancellation_rate": r.get("cancellation_rate_pct", None),
        "delay_score":       r["delay_score"],
        "recommendation":    r["recommendation"],
    })

# --- Day of week rows
for _, r in df_dow.iterrows():
    rows.append({
        "dimension":         "day_of_week",
        "value":             int(r["dow_num"]),
        "label":             r["day_name"],
        "total_flights":     int(r["total_flights"]),
        "avg_delay_mins":    r["avg_delay_mins"],
        "delay_rate_pct":    r["delay_rate_pct"],
        "cancellation_rate": None,
        "delay_score":       r["delay_score"],
        "recommendation":    r["recommendation"],
    })

# --- Month rows
for _, r in df_month.iterrows():
    rows.append({
        "dimension":         "month",
        "value":             int(r["month_num"]),
        "label":             r["month_name"],
        "total_flights":     int(r["total_flights"]),
        "avg_delay_mins":    r["avg_delay_mins"],
        "delay_rate_pct":    r["delay_rate_pct"],
        "cancellation_rate": None,
        "delay_score":       r["delay_score"],
        "recommendation":    r["recommendation"],
    })

# --- Season rows
for _, r in df_season.iterrows():
    rows.append({
        "dimension":         "season",
        "value":             r["season"],
        "label":             r["season"],
        "total_flights":     int(r["total_flights"]),
        "avg_delay_mins":    r["avg_delay_mins"],
        "delay_rate_pct":    r["delay_rate_pct"],
        "cancellation_rate": None,
        "delay_score":       r["delay_score"],
        "recommendation":    r["recommendation"],
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(CSV_OUT, index=False)

con.close()

# ── Print Summary ─────────────────────────────────────────
SEP = "=" * 60

print()
print(SEP)
print("  BY DEPARTURE HOUR")
print(SEP)
print(f"  {'Hour':<8} {'Avg Delay':>10} {'Delay %':>8} {'Score':>7}  {'Rec'}")
print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*7}  {'-'*10}")
for _, r in df_hour.iterrows():
    flag = " ◀ best" if r["recommendation"] == "Best" else (" ✗ avoid" if r["recommendation"] == "Avoid" else "")
    print(f"  {r['hour_label']:<8} {r['avg_delay_mins']:>8.1f}m  {r['delay_rate_pct']:>6.1f}%  {r['delay_score']:>6.4f}  {r['recommendation']}{flag}")

print()
print(SEP)
print("  BY DAY OF WEEK")
print(SEP)
print(f"  {'Day':<12} {'Avg Delay':>10} {'Delay %':>8} {'Rec'}")
print(f"  {'-'*12} {'-'*10} {'-'*8}  {'-'*10}")
for _, r in df_dow.sort_values("avg_delay_mins").iterrows():
    flag = " ◀ best" if r["recommendation"] == "Best" else (" ✗ avoid" if r["recommendation"] == "Avoid" else "")
    print(f"  {r['day_name']:<12} {r['avg_delay_mins']:>8.1f}m  {r['delay_rate_pct']:>6.1f}%  {r['recommendation']}{flag}")

print()
print(SEP)
print("  BY MONTH")
print(SEP)
print(f"  {'Month':<12} {'Avg Delay':>10} {'Delay %':>8} {'Rec'}")
print(f"  {'-'*12} {'-'*10} {'-'*8}  {'-'*10}")
for _, r in df_month.sort_values("avg_delay_mins").iterrows():
    flag = " ◀ best" if r["recommendation"] == "Best" else (" ✗ avoid" if r["recommendation"] == "Avoid" else "")
    print(f"  {r['month_name']:<12} {r['avg_delay_mins']:>8.1f}m  {r['delay_rate_pct']:>6.1f}%  {r['recommendation']}{flag}")

print()
print(SEP)
print("  BY SEASON")
print(SEP)
for _, r in df_season.iterrows():
    print(f"  {r['season']:<10} {r['avg_delay_mins']:>6.1f} min avg  |  {r['delay_rate_pct']:.1f}% delay rate  |  {r['recommendation']}")

print()
print(SEP)
print("  FINAL RECOMMENDATIONS")
print(SEP)

best_hours  = df_hour[df_hour["recommendation"] == "Best"]["hour_label"].tolist()
avoid_hours = df_hour[df_hour["recommendation"] == "Avoid"]["hour_label"].tolist()
best_days   = df_dow[df_dow["recommendation"] == "Best"]["day_name"].tolist()
avoid_days  = df_dow[df_dow["recommendation"] == "Avoid"]["day_name"].tolist()
best_months = df_month[df_month["recommendation"] == "Best"]["month_name"].tolist()
avoid_months= df_month[df_month["recommendation"] == "Avoid"]["month_name"].tolist()
best_season = df_season[df_season["delay_score"] == df_season["delay_score"].min()]["season"].values[0]
worst_season= df_season[df_season["delay_score"] == df_season["delay_score"].max()]["season"].values[0]

print(f"  Best departure times  : {', '.join(best_hours)}")
print(f"  Times to avoid        : {', '.join(avoid_hours)}")
print()
print(f"  Best days to fly      : {', '.join(best_days)}")
print(f"  Days to avoid         : {', '.join(avoid_days)}")
print()
print(f"  Best months           : {', '.join(best_months)}")
print(f"  Months to avoid       : {', '.join(avoid_months)}")
print()
print(f"  Best season           : {best_season}")
print(f"  Worst season          : {worst_season}")
print()
print(f"  Output saved          : {CSV_OUT}")
print(f"  Total rows            : {len(df_out)}")
print(f"  Columns               : dimension, value, label, total_flights,")
print(f"                          avg_delay_mins, delay_rate_pct,")
print(f"                          cancellation_rate, delay_score, recommendation")
print(SEP)
