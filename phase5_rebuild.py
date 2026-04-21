"""
Phase 5 Rebuild — Chunked Feature Engineering
Run: python phase5_rebuild.py
"""

import pandas as pd
import numpy as np
import duckdb
import os
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_IN  = "D:/project/data/processed/bts_weather_joined.parquet"
PARQUET_OUT = "D:/project/data/processed/features_final.parquet"

HUB_AIRPORTS = {
    "ATL","LAX","ORD","DFW","DEN",
    "JFK","SFO","CLT","LAS","PHX",
    "MIA","IAH","SEA","EWR","BOS"
}

HOLIDAY_PERIODS = [
    (1,1,2),(5,23,27),(7,1,7),
    (9,1,2),(11,20,30),(12,20,31),
]

# ── Step 1: Compute all lookup tables via DuckDB streaming ──
print("[setup] Computing lookup tables via DuckDB...")
con = duckdb.connect()

print("  [1/5] Monthly flight volumes...")
monthly_vol = con.execute(f"""
    SELECT
        AIRPORT_CODE,
        YEAR(CAST(FL_DATE AS DATE))  AS yr,
        MONTH(CAST(FL_DATE AS DATE)) AS mo,
        COUNT(*) AS monthly_flights
    FROM read_parquet('{PARQUET_IN}')
    GROUP BY ALL
""").df()
monthly_vol["flight_volume_index"] = monthly_vol.groupby("AIRPORT_CODE")["monthly_flights"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
).round(4)

print("  [2/5] Airport delay rates (excl 2020)...")
airport_stats = con.execute(f"""
    SELECT
        AIRPORT_CODE,
        ROUND(AVG(CAST(DEP_DEL15 AS FLOAT)), 4) AS airport_hist_delay_rate,
        ROUND(AVG(DepDelayMinutes), 2)            AS airport_hist_avg_delay_mins
    FROM read_parquet('{PARQUET_IN}')
    WHERE YEAR(CAST(FL_DATE AS DATE)) != 2020
    AND DEP_DEL15 IS NOT NULL
    GROUP BY AIRPORT_CODE
""").df()

print("  [3/5] Daily flight counts + congestion...")
daily_counts = con.execute(f"""
    SELECT
        AIRPORT_CODE,
        CAST(FL_DATE AS VARCHAR) AS FL_DATE_STR,
        COUNT(*) AS daily_flight_count
    FROM read_parquet('{PARQUET_IN}')
    GROUP BY AIRPORT_CODE, CAST(FL_DATE AS VARCHAR)
""").df()
daily_counts["mo"] = pd.to_datetime(daily_counts["FL_DATE_STR"]).dt.month
monthly_avg = daily_counts.groupby(["AIRPORT_CODE","mo"])["daily_flight_count"].mean().reset_index()
monthly_avg.rename(columns={"daily_flight_count":"avg_daily"}, inplace=True)
daily_counts = daily_counts.merge(monthly_avg, on=["AIRPORT_CODE","mo"], how="left")
daily_counts["congestion_index"] = (daily_counts["daily_flight_count"] / (daily_counts["avg_daily"] + 1e-9)).round(4)
daily_counts = daily_counts[["AIRPORT_CODE","FL_DATE_STR","daily_flight_count","congestion_index"]]

print("  [4/5] Route frequencies...")
route_freq = con.execute(f"""
    SELECT AIRPORT_CODE, Dest, COUNT(*) AS route_total_flights
    FROM read_parquet('{PARQUET_IN}')
    GROUP BY AIRPORT_CODE, Dest
""").df()

print("  [5/5] Carrier delay rates (excl 2020)...")
carrier_stats = con.execute(f"""
    SELECT
        Reporting_Airline,
        ROUND(AVG(CAST(DEP_DEL15 AS FLOAT)), 4) AS carrier_hist_delay_rate,
        COUNT(*) AS carrier_total_flights
    FROM read_parquet('{PARQUET_IN}')
    WHERE YEAR(CAST(FL_DATE AS DATE)) != 2020
    AND DEP_DEL15 IS NOT NULL
    GROUP BY Reporting_Airline
""").df()

con.close()
print("[ok] All lookup tables ready.\n")

# ── Step 2: Process chunks ────────────────────────────────
print("[start] Processing 2M-row chunks...")
print(f"        Output: {PARQUET_OUT}\n")

writer        = None
total_written = 0
pf            = pq.ParquetFile(PARQUET_IN)

for n, batch in enumerate(pf.iter_batches(batch_size=2_000_000), 1):
    chunk = batch.to_pandas()
    print(f"  Chunk {n}: {len(chunk):,} rows", end=" ... ", flush=True)

    # Convert FL_DATE to string for consistent merging
    chunk["FL_DATE"] = pd.to_datetime(chunk["FL_DATE"]).dt.strftime("%Y-%m-%d")

    # Calendar features
    dt = pd.to_datetime(chunk["FL_DATE"])
    chunk["year"]         = dt.dt.year
    chunk["month"]        = dt.dt.month
    chunk["day_of_month"] = dt.dt.day
    chunk["day_of_week"]  = dt.dt.dayofweek
    chunk["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    chunk["quarter"]      = dt.dt.quarter
    chunk["is_weekend"]   = dt.dt.dayofweek.isin([5,6]).astype(int)

    chunk["season"] = chunk["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Fall",10:"Fall",11:"Fall"
    })

    dep = pd.to_numeric(chunk["CRSDepTime"], errors="coerce").fillna(0).astype(int) // 100
    chunk["dep_hour"] = dep
    bins   = [-1,4,11,16,20,23]
    labels = ["RedEye","Morning","Afternoon","Evening","Night"]
    chunk["time_of_day"] = pd.cut(dep, bins=bins, labels=labels).astype(str)

    hol = pd.Series(0, index=chunk.index)
    for (m, ds, de) in HOLIDAY_PERIODS:
        hol |= ((chunk["month"]==m) & (chunk["day_of_month"]>=ds) & (chunk["day_of_month"]<=de)).astype(int)
    chunk["is_holiday_period"] = hol

    # COVID
    chunk["is_covid_year"] = (chunk["year"] == 2020).astype(int)
    chunk = chunk.merge(
        monthly_vol.rename(columns={"yr":"year","mo":"month"}),
        on=["AIRPORT_CODE","year","month"], how="left"
    )

    # Weather flags
    chunk["is_freezing"]     = (chunk["min_temp_c"] < 0).astype(int)
    chunk["is_extreme_heat"] = (chunk["max_temp_c"] > 38).astype(int)
    chunk["temp_range_c"]    = (chunk["max_temp_c"] - chunk["min_temp_c"]).round(2)

    chunk["wind_category"] = pd.cut(
        chunk["max_wind_ms"].fillna(0),
        bins=[-1,5,10,15,20,9999],
        labels=["Calm","Moderate","Strong","Very Strong","Severe"]
    ).astype(str)
    chunk["is_high_wind"] = (chunk["max_wind_ms"].fillna(0) >= 15).astype(int)

    chunk["visibility_category"] = pd.cut(
        chunk["min_visibility_km"].fillna(99),
        bins=[-1,1.6,5.0,8.0,9999],
        labels=["LIFR","IFR","MVFR","VFR"]
    ).astype(str)
    chunk["is_low_visibility"] = (chunk["min_visibility_km"].fillna(99) < 5.0).astype(int)

    chunk["precip_category"] = pd.cut(
        chunk["total_precip_mm"].fillna(0),
        bins=[-0.001,0,2.5,10,50,9999],
        labels=["None","Light","Moderate","Heavy","Extreme"]
    ).astype(str)
    chunk["is_precipitation"] = (chunk["total_precip_mm"].fillna(0) > 0).astype(int)
    chunk["is_low_ceiling"]   = (chunk["min_ceiling_m"].fillna(9999) < 300).astype(int)

    chunk["weather_severity_score"] = (
        chunk["is_freezing"] + chunk["is_extreme_heat"] +
        chunk["is_high_wind"] + chunk["is_low_visibility"] +
        chunk["is_low_ceiling"]
    )

    # Airport features
    chunk["is_hub"] = chunk["AIRPORT_CODE"].isin(HUB_AIRPORTS).astype(int)
    chunk = chunk.merge(airport_stats, on="AIRPORT_CODE", how="left")
    chunk = chunk.merge(
        daily_counts, left_on=["AIRPORT_CODE","FL_DATE"],
        right_on=["AIRPORT_CODE","FL_DATE_STR"], how="left"
    ).drop(columns=["FL_DATE_STR"], errors="ignore")

    # Route / carrier features
    chunk["distance_bucket"] = pd.cut(
        chunk["Distance"].fillna(0),
        bins=[-1,500,1500,2500,99999],
        labels=["Short","Medium","Long","Ultra"]
    ).astype(str)
    chunk = chunk.merge(route_freq, on=["AIRPORT_CODE","Dest"], how="left")
    chunk = chunk.merge(carrier_stats, on="Reporting_Airline", how="left")

    # Derive DEP_DEL15 / ARR_DEL15 if missing
    if "DEP_DEL15" not in chunk.columns or chunk["DEP_DEL15"].isna().all():
        chunk["DEP_DEL15"] = (chunk["DepDelayMinutes"].fillna(0) >= 15).astype(int)
    if "ARR_DEL15" not in chunk.columns or chunk["ARR_DEL15"].isna().all():
        chunk["ARR_DEL15"] = (chunk["ArrDelayMinutes"].fillna(0) >= 15).astype(int)

    # Drop rows with no target
    chunk = chunk[chunk["DEP_DEL15"].notna()].copy()

    # Write
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(PARQUET_OUT, table.schema, compression="zstd")
    writer.write_table(table)

    total_written += len(chunk)
    print(f"done. Total so far: {total_written:,}")

if writer:
    writer.close()

# Verify
size_mb = os.path.getsize(PARQUET_OUT) / 1e6
meta    = pq.read_metadata(PARQUET_OUT)
print(f"\n[done] features_final.parquet")
print(f"       Rows     : {total_written:,}")
print(f"       Columns  : {meta.num_columns}")
print(f"       File size: {size_mb:.1f} MB")
print(f"\n[next] Upload features_final.parquet to Google Drive and run Phase 6 Colab notebook.")
