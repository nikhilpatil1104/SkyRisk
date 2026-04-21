"""
=============================================================
 Flight Delay Prediction — Phase 2: BTS Data Download
=============================================================
 Project root : D:\\FlightDelayProject
 Output       : D:\\FlightDelayProject\\data\\raw\\bts\\
 Years        : 2015 – 2024
 Airports     : 20 target airports (see AIRPORTS set below)

 What this script does:
   1. Downloads monthly On-Time Performance zip files from BTS
   2. Extracts the CSV inside each zip
   3. Filters to rows where ORIGIN or DEST is one of your 20 airports
   4. Saves filtered CSV per month (keeps disk usage manageable)
   5. Cleans up the raw zip and full CSV after filtering

 Run from anywhere:
   python phase2_bts_download.py

 Dependencies:
   pip install requests tqdm pandas
=============================================================
"""

import os
import zipfile
import requests
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PROJECT_ROOT = Path("D:/project")
RAW_BTS_DIR  = PROJECT_ROOT / "data" / "raw" / "bts"

YEARS  = range(2015, 2025)           # 2015 through 2024 inclusive

# Your 20 target airports
AIRPORTS = {
    # 15 major hubs
    "ATL", "LAX", "ORD", "DFW", "DEN",
    "JFK", "SFO", "CLT", "LAS", "PHX",
    "MIA", "IAH", "SEA", "EWR", "BOS",
    # 5 mid-sized
    "SLC", "SAN", "TPA", "PDX", "AUS",
}

# BTS PREZIP base URL — pattern is consistent across all months/years
BTS_BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Columns to keep — reduces file size significantly
KEEP_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DEP_DEL15",          # 1 if departure delay >= 15 min (your target label)
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ARR_DEL15",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_dirs():
    RAW_BTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output directory ready: {RAW_BTS_DIR}")


def download_and_filter(year: int, month: int) -> dict:
    """
    Downloads the BTS zip for a given year/month, extracts it in memory,
    filters to the 20 target airports, and writes a small filtered CSV.
    Returns a status dict.
    """
    url = BTS_BASE_URL.format(year=year, month=month)
    out_file = RAW_BTS_DIR / f"bts_filtered_{year}_{month:02d}.csv"

    # Skip if already downloaded
    if out_file.exists():
        return {"year": year, "month": month, "status": "skipped (exists)", "rows": None}

    try:
        # Stream download
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        raw_bytes = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            raw_bytes.write(chunk)
        raw_bytes.seek(0)

        # Extract CSV from zip (BTS zips contain exactly one CSV)
        with zipfile.ZipFile(raw_bytes) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file, dtype=str, low_memory=False)

        # Normalize column names (BTS sometimes ships with trailing spaces)
        df.columns = df.columns.str.strip()

        # Filter to your 20 airports
        mask = df["Origin"].isin(AIRPORTS) | df["Dest"].isin(AIRPORTS)
        df_filtered = df[mask]

        # Keep only useful columns (intersect with what's actually in the file)
        available_cols = [c for c in KEEP_COLUMNS if c in df_filtered.columns]
        df_filtered = df_filtered[available_cols]

        # Save
        df_filtered.to_csv(out_file, index=False)

        return {
            "year": year,
            "month": month,
            "status": "ok",
            "rows": len(df_filtered),
            "file": str(out_file),
        }

    except requests.HTTPError as e:
        return {"year": year, "month": month, "status": f"HTTP error: {e}", "rows": None}
    except Exception as e:
        return {"year": year, "month": month, "status": f"error: {e}", "rows": None}


def print_summary(results: list):
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)

    ok       = [r for r in results if r["status"] == "ok"]
    skipped  = [r for r in results if r["status"].startswith("skipped")]
    failed   = [r for r in results if r not in ok and r not in skipped]

    total_rows = sum(r["rows"] for r in ok if r["rows"])

    print(f"  Downloaded & filtered : {len(ok):>4} files")
    print(f"  Skipped (exist)       : {len(skipped):>4} files")
    print(f"  Failed                : {len(failed):>4} files")
    print(f"  Total filtered rows   : {total_rows:,}")
    print(f"  Output directory      : {RAW_BTS_DIR}")

    if failed:
        print("\n  Failed downloads:")
        for r in failed:
            print(f"    {r['year']}-{r['month']:02d} → {r['status']}")

    print("=" * 60)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    make_dirs()

    # Build full list of (year, month) tuples — 120 total
    tasks = [(y, m) for y in YEARS for m in range(1, 13)]

    print(f"\n[start] {len(tasks)} files to process ({min(YEARS)}–{max(YEARS)})")
    print(f"[start] Filtering to {len(AIRPORTS)} airports: {sorted(AIRPORTS)}\n")

    results = []

    for year, month in tqdm(tasks, desc="BTS files", unit="file"):
        result = download_and_filter(year, month)
        results.append(result)

        # Show inline status for each file
        if result["status"] == "ok":
            tqdm.write(f"  ✓ {year}-{month:02d}  →  {result['rows']:,} rows")
        elif result["status"].startswith("skipped"):
            tqdm.write(f"  ↷ {year}-{month:02d}  →  already exists")
        else:
            tqdm.write(f"  ✗ {year}-{month:02d}  →  {result['status']}")

    print_summary(results)


if __name__ == "__main__":
    main()
