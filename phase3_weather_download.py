"""
=============================================================
 Flight Delay Prediction — Phase 3: NOAA Weather Download
=============================================================
 Project root : D:\\project
 Output       : D:\\project\\data\\raw\\weather\\
 Years        : 2015 – 2024
 Source       : NOAA Integrated Surface Database (ISD)
                Global Hourly Data

 What this script does:
   1. Downloads hourly weather CSVs for all 20 airports
      from ncei.noaa.gov (one file per station per year)
   2. Keeps only the columns needed for delay prediction
   3. Parses and cleans temperature, wind, visibility,
      precipitation from NOAA's packed format
   4. Saves one clean CSV per airport per year
   5. Fully resumable — skips files already downloaded

 Run:
   python phase3_weather_download.py

 Dependencies:
   pip install requests tqdm pandas
=============================================================
"""

import requests
import pandas as pd
from io import StringIO
from tqdm import tqdm
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PROJECT_ROOT    = Path("D:/project")
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"

YEARS = range(2015, 2025)   # 2015 through 2024

# Station ID → Airport code mapping (verified)
# Format: NOAA station ID (no dash) → IATA code
STATIONS = {
    "72219013874": "ATL",
    "72295023174": "LAX",
    "72530094846": "ORD",
    "72259303985": "DFW",
    "72565003017": "DEN",
    "74486094789": "JFK",
    "72494023234": "SFO",
    "72314013881": "CLT",
    "72386023169": "LAS",
    "72278023183": "PHX",
    "72202012839": "MIA",
    "72243012960": "IAH",
    "72793024233": "SEA",
    "72502014734": "EWR",
    "72509014739": "BOS",
    "72572024127": "SLC",
    "72290023188": "SAN",
    "72211012842": "TPA",
    "72698024229": "PDX",
    "72254013904": "AUS",
}

NOAA_BASE_URL = (
    "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"
)

# ─────────────────────────────────────────────
# NOAA FIELD PARSING
# ─────────────────────────────────────────────
# NOAA packs numeric fields as "VALUE,QUALITY_CODE"
# e.g. TMP = "+0217,1"  means 21.7°C, quality flag 1
# Missing values are coded as +9999 or -9999

def parse_noaa_numeric(series: pd.Series, scale: float = 10.0) -> pd.Series:
    """
    Extracts the numeric value from NOAA packed fields like '+0217,1'.
    Divides by scale (default 10 — most NOAA fields store 1 decimal scaled).
    Replaces 9999 / -9999 sentinel values with NaN.
    """
    extracted = series.astype(str).str.split(",").str[0]
    numeric   = pd.to_numeric(extracted, errors="coerce")
    numeric   = numeric.where(~numeric.abs().eq(9999), other=pd.NA)
    return numeric / scale


def parse_precipitation(series: pd.Series) -> pd.Series:
    """
    AA1 precipitation field format: "depth,condition,quality,period"
    depth is in mm × 10. We extract depth only.
    """
    extracted = series.astype(str).str.split(",").str[0]
    numeric   = pd.to_numeric(extracted, errors="coerce")
    numeric   = numeric.where(~numeric.abs().eq(9999), other=pd.NA)
    return numeric / 10.0   # convert to mm


def clean_weather_df(df: pd.DataFrame, airport_code: str) -> pd.DataFrame:
    """
    Selects and cleans relevant columns from a raw NOAA hourly CSV.
    Returns a tidy dataframe with human-readable column names.
    """
    out = pd.DataFrame()

    # Timestamp — parse to datetime, extract date and hour
    out["DATETIME"]     = pd.to_datetime(df["DATE"], errors="coerce")
    out["FL_DATE"]      = out["DATETIME"].dt.date.astype(str)
    out["HOUR"]         = out["DATETIME"].dt.hour
    out["AIRPORT_CODE"] = airport_code
    out["STATION"]      = df["STATION"]

    # Temperature (°C) — TMP field, scale 10
    if "TMP" in df.columns:
        out["TEMP_C"] = parse_noaa_numeric(df["TMP"], scale=10.0)

    # Dew point (°C) — DEW field, scale 10
    if "DEW" in df.columns:
        out["DEW_POINT_C"] = parse_noaa_numeric(df["DEW"], scale=10.0)

    # Visibility (km) — VIS field, scale 1 (stored in metres, convert to km)
    if "VIS" in df.columns:
        out["VISIBILITY_KM"] = parse_noaa_numeric(df["VIS"], scale=1.0) / 1000.0

    # Wind speed (m/s) — WND field format: "direction,dq,type,speed,sq"
    if "WND" in df.columns:
        wind_speed_raw = df["WND"].astype(str).str.split(",").str[3]
        wind_numeric   = pd.to_numeric(wind_speed_raw, errors="coerce")
        out["WIND_SPEED_MS"] = wind_numeric.where(
            ~wind_numeric.eq(9999), other=pd.NA
        ) / 10.0

        wind_dir_raw = df["WND"].astype(str).str.split(",").str[0]
        wind_dir     = pd.to_numeric(wind_dir_raw, errors="coerce")
        out["WIND_DIR_DEG"] = wind_dir.where(~wind_dir.eq(999), other=pd.NA)

    # Precipitation (mm) — AA1 field
    if "AA1" in df.columns:
        out["PRECIP_MM"] = parse_precipitation(df["AA1"])

    # Sea level pressure (hPa) — SLP field, scale 10
    if "SLP" in df.columns:
        out["PRESSURE_HPA"] = parse_noaa_numeric(df["SLP"], scale=10.0)

    # Sky condition / ceiling (metres) — GA1 field
    # Format: "coverage,cq,height,hq,char,charq"
    if "GA1" in df.columns:
        ceiling_raw = df["GA1"].astype(str).str.split(",").str[2]
        ceiling_num = pd.to_numeric(ceiling_raw, errors="coerce")
        out["CEILING_M"] = ceiling_num.where(~ceiling_num.eq(99999), other=pd.NA)

    # Drop rows where datetime failed to parse
    out = out.dropna(subset=["DATETIME"])
    out = out.drop(columns=["DATETIME"])

    return out


# ─────────────────────────────────────────────
# DOWNLOAD LOGIC
# ─────────────────────────────────────────────

def download_with_progress(url: str, label: str) -> str | None:
    """
    Downloads a URL with a live byte progress bar.
    Returns raw text content or None on failure.
    """
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total     = int(response.headers.get("content-length", 0))
        chunks    = []

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  Downloading {label}",
            leave=False,
            ncols=80,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=65536):
                chunks.append(chunk)
                pbar.update(len(chunk))

        return b"".join(chunks).decode("utf-8", errors="replace")

    except requests.HTTPError as e:
        return None


def download_station_year(station_id: str, airport_code: str, year: int) -> dict:
    label    = f"{airport_code} {year}"
    out_file = RAW_WEATHER_DIR / f"weather_{airport_code}_{year}.csv"

    if out_file.exists():
        return {"label": label, "status": "skipped", "rows": None}

    url = NOAA_BASE_URL.format(year=year, station=station_id)

    try:
        raw_text = download_with_progress(url, label)

        if raw_text is None:
            return {"label": label, "status": "HTTP error (file may not exist)", "rows": None}

        df = pd.read_csv(StringIO(raw_text), dtype=str, low_memory=False)

        if df.empty:
            return {"label": label, "status": "empty file", "rows": 0}

        df_clean = clean_weather_df(df, airport_code)
        df_clean.to_csv(out_file, index=False)

        return {"label": label, "status": "ok", "rows": len(df_clean)}

    except Exception as e:
        return {"label": label, "status": f"error: {e}", "rows": None}


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def print_summary(results: list):
    print("\n" + "=" * 55)
    print("  WEATHER DOWNLOAD SUMMARY")
    print("=" * 55)

    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] not in ("ok", "skipped")]
    total_rows = sum(r["rows"] for r in ok if r["rows"])

    print(f"  Downloaded & cleaned  : {len(ok):>4}")
    print(f"  Skipped (exist)       : {len(skipped):>4}")
    print(f"  Failed                : {len(failed):>4}")
    print(f"  Total cleaned rows    : {total_rows:,}")
    print(f"  Output dir            : {RAW_WEATHER_DIR}")

    if failed:
        print("\n  Failed files:")
        for r in failed:
            print(f"    {r['label']:20s} → {r['status']}")

    print("=" * 55)
    print("\n  Next step: run phase4_spark_eda.py to load and explore")
    print("  both BTS and weather data in PySpark.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output directory ready: {RAW_WEATHER_DIR}\n")

    # Build task list: 20 airports × 10 years = 200 files
    tasks = [
        (station_id, airport_code, year)
        for station_id, airport_code in STATIONS.items()
        for year in YEARS
    ]
    total = len(tasks)

    print(f"[start] {total} files to process (20 airports × 10 years)")
    print(f"[start] Output: one CSV per airport per year\n")

    results = []

    for i, (station_id, airport_code, year) in enumerate(tasks, 1):
        label = f"{airport_code} {year}"
        print(f"[{i:>3}/{total}] {label}", end="  ", flush=True)

        result = download_station_year(station_id, airport_code, year)
        results.append(result)

        if result["status"] == "ok":
            print(f"✓  {result['rows']:,} rows saved")
        elif result["status"] == "skipped":
            print("↷  already exists")
        else:
            print(f"✗  {result['status']}")

    print_summary(results)


if __name__ == "__main__":
    main()
