"""
Phase 4 — CSV Export
Reads bts_weather_joined.parquet and writes bts_weather_joined.csv
Run AFTER phase4 is complete. Uses chunked writing — safe on 16GB RAM.
"""
import pyarrow.parquet as pq
import os

PARQUET_IN = "D:/project/data/processed/bts_weather_joined.parquet"
CSV_OUT    = "D:/project/data/processed/bts_weather_joined.csv"

print(f"[export] Reading: {PARQUET_IN}")
print(f"[export] Writing: {CSV_OUT}")
print("[export] Writing in 1M-row chunks...")

pf      = pq.ParquetFile(PARQUET_IN)
first   = True
written = 0

for batch in pf.iter_batches(batch_size=1_000_000):
    chunk = batch.to_pandas()
    chunk.to_csv(CSV_OUT, mode="w" if first else "a", header=first, index=False)
    first    = False
    written += len(chunk)
    print(f"  Rows written: {written:,}")

size_mb = os.path.getsize(CSV_OUT) / 1e6
print(f"\n[done] bts_weather_joined.csv")
print(f"       Rows      : {written:,}")
print(f"       Size      : {size_mb:.0f} MB")
print(f"       Location  : {CSV_OUT}")
print("\n[note] Parquet remains the input for Phase 5 — CSV is for inspection only.")
