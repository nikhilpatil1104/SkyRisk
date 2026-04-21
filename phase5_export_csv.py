"""
Phase 5 — CSV Export
Reads features_final.parquet and writes features_final.csv
Run AFTER phase5_rebuild.py completes. Chunked — safe on 16GB RAM.
"""
import pyarrow.parquet as pq
import os

PARQUET_IN = "D:/project/data/processed/features_final.parquet"
CSV_OUT    = "D:/project/data/processed/features_final.csv"

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
print(f"\n[done] features_final.csv")
print(f"       Rows      : {written:,}")
print(f"       Size      : {size_mb:.0f} MB")
print(f"       Location  : {CSV_OUT}")
print("\n[note] Parquet remains the input for Phase 6 — CSV is for inspection only.")
