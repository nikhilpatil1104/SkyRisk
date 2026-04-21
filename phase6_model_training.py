"""
=============================================================
 Phase 6 — Model Training (Chunked, Low Memory)
=============================================================
 Reads features_final.parquet in 2M-row chunks
 Uses partial_fit for SGD (full data) and a 3M stratified
 sample for Random Forest and GBT.
 Run: python phase6_model_training.py
=============================================================
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model       import SGDClassifier
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.impute             import SimpleImputer
from sklearn.metrics            import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

PARQUET_IN = "D:/project/data/processed/features_final.parquet"
MODELS_DIR = "D:/project/models"
os.makedirs(MODELS_DIR, exist_ok=True)

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
TARGET      = "DEP_DEL15"
CHUNK_SIZE  = 2_000_000
TRAIN_YEARS = list(range(2015, 2022))
TEST_YEARS  = list(range(2022, 2025))

SEP = "=" * 54

def encode_chunk(chunk, encoders):
    for col in CATEGORICAL_COLS:
        if col not in chunk.columns:
            chunk[col] = "Unknown"
        vals  = chunk[col].astype(str).fillna("Unknown")
        known = set(encoders[col].classes_)
        chunk[col] = encoders[col].transform(vals.map(lambda x: x if x in known else "Unknown"))
    return chunk

# ── PASS 1: fit encoders + collect class balance ──────────
print(SEP)
print("  PHASE 6 — MODEL TRAINING")
print(SEP)
print()
print("[pass 1/3] Scanning training data for encoders...")

encoders    = {col: LabelEncoder() for col in CATEGORICAL_COLS}
all_cats    = {col: set() for col in CATEGORICAL_COLS}
y_sample    = []
total_train = 0

pf = pq.ParquetFile(PARQUET_IN)
for n, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE), 1):
    chunk = batch.to_pandas()
    chunk["year"] = pd.to_datetime(chunk["FL_DATE"]).dt.year
    chunk = chunk[chunk["year"].isin(TRAIN_YEARS) & chunk[TARGET].notna()]
    if len(chunk) == 0:
        continue
    for col in CATEGORICAL_COLS:
        all_cats[col].update(chunk[col].astype(str).fillna("Unknown").unique())
    if len(y_sample) < 500_000:
        y_sample.extend(chunk[TARGET].astype(int).tolist())
    total_train += len(chunk)
    print(f"  Chunk {n}: {len(chunk):,} train rows  (total: {total_train:,})")

for col in CATEGORICAL_COLS:
    encoders[col].fit(sorted(list(all_cats[col])) + ["Unknown"])

class_weights = compute_class_weight("balanced", classes=np.array([0,1]), y=np.array(y_sample))
weight_dict   = {0: class_weights[0], 1: class_weights[1]}
del y_sample

print(f"  Train rows   : {total_train:,}")
print(f"  Class weights: 0={weight_dict[0]:.3f}  1={weight_dict[1]:.3f}")

# ── PASS 2: fit imputer + scaler on 4M-row sample ────────
print()
print("[pass 2/3] Fitting imputer and scaler...")

sample_chunks = []
sample_count  = 0
pf = pq.ParquetFile(PARQUET_IN)
for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
    chunk = batch.to_pandas()
    chunk["year"] = pd.to_datetime(chunk["FL_DATE"]).dt.year
    chunk = chunk[chunk["year"].isin(TRAIN_YEARS) & chunk[TARGET].notna()].copy()
    if len(chunk) == 0:
        continue
    chunk = encode_chunk(chunk, encoders)
    sample_chunks.append(chunk)
    sample_count += len(chunk)
    if sample_count >= 4_000_000:
        break

sample_df    = pd.concat(sample_chunks, ignore_index=True)
ALL_FEATURES = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c in sample_df.columns]
X_sample     = sample_df[ALL_FEATURES].values.astype(float)

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
imputer.fit(X_sample)
scaler.fit(imputer.transform(X_sample))
del sample_df, sample_chunks, X_sample
print(f"  Done. Features: {len(ALL_FEATURES)}")

# ── PASS 3: Train models ──────────────────────────────────
print()
print("[pass 3/3] Training models...")

sgd = SGDClassifier(loss="log_loss", class_weight=weight_dict,
                    max_iter=1, random_state=42, n_jobs=-1)

rf_X, rf_y = [], []
RF_SAMPLE_TARGET = 3_000_000

pf = pq.ParquetFile(PARQUET_IN)
for n, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE), 1):
    chunk = batch.to_pandas()
    chunk["year"] = pd.to_datetime(chunk["FL_DATE"]).dt.year
    train = chunk[chunk["year"].isin(TRAIN_YEARS) & chunk[TARGET].notna()].copy()
    if len(train) == 0:
        continue

    train    = encode_chunk(train, encoders)
    X        = imputer.transform(train[ALL_FEATURES].values.astype(float))
    y        = train[TARGET].astype(int).values
    X_scaled = scaler.transform(X)

    sgd.partial_fit(X_scaled, y, classes=np.array([0,1]))

    current_sample = sum(len(a) for a in rf_X)
    if current_sample < RF_SAMPLE_TARGET:
        n_take = min(len(X), int(len(X) * 0.20))
        idx    = np.random.choice(len(X), size=n_take, replace=False)
        rf_X.append(X[idx])
        rf_y.append(y[idx])

    print(f"  Chunk {n}: SGD trained | RF sample: {sum(len(a) for a in rf_X):,} rows")

print()
joblib.dump(sgd, f"{MODELS_DIR}/sgd_logistic.pkl")
print("  SGD Logistic Regression saved.")

X_rf = np.vstack(rf_X)
y_rf = np.concatenate(rf_y)
sw   = np.where(y_rf == 1, weight_dict[1], weight_dict[0])
print(f"  RF/GBT sample size: {len(X_rf):,} rows")

print()
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                             max_features="sqrt", class_weight=weight_dict,
                             n_jobs=-1, random_state=42)
rf.fit(X_rf, y_rf)
joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
print("  Random Forest saved.")

print()
print("  Training Gradient Boosted Trees...")
gbt = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, min_samples_leaf=10, random_state=42)
gbt.fit(X_rf, y_rf, sample_weight=sw)
joblib.dump(gbt, f"{MODELS_DIR}/gbt.pkl")
print("  GBT saved.")

# ── Evaluation on test set (chunked) ─────────────────────
print()
print("[eval] Evaluating on test set (2022-2024)...")

y_true_all, airports_all               = [], []
sgd_probs, sgd_preds                   = [], []
rf_probs,  rf_preds                    = [], []
gbt_probs, gbt_preds                   = [], []

pf = pq.ParquetFile(PARQUET_IN)
for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
    chunk = batch.to_pandas()
    chunk["year"] = pd.to_datetime(chunk["FL_DATE"]).dt.year
    test = chunk[chunk["year"].isin(TEST_YEARS) & chunk[TARGET].notna()].copy()
    if len(test) == 0:
        continue

    test     = encode_chunk(test, encoders)
    X_imp    = imputer.transform(test[ALL_FEATURES].values.astype(float))
    X_scaled = scaler.transform(X_imp)
    y_test   = test[TARGET].astype(int).values

    y_true_all.extend(y_test.tolist())
    airports_all.extend(test["AIRPORT_CODE"].tolist())

    sgd_probs.extend(sgd.predict_proba(X_scaled)[:, 1].tolist())
    sgd_preds.extend(sgd.predict(X_scaled).tolist())
    rf_probs.extend(rf.predict_proba(X_imp)[:, 1].tolist())
    rf_preds.extend(rf.predict(X_imp).tolist())
    gbt_probs.extend(gbt.predict_proba(X_imp)[:, 1].tolist())
    gbt_preds.extend(gbt.predict(X_imp).tolist())
    print(f"  Evaluated chunk — test rows so far: {len(y_true_all):,}")

y_true = np.array(y_true_all)

def evaluate(y_true, y_pred, y_prob, name):
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print()
    print(SEP)
    print("  " + name)
    print(SEP)
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Confusion Matrix (Test 2022-2024):")
    print(f"  {'':22s}  Pred 0       Pred 1")
    print(f"  {'Actual 0 (On-time)':<22s}  {tn:>10,}   {fp:>10,}")
    print(f"  {'Actual 1 (Delayed)':<22s}  {fn:>10,}   {tp:>10,}")
    print(SEP)
    return {"model": name, "auc": round(auc,4), "f1": round(f1,4),
            "precision": round(prec,4), "recall": round(rec,4)}

results = [
    evaluate(y_true, np.array(sgd_preds), np.array(sgd_probs), "SGD Logistic Regression"),
    evaluate(y_true, np.array(rf_preds),  np.array(rf_probs),  "Random Forest"),
    evaluate(y_true, np.array(gbt_preds), np.array(gbt_probs), "Gradient Boosted Trees"),
]

comparison = pd.DataFrame(results).sort_values("auc", ascending=False)
print()
print(SEP)
print("  FINAL COMPARISON — Test Set (2022-2024)")
print(SEP)
print(comparison.to_string(index=False))
print(SEP)

# Per-airport breakdown
airport_df = pd.DataFrame({
    "AIRPORT_CODE": airports_all,
    "actual":       y_true,
    "predicted":    np.array(gbt_preds),
    "correct":      (y_true == np.array(gbt_preds)).astype(int)
})
per_airport = (
    airport_df.groupby("AIRPORT_CODE")
    .agg(
        flights             = ("actual",    "count"),
        actual_delay_pct    = ("actual",    lambda x: round(x.mean()*100, 1)),
        predicted_delay_pct = ("predicted", lambda x: round(x.mean()*100, 1)),
        accuracy_pct        = ("correct",   lambda x: round(x.mean()*100, 1))
    )
    .sort_values("actual_delay_pct", ascending=False)
)
print()
print("Per-airport performance (GBT):")
print(per_airport.to_string())

comparison.to_csv(f"{MODELS_DIR}/model_results.csv", index=False)
per_airport.to_csv(f"{MODELS_DIR}/per_airport_results.csv")

print()
print(SEP)
print("  PHASE 6 COMPLETE")
print(SEP)
print(f"  Models     : {MODELS_DIR}")
print(f"  Test rows  : {len(y_true):,}")
print(f"  Best model : {comparison.iloc[0]['model']}")
print(f"  Best AUC   : {comparison.iloc[0]['auc']:.4f}")
print(SEP)
