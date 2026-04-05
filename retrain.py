"""
retrain.py — Standalone retraining script.

Fetches all labeled data from Supabase, re-runs the same feature-engineering
and training pipeline as the notebook (CRISP-DM Phase 4–5), and writes:
  • model.sav              — serialised sklearn Pipeline (joblib)
  • model_metadata.json    — features, threshold, training stats
  • metrics.json           — recall, F1, ROC-AUC, accuracy on held-out test set

Run locally:
    python retrain.py

Run in CI (GitHub Actions):
    SUPABASE_URL=... SUPABASE_KEY=... python retrain.py
"""

import io
import json
import os
import sys
import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_recall_curve, recall_score,
    roc_auc_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from supabase import create_client

# ── Config ─────────────────────────────────────────────────────────────────────
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
RECALL_FLOOR   = 0.75          # minimum fraud recall for threshold search
MODEL_OUT      = "model.sav"
METADATA_OUT   = "model_metadata.json"
METRICS_OUT    = "metrics.json"

CATEGORICAL_FEATURES = [
    "payment_method", "device_type", "gender", "customer_segment",
    "loyalty_tier", "carrier", "shipping_method", "distance_band",
    "ip_country", "shipping_state",
]
NUMERIC_FEATURES = [
    "promo_used", "order_subtotal", "shipping_fee", "tax_amount",
    "risk_score", "customer_is_active", "late_delivery",
    "item_count", "total_items", "avg_unit_price", "unique_products",
    "zip_mismatch", "ip_international", "ip_high_risk_country",
    "order_hour", "order_day_of_week", "order_month",
    "is_weekend", "is_night_order", "order_before_account",
    "account_age_days_capped", "customer_age_years",
    "is_high_value", "promo_high_value", "product_diversity_ratio",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "is_fraud"


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_all(sb, table):
    """Page through a Supabase table and return all rows as a DataFrame."""
    rows, page, page_size = [], 0, 1000
    while True:
        resp = (
            sb.table(table)
            .select("*")
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return pd.DataFrame(rows)


def fetch_and_join(sb):
    """Replicate the notebook's master SQL join using separate Supabase queries."""
    print("Fetching data from Supabase...")
    orders    = _fetch_all(sb, "orders")
    customers = _fetch_all(sb, "customers")
    shipments = _fetch_all(sb, "shipments")
    items     = _fetch_all(sb, "order_items")

    print(f"  orders={len(orders)}, customers={len(customers)}, "
          f"shipments={len(shipments)}, order_items={len(items)}")

    # Aggregate order_items per order
    if not items.empty:
        items_agg = items.groupby("order_id").agg(
            item_count     = ("order_item_id", "count"),
            total_items    = ("quantity",       "sum"),
            avg_unit_price = ("unit_price",     "mean"),
            unique_products = ("product_id",    "nunique"),
        ).reset_index()
    else:
        items_agg = pd.DataFrame(columns=[
            "order_id", "item_count", "total_items",
            "avg_unit_price", "unique_products",
        ])

    df = (
        orders
        .merge(
            customers.rename(columns={
                "gender":           "gender",
                "birthdate":        "birthdate",
                "created_at":       "customer_created_at",
                "customer_segment": "customer_segment",
                "loyalty_tier":     "loyalty_tier",
                "is_active":        "customer_is_active",
            }),
            on="customer_id", how="left", suffixes=("", "_cust")
        )
        .merge(
            shipments.rename(columns={
                "carrier":         "carrier",
                "shipping_method": "shipping_method",
                "distance_band":   "distance_band",
                "late_delivery":   "late_delivery",
            }),
            on="order_id", how="left", suffixes=("", "_ship")
        )
        .merge(items_agg, on="order_id", how="left")
    )

    # Resolve any duplicate column names from suffix collisions
    for base in ["gender", "birthdate", "customer_created_at",
                 "customer_segment", "loyalty_tier", "customer_is_active",
                 "carrier", "shipping_method", "distance_band", "late_delivery"]:
        cust_col = f"{base}_cust"
        ship_col = f"{base}_ship"
        if cust_col in df.columns and base not in df.columns:
            df.rename(columns={cust_col: base}, inplace=True)
        elif ship_col in df.columns and base not in df.columns:
            df.rename(columns={ship_col: base}, inplace=True)

    return df


# ── Feature engineering (mirrors notebook Cell 13) ────────────────────────────

def engineer_features(df):
    df = df.copy()

    df["zip_mismatch"]         = (df["billing_zip"] != df["shipping_zip"]).astype(int)
    df["ip_international"]     = (df["ip_country"] != "US").astype(int)
    df["ip_high_risk_country"] = df["ip_country"].isin(["NG", "IN", "BR"]).astype(int)

    dt      = pd.to_datetime(df["order_datetime"],      errors="coerce")
    cust_dt = pd.to_datetime(df.get("customer_created_at"), errors="coerce")
    birth   = pd.to_datetime(df.get("birthdate"),       errors="coerce")

    df["order_hour"]         = dt.dt.hour
    df["order_day_of_week"]  = dt.dt.dayofweek
    df["order_month"]        = dt.dt.month
    df["is_weekend"]         = (dt.dt.dayofweek >= 5).astype(int)
    df["is_night_order"]     = ((dt.dt.hour >= 23) | (dt.dt.hour <= 4)).astype(int)

    account_age = (dt - cust_dt).dt.days
    df["order_before_account"]    = (account_age < 0).astype(int)
    df["account_age_days_capped"] = account_age.clip(lower=0)

    df["customer_age_years"] = ((dt - birth).dt.days // 365).fillna(30).astype(int)

    order_total = pd.to_numeric(df.get("order_total", 0), errors="coerce").fillna(0)
    df["is_high_value"]    = (order_total > 500).astype(int)
    df["promo_high_value"] = (
        (pd.to_numeric(df["promo_used"], errors="coerce").fillna(0) == 1)
        & (df["is_high_value"] == 1)
    ).astype(int)

    item_count = pd.to_numeric(df["item_count"], errors="coerce").fillna(1).clip(lower=1)
    df["product_diversity_ratio"] = (
        pd.to_numeric(df["unique_products"], errors="coerce").fillna(1) / item_count
    )

    return df


def cap_outliers_iqr(df, cols):
    """Winsorize at 1.5×IQR (mirrors pl.cap_outliers_iqr)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def optimize_threshold(model, X_val, y_val, recall_floor=RECALL_FLOOR):
    """Return threshold maximising F1 subject to Recall >= recall_floor."""
    y_prob = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, y_prob)
    f1s   = 2 * (prec * rec) / (prec + rec + 1e-9)
    valid = rec[:-1] >= recall_floor
    if valid.any():
        idx = f1s[:-1][valid].argmax()
        return float(thresholds[valid][idx])
    return 0.5


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("ERROR: SUPABASE_URL and SUPABASE_KEY must be set.")

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ── 1. Fetch & join ────────────────────────────────────────────────────────
    df_raw = fetch_and_join(sb)

    if TARGET not in df_raw.columns:
        sys.exit(f"ERROR: '{TARGET}' column not found — check orders table schema.")

    df_raw = df_raw.dropna(subset=[TARGET])
    print(f"Labeled rows: {len(df_raw)}  |  Fraud rate: {df_raw[TARGET].mean():.4f}")

    if len(df_raw) < 100:
        sys.exit(f"ERROR: only {len(df_raw)} labeled rows — need at least 100 to retrain.")

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("Engineering features...")
    df = engineer_features(df_raw)
    df = cap_outliers_iqr(df, [
        "order_subtotal", "shipping_fee", "tax_amount", "order_total",
        "avg_unit_price", "account_age_days_capped", "customer_age_years",
        "product_diversity_ratio", "item_count", "total_items", "unique_products",
    ])

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing columns after feature engineering: {missing}")

    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int)

    # ── 3. Train / test split (stratified 75/25 — matches notebook) ────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print("Training Random Forest pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── 5. Threshold optimisation ──────────────────────────────────────────────
    threshold = optimize_threshold(pipeline, X_test, y_test)
    print(f"Optimal threshold: {threshold:.4f}")

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "recall_fraud": round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1_fraud":     round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "roc_auc":      round(roc_auc_score(y_test, y_prob), 4),
        "accuracy":     round(accuracy_score(y_test, y_pred), 4),
    }
    print(f"Metrics: {metrics}")

    # ── 7. Serialise ───────────────────────────────────────────────────────────
    print(f"Saving {MODEL_OUT}...")
    joblib.dump(pipeline, MODEL_OUT)

    metadata = {
        "model_name":       "fraud_detection_pipeline",
        "model_version":    "auto",
        "trained_at_utc":   datetime.datetime.utcnow().isoformat(),
        "source_table":     "orders (Supabase)",
        "num_training_rows": int(len(X_train)),
        "num_test_rows":    int(len(X_test)),
        "features":         ALL_FEATURES,
        "final_threshold":  threshold,
    }
    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {METADATA_OUT}")

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {METRICS_OUT}")

    print("Retraining complete.")
    return metrics


if __name__ == "__main__":
    main()
