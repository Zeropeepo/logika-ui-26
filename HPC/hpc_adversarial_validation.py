#!/usr/bin/env python3
"""
HPC Script 1: Adversarial Validation & Temporal Shift Diagnostic
Answers:
1. Is there distribution shift between Train and Test? (AUC > 0.55 = shift)
2. Which features have drifted most?
3. Does customer_id leak or overlap across Train and Test?
4. What is the temporal distribution of decision_datetime?
"""
import os
import sys
import glob
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

def find_data_file(filename):
    candidates = [
        f"data/{filename}",
        f"../data/{filename}",
        f"../data/processed/{filename}",
        f"../data/raw/{filename}",
        f"HPC/data/{filename}",
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not locate {filename} in searched paths: {candidates}")

def main():
    print("=" * 80)
    print("PHASE 1: ADVERSARIAL VALIDATION & DISTRIBUTION SHIFT AUDIT")
    print("=" * 80)

    # 1. Locate features or raw data
    try:
        tr_path = find_data_file("train_features_v5.parquet")
        te_path = find_data_file("test_features_v5.parquet")
    except FileNotFoundError:
        tr_path = find_data_file("train_features_v4.parquet")
        te_path = find_data_file("test_features_v4.parquet")

    print(f"[*] Loading Train: {tr_path}")
    print(f"[*] Loading Test : {te_path}")

    tr_df = pl.read_parquet(tr_path)
    te_df = pl.read_parquet(te_path)

    # Check customer ID overlap
    tr_custs = set(tr_df["customer_id"].to_list())
    te_custs = set(te_df["customer_id"].to_list())
    overlap = tr_custs.intersection(te_custs)
    print(f"\n[*] Unique Train Customers: {len(tr_custs):,}")
    print(f"[*] Unique Test Customers : {len(te_custs):,}")
    print(f"[*] Customer ID Overlap   : {len(overlap):,} ({len(overlap)/len(te_custs)*100:.2f}%)")
    if len(overlap) > 0:
        print("    [!] WARNING: Customer IDs repeat across train/test! GroupKFold required.")
    else:
        print("    [+] Zero customer overlap. Customers are completely distinct.")

    # Check temporal range
    if "decision_datetime" in tr_df.columns:
        tr_d_min, tr_d_max = tr_df["decision_datetime"].min(), tr_df["decision_datetime"].max()
        te_d_min, te_d_max = te_df["decision_datetime"].min(), te_df["decision_datetime"].max()
        print(f"\n[*] Train Decision Date Range: {tr_d_min} to {tr_d_max}")
        print(f"[*] Test Decision Date Range : {te_d_min} to {te_d_max}")

    # Prepare Adversarial dataset
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff", "is_test"}
    feat_cols = [c for c in tr_df.columns if c in te_df.columns and c not in ignore_cols]
    print(f"\n[*] Running Adversarial Validation across {len(feat_cols)} features...")

    # Sample if large
    n_sample = min(25000, len(tr_df), len(te_df))
    pdf_tr = tr_df.sample(n_sample, seed=42).select(feat_cols).to_pandas()
    pdf_te = te_df.sample(n_sample, seed=42).select(feat_cols).to_pandas()

    pdf_tr["is_test"] = 0
    pdf_te["is_test"] = 1

    comb = pd.concat([pdf_tr, pdf_te], axis=0).reset_index(drop=True)
    X = comb[feat_cols]
    y = comb["is_test"].values

    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("category")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_adv = np.zeros(len(y), dtype=np.float32)
    feature_importances = np.zeros(len(feat_cols), dtype=np.float32)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 5,
        "n_estimators": 250,
        "verbose": -1,
        "n_jobs": 4,
        "random_state": 42
    }

    for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X.iloc[idx_tr], y[idx_tr], eval_set=[(X.iloc[idx_va], y[idx_va])], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_adv[idx_va] = clf.predict_proba(X.iloc[idx_va])[:, 1]
        feature_importances += clf.feature_importances_ / 3.0

    adv_auc = roc_auc_score(y, oof_adv)
    print("\n" + "=" * 80)
    print(f"ADVERSARIAL VALIDATION ROC AUC: {adv_auc:.4f}")
    if adv_auc < 0.55:
        print("[+] CONCLUSION: No severe distribution shift between Train and Test.")
    elif adv_auc < 0.65:
        print("[!] CONCLUSION: Mild distribution shift detected. Certain features differ.")
    else:
        print("[!] CRITICAL CONCLUSION: Severe distribution shift between Train and Test!")
    print("=" * 80)

    # Top shifted features
    imp_df = pd.DataFrame({"feature": feat_cols, "importance": feature_importances}).sort_values(by="importance", ascending=False)
    print("\nTop 15 Most Shifted Features (Train vs Test):")
    for idx, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:<38}: {row['importance']:.1f}")

    # Save adversarial weights
    tr_adv_prob = oof_adv[:n_sample]
    # Weight = P(test) / P(train)
    adv_weights = np.clip(tr_adv_prob / (1.0 - tr_adv_prob + 1e-6), 0.1, 10.0)
    print(f"\n[*] Sample re-weighting computed (Mean weight: {adv_weights.mean():.3f})")

if __name__ == "__main__":
    main()
