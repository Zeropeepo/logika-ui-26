#!/usr/bin/env python3
"""
EXP 06: Adversarial Sample Re-weighting Pipeline (Covariate Shift Compensation)
Technique:
- Trains discriminator to compute P(test | x)
- Sets sample weight: w_i = clip(P(test | x) / (1 - P(test | x)), 0.1, 8.0)
- Trains LightGBM using sample weights to prioritize test-like applicants
Directly addresses train-test domain discrepancy.
"""
import os
import sys
import time
import polars as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_hpc import (
    load_clean_dataset, evaluate_metrics, log_experiment_to_csv,
    normalize_rank
)

def run_experiment():
    print("=" * 80)
    print("STARTING EXP 06: ADVERSARIAL SAMPLE RE-WEIGHTING PIPELINE")
    print("=" * 80)
    t_start = time.time()
    
    clean_tr, clean_te, y, fold_col, masks = load_clean_dataset()
    
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"}
    features = [c for c in clean_tr.columns if c not in ignore_cols]
    
    pdf_tr = clean_tr.select(features).to_pandas()
    pdf_te = clean_te.select(features).to_pandas()
    for c in pdf_tr.select_dtypes(include=["object", "string"]).columns:
        pdf_tr[c] = pdf_tr[c].astype("category")
        pdf_te[c] = pdf_te[c].astype("category")
        
    # 1. Compute Adversarial Weights
    print("[*] Training Adversarial Domain Discriminator to estimate P(test | x)...")
    df_adv_tr = pdf_tr.sample(25000, random_state=42).copy()
    df_adv_te = pdf_te.sample(25000, random_state=42).copy()
    df_adv_tr["is_test"] = 0
    df_adv_te["is_test"] = 1
    
    comb_adv = pd.concat([df_adv_tr, df_adv_te], axis=0).reset_index(drop=True)
    X_adv = comb_adv[features]
    y_adv = comb_adv["is_test"].values
    
    adv_clf = lgb.LGBMClassifier(
        objective="binary", metric="auc", learning_rate=0.08,
        num_leaves=24, max_depth=5, n_estimators=150,
        n_jobs=8, random_state=42, verbose=-1
    )
    adv_clf.fit(X_adv, y_adv)
    
    # Predict on entire train set
    p_test_tr = adv_clf.predict_proba(pdf_tr)[:, 1]
    # Ratio = P(test) / (1 - P(test))
    # Normalized so mean weight is 1.0
    raw_weights = p_test_tr / (1.0 - p_test_tr + 1e-5)
    sample_weights = np.clip(raw_weights / np.mean(raw_weights), 0.15, 6.0).astype(np.float32)
    print(f"[*] Adversarial Weights: Min={sample_weights.min():.3f} | Mean={sample_weights.mean():.3f} | Max={sample_weights.max():.3f}")
    
    # 2. Train Re-weighted LightGBM
    print("[*] Training 5-Fold LightGBM with Adversarial Sample Re-weighting...")
    oof_adv_lgb = np.zeros(len(y), dtype=np.float32)
    test_adv_lgb = np.zeros(len(clean_te), dtype=np.float32)
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.035,
        "num_leaves": 18,
        "max_depth": 5,
        "min_child_samples": 120,
        "subsample": 0.8,
        "colsample_bytree": 0.40,
        "reg_alpha": 8.0,
        "reg_lambda": 45.0,
        "n_estimators": 750,
        "n_jobs": 8,
        "random_state": 42,
        "verbose": -1
    }
    
    for f in range(5):
        idx_tr = np.where(fold_col != f)[0]
        idx_va = np.where(fold_col == f)[0]
        
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            pdf_tr.iloc[idx_tr], y[idx_tr],
            sample_weight=sample_weights[idx_tr]
        )
        oof_adv_lgb[idx_va] = clf.predict_proba(pdf_tr.iloc[idx_va])[:, 1]
        test_adv_lgb += clf.predict_proba(pdf_te)[:, 1] / 5.0
        
    metrics = evaluate_metrics(y, oof_adv_lgb, fold_col, masks)
    print("\n" + "=" * 80)
    print(f"EXP 06 COMPLETE: Overall KS={metrics['overall_ks']:.4f} | Min Fold={metrics['min_fold_ks']:.4f} | Pop1={metrics['pop1_ks']:.4f} | Pop2={metrics['pop2_ks']:.4f} | Pop3={metrics['pop3_ks']:.4f}")
    print(f"Time Taken: {time.time() - t_start:.1f}s")
    print("=" * 80)
    
    os.makedirs("preds", exist_ok=True)
    pl.DataFrame({"application_id": clean_tr["application_id"], "oof_adv_lgb": oof_adv_lgb}).write_parquet("preds/oof_exp06_adv_reweighted.parquet")
    pl.DataFrame({"application_id": clean_te["application_id"], "test_adv_lgb": test_adv_lgb}).write_parquet("preds/test_exp06_adv_reweighted.parquet")
    
    os.makedirs("submissions", exist_ok=True)
    p_norm = normalize_rank(test_adv_lgb)
    pl.DataFrame({"application_id": clean_te["application_id"], "default_probability": p_norm.astype(np.float64)}).write_csv("submissions/sub_exp06_adv_reweighted.csv")
    
    log_experiment_to_csv("EXP-06", "Adversarial_Sample_Reweighted", "P(test)/P(train) Sample Weighting", metrics)
    return metrics

if __name__ == "__main__":
    run_experiment()
