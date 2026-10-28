#!/usr/bin/env python3
"""
EXP 03: Decoupled Population Experts with Specialized Regularization
Features:
- Pop 1 (Full): Deep trees (depth 6) with high L2 lambda to extract rich historical interactions
- Pop 2 (Thin): Clean non-empty features + DART mode to avoid overfitting limited bureau/ecom
- Pop 3 (Rejected): Shallow high-min_child tree to prevent Fold 4 collapse
"""
import os
import sys
import time
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_hpc import (
    load_clean_dataset, evaluate_metrics, log_experiment_to_csv,
    normalize_rank
)

def run_experiment():
    print("=" * 80)
    print("STARTING EXP 03: DECOUPLED POPULATION EXPERTS (SPECIALIZED ARCHITECTURES)")
    print("=" * 80)
    t_start = time.time()
    
    clean_tr, clean_te, y, fold_col, masks = load_clean_dataset()
    
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"}
    all_features = [c for c in clean_tr.columns if c not in ignore_cols]
    
    pdf_tr = clean_tr.select(all_features).to_pandas()
    pdf_te = clean_te.select(all_features).to_pandas()
    for c in pdf_tr.select_dtypes(include=["object", "string"]).columns:
        pdf_tr[c] = pdf_tr[c].astype("category")
        pdf_te[c] = pdf_te[c].astype("category")
        
    oof_expert = np.zeros(len(y), dtype=np.float32)
    test_expert = np.zeros(len(clean_te), dtype=np.float32)
    
    # -------------------------------------------------------------------------
    # POP 1: FULL-FILE EXPERT (Deep regularized GBDT)
    # -------------------------------------------------------------------------
    print("\n--- Training Pop 1 Expert (Deep Regularized GBDT on 46k rows) ---")
    p1_tr_idx = np.where(masks["tr_pop1"])[0]
    p1_te_idx = np.where(masks["te_pop1"])[0]
    
    p1_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 24,
        "max_depth": 6,
        "min_child_samples": 120,
        "subsample": 0.8,
        "colsample_bytree": 0.35,
        "reg_alpha": 10.0,
        "reg_lambda": 60.0,
        "n_estimators": 800,
        "n_jobs": 8,
        "random_state": 42,
        "verbose": -1
    }
    
    for f in range(5):
        tr_f = np.where(masks["tr_pop1"] & (fold_col != f))[0]
        va_f = np.where(masks["tr_pop1"] & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(**p1_params)
        clf.fit(pdf_tr.iloc[tr_f], y[tr_f])
        oof_expert[va_f] = clf.predict_proba(pdf_tr.iloc[va_f])[:, 1]
        test_expert[p1_te_idx] += clf.predict_proba(pdf_te.iloc[p1_te_idx])[:, 1] / 5.0
        
    # -------------------------------------------------------------------------
    # POP 2: THIN-FILE EXPERT (Clean Features + DART mode)
    # -------------------------------------------------------------------------
    pop2_feats = [c for c in all_features if not c.startswith("repay_") and not c.startswith("acct_") and not c.startswith("prev_")]
    print(f"\n--- Training Pop 2 Expert (DART Dropout Trees on {len(pop2_feats)} Clean Features) ---")
    p2_tr_idx = np.where(masks["tr_pop2"])[0]
    p2_te_idx = np.where(masks["te_pop2"])[0]
    
    p2_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "dart",
        "learning_rate": 0.04,
        "num_leaves": 14,
        "max_depth": 4,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.40,
        "reg_alpha": 6.0,
        "reg_lambda": 35.0,
        "n_estimators": 600,
        "n_jobs": 8,
        "random_state": 42,
        "verbose": -1
    }
    
    for f in range(5):
        tr_f = np.where(masks["tr_pop2"] & (fold_col != f))[0]
        va_f = np.where(masks["tr_pop2"] & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(**p2_params)
        clf.fit(pdf_tr[pop2_feats].iloc[tr_f], y[tr_f])
        oof_expert[va_f] = clf.predict_proba(pdf_tr[pop2_feats].iloc[va_f])[:, 1]
        test_expert[p2_te_idx] += clf.predict_proba(pdf_te[pop2_feats].iloc[p2_te_idx])[:, 1] / 5.0
        
    # -------------------------------------------------------------------------
    # POP 3: REJECTED PREVIOUS APPS (Robust Shallow Regularized GBDT)
    # -------------------------------------------------------------------------
    pop3_feats = [c for c in all_features if not c.startswith("repay_")]
    print(f"\n--- Training Pop 3 Expert (Robust Shallow Trees on {len(pop3_feats)} Features) ---")
    p3_tr_idx = np.where(masks["tr_pop3"])[0]
    p3_te_idx = np.where(masks["te_pop3"])[0]
    
    p3_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 50, # High to avoid fold 4 collapse
        "subsample": 0.8,
        "colsample_bytree": 0.45,
        "reg_alpha": 8.0,
        "reg_lambda": 30.0,
        "n_estimators": 400,
        "n_jobs": 8,
        "random_state": 42,
        "verbose": -1
    }
    
    for f in range(5):
        tr_f = np.where(masks["tr_pop3"] & (fold_col != f))[0]
        va_f = np.where(masks["tr_pop3"] & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(**p3_params)
        clf.fit(pdf_tr[pop3_feats].iloc[tr_f], y[tr_f])
        oof_expert[va_f] = clf.predict_proba(pdf_tr[pop3_feats].iloc[va_f])[:, 1]
        test_expert[p3_te_idx] += clf.predict_proba(pdf_te[pop3_feats].iloc[p3_te_idx])[:, 1] / 5.0
        
    metrics = evaluate_metrics(y, oof_expert, fold_col, masks)
    print("\n" + "=" * 80)
    print(f"EXP 03 COMPLETE: Overall KS={metrics['overall_ks']:.4f} | Min Fold={metrics['min_fold_ks']:.4f} | Pop1={metrics['pop1_ks']:.4f} | Pop2={metrics['pop2_ks']:.4f} | Pop3={metrics['pop3_ks']:.4f}")
    print(f"Time Taken: {time.time() - t_start:.1f}s")
    print("=" * 80)
    
    os.makedirs("preds", exist_ok=True)
    pl.DataFrame({"application_id": clean_tr["application_id"], "oof_expert": oof_expert}).write_parquet("preds/oof_exp03_pop_expert.parquet")
    pl.DataFrame({"application_id": clean_te["application_id"], "test_expert": test_expert}).write_parquet("preds/test_exp03_pop_expert.parquet")
    
    os.makedirs("submissions", exist_ok=True)
    p_norm = normalize_rank(test_expert)
    pl.DataFrame({"application_id": clean_te["application_id"], "default_probability": p_norm.astype(np.float64)}).write_csv("submissions/sub_exp03_pop_expert_dart.csv")
    
    log_experiment_to_csv("EXP-03", "Pop_Expert_Specialized_DART", "Pop1 Deep + Pop2 DART + Pop3 High-MinChild", metrics)
    return metrics

if __name__ == "__main__":
    run_experiment()
