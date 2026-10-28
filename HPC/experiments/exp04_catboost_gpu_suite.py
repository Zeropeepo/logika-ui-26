#!/usr/bin/env python3
"""
EXP 04: Heavy CatBoost GPU with Native Categorical Feature Interactions
Powered by NVIDIA A100-SXM4 80GB.
Features:
- task_type='GPU'
- border_count=254 (high precision quantization)
- Native categorical combinations on 2-way interaction strings
- Multi-seed bagging (Seed 42 + Seed 2026)
"""
import os
import sys
import time
import polars as pl
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_hpc import (
    load_clean_dataset, evaluate_metrics, log_experiment_to_csv,
    cleanup_gpu, normalize_rank
)

def run_experiment():
    print("=" * 80)
    print("STARTING EXP 04: HEAVY CATBOOST GPU WITH NATIVE CATEGORICAL INTERACTIONS")
    print("=" * 80)
    t_start = time.time()
    
    clean_tr, clean_te, y, fold_col, masks = load_clean_dataset()
    
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"}
    features = [c for c in clean_tr.columns if c not in ignore_cols]
    
    cat_cols = ["region", "loan_purpose", "employment_type", "housing_status", "cat_region_employment", "cat_purpose_housing"]
    cat_cols = [c for c in cat_cols if c in features]
    
    pdf_tr = clean_tr.select(features).to_pandas().fillna(0)
    pdf_te = clean_te.select(features).to_pandas().fillna(0)
    for c in cat_cols:
        pdf_tr[c] = pdf_tr[c].astype(str)
        pdf_te[c] = pdf_te[c].astype(str)
        
    cb_params = {
        "iterations": 1000,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 25.0,
        "border_count": 254,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "random_seed": 42,
        "verbose": 200
    }
    
    oof_cb = np.zeros(len(y), dtype=np.float32)
    test_cb = np.zeros(len(clean_te), dtype=np.float32)
    seeds = [42, 2026]
    
    for s_idx, seed in enumerate(seeds):
        print(f"\n[*] Running CatBoost GPU with Seed {seed} ({s_idx + 1}/{len(seeds)})...")
        cb_params["random_seed"] = seed
        
        for f in range(5):
            t_f = time.time()
            idx_tr = np.where(fold_col != f)[0]
            idx_va = np.where(fold_col == f)[0]
            
            clf = CatBoostClassifier(**cb_params)
            clf.fit(
                pdf_tr.iloc[idx_tr], y[idx_tr],
                cat_features=cat_cols,
                eval_set=(pdf_tr.iloc[idx_va], y[idx_va]),
                early_stopping_rounds=60,
                verbose=False
            )
            
            oof_cb[idx_va] += clf.predict_proba(pdf_tr.iloc[idx_va])[:, 1] / len(seeds)
            test_cb += clf.predict_proba(pdf_te)[:, 1] / (5.0 * len(seeds))
            cleanup_gpu()
            print(f"    Seed {seed} Fold {f} finished in {time.time() - t_f:.1f}s")
            
    metrics = evaluate_metrics(y, oof_cb, fold_col, masks)
    print("\n" + "=" * 80)
    print(f"EXP 04 COMPLETE: Overall KS={metrics['overall_ks']:.4f} | Min Fold={metrics['min_fold_ks']:.4f} | Pop1={metrics['pop1_ks']:.4f} | Pop2={metrics['pop2_ks']:.4f} | Pop3={metrics['pop3_ks']:.4f}")
    print(f"Time Taken: {time.time() - t_start:.1f}s")
    print("=" * 80)
    
    os.makedirs("preds", exist_ok=True)
    pl.DataFrame({"application_id": clean_tr["application_id"], "oof_catboost": oof_cb}).write_parquet("preds/oof_exp04_catboost_gpu.parquet")
    pl.DataFrame({"application_id": clean_te["application_id"], "test_catboost": test_cb}).write_parquet("preds/test_exp04_catboost_gpu.parquet")
    
    os.makedirs("submissions", exist_ok=True)
    p_norm = normalize_rank(test_cb)
    pl.DataFrame({"application_id": clean_te["application_id"], "default_probability": p_norm.astype(np.float64)}).write_csv("submissions/sub_exp04_catboost_gpu.csv")
    
    log_experiment_to_csv("EXP-04", "CatBoost_GPU_A100_Suite", "2-Seed GPU CatBoost with Native Categoricals", metrics)
    return metrics

if __name__ == "__main__":
    run_experiment()
