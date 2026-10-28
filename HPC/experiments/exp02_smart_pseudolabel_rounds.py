#!/usr/bin/env python3
"""
EXP 02: Multi-Round Teacher-Student Self-Distillation Pseudo-Labeling
Features:
- Teacher = 0.55677 Anchor (ens_q4cb.csv)
- Round 1: High-confidence soft pseudo-labeling (P >= 0.88 & P <= 0.12)
- Round 2: Consensus pseudo-labeling between Teacher and Student
- LightGBM DART + Sample Weight Calibration
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
    find_file, normalize_rank
)

def run_experiment():
    print("=" * 80)
    print("STARTING EXP 02: MULTI-ROUND SMART PSEUDO-LABEL DISTILLATION")
    print("=" * 80)
    t_start = time.time()
    
    clean_tr, clean_te, y, fold_col, masks = load_clean_dataset()
    anchor_path = find_file("ens_q4cb.csv")
    print(f"[*] Teacher Anchor: {anchor_path}")
    
    teacher_df = pl.read_csv(anchor_path)
    teacher_preds = teacher_df["default_probability"].to_numpy()
    
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"}
    features = [c for c in clean_tr.columns if c not in ignore_cols]
    
    pdf_tr = clean_tr.select(features).to_pandas()
    pdf_te = clean_te.select(features).to_pandas()
    for c in pdf_tr.select_dtypes(include=["object", "string"]).columns:
        pdf_tr[c] = pdf_tr[c].astype("category")
        pdf_te[c] = pdf_te[c].astype("category")
        
    dart_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "dart",
        "learning_rate": 0.04,
        "num_leaves": 22,
        "max_depth": 5,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.45,
        "n_estimators": 500,
        "n_jobs": 8,
        "random_state": 42,
        "verbose": -1
    }
    
    # --- ROUND 1: High Confidence Distillation ---
    print("\n--- ROUND 1: High-Confidence Thresholding (P >= 0.88, P <= 0.12) ---")
    pos_r1 = np.where(teacher_preds >= 0.88)[0]
    neg_r1 = np.where(teacher_preds <= 0.12)[0]
    pseudo_idx_r1 = np.concatenate([pos_r1, neg_r1])
    pseudo_y_r1 = np.zeros(len(pseudo_idx_r1), dtype=np.int32)
    pseudo_y_r1[:len(pos_r1)] = 1
    
    print(f"[*] R1 Confident Test Samples: {len(pseudo_idx_r1):,} ({len(pos_r1):,} pos, {len(neg_r1):,} neg)")
    pdf_pseudo_r1 = pdf_te.iloc[pseudo_idx_r1].copy()
    
    oof_r1 = np.zeros(len(y), dtype=np.float32)
    test_r1 = np.zeros(len(clean_te), dtype=np.float32)
    
    for f in range(5):
        idx_tr = np.where(fold_col != f)[0]
        idx_va = np.where(fold_col == f)[0]
        
        X_aug = pd.concat([pdf_tr.iloc[idx_tr], pdf_pseudo_r1], axis=0)
        y_aug = np.concatenate([y[idx_tr], pseudo_y_r1])
        w_aug = np.concatenate([np.ones(len(idx_tr), dtype=np.float32), np.full(len(pseudo_idx_r1), 0.6, dtype=np.float32)])
        
        clf = lgb.LGBMClassifier(**dart_params)
        clf.fit(X_aug, y_aug, sample_weight=w_aug)
        
        oof_r1[idx_va] = clf.predict_proba(pdf_tr.iloc[idx_va])[:, 1]
        test_r1 += clf.predict_proba(pdf_te)[:, 1] / 5.0
        
    m1 = evaluate_metrics(y, oof_r1, fold_col, masks)
    print(f"[*] Round 1 Student KS: Overall={m1['overall_ks']:.4f} | Min Fold={m1['min_fold_ks']:.4f} | Pop2={m1['pop2_ks']:.4f}")
    
    # --- ROUND 2: Consensus Expansion ---
    print("\n--- ROUND 2: Teacher-Student Consensus Distillation ---")
    r_teach = normalize_rank(teacher_preds)
    r_stud1 = normalize_rank(test_r1)
    consensus = 0.5 * r_teach + 0.5 * r_stud1
    
    pos_r2 = np.where((consensus >= 0.82) & (r_teach >= 0.80))[0]
    neg_r2 = np.where((consensus <= 0.18) & (r_teach <= 0.20))[0]
    pseudo_idx_r2 = np.concatenate([pos_r2, neg_r2])
    pseudo_y_r2 = np.zeros(len(pseudo_idx_r2), dtype=np.int32)
    pseudo_y_r2[:len(pos_r2)] = 1
    
    print(f"[*] R2 Consensus Test Samples: {len(pseudo_idx_r2):,} ({len(pos_r2):,} pos, {len(neg_r2):,} neg)")
    pdf_pseudo_r2 = pdf_te.iloc[pseudo_idx_r2].copy()
    
    oof_r2 = np.zeros(len(y), dtype=np.float32)
    test_r2 = np.zeros(len(clean_te), dtype=np.float32)
    
    for f in range(5):
        idx_tr = np.where(fold_col != f)[0]
        idx_va = np.where(fold_col == f)[0]
        
        X_aug = pd.concat([pdf_tr.iloc[idx_tr], pdf_pseudo_r2], axis=0)
        y_aug = np.concatenate([y[idx_tr], pseudo_y_r2])
        w_aug = np.concatenate([np.ones(len(idx_tr), dtype=np.float32), np.full(len(pseudo_idx_r2), 0.5, dtype=np.float32)])
        
        clf = lgb.LGBMClassifier(**dart_params)
        clf.fit(X_aug, y_aug, sample_weight=w_aug)
        
        oof_r2[idx_va] = clf.predict_proba(pdf_tr.iloc[idx_va])[:, 1]
        test_r2 += clf.predict_proba(pdf_te)[:, 1] / 5.0
        
    m2 = evaluate_metrics(y, oof_r2, fold_col, masks)
    print("\n" + "=" * 80)
    print(f"EXP 02 COMPLETE (Round 2 Consensus): Overall KS={m2['overall_ks']:.4f} | Min Fold={m2['min_fold_ks']:.4f} | Pop1={m2['pop1_ks']:.4f} | Pop2={m2['pop2_ks']:.4f} | Pop3={m2['pop3_ks']:.4f}")
    print(f"Time Taken: {time.time() - t_start:.1f}s")
    print("=" * 80)
    
    os.makedirs("preds", exist_ok=True)
    pl.DataFrame({"application_id": clean_tr["application_id"], "oof_pseudolabel": oof_r2}).write_parquet("preds/oof_exp02_pseudolabel.parquet")
    pl.DataFrame({"application_id": clean_te["application_id"], "test_pseudolabel": test_r2}).write_parquet("preds/test_exp02_pseudolabel.parquet")
    
    os.makedirs("submissions", exist_ok=True)
    p_norm = normalize_rank(test_r2)
    pl.DataFrame({"application_id": clean_te["application_id"], "default_probability": p_norm.astype(np.float64)}).write_csv("submissions/sub_exp02_pseudolabel_r2.csv")
    
    log_experiment_to_csv("EXP-02", "Smart_Pseudolabel_Rounds", "R1(p>0.88)+R2 Consensus Distillation", m2)
    return m2

if __name__ == "__main__":
    run_experiment()
