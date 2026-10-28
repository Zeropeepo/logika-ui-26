import os
import time
import torch
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from dotenv import load_dotenv
load_dotenv()

from tabpfn import TabPFNClassifier

EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def main():
    print("Loading data...")
    tr_df = pl.read_parquet("data/processed/train_features_v3.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    
    # Fold 0
    fold_idx = df_folds["fold_seed42"].to_numpy()
    print(f"Using fold column: {fold_idx.shape}, val count fold 0: {(fold_idx == 0).sum()}")
    
    feature_cols = [c for c in tr_df.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if tr_df[c].dtype == pl.String]
    
    # Convert to pandas
    pdf = tr_df.select(feature_cols + ["target"]).to_pandas()
    # Categorical handling for TabPFN: convert strings to category dtype or codes
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category").cat.codes
        
    y = pdf["target"].to_numpy()
    X = pdf.drop(columns=["target"])
    
    train_mask = (fold_idx != 0)
    val_mask = (fold_idx == 0)
    
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    X_val_full = X[val_mask]
    y_val_full = y[val_mask]
    
    print(f"Full Train shape: {X_train_full.shape}, Full Val shape: {X_val_full.shape}")
    print(f"CUDA memory before test: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    # Test varying N_train
    sample_sizes = [500, 1000, 2500, 5000]
    n_val_test = 2000  # test slice first
    
    for n_samples in sample_sizes:
        print(f"\n--- Benchmark N_train = {n_samples}, N_val = {n_val_test} ---")
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(len(X_train_full), size=n_samples, replace=False)
        X_tr = X_train_full.iloc[idx_sub]
        y_tr = y_train_full[idx_sub]
        
        idx_val = rng.choice(len(X_val_full), size=n_val_test, replace=False)
        X_va = X_val_full.iloc[idx_val]
        y_va = y_val_full[idx_val]
        
        t0 = time.time()
        clf = TabPFNClassifier(device='cuda')
        clf.fit(X_tr, y_tr)
        t_fit = time.time() - t0
        
        t1 = time.time()
        preds = clf.predict_proba(X_va)[:, 1]
        t_pred = time.time() - t1
        
        ks = official_ks_metric(y_va, preds)
        vram_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"Fit time: {t_fit:.2f}s | Pred time: {t_pred:.2f}s | Peak VRAM: {vram_peak:.1f} MB | Val KS: {ks:.4f}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    main()
