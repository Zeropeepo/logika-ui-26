"""
Evaluate v4 feature set with LightGBM across 5 folds (seed 42)
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import polars as pl
import numpy as np
import lightgbm as lgb
from scipy.stats import ks_2samp
from src.utils_watchdog import cleanup_memory, log_resource_status

EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def main():
    print("Loading v4 train dataset...")
    df_train = pl.read_parquet("data/processed/train_features_v4.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    
    y = df_train["target"].to_numpy()
    feature_cols = [c for c in df_train.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if df_train[c].dtype == pl.String]
    print(f"Total features: {len(feature_cols)}, Categorical: {cat_cols}")
    
    # Cast categoricals
    pdf = df_train.select(feature_cols).to_pandas()
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category")
        
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    oof_preds = np.zeros(len(y), dtype=np.float32)
    fold_ks = []
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 12,
        "max_depth": 5,
        "min_child_samples": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.40,
        "reg_alpha": 10.0,
        "reg_lambda": 50.0,
        "n_estimators": 1000,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    
    t0 = time.time()
    for fold in range(5):
        tr_idx = np.where(fold_col != fold)[0]
        va_idx = np.where(fold_col == fold)[0]
        
        X_tr, y_tr = pdf.iloc[tr_idx], y[tr_idx]
        X_va, y_va = pdf.iloc[va_idx], y[va_idx]
        
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_preds = clf.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = val_preds
        ks = official_ks_metric(y_va, val_preds)
        fold_ks.append(ks)
        print(f"Fold {fold} KS: {ks:.4f} (best iteration: {clf.best_iteration_})")
        cleanup_memory()
        
    overall_ks = official_ks_metric(y, oof_preds)
    mean_ks = np.mean(fold_ks)
    std_ks = np.std(fold_ks)
    print(f"\n--- LIGHTGBM v4 RESULTS ---")
    print(f"Mean KS (5 folds): {mean_ks:.4f} ± {std_ks:.4f} | Overall OOF KS: {overall_ks:.4f} | Time: {time.time() - t0:.1f}s")
    log_resource_status()

if __name__ == "__main__":
    main()
