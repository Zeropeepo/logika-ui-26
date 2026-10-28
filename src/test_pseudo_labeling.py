"""
Empirical Evaluation of Pseudo-Labeling & Semi-Supervised Learning
Tests lightweight LightGBM on Fold 0 ground-truth validation across:
1. Baseline: Pure Train (56k rows, 409 features)
2. Hard Pseudo-Labels (Conservative: Top 5% bad, Bottom 20% good)
3. Hard Pseudo-Labels (Aggressive: Top 10% bad, Bottom 30% good)
4. Soft Pseudo-Labels (All 30k test rows, confidence-weighted: w = (2 * |p - 0.5|)^2)
Measures exact impact on validation KS statistic.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ks_2samp

EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def main():
    print("=" * 70)
    print("EMPIRICAL TEST: PSEUDO-LABELING ON MASSIVE V5 FEATURES (LIGHTWEIGHT GBDT)")
    print("=" * 70)
    
    # Load v5 datasets
    df_train = pl.read_parquet("data/processed/train_features_v5.parquet")
    df_test = pl.read_parquet("data/processed/test_features_v5.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    df_test_preds = pl.read_parquet("preds/test_v4_all_models.parquet")
    
    # Build blended test probabilities
    p_test = (
        0.4423 * df_test_preds["test_lr"].to_numpy() +
        0.2981 * df_test_preds["test_lgb"].to_numpy() +
        0.1581 * df_test_preds["test_tab"].to_numpy() +
        0.1003 * df_test_preds["test_xgb"].to_numpy() +
        0.0013 * df_test_preds["test_cat"].to_numpy()
    )
    
    feature_cols = [c for c in df_train.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if df_train[c].dtype == pl.String]
    print(f"Total features in v5: {len(feature_cols)}, Categoricals: {cat_cols}")
    
    # Pandas conversion
    pdf_tr = df_train.select(feature_cols).to_pandas()
    pdf_te = df_test.select(feature_cols).to_pandas()
    for c in cat_cols:
        pdf_tr[c] = pdf_tr[c].astype("category")
        pdf_te[c] = pdf_te[c].astype("category")
        
    y_train = df_train["target"].to_numpy().astype(np.float32)
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    # Fixed Fold 0 split
    tr_idx = np.where(fold_col != 0)[0] # 56,000 rows
    va_idx = np.where(fold_col == 0)[0] # 14,000 rows
    
    X_va = pdf_tr.iloc[va_idx]
    y_va = y_train[va_idx]
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 14,
        "max_depth": 5,
        "min_child_samples": 200,
        "subsample": 0.80,
        "colsample_bytree": 0.40,
        "reg_alpha": 10.0,
        "reg_lambda": 50.0,
        "n_estimators": 500,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    
    results = []
    
    # ---------------------------------------------------------
    # 1. BASELINE: Pure Train Fold (No Pseudo-Labels)
    # ---------------------------------------------------------
    t0 = time.time()
    clf_base = lgb.LGBMClassifier(**lgb_params)
    clf_base.fit(
        pdf_tr.iloc[tr_idx], y_train[tr_idx],
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    preds_base = clf_base.predict_proba(X_va)[:, 1]
    ks_base = official_ks_metric(y_va, preds_base)
    t_base = time.time() - t0
    results.append(("1. Baseline (No Pseudo-Labels)", len(tr_idx), ks_base, 0.0000, t_base))
    print(f"1. Baseline: Train N={len(tr_idx)} | Val KS: {ks_base:.4f} (best iter: {clf_base.best_iteration_}) | {t_base:.1f}s")
    
    # ---------------------------------------------------------
    # 2. HARD PSEUDO-LABELS (Conservative: Top 5% bad, Bottom 20% good)
    # ---------------------------------------------------------
    t0 = time.time()
    tau_bad_c = np.percentile(p_test, 95)   # Top 5%
    tau_good_c = np.percentile(p_test, 20)  # Bottom 20%
    
    mask_bad_c = (p_test >= tau_bad_c)
    mask_good_c = (p_test <= tau_good_c)
    mask_c = mask_bad_c | mask_good_c
    
    X_pseudo_c = pdf_te[mask_c]
    y_pseudo_c = np.where(mask_bad_c[mask_c], 1.0, 0.0).astype(np.float32)
    
    X_tr_c = pd.concat([pdf_tr.iloc[tr_idx], X_pseudo_c], ignore_index=True)
    y_tr_c = np.concatenate([y_train[tr_idx], y_pseudo_c])
    
    clf_c = lgb.LGBMClassifier(**lgb_params)
    clf_c.fit(
        X_tr_c, y_tr_c,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    preds_c = clf_c.predict_proba(X_va)[:, 1]
    ks_c = official_ks_metric(y_va, preds_c)
    t_c = time.time() - t0
    delta_c = ks_c - ks_base
    results.append(("2. Hard Pseudo (Top 5% / Bot 20%)", len(X_tr_c), ks_c, delta_c, t_c))
    print(f"2. Hard Conservative: Added {len(X_pseudo_c)} rows (N={len(X_tr_c)}) | Val KS: {ks_c:.4f} (Δ: {delta_c:+.4f}) | {t_c:.1f}s")
    
    # ---------------------------------------------------------
    # 3. HARD PSEUDO-LABELS (Aggressive: Top 10% bad, Bottom 30% good)
    # ---------------------------------------------------------
    t0 = time.time()
    tau_bad_a = np.percentile(p_test, 90)   # Top 10%
    tau_good_a = np.percentile(p_test, 30)  # Bottom 30%
    
    mask_bad_a = (p_test >= tau_bad_a)
    mask_good_a = (p_test <= tau_good_a)
    mask_a = mask_bad_a | mask_good_a
    
    X_pseudo_a = pdf_te[mask_a]
    y_pseudo_a = np.where(mask_bad_a[mask_a], 1.0, 0.0).astype(np.float32)
    
    X_tr_a = pd.concat([pdf_tr.iloc[tr_idx], X_pseudo_a], ignore_index=True)
    y_tr_a = np.concatenate([y_train[tr_idx], y_pseudo_a])
    
    clf_a = lgb.LGBMClassifier(**lgb_params)
    clf_a.fit(
        X_tr_a, y_tr_a,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    preds_a = clf_a.predict_proba(X_va)[:, 1]
    ks_a = official_ks_metric(y_va, preds_a)
    t_a = time.time() - t0
    delta_a = ks_a - ks_base
    results.append(("3. Hard Aggressive (Top 10% / Bot 30%)", len(X_tr_a), ks_a, delta_a, t_a))
    print(f"3. Hard Aggressive: Added {len(X_pseudo_a)} rows (N={len(X_tr_a)}) | Val KS: {ks_a:.4f} (Δ: {delta_a:+.4f}) | {t_a:.1f}s")
    
    # ---------------------------------------------------------
    # 4. SOFT PSEUDO-LABELS (All 30k test rows, confidence-weighted)
    # ---------------------------------------------------------
    t0 = time.time()
    # Weights: w = (2 * |p - 0.5|)^2, scaled so max weight is 1.0
    w_test = (2.0 * np.abs(p_test - 0.5)) ** 2
    w_train = np.ones(len(tr_idx), dtype=np.float32)
    w_combined = np.concatenate([w_train, w_test]).astype(np.float32)
    
    X_tr_s = pd.concat([pdf_tr.iloc[tr_idx], pdf_te], ignore_index=True)
    y_tr_s = np.concatenate([y_train[tr_idx], p_test]).astype(np.float32)
    
    reg_params = {k: v for k, v in lgb_params.items()}
    clf_s = lgb.LGBMRegressor(**reg_params)
    clf_s.fit(
        X_tr_s, y_tr_s,
        sample_weight=w_combined,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    preds_s = clf_s.predict(X_va)
    ks_s = official_ks_metric(y_va, preds_s)
    t_s = time.time() - t0
    delta_s = ks_s - ks_base
    results.append(("4. Soft Confidence-Weighted (All 30k)", len(X_tr_s), ks_s, delta_s, t_s))
    print(f"4. Soft Weighted: Added {len(pdf_te)} rows (N={len(X_tr_s)}) | Val KS: {ks_s:.4f} (Δ: {delta_s:+.4f}) | {t_s:.1f}s")
    
    print("\n" + "=" * 75)
    print(f"{'Method':<38} {'Train N':<10} {'Val KS':<8} {'Delta KS':<10} {'Time':<6}")
    print("-" * 75)
    for name, n_rows, ks, delta, elap in results:
        sign = "+" if delta >= 0 else ""
        print(f"{name:<38} {n_rows:<10} {ks:<8.4f} {sign}{delta:<9.4f} {elap:<5.1f}s")
    print("=" * 75)

if __name__ == "__main__":
    main()
