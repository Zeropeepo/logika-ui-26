"""
Experiment: Decoupled Population-Specific Expert Models
Decouples applicants into 3 distinct populations to prevent missing-data distortion:
- Pop 1 (Full-file: 46k train / 19.8k test) -> Expert model on full 352 features
- Pop 2 (Thin-file: 21k train / 9k test)    -> Expert model on pure bureau + ecom + demographics (zero empty repayment/account noise)
- Pop 3 (Rejected: 2.7k train / 1.15k test)-> Expert model on prev_apps rejection + bureau + ecom
Evaluates min_fold_ks, mean_fold_ks, and per-population KS.
Generates OOF and test predictions: preds/oof_population_experts.parquet, preds/test_population_experts.parquet
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.utils_watchdog import cleanup_memory, log_resource_status

EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]
DATA_PROCESSED = "data/processed"
PREDS_DIR = "preds"

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return ks_2samp(preds_bad, preds_good).statistic

def main():
    print("=" * 80)
    print("TRAINING POPULATION-SPECIFIC EXPERT MODELS (POPS 1, 2, 3)")
    print("=" * 80)
    log_resource_status()
    
    # 1. Load full datasets
    tr_df = pl.read_parquet(f"{DATA_PROCESSED}/train_features_v4.parquet")
    te_df = pl.read_parquet(f"{DATA_PROCESSED}/test_features_v4.parquet")
    df_folds = pl.read_parquet(f"{DATA_PROCESSED}/cv_folds_70k.parquet")
    
    y = tr_df["target"].to_numpy().astype(np.int32)
    fold_col = df_folds["fold_seed42"].to_numpy()
    test_app_ids = te_df["application_id"].to_list()
    
    # Population masks (Train)
    tr_has_prev = tr_df["cov_has_prev_apps"].to_numpy()
    tr_has_repay = tr_df["cov_has_repayment"].to_numpy()
    tr_has_acct = tr_df["cov_has_account"].to_numpy()
    
    tr_pop1 = (tr_has_repay == 1) & (tr_has_acct == 1)
    tr_pop2 = (tr_has_prev == 0) & (tr_has_acct == 0)
    tr_pop3 = (tr_has_prev == 1) & (tr_has_repay == 0)
    # Remaining small edge cases (e.g. prev=0, acct=1) group into Pop 2
    tr_pop_rest = (~tr_pop1) & (~tr_pop2) & (~tr_pop3)
    tr_pop2 = tr_pop2 | tr_pop_rest
    
    # Population masks (Test)
    te_has_prev = te_df["cov_has_prev_apps"].to_numpy()
    te_has_repay = te_df["cov_has_repayment"].to_numpy()
    te_has_acct = te_df["cov_has_account"].to_numpy()
    
    te_pop1 = (te_has_repay == 1) & (te_has_acct == 1)
    te_pop2 = (te_has_prev == 0) & (te_has_acct == 0)
    te_pop3 = (te_has_prev == 1) & (te_has_repay == 0)
    te_pop_rest = (~te_pop1) & (~te_pop2) & (~te_pop3)
    te_pop2 = te_pop2 | te_pop_rest
    
    print(f"Population breakdown Train: Pop1={tr_pop1.sum():,}, Pop2={tr_pop2.sum():,}, Pop3={tr_pop3.sum():,}")
    print(f"Population breakdown Test : Pop1={te_pop1.sum():,}, Pop2={te_pop2.sum():,}, Pop3={te_pop3.sum():,}")
    
    all_features = [c for c in tr_df.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in all_features if tr_df[c].dtype == pl.String]
    
    # Prepare Pandas
    pdf_tr_all = tr_df.select(all_features).to_pandas()
    pdf_te_all = te_df.select(all_features).to_pandas()
    for c in cat_cols:
        pdf_tr_all[c] = pdf_tr_all[c].astype("category")
        pdf_te_all[c] = pdf_te_all[c].astype("category")
        
    oof_pop_expert = np.zeros(len(y), dtype=np.float32)
    test_pop_expert = np.zeros(len(te_df), dtype=np.float32)
    
    # =========================================================================
    # EXPERT 1: FULL-FILE POPULATION (Pop 1: 46k train / 19.8k test)
    # =========================================================================
    print("\n--- Training Expert Model 1 (Full-File Population) ---")
    t0 = time.time()
    pop1_tr_indices = np.where(tr_pop1)[0]
    pop1_te_indices = np.where(te_pop1)[0]
    
    lgb_params_p1 = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.40,
        "reg_alpha": 10.0,
        "reg_lambda": 50.0,
        "n_estimators": 1000,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    
    p1_test_preds = np.zeros(len(pop1_te_indices), dtype=np.float32)
    for f in range(5):
        tr_fold = np.where(tr_pop1 & (fold_col != f))[0]
        va_fold = np.where(tr_pop1 & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(**lgb_params_p1)
        clf.fit(
            pdf_tr_all.iloc[tr_fold], y[tr_fold],
            eval_set=[(pdf_tr_all.iloc[va_fold], y[va_fold])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        val_pred = clf.predict_proba(pdf_tr_all.iloc[va_fold])[:, 1]
        oof_pop_expert[va_fold] = val_pred
        p1_test_preds += clf.predict_proba(pdf_te_all.iloc[pop1_te_indices])[:, 1] / 5.0
        print(f"  Pop 1 Fold {f} KS: {official_ks_metric(y[va_fold], val_pred):.4f}")
        
    test_pop_expert[pop1_te_indices] = p1_test_preds
    ks_pop1 = official_ks_metric(y[tr_pop1], oof_pop_expert[tr_pop1])
    print(f"--> Expert Model 1 Overall Pop 1 KS: {ks_pop1:.4f} (took {time.time() - t0:.1f}s)")
    
    # =========================================================================
    # EXPERT 2: THIN-FILE POPULATION (Pop 2: 21k train / 9k test)
    # Filter features: ONLY bureau + ecom + demographic + calendar (NO empty repayment/account features)
    # =========================================================================
    print("\n--- Training Expert Model 2 (Thin-File / New-to-Bank Population) ---")
    t0 = time.time()
    pop2_tr_indices = np.where(tr_pop2)[0]
    pop2_te_indices = np.where(te_pop2)[0]
    
    # Clean feature set for Pop 2: exclude repay_*, acct_*, prev_*
    pop2_features = [
        c for c in all_features
        if not c.startswith("repay_") and not c.startswith("acct_") and not c.startswith("prev_")
        and not c.startswith("rank_repay_") and not c.startswith("rank_acct_")
    ]
    print(f"  Pop 2 Clean Feature Count: {len(pop2_features)} (Filtered out {len(all_features) - len(pop2_features)} empty features!)")
    
    lgb_params_p2 = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.04,
        "num_leaves": 10,
        "max_depth": 4,
        "min_child_samples": 120,
        "subsample": 0.8,
        "colsample_bytree": 0.45,
        "reg_alpha": 8.0,
        "reg_lambda": 40.0,
        "n_estimators": 800,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    
    p2_test_preds = np.zeros(len(pop2_te_indices), dtype=np.float32)
    for f in range(5):
        tr_fold = np.where(tr_pop2 & (fold_col != f))[0]
        va_fold = np.where(tr_pop2 & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(**lgb_params_p2)
        clf.fit(
            pdf_tr_all[pop2_features].iloc[tr_fold], y[tr_fold],
            eval_set=[(pdf_tr_all[pop2_features].iloc[va_fold], y[va_fold])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        val_pred = clf.predict_proba(pdf_tr_all[pop2_features].iloc[va_fold])[:, 1]
        oof_pop_expert[va_fold] = val_pred
        p2_test_preds += clf.predict_proba(pdf_te_all[pop2_features].iloc[pop2_te_indices])[:, 1] / 5.0
        print(f"  Pop 2 Fold {f} KS: {official_ks_metric(y[va_fold], val_pred):.4f}")
        
    test_pop_expert[pop2_te_indices] = p2_test_preds
    ks_pop2 = official_ks_metric(y[tr_pop2], oof_pop_expert[tr_pop2])
    print(f"--> Expert Model 2 Overall Pop 2 KS: {ks_pop2:.4f} (took {time.time() - t0:.1f}s)")
    
    # =========================================================================
    # EXPERT 3: REJECTED PREVIOUS APPS (Pop 3: 2.7k train / 1.15k test)
    # High-risk cohort: has account, previous apps, but NO repayment
    # =========================================================================
    print("\n--- Training Expert Model 3 (Rejected Previous Apps Population) ---")
    t0 = time.time()
    pop3_tr_indices = np.where(tr_pop3)[0]
    pop3_te_indices = np.where(te_pop3)[0]
    
    pop3_features = [c for c in all_features if not c.startswith("repay_") and not c.startswith("rank_repay_")]
    
    lgb_params_p3 = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.50,
        "reg_alpha": 5.0,
        "reg_lambda": 30.0,
        "n_estimators": 500,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    
    p3_test_preds = np.zeros(len(pop3_te_indices), dtype=np.float32)
    for f in range(5):
        tr_fold = np.where(tr_pop3 & (fold_col != f))[0]
        va_fold = np.where(tr_pop3 & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(**lgb_params_p3)
        clf.fit(
            pdf_tr_all[pop3_features].iloc[tr_fold], y[tr_fold],
            eval_set=[(pdf_tr_all[pop3_features].iloc[va_fold], y[va_fold])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        val_pred = clf.predict_proba(pdf_tr_all[pop3_features].iloc[va_fold])[:, 1]
        oof_pop_expert[va_fold] = val_pred
        p3_test_preds += clf.predict_proba(pdf_te_all[pop3_features].iloc[pop3_te_indices])[:, 1] / 5.0
        print(f"  Pop 3 Fold {f} KS: {official_ks_metric(y[va_fold], val_pred):.4f}")
        
    test_pop_expert[pop3_te_indices] = p3_test_preds
    ks_pop3 = official_ks_metric(y[tr_pop3], oof_pop_expert[tr_pop3])
    print(f"--> Expert Model 3 Overall Pop 3 KS: {ks_pop3:.4f} (took {time.time() - t0:.1f}s)")
    
    # =========================================================================
    # OVERALL & FOLD ROBUSTNESS EVALUATION
    # =========================================================================
    overall_ks = official_ks_metric(y, oof_pop_expert)
    fold_scores = [official_ks_metric(y[fold_col == f], oof_pop_expert[fold_col == f]) for f in range(5)]
    min_f = min(fold_scores)
    mean_f = np.mean(fold_scores)
    std_f = np.std(fold_scores)
    
    print("\n" + "=" * 80)
    print("POPULATION-EXPERT COMPOSITE RESULTS:")
    print(f"  Overall OOF KS: {overall_ks:.4f}")
    print(f"  Min Fold KS   : {min_f:.4f} (Worst-case fold!)")
    print(f"  Mean Fold KS  : {mean_f:.4f} ± {std_f:.4f}")
    print(f"  Fold Breakdown: {[round(x, 4) for x in fold_scores]}")
    print(f"  Pop 1 KS (66%): {ks_pop1:.4f}")
    print(f"  Pop 2 KS (30%): {ks_pop2:.4f}")
    print(f"  Pop 3 KS ( 4%): {ks_pop3:.4f}")
    print("=" * 80)
    
    # Save OOF and test predictions
    pl.DataFrame({
        "application_id": tr_df["application_id"],
        "oof_pop_expert": oof_pop_expert
    }).write_parquet(f"{PREDS_DIR}/oof_population_experts.parquet")
    
    pl.DataFrame({
        "application_id": te_df["application_id"],
        "test_pop_expert": test_pop_expert
    }).write_parquet(f"{PREDS_DIR}/test_population_experts.parquet")
    print(f"Saved population expert predictions to {PREDS_DIR}/")

if __name__ == "__main__":
    main()
