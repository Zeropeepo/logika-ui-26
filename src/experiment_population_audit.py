"""
Experiment 1: Population Breakdown & Lowest Fold Robustness Audit
Evaluates all existing models across:
1. Every individual fold KS: min_fold, mean_fold, std_fold.
2. Every population segment KS:
   - Pop 1 (Full-file: repayment=1, account=1)
   - Pop 2 (Thin-file: prev_apps=0, account=0)
   - Pop 3 (Rejected prev apps: prev_apps=1, repayment=0)
Pinpoints exactly where models fail and why certain folds drop.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import polars as pl
import numpy as np
from scipy.stats import ks_2samp

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
    print("AUDIT: FOLD ROBUSTNESS (LOWEST FOLD) & POPULATION SEGMENT PERFORMANCE")
    print("=" * 80)
    
    # 1. Load data & predictions
    df_folds = pl.read_parquet(f"{DATA_PROCESSED}/cv_folds_70k.parquet")
    df_v4 = pl.read_parquet(f"{DATA_PROCESSED}/train_features_v4.parquet").select([
        "target", "cov_has_prev_apps", "cov_has_repayment", "cov_has_account"
    ])
    df_oof_v4 = pl.read_parquet(f"{PREDS_DIR}/oof_v4_all_models.parquet")
    df_oof_gru = pl.read_parquet(f"{PREDS_DIR}/oof_ecom_gru.parquet")
    df_oof_master = pl.read_parquet(f"{PREDS_DIR}/oof_phase4_phase5_master.parquet")
    
    y = df_v4["target"].to_numpy()
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    # Populations
    has_prev = df_v4["cov_has_prev_apps"].to_numpy()
    has_repay = df_v4["cov_has_repayment"].to_numpy()
    has_acct = df_v4["cov_has_account"].to_numpy()
    
    pop_full = (has_repay == 1) & (has_acct == 1)
    pop_thin = (has_prev == 0) & (has_acct == 0)
    pop_rejd = (has_prev == 1) & (has_repay == 0)
    
    print(f"Total rows: {len(y)}")
    print(f"  Pop 1 (Full-file: repayment+account)     : {pop_full.sum():,} rows ({pop_full.mean()*100:.1f}%) | Target rate: {y[pop_full].mean():.3f}")
    print(f"  Pop 2 (Thin-file: NO prev apps, NO acct): {pop_thin.sum():,} rows ({pop_thin.mean()*100:.1f}%) | Target rate: {y[pop_thin].mean():.3f}")
    print(f"  Pop 3 (Rejected prev apps: NO repayment): {pop_rejd.sum():,} rows ({pop_rejd.mean()*100:.1f}%)  | Target rate: {y[pop_rejd].mean():.3f}")
    
    models = {
        "LightGBM": df_oof_v4["oof_lgb"].to_numpy(),
        "XGBoost": df_oof_v4["oof_xgb"].to_numpy(),
        "CatBoost": df_oof_v4["oof_cat"].to_numpy(),
        "LogReg": df_oof_v4["oof_lr"].to_numpy(),
        "Tab-ResNet": df_oof_v4["oof_tab"].to_numpy(),
        "Ecom-GRU": df_oof_gru["oof_gru"].to_numpy(),
        "Cand-A Stacker": df_oof_master["oof_candidate_A_stack"].to_numpy(),
    }
    
    # 2. Compute Fold Metrics
    print("\n" + "=" * 90)
    print(f"{'Model':<16} {'Overall OOF':<12} {'Min Fold':<10} {'Mean Fold':<10} {'Std Fold':<10} {'Fold Scores (0..4)':<35}")
    print("-" * 90)
    
    for name, preds in models.items():
        fold_scores = []
        for f in range(5):
            va = (fold_col == f)
            ks = official_ks_metric(y[va], preds[va])
            fold_scores.append(ks)
            
        ov_ks = official_ks_metric(y, preds)
        min_f = min(fold_scores)
        mean_f = np.mean(fold_scores)
        std_f = np.std(fold_scores)
        f_str = " ".join([f"{s:.4f}" for s in fold_scores])
        print(f"{name:<16} {ov_ks:<12.4f} {min_f:<10.4f} {mean_f:<10.4f} {std_f:<10.4f} [{f_str}]")
        
    # 3. Compute Population KS
    print("\n" + "=" * 80)
    print("PER-POPULATION KS SEPARATION PERFORMANCE")
    print(f"{'Model':<16} {'Pop 1 (Full 66%)':<18} {'Pop 2 (Thin 30%)':<18} {'Pop 3 (Rejd 4%)':<18}")
    print("-" * 80)
    
    for name, preds in models.items():
        ks_p1 = official_ks_metric(y[pop_full], preds[pop_full])
        ks_p2 = official_ks_metric(y[pop_thin], preds[pop_thin])
        ks_p3 = official_ks_metric(y[pop_rejd], preds[pop_rejd])
        print(f"{name:<16} {ks_p1:<18.4f} {ks_p2:<18.4f} {ks_p3:<18.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
