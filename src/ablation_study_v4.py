"""
Ablation Study: Feature Engineering Impact Analysis
Evaluates feature families using a fast, lightweight LightGBM model on stratified validation:
1. Full v4 Feature Bank (Reference)
2. Leave-One-Group-Out (LOGO) Ablation:
   - Without Account Slopes & Balance Acceleration
   - Without Bureau Longitudinal Slopes & Limit Cuts
   - Without Repayment Streaks & DPD Recency
   - Without E-commerce Burstiness & Discretionary Spend
   - Without Previous Application Dynamics & Haircuts
   - Without Cross-Table Non-Linear Domain Interactions
Computes exact delta KS for every feature family.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import polars as pl
import numpy as np
import lightgbm as lgb
from scipy.stats import ks_2samp

EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

FEATURE_FAMILIES = {
    "acct_slopes": [
        "acct_slope_closing_balance", "acct_slope_inflows", "acct_slope_outflows",
        "acct_slope_low_balance_days", "acct_slope_net_cashflow", "acct_balance_acceleration",
        "acct_net_cashflow_3m", "acct_net_cashflow_6m", "acct_income_credits_ratio", "acct_min_to_avg_bal_ratio"
    ],
    "bureau_slopes": [
        "bureau_slope_score", "bureau_slope_out_bal", "bureau_slope_util",
        "bureau_score_acceleration", "bureau_limit_cut_flag", "bureau_inquiries_surge_rate", "bureau_inquiries_accel"
    ],
    "repayment_streaks": [
        "repay_late_streak_cutoff", "repay_max_late_streak_all", "repay_days_since_last_30_dpd",
        "repay_days_since_last_60_dpd", "repay_days_since_last_90_dpd", "repay_mean_days_early",
        "repay_partial_payment_ratio_min", "repay_partial_payment_ratio_mean", "repay_dpd_slope", "repay_dpd_recent_vs_all_ratio"
    ],
    "ecom_burstiness": [
        "ecom_gap_skew_days", "ecom_burstiness_metric", "ecom_night_spend_share",
        "ecom_cod_velocity_14d_vs_180d", "ecom_discretionary_spend_ratio"
    ],
    "prev_apps_dynamics": [
        "prev_apps_approval_rate", "prev_apps_rejection_rate", "prev_apps_cancellation_rate",
        "prev_apps_haircut_mean", "prev_days_since_latest_app", "prev_days_since_latest_rejection", "prev_apps_frequency_per_month"
    ],
    "cross_silo_domain": [
        "cross_dti_ratio", "cross_installment_to_net_cashflow", "cross_balance_to_loan",
        "cross_total_leverage_ratio", "cross_coverage_ratio", "cross_rate_bureau_spread"
    ]
}

def evaluate_subset(pdf, feature_subset, cat_cols, y, tr_idx, va_idx):
    active_cats = [c for c in cat_cols if c in feature_subset]
    X_tr = pdf[feature_subset].iloc[tr_idx]
    y_tr = y[tr_idx]
    X_va = pdf[feature_subset].iloc[va_idx]
    y_va = y[va_idx]
    
    clf = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.06,
        num_leaves=12,
        max_depth=4,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=10.0,
        reg_lambda=50.0,
        n_estimators=400,
        random_state=42,
        verbose=-1,
        n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict_proba(X_va)[:, 1]
    return official_ks_metric(y_va, preds)

def main():
    print("=" * 65)
    print("RUNNING FEATURE ENGINEERING ABLATION STUDY (LIGHTWEIGHT MODEL)")
    print("=" * 65)
    
    df_train = pl.read_parquet("data/processed/train_features_v4.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    
    y = df_train["target"].to_numpy().astype(np.int32)
    all_features = [c for c in df_train.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in all_features if df_train[c].dtype == pl.String]
    
    pdf = df_train.select(all_features).to_pandas()
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category")
        
    fold_col = df_folds["fold_seed42"].to_numpy()
    tr_idx = np.where(fold_col != 0)[0]
    va_idx = np.where(fold_col == 0)[0]
    
    # 1. Full v4 Baseline
    t0 = time.time()
    ref_ks = evaluate_subset(pdf, all_features, cat_cols, y, tr_idx, va_idx)
    print(f"FULL v4 Feature Bank (347 features) -> Fold 0 KS: {ref_ks:.4f} (took {time.time() - t0:.1f}s)")
    
    results = [("FULL_v4_ALL", len(all_features), ref_ks, 0.0000)]
    
    print("\n--- Leave-One-Group-Out (LOGO) Ablation ---")
    for group_name, group_cols in FEATURE_FAMILIES.items():
        t_start = time.time()
        # Drop this group
        subset = [c for c in all_features if c not in group_cols]
        ks = evaluate_subset(pdf, subset, cat_cols, y, tr_idx, va_idx)
        delta = ks - ref_ks
        results.append((f"WITHOUT_{group_name}", len(subset), ks, delta))
        print(f"  Without {group_name:<20} ({len(group_cols):>2} feats removed) -> KS: {ks:.4f} | Drop: {delta:+.4f} (took {time.time() - t_start:.1f}s)")
        
    print("\n" + "=" * 65)
    print(f"{'Experiment':<32} {'Feats':<6} {'Val KS':<8} {'Delta':<8}")
    print("-" * 65)
    for name, n_feat, ks, d in results:
        sign = "+" if d >= 0 else ""
        print(f"{name:<32} {n_feat:<6} {ks:<8.4f} {sign}{d:<.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
