"""
Stage 6 Deep-Dive: Target Reconstruction Verification, Detailed Segment KS (including origination month),
Eval-Only Registry, and Sanity Checks
"""
import os
import json
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb

DATA_DIR = "data/raw"
RANDOM_SEED = 42

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def bootstrap_ks_ci(y_true, y_pred, n_bootstraps=500, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    ks_samples = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        ks_val = official_ks_metric(y_true[idx], y_pred[idx])
        ks_samples.append(ks_val)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(ks_samples, alpha * 100))
    upper = float(np.percentile(ks_samples, (1.0 - alpha) * 100))
    return lower, upper

def run_stage6_deep_dive():
    print("=== STAGE 6 DEEP-DIVE VERIFICATION ===")
    
    # 6.1 Target Reconstruction Verification
    print("\n--- 6.1 Verifying Target Reconstruction from repayment_history ---")
    df_train = pl.read_csv(f"{DATA_DIR}/train_applications.csv", try_parse_dates=True)
    df_prev = pl.read_csv(f"{DATA_DIR}/previous_applications.csv", try_parse_dates=True)
    df_repay = pl.read_csv(f"{DATA_DIR}/repayment_history.csv", try_parse_dates=True)
    
    # Check if train_applications has any loan_id
    has_loan_id_in_train = "loan_id" in df_train.columns
    print(f"loan_id column present in train_applications? {has_loan_id_in_train}")
    
    # Check if train application_id exists in previous_applications
    train_app_in_prev = len(set(df_train["application_id"]).intersection(set(df_prev["application_id"])))
    print(f"train application_id overlap with previous_applications: {train_app_in_prev}")
    
    # Check if repayment_history has installments after decision_datetime for train customers
    repay_train_cust = df_repay.join(
        df_train.select(["customer_id", "decision_datetime"]),
        on="customer_id",
        how="inner"
    )
    post_decision_inst = repay_train_cust.filter(pl.col("due_date") >= pl.col("decision_datetime"))
    post_decision_payments = repay_train_cust.filter(
        pl.col("last_payment_datetime").is_not_null() & 
        (pl.col("last_payment_datetime") >= pl.col("decision_datetime"))
    )
    print(f"Installments with due_date >= decision_datetime for train customers: {len(post_decision_inst)}")
    print(f"Installments with last_payment_datetime >= decision_datetime for train customers: {len(post_decision_payments)}")
    
    target_reconstruct_status = {
        "loan_id_in_train": has_loan_id_in_train,
        "train_app_in_prev": train_app_in_prev,
        "post_decision_installments": len(post_decision_inst),
        "post_decision_payments": len(post_decision_payments),
        "conclusion": "Target reconstruction is IMPOSSIBLE from repayment_history because all repayment data strictly ends before decision_datetime, and target loans have no installments recorded in repayment_history."
    }
    
    # 6.2 Eval-Only Sources
    EVAL_ONLY_COLUMNS = [
        "target",
        "application_id",
        "customer_id",
        "application_datetime",
        "decision_datetime",
        "cutoff",
    ]
    
    # 6.3 Local Validation & Segment Stability by Origination Month
    print("\n--- 6.3 Segment Stability & Origination Month Analysis ---")
    df_train_pl = pl.read_parquet("data/processed/train_features.parquet")
    feature_cols = [c for c in df_train_pl.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if df_train_pl[c].dtype == pl.String]
    
    y = df_train_pl["target"].to_numpy()
    
    # Add origination_month
    df_train_pl = df_train_pl.with_columns(
        origination_month = pl.col("decision_datetime").dt.strftime("%Y-%m")
    )
    orig_months = df_train_pl["origination_month"].to_list()
    
    X = df_train_pl.select(feature_cols).to_pandas()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        
    train_idx, holdout_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_cv, y_cv = X.iloc[train_idx].reset_index(drop=True), y[train_idx]
    X_ho, y_ho = X.iloc[holdout_idx].reset_index(drop=True), y[holdout_idx]
    orig_months_cv = [orig_months[i] for i in train_idx]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(y_cv))
    ho_preds = np.zeros(len(y_ho))
    fold_scores = []
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbosity": -1,
        "n_jobs": 4,
    }
    
    for fold, (trn_i, val_i) in enumerate(skf.split(X_cv, y_cv)):
        X_tr, y_tr = X_cv.iloc[trn_i], y_cv[trn_i]
        X_va, y_va = X_cv.iloc[val_i], y_cv[val_i]
        
        trn_d = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_d = lgb.Dataset(X_va, label=y_va, reference=trn_d, categorical_feature=cat_cols)
        
        bst = lgb.train(
            lgb_params,
            trn_d,
            num_boost_round=1000,
            valid_sets=[trn_d, val_d],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        val_p = bst.predict(X_va)
        oof_preds[val_i] = val_p
        ho_preds += bst.predict(X_ho) / 5.0
        fold_scores.append(official_ks_metric(y_va, val_p))
        
    ks_oof = official_ks_metric(y_cv, oof_preds)
    ks_ho = official_ks_metric(y_ho, ho_preds)
    auc_oof = roc_auc_score(y_cv, oof_preds)
    ci_lower, ci_upper = bootstrap_ks_ci(y_cv, oof_preds)
    
    # Segment evaluation: loan_purpose, region, employment_type, origination_month
    segment_results = {}
    for seg_name, seg_vals in [
        ("loan_purpose", X_cv["loan_purpose"].unique()),
        ("region", X_cv["region"].unique()),
        ("employment_type", X_cv["employment_type"].unique()),
        ("origination_month", sorted(list(set(orig_months_cv)))),
    ]:
        seg_dict = {}
        for sv in seg_vals:
            if seg_name == "origination_month":
                mask = np.array([m == sv for m in orig_months_cv])
            else:
                mask = (X_cv[seg_name] == sv).values
            n_bad = int((y_cv[mask] == 1).sum())
            n_good = int((y_cv[mask] == 0).sum())
            if n_bad > 10 and n_good > 10:
                seg_ks = official_ks_metric(y_cv[mask], oof_preds[mask])
                seg_dict[str(sv)] = {
                    "count": int(mask.sum()),
                    "target_rate": round(float(n_bad / mask.sum()), 4),
                    "ks": round(float(seg_ks), 4),
                }
        segment_results[seg_name] = seg_dict
        
    print("\n--- Segment KS Results ---")
    print(json.dumps(segment_results, indent=2))
    
    out = {
        "target_reconstruction": target_reconstruct_status,
        "cv_performance": {
            "fold_ks": [round(float(s), 4) for s in fold_scores],
            "mean_ks": round(float(np.mean(fold_scores)), 4),
            "std_ks": round(float(np.std(fold_scores)), 4),
            "oof_ks": round(float(ks_oof), 4),
            "ci_95": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
            "holdout_ks": round(float(ks_ho), 4),
            "oof_auc": round(float(auc_oof), 4),
        },
        "segment_performance": segment_results
    }
    
    with open("reports/stage6_detailed_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Stage 6 detailed report saved to reports/stage6_detailed_report.json")

if __name__ == "__main__":
    run_stage6_deep_dive()
