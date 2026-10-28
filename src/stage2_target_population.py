"""
Stage 2: Target & Population Analysis, Temporal Split Check, Adversarial Validation, Univariate Signal
"""
import os
import json
import polars as pl
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb

DATA_DIR = "data/raw"

def ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def run_stage2():
    print("Running Stage 2 Analysis...")
    df_train = pl.read_csv(f"{DATA_DIR}/train_applications.csv", try_parse_dates=True)
    df_test = pl.read_csv(f"{DATA_DIR}/test_applications.csv", try_parse_dates=True)
    df_cust = pl.read_csv(f"{DATA_DIR}/customer_profiles.csv")
    
    # 1. Target rate in train
    y_train = df_train["target"].to_numpy()
    target_rate = float(y_train.mean())
    print(f"Train target rate: {target_rate:.4f} ({y_train.sum()} / {len(y_train)})")
    
    # Join with customer_profiles for demographics
    df_train_full = df_train.join(df_cust, on="customer_id", how="left")
    df_test_full = df_test.join(df_cust, on="customer_id", how="left")
    
    # Target breakdown by category
    cat_breakdowns = {}
    for col in ["loan_purpose", "term_months", "region", "employment_type", "housing_status"]:
        agg = df_train_full.group_by(col).agg(
            count = pl.len(),
            target_mean = pl.col("target").mean(),
            target_sum = pl.col("target").sum()
        ).sort(col).to_dicts()
        cat_breakdowns[col] = agg
        
    # 2. Date ranges & Temporal vs Random split check
    train_app_min, train_app_max = str(df_train["application_datetime"].min()), str(df_train["application_datetime"].max())
    train_dec_min, train_dec_max = str(df_train["decision_datetime"].min()), str(df_train["decision_datetime"].max())
    test_app_min, test_app_max = str(df_test["application_datetime"].min()), str(df_test["application_datetime"].max())
    test_dec_min, test_dec_max = str(df_test["decision_datetime"].min()), str(df_test["decision_datetime"].max())
    
    # 3. Decision - Application datetime difference
    train_diff_hours = (df_train["decision_datetime"] - df_train["application_datetime"]).dt.total_hours()
    test_diff_hours = (df_test["decision_datetime"] - df_test["application_datetime"]).dt.total_hours()
    
    diff_stats = {
        "train_min_hours": float(train_diff_hours.min()),
        "train_p50_hours": float(train_diff_hours.median()),
        "train_max_hours": float(train_diff_hours.max()),
        "train_neg_count": int((train_diff_hours < 0).sum()),
        "test_min_hours": float(test_diff_hours.min()),
        "test_p50_hours": float(test_diff_hours.median()),
        "test_max_hours": float(test_diff_hours.max()),
        "test_neg_count": int((test_diff_hours < 0).sum()),
    }
    
    # 4. Amortization formula check
    # P = loan_amount, r = annual_interest_rate / 12, n = term_months
    # installment = P * r * (1+r)^n / ((1+r)^n - 1)
    def calc_amort(df):
        P = df["loan_amount"].cast(pl.Float64)
        r = df["annual_interest_rate"].cast(pl.Float64) / 12.0
        n = df["term_months"].cast(pl.Float64)
        formula = P * (r * (1.0 + r).pow(n)) / ((1.0 + r).pow(n) - 1.0)
        return formula

    calc_inst = calc_amort(df_train)
    actual_inst = df_train["proposed_monthly_installment"].cast(pl.Float64)
    abs_diff = (calc_inst - actual_inst).abs()
    amort_check = {
        "max_abs_diff": float(abs_diff.max()),
        "median_abs_diff": float(abs_diff.median()),
        "within_1_idr": int((abs_diff <= 1.0).sum()),
        "total": len(df_train)
    }
    
    # DTI ratios
    df_train_full = df_train_full.with_columns(
        installment_income_ratio = pl.col("proposed_monthly_installment") / pl.col("declared_monthly_income"),
        loan_income_ratio = pl.col("loan_amount") / pl.col("declared_monthly_income"),
    )
    
    # 5. Adversarial Validation: Train (0) vs Test (1)
    print("Running Adversarial Validation...")
    adv_features = [
        "loan_amount", "annual_interest_rate", "proposed_monthly_installment",
        "loan_purpose", "age", "region", "employment_type", "employment_tenure_months",
        "declared_monthly_income", "dependents", "housing_status"
    ]
    # Encode categoricals for simple lgb
    X_train_adv = df_train_full.select(adv_features).to_pandas()
    X_test_adv = df_test_full.select(adv_features).to_pandas()
    for c in ["loan_purpose", "region", "employment_type", "housing_status"]:
        X_train_adv[c] = X_train_adv[c].astype("category")
        X_test_adv[c] = X_test_adv[c].astype("category")
        
    X_train_adv["is_test"] = 0
    X_test_adv["is_test"] = 1
    
    import pandas as pd
    adv_df = pd.concat([X_train_adv, X_test_adv], ignore_index=True)
    y_adv = adv_df["is_test"].values
    X_adv = adv_df[adv_features]
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    adv_preds = np.zeros(len(y_adv))
    for tr_idx, val_idx in kf.split(X_adv):
        trn_data = lgb.Dataset(X_adv.iloc[tr_idx], label=y_adv[tr_idx])
        val_data = lgb.Dataset(X_adv.iloc[val_idx], label=y_adv[val_idx], reference=trn_data)
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": 42,
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
        bst = lgb.train(params, trn_data, num_boost_round=100)
        adv_preds[val_idx] = bst.predict(X_adv.iloc[val_idx])
        
    adv_auc = float(roc_auc_score(y_adv, adv_preds))
    print(f"Adversarial Validation AUC (train vs test): {adv_auc:.4f}")
    
    # 6. Univariate KS & AUC on train features
    print("Computing univariate KS and AUC on application + profile features...")
    univariate_signals = {}
    for col in adv_features + ["installment_income_ratio", "loan_income_ratio"]:
        if col in ["loan_purpose", "region", "employment_type", "housing_status"]:
            # compute target mean per category to order it, then measure AUC/KS
            mapping = df_train_full.group_by(col).agg(pl.col("target").mean().alias("tm")).to_dicts()
            map_dict = {d[col]: d["tm"] for d in mapping}
            s = df_train_full[col].map_elements(lambda x: map_dict.get(x, 0.0), return_dtype=pl.Float64).to_numpy()
        else:
            s = df_train_full[col].to_numpy()
            
        auc = roc_auc_score(y_train, s)
        if auc < 0.5:
            auc = 1.0 - auc # separation strength
        ks = ks_metric(y_train, s)
        univariate_signals[col] = {
            "auc": round(float(auc), 4),
            "ks": round(float(ks), 4)
        }
    
    # Save Stage 2 results
    stage2_results = {
        "target_rate": target_rate,
        "cat_breakdowns": cat_breakdowns,
        "date_ranges": {
            "train_application_range": [train_app_min, train_app_max],
            "train_decision_range": [train_dec_min, train_dec_max],
            "test_application_range": [test_app_min, test_app_max],
            "test_decision_range": [test_dec_min, test_dec_max],
        },
        "diff_hours": diff_stats,
        "amort_check": amort_check,
        "adversarial_auc": adv_auc,
        "univariate_signals": univariate_signals,
    }
    
    with open("reports/stage2_report.json", "w") as f:
        json.dump(stage2_results, f, indent=2)
    print("Stage 2 report saved to reports/stage2_report.json")

if __name__ == "__main__":
    run_stage2()
