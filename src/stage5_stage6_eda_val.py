"""
Stage 5 & Stage 6: Merged Feature Table EDA, Strict Evaluation Schema, and Eval-Only Sources Registry
"""
import os
import json
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

EVAL_ONLY_COLUMNS = [
    "target",
    "application_id",
    "customer_id",
    "application_datetime",
    "decision_datetime",
    "cutoff",
]

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

def run_stage5_6():
    print("Loading train features...")
    df_train = pl.read_parquet("data/processed/train_features.parquet")
    
    y = df_train["target"].to_numpy()
    
    feature_cols = [c for c in df_train.columns if c not in EVAL_ONLY_COLUMNS]
    print(f"Total candidate model features: {len(feature_cols)}")
    
    # 1. Univariate KS & AUC ranking
    print("Computing univariate KS and AUC ranking...")
    results = []
    for col in feature_cols:
        s = df_train[col]
        # Handle string/cat columns by target rate encoding
        if s.dtype == pl.String:
            # Map by train mean
            tm = df_train.group_by(col).agg(pl.col("target").mean().alias("m")).to_dicts()
            td = {d[col]: d["m"] for d in tm}
            vals = s.map_elements(lambda x: td.get(x, 0.1358), return_dtype=pl.Float64).to_numpy()
        else:
            # fillna with median for univariate ranking
            vals = s.to_numpy()
            if np.isnan(vals).any():
                med = float(s.drop_nulls().median()) if len(s.drop_nulls()) > 0 else 0.0
                vals = np.nan_to_num(vals, nan=med)
        
        auc = roc_auc_score(y, vals)
        if auc < 0.5:
            vals_for_ks = -vals
            auc = 1.0 - auc
        else:
            vals_for_ks = vals
            
        ks = official_ks_metric(y, vals_for_ks)
        results.append({
            "feature": col,
            "ks": round(float(ks), 4),
            "auc": round(float(auc), 4),
        })
        
    results = sorted(results, key=lambda x: x["ks"], reverse=True)
    
    # 2. Multicollinearity / Top Correlations
    print("Computing top correlations among top 30 features...")
    top_30_features = [r["feature"] for r in results[:30] if df_train[r["feature"]].dtype != pl.String]
    df_top = df_train.select(top_30_features).to_pandas().fillna(df_train.select(top_30_features).to_pandas().median())
    corr_matrix = df_top.corr().abs()
    
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c_val = corr_matrix.iloc[i, j]
            if c_val >= 0.80:
                high_corr_pairs.append({
                    "feat_1": cols[i],
                    "feat_2": cols[j],
                    "abs_corr": round(float(c_val), 4)
                })
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x["abs_corr"], reverse=True)
    
    # 3. High Risk vs Low Risk Profiles
    # Segment analysis on top signals: bureau_score_mean, annual_interest_rate, repay_max_dpd
    df_p = df_train.select([
        "target", "annual_interest_rate", "bureau_score_mean", "repay_max_dpd",
        "installment_income_ratio", "acct_outflow_inflow_ratio", "ecom_cod_share"
    ]).to_pandas()
    
    bad_mask = df_p["target"] == 1
    good_mask = df_p["target"] == 0
    
    segment_comparison = {
        "bureau_score_mean": {
            "bad_median": float(df_p.loc[bad_mask, "bureau_score_mean"].median()),
            "good_median": float(df_p.loc[good_mask, "bureau_score_mean"].median()),
        },
        "annual_interest_rate": {
            "bad_median": float(df_p.loc[bad_mask, "annual_interest_rate"].median()),
            "good_median": float(df_p.loc[good_mask, "annual_interest_rate"].median()),
        },
        "repay_max_dpd": {
            "bad_median": float(df_p.loc[bad_mask, "repay_max_dpd"].fillna(0).median()),
            "good_median": float(df_p.loc[good_mask, "repay_max_dpd"].fillna(0).median()),
        },
        "installment_income_ratio": {
            "bad_median": float(df_p.loc[bad_mask, "installment_income_ratio"].median()),
            "good_median": float(df_p.loc[good_mask, "installment_income_ratio"].median()),
        },
        "acct_outflow_inflow_ratio": {
            "bad_median": float(df_p.loc[bad_mask, "acct_outflow_inflow_ratio"].fillna(1).median()),
            "good_median": float(df_p.loc[good_mask, "acct_outflow_inflow_ratio"].fillna(1).median()),
        },
        "ecom_cod_share": {
            "bad_median": float(df_p.loc[bad_mask, "ecom_cod_share"].fillna(0).median()),
            "good_median": float(df_p.loc[good_mask, "ecom_cod_share"].fillna(0).median()),
        }
    }
    
    # 4. Save eval_only_sources.md
    with open("reports/eval_only_sources.md", "w") as f:
        f.write("""# Eval-Only Sources & Strict Separation Registry

Kolom-kolom berikut terdaftar dalam `EVAL_ONLY_COLUMNS` dan DILARANG KERAS masuk sebagai fitur numerik atau kategorikal dalam model:

1. `target`: Label ground truth (end-of-day loan DPD >= 90 dalam 12 bulan kalender). Hanya untuk kalkulasi loss/loss gradient dan evaluasi metrik (KS/AUC).
2. `application_id`: Identifier aplikasi (opaque string). Tidak boleh dijadikan fitur model.
3. `customer_id`: Identifier nasabah (opaque string). Tidak boleh dijadikan fitur model.
4. `application_datetime`: Timestamp pengajuan aplikasi. Hanya untuk audit audit & window relative calculation.
5. `decision_datetime`: Cutoff timestamp keputusan kredit.
6. `cutoff`: Helper timestamp cutoff.

Pipeline fitur dan training script memiliki asersi ketat untuk memastikan tidak ada kolom di atas yang masuk ke feature matrix X.
""")
        
    eda_val_summary = {
        "top_20_univariate_signals": results[:20],
        "high_correlation_pairs": high_corr_pairs[:15],
        "segment_comparison": segment_comparison,
        "recommended_cv_scheme": "Stratified 5-Fold Cross Validation (Seed 42) + Isolated 20% Stratified Holdout Set. Reason: Random customer-level split across 2025, zero customer overlap, identical train/test distributions (Adversarial AUC 0.5052)."
    }
    
    with open("reports/stage5_stage6_summary.json", "w") as f:
        json.dump(eda_val_summary, f, indent=2)
    print("Stage 5 & 6 summary saved to reports/stage5_stage6_summary.json")

if __name__ == "__main__":
    run_stage5_6()
