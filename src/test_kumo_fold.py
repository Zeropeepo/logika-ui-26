"""
Test single fold KumoTabular population forward pass to verify timing and memory.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import torch
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import sdm

from src.utils_watchdog import cleanup_memory, log_resource_status

DATA_PROCESSED = "data/processed"

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return ks_2samp(preds_bad, preds_good).statistic

def test_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tr_df = pl.read_parquet(f"{DATA_PROCESSED}/train_features_v5.parquet")
    df_folds = pl.read_parquet(f"{DATA_PROCESSED}/cv_folds_70k.parquet")
    
    y = tr_df["target"].to_numpy().astype(np.int32)
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    # Population 2 (thin-file: 21,000 rows)
    tr_has_prev = tr_df["cov_has_prev_apps"].to_numpy()
    tr_has_repay = tr_df["cov_has_repayment"].to_numpy()
    tr_has_acct = tr_df["cov_has_account"].to_numpy()
    tr_pop1 = (tr_has_repay == 1) & (tr_has_acct == 1)
    tr_pop3 = (tr_has_prev == 1) & (tr_has_repay == 0)
    tr_pop2 = (~tr_pop1) & (~tr_pop3)
    
    p2_feats = [
        "bureau_score_latest", "bureau_score_mean", "bureau_score_min", "bureau_score_max",
        "bureau_outstanding_balance_latest", "bureau_inquiries_latest", "bureau_inquiries_total",
        "declared_monthly_income", "loan_amount", "proposed_monthly_installment", "annual_interest_rate",
        "age", "employment_tenure_months", "dependents", "housing_status", "loan_purpose",
        "employment_type", "region", "installment_income_ratio", "loan_income_ratio",
        "ecom_discretionary_spend_ratio", "ecom_spend_30d", "ecom_spend_90d", "ecom_comp_orders_all",
        "ecom_gap_mean_days", "ecom_weekend_spend_share", "ecom_share_ewallet", "ecom_share_credit",
        "cross_income_per_dep", "cross_career_stability_ratio", "cross_dsr_squared"
    ]
    # Filter only available feats
    p2_feats = [c for c in p2_feats if c in tr_df.columns]
    print(f"Testing Pop 2 on Fold 0 with {len(p2_feats)} features...")
    
    # Fold 0
    f = 0
    tr_mask = tr_pop2 & (fold_col != f)
    va_mask = tr_pop2 & (fold_col == f)
    
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]
    
    # Context sample of 1200 stratified
    rng = np.random.RandomState(42)
    pos_idx = tr_idx[y[tr_idx] == 1]
    neg_idx = tr_idx[y[tr_idx] == 0]
    
    n_pos = int(1200 * y[tr_idx].mean())
    n_neg = 1200 - n_pos
    ctx_idx = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False)
    ])
    rng.shuffle(ctx_idx)
    
    df_ctx = tr_df[ctx_idx].select(p2_feats + ["target"]).to_pandas().fillna(0)
    df_va = tr_df[va_idx].select(p2_feats).to_pandas().fillna(0)
    
    for c in ["housing_status", "loan_purpose", "employment_type", "region"]:
        if c in df_ctx.columns:
            df_ctx[c] = df_ctx[c].astype(str)
            df_va[c] = df_va[c].astype(str)
            
    t0 = time.time()
    tbl_ctx = sdm.TableTensor.from_pandas(df_ctx, stypes=sdm.infer_stypes(df_ctx, overrides={"target": "categorical"}), device=device)
    tbl_va = sdm.TableTensor.from_pandas(df_va, stypes=sdm.infer_stypes(df_va), device=device)
    
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        model = sdm.models.KumoTabular(task="classification", size="small", device=device)
        model.fit(tbl_ctx.drop_columns("target"), tbl_ctx["target"], num_estimators=2)
        
        chunk_size = 5000
        preds = []
        for i in range(0, len(tbl_va), chunk_size):
            end_i = min(i + chunk_size, len(tbl_va))
            res = model.predict(tbl_va[i:end_i])
            p = res.blocks[sdm.Stype.numerical][:, 1].detach().cpu().numpy()
            preds.append(p)
            
    preds = np.concatenate(preds)
    ks = official_ks_metric(y[va_idx], preds)
    print(f"Pop 2 Fold 0 Kumo KS: {ks:.4f} (took {time.time() - t0:.2f}s)")

if __name__ == "__main__":
    test_run()
