"""
Kumo Tabular Population Foundation Inference (exp067_kumo_fold_population)
Runs NVIDIA's Kumo Tabular Foundation Model (sdm.models.KumoTabular) on each population segment:
- Pop 1 (Full-file): In-context forward pass on 19.8k test rows using Pop 1 context
- Pop 2 (Thin-file): In-context forward pass on 9.0k test rows using Pop 2 context (pure bureau+ecom+demographics)
- Pop 3 (Rejected): In-context forward pass on 1.15k test rows using Pop 3 context
Combines all 30,000 test predictions into preds/test_kumo_population.parquet.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import torch
import polars as pl
import numpy as np
import pandas as pd
import sdm

from src.utils_watchdog import cleanup_memory, log_resource_status

DATA_PROCESSED = "data/processed"
PREDS_DIR = "preds"

# Top representative domain features identified from Tab-FM profiling
KUMO_FEATURES_POP1 = [
    "repay_max_max_dpd_all", "repay_max_max_dpd", "repay_weighted_underpaid", "repay_inst_total",
    "bureau_life_score_latest", "bureau_life_score_mean", "bureau_life_out_bal_latest",
    "bureau_life_inquiries_latest", "bureau_score_drop_from_max", "acct_mean_balance_all",
    "acct_mean_inflows_all", "acct_mean_outflows_all", "income_credits_to_declared_ratio",
    "ecom_discretionary_spend_ratio", "ecom_spend_30d", "ecom_comp_orders_all", "ecom_gap_mean_days",
    "declared_monthly_income", "loan_amount", "proposed_monthly_installment", "annual_interest_rate",
    "age", "employment_tenure_months", "dependents", "disposable_income", "rate_score_ratio"
]

KUMO_FEATURES_POP2 = [
    "bureau_life_score_latest", "bureau_life_score_mean", "bureau_life_out_bal_latest",
    "bureau_life_inquiries_latest", "bureau_score_drop_from_max", "bureau_latest_score_Bureau_A",
    "bureau_latest_score_Bureau_B", "ecom_discretionary_spend_ratio", "ecom_spend_30d",
    "ecom_spend_90d", "ecom_comp_orders_all", "ecom_gap_mean_days", "ecom_gap_std_days",
    "declared_monthly_income", "loan_amount", "proposed_monthly_installment", "annual_interest_rate",
    "age", "employment_tenure_months", "dependents", "loan_purpose", "employment_type", "region"
]

KUMO_FEATURES_POP3 = [
    "prev_apps_approval_rate", "prev_apps_rejection_rate", "prev_days_since_latest_app",
    "bureau_life_score_latest", "bureau_life_out_bal_latest", "acct_mean_balance_all",
    "acct_mean_inflows_all", "ecom_spend_30d", "ecom_discretionary_spend_ratio",
    "declared_monthly_income", "loan_amount", "proposed_monthly_installment", "annual_interest_rate",
    "age", "employment_tenure_months", "dependents"
]

def run_kumo_inference():
    print("=" * 80)
    print("RUNNING KUMO TABULAR POPULATION INFERENCE (NVIDIA FOUNDATION MODEL)")
    print("=" * 80)
    log_resource_status()
    
    tr_df = pl.read_parquet(f"{DATA_PROCESSED}/train_features_v4.parquet")
    te_df = pl.read_parquet(f"{DATA_PROCESSED}/test_features_v4.parquet")
    
    # Populations
    tr_pop1 = ((tr_df["cov_has_repayment"] == 1) & (tr_df["cov_has_account"] == 1)).to_numpy()
    tr_pop2 = ((tr_df["cov_has_prev_apps"] == 0) & (tr_df["cov_has_account"] == 0)).to_numpy()
    tr_pop3 = ((tr_df["cov_has_prev_apps"] == 1) & (tr_df["cov_has_repayment"] == 0)).to_numpy()
    tr_pop2 = tr_pop2 | ((~tr_pop1) & (~tr_pop2) & (~tr_pop3))
    
    te_pop1 = ((te_df["cov_has_repayment"] == 1) & (te_df["cov_has_account"] == 1)).to_numpy()
    te_pop2 = ((te_df["cov_has_prev_apps"] == 0) & (te_df["cov_has_account"] == 0)).to_numpy()
    te_pop3 = ((te_df["cov_has_prev_apps"] == 1) & (te_df["cov_has_repayment"] == 0)).to_numpy()
    te_pop2 = te_pop2 | ((~te_pop1) & (~te_pop2) & (~te_pop3))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    test_kumo_probs = np.zeros(len(te_df), dtype=np.float32)
    
    # -------------------------------------------------------------------------
    # POPULATION 1 (FULL-FILE: 19,845 test rows)
    # -------------------------------------------------------------------------
    print("\n--- Running Kumo Tabular on Population 1 (Full-File: 19,845 rows) ---")
    t0 = time.time()
    
    # Select stratified context sample (1500 rows)
    rng = np.random.RandomState(42)
    p1_tr_idx = np.where(tr_pop1)[0]
    p1_ctx_idx = rng.choice(p1_tr_idx, size=1500, replace=False)
    p1_te_idx = np.where(te_pop1)[0]
    
    df_ctx1 = tr_df[p1_ctx_idx].select(KUMO_FEATURES_POP1 + ["target"]).to_pandas().fillna(0)
    df_que1 = te_df[p1_te_idx].select(KUMO_FEATURES_POP1).to_pandas().fillna(0)
    df_que1["target"] = 0 # dummy
    
    tbl_ctx1 = sdm.TableTensor.from_pandas(df_ctx1, stypes=sdm.infer_stypes(df_ctx1, overrides={"target": "categorical"}), device=device)
    tbl_que1 = sdm.TableTensor.from_pandas(df_que1, stypes=sdm.infer_stypes(df_que1, overrides={"target": "categorical"}), device=device)
    
    model = sdm.models.KumoTabular(task="classification", size="small", device=device)
    
    # Batch query in chunks of 5000 to keep VRAM clean
    chunk_size = 5000
    p1_probs = []
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for i in range(0, len(tbl_que1), chunk_size):
            end_i = min(i + chunk_size, len(tbl_que1))
            res = model(
                x_context=tbl_ctx1.drop_columns("target"),
                y_context=tbl_ctx1["target"],
                x_query=tbl_que1[i:end_i].drop_columns("target"),
                num_estimators=2
            )
            # Extract class 1 prob
            p = res.blocks[sdm.Stype.numerical][:, 1].detach().cpu().numpy()
            p1_probs.append(p)
            cleanup_memory()
            
    test_kumo_probs[p1_te_idx] = np.concatenate(p1_probs)
    print(f"Pop 1 completed in {time.time() - t0:.1f}s | Probs range: [{test_kumo_probs[p1_te_idx].min():.4f}, {test_kumo_probs[p1_te_idx].max():.4f}]")
    del tbl_ctx1, tbl_que1, df_ctx1, df_que1; cleanup_memory()
    
    # -------------------------------------------------------------------------
    # POPULATION 2 (THIN-FILE: 9,000 test rows)
    # -------------------------------------------------------------------------
    print("\n--- Running Kumo Tabular on Population 2 (Thin-File: 9,000 rows) ---")
    t0 = time.time()
    
    p2_tr_idx = np.where(tr_pop2)[0]
    p2_ctx_idx = rng.choice(p2_tr_idx, size=1500, replace=False)
    p2_te_idx = np.where(te_pop2)[0]
    
    # Fill categoricals with string
    df_ctx2 = tr_df[p2_ctx_idx].select(KUMO_FEATURES_POP2 + ["target"]).to_pandas().fillna(0)
    df_que2 = te_df[p2_te_idx].select(KUMO_FEATURES_POP2).to_pandas().fillna(0)
    df_que2["target"] = 0
    
    # String fill
    for c in ["loan_purpose", "employment_type", "region"]:
        if c in df_ctx2.columns:
            df_ctx2[c] = df_ctx2[c].astype(str)
            df_que2[c] = df_que2[c].astype(str)
            
    tbl_ctx2 = sdm.TableTensor.from_pandas(df_ctx2, stypes=sdm.infer_stypes(df_ctx2, overrides={"target": "categorical"}), device=device)
    tbl_que2 = sdm.TableTensor.from_pandas(df_que2, stypes=sdm.infer_stypes(df_que2, overrides={"target": "categorical"}), device=device)
    
    p2_probs = []
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for i in range(0, len(tbl_que2), chunk_size):
            end_i = min(i + chunk_size, len(tbl_que2))
            res = model(
                x_context=tbl_ctx2.drop_columns("target"),
                y_context=tbl_ctx2["target"],
                x_query=tbl_que2[i:end_i].drop_columns("target"),
                num_estimators=2
            )
            p = res.blocks[sdm.Stype.numerical][:, 1].detach().cpu().numpy()
            p2_probs.append(p)
            cleanup_memory()
            
    test_kumo_probs[p2_te_idx] = np.concatenate(p2_probs)
    print(f"Pop 2 completed in {time.time() - t0:.1f}s | Probs range: [{test_kumo_probs[p2_te_idx].min():.4f}, {test_kumo_probs[p2_te_idx].max():.4f}]")
    del tbl_ctx2, tbl_que2, df_ctx2, df_que2; cleanup_memory()
    
    # -------------------------------------------------------------------------
    # POPULATION 3 (REJECTED PREV APPS: 1,155 test rows)
    # -------------------------------------------------------------------------
    print("\n--- Running Kumo Tabular on Population 3 (Rejected Apps: 1,155 rows) ---")
    t0 = time.time()
    
    p3_tr_idx = np.where(tr_pop3)[0]
    p3_ctx_idx = rng.choice(p3_tr_idx, size=min(1000, len(p3_tr_idx)), replace=False)
    p3_te_idx = np.where(te_pop3)[0]
    
    df_ctx3 = tr_df[p3_ctx_idx].select(KUMO_FEATURES_POP3 + ["target"]).to_pandas().fillna(0)
    df_que3 = te_df[p3_te_idx].select(KUMO_FEATURES_POP3).to_pandas().fillna(0)
    df_que3["target"] = 0
    
    tbl_ctx3 = sdm.TableTensor.from_pandas(df_ctx3, stypes=sdm.infer_stypes(df_ctx3, overrides={"target": "categorical"}), device=device)
    tbl_que3 = sdm.TableTensor.from_pandas(df_que3, stypes=sdm.infer_stypes(df_que3, overrides={"target": "categorical"}), device=device)
    
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        res = model(
            x_context=tbl_ctx3.drop_columns("target"),
            y_context=tbl_ctx3["target"],
            x_query=tbl_que3.drop_columns("target"),
            num_estimators=2
        )
        p3_probs = res.blocks[sdm.Stype.numerical][:, 1].detach().cpu().numpy()
        
    test_kumo_probs[p3_te_idx] = p3_probs
    print(f"Pop 3 completed in {time.time() - t0:.1f}s | Probs range: [{test_kumo_probs[p3_te_idx].min():.4f}, {test_kumo_probs[p3_te_idx].max():.4f}]")
    del tbl_ctx3, tbl_que3, df_ctx3, df_que3, model; cleanup_memory()
    
    # Save predictions
    out_df = pl.DataFrame({
        "application_id": te_df["application_id"],
        "test_kumo": test_kumo_probs
    })
    out_path = f"{PREDS_DIR}/test_kumo_population.parquet"
    out_df.write_parquet(out_path)
    print(f"\nSaved Kumo Population Predictions to {out_path} ({len(out_df)} rows)")
    log_resource_status()

if __name__ == "__main__":
    run_kumo_inference()
