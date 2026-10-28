#!/usr/bin/env python3
"""
EXP 01: Kumo-Tabular Foundation Model Maximum Capacity & Multi-Bagging
Designed for NVIDIA A100-SXM4 80GB (shared node tolerant).
Features:
- Population-specific feature subspaces (Full, Thin, Rejected)
- High context sample points (3000-4000)
- In-context bagging (4-5 stratified bags per fold)
- Automated orientation calibration
"""
import os
import sys
import time
import torch
import polars as pl
import numpy as np
import pandas as pd
import sdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_hpc import (
    load_clean_dataset, evaluate_metrics, log_experiment_to_csv,
    cleanup_gpu, normalize_rank
)

FEATS_POP1 = [
    "repay_max_max_dpd", "repay_frac_ontime", "repay_total_amount_paid", "repay_weighted_underpaid",
    "repay_installment_count", "repay_cumulative_shortage", "repay_dpd_p90", "repay_dpd_decay_30d",
    "rate_score_ratio", "income_credits_to_declared_ratio", "acct_income_credits_ratio",
    "acct_mean_balance_all", "acct_mean_inflows_all", "acct_mean_outflows_all", "acct_balance_volatility",
    "acct_cash_burn_months_ratio", "acct_balance_runway_months", "acct_low_bal_days_intensity",
    "bureau_life_score_latest", "bureau_life_score_mean", "bureau_life_score_min", "bureau_life_out_bal_latest",
    "bureau_life_inquiries_latest", "bureau_score_drop_from_max", "bureau_recalculated_utilization",
    "declared_monthly_income", "loan_amount", "proposed_monthly_installment", "annual_interest_rate",
    "age", "employment_tenure_months", "dependents", "housing_status", "loan_purpose", "employment_type", "region",
    "cross_income_per_dep", "cross_career_stability_ratio", "cross_dsr_squared",
    "ecom_order_p10", "ecom_order_p25", "ecom_spend_groceries", "ecom_share_personal_care",
    "ecom_gap_mean_days", "ecom_gap_std_days", "ecom_median_order_val", "ecom_orders_14d",
    "ecom_discretionary_spend_ratio", "ecom_spend_30d", "ecom_comp_orders_all",
    "woe_region", "woe_loan_purpose", "woe_employment_type", "woe_housing_status"
]

FEATS_POP2 = [
    "bureau_score_max", "bureau_score_mean", "bureau_score_min",
    "bureau_life_score_first", "bureau_life_score_latest", "bureau_life_out_bal_latest",
    "bureau_life_inquiries_latest", "bureau_recalculated_utilization", "bureau_score_Bureau_A",
    "bureau_score_Bureau_B", "declared_monthly_income", "loan_amount", "proposed_monthly_installment",
    "annual_interest_rate", "age", "employment_tenure_months", "dependents",
    "housing_status", "loan_purpose", "employment_type", "region",
    "installment_income_ratio", "loan_income_ratio", "cross_income_per_dep",
    "cross_career_stability_ratio", "cross_dsr_squared", "cross_interest_burden_index",
    "ecom_gap_mean_days", "ecom_gap_std_days", "ecom_order_p10", "ecom_order_p25",
    "ecom_spend_groceries", "ecom_spending_90d", "ecom_spending_180d", "ecom_mean_order_value",
    "ecom_orders_14d", "ecom_share_personal_care", "ecom_spend_60d", "ecom_share_groceries_right",
    "ecom_electronics_share", "ecom_median_order_val", "ecom_weekend_spend_share",
    "ecom_share_ewallet", "ecom_share_credit", "ecom_discretionary_spend_ratio", "ecom_spend_30d",
    "woe_region", "woe_loan_purpose", "woe_employment_type", "woe_housing_status"
]

FEATS_POP3 = [
    "prev_apps_rejection_rate", "prev_apps_approval_rate",
    "prev_days_since_latest_app", "bureau_life_score_min", "bureau_life_score_latest",
    "bureau_life_score_first", "bureau_life_out_bal_latest", "bureau_score_Bureau_A",
    "acct_balance_volatility", "acct_outflow_trend_3m_vs_6m", "acct_mean_balance_all",
    "acct_mean_inflows_all", "declared_monthly_income", "loan_amount", "proposed_monthly_installment",
    "annual_interest_rate", "age", "employment_tenure_months", "dependents",
    "housing_status", "loan_purpose", "employment_type", "region",
    "ecom_order_p10", "ecom_order_p25", "ecom_gap_mean_days", "ecom_gap_std_days",
    "ecom_spend_groceries", "ecom_spend_14d", "ecom_discretionary_spend_ratio",
    "ecom_mean_order_value", "ecom_item_total_items", "ecom_latest_order_days", "ecom_share_personal_care",
    "woe_region", "woe_loan_purpose", "woe_employment_type", "woe_housing_status"
]

def run_experiment():
    print("=" * 80)
    print("STARTING EXP 01: KUMO-TABULAR FOUNDATION MAXIMUM CAPACITY SUITE")
    print("=" * 80)
    t_start = time.time()
    
    clean_tr, clean_te, y, fold_col, masks = load_clean_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    
    free_gb = 0.0
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1024**3
        print(f"[*] Current Free VRAM: {free_gb:.1f} GB")
        
    kumo_size = "medium" if free_gb >= 15.0 else "small"
    ctx_size = 3500 if free_gb >= 30.0 else (2500 if free_gb >= 15.0 else 1500)
    n_estimators = 6 if free_gb >= 30.0 else (4 if free_gb >= 15.0 else 2)
    n_bags = 4 if free_gb >= 20.0 else 2
    chunk_size = 6000
    
    print(f"[*] Config: size='{kumo_size}' | context={ctx_size} | estimators={n_estimators} | bags={n_bags}")
    
    pop_configs = [
        ("Pop 1 (Full-File)", masks["tr_pop1"], masks["te_pop1"], FEATS_POP1, ctx_size),
        ("Pop 2 (Thin-File)", masks["tr_pop2"], masks["te_pop2"], FEATS_POP2, int(ctx_size * 0.8)),
        ("Pop 3 (Rejected)",  masks["tr_pop3"], masks["te_pop3"], FEATS_POP3, int(ctx_size * 0.6))
    ]
    
    cat_cols_all = ["housing_status", "loan_purpose", "employment_type", "region", "cat_region_employment", "cat_purpose_housing"]
    oof_kumo = np.zeros(len(y), dtype=np.float32)
    test_kumo = np.zeros(len(clean_te), dtype=np.float32)
    
    for pop_name, tr_mask, te_mask, feat_list, c_size in pop_configs:
        feat_list = [c for c in feat_list if c in clean_tr.columns and c in clean_te.columns]
        pop_te_idx = np.where(te_mask)[0]
        
        print(f"\n[*] Processing {pop_name} ({len(feat_list)} Features | Ctx: {c_size})...")
        df_te_pop = clean_te[pop_te_idx].select(feat_list).to_pandas().fillna(0)
        for c in cat_cols_all:
            if c in df_te_pop.columns:
                df_te_pop[c] = df_te_pop[c].astype(str)
        tbl_te_pop = sdm.TableTensor.from_pandas(df_te_pop, stypes=sdm.infer_stypes(df_te_pop), device=device)
        
        pop_test_accum = np.zeros(len(pop_te_idx), dtype=np.float32)
        
        for f in range(5):
            t_f = time.time()
            tr_f_idx = np.where(tr_mask & (fold_col != f))[0]
            va_f_idx = np.where(tr_mask & (fold_col == f))[0]
            va_fold_preds = np.zeros(len(va_f_idx), dtype=np.float32)
            
            for b in range(n_bags):
                rng = np.random.RandomState(42 + f * 100 + b)
                pos_idx = tr_f_idx[y[tr_f_idx] == 1]
                neg_idx = tr_f_idx[y[tr_f_idx] == 0]
                
                n_pos = int(c_size * y[tr_f_idx].mean())
                n_neg = c_size - n_pos
                ctx_idx = np.concatenate([
                    rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
                    rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
                ])
                rng.shuffle(ctx_idx)
                
                df_ctx = clean_tr[ctx_idx].select(feat_list + ["target"]).to_pandas().fillna(0)
                y_ctx = df_ctx["target"].to_numpy()
                df_va = clean_tr[va_f_idx].select(feat_list).to_pandas().fillna(0)
                
                for c in cat_cols_all:
                    if c in df_ctx.columns:
                        df_ctx[c] = df_ctx[c].astype(str)
                        df_va[c] = df_va[c].astype(str)
                        
                tbl_ctx = sdm.TableTensor.from_pandas(df_ctx, stypes=sdm.infer_stypes(df_ctx, overrides={"target": "categorical"}), device=device)
                tbl_va = sdm.TableTensor.from_pandas(df_va, stypes=sdm.infer_stypes(df_va), device=device)
                
                with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    model = sdm.models.KumoTabular(task="classification", size=kumo_size, device=device)
                    model.fit(tbl_ctx.drop_columns("target"), tbl_ctx["target"], num_estimators=n_estimators)
                    
                    # Calibration check
                    res_ctx = model.predict(tbl_ctx.drop_columns("target")[:300]).blocks[sdm.Stype.numerical].detach().cpu().numpy()
                    target_col = 0 if np.corrcoef(res_ctx[:, 0], y_ctx[:300])[0, 1] > 0 else 1
                    
                    # Validation query
                    va_b = []
                    for i in range(0, len(tbl_va), chunk_size):
                        end_i = min(i + chunk_size, len(tbl_va))
                        res_v = model.predict(tbl_va[i:end_i])
                        va_b.append(res_v.blocks[sdm.Stype.numerical][:, target_col].detach().cpu().numpy())
                    va_fold_preds += np.concatenate(va_b) / n_bags
                    
                    # Test query
                    te_b = []
                    for i in range(0, len(tbl_te_pop), chunk_size):
                        end_i = min(i + chunk_size, len(tbl_te_pop))
                        res_t = model.predict(tbl_te_pop[i:end_i])
                        te_b.append(res_t.blocks[sdm.Stype.numerical][:, target_col].detach().cpu().numpy())
                    pop_test_accum += np.concatenate(te_b) / (5.0 * n_bags)
                    
                del tbl_ctx, tbl_va, df_ctx, df_va, model
                cleanup_gpu()
                
            oof_kumo[va_f_idx] = va_fold_preds
            print(f"    Fold {f} KS: {official_ks_metric(y[va_f_idx], va_fold_preds):.4f} ({time.time() - t_f:.1f}s)")
            
        test_kumo[pop_te_idx] = pop_test_accum
        del tbl_te_pop, df_te_pop
        cleanup_gpu()
        
    metrics = evaluate_metrics(y, oof_kumo, fold_col, masks)
    print("\n" + "=" * 80)
    print(f"EXP 01 COMPLETE: Overall KS={metrics['overall_ks']:.4f} | Min Fold={metrics['min_fold_ks']:.4f} | Pop1={metrics['pop1_ks']:.4f} | Pop2={metrics['pop2_ks']:.4f} | Pop3={metrics['pop3_ks']:.4f}")
    print(f"Time Taken: {time.time() - t_start:.1f}s")
    print("=" * 80)
    
    # Save predictions
    os.makedirs("preds", exist_ok=True)
    pl.DataFrame({"application_id": clean_tr["application_id"], "oof_kumo": oof_kumo}).write_parquet("preds/oof_exp01_kumo.parquet")
    pl.DataFrame({"application_id": clean_te["application_id"], "test_kumo": test_kumo}).write_parquet("preds/test_exp01_kumo.parquet")
    
    # Save standalone submission
    os.makedirs("submissions", exist_ok=True)
    p_norm = normalize_rank(test_kumo)
    pl.DataFrame({"application_id": clean_te["application_id"], "default_probability": p_norm.astype(np.float64)}).write_csv("submissions/sub_exp01_kumo_foundation.csv")
    
    log_experiment_to_csv("EXP-01", "KumoTabular_Capacity_Suite", f"size={kumo_size}, ctx={ctx_size}, bags={n_bags}", metrics)
    return metrics

if __name__ == "__main__":
    run_experiment()
