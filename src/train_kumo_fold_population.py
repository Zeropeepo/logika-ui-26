"""
exp067_kumo_fold_population: Complete 5-Fold Kumo-Tabular Population Foundation Model Pipeline
Implements:
1. Population-specific feature subspaces (Pop 1 Full, Pop 2 Thin, Pop 3 Rejected)
2. 5-Fold Cross Validation with Stratified Multi-Context Bagging
3. Exact out-of-fold inference (oof_kumo_population.parquet) and test inference (test_kumo_population.parquet)
4. Automated context orientation calibration (guarantees positive correlation with default target)
5. Comprehensive metric reporting: min_fold_ks, mean_fold_ks, std_fold_ks, and per-population KS.
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
PREDS_DIR = "preds"

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return ks_2samp(preds_bad, preds_good).statistic

KUMO_FEATS_POP1 = [
    "repay_max_max_dpd", "repay_frac_ontime", "repay_total_amount_paid", "repay_weighted_underpaid",
    "repay_installment_count", "repay_cumulative_shortage", "repay_dpd_p90", "repay_dpd_decay_30d",
    "composite_risk_score", "rate_score_ratio", "income_credits_to_declared_ratio", "acct_income_credits_ratio",
    "acct_mean_balance_all", "acct_mean_inflows_all", "acct_mean_outflows_all", "acct_balance_volatility",
    "acct_cash_burn_months_ratio", "acct_balance_runway_months", "acct_low_bal_days_intensity",
    "bureau_life_score_latest", "bureau_life_score_mean", "bureau_life_score_min", "bureau_life_out_bal_latest",
    "bureau_life_inquiries_latest", "bureau_score_drop_from_max", "bureau_recalculated_utilization",
    "declared_monthly_income", "loan_amount", "proposed_monthly_installment", "annual_interest_rate",
    "age", "employment_tenure_months", "dependents", "housing_status", "loan_purpose", "employment_type", "region",
    "cross_income_per_dep", "cross_career_stability_ratio", "cross_dsr_squared",
    "ecom_order_p10", "ecom_order_p25", "ecom_spend_groceries", "ecom_share_personal_care",
    "ecom_gap_mean_days", "ecom_gap_std_days", "ecom_median_order_val", "ecom_orders_14d",
    "ecom_discretionary_spend_ratio", "ecom_spend_30d", "ecom_comp_orders_all"
]

KUMO_FEATS_POP2 = [
    "composite_risk_score", "bureau_score_max", "bureau_score_mean", "bureau_score_min",
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
    "ecom_share_ewallet", "ecom_share_credit", "ecom_discretionary_spend_ratio", "ecom_spend_30d"
]

KUMO_FEATS_POP3 = [
    "composite_risk_score", "prev_apps_rejection_rate", "prev_apps_approval_rate",
    "prev_days_since_latest_app", "bureau_life_score_min", "bureau_life_score_latest",
    "bureau_life_score_first", "bureau_life_out_bal_latest", "bureau_score_Bureau_A",
    "acct_balance_volatility", "acct_outflow_trend_3m_vs_6m", "acct_mean_balance_all",
    "acct_mean_inflows_all", "declared_monthly_income", "loan_amount", "proposed_monthly_installment",
    "annual_interest_rate", "age", "employment_tenure_months", "dependents",
    "housing_status", "loan_purpose", "employment_type", "region",
    "ecom_order_p10", "ecom_order_p25", "ecom_gap_mean_days", "ecom_gap_std_days",
    "ecom_spend_groceries", "ecom_spend_14d", "ecom_discretionary_spend_ratio",
    "ecom_mean_order_value", "ecom_item_total_items", "ecom_latest_order_days", "ecom_share_personal_care"
]

def main():
    print("=" * 85)
    print("RUNNING exp067_kumo_fold_population: 5-FOLD IN-CONTEXT KUMO-TABULAR PIPELINE")
    print("=" * 85)
    log_resource_status()
    t_start = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load full datasets
    tr_df = pl.read_parquet(f"{DATA_PROCESSED}/train_features_v5.parquet")
    te_df = pl.read_parquet(f"{DATA_PROCESSED}/test_features_v5.parquet")
    df_folds = pl.read_parquet(f"{DATA_PROCESSED}/cv_folds_70k.parquet")
    
    y = tr_df["target"].to_numpy().astype(np.int32)
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    # Populations
    tr_has_prev = tr_df["cov_has_prev_apps"].to_numpy()
    tr_has_repay = tr_df["cov_has_repayment"].to_numpy()
    tr_has_acct = tr_df["cov_has_account"].to_numpy()
    
    tr_pop1 = (tr_has_repay == 1) & (tr_has_acct == 1)
    tr_pop3 = (tr_has_prev == 1) & (tr_has_repay == 0)
    tr_pop2 = (~tr_pop1) & (~tr_pop3)
    
    te_has_prev = te_df["cov_has_prev_apps"].to_numpy()
    te_has_repay = te_df["cov_has_repayment"].to_numpy()
    te_has_acct = te_df["cov_has_account"].to_numpy()
    
    te_pop1 = (te_has_repay == 1) & (te_has_acct == 1)
    te_pop3 = (te_has_prev == 1) & (te_has_repay == 0)
    te_pop2 = (~te_pop1) & (~te_pop3)
    
    print(f"Train Cohorts: Pop1={tr_pop1.sum():,}, Pop2={tr_pop2.sum():,}, Pop3={tr_pop3.sum():,}")
    print(f"Test Cohorts : Pop1={te_pop1.sum():,}, Pop2={te_pop2.sum():,}, Pop3={te_pop3.sum():,}")
    
    oof_kumo = np.zeros(len(y), dtype=np.float32)
    test_kumo = np.zeros(len(te_df), dtype=np.float32)
    
    populations_config = [
        ("Pop 1 (Full-File)", 1, tr_pop1, te_pop1, KUMO_FEATS_POP1, 1500),
        ("Pop 2 (Thin-File)", 2, tr_pop2, te_pop2, KUMO_FEATS_POP2, 1200),
        ("Pop 3 (Rejected)",  3, tr_pop3, te_pop3, KUMO_FEATS_POP3, 900)
    ]
    
    chunk_size = 5000
    cat_cols_all = ["housing_status", "loan_purpose", "employment_type", "region"]
    
    for pop_name, pop_id, tr_pop_mask, te_pop_mask, feat_list, ctx_size in populations_config:
        print("\n" + "=" * 80)
        print(f"STARTING {pop_name} - {len(feat_list)} Features | Context Size: {ctx_size}")
        print("=" * 80)
        
        # Verify columns exist
        feat_list = [c for c in feat_list if c in tr_df.columns and c in te_df.columns]
        
        pop_tr_idx_all = np.where(tr_pop_mask)[0]
        pop_te_idx = np.where(te_pop_mask)[0]
        
        # Prepare test table for this pop
        df_te_pop = te_df[pop_te_idx].select(feat_list).to_pandas().fillna(0)
        for c in cat_cols_all:
            if c in df_te_pop.columns:
                df_te_pop[c] = df_te_pop[c].astype(str)
        tbl_te_pop = sdm.TableTensor.from_pandas(df_te_pop, stypes=sdm.infer_stypes(df_te_pop), device=device)
        
        pop_test_accum = np.zeros(len(pop_te_idx), dtype=np.float32)
        pop_fold_scores = []
        
        for f in range(5):
            t_f0 = time.time()
            tr_fold_mask = tr_pop_mask & (fold_col != f)
            va_fold_mask = tr_pop_mask & (fold_col == f)
            
            tr_fold_idx = np.where(tr_fold_mask)[0]
            va_fold_idx = np.where(va_fold_mask)[0]
            
            # Multi-context bagging: 2 diverse stratified context bags
            va_fold_preds = np.zeros(len(va_fold_idx), dtype=np.float32)
            n_bags = 2
            
            for b in range(n_bags):
                rng = np.random.RandomState(42 + f * 10 + b)
                pos_idx = tr_fold_idx[y[tr_fold_idx] == 1]
                neg_idx = tr_fold_idx[y[tr_fold_idx] == 0]
                
                n_pos = int(ctx_size * y[tr_fold_idx].mean())
                n_neg = ctx_size - n_pos
                ctx_idx = np.concatenate([
                    rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
                    rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
                ])
                rng.shuffle(ctx_idx)
                
                df_ctx = tr_df[ctx_idx].select(feat_list + ["target"]).to_pandas().fillna(0)
                y_ctx = df_ctx["target"].to_numpy()
                df_va = tr_df[va_fold_idx].select(feat_list).to_pandas().fillna(0)
                
                for c in cat_cols_all:
                    if c in df_ctx.columns:
                        df_ctx[c] = df_ctx[c].astype(str)
                        df_va[c] = df_va[c].astype(str)
                        
                tbl_ctx = sdm.TableTensor.from_pandas(
                    df_ctx,
                    stypes=sdm.infer_stypes(df_ctx, overrides={"target": "categorical"}),
                    device=device
                )
                tbl_va = sdm.TableTensor.from_pandas(df_va, stypes=sdm.infer_stypes(df_va), device=device)
                
                with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    model = sdm.models.KumoTabular(task="classification", size="small", device=device)
                    model.fit(tbl_ctx.drop_columns("target"), tbl_ctx["target"], num_estimators=2)
                    
                    # Calibration check on context: find column that positively correlates with target=1
                    res_ctx = model.predict(tbl_ctx.drop_columns("target")[:300]).blocks[sdm.Stype.numerical].detach().cpu().numpy()
                    corr0 = np.corrcoef(res_ctx[:, 0], y_ctx[:300])[0, 1]
                    target_col = 0 if corr0 > 0 else 1
                    
                    # Predict on validation fold
                    va_preds_b = []
                    for i in range(0, len(tbl_va), chunk_size):
                        end_i = min(i + chunk_size, len(tbl_va))
                        res_va = model.predict(tbl_va[i:end_i])
                        va_preds_b.append(res_va.blocks[sdm.Stype.numerical][:, target_col].detach().cpu().numpy())
                    va_fold_preds += np.concatenate(va_preds_b) / n_bags
                    
                    # Predict on test
                    te_preds_b = []
                    for i in range(0, len(tbl_te_pop), chunk_size):
                        end_i = min(i + chunk_size, len(tbl_te_pop))
                        res_te = model.predict(tbl_te_pop[i:end_i])
                        te_preds_b.append(res_te.blocks[sdm.Stype.numerical][:, target_col].detach().cpu().numpy())
                    pop_test_accum += np.concatenate(te_preds_b) / (5.0 * n_bags)
                    
                del tbl_ctx, tbl_va, df_ctx, df_va, model
                cleanup_memory()
                
            oof_kumo[va_fold_idx] = va_fold_preds
            fold_ks = official_ks_metric(y[va_fold_idx], va_fold_preds)
            pop_fold_scores.append(fold_ks)
            print(f"  {pop_name} Fold {f} KS: {fold_ks:.4f} (took {time.time() - t_f0:.1f}s)")
            
        test_kumo[pop_te_idx] = pop_test_accum
        del tbl_te_pop, df_te_pop
        cleanup_memory()
        
        pop_ks = official_ks_metric(y[tr_pop_mask], oof_kumo[tr_pop_mask])
        print(f"--> {pop_name} Overall Pop KS: {pop_ks:.4f} | Folds: {[round(s, 4) for s in pop_fold_scores]}")
        
    # Overall evaluation
    overall_ks = official_ks_metric(y, oof_kumo)
    fold_scores_all = [official_ks_metric(y[fold_col == f], oof_kumo[fold_col == f]) for f in range(5)]
    min_f = min(fold_scores_all)
    mean_f = np.mean(fold_scores_all)
    std_f = np.std(fold_scores_all)
    
    print("\n" + "=" * 85)
    print("exp067_kumo_fold_population FULL 5-FOLD RESULTS:")
    print(f"  Overall OOF KS: {overall_ks:.4f}")
    print(f"  Min Fold KS   : {min_f:.4f} (Worst fold robustness check)")
    print(f"  Mean Fold KS  : {mean_f:.4f} ± {std_f:.4f}")
    print(f"  Fold Breakdown: {[round(s, 4) for s in fold_scores_all]}")
    print(f"  Pop 1 KS (66%): {official_ks_metric(y[tr_pop1], oof_kumo[tr_pop1]):.4f}")
    print(f"  Pop 2 KS (30%): {official_ks_metric(y[tr_pop2], oof_kumo[tr_pop2]):.4f}")
    print(f"  Pop 3 KS ( 4%): {official_ks_metric(y[tr_pop3], oof_kumo[tr_pop3]):.4f}")
    print(f"Total Pipeline Runtime: {time.time() - t_start:.1f}s")
    print("=" * 85)
    
    # Save OOF and test predictions
    pl.DataFrame({
        "application_id": tr_df["application_id"],
        "oof_kumo": oof_kumo
    }).write_parquet(f"{PREDS_DIR}/oof_kumo_population.parquet")
    
    pl.DataFrame({
        "application_id": te_df["application_id"],
        "test_kumo": test_kumo
    }).write_parquet(f"{PREDS_DIR}/test_kumo_population.parquet")
    print(f"Saved: {PREDS_DIR}/oof_kumo_population.parquet & {PREDS_DIR}/test_kumo_population.parquet")

if __name__ == "__main__":
    main()
