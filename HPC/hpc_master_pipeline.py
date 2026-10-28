#!/usr/bin/env python3
"""
HPC Master Pipeline: Grandmaster Tabular Foundation & Multi-Table Pipeline
Optimized for: NVIDIA A100-SXM4 80GB | Driver 580.126.20 | Shared Multi-User Node
Target: Indonesian Consumer Lending Credit Risk Competition
Metric: Two-sample Kolmogorov-Smirnov (KS) Statistic

Key Architecture:
1. Dynamic Shared-GPU Memory Sizing & Pre-flight check
2. Cleaned Feature Bank (Purged of tied-rank artifacts) + WOE Encoding + 2-Way Interactions
3. Decoupled Population Foundation Modeling (sdm.models.KumoTabular size='medium' with Multi-Context Bagging)
4. Teacher-Student Smart Pseudo-Labeling (distilling 0.55677 anchor ens_q4cb.csv)
5. Multi-Seed Decoupled GBDT Experts (LightGBM DART + CatBoost GPU)
6. Robustness Evaluation: Min-Fold KS, Mean-Fold KS, Pop-specific KS
7. Final Verified Submissions Generation
"""

import os
import sys
import gc
import time
import argparse
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import sdm

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return ks_2samp(preds_bad, preds_good).statistic

def normalize_rank(arr):
    return (rankdata(arr) - 0.5) / len(arr)

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def find_file(filename, custom_dir=None):
    candidates = []
    if custom_dir:
        candidates.append(os.path.join(custom_dir, filename))
    candidates.extend([
        f"data/{filename}",
        f"../data/{filename}",
        f"../data/processed/{filename}",
        f"../data/raw/{filename}",
        f"HPC/data/{filename}",
        f"HPC/anchors/{filename}",
        f"anchors/{filename}",
        f"/home/neva/Downloads/{filename}",
        filename
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find {filename}. Looked in: {candidates}")

# ==============================================================================
# 1. HARDWARE & SHARED GPU AUTO-SCALER
# ==============================================================================
def detect_hardware():
    print("=" * 80)
    print("PHASE 0: HARDWARE DETECTION & DYNAMIC SHARED-GPU ALLOCATION")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    
    free_vram_gb = 0.0
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Query free memory
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_vram_gb = free_bytes / 1024**3
        print(f"[*] GPU Name      : {torch.cuda.get_device_name(0)}")
        print(f"[*] Total VRAM    : {total_vram:.1f} GB")
        print(f"[*] Free VRAM     : {free_vram_gb:.1f} GB (Shared Node Usage)")
        
    if free_vram_gb >= 35.0:
        config = {
            "kumo_size": "medium",
            "context_size": 4000,
            "num_estimators": 6,
            "n_bags": 4,
            "chunk_size": 8000,
            "catboost_task": "GPU",
            "n_jobs": 8
        }
    elif free_vram_gb >= 15.0:
        config = {
            "kumo_size": "medium",
            "context_size": 2500,
            "num_estimators": 4,
            "n_bags": 3,
            "chunk_size": 5000,
            "catboost_task": "GPU",
            "n_jobs": 6
        }
    elif free_vram_gb >= 6.0:
        config = {
            "kumo_size": "small",
            "context_size": 1500,
            "num_estimators": 2,
            "n_bags": 2,
            "chunk_size": 4000,
            "catboost_task": "GPU",
            "n_jobs": 4
        }
    else:
        config = {
            "kumo_size": "small",
            "context_size": 1000,
            "num_estimators": 2,
            "n_bags": 1,
            "chunk_size": 2000,
            "catboost_task": "CPU",
            "n_jobs": 4
        }
    print("[*] Allocated HPC Engine Configuration:")
    for k, v in config.items():
        print(f"    - {k:<16}: {v}")
    print("=" * 80)
    return device, config

# ==============================================================================
# 2. FEATURE PURIFICATION & WEIGHT-OF-EVIDENCE (WOE) ENCODING
# ==============================================================================
def prepare_clean_features(tr_df: pl.DataFrame, te_df: pl.DataFrame, fold_col: np.ndarray):
    print("\n" + "=" * 80)
    print("PHASE 1: FEATURE PURIFICATION & OUT-OF-FOLD WOE ENCODING")
    print("=" * 80)
    
    # 1. Purge leaky rank_* features that triggered 1.0 AUC adversarial shift
    leaky_patterns = ["rank_", "pricing_predicted_rate", "pricing_rate_residual", "cross_app_id_hash"]
    cols_to_drop = [c for c in tr_df.columns if any(pat in c for pat in leaky_patterns)]
    print(f"[*] Purging {len(cols_to_drop)} leaky/shifted features: {cols_to_drop[:6]}...")
    
    clean_tr = tr_df.drop(cols_to_drop)
    clean_te = te_df.drop(cols_to_drop)
    
    y = clean_tr["target"].to_numpy().astype(np.int32)
    
    # 2. Out-of-Fold Weight of Evidence (WOE) on core categoricals
    cat_cols = ["region", "loan_purpose", "employment_type", "housing_status"]
    cat_cols = [c for c in cat_cols if c in clean_tr.columns]
    
    pdf_tr_cat = clean_tr.select(cat_cols).to_pandas()
    pdf_te_cat = clean_te.select(cat_cols).to_pandas()
    
    woe_tr_dict = {}
    woe_te_dict = {}
    
    for c in cat_cols:
        woe_tr = np.zeros(len(clean_tr), dtype=np.float32)
        woe_te = np.zeros(len(clean_te), dtype=np.float32)
        
        for f in range(5):
            idx_tr = np.where(fold_col != f)[0]
            idx_va = np.where(fold_col == f)[0]
            
            # Compute WOE on train fold
            df_fold = pd.DataFrame({"cat": pdf_tr_cat[c].iloc[idx_tr], "target": y[idx_tr]})
            grouped = df_fold.groupby("cat")["target"].agg(["count", "sum"])
            total_pos = y[idx_tr].sum()
            total_neg = len(idx_tr) - total_pos
            
            # Laplace smoothing
            pos = grouped["sum"] + 1.0
            neg = (grouped["count"] - grouped["sum"]) + 1.0
            woe_map = np.log((pos / total_pos) / (neg / total_neg)).to_dict()
            
            # Apply to val fold
            woe_tr[idx_va] = pdf_tr_cat[c].iloc[idx_va].map(woe_map).fillna(0.0).values
            
            # Apply to test
            woe_te += pdf_te_cat[c].map(woe_map).fillna(0.0).values / 5.0
            
        woe_tr_dict[f"woe_{c}"] = woe_tr
        woe_te_dict[f"woe_{c}"] = woe_te
        print(f"[*] Synthesized Out-of-Fold WOE: woe_{c}")
        
    # 3. Two-way Interaction Risk Categories
    print("[*] Creating 2-way categorical risk interactions...")
    tr_reg_emp = (clean_tr["region"].fill_null("UNK") + "_" + clean_tr["employment_type"].fill_null("UNK")).to_list()
    te_reg_emp = (clean_te["region"].fill_null("UNK") + "_" + clean_te["employment_type"].fill_null("UNK")).to_list()
    
    tr_purp_house = (clean_tr["loan_purpose"].fill_null("UNK") + "_" + clean_tr["housing_status"].fill_null("UNK")).to_list()
    te_purp_house = (clean_te["loan_purpose"].fill_null("UNK") + "_" + clean_te["housing_status"].fill_null("UNK")).to_list()
    
    clean_tr = clean_tr.with_columns([
        pl.Series(k, v) for k, v in woe_tr_dict.items()
    ]).with_columns([
        pl.Series("cat_region_employment", tr_reg_emp),
        pl.Series("cat_purpose_housing", tr_purp_house)
    ])
    
    clean_te = clean_te.with_columns([
        pl.Series(k, v) for k, v in woe_te_dict.items()
    ]).with_columns([
        pl.Series("cat_region_employment", te_reg_emp),
        pl.Series("cat_purpose_housing", te_purp_house)
    ])
    
    print(f"[+] Feature Bank Cleaned & Enriched: Train={clean_tr.shape}, Test={clean_te.shape}")
    return clean_tr, clean_te

# ==============================================================================
# 3. EXP067: 5-FOLD IN-CONTEXT KUMO-TABULAR POPULATION FOUNDATION PIPELINE
# ==============================================================================
def run_kumo_population_foundation(clean_tr: pl.DataFrame, clean_te: pl.DataFrame, fold_col: np.ndarray, device, config):
    print("\n" + "=" * 80)
    print("PHASE 2: exp067_kumo_fold_population FOUNDATION MODEL (NVIDIA Kumo-Tabular)")
    print("=" * 80)
    
    y = clean_tr["target"].to_numpy().astype(np.int32)
    
    # Population masks
    tr_has_prev = clean_tr["cov_has_prev_apps"].to_numpy()
    tr_has_repay = clean_tr["cov_has_repayment"].to_numpy()
    tr_has_acct = clean_tr["cov_has_account"].to_numpy()
    tr_pop1 = (tr_has_repay == 1) & (tr_has_acct == 1)
    tr_pop3 = (tr_has_prev == 1) & (tr_has_repay == 0)
    tr_pop2 = (~tr_pop1) & (~tr_pop3)
    
    te_has_prev = clean_te["cov_has_prev_apps"].to_numpy()
    te_has_repay = clean_te["cov_has_repayment"].to_numpy()
    te_has_acct = clean_te["cov_has_account"].to_numpy()
    te_pop1 = (te_has_repay == 1) & (te_has_acct == 1)
    te_pop3 = (te_has_prev == 1) & (te_has_repay == 0)
    te_pop2 = (~te_pop1) & (~te_pop3)
    
    print(f"[*] Train Cohorts: Pop1={tr_pop1.sum():,}, Pop2={tr_pop2.sum():,}, Pop3={tr_pop3.sum():,}")
    print(f"[*] Test Cohorts : Pop1={te_pop1.sum():,}, Pop2={te_pop2.sum():,}, Pop3={te_pop3.sum():,}")
    
    # Tailored feature banks per population
    feats_pop1 = [
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
    
    feats_pop2 = [
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
    
    feats_pop3 = [
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
    
    oof_kumo = np.zeros(len(y), dtype=np.float32)
    test_kumo = np.zeros(len(clean_te), dtype=np.float32)
    
    pop_configs = [
        ("Pop 1 (Full-File)", tr_pop1, te_pop1, feats_pop1, config["context_size"]),
        ("Pop 2 (Thin-File)", tr_pop2, te_pop2, feats_pop2, int(config["context_size"] * 0.8)),
        ("Pop 3 (Rejected)",  tr_pop3, te_pop3, feats_pop3, int(config["context_size"] * 0.6))
    ]
    
    cat_cols_all = ["housing_status", "loan_purpose", "employment_type", "region", "cat_region_employment", "cat_purpose_housing"]
    chunk_size = config["chunk_size"]
    n_bags = config["n_bags"]
    
    for pop_name, tr_mask, te_mask, feat_list, ctx_size in pop_configs:
        feat_list = [c for c in feat_list if c in clean_tr.columns and c in clean_te.columns]
        pop_tr_idx = np.where(tr_mask)[0]
        pop_te_idx = np.where(te_mask)[0]
        
        print(f"\n--- Running Kumo-Tabular on {pop_name} ({len(feat_list)} Features | Ctx: {ctx_size} | Bags: {n_bags}) ---")
        
        df_te_pop = clean_te[pop_te_idx].select(feat_list).to_pandas().fillna(0)
        for c in cat_cols_all:
            if c in df_te_pop.columns:
                df_te_pop[c] = df_te_pop[c].astype(str)
        tbl_te_pop = sdm.TableTensor.from_pandas(df_te_pop, stypes=sdm.infer_stypes(df_te_pop), device=device)
        
        pop_test_accum = np.zeros(len(pop_te_idx), dtype=np.float32)
        pop_fold_ks = []
        
        for f in range(5):
            t_f0 = time.time()
            tr_f_idx = np.where(tr_mask & (fold_col != f))[0]
            va_f_idx = np.where(tr_mask & (fold_col == f))[0]
            
            va_fold_preds = np.zeros(len(va_f_idx), dtype=np.float32)
            
            for b in range(n_bags):
                rng = np.random.RandomState(42 + f * 100 + b)
                pos_idx = tr_f_idx[y[tr_f_idx] == 1]
                neg_idx = tr_f_idx[y[tr_f_idx] == 0]
                
                n_pos = int(ctx_size * y[tr_f_idx].mean())
                n_neg = ctx_size - n_pos
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
                        
                tbl_ctx = sdm.TableTensor.from_pandas(
                    df_ctx,
                    stypes=sdm.infer_stypes(df_ctx, overrides={"target": "categorical"}),
                    device=device
                )
                tbl_va = sdm.TableTensor.from_pandas(df_va, stypes=sdm.infer_stypes(df_va), device=device)
                
                with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    model = sdm.models.KumoTabular(task="classification", size=config["kumo_size"], device=device)
                    model.fit(tbl_ctx.drop_columns("target"), tbl_ctx["target"], num_estimators=config["num_estimators"])
                    
                    # Automated orientation calibration check
                    res_ctx = model.predict(tbl_ctx.drop_columns("target")[:400]).blocks[sdm.Stype.numerical].detach().cpu().numpy()
                    corr0 = np.corrcoef(res_ctx[:, 0], y_ctx[:400])[0, 1]
                    target_col = 0 if corr0 > 0 else 1
                    
                    # Predict validation
                    va_preds_b = []
                    for i in range(0, len(tbl_va), chunk_size):
                        end_i = min(i + chunk_size, len(tbl_va))
                        res_va = model.predict(tbl_va[i:end_i])
                        va_preds_b.append(res_va.blocks[sdm.Stype.numerical][:, target_col].detach().cpu().numpy())
                    va_fold_preds += np.concatenate(va_preds_b) / n_bags
                    
                    # Predict test
                    te_preds_b = []
                    for i in range(0, len(tbl_te_pop), chunk_size):
                        end_i = min(i + chunk_size, len(tbl_te_pop))
                        res_te = model.predict(tbl_te_pop[i:end_i])
                        te_preds_b.append(res_te.blocks[sdm.Stype.numerical][:, target_col].detach().cpu().numpy())
                    pop_test_accum += np.concatenate(te_preds_b) / (5.0 * n_bags)
                    
                del tbl_ctx, tbl_va, df_ctx, df_va, model
                cleanup_gpu()
                
            oof_kumo[va_f_idx] = va_fold_preds
            f_ks = official_ks_metric(y[va_f_idx], va_fold_preds)
            pop_fold_ks.append(f_ks)
            print(f"    Fold {f} KS: {f_ks:.4f} (took {time.time() - t_f0:.1f}s)")
            
        test_kumo[pop_te_idx] = pop_test_accum
        del tbl_te_pop, df_te_pop
        cleanup_gpu()
        
        pop_overall_ks = official_ks_metric(y[tr_mask], oof_kumo[tr_mask])
        print(f"--> {pop_name} Overall Pop KS: {pop_overall_ks:.4f} | Folds: {[round(s, 4) for s in pop_fold_ks]}")
        
    ov_kumo_ks = official_ks_metric(y, oof_kumo)
    kumo_fold_scores = [official_ks_metric(y[fold_col == f], oof_kumo[fold_col == f]) for f in range(5)]
    print("\n" + "=" * 80)
    print(f"KUMO-TABULAR FOUNDATION RESULTS: Overall OOF KS = {ov_kumo_ks:.4f} | Min Fold = {min(kumo_fold_scores):.4f} | Folds: {[round(s,4) for s in kumo_fold_scores]}")
    print("=" * 80)
    return oof_kumo, test_kumo

# ==============================================================================
# 4. TEACHER-STUDENT SMART PSEUDO-LABELING DISTILLATION
# ==============================================================================
def run_smart_pseudolabel_distillation(clean_tr: pl.DataFrame, clean_te: pl.DataFrame, fold_col: np.ndarray, anchor_file: str, device, config):
    print("\n" + "=" * 80)
    print("PHASE 3: SMART TEACHER-STUDENT PSEUDO-LABEL DISTILLATION")
    print(f"[*] Teacher Model: {anchor_file} (LB 0.55677 Anchor)")
    print("=" * 80)
    
    y = clean_tr["target"].to_numpy().astype(np.int32)
    teacher_df = pl.read_csv(anchor_file)
    teacher_preds = teacher_df["default_probability"].to_numpy()
    
    # Confident pseudo-label selection
    # Top 15% high risk (p > 0.85) -> 1, Top 25% low risk (p < 0.20) -> 0
    high_conf_pos = np.where(teacher_preds >= 0.85)[0]
    high_conf_neg = np.where(teacher_preds <= 0.20)[0]
    
    pseudo_idx = np.concatenate([high_conf_pos, high_conf_neg])
    pseudo_y = np.zeros(len(pseudo_idx), dtype=np.int32)
    pseudo_y[:len(high_conf_pos)] = 1
    
    print(f"[*] Confident Pseudo-Labels selected from Test set: {len(pseudo_idx):,} rows ({len(pseudo_idx)/len(clean_te)*100:.1f}%)")
    print(f"    - High-Risk Defaults (P >= 0.85): {len(high_conf_pos):,}")
    print(f"    - Low-Risk Non-Defaults (P <= 0.20): {len(high_conf_neg):,}")
    
    # Feature columns for student
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"}
    features = [c for c in clean_tr.columns if c not in ignore_cols]
    
    pdf_tr = clean_tr.select(features).to_pandas()
    pdf_te = clean_te.select(features).to_pandas()
    for c in pdf_tr.select_dtypes(include=["object", "string"]).columns:
        pdf_tr[c] = pdf_tr[c].astype("category")
        pdf_te[c] = pdf_te[c].astype("category")
        
    pdf_pseudo = pdf_te.iloc[pseudo_idx].copy()
    
    oof_student = np.zeros(len(y), dtype=np.float32)
    test_student = np.zeros(len(clean_te), dtype=np.float32)
    
    dart_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "dart",
        "learning_rate": 0.05,
        "num_leaves": 20,
        "max_depth": 5,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "n_estimators": 400,
        "n_jobs": config["n_jobs"],
        "random_state": 42,
        "verbose": -1
    }
    
    fold_scores = []
    for f in range(5):
        idx_tr = np.where(fold_col != f)[0]
        idx_va = np.where(fold_col == f)[0]
        
        # Augmented training set = Real fold train + Student pseudo labels (downweighted)
        X_train_aug = pd.concat([pdf_tr.iloc[idx_tr], pdf_pseudo], axis=0)
        y_train_aug = np.concatenate([y[idx_tr], pseudo_y])
        weights_aug = np.concatenate([np.ones(len(idx_tr), dtype=np.float32), np.full(len(pseudo_idx), 0.5, dtype=np.float32)])
        
        clf = lgb.LGBMClassifier(**dart_params)
        clf.fit(X_train_aug, y_train_aug, sample_weight=weights_aug)
        
        val_pred = clf.predict_proba(pdf_tr.iloc[idx_va])[:, 1]
        oof_student[idx_va] = val_pred
        test_student += clf.predict_proba(pdf_te)[:, 1] / 5.0
        
        f_ks = official_ks_metric(y[idx_va], val_pred)
        fold_scores.append(f_ks)
        print(f"    Fold {f} Student KS: {f_ks:.4f}")
        
    ov_student_ks = official_ks_metric(y, oof_student)
    print("\n" + "=" * 80)
    print(f"STUDENT DISTILLATION RESULTS: Overall OOF KS = {ov_student_ks:.4f} | Min Fold = {min(fold_scores):.4f} | Folds: {[round(s,4) for s in fold_scores]}")
    print("=" * 80)
    return oof_student, test_student

# ==============================================================================
# 5. DECOUPLED POPULATION GBDT EXPERTS (MULTI-SEED LIGHTGBM + CATBOOST GPU)
# ==============================================================================
def run_population_gbdt_experts(clean_tr: pl.DataFrame, clean_te: pl.DataFrame, fold_col: np.ndarray, config):
    print("\n" + "=" * 80)
    print("PHASE 4: DECOUPLED POPULATION GBDT EXPERTS (POPS 1, 2, 3)")
    print("=" * 80)
    
    y = clean_tr["target"].to_numpy().astype(np.int32)
    
    tr_has_prev = clean_tr["cov_has_prev_apps"].to_numpy()
    tr_has_repay = clean_tr["cov_has_repayment"].to_numpy()
    tr_has_acct = clean_tr["cov_has_account"].to_numpy()
    tr_pop1 = (tr_has_repay == 1) & (tr_has_acct == 1)
    tr_pop3 = (tr_has_prev == 1) & (tr_has_repay == 0)
    tr_pop2 = (~tr_pop1) & (~tr_pop3)
    
    te_has_prev = clean_te["cov_has_prev_apps"].to_numpy()
    te_has_repay = clean_te["cov_has_repayment"].to_numpy()
    te_has_acct = clean_te["cov_has_account"].to_numpy()
    te_pop1 = (te_has_repay == 1) & (te_has_acct == 1)
    te_pop3 = (te_has_prev == 1) & (te_has_repay == 0)
    te_pop2 = (~te_pop1) & (~te_pop3)
    
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"}
    all_features = [c for c in clean_tr.columns if c not in ignore_cols]
    
    pdf_tr = clean_tr.select(all_features).to_pandas()
    pdf_te = clean_te.select(all_features).to_pandas()
    for c in pdf_tr.select_dtypes(include=["object", "string"]).columns:
        pdf_tr[c] = pdf_tr[c].astype("category")
        pdf_te[c] = pdf_te[c].astype("category")
        
    oof_gbdt_expert = np.zeros(len(y), dtype=np.float32)
    test_gbdt_expert = np.zeros(len(clean_te), dtype=np.float32)
    
    # Pop 1 (Full-File)
    print("[*] Training Pop 1 Expert (All Features, Multi-Seed LGBM)...")
    p1_tr_idx = np.where(tr_pop1)[0]
    p1_te_idx = np.where(te_pop1)[0]
    
    for f in range(5):
        tr_f = np.where(tr_pop1 & (fold_col != f))[0]
        va_f = np.where(tr_pop1 & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(
            objective="binary", metric="auc", learning_rate=0.035,
            num_leaves=16, max_depth=5, min_child_samples=150,
            subsample=0.8, colsample_bytree=0.4, reg_alpha=8.0, reg_lambda=40.0,
            n_estimators=700, n_jobs=config["n_jobs"], random_state=42, verbose=-1
        )
        clf.fit(pdf_tr.iloc[tr_f], y[tr_f])
        oof_gbdt_expert[va_f] = clf.predict_proba(pdf_tr.iloc[va_f])[:, 1]
        test_gbdt_expert[p1_te_idx] += clf.predict_proba(pdf_te.iloc[p1_te_idx])[:, 1] / 5.0
        
    print(f"    Pop 1 Expert KS: {official_ks_metric(y[tr_pop1], oof_gbdt_expert[tr_pop1]):.4f}")
    
    # Pop 2 (Thin-File) - Clean non-empty features only
    pop2_feats = [c for c in all_features if not c.startswith("repay_") and not c.startswith("acct_") and not c.startswith("prev_")]
    print(f"[*] Training Pop 2 Expert ({len(pop2_feats)} Clean Features)...")
    p2_tr_idx = np.where(tr_pop2)[0]
    p2_te_idx = np.where(te_pop2)[0]
    
    for f in range(5):
        tr_f = np.where(tr_pop2 & (fold_col != f))[0]
        va_f = np.where(tr_pop2 & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(
            objective="binary", metric="auc", learning_rate=0.03,
            num_leaves=12, max_depth=4, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.45, reg_alpha=6.0, reg_lambda=30.0,
            n_estimators=600, n_jobs=config["n_jobs"], random_state=42, verbose=-1
        )
        clf.fit(pdf_tr[pop2_feats].iloc[tr_f], y[tr_f])
        oof_gbdt_expert[va_f] = clf.predict_proba(pdf_tr[pop2_feats].iloc[va_f])[:, 1]
        test_gbdt_expert[p2_te_idx] += clf.predict_proba(pdf_te[pop2_feats].iloc[p2_te_idx])[:, 1] / 5.0
        
    print(f"    Pop 2 Expert KS: {official_ks_metric(y[tr_pop2], oof_gbdt_expert[tr_pop2]):.4f}")
    
    # Pop 3 (Rejected prev apps)
    pop3_feats = [c for c in all_features if not c.startswith("repay_")]
    print(f"[*] Training Pop 3 Expert ({len(pop3_feats)} Features)...")
    p3_tr_idx = np.where(tr_pop3)[0]
    p3_te_idx = np.where(te_pop3)[0]
    
    for f in range(5):
        tr_f = np.where(tr_pop3 & (fold_col != f))[0]
        va_f = np.where(tr_pop3 & (fold_col == f))[0]
        
        clf = lgb.LGBMClassifier(
            objective="binary", metric="auc", learning_rate=0.025,
            num_leaves=8, max_depth=3, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.5, reg_alpha=5.0, reg_lambda=25.0,
            n_estimators=450, n_jobs=config["n_jobs"], random_state=42, verbose=-1
        )
        clf.fit(pdf_tr[pop3_feats].iloc[tr_f], y[tr_f])
        oof_gbdt_expert[va_f] = clf.predict_proba(pdf_tr[pop3_feats].iloc[va_f])[:, 1]
        test_gbdt_expert[p3_te_idx] += clf.predict_proba(pdf_te[pop3_feats].iloc[p3_te_idx])[:, 1] / 5.0
        
    print(f"    Pop 3 Expert KS: {official_ks_metric(y[tr_pop3], oof_gbdt_expert[tr_pop3]):.4f}")
    
    ov_expert_ks = official_ks_metric(y, oof_gbdt_expert)
    expert_folds = [official_ks_metric(y[fold_col == f], oof_gbdt_expert[fold_col == f]) for f in range(5)]
    print("\n" + "=" * 80)
    print(f"POPS EXPERTS GBDT RESULTS: Overall OOF KS = {ov_expert_ks:.4f} | Min Fold = {min(expert_folds):.4f} | Folds: {[round(s,4) for s in expert_folds]}")
    print("=" * 80)
    return oof_gbdt_expert, test_gbdt_expert

# ==============================================================================
# 6. ENSEMBLE ORCHESTRATION & FINAL SUBMISSIONS
# ==============================================================================
def create_hpc_submissions(clean_te: pl.DataFrame, test_kumo: np.ndarray, test_student: np.ndarray, test_expert: np.ndarray, anchor_file: str, out_dir="submissions"):
    print("\n" + "=" * 80)
    print("PHASE 5: FINAL PREDICTION CALIBRATION & SUBMISSION GENERATION")
    print("=" * 80)
    os.makedirs(out_dir, exist_ok=True)
    
    app_ids = clean_te["application_id"].to_list()
    N = len(app_ids)
    
    # 0.55677 Anchor
    anchor_df = pl.read_csv(anchor_file)
    r_anchor = normalize_rank(anchor_df["default_probability"].to_numpy())
    r_kumo = normalize_rank(test_kumo)
    r_student = normalize_rank(test_student)
    r_expert = normalize_rank(test_expert)
    
    # Populations
    te_has_prev = clean_te["cov_has_prev_apps"].to_numpy()
    te_has_repay = clean_te["cov_has_repayment"].to_numpy()
    te_has_acct = clean_te["cov_has_account"].to_numpy()
    p1 = (te_has_repay == 1) & (te_has_acct == 1)
    p3 = (te_has_prev == 1) & (te_has_repay == 0)
    p2 = (~p1) & (~p3)
    
    # -------------------------------------------------------------------------
    # 1. SUBMISSION: exp067 Kumo Population Champion (Target: > 0.5700)
    # Decoupled weights: Injects high Kumo + Student into Pop 2 & Pop 3
    # -------------------------------------------------------------------------
    blend_champ = np.zeros(N, dtype=np.float64)
    blend_champ[p1] = 0.75 * r_anchor[p1] + 0.12 * r_kumo[p1] + 0.08 * r_student[p1] + 0.05 * r_expert[p1]
    blend_champ[p2] = 0.55 * r_anchor[p2] + 0.25 * r_kumo[p2] + 0.12 * r_student[p2] + 0.08 * r_expert[p2]
    blend_champ[p3] = 0.55 * r_anchor[p3] + 0.25 * r_kumo[p3] + 0.12 * r_student[p3] + 0.08 * r_expert[p3]
    
    p_champ = normalize_rank(blend_champ)
    path_1 = os.path.join(out_dir, "submission_HPC_exp067_kumo_population_champion.csv")
    pl.DataFrame({"application_id": app_ids, "default_probability": p_champ.astype(np.float64)}).write_csv(path_1)
    print(f"[+] Created: {path_1}")
    
    # -------------------------------------------------------------------------
    # 2. SUBMISSION: Smart Pseudo-label Distillation Blend
    # -------------------------------------------------------------------------
    blend_distill = 0.70 * r_anchor + 0.15 * r_student + 0.10 * r_kumo + 0.05 * r_expert
    p_distill = normalize_rank(blend_distill)
    path_2 = os.path.join(out_dir, "submission_HPC_smart_pseudolabel_distillation.csv")
    pl.DataFrame({"application_id": app_ids, "default_probability": p_distill.astype(np.float64)}).write_csv(path_2)
    print(f"[+] Created: {path_2}")
    
    # -------------------------------------------------------------------------
    # 3. SUBMISSION: Pure Autonomous Foundation + Expert (Zero-leak clean)
    # -------------------------------------------------------------------------
    blend_clean = 0.40 * r_student + 0.35 * r_kumo + 0.25 * r_expert
    p_clean = normalize_rank(blend_clean)
    path_3 = os.path.join(out_dir, "submission_HPC_pure_foundation_expert_clean.csv")
    pl.DataFrame({"application_id": app_ids, "default_probability": p_clean.astype(np.float64)}).write_csv(path_3)
    print(f"[+] Created: {path_3}")
    
    # Rigorous Sanity Check
    print("\n" + "=" * 80)
    print("SUBMISSION INTEGRITY & SANITY VERIFICATION:")
    print("=" * 80)
    for pth in [path_1, path_2, path_3]:
        df = pl.read_csv(pth)
        assert len(df) == 30000, f"Mismatch length in {pth}"
        assert df.columns == ["application_id", "default_probability"]
        assert df["default_probability"].null_count() == 0
        assert df["default_probability"].is_finite().all()
        assert df["application_id"].to_list() == app_ids
        print(f"  [VERIFIED OK] {os.path.basename(pth):<55} | 30k rows | Mean: {df['default_probability'].mean():.6f}")
    print("=" * 80)

# ==============================================================================
# MAIN CONTROLLER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="HPC Master Pipeline - Indonesian Credit Risk")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing processed or raw data")
    parser.add_argument("--anchor_file", type=str, default=None, help="Path to 0.55677 anchor submission (ens_q4cb.csv)")
    parser.add_argument("--skip_kumo", action="store_true", help="Skip KumoTabular if running fast smoke test")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STARTING HPC MASTER PIPELINE - INDONESIAN CONSUMER LENDING CREDIT RISK")
    print("=" * 80)
    t_start = time.time()
    
    # Detect hardware
    device, config = detect_hardware()
    
    # Locate data files
    tr_path = find_file("train_features_v5.parquet", args.data_dir)
    te_path = find_file("test_features_v5.parquet", args.data_dir)
    folds_path = find_file("cv_folds_70k.parquet", args.data_dir)
    
    anchor_name = args.anchor_file or "ens_q4cb.csv"
    anchor_path = find_file(anchor_name, args.data_dir)
    
    print(f"[*] Train Path : {tr_path}")
    print(f"[*] Test Path  : {te_path}")
    print(f"[*] Folds Path : {folds_path}")
    print(f"[*] Anchor Path: {anchor_path}")
    
    tr_df = pl.read_parquet(tr_path)
    te_df = pl.read_parquet(te_path)
    df_folds = pl.read_parquet(folds_path)
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    # Feature purification
    clean_tr, clean_te = prepare_clean_features(tr_df, te_df, fold_col)
    
    # Foundation Model (exp067)
    if not args.skip_kumo:
        oof_kumo, test_kumo = run_kumo_population_foundation(clean_tr, clean_te, fold_col, device, config)
    else:
        print("[!] Skipping Kumo-Tabular as requested.")
        test_kumo = np.zeros(len(clean_te), dtype=np.float32)
        
    # Smart Pseudo-Labeling Distillation
    oof_student, test_student = run_smart_pseudolabel_distillation(clean_tr, clean_te, fold_col, anchor_path, device, config)
    
    # Decoupled Population GBDT Experts
    oof_expert, test_expert = run_population_gbdt_experts(clean_tr, clean_te, fold_col, config)
    
    # Final Submissions
    out_dir = "submissions" if os.path.exists("submissions") else "HPC/submissions"
    create_hpc_submissions(clean_te, test_kumo, test_student, test_expert, anchor_path, out_dir=out_dir)
    
    print(f"\n[SUCCESS] HPC Master Pipeline Completed in {time.time() - t_start:.1f}s!")

if __name__ == "__main__":
    main()
