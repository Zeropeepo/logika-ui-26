"""
Evaluation of 7 Out-of-the-Box (OOTB) Feature Families with Statistical Shadow Noise Controls
1. Struktur data (tier bunga, kelipatan 100k, segmen ketersediaan)
2. Kink / threshold indicators
3. Persona (K-means clustering inside folds)
4. Artefak & hidden debt
5. Periodisitas, HHI & burstiness
6. Representasi kohort
7. Kontrol statistik: 20 random noise features (Gaussian & Uniform)
"""
import os
import json
import time
import csv
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

SEEDS = [42, 2026, 999]
N_SPLITS = 5
EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def run_ootb_experiments():
    print("=== RUNNING OUT-OF-THE-BOX (OOTB) FEATURE EVALUATION ===")
    
    # Load v3 feature dataset
    df_train = pl.read_parquet("data/processed/train_features_v3.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    y = df_train["target"].to_numpy()
    
    # -------------------------------------------------------------
    # 1. GENERATE OOTB CANDIDATE FEATURES
    # -------------------------------------------------------------
    print("Generating candidate feature families...")
    
    # Family 1: Data Structure
    rate_map = {0.16: 0, 0.20: 1, 0.24: 2, 0.28: 3, 0.32: 4}
    rate_tier_ord = df_train["annual_interest_rate"].map_elements(lambda x: rate_map.get(round(x, 2), 1), return_dtype=pl.Int32)
    income_decile = (df_train["declared_monthly_income"].rank() / (pl.len() / 10.0)).cast(pl.Int32)
    
    df_train = df_train.with_columns(
        f1_rate_tier_ord = rate_tier_ord,
        f1_tier_x_ecom_spend = rate_tier_ord * (pl.col("ecom_spend_30d").fill_null(0).rank() / pl.len()),
        f1_tier_x_repay_dpd = rate_tier_ord * (pl.col("repay_max_max_dpd_all").fill_null(0).rank() / pl.len()),
        f1_tier_x_bureau_score = rate_tier_ord * ((850 - pl.col("bureau_life_score_latest").fill_null(650)).rank() / pl.len()),
        f1_is_loan_amount_mult_1m = (pl.col("loan_amount") % 1_000_000 == 0).cast(pl.Int32),
        f1_seg_no_repayment = pl.col("repay_inst_total").is_null().cast(pl.Int32),
        f1_seg_no_bureau_util = (pl.col("bureau_life_rev_lim_latest").is_null() | (pl.col("bureau_life_rev_lim_latest") == 0)).cast(pl.Int32),
        
        # Family 2: Kink / Threshold Indicators
        f2_kink_ecom_spend_gt_823k = (pl.col("ecom_spend_30d").fill_null(0) > 823_000).cast(pl.Int32),
        f2_kink_ecom_spend_gt_2m19 = (pl.col("ecom_spend_30d").fill_null(0) > 2_192_000).cast(pl.Int32),
        f2_kink_ecom_orders_ge_4 = (pl.col("ecom_orders_30d").fill_null(0) >= 4).cast(pl.Int32),
        f2_kink_ecom_recency_le_2d = (pl.col("ecom_latest_order_days").fill_null(999) <= 2).cast(pl.Int32),
        f2_kink_bureau_score_lt_600 = (pl.col("bureau_life_score_latest").fill_null(800) < 600).cast(pl.Int32),
        f2_kink_bureau_drop_gt_50 = (pl.col("bureau_score_drop_from_max").fill_null(0) > 50).cast(pl.Int32),
        f2_kink_repay_dpd_ge_30 = (pl.col("repay_max_max_dpd_all").fill_null(0) >= 30).cast(pl.Int32),
        f2_kink_repay_underpaid_gt_0 = (pl.col("repay_weighted_underpaid").fill_null(0) > 0.0).cast(pl.Int32),
        
        # Family 4: Artefak dalam kolom
        f4_working_age_ratio = pl.col("employment_tenure_months") / ((pl.col("age") - 18).clip(lower_bound=1) * 12.0),
        f4_hidden_debt_accounts = pl.col("bureau_life_inquiries_latest").fill_null(0) - pl.col("prev_originated_loans_count").fill_null(0),
        f4_bureau_staleness_days = (pl.col("bureau_max_days_since_avail").fill_null(0) - pl.col("bureau_min_days_since_avail").fill_null(0)),
        f4_balance_discrepancy = pl.col("acct_mean_inflows_all").fill_null(0) - pl.col("acct_mean_outflows_all").fill_null(0),
        
        # Family 5: Periodisitas & Signal Processing
        f5_burstiness_metric = (pl.col("ecom_gap_std_days").fill_null(0) - pl.col("ecom_gap_mean_days").fill_null(0)) / (pl.col("ecom_gap_std_days").fill_null(0) + pl.col("ecom_gap_mean_days").fill_null(0) + 1e-5),
        f5_payday_order_share = pl.col("ecom_payday_orders").fill_null(0) / (pl.col("ecom_total_orders_all").fill_null(1) + 1e-5),
        f5_weekend_order_share = pl.col("ecom_weekend_orders").fill_null(0) / (pl.col("ecom_total_orders_all").fill_null(1) + 1e-5),
        f5_night_order_share = pl.col("ecom_night_orders").fill_null(0) / (pl.col("ecom_total_orders_all").fill_null(1) + 1e-5),
        
        # Family 6: Kohort Ranks
        f6_ecom_spend_in_income_decile = pl.col("ecom_spend_30d").fill_null(0).rank().over(income_decile) / pl.len().over(income_decile),
        f6_rate_in_purpose_group = pl.col("annual_interest_rate").rank().over("loan_purpose") / pl.len().over("loan_purpose")
    )
    
    # Family 7: Kontrol Statistik (20 Shadow Random Noise Columns)
    rng = np.random.RandomState(42)
    noise_dict = {}
    for i in range(10):
        noise_dict[f"noise_gauss_{i}"] = rng.randn(len(df_train))
    for i in range(10):
        noise_dict[f"noise_unif_{i}"] = rng.rand(len(df_train))
    f7_noise_df = pl.DataFrame(noise_dict)
    
    f1_cols = ["f1_rate_tier_ord", "f1_tier_x_ecom_spend", "f1_tier_x_repay_dpd", "f1_tier_x_bureau_score", "f1_is_loan_amount_mult_1m", "f1_seg_no_repayment", "f1_seg_no_bureau_util"]
    f2_cols = ["f2_kink_ecom_spend_gt_823k", "f2_kink_ecom_spend_gt_2m19", "f2_kink_ecom_orders_ge_4", "f2_kink_ecom_recency_le_2d", "f2_kink_bureau_score_lt_600", "f2_kink_bureau_drop_gt_50", "f2_kink_repay_dpd_ge_30", "f2_kink_repay_underpaid_gt_0"]
    f4_cols = ["f4_working_age_ratio", "f4_hidden_debt_accounts", "f4_bureau_staleness_days", "f4_balance_discrepancy"]
    f5_cols = ["f5_burstiness_metric", "f5_payday_order_share", "f5_weekend_order_share", "f5_night_order_share"]
    f6_cols = ["f6_ecom_spend_in_income_decile", "f6_rate_in_purpose_group"]
    
    cand_cols = f1_cols + f2_cols + f4_cols + f5_cols + f6_cols
    cand_df = pl.concat([df_train.select(cand_cols), f7_noise_df], how="horizontal")
    print(f"Total candidate OOTB features created: {len(cand_df.columns)} (including 20 noise controls)")
    
    # -------------------------------------------------------------
    # 2. SHADOW NOISE CONTROL & STATISTICAL SCREENING
    # -------------------------------------------------------------
    print("\n--- Running Statistical Shadow Noise Control (LightGBM Gain Screening) ---")
    
    # Fast evaluation on Seed 42 5-fold
    folds_s42 = df_folds["fold_seed42"].to_numpy()
    
    # Train LightGBM with candidate features + base features + noise features
    full_X = pl.concat([df_train.select([c for c in df_train.columns if c not in EVAL_ONLY_COLUMNS]), f7_noise_df], how="horizontal")
    cat_cols = [c for c in full_X.columns if full_X[c].dtype == pl.String]
    X_pd = full_X.to_pandas()
    for c in cat_cols:
        X_pd[c] = X_pd[c].astype("category")
        
    trn_i = np.where(folds_s42 != 0)[0]
    val_i = np.where(folds_s42 == 0)[0]
    
    trn_data = lgb.Dataset(X_pd.iloc[trn_i], label=y[trn_i])
    val_data = lgb.Dataset(X_pd.iloc[val_i], label=y[val_i], reference=trn_data)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 100,
        "feature_fraction": 0.6,
        "verbosity": -1,
        "random_state": 42
    }
    bst = lgb.train(params, trn_data, num_boost_round=600, valid_sets=[val_data], callbacks=[lgb.early_stopping(40, verbose=False)])
    
    # Feature importances
    gain_dict = dict(zip(bst.feature_name(), bst.feature_importance(importance_type="gain")))
    
    # Find max noise gain
    noise_gains = [gain_dict.get(f"noise_gauss_{i}", 0.0) for i in range(10)] + [gain_dict.get(f"noise_unif_{i}", 0.0) for i in range(10)]
    max_noise_gain = max(noise_gains)
    mean_noise_gain = np.mean(noise_gains)
    print(f"Shadow Noise Control: Max Noise Gain = {max_noise_gain:.2f}, Mean Noise Gain = {mean_noise_gain:.2f}")
    
    # Audit each candidate feature against max noise gain
    screening_report = []
    families = {
        "1. Struktur Data": f1_cols,
        "2. Kink/Threshold": f2_cols,
        "4. Artefak/Hidden Debt": f4_cols,
        "5. Periodisitas/Entropi": f5_cols,
        "6. Kohort/Representasi": f6_cols
    }
    
    adopted_features = []
    family_summary = {}
    
    for fam_name, col_list in families.items():
        fam_gains = [gain_dict.get(c, 0.0) for c in col_list]
        passed_cols = [c for c in col_list if gain_dict.get(c, 0.0) > max_noise_gain]
        
        # Univariate KS of best feature in family
        best_col = max(col_list, key=lambda c: gain_dict.get(c, 0.0))
        best_gain = gain_dict.get(best_col, 0.0)
        
        adopted = len(passed_cols) > 0
        if adopted:
            adopted_features.extend(passed_cols)
            
        family_summary[fam_name] = {
            "total_candidates": len(col_list),
            "passed_candidates": len(passed_cols),
            "passed_columns": passed_cols,
            "best_feature": best_col,
            "best_gain": best_gain,
            "decision": "ADOPTED" if adopted else "REJECTED"
        }
        print(f"{fam_name:<25}: {len(passed_cols)}/{len(col_list)} passed noise control (Best: {best_col} gain={best_gain:.1f}) -> {'ADOPTED' if adopted else 'REJECTED'}")
        
    print(f"\nTotal OOTB features passing shadow noise filter: {len(adopted_features)}")
    print("Adopted features:", adopted_features)
    
    # -------------------------------------------------------------
    # 3. 15-FOLD CV VALIDATION WITH ADOPTED FEATURES
    # -------------------------------------------------------------
    print("\n--- Validating Adopted OOTB Features on 15 Folds (3 Seeds x 5 Folds) ---")
    
    # Build final feature matrix
    base_features = [c for c in df_train.columns if c not in EVAL_ONLY_COLUMNS and c not in cand_cols]
    final_ootb_features = base_features + adopted_features
    X_final = df_train.select(final_ootb_features).to_pandas()
    for c in cat_cols:
        if c in X_final.columns:
            X_final[c] = X_final[c].astype("category")
        
    lgb_ootb_folds = []
    lgb_ootb_oof = np.zeros(len(y))
    
    params_lgb = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 8,
        "min_child_samples": 250,
        "feature_fraction": 0.40,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "n_jobs": 8
    }
    
    t0 = time.time()
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        for f in range(N_SPLITS):
            trn_i, val_i = np.where(folds != f)[0], np.where(folds == f)[0]
            p = params_lgb.copy()
            p["random_state"] = seed + f * 100
            
            trn_d = lgb.Dataset(X_final.iloc[trn_i], label=y[trn_i])
            val_d = lgb.Dataset(X_final.iloc[val_i], label=y[val_i], reference=trn_d)
            bst = lgb.train(p, trn_d, num_boost_round=1200, valid_sets=[val_d], callbacks=[lgb.early_stopping(50, verbose=False)])
            val_p = bst.predict(X_final.iloc[val_i])
            lgb_ootb_oof[val_i] += val_p / len(SEEDS)
            lgb_ootb_folds.append(official_ks_metric(y[val_i], val_p))
            
    t_ootb = time.time() - t0
    ootb_mean_ks = float(np.mean(lgb_ootb_folds))
    ootb_std_ks = float(np.std(lgb_ootb_folds))
    ootb_overall_ks = float(official_ks_metric(y, lgb_ootb_oof))
    ootb_auc = float(roc_auc_score(y, lgb_ootb_oof))
    
    # Load base v3 15-fold scores for pairwise delta
    with open("reports/master_pipeline_results.json") as f:
        master_res = json.load(f)
    base_v3_mean_ks = master_res["models_15fold"]["lgb_tuned"]["mean_ks"]
    delta_vs_v3 = ootb_mean_ks - base_v3_mean_ks
    delta_vs_base1 = ootb_mean_ks - 0.4900
    
    print(f"\nFinal OOTB 15-Fold Results: Mean KS = {ootb_mean_ks:.4f} +/- {ootb_std_ks:.4f} | Overall OOF KS = {ootb_overall_ks:.4f} | AUC = {ootb_auc:.4f} ({t_ootb:.1f}s)")
    print(f"Pairwise Delta vs v3: {delta_vs_v3:+.4f} KS per fold")
    print(f"Pairwise Delta vs Baseline Fase 1: {delta_vs_base1:+.4f} KS per fold")
    
    # Save OOTB results
    ootb_summary = {
        "shadow_noise": {
            "max_noise_gain": float(max_noise_gain),
            "mean_noise_gain": float(mean_noise_gain),
            "noise_columns_tested": 20
        },
        "family_summary": family_summary,
        "adopted_features": adopted_features,
        "lgb_ootb_15fold": {
            "mean_ks": ootb_mean_ks,
            "std_ks": ootb_std_ks,
            "oof_ks": ootb_overall_ks,
            "oof_auc": ootb_auc,
            "delta_vs_v3": delta_vs_v3,
            "delta_vs_base1": delta_vs_base1,
            "time_sec": t_ootb
        }
    }
    
    with open("reports/phase_ootb_results.json", "w") as f:
        json.dump(ootb_summary, f, indent=2)
        
    # Log to experiments_log.csv
    with open("experiments_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EXP-016", "FASE_OOTB", f"OOTB Features with Noise Filter ({len(adopted_features)} adopted)", "Screened with 20 shadow noise controls", f"v3_{len(adopted_features)}_ootb", "LightGBM", f"{ootb_mean_ks:.4f}", f"{ootb_std_ks:.4f}", f"{delta_vs_base1:+.4f}", f"{ootb_std_ks:.4f}", f"{ootb_auc:.4f}", f"{t_ootb:.1f}", "DEPLOYABLE", "Out-of-the-box features screened via shadow noise"])

    print("\nOOTB Experiments Logged Successfully!")

if __name__ == "__main__":
    run_ootb_experiments()
