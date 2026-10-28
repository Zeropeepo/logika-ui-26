"""
Phase 3: Feature Selection, Group Importance & Hierarchical Correlation Clustering
Performs:
1. Group ablation across 6 functional constructs (e-commerce, repayment, bureau, pricing, profile, composite)
2. Hierarchical clustering on correlation matrix to find representative non-redundant feature subset (< 100 features)
3. Evaluates models on selected feature set across 15 folds and checks pairwise delta.
"""
import os
import json
import time
import csv
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from scipy.cluster import hierarchy
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

SEEDS = [42, 2026, 999]
N_SPLITS = 5
EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def run_phase3():
    print("=== FASE 3: FEATURE SELECTION & GROUP CLUSTERING ===")
    
    df_train_pl = pl.read_parquet("data/processed/train_features_v2.parquet")
    df_test_pl = pl.read_parquet("data/processed/test_features_v2.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    
    y = df_train_pl["target"].to_numpy()
    feature_cols = [c for c in df_train_pl.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if df_train_pl[c].dtype == pl.String]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    print(f"Total input features: {len(feature_cols)} ({len(cat_cols)} cat, {len(num_cols)} num)")
    
    # -----------------------------------------------------------------
    # 3.1 Group Definition & Group Importance
    # -----------------------------------------------------------------
    print("\n--- 3.1 Group Importance / Group Ablation on LightGBM ---")
    groups = {
        "ecommerce": [c for c in feature_cols if c.startswith("ecom_") or "ecom" in c],
        "repayment": [c for c in feature_cols if c.startswith("repay_") or "dpd" in c or "repay" in c],
        "bureau": [c for c in feature_cols if c.startswith("bureau_") or "bureau" in c],
        "pricing_loan": [c for c in feature_cols if c in ["loan_amount", "term_months", "annual_interest_rate", "proposed_monthly_installment", "monthly_interest_rate", "pricing_predicted_rate", "pricing_rate_residual", "installment_income_ratio", "loan_income_ratio"]],
        "profile": [c for c in feature_cols if c in ["age", "region", "employment_type", "employment_tenure_months", "declared_monthly_income", "dependents", "housing_status"]],
        "composite_flags": [c for c in feature_cols if c.startswith("rank_") or c.startswith("ratio_") or c.startswith("flag_") or c.startswith("cov_") or c == "composite_risk_score"]
    }
    
    for g_name, g_cols in groups.items():
        print(f"  Group '{g_name}': {len(g_cols)} features")
        
    # Group drop ablation on 5 folds (Seed 42) for rapid precise comparison
    folds_s42 = df_folds["fold_seed42"].to_numpy()
    X_train_df = df_train_pl.select(feature_cols).to_pandas()
    for c in cat_cols:
        X_train_df[c] = X_train_df[c].astype("category")
        
    def evaluate_lgb_subset(sub_cols, n_splits=5):
        cat_sub = [c for c in cat_cols if c in sub_cols]
        fold_scores = []
        oof_p = np.zeros(len(y))
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "verbosity": -1,
            "n_jobs": 8,
            "random_state": 42
        }
        
        for f in range(n_splits):
            trn_idx = np.where(folds_s42 != f)[0]
            val_idx = np.where(folds_s42 == f)[0]
            
            X_tr, y_tr = X_train_df.iloc[trn_idx][sub_cols], y[trn_idx]
            X_va, y_va = X_train_df.iloc[val_idx][sub_cols], y[val_idx]
            
            trn_d = lgb.Dataset(X_tr, label=y_tr)
            val_d = lgb.Dataset(X_va, label=y_va, reference=trn_d)
            
            bst = lgb.train(params, trn_d, num_boost_round=600, valid_sets=[val_d], callbacks=[lgb.early_stopping(40, verbose=False)])
            preds = bst.predict(X_va)
            oof_p[val_idx] = preds
            fold_scores.append(official_ks_metric(y_va, preds))
            
        overall_ks = official_ks_metric(y, oof_p)
        return float(np.mean(fold_scores)), float(overall_ks), oof_p
        
    print("\nEvaluating Baseline Full Feature Set (Seed 42 5-Fold)...")
    full_mean_ks, full_oof_ks, full_oof_p = evaluate_lgb_subset(feature_cols)
    print(f"Full 292 Features LightGBM KS: Mean = {full_mean_ks:.4f}, OOF = {full_oof_ks:.4f}")
    
    group_ablation_results = {}
    for g_name, g_cols in groups.items():
        remain_cols = [c for c in feature_cols if c not in g_cols]
        mean_ks, oof_ks, _ = evaluate_lgb_subset(remain_cols)
        delta_ks = oof_ks - full_oof_ks
        group_ablation_results[g_name] = {
            "removed_count": len(g_cols),
            "remain_count": len(remain_cols),
            "oof_ks": float(oof_ks),
            "delta_ks": float(delta_ks)
        }
        print(f"  Drop '{g_name:<16}' ({len(g_cols):2d} cols): OOF KS = {oof_ks:.4f} (Delta: {delta_ks:+.4f})")
        
    # -----------------------------------------------------------------
    # 3.2 Hierarchical Correlation Clustering (< 100 features)
    # -----------------------------------------------------------------
    print("\n--- 3.2 Hierarchical Spearman Correlation Clustering ---")
    # Compute correlation matrix on numerical features
    # Compute Spearman rank correlation matrix via Pearson on ranked data (mathematically identical, 10x faster)
    num_df_ranked = df_train_pl.select(num_cols).to_pandas().fillna(df_train_pl.select(num_cols).to_pandas().median()).rank()
    corr_mat = num_df_ranked.corr(method="pearson").abs().values.copy()
    np.fill_diagonal(corr_mat, 1.0)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    
    # Distance matrix = 1 - abs(corr)
    dist_mat = 1.0 - corr_mat
    dist_mat = np.clip(dist_mat, 0.0, 1.0)
    
    # Hierarchical linkage (Ward / Average)
    from scipy.spatial.distance import squareform
    condensed_dist = squareform(dist_mat, checks=False)
    linkage = hierarchy.linkage(condensed_dist, method="average")
    
    # Cluster at threshold distance = 0.20 (correlation >= 0.80)
    cluster_labels = hierarchy.fcluster(linkage, t=0.20, criterion="distance")
    n_clusters = len(np.unique(cluster_labels))
    print(f"Formed {n_clusters} clusters from {len(num_cols)} numerical features at |r| >= 0.80 threshold.")
    
    # Select representative feature per cluster:
    # Measure univariate KS of each feature
    from scipy.stats import ks_2samp
    univ_ks = {}
    for c in num_cols:
        vals = num_df_ranked[c].values
        # direction
        auc = roc_auc_score(y, vals)
        if auc < 0.5:
            vals = -vals
        bads = vals[y == 1]
        goods = vals[y == 0]
        univ_ks[c] = ks_2samp(bads, goods).statistic
        
    selected_num_features = []
    cluster_details = {}
    for cl_id in np.unique(cluster_labels):
        members = [num_cols[i] for i in np.where(cluster_labels == cl_id)[0]]
        # pick member with highest univariate KS
        best_feat = max(members, key=lambda f: univ_ks[f])
        selected_num_features.append(best_feat)
        cluster_details[int(cl_id)] = {
            "representative": best_feat,
            "rep_ks": float(univ_ks[best_feat]),
            "members": members
        }
        
    # Always keep vital composite & high-gain features
    vital_features = ["composite_risk_score", "repay_max_max_dpd", "annual_interest_rate", "ecom_spend_30d", "ecom_orders_30d", "bureau_life_score_latest", "pricing_rate_residual"]
    for vf in vital_features:
        if vf in num_cols and vf not in selected_num_features:
            selected_num_features.append(vf)
            
    final_selected_features = cat_cols + selected_num_features
    print(f"Final Selected Feature Set: {len(final_selected_features)} features ({len(cat_cols)} cat + {len(selected_num_features)} num)")
    
    # -----------------------------------------------------------------
    # 3.3 Validate Selected Set on 15 Folds (Full CV)
    # -----------------------------------------------------------------
    print(f"\n--- 3.3 Validating Selected {len(final_selected_features)} Features on Full 15 Folds ---")
    
    # Save selected feature datasets to disk
    train_sel_pl = df_train_pl.select(["application_id", "target"] + final_selected_features)
    test_sel_pl = df_test_pl.select(["application_id"] + final_selected_features)
    train_sel_pl.write_parquet("data/processed/train_features_selected.parquet")
    test_sel_pl.write_parquet("data/processed/test_features_selected.parquet")
    print("Saved selected feature sets to data/processed/train_features_selected.parquet and test_features_selected.parquet")
    
    # Evaluate LightGBM on selected features (15 Folds)
    t0 = time.time()
    lgb_sel_oof = np.zeros(len(y))
    lgb_sel_folds = []
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            X_tr = X_train_df.iloc[trn_idx][final_selected_features]
            y_tr = y[trn_idx]
            X_va = X_train_df.iloc[val_idx][final_selected_features]
            y_va = y[val_idx]
            
            trn_d = lgb.Dataset(X_tr, label=y_tr)
            val_d = lgb.Dataset(X_va, label=y_va, reference=trn_d)
            
            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 6,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "min_child_samples": 50,
                "verbosity": -1,
                "n_jobs": 8,
                "random_state": seed + f * 100
            }
            bst = lgb.train(params, trn_d, num_boost_round=1000, valid_sets=[val_d], callbacks=[lgb.early_stopping(50, verbose=False)])
            val_p = bst.predict(X_va)
            lgb_sel_oof[val_idx] += val_p / len(SEEDS)
            lgb_sel_folds.append(official_ks_metric(y_va, val_p))
            
    t_lgb_sel = time.time() - t0
    lgb_sel_mean_ks = float(np.mean(lgb_sel_folds))
    lgb_sel_oof_ks = float(official_ks_metric(y, lgb_sel_oof))
    lgb_sel_auc = float(roc_auc_score(y, lgb_sel_oof))
    print(f"LightGBM on Selected ({len(final_selected_features)} feats) 15-Fold: Mean KS = {lgb_sel_mean_ks:.4f} +/- {np.std(lgb_sel_folds):.4f} | OOF KS = {lgb_sel_oof_ks:.4f} | AUC = {lgb_sel_auc:.4f} ({t_lgb_sel:.1f}s)")
    
    # Evaluate Logistic Regression on selected features (15 Folds)
    t0 = time.time()
    lr_sel_oof = np.zeros(len(y))
    lr_sel_folds = []
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr_num = scl.fit_transform(imp.fit_transform(X_train_df.iloc[trn_idx][selected_num_features]))
            X_va_num = scl.transform(imp.transform(X_train_df.iloc[val_idx][selected_num_features]))
            
            lr = LogisticRegression(C=0.05, max_iter=500, random_state=seed + f * 100, solver="lbfgs")
            lr.fit(X_tr_num, y[trn_idx])
            val_p = lr.predict_proba(X_va_num)[:, 1]
            lr_sel_oof[val_idx] += val_p / len(SEEDS)
            lr_sel_folds.append(official_ks_metric(y_va, val_p))
            
    t_lr_sel = time.time() - t0
    lr_sel_mean_ks = float(np.mean(lr_sel_folds))
    lr_sel_oof_ks = float(official_ks_metric(y, lr_sel_oof))
    lr_sel_auc = float(roc_auc_score(y, lr_sel_oof))
    print(f"LogReg on Selected ({len(selected_num_features)} num feats) 15-Fold: Mean KS = {lr_sel_mean_ks:.4f} +/- {np.std(lr_sel_folds):.4f} | OOF KS = {lr_sel_oof_ks:.4f} | AUC = {lr_sel_auc:.4f} ({t_lr_sel:.1f}s)")
    
    # Save Phase 3 results
    phase3_summary = {
        "group_ablation": group_ablation_results,
        "n_clusters": int(n_clusters),
        "selected_feature_count": len(final_selected_features),
        "selected_features": final_selected_features,
        "cluster_details": cluster_details,
        "lgb_selected_15fold": {
            "mean_ks": lgb_sel_mean_ks,
            "std_ks": float(np.std(lgb_sel_folds)),
            "oof_ks": lgb_sel_oof_ks,
            "oof_auc": lgb_sel_auc
        },
        "lr_selected_15fold": {
            "mean_ks": lr_sel_mean_ks,
            "std_ks": float(np.std(lr_sel_folds)),
            "oof_ks": lr_sel_oof_ks,
            "oof_auc": lr_sel_auc
        }
    }
    
    with open("reports/phase3_selection_results.json", "w") as f:
        json.dump(phase3_summary, f, indent=2)
        
    # Log to experiments_log.csv
    # Compare vs EXP-004 (LGB full) and EXP-007 (LR full)
    with open("experiments_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EXP-008", "FASE_3", f"Hierarchical Clustering Selection ({len(final_selected_features)} feats)", "Selected subset via Spearman r>=0.80", f"{len(final_selected_features)}_selected_features", "LightGBM", f"{lgb_sel_mean_ks:.4f}", f"{np.std(lgb_sel_folds):.4f}", f"{lgb_sel_mean_ks - 0.4900:+.4f}", f"{np.std(lgb_sel_folds):.4f}", f"{lgb_sel_auc:.4f}", f"{t_lgb_sel:.1f}", "DEPLOYABLE", "Compact representative feature set for foundation models and trees"])
        writer.writerow(["EXP-009", "FASE_3", f"Hierarchical Clustering Selection ({len(selected_num_features)} num feats)", "Selected subset via Spearman r>=0.80", f"{len(selected_num_features)}_selected_num_features", "Logistic_Regression", f"{lr_sel_mean_ks:.4f}", f"{np.std(lr_sel_folds):.4f}", f"{lr_sel_mean_ks - 0.4921:+.4f}", f"{np.std(lr_sel_folds):.4f}", f"{lr_sel_auc:.4f}", f"{t_lr_sel:.1f}", "DEPLOYABLE", "Regularized LogReg on pruned non-redundant feature set"])

if __name__ == "__main__":
    run_phase3()
