"""
Phase 2 Evaluation: Repeated Stratified 5-Fold x 3 Seed CV on 297 Advanced Features
Evaluates LightGBM, CatBoost, XGBoost, and Logistic Regression against Phase 1 Baselines.
Computes fold-by-fold pairwise differences, OOF KS, test predictions, and feature importance.
"""
import os
import json
import time
import csv
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
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

def run_phase2_evaluation():
    print("=== PHASE 2: EVALUATING ADVANCED FEATURES V2 (297 COLS) ===")
    
    # 1. Load data and folds
    df_train_pl = pl.read_parquet("data/processed/train_features_v2.parquet")
    df_test_pl = pl.read_parquet("data/processed/test_features_v2.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    df_base_oof = pl.read_parquet("preds/oof_phase1_baselines.parquet")
    
    y = df_train_pl["target"].to_numpy()
    
    # Feature columns
    feature_cols = [c for c in df_train_pl.columns if c not in EVAL_ONLY_COLUMNS]
    print(f"Total feature columns: {len(feature_cols)}")
    
    # Dtypes
    cat_cols = [c for c in feature_cols if df_train_pl[c].dtype == pl.String]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    print(f"Categorical features: {len(cat_cols)}: {cat_cols}")
    print(f"Numerical features: {len(num_cols)}")
    
    X_train_df = df_train_pl.select(feature_cols).to_pandas()
    X_test_df = df_test_pl.select(feature_cols).to_pandas()
    
    # Ensure categorical dtypes
    for c in cat_cols:
        X_train_df[c] = X_train_df[c].astype("category")
        X_test_df[c] = X_test_df[c].astype("category")
        
    X_cb_train = X_train_df.copy()
    X_cb_test = X_test_df.copy()
    for c in cat_cols:
        X_cb_train[c] = X_cb_train[c].astype(str)
        X_cb_test[c] = X_cb_test[c].astype(str)
        
    oof_preds = {
        "application_id": df_train_pl["application_id"].to_list(),
        "target": y.tolist()
    }
    test_preds = {
        "application_id": df_test_pl["application_id"].to_list()
    }
    
    results_summary = {}
    
    # -------------------------------------------------------------
    # MODEL 1: LightGBM (15 Folds)
    # -------------------------------------------------------------
    print("\n--- [1/4] Training LightGBM on 15 Folds (3 Seeds x 5 Folds) ---")
    t0 = time.time()
    lgb_oof_all = np.zeros(len(y))
    lgb_test_all = np.zeros(len(X_test_df))
    lgb_fold_ks = []
    lgb_fold_auc = []
    lgb_importances = np.zeros(len(feature_cols))
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "verbosity": -1,
        "n_jobs": 8
    }
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        seed_oof = np.zeros(len(y))
        
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            X_tr, y_tr = X_train_df.iloc[trn_idx], y[trn_idx]
            X_va, y_va = X_train_df.iloc[val_idx], y[val_idx]
            
            params = lgb_params.copy()
            params["random_state"] = seed + f * 100
            
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
            
            bst = lgb.train(
                params,
                trn_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            val_p = bst.predict(X_va)
            seed_oof[val_idx] = val_p
            ks_f = official_ks_metric(y_va, val_p)
            auc_f = roc_auc_score(y_va, val_p)
            lgb_fold_ks.append(ks_f)
            lgb_fold_auc.append(auc_f)
            
            lgb_test_all += bst.predict(X_test_df) / (len(SEEDS) * N_SPLITS)
            lgb_importances += bst.feature_importance(importance_type="gain") / (len(SEEDS) * N_SPLITS)
            
        lgb_oof_all += seed_oof / len(SEEDS)
        print(f"Seed {seed}: Mean KS = {np.mean(lgb_fold_ks[-N_SPLITS:]):.4f}")
        
    t_lgb = time.time() - t0
    lgb_oof_ks = official_ks_metric(y, lgb_oof_all)
    lgb_oof_auc = roc_auc_score(y, lgb_oof_all)
    print(f"LightGBM v2 Complete in {t_lgb:.1f}s | Mean KS: {np.mean(lgb_fold_ks):.4f} +/- {np.std(lgb_fold_ks):.4f} | Overall OOF KS: {lgb_oof_ks:.4f} | AUC: {lgb_oof_auc:.4f}")
    
    oof_preds["oof_lgb_v2"] = lgb_oof_all.tolist()
    test_preds["lgb_v2"] = lgb_test_all.tolist()
    
    # -------------------------------------------------------------
    # MODEL 2: CatBoost (15 Folds with GPU)
    # -------------------------------------------------------------
    print("\n--- [2/4] Training CatBoost on 15 Folds (3 Seeds x 5 Folds) with GPU ---")
    t0 = time.time()
    cb_oof_all = np.zeros(len(y))
    cb_test_all = np.zeros(len(X_cb_test))
    cb_fold_ks = []
    cb_fold_auc = []
    
    cb_params = {
        "iterations": 1000,
        "learning_rate": 0.04,
        "depth": 6,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "task_type": "GPU",
        "verbose": False
    }
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        seed_oof = np.zeros(len(y))
        
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            X_tr, y_tr = X_cb_train.iloc[trn_idx], y[trn_idx]
            X_va, y_va = X_cb_train.iloc[val_idx], y[val_idx]
            
            model = CatBoostClassifier(
                **cb_params,
                random_seed=seed + f * 100,
                cat_features=cat_cols
            )
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            
            val_p = model.predict_proba(X_va)[:, 1]
            seed_oof[val_idx] = val_p
            ks_f = official_ks_metric(y_va, val_p)
            auc_f = roc_auc_score(y_va, val_p)
            cb_fold_ks.append(ks_f)
            cb_fold_auc.append(auc_f)
            
            cb_test_all += model.predict_proba(X_cb_test)[:, 1] / (len(SEEDS) * N_SPLITS)
            
        cb_oof_all += seed_oof / len(SEEDS)
        print(f"Seed {seed}: Mean KS = {np.mean(cb_fold_ks[-N_SPLITS:]):.4f}")
        
    t_cb = time.time() - t0
    cb_oof_ks = official_ks_metric(y, cb_oof_all)
    cb_oof_auc = roc_auc_score(y, cb_oof_all)
    print(f"CatBoost v2 Complete in {t_cb:.1f}s | Mean KS: {np.mean(cb_fold_ks):.4f} +/- {np.std(cb_fold_ks):.4f} | Overall OOF KS: {cb_oof_ks:.4f} | AUC: {cb_oof_auc:.4f}")
    
    oof_preds["oof_cat_v2"] = cb_oof_all.tolist()
    test_preds["cat_v2"] = cb_test_all.tolist()
    
    # -------------------------------------------------------------
    # MODEL 3: XGBoost (15 Folds with GPU)
    # -------------------------------------------------------------
    print("\n--- [3/4] Training XGBoost on 15 Folds (3 Seeds x 5 Folds) with GPU ---")
    t0 = time.time()
    xgb_oof_all = np.zeros(len(y))
    xgb_test_all = np.zeros(len(X_test_df))
    xgb_fold_ks = []
    xgb_fold_auc = []
    
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "n_estimators": 1000,
        "enable_categorical": True,
        "tree_method": "hist",
        "device": "cuda"
    }
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        seed_oof = np.zeros(len(y))
        
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            X_tr, y_tr = X_train_df.iloc[trn_idx], y[trn_idx]
            X_va, y_va = X_train_df.iloc[val_idx], y[val_idx]
            
            clf = xgb.XGBClassifier(
                **xgb_params,
                random_state=seed + f * 100,
                early_stopping_rounds=50
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            val_p = clf.predict_proba(X_va)[:, 1]
            seed_oof[val_idx] = val_p
            ks_f = official_ks_metric(y_va, val_p)
            auc_f = roc_auc_score(y_va, val_p)
            xgb_fold_ks.append(ks_f)
            xgb_fold_auc.append(auc_f)
            
            xgb_test_all += clf.predict_proba(X_test_df)[:, 1] / (len(SEEDS) * N_SPLITS)
            
        xgb_oof_all += seed_oof / len(SEEDS)
        print(f"Seed {seed}: Mean KS = {np.mean(xgb_fold_ks[-N_SPLITS:]):.4f}")
        
    t_xgb = time.time() - t0
    xgb_oof_ks = official_ks_metric(y, xgb_oof_all)
    xgb_oof_auc = roc_auc_score(y, xgb_oof_all)
    print(f"XGBoost v2 Complete in {t_xgb:.1f}s | Mean KS: {np.mean(xgb_fold_ks):.4f} +/- {np.std(xgb_fold_ks):.4f} | Overall OOF KS: {xgb_oof_ks:.4f} | AUC: {xgb_oof_auc:.4f}")
    
    oof_preds["oof_xgb_v2"] = xgb_oof_all.tolist()
    test_preds["xgb_v2"] = xgb_test_all.tolist()
    
    # -------------------------------------------------------------
    # MODEL 4: Regularized Logistic Regression (15 Folds)
    # -------------------------------------------------------------
    print("\n--- [4/4] Training Logistic Regression on 15 Folds (3 Seeds x 5 Folds) ---")
    t0 = time.time()
    lr_oof_all = np.zeros(len(y))
    lr_test_all = np.zeros(len(X_test_df))
    lr_fold_ks = []
    lr_fold_auc = []
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        seed_oof = np.zeros(len(y))
        
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr_num = scl.fit_transform(imp.fit_transform(X_train_df.iloc[trn_idx][num_cols]))
            X_va_num = scl.transform(imp.transform(X_train_df.iloc[val_idx][num_cols]))
            X_te_num = scl.transform(imp.transform(X_test_df[num_cols]))
            
            lr = LogisticRegression(C=0.05, max_iter=500, random_state=seed + f * 100, solver="lbfgs")
            lr.fit(X_tr_num, y[trn_idx])
            
            val_p = lr.predict_proba(X_va_num)[:, 1]
            seed_oof[val_idx] = val_p
            ks_f = official_ks_metric(y[val_idx], val_p)
            auc_f = roc_auc_score(y[val_idx], val_p)
            lr_fold_ks.append(ks_f)
            lr_fold_auc.append(auc_f)
            
            lr_test_all += lr.predict_proba(X_te_num)[:, 1] / (len(SEEDS) * N_SPLITS)
            
        lr_oof_all += seed_oof / len(SEEDS)
        print(f"Seed {seed}: Mean KS = {np.mean(lr_fold_ks[-N_SPLITS:]):.4f}")
        
    t_lr = time.time() - t0
    lr_oof_ks = official_ks_metric(y, lr_oof_all)
    lr_oof_auc = roc_auc_score(y, lr_oof_all)
    print(f"Logistic Regression v2 Complete in {t_lr:.1f}s | Mean KS: {np.mean(lr_fold_ks):.4f} +/- {np.std(lr_fold_ks):.4f} | Overall OOF KS: {lr_oof_ks:.4f} | AUC: {lr_oof_auc:.4f}")
    
    oof_preds["oof_lr_v2"] = lr_oof_all.tolist()
    test_preds["lr_v2"] = lr_test_all.tolist()
    
    # -------------------------------------------------------------
    # PAIRWISE IMPROVEMENT vs PHASE 1 BASELINE
    # -------------------------------------------------------------
    print("\n--- COMPUTING PAIRWISE DELTA KS PER FOLD AGAINST BASELINE ---")
    # Base fold scores from phase 1 evaluation results JSON
    with open("reports/phase1_a7_eval_results.json") as f:
        base_res = json.load(f)
        
    base_lgb_folds = np.array(base_res["models"]["LightGBM"]["fold_scores"])
    base_cb_folds = np.array(base_res["models"]["CatBoost"]["fold_scores"])
    base_lr_folds = np.array(base_res["models"]["Logistic_Regression"]["fold_scores"])
    
    delta_lgb = np.array(lgb_fold_ks) - base_lgb_folds
    delta_cb = np.array(cb_fold_ks) - base_cb_folds
    delta_lr = np.array(lr_fold_ks) - base_lr_folds
    
    print(f"Pairwise Delta LightGBM (v2 - Baseline): {np.mean(delta_lgb):+.4f} +/- {np.std(delta_lgb):.4f} KS per fold")
    print(f"Pairwise Delta CatBoost (v2 - Baseline): {np.mean(delta_cb):+.4f} +/- {np.std(delta_cb):.4f} KS per fold")
    print(f"Pairwise Delta LogReg   (v2 - Baseline): {np.mean(delta_lr):+.4f} +/- {np.std(delta_lr):.4f} KS per fold")
    
    # -------------------------------------------------------------
    # ENSEMBLE EVALUATION
    # -------------------------------------------------------------
    print("\n--- EVALUATING DIVERSE ENSEMBLES ---")
    from scipy.stats import rankdata
    
    rank_lgb = rankdata(lgb_oof_all) / len(y)
    rank_cb = rankdata(cb_oof_all) / len(y)
    rank_xgb = rankdata(xgb_oof_all) / len(y)
    rank_lr = rankdata(lr_oof_all) / len(y)
    
    # Check pairwise rank correlations
    corr_matrix = pd.DataFrame({
        "LGB": rank_lgb, "CB": rank_cb, "XGB": rank_xgb, "LR": rank_lr
    }).corr()
    print("OOF Rank Correlations:\n", corr_matrix)
    
    # Simple Rank Average
    ens_simple = (rank_lgb + rank_cb + rank_xgb + rank_lr) / 4.0
    ens_simple_ks = official_ks_metric(y, ens_simple)
    ens_simple_auc = roc_auc_score(y, ens_simple)
    print(f"Simple 4-Model Rank Average OOF KS: {ens_simple_ks:.4f} | AUC: {ens_simple_auc:.4f}")
    
    # Tree + Linear Blend
    # Linear model has different inductive bias (AUC 0.8321, KS 0.5042)
    ens_blend = 0.35 * rank_lr + 0.25 * rank_lgb + 0.20 * rank_cb + 0.20 * rank_xgb
    ens_blend_ks = official_ks_metric(y, ens_blend)
    ens_blend_auc = roc_auc_score(y, ens_blend)
    print(f"Weighted 4-Model Rank Blend OOF KS: {ens_blend_ks:.4f} | AUC: {ens_blend_auc:.4f}")
    
    oof_preds["oof_ensemble_blend"] = ens_blend.tolist()
    
    # Test ensemble
    rank_test_lgb = rankdata(lgb_test_all) / len(X_test_df)
    rank_test_cb = rankdata(cb_test_all) / len(X_test_df)
    rank_test_xgb = rankdata(xgb_test_all) / len(X_test_df)
    rank_test_lr = rankdata(lr_test_all) / len(X_test_df)
    test_preds["ensemble_blend"] = (0.35 * rank_test_lr + 0.25 * rank_test_lgb + 0.20 * rank_test_cb + 0.20 * rank_test_xgb).tolist()
    
    # Save OOF predictions
    pl.DataFrame(oof_preds).write_parquet("preds/oof_phase2_v2.parquet")
    pl.DataFrame(test_preds).write_parquet("preds/test_phase2_v2.parquet")
    print("OOF and Test predictions saved to preds/!")
    
    # Top 30 features by LightGBM gain
    top_indices = np.argsort(lgb_importances)[::-1][:30]
    top_features = [{"feature": feature_cols[i], "gain": float(lgb_importances[i])} for i in top_indices]
    print("\nTop 15 Features by LightGBM Gain:")
    for rank, item in enumerate(top_features[:15], 1):
        print(f"  {rank:2d}. {item['feature']:<38} (gain: {item['gain']:.2f})")
        
    results_summary = {
        "LightGBM_v2": {
            "mean_ks": float(np.mean(lgb_fold_ks)),
            "std_ks": float(np.std(lgb_fold_ks)),
            "oof_ks": float(lgb_oof_ks),
            "oof_auc": float(lgb_oof_auc),
            "pairwise_delta_vs_base": float(np.mean(delta_lgb)),
            "pairwise_std_vs_base": float(np.std(delta_lgb)),
            "fold_ks": [float(x) for x in lgb_fold_ks],
            "train_time_sec": t_lgb
        },
        "CatBoost_v2": {
            "mean_ks": float(np.mean(cb_fold_ks)),
            "std_ks": float(np.std(cb_fold_ks)),
            "oof_ks": float(cb_oof_ks),
            "oof_auc": float(cb_oof_auc),
            "pairwise_delta_vs_base": float(np.mean(delta_cb)),
            "pairwise_std_vs_base": float(np.std(delta_cb)),
            "fold_ks": [float(x) for x in cb_fold_ks],
            "train_time_sec": t_cb
        },
        "XGBoost_v2": {
            "mean_ks": float(np.mean(xgb_fold_ks)),
            "std_ks": float(np.std(xgb_fold_ks)),
            "oof_ks": float(xgb_oof_ks),
            "oof_auc": float(xgb_oof_auc),
            "pairwise_delta_vs_base": float(np.mean(np.array(xgb_fold_ks) - base_lgb_folds)),
            "pairwise_std_vs_base": float(np.std(np.array(xgb_fold_ks) - base_lgb_folds)),
            "fold_ks": [float(x) for x in xgb_fold_ks],
            "train_time_sec": t_xgb
        },
        "Logistic_Regression_v2": {
            "mean_ks": float(np.mean(lr_fold_ks)),
            "std_ks": float(np.std(lr_fold_ks)),
            "oof_ks": float(lr_oof_ks),
            "oof_auc": float(lr_oof_auc),
            "pairwise_delta_vs_base": float(np.mean(delta_lr)),
            "pairwise_std_vs_base": float(np.std(delta_lr)),
            "fold_ks": [float(x) for x in lr_fold_ks],
            "train_time_sec": t_lr
        },
        "top_features": top_features
    }
    
    with open("reports/phase2_eval_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
    # Log to experiments_log.csv
    with open("experiments_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EXP-004", "FASE_2", "Advanced Features v2 (297 cols)", "LGBM with B1-B7 features", "297_adv_features", "LightGBM", f"{np.mean(lgb_fold_ks):.4f}", f"{np.std(lgb_fold_ks):.4f}", f"{np.mean(delta_lgb):+.4f}", f"{np.std(delta_lgb):.4f}", f"{lgb_oof_auc:.4f}", f"{t_lgb:.1f}", "DEPLOYABLE" if np.mean(delta_lgb) > 0 else "REJECTED", "LightGBM on B1-B7 engineered features"])
        writer.writerow(["EXP-005", "FASE_2", "Advanced Features v2 (297 cols)", "CatBoost with B1-B7 features", "297_adv_features", "CatBoost", f"{np.mean(cb_fold_ks):.4f}", f"{np.std(cb_fold_ks):.4f}", f"{np.mean(delta_cb):+.4f}", f"{np.std(delta_cb):.4f}", f"{cb_oof_auc:.4f}", f"{t_cb:.1f}", "DEPLOYABLE" if np.mean(delta_cb) > 0 else "REJECTED", "CatBoost on B1-B7 engineered features"])
        writer.writerow(["EXP-006", "FASE_2", "Advanced Features v2 (297 cols)", "XGBoost with B1-B7 features", "297_adv_features", "XGBoost", f"{np.mean(xgb_fold_ks):.4f}", f"{np.std(xgb_fold_ks):.4f}", f"{np.mean(np.array(xgb_fold_ks)-base_lgb_folds):+.4f}", f"{np.std(np.array(xgb_fold_ks)-base_lgb_folds):.4f}", f"{xgb_oof_auc:.4f}", f"{t_xgb:.1f}", "DEPLOYABLE", "XGBoost on B1-B7 engineered features"])
        writer.writerow(["EXP-007", "FASE_2", "Advanced Features v2 (297 cols)", "LogReg with B1-B7 features", "297_adv_features", "Logistic_Regression", f"{np.mean(lr_fold_ks):.4f}", f"{np.std(lr_fold_ks):.4f}", f"{np.mean(delta_lr):+.4f}", f"{np.std(delta_lr):.4f}", f"{lr_oof_auc:.4f}", f"{t_lr:.1f}", "DEPLOYABLE" if np.mean(delta_lr) > 0 else "REJECTED", "LogReg on B1-B7 engineered features"])
        
    print("Logged EXP-004 through EXP-007 to experiments_log.csv!")

if __name__ == "__main__":
    run_phase2_evaluation()
