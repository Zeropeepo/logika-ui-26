"""
Phase 1 - A7: Repeated Stratified 5-Fold x 3 Seed Validation Scheme
Generates fixed folds for all 70,000 train rows and evaluates baselines (LR, LightGBM, CatBoost).
"""
import os
import json
import time
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

DATA_DIR = "data/raw"
SEEDS = [42, 2026, 999]
N_SPLITS = 5
EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def generate_and_save_folds(df_train: pl.DataFrame):
    y = df_train["target"].to_numpy()
    app_ids = df_train["application_id"].to_list()
    
    folds_dict = {"application_id": app_ids}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_arr = np.zeros(len(y), dtype=int)
        for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold_arr[val_idx] = fold
        folds_dict[f"fold_seed{seed}"] = fold_arr.tolist()
        
    df_folds = pl.DataFrame(folds_dict)
    os.makedirs("data/processed", exist_ok=True)
    df_folds.write_parquet("data/processed/cv_folds_70k.parquet")
    print(f"Folds saved to data/processed/cv_folds_70k.parquet ({len(df_folds)} rows, seeds {SEEDS})")
    return df_folds

def run_a7_eval_scheme():
    print("=== A7: BUILDING REPEATED STRATIFIED 5-FOLD x 3 SEED CV ===")
    df_train_pl = pl.read_parquet("data/processed/train_features.parquet")
    y = df_train_pl["target"].to_numpy()
    
    # 1. Generate & save folds
    df_folds = generate_and_save_folds(df_train_pl)
    
    # Features
    feature_cols = [c for c in df_train_pl.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if df_train_pl[c].dtype == pl.String]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    X = df_train_pl.select(feature_cols).to_pandas()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        
    # CatBoost features (string cats)
    X_cb = X.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        
    os.makedirs("preds", exist_ok=True)
    
    all_results = {}
    oof_predictions = {}
    
    # Model 1: Logistic Regression
    print("\n--- Evaluating Baseline: Logistic Regression (Repeated 5-Fold x 3 Seed) ---")
    t0 = time.time()
    lr_oof_all_seeds = np.zeros(len(y))
    lr_seed_scores = []
    lr_all_fold_scores = []
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        seed_oof = np.zeros(len(y))
        seed_fold_scores = []
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(X.iloc[trn_idx][num_cols]))
            X_va = scl.transform(imp.transform(X.iloc[val_idx][num_cols]))
            
            lr = LogisticRegression(C=0.1, max_iter=500, random_state=seed)
            lr.fit(X_tr, y[trn_idx])
            val_p = lr.predict_proba(X_va)[:, 1]
            seed_oof[val_idx] = val_p
            ks_f = official_ks_metric(y[val_idx], val_p)
            seed_fold_scores.append(ks_f)
            lr_all_fold_scores.append(ks_f)
            
        seed_ks = official_ks_metric(y, seed_oof)
        lr_seed_scores.append(seed_ks)
        lr_oof_all_seeds += seed_oof / len(SEEDS)
        print(f"  Seed {seed}: OOF KS={seed_ks:.4f}, Folds Mean={np.mean(seed_fold_scores):.4f} +/- {np.std(seed_fold_scores):.4f}")
        
    t_lr = time.time() - t0
    oof_predictions["lr"] = lr_oof_all_seeds
    all_results["Logistic_Regression"] = {
        "seed_scores": lr_seed_scores,
        "fold_scores": lr_all_fold_scores,
        "mean_ks": float(np.mean(lr_all_fold_scores)),
        "std_ks": float(np.std(lr_all_fold_scores)),
        "oof_ks": float(official_ks_metric(y, lr_oof_all_seeds)),
        "oof_auc": float(roc_auc_score(y, lr_oof_all_seeds)),
        "time_sec": round(t_lr, 2)
    }
    
    # Model 2: LightGBM Conservative Baseline
    print("\n--- Evaluating Baseline: LightGBM (Repeated 5-Fold x 3 Seed) ---")
    t0 = time.time()
    lgb_oof_all_seeds = np.zeros(len(y))
    lgb_seed_scores = []
    lgb_all_fold_scores = []
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "n_jobs": 4,
    }
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        seed_oof = np.zeros(len(y))
        seed_fold_scores = []
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            trn_d = lgb.Dataset(X.iloc[trn_idx], label=y[trn_idx], categorical_feature=cat_cols)
            val_d = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], reference=trn_d, categorical_feature=cat_cols)
            
            bst = lgb.train(
                {**lgb_params, "random_state": seed},
                trn_d,
                num_boost_round=1000,
                valid_sets=[trn_d, val_d],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            val_p = bst.predict(X.iloc[val_idx])
            seed_oof[val_idx] = val_p
            ks_f = official_ks_metric(y[val_idx], val_p)
            seed_fold_scores.append(ks_f)
            lgb_all_fold_scores.append(ks_f)
            
        seed_ks = official_ks_metric(y, seed_oof)
        lgb_seed_scores.append(seed_ks)
        lgb_oof_all_seeds += seed_oof / len(SEEDS)
        print(f"  Seed {seed}: OOF KS={seed_ks:.4f}, Folds Mean={np.mean(seed_fold_scores):.4f} +/- {np.std(seed_fold_scores):.4f}")
        
    t_lgb = time.time() - t0
    oof_predictions["lgb"] = lgb_oof_all_seeds
    all_results["LightGBM"] = {
        "seed_scores": lgb_seed_scores,
        "fold_scores": lgb_all_fold_scores,
        "mean_ks": float(np.mean(lgb_all_fold_scores)),
        "std_ks": float(np.std(lgb_all_fold_scores)),
        "oof_ks": float(official_ks_metric(y, lgb_oof_all_seeds)),
        "oof_auc": float(roc_auc_score(y, lgb_oof_all_seeds)),
        "time_sec": round(t_lgb, 2)
    }
    
    # Model 3: CatBoost Baseline
    print("\n--- Evaluating Baseline: CatBoost (Repeated 5-Fold x 3 Seed) ---")
    t0 = time.time()
    cb_oof_all_seeds = np.zeros(len(y))
    cb_seed_scores = []
    cb_all_fold_scores = []
    
    cb_params = {
        "iterations": 800,
        "learning_rate": 0.05,
        "depth": 6,
        "eval_metric": "AUC",
        "verbose": False,
        "thread_count": 4,
    }
    
    for seed in SEEDS:
        fold_col = f"fold_seed{seed}"
        folds = df_folds[fold_col].to_numpy()
        seed_oof = np.zeros(len(y))
        seed_fold_scores = []
        for f in range(N_SPLITS):
            trn_idx = np.where(folds != f)[0]
            val_idx = np.where(folds == f)[0]
            
            trn_pool = Pool(X_cb.iloc[trn_idx], y[trn_idx], cat_features=cat_cols)
            val_pool = Pool(X_cb.iloc[val_idx], y[val_idx], cat_features=cat_cols)
            
            cb_model = CatBoostClassifier(**cb_params, random_seed=seed)
            cb_model.fit(trn_pool, eval_set=val_pool, early_stopping_rounds=50)
            val_p = cb_model.predict_proba(val_pool)[:, 1]
            seed_oof[val_idx] = val_p
            ks_f = official_ks_metric(y[val_idx], val_p)
            seed_fold_scores.append(ks_f)
            cb_all_fold_scores.append(ks_f)
            
        seed_ks = official_ks_metric(y, seed_oof)
        cb_seed_scores.append(seed_ks)
        cb_oof_all_seeds += seed_oof / len(SEEDS)
        print(f"  Seed {seed}: OOF KS={seed_ks:.4f}, Folds Mean={np.mean(seed_fold_scores):.4f} +/- {np.std(seed_fold_scores):.4f}")
        
    t_cb = time.time() - t0
    oof_predictions["cat"] = cb_oof_all_seeds
    all_results["CatBoost"] = {
        "seed_scores": cb_seed_scores,
        "fold_scores": cb_all_fold_scores,
        "mean_ks": float(np.mean(cb_all_fold_scores)),
        "std_ks": float(np.std(cb_all_fold_scores)),
        "oof_ks": float(official_ks_metric(y, cb_oof_all_seeds)),
        "oof_auc": float(roc_auc_score(y, cb_oof_all_seeds)),
        "time_sec": round(t_cb, 2)
    }
    
    # 2. Pairwise Differences (Selisih Berpasangan) per fold (15 folds)
    lgb_folds = np.array(lgb_all_fold_scores)
    lr_folds = np.array(lr_all_fold_scores)
    cb_folds = np.array(cb_all_fold_scores)
    
    delta_lgb_vs_lr = lgb_folds - lr_folds
    delta_cb_vs_lgb = cb_folds - lgb_folds
    
    pairwise_comparison = {
        "LGB_vs_LR": {
            "mean_delta": float(np.mean(delta_lgb_vs_lr)),
            "std_delta": float(np.std(delta_lgb_vs_lr)),
        },
        "CatBoost_vs_LGB": {
            "mean_delta": float(np.mean(delta_cb_vs_lgb)),
            "std_delta": float(np.std(delta_cb_vs_lgb)),
        }
    }
    
    # 3. Save OOF predictions to preds/
    pred_df = pl.DataFrame({
        "application_id": df_train_pl["application_id"],
        "target": y,
        "oof_lr": lr_oof_all_seeds,
        "oof_lgb": lgb_oof_all_seeds,
        "oof_cat": cb_oof_all_seeds,
    })
    pred_df.write_parquet("preds/oof_phase1_baselines.parquet")
    print(f"OOF predictions saved to preds/oof_phase1_baselines.parquet ({len(pred_df)} rows)")
    
    # 4. Save results summary
    out = {
        "models": all_results,
        "pairwise": pairwise_comparison,
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "total_rows": len(y)
    }
    with open("reports/phase1_a7_eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("A7 evaluation results saved to reports/phase1_a7_eval_results.json")
    
    # 5. Log to experiments_log.csv
    import csv
    with open("experiments_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EXP-001", "FASE_1", "Baseline 70k Repeated 5-Fold x 3 Seed", "Folds fixed across all models", "132_clean_features", "Logistic_Regression", f"{all_results['Logistic_Regression']['mean_ks']:.4f}", f"{all_results['Logistic_Regression']['std_ks']:.4f}", "0.0000", "0.0000", f"{all_results['Logistic_Regression']['oof_auc']:.4f}", f"{t_lr:.1f}", "BASELINE_ACUAN", "Linear baseline reference"])
        writer.writerow(["EXP-002", "FASE_1", "Baseline 70k Repeated 5-Fold x 3 Seed", "Folds fixed across all models", "132_clean_features", "LightGBM", f"{all_results['LightGBM']['mean_ks']:.4f}", f"{all_results['LightGBM']['std_ks']:.4f}", f"{np.mean(delta_lgb_vs_lr):+.4f}", f"{np.std(delta_lgb_vs_lr):.4f}", f"{all_results['LightGBM']['oof_auc']:.4f}", f"{t_lgb:.1f}", "BASELINE_ACUAN", "GBDT baseline reference"])
        writer.writerow(["EXP-003", "FASE_1", "Baseline 70k Repeated 5-Fold x 3 Seed", "Folds fixed across all models", "132_clean_features", "CatBoost", f"{all_results['CatBoost']['mean_ks']:.4f}", f"{all_results['CatBoost']['std_ks']:.4f}", f"{np.mean(delta_cb_vs_lgb):+.4f}", f"{np.std(delta_cb_vs_lgb):.4f}", f"{all_results['CatBoost']['oof_auc']:.4f}", f"{t_cb:.1f}", "BASELINE_ACUAN", "CatBoost baseline reference"])
    print("Experiments logged to experiments_log.csv")

if __name__ == "__main__":
    run_a7_eval_scheme()
