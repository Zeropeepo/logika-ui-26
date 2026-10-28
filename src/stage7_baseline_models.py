"""
Stage 7: Baseline Modeling, Cross Validation, Sanity Checks, Ablation, and First Submission
"""
import os
import json
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

RANDOM_SEED = 42
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

def run_stage7():
    print("=== STAGE 7: BASELINE MODELING ===")
    
    # 1. Load Parquet datasets
    df_train_pl = pl.read_parquet("data/processed/train_features.parquet")
    df_test_pl = pl.read_parquet("data/processed/test_features.parquet")
    
    train_apps = df_train_pl["application_id"].to_list()
    test_apps = df_test_pl["application_id"].to_list()
    y_full = df_train_pl["target"].to_numpy()
    
    # Assert column consistency
    feature_cols = [c for c in df_train_pl.columns if c not in EVAL_ONLY_COLUMNS]
    test_feature_cols = [c for c in df_test_pl.columns if c not in EVAL_ONLY_COLUMNS]
    assert feature_cols == test_feature_cols, "Feature columns mismatch between train and test!"
    print(f"Number of clean input features: {len(feature_cols)}")
    
    # Convert to pandas with category types for LightGBM/CatBoost
    cat_cols = [c for c in feature_cols if df_train_pl[c].dtype == pl.String]
    print(f"Categorical features: {cat_cols}")
    
    X_full = df_train_pl.select(feature_cols).to_pandas()
    X_test = df_test_pl.select(feature_cols).to_pandas()
    
    for c in cat_cols:
        X_full[c] = X_full[c].astype("category")
        X_test[c] = X_test[c].astype("category")
        
    # 2. Strict Train-Holdout Split (80% CV, 20% Final Hold-out)
    train_idx, holdout_idx = train_test_split(
        np.arange(len(y_full)),
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_full
    )
    
    X_cv = X_full.iloc[train_idx].reset_index(drop=True)
    y_cv = y_full[train_idx]
    
    X_holdout = X_full.iloc[holdout_idx].reset_index(drop=True)
    y_holdout = y_full[holdout_idx]
    
    print(f"CV Train shape: {X_cv.shape}, Holdout shape: {X_holdout.shape}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    results_summary = []
    
    # --- MODEL B0: Trivial Random Baseline ---
    print("\n--- Running B0: Random Baseline ---")
    rng = np.random.RandomState(RANDOM_SEED)
    oof_b0_rnd = rng.rand(len(y_cv))
    ks_b0_rnd = official_ks_metric(y_cv, oof_b0_rnd)
    auc_b0_rnd = roc_auc_score(y_cv, oof_b0_rnd)
    ho_b0_rnd = rng.rand(len(y_holdout))
    ks_ho_b0_rnd = official_ks_metric(y_holdout, ho_b0_rnd)
    ci_b0 = bootstrap_ks_ci(y_cv, oof_b0_rnd)
    results_summary.append({
        "model": "B0_Random",
        "ks_mean": round(ks_b0_rnd, 4),
        "ks_std": 0.0,
        "ci_95": [round(ci_b0[0], 4), round(ci_b0[1], 4)],
        "ks_holdout": round(ks_ho_b0_rnd, 4),
        "auc_oof": round(auc_b0_rnd, 4),
        "note": "Sanity check random scores"
    })
    print(f"B0 Random: KS OOF={ks_b0_rnd:.4f}, AUC={auc_b0_rnd:.4f}")
    
    # --- MODEL B0: Single Feature Baseline (annual_interest_rate) ---
    print("\n--- Running B0: Single Feature Baseline (annual_interest_rate) ---")
    s_feat = X_cv["annual_interest_rate"].values
    ks_b0_feat = official_ks_metric(y_cv, s_feat)
    auc_b0_feat = roc_auc_score(y_cv, s_feat)
    ho_feat = X_holdout["annual_interest_rate"].values
    ks_ho_b0_feat = official_ks_metric(y_holdout, ho_feat)
    ci_b0_feat = bootstrap_ks_ci(y_cv, s_feat)
    results_summary.append({
        "model": "B0_Single_Feature (annual_interest_rate)",
        "ks_mean": round(ks_b0_feat, 4),
        "ks_std": 0.0,
        "ci_95": [round(ci_b0_feat[0], 4), round(ci_b0_feat[1], 4)],
        "ks_holdout": round(ks_ho_b0_feat, 4),
        "auc_oof": round(auc_b0_feat, 4),
        "note": "Single feature benchmark"
    })
    print(f"B0 Single Feature: KS OOF={ks_b0_feat:.4f}, AUC={auc_b0_feat:.4f}")
    
    # --- MODEL B1: Logistic Regression ---
    print("\n--- Running B1: Logistic Regression ---")
    # Impute numeric features inside fold, encode categories
    num_cols = [c for c in feature_cols if c not in cat_cols]
    oof_b1 = np.zeros(len(y_cv))
    ho_b1_folds = np.zeros(len(y_holdout))
    fold_ks_b1 = []
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        X_tr = X_cv.iloc[trn_idx][num_cols]
        X_va = X_cv.iloc[val_idx][num_cols]
        X_ho = X_holdout[num_cols]
        
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        X_tr_proc = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_va_proc = scaler.transform(imputer.transform(X_va))
        X_ho_proc = scaler.transform(imputer.transform(X_ho))
        
        lr = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED)
        lr.fit(X_tr_proc, y_cv[trn_idx])
        
        val_preds = lr.predict_proba(X_va_proc)[:, 1]
        oof_b1[val_idx] = val_preds
        ho_b1_folds += lr.predict_proba(X_ho_proc)[:, 1] / 5.0
        fold_ks_b1.append(official_ks_metric(y_cv[val_idx], val_preds))
        
    ks_b1 = official_ks_metric(y_cv, oof_b1)
    auc_b1 = roc_auc_score(y_cv, oof_b1)
    ks_ho_b1 = official_ks_metric(y_holdout, ho_b1_folds)
    ci_b1 = bootstrap_ks_ci(y_cv, oof_b1)
    results_summary.append({
        "model": "B1_Logistic_Regression",
        "ks_mean": round(float(np.mean(fold_ks_b1)), 4),
        "ks_std": round(float(np.std(fold_ks_b1)), 4),
        "ci_95": [round(ci_b1[0], 4), round(ci_b1[1], 4)],
        "ks_holdout": round(ks_ho_b1, 4),
        "auc_oof": round(auc_b1, 4),
        "note": "Linear model baseline with median imputation & scaling"
    })
    print(f"B1 Logistic Regression: KS OOF={ks_b1:.4f} (folds: {np.mean(fold_ks_b1):.4f} +/- {np.std(fold_ks_b1):.4f}), AUC={auc_b1:.4f}, Holdout KS={ks_ho_b1:.4f}")
    
    # --- MODEL B2: LightGBM (Conservative baseline) ---
    print("\n--- Running B2: LightGBM (Conservative Baseline) ---")
    oof_b2 = np.zeros(len(y_cv))
    ho_b2_folds = np.zeros(len(y_holdout))
    test_b2_folds = np.zeros(len(X_test))
    fold_ks_b2 = []
    lgb_models = []
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbosity": -1,
        "n_jobs": 4,
    }
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        X_tr, y_tr = X_cv.iloc[trn_idx], y_cv[trn_idx]
        X_va, y_va = X_cv.iloc[val_idx], y_cv[val_idx]
        
        trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, categorical_feature=cat_cols)
        
        bst = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1000,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgb_models.append(bst)
        val_preds = bst.predict(X_va)
        oof_b2[val_idx] = val_preds
        ho_b2_folds += bst.predict(X_holdout) / 5.0
        test_b2_folds += bst.predict(X_test) / 5.0
        fold_ks_b2.append(official_ks_metric(y_va, val_preds))
        print(f"Fold {fold+1} LightGBM: Best iteration = {bst.best_iteration}, KS = {fold_ks_b2[-1]:.4f}")
        
    ks_b2 = official_ks_metric(y_cv, oof_b2)
    auc_b2 = roc_auc_score(y_cv, oof_b2)
    ks_ho_b2 = official_ks_metric(y_holdout, ho_b2_folds)
    ci_b2 = bootstrap_ks_ci(y_cv, oof_b2)
    results_summary.append({
        "model": "B2_LightGBM",
        "ks_mean": round(float(np.mean(fold_ks_b2)), 4),
        "ks_std": round(float(np.std(fold_ks_b2)), 4),
        "ci_95": [round(ci_b2[0], 4), round(ci_b2[1], 4)],
        "ks_holdout": round(ks_ho_b2, 4),
        "auc_oof": round(auc_b2, 4),
        "note": "Conservative GBDT (lr=0.03, leaves=31, colsample=0.8)"
    })
    print(f"B2 LightGBM: KS OOF={ks_b2:.4f} (folds: {np.mean(fold_ks_b2):.4f} +/- {np.std(fold_ks_b2):.4f}), AUC={auc_b2:.4f}, Holdout KS={ks_ho_b2:.4f}")
    
    # --- MODEL B3: CatBoost (Default parameters) ---
    print("\n--- Running B3: CatBoost ---")
    oof_b3 = np.zeros(len(y_cv))
    ho_b3_folds = np.zeros(len(y_holdout))
    fold_ks_b3 = []
    
    # Convert cat_cols to string for CatBoost
    X_cv_cb = X_cv.copy()
    X_ho_cb = X_holdout.copy()
    for c in cat_cols:
        X_cv_cb[c] = X_cv_cb[c].astype(str)
        X_ho_cb[c] = X_ho_cb[c].astype(str)
        
    cb_params = {
        "iterations": 800,
        "learning_rate": 0.05,
        "depth": 6,
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "thread_count": 4,
    }
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_cv_cb, y_cv)):
        X_tr, y_tr = X_cv_cb.iloc[trn_idx], y_cv[trn_idx]
        X_va, y_va = X_cv_cb.iloc[val_idx], y_cv[val_idx]
        
        trn_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)
        ho_pool = Pool(X_ho_cb, cat_features=cat_cols)
        
        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(trn_pool, eval_set=val_pool, early_stopping_rounds=50)
        
        val_preds = cb_model.predict_proba(val_pool)[:, 1]
        oof_b3[val_idx] = val_preds
        ho_b3_folds += cb_model.predict_proba(ho_pool)[:, 1] / 5.0
        fold_ks_b3.append(official_ks_metric(y_va, val_preds))
        print(f"Fold {fold+1} CatBoost: Best iteration = {cb_model.get_best_iteration()}, KS = {fold_ks_b3[-1]:.4f}")
        
    ks_b3 = official_ks_metric(y_cv, oof_b3)
    auc_b3 = roc_auc_score(y_cv, oof_b3)
    ks_ho_b3 = official_ks_metric(y_holdout, ho_b3_folds)
    ci_b3 = bootstrap_ks_ci(y_cv, oof_b3)
    results_summary.append({
        "model": "B3_CatBoost",
        "ks_mean": round(float(np.mean(fold_ks_b3)), 4),
        "ks_std": round(float(np.std(fold_ks_b3)), 4),
        "ci_95": [round(ci_b3[0], 4), round(ci_b3[1], 4)],
        "ks_holdout": round(ks_ho_b3, 4),
        "auc_oof": round(auc_b3, 4),
        "note": "CatBoost with default depth=6, lr=0.05"
    })
    print(f"B3 CatBoost: KS OOF={ks_b3:.4f} (folds: {np.mean(fold_ks_b3):.4f} +/- {np.std(fold_ks_b3):.4f}), AUC={auc_b3:.4f}, Holdout KS={ks_ho_b3:.4f}")
    
    # --- SANITY CHECK 1: Target Shuffle Test ---
    print("\n--- Running Sanity Check 1: Target Shuffle Test ---")
    rng = np.random.RandomState(RANDOM_SEED)
    y_shuffled = rng.permutation(y_cv)
    oof_shuffled = np.zeros(len(y_cv))
    
    # Quick 3-fold shuffle check with LightGBM
    skf_shuf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    for fold, (trn_idx, val_idx) in enumerate(skf_shuf.split(X_cv, y_shuffled)):
        X_tr, y_tr = X_cv.iloc[trn_idx], y_shuffled[trn_idx]
        X_va, y_va = X_cv.iloc[val_idx], y_shuffled[val_idx]
        trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, categorical_feature=cat_cols)
        bst = lgb.train(
            {**lgb_params, "learning_rate": 0.05},
            trn_data,
            num_boost_round=100,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        oof_shuffled[val_idx] = bst.predict(X_va)
    ks_shuffled = official_ks_metric(y_shuffled, oof_shuffled)
    print(f"Target Shuffle Test KS: {ks_shuffled:.4f} (Expected ~0.00)")
    assert ks_shuffled < 0.03, f"Shuffle test failed: KS {ks_shuffled} too high!"
    
    # --- SANITY CHECK 2: Source Ablation Test ---
    print("\n--- Running Sanity Check 2: Source Ablation Test ---")
    sources = {
        "without_previous_applications": [c for c in feature_cols if not c.startswith("prev_")],
        "without_repayment_history": [c for c in feature_cols if not c.startswith("repay_")],
        "without_account_behavior": [c for c in feature_cols if not c.startswith("acct_")],
        "without_bureau_observations": [c for c in feature_cols if not c.startswith("bureau_")],
        "without_ecommerce_transactions": [c for c in feature_cols if not c.startswith("ecom_")],
        "only_application_and_profile": [c for c in feature_cols if not any(c.startswith(p) for p in ["prev_", "repay_", "acct_", "bureau_", "ecom_"])],
    }
    
    ablation_results = {}
    for ab_name, ab_cols in sources.items():
        ab_cats = [c for c in ab_cols if c in cat_cols]
        # Fast 3-fold evaluation
        skf_ab = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        ab_oof = np.zeros(len(y_cv))
        for fold, (trn_idx, val_idx) in enumerate(skf_ab.split(X_cv, y_cv)):
            X_tr, y_tr = X_cv.iloc[trn_idx][ab_cols], y_cv[trn_idx]
            X_va, y_va = X_cv.iloc[val_idx][ab_cols], y_cv[val_idx]
            trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=ab_cats)
            val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, categorical_feature=ab_cats)
            bst = lgb.train(
                {**lgb_params, "learning_rate": 0.05},
                trn_data,
                num_boost_round=300,
                valid_sets=[trn_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            ab_oof[val_idx] = bst.predict(X_va)
        ab_ks = official_ks_metric(y_cv, ab_oof)
        ab_auc = roc_auc_score(y_cv, ab_oof)
        ablation_results[ab_name] = {
            "n_features": len(ab_cols),
            "ks": round(float(ab_ks), 4),
            "auc": round(float(ab_auc), 4),
            "ks_delta_vs_full": round(float(ab_ks - ks_b2), 4)
        }
        print(f"Ablation [{ab_name}] ({len(ab_cols)} feats): KS={ab_ks:.4f}, AUC={ab_auc:.4f}, Delta={ab_ks - ks_b2:.4f}")
        
    # --- FEATURE IMPORTANCE & SHAP FOR BEST MODEL (LightGBM) ---
    print("\n--- Computing Feature Importance ---")
    feat_importances = np.zeros(len(feature_cols))
    for bst in lgb_models:
        feat_importances += bst.feature_importance(importance_type="gain") / len(lgb_models)
        
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": feat_importances
    }).sort_values("importance_gain", ascending=False)
    
    top_15_features = fi_df.head(15).to_dict(orient="records")
    print("Top 10 features by LightGBM Gain Importance:")
    for r in top_15_features[:10]:
        print(f"  {r['feature']:35s}: {r['importance_gain']:.2f}")
        
    # --- PERFORMANCE PER SEGMENT ---
    print("\n--- Evaluating Segment Performance on OOF Predictions ---")
    segment_eval = {}
    for seg_col in ["loan_purpose", "region", "employment_type"]:
        seg_vals = X_cv[seg_col].unique()
        seg_ks = {}
        for sv in seg_vals:
            mask = X_cv[seg_col] == sv
            if sum(y_cv[mask] == 1) > 10 and sum(y_cv[mask] == 0) > 10:
                s_ks = official_ks_metric(y_cv[mask], oof_b2[mask])
                seg_ks[str(sv)] = round(float(s_ks), 4)
        segment_eval[seg_col] = seg_ks
    print("Segment KS Stability:", json.dumps(segment_eval, indent=2))
    
    # --- TEST SUBMISSION PREPARATION & VALIDATION ---
    print("\n--- Preparing First Submission ---")
    # Best model is B2 LightGBM (5-fold ensemble predictions)
    sub_df = pd.DataFrame({
        "application_id": test_apps,
        "default_probability": test_b2_folds
    })
    
    # Strictly validate submission
    assert len(sub_df) == 30000, f"Submission row count mismatch: {len(sub_df)} vs 30000"
    assert sub_df["application_id"].nunique() == 30000, "Duplicate application_id in submission!"
    assert set(sub_df["application_id"]) == set(test_apps), "Mismatch with test application IDs!"
    assert not sub_df["default_probability"].isna().any(), "NaN found in predictions!"
    assert np.all(np.isfinite(sub_df["default_probability"])), "Non-finite values found in predictions!"
    assert (sub_df["default_probability"] >= 0.0).all() and (sub_df["default_probability"] <= 1.0).all(), "Predictions out of [0, 1] range!"
    
    os.makedirs("submissions", exist_ok=True)
    sub_path = "submissions/submission_baseline.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission successfully validated and written to {sub_path} ({len(sub_df)} rows)")
    
    # Prediction distribution comparison: Train OOF vs Test Preds
    pred_dist_check = {
        "train_oof_mean": float(oof_b2.mean()),
        "test_pred_mean": float(test_b2_folds.mean()),
        "train_oof_std": float(oof_b2.std()),
        "test_pred_std": float(test_b2_folds.std()),
        "train_oof_median": float(np.median(oof_b2)),
        "test_pred_median": float(np.median(test_b2_folds)),
        "train_oof_min": float(oof_b2.min()),
        "test_pred_min": float(test_b2_folds.min()),
        "train_oof_max": float(oof_b2.max()),
        "test_pred_max": float(test_b2_folds.max()),
    }
    print("Prediction distribution check:", json.dumps(pred_dist_check, indent=2))
    
    # Save full stage 7 results
    stage7_output = {
        "models_summary": results_summary,
        "sanity_checks": {
            "shuffle_test_ks": round(ks_shuffled, 4),
            "source_ablation": ablation_results,
        },
        "top_features": top_15_features,
        "segment_performance": segment_eval,
        "prediction_distribution_check": pred_dist_check,
    }
    
    with open("reports/stage7_baseline_report.json", "w") as f:
        json.dump(stage7_output, f, indent=2)
    print("Stage 7 report saved to reports/stage7_baseline_report.json")

if __name__ == "__main__":
    run_stage7()
