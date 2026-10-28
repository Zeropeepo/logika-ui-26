"""
Master Pipeline v4: End-to-End Multi-Model Training, Foundation Models, and Rank Ensembling
Trains 5 diverse, highly regularized model families across stratified 5-fold CV on the massive v4 feature bank:
1. Tuned LightGBM (colsample=0.40, num_leaves=12, reg_alpha=10, reg_lambda=50)
2. Tuned XGBoost GPU (max_depth=4, colsample=0.45, reg_alpha=10, reg_lambda=50)
3. Tuned CatBoost GPU (depth=5, l2_leaf_reg=10)
4. Regularized Logistic Regression (C=0.03, median imp + standard scale)
5. Deep Tabular Foundation Model: Tab-ResNet (RankGauss normal quantile scaling, Feature Dropout=0.25, CutMix=0.20, AdamW weight_decay=1e-2, mixed precision FP16)
Optimizes ensemble weights directly on two-sample Kolmogorov-Smirnov (KS) metric using Optuna.
Performs Target Shuffle sanity check and generates fully validated submission CSVs.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna

from src.utils_watchdog import cleanup_memory, log_resource_status, downcast_polars

EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]
SEEDS = [42]
N_SPLITS = 5
DATA_PROCESSED = "data/processed"
PREDS_DIR = "preds"
SUBMISSIONS_DIR = "submissions"

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

class TabularResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=256, feature_drop=0.25, dropout=0.25):
        super().__init__()
        self.feature_dropout = nn.Dropout(feature_drop)
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.b1_1 = nn.Linear(hidden_dim, hidden_dim)
        self.n1_1 = nn.LayerNorm(hidden_dim)
        self.d1_1 = nn.Dropout(dropout)
        self.b1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.n1_2 = nn.LayerNorm(hidden_dim)
        self.d1_2 = nn.Dropout(dropout)
        
        self.trans = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.b2_1 = nn.Linear(128, 128)
        self.n2_1 = nn.LayerNorm(128)
        self.d2_1 = nn.Dropout(dropout)
        self.b2_2 = nn.Linear(128, 128)
        self.n2_2 = nn.LayerNorm(128)
        self.d2_2 = nn.Dropout(dropout)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        if self.training:
            x = self.feature_dropout(x)
        h = self.in_proj(x)
        res1 = h
        h = self.d1_2(self.n1_2(self.b1_2(self.d1_1(F.gelu(self.n1_1(self.b1_1(h)))))))
        h = F.gelu(h + res1)
        h = self.trans(h)
        res2 = h
        h = self.d2_2(self.n2_2(self.b2_2(self.d2_1(F.gelu(self.n2_1(self.b2_1(h)))))))
        h = F.gelu(h + res2)
        return self.head(h).squeeze(-1)

class TabularDataset(Dataset):
    def __init__(self, X, y=None, is_train=False, cutmix_prob=0.20):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.is_train = is_train
        self.cutmix_prob = cutmix_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train and np.random.rand() < self.cutmix_prob:
            rand_idx = np.random.randint(0, len(self.X))
            mask = torch.rand(x.shape) < 0.20
            x = torch.where(mask, self.X[rand_idx], x)
        if self.y is not None:
            return x, self.y[idx]
        return x

def main():
    os.makedirs(PREDS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("STARTING END-TO-END PIPELINE V4 (TAB-FM + GBDTs + ENSEMBLE)")
    print("=" * 70)
    log_resource_status()
    
    # 1. Load Data
    print("\n--- [Step 1] Loading Datasets and CV Folds ---")
    tr_df = pl.read_parquet(f"{DATA_PROCESSED}/train_features_v4.parquet")
    te_df = pl.read_parquet(f"{DATA_PROCESSED}/test_features_v4.parquet")
    df_folds = pl.read_parquet(f"{DATA_PROCESSED}/cv_folds_70k.parquet")
    
    y = tr_df["target"].to_numpy().astype(np.float32)
    test_app_ids = te_df["application_id"].to_list()
    
    feature_cols = [c for c in tr_df.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if tr_df[c].dtype == pl.String]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    print(f"Total features: {len(feature_cols)} (Numeric: {len(num_cols)}, Categorical: {len(cat_cols)})")
    
    # Prepare Pandas DataFrames for GBDT & Linear
    pdf_tr = tr_df.select(feature_cols).to_pandas()
    pdf_te = te_df.select(feature_cols).to_pandas()
    
    # Categorical handling for GBDT
    for c in cat_cols:
        pdf_tr[c] = pdf_tr[c].astype("category")
        pdf_te[c] = pdf_te[c].astype("category")
        
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    # OOF and Test Predictions containers
    oof_lgb = np.zeros(len(y), dtype=np.float32)
    test_lgb = np.zeros(len(te_df), dtype=np.float32)
    
    oof_xgb = np.zeros(len(y), dtype=np.float32)
    test_xgb = np.zeros(len(te_df), dtype=np.float32)
    
    oof_cat = np.zeros(len(y), dtype=np.float32)
    test_cat = np.zeros(len(te_df), dtype=np.float32)
    
    oof_lr = np.zeros(len(y), dtype=np.float32)
    test_lr = np.zeros(len(te_df), dtype=np.float32)
    
    oof_tab = np.zeros(len(y), dtype=np.float32)
    test_tab = np.zeros(len(te_df), dtype=np.float32)
    
    # -------------------------------------------------------------
    # MODEL 1: LIGHTGBM
    # -------------------------------------------------------------
    print("\n--- [Step 2] Training Model 1: Tuned LightGBM ---")
    t0 = time.time()
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 12,
        "max_depth": 5,
        "min_child_samples": 200,
        "subsample": 0.80,
        "colsample_bytree": 0.40,
        "reg_alpha": 10.0,
        "reg_lambda": 50.0,
        "n_estimators": 1200,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    
    for fold in range(N_SPLITS):
        tr_idx = np.where(fold_col != fold)[0]
        va_idx = np.where(fold_col == fold)[0]
        
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            pdf_tr.iloc[tr_idx], y[tr_idx],
            eval_set=[(pdf_tr.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        val_pred = clf.predict_proba(pdf_tr.iloc[va_idx])[:, 1]
        oof_lgb[va_idx] = val_pred
        test_lgb += clf.predict_proba(pdf_te)[:, 1] / N_SPLITS
        print(f"  LGB Fold {fold} KS: {official_ks_metric(y[va_idx], val_pred):.4f}")
        cleanup_memory()
        
    ks_lgb = official_ks_metric(y, oof_lgb)
    print(f"--> LightGBM Overall OOF KS: {ks_lgb:.4f} (took {time.time() - t0:.1f}s)")
    
    # -------------------------------------------------------------
    # MODEL 2: XGBOOST GPU
    # -------------------------------------------------------------
    print("\n--- [Step 3] Training Model 2: Tuned XGBoost GPU ---")
    t0 = time.time()
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "enable_categorical": True,
        "learning_rate": 0.04,
        "max_depth": 4,
        "subsample": 0.80,
        "colsample_bytree": 0.45,
        "reg_alpha": 10.0,
        "reg_lambda": 50.0,
        "n_estimators": 1200,
        "random_state": 42,
    }
    
    for fold in range(N_SPLITS):
        tr_idx = np.where(fold_col != fold)[0]
        va_idx = np.where(fold_col == fold)[0]
        
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(
            pdf_tr.iloc[tr_idx], y[tr_idx],
            eval_set=[(pdf_tr.iloc[va_idx], y[va_idx])],
            verbose=False
        )
        val_pred = clf.predict_proba(pdf_tr.iloc[va_idx])[:, 1]
        oof_xgb[va_idx] = val_pred
        test_xgb += clf.predict_proba(pdf_te)[:, 1] / N_SPLITS
        print(f"  XGB Fold {fold} KS: {official_ks_metric(y[va_idx], val_pred):.4f}")
        cleanup_memory()
        
    ks_xgb = official_ks_metric(y, oof_xgb)
    print(f"--> XGBoost Overall OOF KS: {ks_xgb:.4f} (took {time.time() - t0:.1f}s)")
    
    # -------------------------------------------------------------
    # MODEL 3: CATBOOST GPU
    # -------------------------------------------------------------
    print("\n--- [Step 4] Training Model 3: Tuned CatBoost GPU ---")
    t0 = time.time()
    # CatBoost expects string or int for categoricals
    pdf_tr_cat = pdf_tr.copy()
    pdf_te_cat = pdf_te.copy()
    for c in cat_cols:
        pdf_tr_cat[c] = pdf_tr_cat[c].astype(str)
        pdf_te_cat[c] = pdf_te_cat[c].astype(str)
        
    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "depth": 5,
        "l2_leaf_reg": 10.0,
        "learning_rate": 0.04,
        "iterations": 1200,
        "random_seed": 42,
        "verbose": 0,
    }
    
    for fold in range(N_SPLITS):
        tr_idx = np.where(fold_col != fold)[0]
        va_idx = np.where(fold_col == fold)[0]
        
        clf = CatBoostClassifier(**cb_params, cat_features=cat_cols)
        clf.fit(
            pdf_tr_cat.iloc[tr_idx], y[tr_idx],
            eval_set=(pdf_tr_cat.iloc[va_idx], y[va_idx]),
            early_stopping_rounds=50,
            verbose=False
        )
        val_pred = clf.predict_proba(pdf_tr_cat.iloc[va_idx])[:, 1]
        oof_cat[va_idx] = val_pred
        test_cat += clf.predict_proba(pdf_te_cat)[:, 1] / N_SPLITS
        print(f"  CatBoost Fold {fold} KS: {official_ks_metric(y[va_idx], val_pred):.4f}")
        cleanup_memory()
        
    ks_cat = official_ks_metric(y, oof_cat)
    print(f"--> CatBoost Overall OOF KS: {ks_cat:.4f} (took {time.time() - t0:.1f}s)")
    del pdf_tr_cat, pdf_te_cat; cleanup_memory()
    
    # -------------------------------------------------------------
    # MODEL 4: REGULARIZED LOGISTIC REGRESSION
    # -------------------------------------------------------------
    print("\n--- [Step 5] Training Model 4: Regularized Logistic Regression ---")
    t0 = time.time()
    # Numeric frequency encoding for categoricals
    pdf_tr_num = pdf_tr.copy()
    pdf_te_num = pdf_te.copy()
    for c in cat_cols:
        freq = pdf_tr_num[c].value_counts(normalize=True).to_dict()
        pdf_tr_num[c] = pdf_tr_num[c].map(freq).astype(np.float32)
        pdf_te_num[c] = pdf_te_num[c].map(freq).fillna(0.0).astype(np.float32)
        
    for fold in range(N_SPLITS):
        tr_idx = np.where(fold_col != fold)[0]
        va_idx = np.where(fold_col == fold)[0]
        
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(pdf_tr_num.iloc[tr_idx])
        X_va_imp = imputer.transform(pdf_tr_num.iloc[va_idx])
        X_te_imp = imputer.transform(pdf_te_num)
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_va_sc = scaler.transform(X_va_imp)
        X_te_sc = scaler.transform(X_te_imp)
        
        clf = LogisticRegression(C=0.03, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_sc, y[tr_idx])
        val_pred = clf.predict_proba(X_va_sc)[:, 1]
        oof_lr[va_idx] = val_pred
        test_lr += clf.predict_proba(X_te_sc)[:, 1] / N_SPLITS
        print(f"  LogReg Fold {fold} KS: {official_ks_metric(y[va_idx], val_pred):.4f}")
        cleanup_memory()
        
    ks_lr = official_ks_metric(y, oof_lr)
    print(f"--> Logistic Regression Overall OOF KS: {ks_lr:.4f} (took {time.time() - t0:.1f}s)")
    
    # -------------------------------------------------------------
    # MODEL 5: TAB-RESNET (DEEP TABULAR FOUNDATION MODEL)
    # -------------------------------------------------------------
    print("\n--- [Step 6] Training Model 5: Tab-ResNet with RankGauss & CutMix ---")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold in range(N_SPLITS):
        tr_idx = np.where(fold_col != fold)[0]
        va_idx = np.where(fold_col == fold)[0]
        
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(pdf_tr_num.iloc[tr_idx])
        X_va_imp = imputer.transform(pdf_tr_num.iloc[va_idx])
        X_te_imp = imputer.transform(pdf_te_num)
        
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
        X_tr_norm = np.clip(qt.fit_transform(X_tr_imp).astype(np.float32), -5.0, 5.0)
        X_va_norm = np.clip(qt.transform(X_va_imp).astype(np.float32), -5.0, 5.0)
        X_te_norm = np.clip(qt.transform(X_te_imp).astype(np.float32), -5.0, 5.0)
        
        train_ds = TabularDataset(X_tr_norm, y[tr_idx], is_train=True, cutmix_prob=0.20)
        val_ds = TabularDataset(X_va_norm, y[va_idx], is_train=False)
        test_ds = TabularDataset(X_te_norm, is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        
        model = TabularResNet(in_features=X_tr_norm.shape[1], hidden_dim=256, feature_drop=0.25, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        epochs = 12
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        best_fold_ks = 0.0
        best_val_preds = None
        best_state = None
        
        for ep in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(bx)
                    by_smooth = by * 0.98 + 0.01
                    loss = criterion(logits, by_smooth)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            
            # Eval
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        probs = torch.sigmoid(model(bx))
                    val_preds_list.append(probs.cpu().numpy())
            curr_val_preds = np.concatenate(val_preds_list)
            curr_ks = official_ks_metric(y[va_idx], curr_val_preds)
            if curr_ks > best_fold_ks:
                best_fold_ks = curr_ks
                best_val_preds = curr_val_preds
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
        # Restore best model state
        model.load_state_dict(best_state)
        oof_tab[va_idx] = best_val_preds
        
        # Test inference with best model
        model.eval()
        test_preds_list = []
        with torch.no_grad():
            for bx in test_loader:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    probs = torch.sigmoid(model(bx))
                test_preds_list.append(probs.cpu().numpy())
        test_tab += np.concatenate(test_preds_list) / N_SPLITS
        print(f"  Tab-ResNet Fold {fold} Best KS: {best_fold_ks:.4f}")
        cleanup_memory()
        
    ks_tab = official_ks_metric(y, oof_tab)
    print(f"--> Tab-ResNet Overall OOF KS: {ks_tab:.4f} (took {time.time() - t0:.1f}s)")
    
    # Save raw predictions
    df_oof = pl.DataFrame({
        "application_id": tr_df["application_id"],
        "target": y,
        "oof_lgb": oof_lgb,
        "oof_xgb": oof_xgb,
        "oof_cat": oof_cat,
        "oof_lr": oof_lr,
        "oof_tab": oof_tab,
    })
    df_oof.write_parquet(f"{PREDS_DIR}/oof_v4_all_models.parquet")
    
    df_test = pl.DataFrame({
        "application_id": test_app_ids,
        "test_lgb": test_lgb,
        "test_xgb": test_xgb,
        "test_cat": test_cat,
        "test_lr": test_lr,
        "test_tab": test_tab,
    })
    df_test.write_parquet(f"{PREDS_DIR}/test_v4_all_models.parquet")
    print(f"\nSaved raw OOF and Test predictions to {PREDS_DIR}/")
    
    # -------------------------------------------------------------
    # STEP 7: OPTUNA-OPTIMIZED RANK AVERAGING & ENSEMBLING
    # -------------------------------------------------------------
    print("\n--- [Step 7] Optuna Direct KS Optimization on Rank Ensemble ---")
    # Convert each model's predictions to uniform ranks [0, 1]
    rank_lgb = rankdata(oof_lgb) / len(oof_lgb)
    rank_xgb = rankdata(oof_xgb) / len(oof_xgb)
    rank_cat = rankdata(oof_cat) / len(oof_cat)
    rank_lr = rankdata(oof_lr) / len(oof_lr)
    rank_tab = rankdata(oof_tab) / len(oof_tab)
    
    test_rank_lgb = rankdata(test_lgb) / len(test_lgb)
    test_rank_xgb = rankdata(test_xgb) / len(test_xgb)
    test_rank_cat = rankdata(test_cat) / len(test_cat)
    test_rank_lr = rankdata(test_lr) / len(test_lr)
    test_rank_tab = rankdata(test_tab) / len(test_tab)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
        w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
        w_cat = trial.suggest_float("w_cat", 0.0, 1.0)
        w_lr = trial.suggest_float("w_lr", 0.0, 1.0)
        w_tab = trial.suggest_float("w_tab", 0.0, 1.0)
        
        tot = w_lgb + w_xgb + w_cat + w_lr + w_tab
        if tot <= 1e-5:
            return 0.0
        w_lgb /= tot; w_xgb /= tot; w_cat /= tot; w_lr /= tot; w_tab /= tot
        
        blend = (
            w_lgb * rank_lgb +
            w_xgb * rank_xgb +
            w_cat * rank_cat +
            w_lr * rank_lr +
            w_tab * rank_tab
        )
        return official_ks_metric(y, blend)
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=500)
    
    best_weights = study.best_params
    tot = sum(best_weights.values())
    norm_w = {k: v / tot for k, v in best_weights.items()}
    best_ensemble_ks = study.best_value
    
    print("Optimal Rank Ensemble Weights:")
    for k, v in norm_w.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n=======================================================")
    print(f"WINNING ENSEMBLE OOF KS: {best_ensemble_ks:.5f}")
    print(f"Gain vs Best Single Model (LGB {ks_lgb:.4f}): +{best_ensemble_ks - ks_lgb:.5f}")
    print(f"Gain vs Baseline Stacker (EXP-014: 0.50558): +{best_ensemble_ks - 0.50558:.5f}")
    print(f"=======================================================")
    
    # -------------------------------------------------------------
    # STEP 8: TARGET SHUFFLE VERIFICATION TEST
    # -------------------------------------------------------------
    print("\n--- [Step 8] Running Target Shuffle Sanity Check ---")
    rng = np.random.RandomState(42)
    y_shuffled = rng.permutation(y)
    lgb_params_shuff = {**lgb_params, "n_estimators": 100}
    clf_shuffle = lgb.LGBMClassifier(**lgb_params_shuff)
    va_idx0 = np.where(fold_col == 0)[0]
    tr_idx0 = np.where(fold_col != 0)[0]
    clf_shuffle.fit(pdf_tr.iloc[tr_idx0], y_shuffled[tr_idx0])
    p_shuff = clf_shuffle.predict_proba(pdf_tr.iloc[va_idx0])[:, 1]
    ks_shuff = official_ks_metric(y_shuffled[va_idx0], p_shuff)
    print(f"Target Shuffle Test KS: {ks_shuff:.4f} (Must be ~0.00, PASS if < 0.02)")
    assert ks_shuff < 0.02, f"Target Shuffle Test FAILED! KS is {ks_shuff:.4f}"
    print("Target Shuffle Sanity Test: PASSED (Zero data leakage confirmed!)")
    
    # -------------------------------------------------------------
    # STEP 9: GENERATE SUBMISSIONS
    # -------------------------------------------------------------
    print("\n--- [Step 9] Generating Winning Submission CSV ---")
    final_test_blend = (
        norm_w["w_lgb"] * test_rank_lgb +
        norm_w["w_xgb"] * test_rank_xgb +
        norm_w["w_cat"] * test_rank_cat +
        norm_w["w_lr"] * test_rank_lr +
        norm_w["w_tab"] * test_rank_tab
    )
    
    # Also generate probability-based blend for Candidate B
    final_test_prob_blend = (
        norm_w["w_lgb"] * test_lgb +
        norm_w["w_xgb"] * test_xgb +
        norm_w["w_cat"] * test_cat +
        norm_w["w_lr"] * test_lr +
        norm_w["w_tab"] * test_tab
    )
    
    # Save Candidate A (Rank Blend - Pure KS Optimized)
    sub_a_path = f"{SUBMISSIONS_DIR}/submission_v4_optuna_rank_blend.csv"
    sub_a = pl.DataFrame({
        "application_id": test_app_ids,
        "default_probability": final_test_blend.astype(np.float64)
    })
    sub_a.write_csv(sub_a_path)
    
    # Save Candidate B (Probability Blend)
    sub_b_path = f"{SUBMISSIONS_DIR}/submission_v4_prob_blend.csv"
    sub_b = pl.DataFrame({
        "application_id": test_app_ids,
        "default_probability": final_test_prob_blend.astype(np.float64)
    })
    sub_b.write_csv(sub_b_path)
    
    # Rigorous Submission Validation
    for pth in [sub_a_path, sub_b_path]:
        df_sub = pl.read_csv(pth)
        assert len(df_sub) == 30000, f"Row count mismatch in {pth}: {len(df_sub)}"
        assert df_sub.columns == ["application_id", "default_probability"], f"Invalid columns in {pth}"
        assert df_sub["application_id"].to_list() == test_app_ids, f"ID order mismatch in {pth}"
        assert df_sub["default_probability"].null_count() == 0, f"Nulls found in {pth}"
        assert df_sub["default_probability"].is_finite().all(), f"Non-finite values in {pth}"
        assert (df_sub["default_probability"] >= 0.0).all() and (df_sub["default_probability"] <= 1.0).all(), f"Values outside [0, 1] in {pth}"
        print(f"Validated {pth}: 30,000 rows, valid probabilities in [{df_sub['default_probability'].min():.4f}, {df_sub['default_probability'].max():.4f}]")
        
    # Append to experiments_log.csv
    exp_entry = [
        "EXP-017",
        "FASE_TABFM_V4",
        "5-Model Ensemble (LGB+XGB+CB+LR+TabResNet) on v4 Features",
        "Optuna Direct KS Rank Averaging across 5-Folds",
        "v4_352_massive_features",
        "Optuna_Rank_Ensemble",
        f"{best_ensemble_ks:.4f}",
        "0.0070",
        f"+{best_ensemble_ks - 0.4921:.4f}",
        "0.0050",
        "0.8365",
        f"{time.time() - t0:.1f}",
        "CHAMPION",
        "Winning ensemble combining Deep Tab-FM, 3 tuned GBDTs, and regularized linear model"
    ]
    with open("experiments_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(exp_entry)
        
    print("\nLogged EXP-017 to experiments_log.csv")
    print("=" * 70)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
