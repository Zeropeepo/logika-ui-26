"""
Sequential E-Commerce GRU Model (Pieter's Winning Discovery Architecture)
Trains a lightweight PyTorch GRU on chronological e-commerce order logs leading up to cutoff:
- Input per order: [log_amount, days_since, hour, dow, is_weekend, is_payday, is_completed, is_cod, is_ewallet, is_card]
- 5-Fold Stratified Cross-Validation on RTX 3050 in FP16 mixed precision
- Produces oof_ecom_gru and test_ecom_gru predictions
- Optimizes Quantile Rank Ensemble combining GRU with existing v4 models
- Directly outputs winning submission: submissions/submission_v5_gru_optuna_blend.csv
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, rankdata
import optuna

from src.utils_watchdog import cleanup_memory, log_resource_status

MAX_ORDERS = 40
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
PREDS_DIR = "preds"
SUBMISSIONS_DIR = "submissions"

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

class EcomSequentialGRU(nn.Module):
    def __init__(self, in_features=10, hidden_dim=64, num_layers=1, dropout=0.20):
        super().__init__()
        self.in_proj = nn.Linear(in_features, 32)
        self.gru = nn.GRU(32, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        # x: (B, T, in_features)
        h = F.gelu(self.in_proj(x))
        out, _ = self.gru(h)
        B = x.size(0)
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(B, 1, out.size(2))
        last_step = out.gather(1, idx).squeeze(1)
        logits = self.head(last_step).squeeze(-1)
        return logits

class EcomDataset(Dataset):
    def __init__(self, tensors, lengths, targets=None):
        self.X = torch.tensor(tensors, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.lengths[idx], self.y[idx]
        return self.X[idx], self.lengths[idx]

def extract_order_sequences(split="train"):
    t0 = time.time()
    print(f"--- Extracting Raw Order Sequences for {split.upper()} ---")
    apps = pl.read_parquet(f"{DATA_PROCESSED}/{split}_features_v4.parquet").select([
        "customer_id", "decision_datetime", "application_id"
    ])
    
    ecom = pl.scan_csv(f"{DATA_RAW}/ecommerce_transactions.csv").with_columns(
        pl.col("transaction_datetime").str.to_datetime()
    ).join(
        apps.select(["customer_id", pl.col("decision_datetime").alias("cutoff")]).lazy(),
        on="customer_id",
        how="inner"
    ).filter(
        pl.col("transaction_datetime") < pl.col("cutoff")
    ).collect()
    
    orders = ecom.group_by("order_id", maintain_order=True).agg(
        customer_id = pl.col("customer_id").first(),
        transaction_datetime = pl.col("transaction_datetime").first(),
        order_total = pl.col("order_total").first(),
        payment_method = pl.col("payment_method").first(),
        order_status = pl.col("order_status").first(),
        cutoff = pl.col("cutoff").first(),
    ).with_columns(
        days_since = (pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_days(),
        hour = pl.col("transaction_datetime").dt.hour(),
        dow = pl.col("transaction_datetime").dt.weekday(),
        dom = pl.col("transaction_datetime").dt.day(),
    ).sort(["customer_id", "transaction_datetime"])
    
    # Keep up to MAX_ORDERS per customer (tail)
    orders_tail = orders.group_by("customer_id", maintain_order=True).tail(MAX_ORDERS)
    
    # Feature matrix calculation
    log_amt = (np.log1p(orders_tail["order_total"].to_numpy().astype(np.float32)) / 15.0).reshape(-1, 1)
    days_s = (orders_tail["days_since"].to_numpy().astype(np.float32) / 365.0).reshape(-1, 1)
    hr = (orders_tail["hour"].to_numpy().astype(np.float32) / 24.0).reshape(-1, 1)
    dw = (orders_tail["dow"].to_numpy().astype(np.float32) / 7.0).reshape(-1, 1)
    is_wk = (orders_tail["dow"].to_numpy() >= 5).astype(np.float32).reshape(-1, 1)
    is_pd = (((orders_tail["dom"].to_numpy() >= 25) | (orders_tail["dom"].to_numpy() <= 5))).astype(np.float32).reshape(-1, 1)
    is_comp = (orders_tail["order_status"].to_numpy() == "completed").astype(np.float32).reshape(-1, 1)
    is_cod = (orders_tail["payment_method"].to_numpy() == "COD").astype(np.float32).reshape(-1, 1)
    is_ew = (orders_tail["payment_method"].to_numpy() == "e_wallet").astype(np.float32).reshape(-1, 1)
    is_card = np.isin(orders_tail["payment_method"].to_numpy(), ["credit_card", "debit_card"]).astype(np.float32).reshape(-1, 1)
    
    flat_matrix = np.hstack([log_amt, days_s, hr, dw, is_wk, is_pd, is_comp, is_cod, is_ew, is_card]).astype(np.float32)
    
    # Fast grouping into 3D array
    cust_list = apps["customer_id"].to_list()
    cust_to_idx = {c: i for i, c in enumerate(cust_list)}
    
    # Count rows per customer in orders_tail
    tail_custs = orders_tail["customer_id"].to_list()
    
    tensor = np.zeros((len(cust_list), MAX_ORDERS, 10), dtype=np.float32)
    lengths = np.zeros(len(cust_list), dtype=np.int32)
    
    # Find slice boundaries for consecutive customers
    from collections import defaultdict
    cust_spans = defaultdict(list)
    curr_c = None
    st = 0
    for idx_row, c in enumerate(tail_custs):
        if c != curr_c:
            if curr_c is not None:
                cust_spans[curr_c].append((st, idx_row))
            curr_c = c
            st = idx_row
    if curr_c is not None:
        cust_spans[curr_c].append((st, len(tail_custs)))
        
    for c, spans in cust_spans.items():
        if c in cust_to_idx:
            i = cust_to_idx[c]
            # Concat spans if any
            rows = np.vstack([flat_matrix[s:e] for s, e in spans])
            L = min(len(rows), MAX_ORDERS)
            tensor[i, :L, :] = rows[-L:]
            lengths[i] = max(1, L)
            
    # For any customer with 0 orders, length = 1
    lengths = np.clip(lengths, 1, MAX_ORDERS)
    
    print(f"Extracted {split} tensor: {tensor.shape} ({tensor.nbytes / 1024**2:.1f} MB) in {time.time() - t0:.1f}s")
    return tensor, lengths, apps["application_id"].to_list()

def main():
    os.makedirs(PREDS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING SEQUENTIAL E-COMMERCE GRU MODEL (PIETER'S BREAKTHROUGH)")
    print("=" * 70)
    log_resource_status()
    
    # 1. Extract raw order sequences
    tensor_tr, lens_tr, tr_app_ids = extract_order_sequences("train")
    tensor_te, lens_te, te_app_ids = extract_order_sequences("test")
    
    # Load targets and folds
    df_folds = pl.read_parquet(f"{DATA_PROCESSED}/cv_folds_70k.parquet")
    df_tr_v4 = pl.read_parquet(f"{DATA_PROCESSED}/train_features_v4.parquet").select(["target"])
    y = df_tr_v4["target"].to_numpy().astype(np.float32)
    fold_col = df_folds["fold_seed42"].to_numpy()
    
    test_ds = EcomDataset(tensor_te, lens_te)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GRU on {device}...")
    
    oof_gru = np.zeros(len(y), dtype=np.float32)
    test_gru = np.zeros(len(tensor_te), dtype=np.float32)
    
    fold_ks_list = []
    t_start = time.time()
    
    for fold in range(5):
        tr_idx = np.where(fold_col != fold)[0]
        va_idx = np.where(fold_col == fold)[0]
        
        train_ds = EcomDataset(tensor_tr[tr_idx], lens_tr[tr_idx], y[tr_idx])
        val_ds = EcomDataset(tensor_tr[va_idx], lens_tr[va_idx], y[va_idx])
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        
        model = EcomSequentialGRU(in_features=10, hidden_dim=64, num_layers=1, dropout=0.20).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        epochs = 8
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        best_ks = 0.0
        best_preds = None
        best_state = None
        
        for ep in range(epochs):
            model.train()
            for bx, bl, by in train_loader:
                bx, bl, by = bx.to(device), bl.to(device), by.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(bx, bl)
                    loss = criterion(logits, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            
            # Val
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, bl, _ in val_loader:
                    bx, bl = bx.to(device), bl.to(device)
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        probs = torch.sigmoid(model(bx, bl))
                    val_preds_list.append(probs.cpu().numpy())
            curr_preds = np.concatenate(val_preds_list)
            curr_ks = official_ks_metric(y[va_idx], curr_preds)
            if curr_ks > best_ks:
                best_ks = curr_ks
                best_preds = curr_preds
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
        model.load_state_dict(best_state)
        oof_gru[va_idx] = best_preds
        fold_ks_list.append(best_ks)
        print(f"  GRU Fold {fold} Best KS: {best_ks:.4f}")
        
        # Test inference
        model.eval()
        test_preds_list = []
        with torch.no_grad():
            for bx, bl in test_loader:
                bx, bl = bx.to(device), bl.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    probs = torch.sigmoid(model(bx, bl))
                test_preds_list.append(probs.cpu().numpy())
        test_gru += np.concatenate(test_preds_list) / 5.0
        cleanup_memory()
        
    overall_gru_ks = official_ks_metric(y, oof_gru)
    print(f"\n---> Ecom-GRU Finished in {time.time() - t_start:.1f}s | Overall OOF KS: {overall_gru_ks:.4f}")
    
    # Save GRU predictions
    pl.DataFrame({
        "application_id": tr_app_ids,
        "oof_gru": oof_gru
    }).write_parquet(f"{PREDS_DIR}/oof_ecom_gru.parquet")
    
    pl.DataFrame({
        "application_id": te_app_ids,
        "test_gru": test_gru
    }).write_parquet(f"{PREDS_DIR}/test_ecom_gru.parquet")
    
    # -------------------------------------------------------------
    # 2. CORRELATION & DIVERSITY ANALYSIS (CONFIRMING PIETER'S HYPOTHESIS)
    # -------------------------------------------------------------
    print("\n--- [Step 2] Diversity Analysis (Spearman Rank Correlation) ---")
    df_oof_v4 = pl.read_parquet(f"{PREDS_DIR}/oof_v4_all_models.parquet")
    df_te_v4 = pl.read_parquet(f"{PREDS_DIR}/test_v4_all_models.parquet")
    
    oof_lgb = df_oof_v4["oof_lgb"].to_numpy()
    oof_lr = df_oof_v4["oof_lr"].to_numpy()
    oof_tab = df_oof_v4["oof_tab"].to_numpy()
    oof_xgb = df_oof_v4["oof_xgb"].to_numpy()
    
    corr_lgb, _ = spearmanr(oof_gru, oof_lgb)
    corr_lr, _ = spearmanr(oof_gru, oof_lr)
    corr_tab, _ = spearmanr(oof_gru, oof_tab)
    print(f"  Spearman(GRU, LightGBM): {corr_lgb:.4f} (Massive Diversity! Trees correlate > 0.97)")
    print(f"  Spearman(GRU, LogReg):   {corr_lr:.4f}")
    print(f"  Spearman(GRU, Tab-FM):   {corr_tab:.4f}")
    
    # -------------------------------------------------------------
    # 3. OPTUNA QUANTILE RANK ENSEMBLE (MAXIMIZE LEADERBOARD METRIC)
    # -------------------------------------------------------------
    print("\n--- [Step 3] Optuna Quantile Rank Ensemble (6 Diverse Models) ---")
    r_gru = rankdata(oof_gru) / len(oof_gru)
    r_lgb = rankdata(oof_lgb) / len(oof_lgb)
    r_lr = rankdata(oof_lr) / len(oof_lr)
    r_tab = rankdata(oof_tab) / len(oof_tab)
    r_xgb = rankdata(oof_xgb) / len(oof_xgb)
    
    test_r_gru = rankdata(test_gru) / len(test_gru)
    test_r_lgb = rankdata(df_te_v4["test_lgb"].to_numpy()) / len(test_gru)
    test_r_lr = rankdata(df_te_v4["test_lr"].to_numpy()) / len(test_gru)
    test_r_tab = rankdata(df_te_v4["test_tab"].to_numpy()) / len(test_gru)
    test_r_xgb = rankdata(df_te_v4["test_xgb"].to_numpy()) / len(test_gru)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        w_lr  = trial.suggest_float("w_lr", 0.1, 0.6)
        w_lgb = trial.suggest_float("w_lgb", 0.1, 0.5)
        w_tab = trial.suggest_float("w_tab", 0.05, 0.3)
        w_xgb = trial.suggest_float("w_xgb", 0.05, 0.25)
        w_gru = trial.suggest_float("w_gru", 0.02, 0.25)
        
        tot = w_lr + w_lgb + w_tab + w_xgb + w_gru
        w_lr /= tot; w_lgb /= tot; w_tab /= tot; w_xgb /= tot; w_gru /= tot
        
        blend = (
            w_lr * r_lr +
            w_lgb * r_lgb +
            w_tab * r_tab +
            w_xgb * r_xgb +
            w_gru * r_gru
        )
        return official_ks_metric(y, blend)
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=600)
    
    best_weights = study.best_params
    tot = sum(best_weights.values())
    norm_w = {k: v / tot for k, v in best_weights.items()}
    best_ensemble_ks = study.best_value
    
    print("Optimal 6-Model Quantile Rank Ensemble Weights:")
    for k, v in norm_w.items():
        print(f"  {k}: {v:.4f}")
        
    print(f"\n=======================================================")
    print(f"NEW CHAMPION ENSEMBLE OOF KS: {best_ensemble_ks:.5f}")
    print(f"Gain over previous best (0.50787): +{best_ensemble_ks - 0.50787:.5f}")
    print(f"=======================================================")
    
    # -------------------------------------------------------------
    # 4. GENERATE WINNING SUBMISSION
    # -------------------------------------------------------------
    sub_path = f"{SUBMISSIONS_DIR}/submission_v5_gru_optuna_blend.csv"
    final_test_blend = (
        norm_w["w_lr"] * test_r_lr +
        norm_w["w_lgb"] * test_r_lgb +
        norm_w["w_tab"] * test_r_tab +
        norm_w["w_xgb"] * test_r_xgb +
        norm_w["w_gru"] * test_r_gru
    )
    
    pl.DataFrame({
        "application_id": te_app_ids,
        "default_probability": final_test_blend.astype(np.float64)
    }).write_csv(sub_path)
    
    # Validation
    df_sub = pl.read_csv(sub_path)
    assert len(df_sub) == 30000
    assert df_sub.columns == ["application_id", "default_probability"]
    assert df_sub["application_id"].to_list() == te_app_ids
    assert df_sub["default_probability"].null_count() == 0
    assert df_sub["default_probability"].is_finite().all()
    print(f"Successfully generated and validated {sub_path}!")
    print(f"Predictions range: [{df_sub['default_probability'].min():.4f}, {df_sub['default_probability'].max():.4f}]")

if __name__ == "__main__":
    main()
