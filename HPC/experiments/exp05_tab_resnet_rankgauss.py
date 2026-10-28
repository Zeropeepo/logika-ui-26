#!/usr/bin/env python3
"""
EXP 05: Deep Tabular ResNet with RankGauss Quantile Normal Transformation
Architecture:
- Dense Residual Blocks with LayerNorm + Swish Activation + Dropout
- QuantileTransformer(output_distribution='normal') (RankGauss)
- CosineAnnealingLR + AdamW (Weight Decay 0.05)
- AMP FP16 on NVIDIA A100
Provides highly orthogonal non-linear representations.
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_hpc import (
    load_clean_dataset, evaluate_metrics, log_experiment_to_csv,
    cleanup_gpu, normalize_rank
)

class TabularResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=256, num_blocks=3, dropout=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            TabularResBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.input_layer(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)

def run_experiment():
    print("=" * 80)
    print("STARTING EXP 05: DEEP TABULAR RESNET WITH RANKGAUSS NORMALIZATION")
    print("=" * 80)
    t_start = time.time()
    
    clean_tr, clean_te, y, fold_col, masks = load_clean_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    
    ignore_cols = {"target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff", "cat_region_employment", "cat_purpose_housing", "region", "loan_purpose", "employment_type", "housing_status"}
    num_features = [c for c in clean_tr.columns if c not in ignore_cols and clean_tr[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    print(f"[*] Transforming {len(num_features)} numeric features with RankGauss...")
    
    qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal", random_state=42)
    X_tr_num = clean_tr.select(num_features).to_pandas().fillna(0).values.astype(np.float32)
    X_te_num = clean_te.select(num_features).to_pandas().fillna(0).values.astype(np.float32)
    
    X_tr_scaled = np.nan_to_num(qt.fit_transform(X_tr_num), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)
    X_te_scaled = np.nan_to_num(qt.transform(X_te_num), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)
    
    oof_tab = np.zeros(len(y), dtype=np.float32)
    test_tab = np.zeros(len(clean_te), dtype=np.float32)
    
    batch_size = 512
    epochs = 12
    
    te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32)
    te_loader = DataLoader(TensorDataset(te_tensor), batch_size=batch_size * 2, shuffle=False)
    
    for f in range(5):
        t_f = time.time()
        idx_tr = np.where(fold_col != f)[0]
        idx_va = np.where(fold_col == f)[0]
        
        tr_ds = TensorDataset(torch.tensor(X_tr_scaled[idx_tr]), torch.tensor(y[idx_tr], dtype=torch.float32))
        va_ds = TensorDataset(torch.tensor(X_tr_scaled[idx_va]), torch.tensor(y[idx_va], dtype=torch.float32))
        
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False)
        
        model = TabularResNet(in_features=len(num_features), hidden_dim=256, num_blocks=3, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")
        
        for ep in range(epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    logits = model(bx)
                    loss = criterion(logits, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            
        # Eval validation
        model.eval()
        va_preds = []
        with torch.no_grad():
            for bx, in va_loader:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    p = torch.sigmoid(model(bx)).cpu().numpy()
                va_preds.append(p)
        oof_tab[idx_va] = np.concatenate(va_preds)
        
        # Eval test
        te_preds = []
        with torch.no_grad():
            for bx, in te_loader:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    p = torch.sigmoid(model(bx)).cpu().numpy()
                te_preds.append(p)
        test_tab += np.concatenate(te_preds) / 5.0
        
        del model, optimizer, scheduler
        cleanup_gpu()
        print(f"    Fold {f} KS: {official_ks_metric(y[idx_va], oof_tab[idx_va]):.4f} ({time.time() - t_f:.1f}s)")
        
    metrics = evaluate_metrics(y, oof_tab, fold_col, masks)
    print("\n" + "=" * 80)
    print(f"EXP 05 COMPLETE: Overall KS={metrics['overall_ks']:.4f} | Min Fold={metrics['min_fold_ks']:.4f} | Pop1={metrics['pop1_ks']:.4f} | Pop2={metrics['pop2_ks']:.4f} | Pop3={metrics['pop3_ks']:.4f}")
    print(f"Time Taken: {time.time() - t_start:.1f}s")
    print("=" * 80)
    
    os.makedirs("preds", exist_ok=True)
    pl.DataFrame({"application_id": clean_tr["application_id"], "oof_tab_resnet": oof_tab}).write_parquet("preds/oof_exp05_tab_resnet.parquet")
    pl.DataFrame({"application_id": clean_te["application_id"], "test_tab_resnet": test_tab}).write_parquet("preds/test_exp05_tab_resnet.parquet")
    
    os.makedirs("submissions", exist_ok=True)
    p_norm = normalize_rank(test_tab)
    pl.DataFrame({"application_id": clean_te["application_id"], "default_probability": p_norm.astype(np.float64)}).write_csv("submissions/sub_exp05_tab_resnet.csv")
    
    log_experiment_to_csv("EXP-05", "Deep_Tab_ResNet_RankGauss", "3-Block ResNet + RankGauss + AdamW", metrics)
    return metrics

if __name__ == "__main__":
    run_experiment()
