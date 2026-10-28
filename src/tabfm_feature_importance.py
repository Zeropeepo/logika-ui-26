"""
Tab-FM Feature Importance & Representativeness Analysis
Trains Deep Tabular Foundation Model (Tab-ResNet with RankGauss & Feature Dropout) on v5 features (409 cols).
Computes:
1. Exact validation KS on Fold 0.
2. Input Gradient Attribution Saliency across all 14,000 validation samples.
3. Permutation Feature Importance (measuring exact drop in validation KS: delta_KS = KS_orig - KS_perm).
Identifies the most critical and representative features powering Tab-FM.
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
from scipy.stats import ks_2samp
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer

from src.utils_watchdog import cleanup_memory, log_resource_status

EVAL_ONLY_COLUMNS = ["target", "application_id", "customer_id", "application_datetime", "decision_datetime", "cutoff"]

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

class TabularResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=256, feature_drop=0.20, dropout=0.20):
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
    def __init__(self, X, y=None, is_train=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def main():
    print("=" * 70)
    print("TAB-FM FEATURE IMPORTANCE & REPRESENTATIVENESS PROFILING")
    print("=" * 70)
    log_resource_status()
    
    # 1. Load data
    df = pl.read_parquet("data/processed/train_features_v5.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    
    y = df["target"].to_numpy().astype(np.float32)
    feature_cols = [c for c in df.columns if c not in EVAL_ONLY_COLUMNS]
    cat_cols = [c for c in feature_cols if df[c].dtype == pl.String]
    print(f"Total input features: {len(feature_cols)} (Categorical: {len(cat_cols)})")
    
    pdf = df.select(feature_cols).to_pandas()
    for c in cat_cols:
        freq = pdf[c].value_counts(normalize=True).to_dict()
        pdf[c] = pdf[c].map(freq).fillna(0.0).astype(np.float32)
        
    fold_col = df_folds["fold_seed42"].to_numpy()
    tr_mask = (fold_col != 0)
    va_mask = (fold_col == 0)
    
    print("Transforming manifold via Median Imputer & RankGauss QuantileTransformer...")
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(pdf.iloc[tr_mask])
    X_va_imp = imputer.transform(pdf.iloc[va_mask])
    
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
    X_tr_norm = np.clip(qt.fit_transform(X_tr_imp).astype(np.float32), -5.0, 5.0)
    X_va_norm = np.clip(qt.transform(X_va_imp).astype(np.float32), -5.0, 5.0)
    
    y_tr = y[tr_mask]
    y_va = y[va_mask]
    
    train_ds = TabularDataset(X_tr_norm, y_tr, is_train=True)
    val_ds = TabularDataset(X_va_norm, y_va, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Tab-FM model on {device}...")
    
    model = TabularResNet(in_features=X_tr_norm.shape[1], hidden_dim=256, feature_drop=0.20, dropout=0.20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_ks = 0.0
    best_state = None
    t0 = time.time()
    
    for epoch in range(epochs):
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
        
        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    probs = torch.sigmoid(model(bx))
                val_preds_list.append(probs.cpu().numpy())
        val_preds = np.concatenate(val_preds_list)
        ks = official_ks_metric(y_va, val_preds)
        if ks > best_ks:
            best_ks = ks
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    print(f"Tab-FM training completed in {time.time() - t0:.1f}s | Baseline Val KS: {best_ks:.4f}")
    
    # -------------------------------------------------------------
    # 2. GRADIENT SALIENCY ATTRIBUTION
    # -------------------------------------------------------------
    print("\n--- [Step 2] Computing Gradient Saliency (|x * d(logit)/dx|) ---")
    model.eval()
    X_va_tensor = torch.tensor(X_va_norm, dtype=torch.float32, device=device, requires_grad=True)
    
    # Forward pass without autocast for exact gradients
    logits = model(X_va_tensor)
    loss_sum = logits.sum()
    loss_sum.backward()
    
    grads = X_va_tensor.grad.detach().cpu().numpy()
    saliency = np.mean(np.abs(grads * X_va_norm), axis=0) # shape (N_features,)
    
    top_indices_sal = np.argsort(saliency)[::-1]
    
    # -------------------------------------------------------------
    # 3. PERMUTATION FEATURE IMPORTANCE (MEASURING DROP IN KS)
    # -------------------------------------------------------------
    print("\n--- [Step 3] Running Permutation Feature Importance on Top 35 Candidates ---")
    top_k = 35
    candidate_indices = top_indices_sal[:top_k]
    
    perm_results = []
    
    # Baseline predictions
    with torch.no_grad():
        base_logits = model(torch.tensor(X_va_norm, dtype=torch.float32, device=device))
        base_probs = torch.sigmoid(base_logits).cpu().numpy()
    base_ks = official_ks_metric(y_va, base_probs)
    
    rng = np.random.RandomState(42)
    for idx in candidate_indices:
        feat_name = feature_cols[idx]
        
        # Create permuted copy
        X_perm = X_va_norm.copy()
        X_perm[:, idx] = rng.permutation(X_perm[:, idx])
        
        with torch.no_grad():
            perm_logits = model(torch.tensor(X_perm, dtype=torch.float32, device=device))
            perm_probs = torch.sigmoid(perm_logits).cpu().numpy()
            
        perm_ks = official_ks_metric(y_va, perm_probs)
        drop = base_ks - perm_ks
        sal_val = saliency[idx]
        perm_results.append((feat_name, drop, sal_val, perm_ks))
        
    # Sort by KS Drop (descending)
    perm_results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 78)
    print(f"{'Rank':<4} {'Feature Name':<38} {'KS Drop (Delta)':<18} {'Saliency':<10}")
    print("-" * 78)
    for i, (name, drop, sal, p_ks) in enumerate(perm_results[:30], 1):
        print(f"{i:<4} {name:<38} {drop:<18.5f} {sal:<10.5f}")
    print("=" * 78)
    
    # Save results to CSV
    res_df = pl.DataFrame({
        "rank": list(range(1, len(perm_results) + 1)),
        "feature_name": [x[0] for x in perm_results],
        "ks_drop": [x[1] for x in perm_results],
        "saliency": [x[2] for x in perm_results],
        "permuted_val_ks": [x[3] for x in perm_results]
    })
    res_df.write_csv("reports/tabfm_representative_features.csv")
    print("\nSaved full ranking to reports/tabfm_representative_features.csv")

if __name__ == "__main__":
    main()
