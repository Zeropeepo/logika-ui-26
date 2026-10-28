"""
Deep Tabular Foundation Architecture (Tab-ResNet / RealMLP)
Features:
- RankGauss / Quantile Normal Scaling + Median Imputation
- Input Feature Dropout (p=0.25)
- Tabular CutMix Augmentation (p=0.20)
- Deep Residual MLP with LayerNorm, GELU, and High Decoupled Weight Decay (AdamW)
- Mixed Precision FP16 (autocast)
- Cosine Annealing LR Schedule
- Early stopping on exact two-sample Kolmogorov-Smirnov (KS) Statistic
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
    def __init__(self, in_features, hidden_dim=256, feature_drop=0.25, dropout=0.25):
        super().__init__()
        self.feature_dropout = nn.Dropout(feature_drop)
        
        # Initial projection
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ResBlock 1
        self.block1_1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1_1 = nn.LayerNorm(hidden_dim)
        self.act1_1 = nn.GELU()
        self.drop1_1 = nn.Dropout(dropout)
        self.block1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1_2 = nn.LayerNorm(hidden_dim)
        self.drop1_2 = nn.Dropout(dropout)
        
        # Transition to 128
        self.trans = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ResBlock 2
        self.block2_1 = nn.Linear(128, 128)
        self.norm2_1 = nn.LayerNorm(128)
        self.act2_1 = nn.GELU()
        self.drop2_1 = nn.Dropout(dropout)
        self.block2_2 = nn.Linear(128, 128)
        self.norm2_2 = nn.LayerNorm(128)
        self.drop2_2 = nn.Dropout(dropout)
        
        # Head
        self.head = nn.Linear(128, 1)
        
    def forward(self, x):
        # Input feature dropout
        if self.training:
            x = self.feature_dropout(x)
            
        h = self.in_proj(x)
        
        # Block 1
        res1 = h
        h1 = self.drop1_1(self.act1_1(self.norm1_1(self.block1_1(h))))
        h1 = self.drop1_2(self.norm1_2(self.block1_2(h1)))
        h = F.gelu(h1 + res1)
        
        # Trans
        h = self.trans(h)
        
        # Block 2
        res2 = h
        h2 = self.drop2_1(self.act2_1(self.norm2_1(self.block2_1(h))))
        h2 = self.drop2_2(self.norm2_2(self.block2_2(h2)))
        h = F.gelu(h2 + res2)
        
        out = self.head(h).squeeze(-1)
        return out

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
            # CutMix: swap 20% of features with random other sample
            rand_idx = np.random.randint(0, len(self.X))
            mask = torch.rand(x.shape) < 0.20
            x = torch.where(mask, self.X[rand_idx], x)
            
        if self.y is not None:
            return x, self.y[idx]
        return x

def run_test():
    print("Loading data for Tab-ResNet test...")
    df = pl.read_parquet("data/processed/train_features_v4.parquet")
    df_folds = pl.read_parquet("data/processed/cv_folds_70k.parquet")
    
    y = df["target"].to_numpy().astype(np.float32)
    feature_cols = [c for c in df.columns if c not in EVAL_ONLY_COLUMNS]
    
    # Categorical handling: frequency encoding
    cat_cols = [c for c in feature_cols if df[c].dtype == pl.String]
    print(f"Features: {len(feature_cols)}, Categorical: {cat_cols}")
    
    pdf = df.select(feature_cols).to_pandas()
    for c in cat_cols:
        freq = pdf[c].value_counts(normalize=True).to_dict()
        pdf[c] = pdf[c].map(freq).astype(np.float32)
        
    # Fold 0 test
    fold_col = df_folds["fold_seed42"].to_numpy()
    tr_mask = (fold_col != 0)
    va_mask = (fold_col == 0)
    
    print("Applying Median Imputer and RankGauss (QuantileTransformer)...")
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(pdf.iloc[tr_mask])
    X_va_imp = imputer.transform(pdf.iloc[va_mask])
    
    # RankGauss: quantile transformer with normal output
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
    X_tr_norm = qt.fit_transform(X_tr_imp).astype(np.float32)
    X_va_norm = qt.transform(X_va_imp).astype(np.float32)
    
    # Clip extreme normal tails to [-5, 5]
    X_tr_norm = np.clip(X_tr_norm, -5.0, 5.0)
    X_va_norm = np.clip(X_va_norm, -5.0, 5.0)
    
    y_tr = y[tr_mask]
    y_va = y[va_mask]
    
    train_ds = TabularDataset(X_tr_norm, y_tr, is_train=True, cutmix_prob=0.20)
    val_ds = TabularDataset(X_va_norm, y_va, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = TabularResNet(in_features=X_tr_norm.shape[1], hidden_dim=256, feature_drop=0.25, dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_ks = 0.0
    best_preds = None
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(bx)
                # Label smoothing: y_smooth = y * 0.98 + 0.01
                by_smooth = by * 0.98 + 0.01
                loss = criterion(logits, by_smooth)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(bx)
            
        scheduler.step()
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(bx)
                    probs = torch.sigmoid(logits)
                val_preds_list.append(probs.cpu().numpy())
                
        val_preds = np.concatenate(val_preds_list)
        ks = official_ks_metric(y_va, val_preds)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val KS: {ks:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if ks > best_ks:
            best_ks = ks
            best_preds = val_preds
            
    print(f"\n--- Tab-ResNet Test Completed in {time.time() - t0:.1f}s ---")
    print(f"Best Fold 0 KS: {best_ks:.4f}")
    log_resource_status()

if __name__ == "__main__":
    run_test()
