#!/usr/bin/env python3
"""
Shared Utilities for HPC Experiments (AI Center UB A100 SXM4 80GB)
"""
import os
import sys
import gc
import torch
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return ks_2samp(preds_bad, preds_good).statistic

def normalize_rank(arr):
    return (rankdata(arr) - 0.5) / len(arr)

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def find_file(filename, custom_dir=None):
    candidates = []
    if custom_dir:
        candidates.append(os.path.join(custom_dir, filename))
    candidates.extend([
        f"data/{filename}",
        f"../data/{filename}",
        f"../data/processed/{filename}",
        f"../data/raw/{filename}",
        f"HPC/data/{filename}",
        f"HPC/anchors/{filename}",
        f"anchors/{filename}",
        f"/home/neva/Downloads/{filename}",
        filename
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not locate {filename} in searched paths: {candidates}")

def load_clean_dataset(custom_dir=None):
    """
    Loads train_features_v5 & test_features_v5 with:
    1. Leaky rank_* & pricing_* features purged.
    2. Out-of-fold Weight of Evidence (WOE) on categoricals.
    3. 2-way categorical interaction risk features.
    4. Exact population masks (Pop 1 Full, Pop 2 Thin, Pop 3 Rejected).
    """
    tr_path = find_file("train_features_v5.parquet", custom_dir)
    te_path = find_file("test_features_v5.parquet", custom_dir)
    folds_path = find_file("cv_folds_70k.parquet", custom_dir)

    tr_df = pl.read_parquet(tr_path)
    te_df = pl.read_parquet(te_path)
    df_folds = pl.read_parquet(folds_path)
    fold_col = df_folds["fold_seed42"].to_numpy()
    y = tr_df["target"].to_numpy().astype(np.int32)

    # 1. Purge leaky rank_* features that triggered 1.0 AUC adversarial shift
    leaky_patterns = ["rank_", "pricing_predicted_rate", "pricing_rate_residual", "cross_app_id_hash"]
    cols_to_drop = [c for c in tr_df.columns if any(pat in c for pat in leaky_patterns)]
    clean_tr = tr_df.drop(cols_to_drop)
    clean_te = te_df.drop(cols_to_drop)

    # 2. Out-of-Fold WOE
    cat_cols = ["region", "loan_purpose", "employment_type", "housing_status"]
    cat_cols = [c for c in cat_cols if c in clean_tr.columns]
    pdf_tr_cat = clean_tr.select(cat_cols).to_pandas()
    pdf_te_cat = clean_te.select(cat_cols).to_pandas()

    woe_tr_dict = {}
    woe_te_dict = {}

    for c in cat_cols:
        woe_tr = np.zeros(len(clean_tr), dtype=np.float32)
        woe_te = np.zeros(len(clean_te), dtype=np.float32)

        for f in range(5):
            idx_tr = np.where(fold_col != f)[0]
            idx_va = np.where(fold_col == f)[0]

            df_fold = pd.DataFrame({"cat": pdf_tr_cat[c].iloc[idx_tr], "target": y[idx_tr]})
            grouped = df_fold.groupby("cat")["target"].agg(["count", "sum"])
            total_pos = y[idx_tr].sum()
            total_neg = len(idx_tr) - total_pos

            pos = grouped["sum"] + 1.0
            neg = (grouped["count"] - grouped["sum"]) + 1.0
            woe_map = np.log((pos / total_pos) / (neg / total_neg)).to_dict()

            woe_tr[idx_va] = pdf_tr_cat[c].iloc[idx_va].map(woe_map).fillna(0.0).values
            woe_te += pdf_te_cat[c].map(woe_map).fillna(0.0).values / 5.0

        woe_tr_dict[f"woe_{c}"] = woe_tr
        woe_te_dict[f"woe_{c}"] = woe_te

    # 3. Two-way Interaction Risk Categories
    tr_reg_emp = (clean_tr["region"].fill_null("UNK") + "_" + clean_tr["employment_type"].fill_null("UNK")).to_list()
    te_reg_emp = (clean_te["region"].fill_null("UNK") + "_" + clean_te["employment_type"].fill_null("UNK")).to_list()

    tr_purp_house = (clean_tr["loan_purpose"].fill_null("UNK") + "_" + clean_tr["housing_status"].fill_null("UNK")).to_list()
    te_purp_house = (clean_te["loan_purpose"].fill_null("UNK") + "_" + clean_te["housing_status"].fill_null("UNK")).to_list()

    clean_tr = clean_tr.with_columns([
        pl.Series(k, v) for k, v in woe_tr_dict.items()
    ]).with_columns([
        pl.Series("cat_region_employment", tr_reg_emp),
        pl.Series("cat_purpose_housing", tr_purp_house)
    ])

    clean_te = clean_te.with_columns([
        pl.Series(k, v) for k, v in woe_te_dict.items()
    ]).with_columns([
        pl.Series("cat_region_employment", te_reg_emp),
        pl.Series("cat_purpose_housing", te_purp_house)
    ])

    # Population masks
    tr_has_prev = clean_tr["cov_has_prev_apps"].to_numpy()
    tr_has_repay = clean_tr["cov_has_repayment"].to_numpy()
    tr_has_acct = clean_tr["cov_has_account"].to_numpy()
    tr_pop1 = (tr_has_repay == 1) & (tr_has_acct == 1)
    tr_pop3 = (tr_has_prev == 1) & (tr_has_repay == 0)
    tr_pop2 = (~tr_pop1) & (~tr_pop3)

    te_has_prev = clean_te["cov_has_prev_apps"].to_numpy()
    te_has_repay = clean_te["cov_has_repayment"].to_numpy()
    te_has_acct = clean_te["cov_has_account"].to_numpy()
    te_pop1 = (te_has_repay == 1) & (te_has_acct == 1)
    te_pop3 = (te_has_prev == 1) & (te_has_repay == 0)
    te_pop2 = (~te_pop1) & (~te_pop3)

    masks = {
        "tr_pop1": tr_pop1, "tr_pop2": tr_pop2, "tr_pop3": tr_pop3,
        "te_pop1": te_pop1, "te_pop2": te_pop2, "te_pop3": te_pop3,
    }

    return clean_tr, clean_te, y, fold_col, masks

def evaluate_metrics(y_true, y_pred, fold_col, masks):
    """
    Computes full comprehensive evaluation metrics:
    Overall OOF KS, Min Fold KS, Mean Fold KS, Std Fold KS, and Population KS
    """
    fold_scores = [official_ks_metric(y_true[fold_col == f], y_pred[fold_col == f]) for f in range(5)]
    ov_ks = official_ks_metric(y_true, y_pred)
    p1_ks = official_ks_metric(y_true[masks["tr_pop1"]], y_pred[masks["tr_pop1"]])
    p2_ks = official_ks_metric(y_true[masks["tr_pop2"]], y_pred[masks["tr_pop2"]])
    p3_ks = official_ks_metric(y_true[masks["tr_pop3"]], y_pred[masks["tr_pop3"]])
    
    return {
        "overall_ks": ov_ks,
        "min_fold_ks": min(fold_scores),
        "mean_fold_ks": float(np.mean(fold_scores)),
        "std_fold_ks": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "pop1_ks": p1_ks,
        "pop2_ks": p2_ks,
        "pop3_ks": p3_ks
    }

def log_experiment_to_csv(exp_id, model_name, desc, metrics, log_path="hpc_experiments_summary.csv"):
    header = not os.path.exists(log_path)
    df_row = pd.DataFrame([{
        "exp_id": exp_id,
        "model_name": model_name,
        "description": desc,
        "overall_ks": f"{metrics['overall_ks']:.4f}",
        "min_fold_ks": f"{metrics['min_fold_ks']:.4f}",
        "mean_fold_ks": f"{metrics['mean_fold_ks']:.4f}",
        "std_fold_ks": f"{metrics['std_fold_ks']:.4f}",
        "pop1_ks": f"{metrics['pop1_ks']:.4f}",
        "pop2_ks": f"{metrics['pop2_ks']:.4f}",
        "pop3_ks": f"{metrics['pop3_ks']:.4f}",
        "fold_scores": str([round(s, 4) for s in metrics["fold_scores"]])
    }])
    df_row.to_csv(log_path, mode="a", index=False, header=header)
    print(f"[*] Logged {exp_id} to {log_path}")
