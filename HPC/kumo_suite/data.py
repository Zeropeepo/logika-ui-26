"""ID-checked inputs and deterministic, label-free feature definitions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

COMPACT = """loan_amount annual_interest_rate loan_purpose age region employment_type
employment_tenure_months declared_monthly_income dependents housing_status
installment_income_ratio loan_income_ratio income_per_dependent hours_to_decision
prev_app_count prev_approval_rate prev_rejection_rate prev_recency_days
repay_installment_count repay_max_max_dpd repay_mean_dpd repay_paid_over_due_ratio
repay_recent6_max_dpd repay_recent6_underpaid_ratio repay_days_since_last_delay
acct_observed_months acct_mean_income_credits_3m acct_mean_inflows_3m acct_mean_outflows_3m
acct_mean_balance_3m acct_outflow_inflow_ratio acct_slope_outflows acct_slope_low_balance_days
bureau_score_mean bureau_score_min bureau_score_Bureau_A bureau_score_Bureau_B bureau_score_Bureau_C
bureau_max_delinquent_12m bureau_max_inquiries_6m bureau_recalculated_utilization
ecom_spending_30d ecom_orders_30d ecom_cod_share_30d ecom_ratio_spend_30d_90d
ecom_recency_days ecom_cancellation_rate cov_has_repayment cov_has_prev_apps cov_has_account""".split()
META = {"application_id", "customer_id", "application_datetime", "decision_datetime", "target"}
BLOCKED = {"pricing_predicted_rate", "pricing_rate_residual", "cross_app_id_hash",
           "composite_risk_score", "composite_risk_score_v2"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def population(df):
    return np.where(df.cov_has_repayment.fillna(0).to_numpy() > 0, "full",
                    np.where(df.cov_has_prev_apps.fillna(0).to_numpy() > 0, "prev_no_repay", "thin"))


def load(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    tr, te, folds = [pl.read_parquet(folder / n).to_pandas() for n in ("train.parquet", "test.parquet", "folds.parquet")]
    for df in (tr, te, folds):
        if df.application_id.isna().any() or not df.application_id.is_unique:
            raise ValueError("IDs must be unique and non-null")
    if set(tr.application_id) & set(te.application_id):
        raise ValueError("Train/test IDs overlap")
    if not np.isin(tr.target, [0, 1]).all() or tr.target.nunique() != 2:
        raise ValueError("Need binary targets with both classes")
    if set(tr.application_id) != set(folds.application_id):
        raise ValueError("Fold IDs differ from train")
    folds = tr[["application_id"]].merge(folds, on="application_id", validate="one_to_one", sort=False)
    for seed in (42, 2026, 999):
        col = f"fold_seed{seed}"
        if col not in folds or set(folds[col]) != set(range(5)):
            raise ValueError(f"Invalid {col}")
    if set(tr.customer_id) & set(te.customer_id) or not tr.customer_id.is_unique or not te.customer_id.is_unique:
        raise ValueError("This runner requires disjoint, one-row-per-customer inputs")
    return tr, te, folds, manifest


def columns(df, exp):
    if exp.view == "compact":
        missing = set(COMPACT) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        cols = COMPACT.copy()
    elif exp.view == "enhanced":
        cols = COMPACT + sorted(c for c in df if c.startswith("new_"))
    else:
        cols = [c for c in df if c not in META and c not in BLOCKED and not c.startswith("rank_")]
    if exp.drop != "none":
        prefixes = {"repay": ("repay_", "new_rep_"), "ecom": ("ecom_", "new_eco_"),
                    "bureau": ("bureau_", "new_bur_"), "acct": ("acct_", "new_acc_"),
                    "prev": ("prev_",), "calendar": ("cal_", "hours_to_decision"),
                    "pricing": ("annual_interest_rate", "monthly_interest", "proposed_monthly",
                                "total_repayment", "interest_")}[exp.drop]
        # Derived cross-source columns are excluded too, so source removal is interpretable.
        cols = [c for c in cols if not c.startswith(prefixes + ("cross_", "ratio_", "flag_"))]
        if exp.drop == "repay":
            cols = [c for c in cols if c != "cov_has_repayment"]
        if exp.drop == "prev":
            cols = [c for c in cols if c != "cov_has_prev_apps"]
        if exp.drop == "acct":
            cols = [c for c in cols if c != "cov_has_account"]
        if exp.drop == "pricing":
            cols = [c for c in cols if c not in {"installment_income_ratio", "new_eco_to_installment"}]
    return cols


def transform(train, query, mode):
    """All learned transforms fit on outer training rows only; never query labels."""
    train, query = train.copy(), query.copy()
    numeric = train.select_dtypes(include="number").columns.tolist()
    train[numeric] = train[numeric].replace([np.inf, -np.inf], np.nan)
    query[numeric] = query[numeric].replace([np.inf, -np.inf], np.nan)
    # Remove constants and exact duplicate features on outer training only.
    kept, hashes = [], {}
    for c in train:
        if train[c].nunique(dropna=False) <= 1:
            continue
        key = hashlib.sha256(pd.util.hash_pandas_object(train[c], index=False).values.tobytes()).hexdigest()
        if key in hashes and train[c].equals(train[hashes[key]]):
            continue
        hashes[key] = c
        kept.append(c)
    train, query = train[kept], query[kept]
    for c in train:
        if c not in numeric:
            train[c] = train[c].where(train[c].notna(), None).astype(object)
            query[c] = query[c].where(query[c].notna(), None).astype(object)
        elif mode == "log":
            for df in (train, query):
                df[c] = np.sign(df[c]) * np.log1p(np.abs(df[c]))
        elif mode == "rank":
            ref = np.sort(train[c].dropna().to_numpy())
            for df in (train, query):
                v = df[c].to_numpy()
                ranks = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / (2 * max(1, len(ref)))
                df[c] = np.where(pd.isna(v), np.nan, ranks)
        elif mode == "median_indicator":
            if train[c].isna().any():
                for df in (train, query):
                    df[c + "__missing"] = df[c].isna().astype(float)
            med = train[c].median()
            train[c] = train[c].fillna(med)
            query[c] = query[c].fillna(med)
    return train, query
