"""
Stage 0 & Stage 1: Data Loading Helper & Comprehensive Profiling
"""
import os
import json
import polars as pl
import numpy as np

DATA_DIR = "data/raw"

def get_table_metadata():
    tables = [
        "train_applications.csv",
        "test_applications.csv",
        "customer_profiles.csv",
        "previous_applications.csv",
        "repayment_history.csv",
        "account_behavior.csv",
        "bureau_observations.csv",
        "ecommerce_transactions.csv",
        "sample_submission.csv",
    ]
    return tables

def profile_table(name, df: pl.DataFrame):
    n_rows = len(df)
    n_cols = len(df.columns)
    mem_mb = df.estimated_size() / (1024 * 1024)
    
    col_profiles = {}
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        null_count = s.null_count()
        null_pct = (null_count / n_rows) * 100 if n_rows > 0 else 0
        n_unique = s.n_unique()
        
        info = {
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": round(null_pct, 4),
            "n_unique": n_unique,
        }
        
        if s.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
            valid = s.drop_nulls()
            if len(valid) > 0:
                # quantiles
                q0 = valid.min()
                q1 = valid.quantile(0.01)
                q50 = valid.median()
                q99 = valid.quantile(0.99)
                q100 = valid.max()
                info["stats"] = {
                    "min": float(q0),
                    "p1": float(q1),
                    "p50": float(q50),
                    "p99": float(q99),
                    "max": float(q100),
                    "zero_count": int((valid == 0).sum()),
                    "neg_count": int((valid < 0).sum()),
                }
        elif s.dtype in [pl.Datetime, pl.Date]:
            valid = s.drop_nulls()
            if len(valid) > 0:
                info["date_range"] = {
                    "min": str(valid.min()),
                    "max": str(valid.max()),
                }
        elif s.dtype == pl.String:
            # check if datetime parseable sample
            valid = s.drop_nulls()
            if len(valid) > 0 and n_unique <= 20:
                val_counts = valid.value_counts().to_dicts()
                info["categories"] = {d[col]: d["count"] for d in val_counts}
        
        col_profiles[col] = info

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "mem_mb": round(mem_mb, 2),
        "columns": col_profiles,
    }

def run_profiling():
    print("Loading tables...")
    
    # 1. train_applications
    df_train = pl.read_csv(f"{DATA_DIR}/train_applications.csv", try_parse_dates=True)
    prof_train = profile_table("train_applications", df_train)
    
    # 2. test_applications
    df_test = pl.read_csv(f"{DATA_DIR}/test_applications.csv", try_parse_dates=True)
    prof_test = profile_table("test_applications", df_test)
    
    # 3. customer_profiles
    df_cust = pl.read_csv(f"{DATA_DIR}/customer_profiles.csv")
    prof_cust = profile_table("customer_profiles", df_cust)
    
    # 4. previous_applications
    df_prev = pl.read_csv(f"{DATA_DIR}/previous_applications.csv", try_parse_dates=True)
    prof_prev = profile_table("previous_applications", df_prev)
    
    # 5. repayment_history
    df_repay = pl.read_csv(f"{DATA_DIR}/repayment_history.csv", try_parse_dates=True)
    prof_repay = profile_table("repayment_history", df_repay)
    
    # 6. account_behavior
    df_acct = pl.read_csv(f"{DATA_DIR}/account_behavior.csv", try_parse_dates=True)
    prof_acct = profile_table("account_behavior", df_acct)
    
    # 7. bureau_observations
    df_bureau = pl.read_csv(f"{DATA_DIR}/bureau_observations.csv", try_parse_dates=True)
    prof_bureau = profile_table("bureau_observations", df_bureau)
    
    # 8. ecommerce_transactions
    df_ecom = pl.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv", try_parse_dates=True)
    prof_ecom = profile_table("ecommerce_transactions", df_ecom)
    
    # 9. sample_submission
    df_sub = pl.read_csv(f"{DATA_DIR}/sample_submission.csv")
    prof_sub = profile_table("sample_submission", df_sub)
    
    # Verifications for specific claims
    print("Running specific claim verifications...")
    
    # A. customer_profiles: 1 row per customer_id?
    cust_total = len(df_cust)
    cust_unique = df_cust["customer_id"].n_unique()
    cust_dups = cust_total - cust_unique
    
    # B. customer applications count in train/test
    train_cust_counts = df_train["customer_id"].value_counts()["count"].to_list()
    test_cust_counts = df_test["customer_id"].value_counts()["count"].to_list()
    
    # C. overlap customer train vs test
    train_custs = set(df_train["customer_id"].unique())
    test_custs = set(df_test["customer_id"].unique())
    cust_overlap = len(train_custs.intersection(test_custs))
    
    train_apps = set(df_train["application_id"].unique())
    test_apps = set(df_test["application_id"].unique())
    app_overlap = len(train_apps.intersection(test_apps))
    
    # D. previous_applications overlap with train/test application_id
    prev_apps = set(df_prev["application_id"].unique())
    prev_in_train = len(prev_apps.intersection(train_apps))
    prev_in_test = len(prev_apps.intersection(test_apps))
    
    # E. Bureau: observations per application_id, per provider, diff report vs available
    bureau_apps = df_bureau["application_id"].n_unique()
    bureau_per_app = df_bureau["application_id"].value_counts()["count"]
    bureau_per_prov = df_bureau["bureau_provider"].value_counts().to_dicts()
    
    # report_datetime vs available_datetime difference
    df_bureau_diff = df_bureau.with_columns(
        diff_hours = (pl.col("available_datetime") - pl.col("report_datetime")).dt.total_hours()
    )
    diff_stats = {
        "min_hours": float(df_bureau_diff["diff_hours"].min()),
        "p50_hours": float(df_bureau_diff["diff_hours"].median()),
        "max_hours": float(df_bureau_diff["diff_hours"].max()),
        "neg_count": int((df_bureau_diff["diff_hours"] < 0).sum()),
        "zero_count": int((df_bureau_diff["diff_hours"] == 0).sum()),
    }
    
    # F. Repayment history: installments per loan, loan overlap with train/test?
    inst_per_loan = df_repay["loan_id"].value_counts()["count"]
    repay_loans = set(df_repay["loan_id"].unique())
    prev_loans = set(df_prev["loan_id"].drop_nulls().unique())
    loans_in_prev = len(repay_loans.intersection(prev_loans))
    loans_not_in_prev = len(repay_loans - prev_loans)
    
    # Check if any loan_id or installment corresponds directly to train or test application_id
    # Note: train/test applications don't have loan_id in their table. But do previous_applications link them?
    # Let's check customers in repayment_history vs train/test customers:
    repay_custs = set(df_repay["customer_id"].unique())
    repay_cust_in_train = len(repay_custs.intersection(train_custs))
    repay_cust_in_test = len(repay_custs.intersection(test_custs))
    
    verifications = {
        "customer_profiles": {
            "total_rows": cust_total,
            "unique_customer_ids": cust_unique,
            "duplicate_customer_ids": cust_dups
        },
        "train_customers": {
            "total_rows": len(df_train),
            "unique_customers": len(train_custs),
            "max_apps_per_cust": max(train_cust_counts),
            "min_apps_per_cust": min(train_cust_counts),
            "dist_apps_per_cust": {str(d["app_count"]): d["len"] for d in df_train["customer_id"].value_counts().rename({"count": "app_count"}).group_by("app_count").len().to_dicts()}
        },
        "test_customers": {
            "total_rows": len(df_test),
            "unique_customers": len(test_custs),
            "max_apps_per_cust": max(test_cust_counts),
            "min_apps_per_cust": min(test_cust_counts),
            "dist_apps_per_cust": {str(d["app_count"]): d["len"] for d in df_test["customer_id"].value_counts().rename({"count": "app_count"}).group_by("app_count").len().to_dicts()}
        },
        "overlap": {
            "customer_overlap_train_test": cust_overlap,
            "application_overlap_train_test": app_overlap,
            "prev_apps_in_train": prev_in_train,
            "prev_apps_in_test": prev_in_test,
        },
        "bureau": {
            "unique_applications": bureau_apps,
            "min_obs_per_app": int(bureau_per_app.min()),
            "median_obs_per_app": float(bureau_per_app.median()),
            "max_obs_per_app": int(bureau_per_app.max()),
            "provider_counts": {d["bureau_provider"]: d["count"] for d in bureau_per_prov},
            "diff_available_minus_report_hours": diff_stats,
        },
        "repayment_history": {
            "unique_loans": len(repay_loans),
            "unique_customers": len(repay_custs),
            "min_inst_per_loan": int(inst_per_loan.min()),
            "median_inst_per_loan": float(inst_per_loan.median()),
            "max_inst_per_loan": int(inst_per_loan.max()),
            "repay_loans_in_prev_loans": loans_in_prev,
            "repay_loans_not_in_prev_loans": loans_not_in_prev,
            "repay_cust_in_train": repay_cust_in_train,
            "repay_cust_in_test": repay_cust_in_test,
        }
    }
    
    full_report = {
        "tables": {
            "train_applications": prof_train,
            "test_applications": prof_test,
            "customer_profiles": prof_cust,
            "previous_applications": prof_prev,
            "repayment_history": prof_repay,
            "account_behavior": prof_acct,
            "bureau_observations": prof_bureau,
            "ecommerce_transactions": prof_ecom,
            "sample_submission": prof_sub,
        },
        "verifications": verifications,
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/profiling_report.json", "w") as f:
        json.dump(full_report, f, indent=2)
    print("Profiling report saved to reports/profiling_report.json")

if __name__ == "__main__":
    run_profiling()
