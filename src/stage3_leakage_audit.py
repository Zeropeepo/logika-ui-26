"""
Stage 3: Comprehensive Point-in-Time Leakage Audit
"""
import os
import json
import polars as pl
import numpy as np

DATA_DIR = "data/raw"

def run_stage3():
    print("Running Stage 3: Leakage Audit...")
    
    # Load applications (train + test) with decision_datetime
    df_train = pl.read_csv(f"{DATA_DIR}/train_applications.csv", try_parse_dates=True)
    df_test = pl.read_csv(f"{DATA_DIR}/test_applications.csv", try_parse_dates=True)
    
    # Combined target applications
    apps_train = df_train.select([
        pl.col("application_id"),
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff_datetime"),
        pl.col("application_datetime").alias("app_cutoff_datetime")
    ]).with_columns(split=pl.lit("train"))
    
    apps_test = df_test.select([
        pl.col("application_id"),
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff_datetime"),
        pl.col("application_datetime").alias("app_cutoff_datetime")
    ]).with_columns(split=pl.lit("test"))
    apps = pl.concat([apps_train, apps_test])
    
    audit_results = {}
    
    # 1. Audit previous_applications
    print("Auditing previous_applications...")
    df_prev = pl.read_csv(f"{DATA_DIR}/previous_applications.csv", try_parse_dates=True)
    # Join on customer_id with target apps
    prev_joined = df_prev.join(apps.select(["customer_id", "cutoff_datetime", "app_cutoff_datetime"]), on="customer_id", how="inner")
    
    total_prev = len(prev_joined)
    # Violations: prev decision_datetime >= curr decision_datetime OR prev app_datetime >= curr app_datetime
    prev_viol_dec = prev_joined.filter(pl.col("decision_datetime") >= pl.col("cutoff_datetime"))
    prev_viol_app = prev_joined.filter(pl.col("application_datetime") >= pl.col("app_cutoff_datetime"))
    prev_viol_any = prev_joined.filter(
        (pl.col("decision_datetime") >= pl.col("cutoff_datetime")) |
        (pl.col("application_datetime") >= pl.col("app_cutoff_datetime"))
    )
    
    audit_results["previous_applications"] = {
        "total_candidate_rows": total_prev,
        "violating_decision_cutoff": len(prev_viol_dec),
        "violating_app_cutoff": len(prev_viol_app),
        "violating_any": len(prev_viol_any),
        "pct_violating": round(len(prev_viol_any) / total_prev * 100, 4) if total_prev > 0 else 0,
        "action": "Filter: strictly keep previous_applications where decision_datetime < cutoff_datetime"
    }
    
    # 2. Audit bureau_observations
    print("Auditing bureau_observations...")
    df_bureau = pl.read_csv(f"{DATA_DIR}/bureau_observations.csv", try_parse_dates=True)
    
    # Check A: Direct application_id match
    bureau_app_joined = df_bureau.join(apps.select(["application_id", "cutoff_datetime"]), on="application_id", how="inner")
    total_bureau_app = len(bureau_app_joined)
    
    bureau_app_viol_avail = bureau_app_joined.filter(pl.col("available_datetime") > pl.col("cutoff_datetime"))
    bureau_app_viol_rep = bureau_app_joined.filter(pl.col("report_datetime") > pl.col("cutoff_datetime"))
    # Report before cutoff, but available AFTER cutoff!
    bureau_app_rep_ok_avail_bad = bureau_app_joined.filter(
        (pl.col("report_datetime") <= pl.col("cutoff_datetime")) &
        (pl.col("available_datetime") > pl.col("cutoff_datetime"))
    )
    
    audit_results["bureau_observations_by_application"] = {
        "total_matched_rows": total_bureau_app,
        "violating_available_datetime": len(bureau_app_viol_avail),
        "pct_violating_available": round(len(bureau_app_viol_avail) / total_bureau_app * 100, 4),
        "violating_report_datetime": len(bureau_app_viol_rep),
        "rep_before_cutoff_but_avail_after": len(bureau_app_rep_ok_avail_bad),
        "pct_trap_leaks": round(len(bureau_app_rep_ok_avail_bad) / total_bureau_app * 100, 4),
        "action": "Filter: strictly require available_datetime <= decision_datetime. Reject records where available_datetime > decision_datetime even if report_datetime <= cutoff."
    }
    
    # 3. Audit account_behavior
    print("Auditing account_behavior...")
    df_acct = pl.read_csv(f"{DATA_DIR}/account_behavior.csv", try_parse_dates=True)
    acct_joined = df_acct.join(apps.select(["customer_id", "cutoff_datetime"]), on="customer_id", how="inner")
    total_acct = len(acct_joined)
    
    # Incomplete month rule:
    # month is first day of calendar month. The month completes on (month + 1 month).
    # If (month + 1 month) > cutoff_datetime (or equivalently month's month/year >= cutoff's month/year when decision is in that month),
    # the month was not complete at decision time!
    # In Polars: pl.col("month").dt.offset_by("1mo") > pl.col("cutoff_datetime")
    acct_viol = acct_joined.filter(pl.col("month").dt.offset_by("1mo") > pl.col("cutoff_datetime"))
    
    audit_results["account_behavior"] = {
        "total_candidate_rows": total_acct,
        "violating_incomplete_month": len(acct_viol),
        "pct_violating": round(len(acct_viol) / total_acct * 100, 4),
        "action": "Filter: strictly keep complete months only, i.e., month.dt.offset_by('1mo') <= cutoff_datetime"
    }
    
    # 4. Audit repayment_history
    print("Auditing repayment_history...")
    df_repay = pl.read_csv(f"{DATA_DIR}/repayment_history.csv", try_parse_dates=True)
    repay_joined = df_repay.join(apps.select(["customer_id", "cutoff_datetime"]), on="customer_id", how="inner")
    total_repay = len(repay_joined)
    
    repay_viol_due = repay_joined.filter(pl.col("due_date") >= pl.col("cutoff_datetime"))
    repay_viol_pay = repay_joined.filter(
        pl.col("last_payment_datetime").is_not_null() & 
        (pl.col("last_payment_datetime") >= pl.col("cutoff_datetime"))
    )
    
    # What about days_past_due / max_days_past_due?
    # Are there rows where due_date < cutoff, but last_payment_datetime >= cutoff?
    repay_due_before_pay_after = repay_joined.filter(
        (pl.col("due_date") < pl.col("cutoff_datetime")) &
        pl.col("last_payment_datetime").is_not_null() &
        (pl.col("last_payment_datetime") >= pl.col("cutoff_datetime"))
    )
    
    audit_results["repayment_history"] = {
        "total_candidate_rows": total_repay,
        "violating_due_date_cutoff": len(repay_viol_due),
        "pct_violating_due_date": round(len(repay_viol_due) / total_repay * 100, 4),
        "violating_payment_datetime_cutoff": len(repay_viol_pay),
        "pct_violating_payment_datetime": round(len(repay_viol_pay) / total_repay * 100, 4),
        "due_before_cutoff_but_paid_after": len(repay_due_before_pay_after),
        "action": "Filter: strictly keep installments where due_date < cutoff_datetime. Furthermore, if last_payment_datetime >= cutoff_datetime, payment was made in the future: mask payment or treat as unpaid at cutoff!"
    }
    
    # 5. Audit ecommerce_transactions
    print("Auditing ecommerce_transactions...")
    df_ecom = pl.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv", try_parse_dates=True)
    ecom_joined = df_ecom.join(apps.select(["customer_id", "cutoff_datetime"]), on="customer_id", how="inner")
    total_ecom = len(ecom_joined)
    
    ecom_viol_time = ecom_joined.filter(pl.col("transaction_datetime") >= pl.col("cutoff_datetime"))
    ecom_cancelled = ecom_joined.filter(pl.col("order_status") == "cancelled")
    
    audit_results["ecommerce_transactions"] = {
        "total_candidate_rows": total_ecom,
        "violating_transaction_cutoff": len(ecom_viol_time),
        "pct_violating_time": round(len(ecom_viol_time) / total_ecom * 100, 4),
        "cancelled_order_rows": len(ecom_cancelled),
        "pct_cancelled": round(len(ecom_cancelled) / total_ecom * 100, 4),
        "action": "Filter: strictly keep transaction_datetime < decision_datetime. For spending aggregates, filter out order_status == 'cancelled' and deduplicate by order_id before summing order_total."
    }
    
    # Save audit results
    with open("reports/leakage_audit_report.json", "w") as f:
        json.dump(audit_results, f, indent=2)
    print("Leakage audit report saved to reports/leakage_audit_report.json")

if __name__ == "__main__":
    run_stage3()
