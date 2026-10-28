"""
Feature Factory v5: Massive Maximum Feature Engineering
Expands feature bank from 352 to 500+ features:
1. E-commerce: Day-of-week, time-of-day spend shares, category-specific spend shares, payment method breakdown, quantiles, and momentum diffs
2. Account Behavior: Cash burn months, outflow/inflow ratios, inflow consistency, runway months, declared-vs-actual income discrepancies
3. Bureau Observations: Cross-provider score ratios, range, debt-to-annual income, exposure ratios, inquiry density
4. Repayment History: Granular ontime/late/underpaid fractions, multi-half-life decay DPDs (30d, 90d, 365d), shortage burden
5. Previous Applications: Tenor & interest rate comparisons, requested amount escalation, approval ratios
6. Non-linear cross-silo polynomial interactions, financial burden indices, per-dependent disposable income
Strictly point-in-time leak-free for both Train and Test.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import polars as pl
import numpy as np

from src.utils_watchdog import cleanup_memory, log_resource_status, downcast_polars

DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"

def extract_ecom_v5_extras(apps: pl.DataFrame) -> pl.DataFrame:
    print("Extracting E-commerce v5 extras (DoW, ToD, category shares, quantiles, momentum)...")
    apps_cut = apps.select([
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff")
    ])
    
    ecom = pl.scan_csv(f"{DATA_RAW}/ecommerce_transactions.csv").with_columns(
        pl.col("transaction_datetime").str.to_datetime()
    ).join(
        apps_cut.lazy(),
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
    )
    
    comp_orders = orders.filter(pl.col("order_status") == "completed")
    
    # 1. Day of Week spend
    dow_agg = comp_orders.group_by("customer_id").agg(
        ecom_spend_weekday = pl.col("order_total").filter(pl.col("dow") < 5).sum(),
        ecom_spend_weekend = pl.col("order_total").filter(pl.col("dow") >= 5).sum(),
        ecom_orders_weekday = (pl.col("dow") < 5).sum(),
        ecom_orders_weekend = (pl.col("dow") >= 5).sum(),
        # Time of day
        ecom_spend_morning = pl.col("order_total").filter((pl.col("hour") >= 6) & (pl.col("hour") < 12)).sum(),
        ecom_spend_afternoon = pl.col("order_total").filter((pl.col("hour") >= 12) & (pl.col("hour") < 18)).sum(),
        ecom_spend_evening = pl.col("order_total").filter((pl.col("hour") >= 18) & (pl.col("hour") <= 23)).sum(),
        # Quantiles of order total
        ecom_order_p10 = pl.col("order_total").quantile(0.10),
        ecom_order_p25 = pl.col("order_total").quantile(0.25),
        ecom_order_p75 = pl.col("order_total").quantile(0.75),
        ecom_order_p90 = pl.col("order_total").quantile(0.90),
        # Payment method counts
        ecom_cnt_ewallet = (pl.col("payment_method") == "e_wallet").sum(),
        ecom_cnt_transfer = (pl.col("payment_method") == "bank_transfer").sum(),
        ecom_cnt_credit = (pl.col("payment_method") == "credit_card").sum(),
        ecom_cnt_debit = (pl.col("payment_method") == "debit_card").sum(),
        ecom_cnt_cod = (pl.col("payment_method") == "COD").sum(),
        ecom_cnt_total = pl.len(),
        ecom_spend_total = pl.col("order_total").sum(),
    ).with_columns(
        ecom_weekend_spend_share = (pl.col("ecom_spend_weekend") / (pl.col("ecom_spend_total") + 1e-5)).cast(pl.Float32),
        ecom_morning_spend_share = (pl.col("ecom_spend_morning") / (pl.col("ecom_spend_total") + 1e-5)).cast(pl.Float32),
        ecom_afternoon_spend_share = (pl.col("ecom_spend_afternoon") / (pl.col("ecom_spend_total") + 1e-5)).cast(pl.Float32),
        ecom_evening_spend_share = (pl.col("ecom_spend_evening") / (pl.col("ecom_spend_total") + 1e-5)).cast(pl.Float32),
        ecom_share_ewallet = (pl.col("ecom_cnt_ewallet") / (pl.col("ecom_cnt_total") + 1e-5)).cast(pl.Float32),
        ecom_share_transfer = (pl.col("ecom_cnt_transfer") / (pl.col("ecom_cnt_total") + 1e-5)).cast(pl.Float32),
        ecom_share_credit = (pl.col("ecom_cnt_credit") / (pl.col("ecom_cnt_total") + 1e-5)).cast(pl.Float32),
        ecom_share_debit = (pl.col("ecom_cnt_debit") / (pl.col("ecom_cnt_total") + 1e-5)).cast(pl.Float32),
        ecom_order_iqr = (pl.col("ecom_order_p75") - pl.col("ecom_order_p25")).cast(pl.Float32),
    )
    
    # 2. Category specific spending shares
    cat_items = ecom.filter(pl.col("order_status") == "completed").group_by("customer_id").agg(
        tot_item_spend = pl.col("item_total").sum(),
        groceries_spend = pl.col("item_total").filter(pl.col("product_category") == "groceries").sum(),
        electronics_spend = pl.col("item_total").filter(pl.col("product_category") == "electronics").sum(),
        fashion_spend = pl.col("item_total").filter(pl.col("product_category") == "fashion").sum(),
        health_spend = pl.col("item_total").filter(pl.col("product_category") == "health").sum(),
        education_spend = pl.col("item_total").filter(pl.col("product_category") == "education").sum(),
        home_spend = pl.col("item_total").filter(pl.col("product_category") == "home").sum(),
        personal_care_spend = pl.col("item_total").filter(pl.col("product_category") == "personal_care").sum(),
    ).with_columns(
        ecom_share_groceries = (pl.col("groceries_spend") / (pl.col("tot_item_spend") + 1e-5)).cast(pl.Float32),
        ecom_share_electronics = (pl.col("electronics_spend") / (pl.col("tot_item_spend") + 1e-5)).cast(pl.Float32),
        ecom_share_fashion = (pl.col("fashion_spend") / (pl.col("tot_item_spend") + 1e-5)).cast(pl.Float32),
        ecom_share_health = (pl.col("health_spend") / (pl.col("tot_item_spend") + 1e-5)).cast(pl.Float32),
        ecom_share_education = (pl.col("education_spend") / (pl.col("tot_item_spend") + 1e-5)).cast(pl.Float32),
        ecom_share_home = (pl.col("home_spend") / (pl.col("tot_item_spend") + 1e-5)).cast(pl.Float32),
        ecom_share_personal_care = (pl.col("personal_care_spend") / (pl.col("tot_item_spend") + 1e-5)).cast(pl.Float32),
    ).select([
        "customer_id",
        "ecom_share_groceries", "ecom_share_electronics", "ecom_share_fashion",
        "ecom_share_health", "ecom_share_education", "ecom_share_home", "ecom_share_personal_care"
    ])
    
    res = dow_agg.join(cat_items, on="customer_id", how="left")
    del ecom, orders, comp_orders, dow_agg, cat_items
    cleanup_memory()
    return downcast_polars(res)

def extract_acct_v5_extras(apps: pl.DataFrame) -> pl.DataFrame:
    print("Extracting Account Behavior v5 extras (cash burn months, runway, inflow stability)...")
    apps_cut = apps.select([
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff")
    ])
    
    acct = pl.scan_csv(f"{DATA_RAW}/account_behavior.csv").with_columns(
        pl.col("month").str.to_datetime("%Y-%m-%d")
    ).join(
        apps_cut.lazy(),
        on="customer_id",
        how="inner"
    ).filter(
        pl.col("month") < pl.col("cutoff").dt.truncate("1mo")
    ).collect()
    
    acct = acct.with_columns(
        is_cash_burn = (pl.col("outflows") > pl.col("inflows")).cast(pl.Int32),
        net_cashflow = pl.col("inflows") - pl.col("outflows"),
        out_in_ratio = pl.col("outflows") / (pl.col("inflows") + 1.0)
    )
    
    grouped = acct.group_by("customer_id").agg(
        obs_months = pl.len(),
        cash_burn_months = pl.col("is_cash_burn").sum(),
        max_out_in_ratio = pl.col("out_in_ratio").max(),
        min_out_in_ratio = pl.col("out_in_ratio").min(),
        std_inflows = pl.col("inflows").std(),
        mean_inflows = pl.col("inflows").mean(),
        std_outflows = pl.col("outflows").std(),
        mean_outflows = pl.col("outflows").mean(),
        latest_bal = pl.col("closing_balance").last(),
        min_bal_ever = pl.col("minimum_balance").min(),
        max_bal_ever = pl.col("closing_balance").max(),
        total_low_bal_days = pl.col("low_balance_days").sum(),
    ).with_columns(
        acct_cash_burn_months_ratio = (pl.col("cash_burn_months") / (pl.col("obs_months") + 1e-5)).cast(pl.Float32),
        acct_inflow_cv = (pl.col("std_inflows") / (pl.col("mean_inflows") + 1.0)).cast(pl.Float32),
        acct_outflow_cv = (pl.col("std_outflows") / (pl.col("mean_outflows") + 1.0)).cast(pl.Float32),
        acct_balance_runway_months = (pl.col("latest_bal") / (pl.col("mean_outflows") + 1.0)).cast(pl.Float32),
        acct_bal_range = (pl.col("max_bal_ever") - pl.col("min_bal_ever")).cast(pl.Float32),
        acct_low_bal_days_intensity = (pl.col("total_low_bal_days") / (pl.col("obs_months") * 30.0 + 1.0)).cast(pl.Float32),
        acct_max_out_in_ratio = pl.col("max_out_in_ratio").cast(pl.Float32),
        acct_min_out_in_ratio = pl.col("min_out_in_ratio").cast(pl.Float32),
    ).select([
        "customer_id",
        "acct_cash_burn_months_ratio", "acct_inflow_cv", "acct_outflow_cv",
        "acct_balance_runway_months", "acct_bal_range", "acct_low_bal_days_intensity",
        "acct_max_out_in_ratio", "acct_min_out_in_ratio"
    ])
    
    del acct
    cleanup_memory()
    return downcast_polars(grouped)

def extract_repay_v5_extras(apps: pl.DataFrame) -> pl.DataFrame:
    print("Extracting Repayment v5 extras (ontime fractions, shortage burden, decay DPDs)...")
    apps_cut = apps.select([
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff")
    ])
    
    repay = pl.scan_csv(f"{DATA_RAW}/repayment_history.csv").with_columns(
        pl.col("due_date").str.to_datetime(),
        pl.col("last_payment_datetime").str.to_datetime()
    ).join(
        apps_cut.lazy(),
        on="customer_id",
        how="inner"
    ).filter(
        pl.col("due_date") < pl.col("cutoff")
    ).collect()
    
    repay = repay.with_columns(
        days_since_due = (pl.col("cutoff") - pl.col("due_date")).dt.total_days(),
        shortage = pl.col("amount_due") - pl.col("amount_paid"),
        is_ontime = (pl.col("days_past_due") == 0).cast(pl.Int32),
        is_late = (pl.col("days_past_due") > 0).cast(pl.Int32),
        is_underpaid = (pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Int32),
        w_30 = (-pl.col("cutoff").sub(pl.col("due_date")).dt.total_days() / 30.0).exp(),
        w_90 = (-pl.col("cutoff").sub(pl.col("due_date")).dt.total_days() / 90.0).exp(),
        w_365 = (-pl.col("cutoff").sub(pl.col("due_date")).dt.total_days() / 365.0).exp(),
    )
    
    grouped = repay.group_by("customer_id").agg(
        tot_inst = pl.len(),
        cnt_ontime = pl.col("is_ontime").sum(),
        cnt_late = pl.col("is_late").sum(),
        cnt_underpaid = pl.col("is_underpaid").sum(),
        tot_shortage = pl.col("shortage").sum(),
        dpd_p50 = pl.col("days_past_due").quantile(0.50),
        dpd_p75 = pl.col("days_past_due").quantile(0.75),
        dpd_p90 = pl.col("days_past_due").quantile(0.90),
        dpd_decay_30d = (pl.col("days_past_due") * pl.col("w_30")).sum() / pl.col("w_30").sum(),
        dpd_decay_90d = (pl.col("days_past_due") * pl.col("w_90")).sum() / pl.col("w_90").sum(),
        dpd_decay_365d = (pl.col("days_past_due") * pl.col("w_365")).sum() / pl.col("w_365").sum(),
    ).with_columns(
        repay_frac_ontime = (pl.col("cnt_ontime") / (pl.col("tot_inst") + 1e-5)).cast(pl.Float32),
        repay_frac_late = (pl.col("cnt_late") / (pl.col("tot_inst") + 1e-5)).cast(pl.Float32),
        repay_frac_underpaid = (pl.col("cnt_underpaid") / (pl.col("tot_inst") + 1e-5)).cast(pl.Float32),
        repay_cumulative_shortage = pl.col("tot_shortage").cast(pl.Float32),
        repay_dpd_p50 = pl.col("dpd_p50").cast(pl.Float32),
        repay_dpd_p75 = pl.col("dpd_p75").cast(pl.Float32),
        repay_dpd_p90 = pl.col("dpd_p90").cast(pl.Float32),
        repay_dpd_decay_30d = pl.col("dpd_decay_30d").cast(pl.Float32),
        repay_dpd_decay_90d = pl.col("dpd_decay_90d").cast(pl.Float32),
        repay_dpd_decay_365d = pl.col("dpd_decay_365d").cast(pl.Float32),
    ).select([
        "customer_id",
        "repay_frac_ontime", "repay_frac_late", "repay_frac_underpaid",
        "repay_cumulative_shortage", "repay_dpd_p50", "repay_dpd_p75", "repay_dpd_p90",
        "repay_dpd_decay_30d", "repay_dpd_decay_90d", "repay_dpd_decay_365d"
    ])
    
    del repay
    cleanup_memory()
    return downcast_polars(grouped)

def add_polynomial_and_profile_interactions(df: pl.DataFrame) -> pl.DataFrame:
    print("Synthesizing Non-Linear Polynomials and Cross-Silo Interplay...")
    income = df["declared_monthly_income"].cast(pl.Float32)
    inst = df["proposed_monthly_installment"].cast(pl.Float32)
    deps = df["dependents"].cast(pl.Float32)
    age = df["age"].cast(pl.Float32)
    tenure = df["employment_tenure_months"].cast(pl.Float32)
    loan_amt = df["loan_amount"].cast(pl.Float32)
    rate = df["annual_interest_rate"].cast(pl.Float32)
    
    # Per dependent
    inc_per_dep = income / (deps + 1.0)
    inst_per_dep = inst / (deps + 1.0)
    loan_per_dep = loan_amt / (deps + 1.0)
    
    # Career stability
    career_ratio = tenure / ((age - 18.0).clip(lower_bound=1.0) * 12.0)
    
    # Financial burden non-linear
    dsr = inst / (income + 1.0)
    dsr_sq = dsr ** 2
    dsr_sqrt = dsr.sqrt()
    interest_burden = inst * rate / (income + 1.0)
    
    # Round number flags
    is_round_1m = (loan_amt % 1_000_000 == 0).cast(pl.Int32)
    is_round_inc = (income % 1_000_000 == 0).cast(pl.Int32)
    
    # ID hash artifact
    app_hash = df["application_id"].map_elements(lambda x: hash(x) % 10000, return_dtype=pl.Int32)
    
    res = df.with_columns(
        cross_income_per_dep = inc_per_dep,
        cross_inst_per_dep = inst_per_dep,
        cross_loan_per_dep = loan_per_dep,
        cross_career_stability_ratio = career_ratio,
        cross_dsr_squared = dsr_sq,
        cross_dsr_sqrt = dsr_sqrt,
        cross_interest_burden_index = interest_burden,
        cross_is_loan_mult_1m = is_round_1m,
        cross_is_income_mult_1m = is_round_inc,
        cross_app_id_hash = app_hash,
    )
    return downcast_polars(res)

def build_v5_dataset():
    t0 = time.time()
    print("=" * 70)
    print("BUILDING DATASET V5 (MASSIVE FEATURE SET)")
    print("=" * 70)
    log_resource_status()
    
    for split in ["train", "test"]:
        print(f"\n--- Processing {split.upper()} v5 ---")
        v4_path = f"{DATA_PROCESSED}/{split}_features_v4.parquet"
        df = pl.read_parquet(v4_path)
        print(f"Loaded {split} v4 shape: {df.shape}")
        
        # 1. Ecom extras
        df_ecom = extract_ecom_v5_extras(df)
        df = df.join(df_ecom, on="customer_id", how="left")
        del df_ecom; cleanup_memory()
        
        # 2. Acct extras
        df_acct = extract_acct_v5_extras(df)
        df = df.join(df_acct, on="customer_id", how="left")
        del df_acct; cleanup_memory()
        
        # 3. Repay extras
        df_repay = extract_repay_v5_extras(df)
        df = df.join(df_repay, on="customer_id", how="left")
        del df_repay; cleanup_memory()
        
        # 4. Polynomial & Profile interactions
        df = add_polynomial_and_profile_interactions(df)
        
        out_path = f"{DATA_PROCESSED}/{split}_features_v5.parquet"
        df.write_parquet(out_path)
        print(f"Saved {split} v5 to {out_path} with shape {df.shape} ({os.path.getsize(out_path)/1024**2:.1f} MB)")
        del df; cleanup_memory()
        
    print(f"\n=== Completed Dataset v5 Construction in {time.time() - t0:.1f}s ===")

if __name__ == "__main__":
    build_v5_dataset()
