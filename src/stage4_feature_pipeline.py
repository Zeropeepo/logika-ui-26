"""
Stage 4: Point-in-Time Feature Engineering Pipeline
Extracts features for train and test applications identically.
"""
import os
import json
import polars as pl
import numpy as np

DATA_DIR = "data/raw"

def extract_application_and_profile_features(apps: pl.DataFrame, df_cust: pl.DataFrame) -> pl.DataFrame:
    print("Extracting application & customer profile features...")
    df = apps.join(df_cust, on="customer_id", how="left")
    
    # Feature expressions
    df = df.with_columns(
        hours_to_decision = (pl.col("decision_datetime") - pl.col("application_datetime")).dt.total_hours(),
        installment_income_ratio = pl.col("proposed_monthly_installment") / pl.col("declared_monthly_income"),
        loan_income_ratio = pl.col("loan_amount") / pl.col("declared_monthly_income"),
        monthly_interest_rate = pl.col("annual_interest_rate") / 12.0,
        total_repayment_amount = pl.col("proposed_monthly_installment") * pl.col("term_months"),
        interest_amount = (pl.col("proposed_monthly_installment") * pl.col("term_months")) - pl.col("loan_amount"),
        interest_principal_ratio = ((pl.col("proposed_monthly_installment") * pl.col("term_months")) - pl.col("loan_amount")) / pl.col("loan_amount"),
        income_per_dependent = pl.col("declared_monthly_income") / (pl.col("dependents") + 1.0),
        tenure_age_ratio = pl.col("employment_tenure_months") / (pl.col("age") * 12.0),
    )
    return df

def extract_previous_application_features(apps: pl.DataFrame, df_prev: pl.DataFrame) -> pl.DataFrame:
    print("Extracting previous application features...")
    # Join with cutoff
    prev_j = df_prev.join(
        apps.select([pl.col("customer_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="customer_id",
        how="inner"
    )
    # Strictly filter before cutoff
    prev_valid = prev_j.filter(pl.col("decision_datetime") < pl.col("cutoff"))
    
    prev_valid = prev_valid.with_columns(
        days_since_decision = (pl.col("cutoff") - pl.col("decision_datetime")).dt.total_days(),
        days_since_app = (pl.col("cutoff") - pl.col("application_datetime")).dt.total_days(),
    )
    
    agg = prev_valid.group_by("customer_id").agg(
        prev_app_count = pl.len(),
        prev_approved_count = (pl.col("decision") == "approved").sum(),
        prev_rejected_count = (pl.col("decision") == "rejected").sum(),
        prev_cancelled_count = (pl.col("decision") == "cancelled").sum(),
        prev_originated_loans_count = pl.col("loan_id").is_not_null().sum(),
        prev_total_approved_amount = pl.col("approved_amount").sum(),
        prev_mean_approved_amount = pl.col("approved_amount").mean(),
        prev_total_requested_amount = pl.col("requested_amount").sum(),
        prev_recency_days = pl.col("days_since_decision").min(),
        prev_app_recency_days = pl.col("days_since_app").min(),
    ).with_columns(
        prev_approval_rate = pl.col("prev_approved_count") / pl.col("prev_app_count"),
        prev_rejection_rate = pl.col("prev_rejected_count") / pl.col("prev_app_count"),
        prev_ratio_approved_requested = pl.col("prev_total_approved_amount") / (pl.col("prev_total_requested_amount") + 1e-5),
    )
    return agg

def extract_repayment_history_features(apps: pl.DataFrame, df_repay: pl.DataFrame) -> pl.DataFrame:
    print("Extracting repayment history features...")
    repay_j = df_repay.join(
        apps.select([pl.col("customer_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="customer_id",
        how="inner"
    )
    repay_valid = repay_j.filter(pl.col("due_date") < pl.col("cutoff"))
    
    # Calculate days since due
    repay_valid = repay_valid.with_columns(
        days_since_due = (pl.col("cutoff") - pl.col("due_date")).dt.total_days(),
        is_ontime = (pl.col("days_past_due") == 0).cast(pl.Int32),
        is_underpaid = (pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Int32),
        dpd_ge_30 = (pl.col("days_past_due") >= 30).cast(pl.Int32),
        dpd_ge_60 = (pl.col("days_past_due") >= 60).cast(pl.Int32),
        dpd_ge_90 = (pl.col("days_past_due") >= 90).cast(pl.Int32),
    )
    
    agg = repay_valid.group_by("customer_id").agg(
        repay_installment_count = pl.len(),
        repay_loans_count = pl.col("loan_id").n_unique(),
        repay_total_amount_due = pl.col("amount_due").sum(),
        repay_total_amount_paid = pl.col("amount_paid").sum(),
        repay_ontime_ratio = pl.col("is_ontime").mean(),
        repay_underpaid_ratio = pl.col("is_underpaid").mean(),
        repay_max_dpd = pl.col("days_past_due").max(),
        repay_mean_dpd = pl.col("days_past_due").mean(),
        repay_max_max_dpd = pl.col("max_days_past_due").max(),
        repay_dpd_ge_30_count = pl.col("dpd_ge_30").sum(),
        repay_dpd_ge_60_count = pl.col("dpd_ge_60").sum(),
        repay_dpd_ge_90_count = pl.col("dpd_ge_90").sum(),
        repay_min_days_since_due = pl.col("days_since_due").min(),
        # Active loans: loans with due date in last 60 days
        repay_active_loans_count = pl.col("loan_id").filter(pl.col("days_since_due") <= 60).n_unique(),
    ).with_columns(
        repay_paid_over_due_ratio = pl.col("repay_total_amount_paid") / (pl.col("repay_total_amount_due") + 1e-5)
    )
    
    # Recent 6 installments trend
    repay_sorted = repay_valid.sort(["customer_id", "due_date"], descending=[False, True])
    # Top 6 per customer
    repay_recent6 = repay_sorted.group_by("customer_id").head(6)
    recent6_agg = repay_recent6.group_by("customer_id").agg(
        repay_recent6_max_dpd = pl.col("days_past_due").max(),
        repay_recent6_mean_dpd = pl.col("days_past_due").mean(),
        repay_recent6_underpaid_ratio = pl.col("is_underpaid").mean(),
        repay_recent6_ontime_ratio = pl.col("is_ontime").mean(),
    )
    
    agg = agg.join(recent6_agg, on="customer_id", how="left")
    return agg

def extract_account_behavior_features(apps: pl.DataFrame, df_acct: pl.DataFrame) -> pl.DataFrame:
    print("Extracting account behavior features...")
    acct_j = df_acct.join(
        apps.select([pl.col("customer_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="customer_id",
        how="inner"
    )
    # Strictly complete months
    acct_valid = acct_j.filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime("us")) <= pl.col("cutoff"))
    
    # Months difference relative to cutoff
    # approximate months as days / 30.4375 or use date subtraction
    acct_valid = acct_valid.with_columns(
        days_since_month_start = (pl.col("cutoff") - pl.col("month").cast(pl.Datetime("us"))).dt.total_days()
    )
    
    # 3m (approx <= 100 days), 6m (approx <= 190 days), 12m (approx <= 375 days)
    agg_all = acct_valid.group_by("customer_id").agg(
        acct_observed_months = pl.len(),
        acct_mean_balance_all = pl.col("average_balance").mean(),
        acct_std_balance_all = pl.col("average_balance").std(),
        acct_total_inflows_all = pl.col("inflows").sum(),
        acct_total_outflows_all = pl.col("outflows").sum(),
        acct_mean_inflows_all = pl.col("inflows").mean(),
        acct_mean_outflows_all = pl.col("outflows").mean(),
        acct_mean_income_credits_all = pl.col("income_like_credits").mean(),
        acct_total_low_balance_days_all = pl.col("low_balance_days").sum(),
        acct_mean_low_balance_days_all = pl.col("low_balance_days").mean(),
    ).with_columns(
        acct_outflow_inflow_ratio = pl.col("acct_total_outflows_all") / (pl.col("acct_total_inflows_all") + 1e-5),
        acct_balance_volatility = pl.col("acct_std_balance_all") / (pl.col("acct_mean_balance_all") + 1e-5),
    )
    
    # 3-month window
    acct_3m = acct_valid.filter(pl.col("days_since_month_start") <= 100).group_by("customer_id").agg(
        acct_mean_balance_3m = pl.col("average_balance").mean(),
        acct_mean_inflows_3m = pl.col("inflows").mean(),
        acct_mean_outflows_3m = pl.col("outflows").mean(),
        acct_mean_income_credits_3m = pl.col("income_like_credits").mean(),
        acct_low_balance_days_3m = pl.col("low_balance_days").sum(),
    )
    
    # 6-month window
    acct_6m = acct_valid.filter(pl.col("days_since_month_start") <= 190).group_by("customer_id").agg(
        acct_mean_balance_6m = pl.col("average_balance").mean(),
        acct_mean_inflows_6m = pl.col("inflows").mean(),
        acct_mean_outflows_6m = pl.col("outflows").mean(),
        acct_mean_income_credits_6m = pl.col("income_like_credits").mean(),
        acct_low_balance_days_6m = pl.col("low_balance_days").sum(),
    )
    
    # 12-month window
    acct_12m = acct_valid.filter(pl.col("days_since_month_start") <= 375).group_by("customer_id").agg(
        acct_mean_balance_12m = pl.col("average_balance").mean(),
        acct_mean_inflows_12m = pl.col("inflows").mean(),
        acct_mean_outflows_12m = pl.col("outflows").mean(),
        acct_mean_income_credits_12m = pl.col("income_like_credits").mean(),
        acct_low_balance_days_12m = pl.col("low_balance_days").sum(),
    )
    
    combined = agg_all.join(acct_3m, on="customer_id", how="left")
    combined = combined.join(acct_6m, on="customer_id", how="left")
    combined = combined.join(acct_12m, on="customer_id", how="left")
    
    combined = combined.with_columns(
        acct_inflow_trend_3m_vs_6m = pl.col("acct_mean_inflows_3m") / (pl.col("acct_mean_inflows_6m") + 1e-5),
        acct_outflow_trend_3m_vs_6m = pl.col("acct_mean_outflows_3m") / (pl.col("acct_mean_outflows_6m") + 1e-5),
        acct_balance_trend_3m_vs_6m = pl.col("acct_mean_balance_3m") / (pl.col("acct_mean_balance_6m") + 1e-5),
    )
    return combined

def extract_bureau_features(apps: pl.DataFrame, df_bureau: pl.DataFrame) -> pl.DataFrame:
    print("Extracting bureau observation features...")
    # Joined on application_id!
    bureau_j = df_bureau.join(
        apps.select([pl.col("application_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="application_id",
        how="inner"
    )
    bureau_valid = bureau_j.filter(pl.col("available_datetime") <= pl.col("cutoff"))
    
    bureau_valid = bureau_valid.with_columns(
        days_since_available = (pl.col("cutoff") - pl.col("available_datetime")).dt.total_days(),
        is_unscored = pl.col("score").is_null().cast(pl.Int32),
    )
    
    agg = bureau_valid.group_by("application_id").agg(
        bureau_provider_count = pl.len(),
        bureau_score_min = pl.col("score").min(),
        bureau_score_mean = pl.col("score").mean(),
        bureau_score_max = pl.col("score").max(),
        bureau_score_std = pl.col("score").std(),
        bureau_has_unscored = pl.col("is_unscored").max(),
        bureau_all_unscored = pl.col("is_unscored").min(),
        bureau_total_active_accounts = pl.col("active_accounts").sum(),
        bureau_max_active_accounts = pl.col("active_accounts").max(),
        bureau_total_outstanding_balance = pl.col("outstanding_balance").sum(),
        bureau_total_revolving_limit = pl.col("revolving_credit_limit").sum(),
        bureau_total_revolving_balance = pl.col("revolving_outstanding_balance").sum(),
        bureau_total_inquiries_6m = pl.col("inquiries_last_6m").sum(),
        bureau_max_inquiries_6m = pl.col("inquiries_last_6m").max(),
        bureau_total_delinquent_12m = pl.col("delinquent_accounts_last_12m").sum(),
        bureau_max_delinquent_12m = pl.col("delinquent_accounts_last_12m").max(),
        bureau_days_since_available = pl.col("days_since_available").min(),
    ).with_columns(
        bureau_recalculated_utilization = pl.when(pl.col("bureau_total_revolving_limit") > 0)
            .then(pl.col("bureau_total_revolving_balance") / pl.col("bureau_total_revolving_limit"))
            .otherwise(None),
    )
    
    # Individual provider scores via pivot/aggregation
    prov_pivot = bureau_valid.group_by("application_id").agg(
        bureau_score_Bureau_A = pl.col("score").filter(pl.col("bureau_provider") == "Bureau_A").first(),
        bureau_score_Bureau_B = pl.col("score").filter(pl.col("bureau_provider") == "Bureau_B").first(),
        bureau_score_Bureau_C = pl.col("score").filter(pl.col("bureau_provider") == "Bureau_C").first(),
    )
    
    agg = agg.join(prov_pivot, on="application_id", how="left")
    return agg

def extract_ecommerce_features(apps: pl.DataFrame, df_ecom: pl.DataFrame) -> pl.DataFrame:
    print("Extracting ecommerce transaction features...")
    ecom_j = df_ecom.join(
        apps.select([pl.col("customer_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="customer_id",
        how="inner"
    )
    ecom_valid = ecom_j.filter(pl.col("transaction_datetime") < pl.col("cutoff"))
    
    ecom_valid = ecom_valid.with_columns(
        days_since_order = (pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_days(),
    )
    
    # 1. Order level (deduplicated by order_id)
    orders = ecom_valid.group_by("order_id").agg(
        customer_id = pl.col("customer_id").first(),
        transaction_datetime = pl.col("transaction_datetime").first(),
        days_since_order = pl.col("days_since_order").first(),
        order_total = pl.col("order_total").first(),
        payment_method = pl.col("payment_method").first(),
        order_status = pl.col("order_status").first(),
    )
    
    # Overall order aggregations
    orders_agg = orders.group_by("customer_id").agg(
        ecom_total_orders = pl.len(),
        ecom_cancelled_orders = (pl.col("order_status") == "cancelled").sum(),
        ecom_completed_orders = (pl.col("order_status") == "completed").sum(),
    ).with_columns(
        ecom_cancellation_rate = pl.col("ecom_cancelled_orders") / pl.col("ecom_total_orders"),
    )
    
    # Completed orders only for spending & methods
    completed_orders = orders.filter(pl.col("order_status") == "completed")
    
    comp_agg = completed_orders.group_by("customer_id").agg(
        ecom_total_spending = pl.col("order_total").sum(),
        ecom_mean_order_value = pl.col("order_total").mean(),
        ecom_max_order_value = pl.col("order_total").max(),
        ecom_recency_days = pl.col("days_since_order").min(),
        # Windows
        ecom_spending_30d = pl.col("order_total").filter(pl.col("days_since_order") <= 30).sum(),
        ecom_spending_90d = pl.col("order_total").filter(pl.col("days_since_order") <= 90).sum(),
        ecom_spending_180d = pl.col("order_total").filter(pl.col("days_since_order") <= 180).sum(),
        ecom_orders_30d = (pl.col("days_since_order") <= 30).sum(),
        ecom_orders_90d = (pl.col("days_since_order") <= 90).sum(),
        ecom_orders_180d = (pl.col("days_since_order") <= 180).sum(),
        # Payment method counts
        ecom_cod_orders = (pl.col("payment_method") == "COD").sum(),
        ecom_ewallet_orders = (pl.col("payment_method") == "e_wallet").sum(),
        ecom_transfer_orders = (pl.col("payment_method") == "bank_transfer").sum(),
        ecom_credit_orders = (pl.col("payment_method") == "credit_card").sum(),
    ).with_columns(
        ecom_cod_share = pl.col("ecom_cod_orders") / (pl.col("ecom_total_spending").is_not_null().cast(pl.Float64) * 0 + pl.col("ecom_orders_180d") + 1e-5), # will adjust below
    )
    
    # Item level aggregations for completed orders: category shares
    completed_items = ecom_valid.filter(pl.col("order_status") == "completed")
    cat_agg = completed_items.group_by("customer_id").agg(
        ecom_item_total_items = pl.len(),
        ecom_items_groceries = (pl.col("product_category") == "groceries").sum(),
        ecom_items_electronics = (pl.col("product_category") == "electronics").sum(),
        ecom_items_fashion = (pl.col("product_category") == "fashion").sum(),
    ).with_columns(
        ecom_groceries_share = pl.col("ecom_items_groceries") / pl.col("ecom_item_total_items"),
        ecom_electronics_share = pl.col("ecom_items_electronics") / pl.col("ecom_item_total_items"),
        ecom_fashion_share = pl.col("ecom_items_fashion") / pl.col("ecom_item_total_items"),
    )
    
    # Merge order aggregations
    ecom_features = orders_agg.join(comp_agg, on="customer_id", how="left")
    # Fix COD share using completed orders count
    ecom_features = ecom_features.with_columns(
        ecom_cod_share = pl.col("ecom_cod_orders") / pl.col("ecom_completed_orders")
    )
    ecom_features = ecom_features.join(cat_agg, on="customer_id", how="left")
    return ecom_features

def build_features(apps: pl.DataFrame,
                   df_cust: pl.DataFrame,
                   df_prev: pl.DataFrame,
                   df_repay: pl.DataFrame,
                   df_acct: pl.DataFrame,
                   df_bureau: pl.DataFrame,
                   df_ecom: pl.DataFrame) -> pl.DataFrame:
    
    # 1. Base app + customer profile
    df = extract_application_and_profile_features(apps, df_cust)
    
    # 2. Previous apps
    feat_prev = extract_previous_application_features(apps, df_prev)
    df = df.join(feat_prev, on="customer_id", how="left")
    
    # 3. Repayment history
    feat_repay = extract_repayment_history_features(apps, df_repay)
    df = df.join(feat_repay, on="customer_id", how="left")
    
    # 4. Account behavior
    feat_acct = extract_account_behavior_features(apps, df_acct)
    df = df.join(feat_acct, on="customer_id", how="left")
    
    # 5. Bureau observations
    feat_bureau = extract_bureau_features(apps, df_bureau)
    df = df.join(feat_bureau, on="application_id", how="left")
    
    # 6. Ecommerce transactions
    feat_ecom = extract_ecommerce_features(apps, df_ecom)
    df = df.join(feat_ecom, on="customer_id", how="left")
    
    # Add cross-source interaction features:
    df = df.with_columns(
        income_credits_to_declared_ratio = pl.col("acct_mean_income_credits_all") / (pl.col("declared_monthly_income") + 1e-5),
        total_debt_to_income = (pl.col("bureau_total_outstanding_balance") + pl.col("loan_amount")) / (pl.col("declared_monthly_income") * 12.0 + 1e-5),
    )
    
    return df

def run_pipeline():
    print("Loading all raw tables for feature pipeline...")
    df_train = pl.read_csv(f"{DATA_DIR}/train_applications.csv", try_parse_dates=True)
    df_test = pl.read_csv(f"{DATA_DIR}/test_applications.csv", try_parse_dates=True)
    df_cust = pl.read_csv(f"{DATA_DIR}/customer_profiles.csv")
    df_prev = pl.read_csv(f"{DATA_DIR}/previous_applications.csv", try_parse_dates=True)
    df_repay = pl.read_csv(f"{DATA_DIR}/repayment_history.csv", try_parse_dates=True)
    df_acct = pl.read_csv(f"{DATA_DIR}/account_behavior.csv", try_parse_dates=True)
    df_bureau = pl.read_csv(f"{DATA_DIR}/bureau_observations.csv", try_parse_dates=True)
    df_ecom = pl.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv", try_parse_dates=True)
    
    print("--- Building Train Features ---")
    train_feat = build_features(df_train, df_cust, df_prev, df_repay, df_acct, df_bureau, df_ecom)
    
    print("--- Building Test Features ---")
    test_feat = build_features(df_test, df_cust, df_prev, df_repay, df_acct, df_bureau, df_ecom)
    
    # Validations
    print("Validating feature tables...")
    assert len(train_feat) == len(df_train), f"Train row mismatch: {len(train_feat)} vs {len(df_train)}"
    assert len(test_feat) == len(df_test), f"Test row mismatch: {len(test_feat)} vs {len(df_test)}"
    assert train_feat["application_id"].n_unique() == len(train_feat), "Train fan-out detected!"
    assert test_feat["application_id"].n_unique() == len(test_feat), "Test fan-out detected!"
    
    # Check columns
    train_cols = set(train_feat.columns)
    test_cols = set(test_feat.columns)
    # train has target, test doesn't
    diff_cols = train_cols - test_cols
    assert diff_cols == {"target"}, f"Unexpected column difference between train and test: {diff_cols}"
    
    # Save to parquet
    os.makedirs("data/processed", exist_ok=True)
    train_path = "data/processed/train_features.parquet"
    test_path = "data/processed/test_features.parquet"
    
    train_feat.write_parquet(train_path)
    test_feat.write_parquet(test_path)
    print(f"Train features saved to {train_path} ({len(train_feat)} rows, {len(train_feat.columns)} cols)")
    print(f"Test features saved to {test_path} ({len(test_feat)} rows, {len(test_feat.columns)} cols)")

if __name__ == "__main__":
    run_pipeline()
