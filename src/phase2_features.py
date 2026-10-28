"""
Phase 2: Comprehensive Advanced Feature Engineering Pipeline
Extracts B1 (E-commerce multi-window & velocity), B2 (Bureau multi-point from all 797k observations),
B3 (Repayment granular & EWMA), B4 (Interest rate residuals via OOF), B5 (Calendar & coverage),
B6 (Latent composite ranks), B7 (Interaction ratios & risk flags).
Identical, strictly point-in-time safe for Train and Test.
"""
import os
import json
import time
import polars as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb

DATA_DIR = "data/raw"
RANDOM_SEED = 42

def extract_b1_ecommerce_advanced(apps: pl.DataFrame, df_ecom: pl.DataFrame) -> pl.DataFrame:
    print("Extracting B1: Advanced E-commerce multi-window, velocity & categories...")
    ecom_j = df_ecom.join(
        apps.select([pl.col("customer_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="customer_id",
        how="inner"
    )
    # Strict cutoff filter
    ecom_valid = ecom_j.filter(pl.col("transaction_datetime") < pl.col("cutoff"))
    
    ecom_valid = ecom_valid.with_columns(
        days_since = (pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_days(),
        hour = pl.col("transaction_datetime").dt.hour(),
        day_of_week = pl.col("transaction_datetime").dt.weekday(),
        day_of_month = pl.col("transaction_datetime").dt.day(),
    )
    
    # Order level deduplication
    orders = ecom_valid.group_by("order_id").agg(
        customer_id = pl.col("customer_id").first(),
        transaction_datetime = pl.col("transaction_datetime").first(),
        days_since = pl.col("days_since").first(),
        order_total = pl.col("order_total").first(),
        payment_method = pl.col("payment_method").first(),
        order_status = pl.col("order_status").first(),
        hour = pl.col("hour").first(),
        day_of_week = pl.col("day_of_week").first(),
        day_of_month = pl.col("day_of_month").first(),
    )
    
    # 1. Overall order aggregations & cancellation ratios
    order_all_agg = orders.group_by("customer_id").agg(
        ecom_total_orders_all = pl.len(),
        ecom_canc_orders_all = (pl.col("order_status") == "cancelled").sum(),
        ecom_canc_orders_30d = ((pl.col("order_status") == "cancelled") & (pl.col("days_since") <= 30)).sum(),
        ecom_canc_orders_90d = ((pl.col("order_status") == "cancelled") & (pl.col("days_since") <= 90)).sum(),
        ecom_first_order_days = pl.col("days_since").max(),
        ecom_latest_order_days = pl.col("days_since").min(),
        ecom_history_length_days = pl.col("days_since").max() - pl.col("days_since").min(),
    ).with_columns(
        ecom_canc_rate_all = pl.col("ecom_canc_orders_all") / pl.col("ecom_total_orders_all"),
        ecom_canc_rate_30d = pl.col("ecom_canc_orders_30d") / (pl.col("ecom_canc_orders_all").is_not_null().cast(pl.Float64) * 0 + 1e-5), # will update below
    )
    
    # Completed orders only for spending & timings
    comp_orders = orders.filter(pl.col("order_status") == "completed").sort(["customer_id", "transaction_datetime"])
    
    # Inter-order gaps (time between consecutive completed orders)
    comp_orders = comp_orders.with_columns(
        prev_dt = pl.col("transaction_datetime").shift(1).over("customer_id")
    ).with_columns(
        order_gap_days = (pl.col("transaction_datetime") - pl.col("prev_dt")).dt.total_days()
    )
    
    gap_agg = comp_orders.group_by("customer_id").agg(
        ecom_gap_mean_days = pl.col("order_gap_days").mean(),
        ecom_gap_std_days = pl.col("order_gap_days").std(),
        ecom_gap_max_days = pl.col("order_gap_days").max(),
        ecom_gap_min_days = pl.col("order_gap_days").min(),
        # Timing patterns
        ecom_weekend_orders = (pl.col("day_of_week") >= 6).sum(),
        ecom_night_orders = ((pl.col("hour") >= 22) | (pl.col("hour") <= 5)).sum(),
        ecom_payday_orders = (pl.col("day_of_month") >= 25).sum(),
    )
    
    # Windows: 7d, 14d, 30d, 60d, 90d, 180d, 365d
    windows_agg = comp_orders.group_by("customer_id").agg(
        ecom_comp_orders_all = pl.len(),
        ecom_spend_all = pl.col("order_total").sum(),
        ecom_mean_order_val = pl.col("order_total").mean(),
        ecom_std_order_val = pl.col("order_total").std(),
        ecom_max_order_val = pl.col("order_total").max(),
        ecom_median_order_val = pl.col("order_total").median(),
        
        # 7 days
        ecom_spend_7d = pl.col("order_total").filter(pl.col("days_since") <= 7).sum(),
        ecom_orders_7d = (pl.col("days_since") <= 7).sum(),
        ecom_cod_orders_7d = ((pl.col("days_since") <= 7) & (pl.col("payment_method") == "COD")).sum(),
        ecom_cod_spend_7d = pl.col("order_total").filter((pl.col("days_since") <= 7) & (pl.col("payment_method") == "COD")).sum(),
        
        # 14 days
        ecom_spend_14d = pl.col("order_total").filter(pl.col("days_since") <= 14).sum(),
        ecom_orders_14d = (pl.col("days_since") <= 14).sum(),
        ecom_cod_orders_14d = ((pl.col("days_since") <= 14) & (pl.col("payment_method") == "COD")).sum(),
        ecom_cod_spend_14d = pl.col("order_total").filter((pl.col("days_since") <= 14) & (pl.col("payment_method") == "COD")).sum(),
        
        # 30 days
        ecom_spend_30d = pl.col("order_total").filter(pl.col("days_since") <= 30).sum(),
        ecom_orders_30d = (pl.col("days_since") <= 30).sum(),
        ecom_cod_orders_30d = ((pl.col("days_since") <= 30) & (pl.col("payment_method") == "COD")).sum(),
        ecom_cod_spend_30d = pl.col("order_total").filter((pl.col("days_since") <= 30) & (pl.col("payment_method") == "COD")).sum(),
        
        # 60 days
        ecom_spend_60d = pl.col("order_total").filter(pl.col("days_since") <= 60).sum(),
        ecom_orders_60d = (pl.col("days_since") <= 60).sum(),
        
        # 90 days
        ecom_spend_90d = pl.col("order_total").filter(pl.col("days_since") <= 90).sum(),
        ecom_orders_90d = (pl.col("days_since") <= 90).sum(),
        ecom_cod_orders_90d = ((pl.col("days_since") <= 90) & (pl.col("payment_method") == "COD")).sum(),
        
        # 180 days
        ecom_spend_180d = pl.col("order_total").filter(pl.col("days_since") <= 180).sum(),
        ecom_orders_180d = (pl.col("days_since") <= 180).sum(),
        
        # 365 days
        ecom_spend_365d = pl.col("order_total").filter(pl.col("days_since") <= 365).sum(),
        ecom_orders_365d = (pl.col("days_since") <= 365).sum(),
        
        # Payment method total counts
        ecom_cod_orders_all = (pl.col("payment_method") == "COD").sum(),
        ecom_ewallet_orders_all = (pl.col("payment_method") == "e_wallet").sum(),
        ecom_transfer_orders_all = (pl.col("payment_method") == "bank_transfer").sum(),
        ecom_credit_orders_all = (pl.col("payment_method") == "credit_card").sum(),
    ).with_columns(
        # Ratios & Velocity / Momentum
        ecom_ratio_spend_7d_30d = (pl.col("ecom_spend_7d") * 4.0) / (pl.col("ecom_spend_30d") + 1e-5),
        ecom_ratio_spend_14d_30d = (pl.col("ecom_spend_14d") * 2.0) / (pl.col("ecom_spend_30d") + 1e-5),
        ecom_ratio_spend_30d_90d = (pl.col("ecom_spend_30d") * 3.0) / (pl.col("ecom_spend_90d") + 1e-5),
        ecom_ratio_spend_30d_180d = (pl.col("ecom_spend_30d") * 6.0) / (pl.col("ecom_spend_180d") + 1e-5),
        ecom_ratio_spend_90d_365d = (pl.col("ecom_spend_90d") * 4.0) / (pl.col("ecom_spend_365d") + 1e-5),
        
        ecom_ratio_orders_7d_30d = (pl.col("ecom_orders_7d") * 4.0) / (pl.col("ecom_orders_30d") + 1e-5),
        ecom_ratio_orders_14d_30d = (pl.col("ecom_orders_14d") * 2.0) / (pl.col("ecom_orders_30d") + 1e-5),
        ecom_ratio_orders_30d_90d = (pl.col("ecom_orders_30d") * 3.0) / (pl.col("ecom_orders_90d") + 1e-5),
        
        # COD share per window
        ecom_cod_share_all = pl.col("ecom_cod_orders_all") / (pl.col("ecom_comp_orders_all") + 1e-5),
        ecom_cod_share_30d = pl.col("ecom_cod_orders_30d") / (pl.col("ecom_orders_30d") + 1e-5),
        ecom_cod_share_14d = pl.col("ecom_cod_orders_14d") / (pl.col("ecom_orders_14d") + 1e-5),
        ecom_cod_share_7d = pl.col("ecom_cod_orders_7d") / (pl.col("ecom_orders_7d") + 1e-5),
        ecom_cod_spend_share_30d = pl.col("ecom_cod_spend_30d") / (pl.col("ecom_spend_30d") + 1e-5),
    )
    
    # Item level product category aggregations
    comp_items = ecom_valid.filter(pl.col("order_status") == "completed")
    item_cat_agg = comp_items.group_by("customer_id").agg(
        ecom_items_total = pl.len(),
        ecom_items_groceries = (pl.col("product_category") == "groceries").sum(),
        ecom_items_electronics = (pl.col("product_category") == "electronics").sum(),
        ecom_items_fashion = (pl.col("product_category") == "fashion").sum(),
        ecom_items_personal_care = (pl.col("product_category") == "personal_care").sum(),
        ecom_items_home = (pl.col("product_category") == "home").sum(),
        # Category spending
        ecom_spend_groceries = pl.col("item_total").filter(pl.col("product_category") == "groceries").sum(),
        ecom_spend_electronics = pl.col("item_total").filter(pl.col("product_category") == "electronics").sum(),
        ecom_spend_fashion = pl.col("item_total").filter(pl.col("product_category") == "fashion").sum(),
    ).with_columns(
        ecom_share_groceries = pl.col("ecom_items_groceries") / (pl.col("ecom_items_total") + 1e-5),
        ecom_share_electronics = pl.col("ecom_items_electronics") / (pl.col("ecom_items_total") + 1e-5),
        ecom_share_fashion = pl.col("ecom_items_fashion") / (pl.col("ecom_items_total") + 1e-5),
        ecom_spend_share_electronics = pl.col("ecom_spend_electronics") / (pl.col("ecom_items_total").is_not_null().cast(pl.Float64) * 0 + pl.col("ecom_spend_groceries") + pl.col("ecom_spend_electronics") + pl.col("ecom_spend_fashion") + 1e-5),
    )
    
    # Merge all e-commerce tables
    res = order_all_agg.join(gap_agg, on="customer_id", how="left")
    res = res.join(windows_agg, on="customer_id", how="left")
    res = res.join(item_cat_agg, on="customer_id", how="left")
    return res

def extract_b2_bureau_multi_point(apps: pl.DataFrame, df_bureau: pl.DataFrame) -> pl.DataFrame:
    print("Extracting B2: Multi-point Bureau features from ALL 797k lifetime observations...")
    # Join on customer_id with cutoff
    b_j = df_bureau.join(
        apps.select([pl.col("customer_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="customer_id",
        how="inner"
    )
    # Strictly safe available_datetime <= cutoff
    b_valid = b_j.filter(pl.col("available_datetime") <= pl.col("cutoff")).sort(["customer_id", "available_datetime"])
    
    b_valid = b_valid.with_columns(
        days_since_avail = (pl.col("cutoff") - pl.col("available_datetime")).dt.total_days(),
        days_since_rep = (pl.col("cutoff") - pl.col("report_datetime")).dt.total_days(),
        is_unscored = pl.col("score").is_null().cast(pl.Int32),
    )
    
    # Lifetime multi-point aggregations across ALL reports
    b_life = b_valid.group_by("customer_id").agg(
        bureau_lifetime_obs_count = pl.len(),
        bureau_lifetime_unique_apps = pl.col("application_id").n_unique(),
        bureau_lifetime_providers_count = pl.col("bureau_provider").n_unique(),
        
        # Scores
        bureau_life_score_min = pl.col("score").min(),
        bureau_life_score_max = pl.col("score").max(),
        bureau_life_score_mean = pl.col("score").mean(),
        bureau_life_score_std = pl.col("score").std(),
        bureau_life_score_first = pl.col("score").first(),
        bureau_life_score_latest = pl.col("score").last(),
        
        # Balances & Limits
        bureau_life_out_bal_max = pl.col("outstanding_balance").max(),
        bureau_life_out_bal_mean = pl.col("outstanding_balance").mean(),
        bureau_life_out_bal_first = pl.col("outstanding_balance").first(),
        bureau_life_out_bal_latest = pl.col("outstanding_balance").last(),
        
        bureau_life_rev_bal_latest = pl.col("revolving_outstanding_balance").last(),
        bureau_life_rev_lim_latest = pl.col("revolving_credit_limit").last(),
        
        # Inquiries & Delinquency
        bureau_life_inquiries_max = pl.col("inquiries_last_6m").max(),
        bureau_life_inquiries_first = pl.col("inquiries_last_6m").first(),
        bureau_life_inquiries_latest = pl.col("inquiries_last_6m").last(),
        
        bureau_life_delinq_max = pl.col("delinquent_accounts_last_12m").max(),
        bureau_life_delinq_latest = pl.col("delinquent_accounts_last_12m").last(),
        bureau_life_ever_delinquent = (pl.col("delinquent_accounts_last_12m") > 0).max(),
        
        # Recency
        bureau_min_days_since_avail = pl.col("days_since_avail").min(),
        bureau_max_days_since_avail = pl.col("days_since_avail").max(),
    ).with_columns(
        # Dynamics: Change over time
        bureau_score_delta_latest_vs_first = pl.col("bureau_life_score_latest") - pl.col("bureau_life_score_first"),
        bureau_score_drop_from_max = pl.col("bureau_life_score_max") - pl.col("bureau_life_score_latest"),
        bureau_debt_growth = pl.col("bureau_life_out_bal_latest") - pl.col("bureau_life_out_bal_first"),
        bureau_inquiry_surge = pl.col("bureau_life_inquiries_latest") - pl.col("bureau_life_inquiries_first"),
        bureau_recalculated_util_latest = pl.when(pl.col("bureau_life_rev_lim_latest") > 0)
            .then(pl.col("bureau_life_rev_bal_latest") / pl.col("bureau_life_rev_lim_latest"))
            .otherwise(None),
    )
    
    # Provider-specific latest scores & disparities
    prov_latest = b_valid.group_by(["customer_id", "bureau_provider"]).agg(
        prov_score_latest = pl.col("score").last()
    )
    
    prov_pivot = prov_latest.pivot(on="bureau_provider", index="customer_id", values="prov_score_latest")
    prov_cols = [c for c in prov_pivot.columns if c != "customer_id"]
    rename_dict = {c: f"bureau_latest_score_{c}" for c in prov_cols}
    prov_pivot = prov_pivot.rename(rename_dict)
    
    # Provider disparity calculations
    b_features = b_life.join(prov_pivot, on="customer_id", how="left")
    if "bureau_latest_score_Bureau_A" in b_features.columns and "bureau_latest_score_Bureau_B" in b_features.columns:
        b_features = b_features.with_columns(
            bureau_disparity_A_minus_B = pl.col("bureau_latest_score_Bureau_A") - pl.col("bureau_latest_score_Bureau_B")
        )
    if "bureau_latest_score_Bureau_A" in b_features.columns and "bureau_latest_score_Bureau_C" in b_features.columns:
        b_features = b_features.with_columns(
            bureau_disparity_A_minus_C = pl.col("bureau_latest_score_Bureau_A") - pl.col("bureau_latest_score_Bureau_C")
        )
        
    return b_features

def extract_b3_repayment_granular(apps: pl.DataFrame, df_repay: pl.DataFrame) -> pl.DataFrame:
    print("Extracting B3: Granular Repayment, EWMA DPD & recent K installment trends...")
    repay_j = df_repay.join(
        apps.select([pl.col("customer_id"), pl.col("decision_datetime").alias("cutoff")]),
        on="customer_id",
        how="inner"
    )
    repay_valid = repay_j.filter(pl.col("due_date") < pl.col("cutoff")).sort(["customer_id", "due_date"])
    
    repay_valid = repay_valid.with_columns(
        days_since_due = (pl.col("cutoff") - pl.col("due_date")).dt.total_days(),
        is_ontime = (pl.col("days_past_due") == 0).cast(pl.Float64),
        is_underpaid = (pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Float64),
        paid_shortage = pl.col("amount_due") - pl.col("amount_paid"),
    ).with_columns(
        # Recency decay weight: w = exp(-days_since_due / 180)
        recency_weight = (-pl.col("days_since_due") / 180.0).exp(),
    )
    
    # Lifetime loan & installment metrics
    repay_life = repay_valid.group_by("customer_id").agg(
        repay_inst_total = pl.len(),
        repay_loans_total = pl.col("loan_id").n_unique(),
        repay_total_due = pl.col("amount_due").sum(),
        repay_total_paid = pl.col("amount_paid").sum(),
        repay_total_shortage = pl.col("paid_shortage").sum(),
        
        # Weighted on-time and paid ratio
        repay_weighted_ontime = (pl.col("is_ontime") * pl.col("recency_weight")).sum() / pl.col("recency_weight").sum(),
        repay_weighted_underpaid = (pl.col("is_underpaid") * pl.col("recency_weight")).sum() / pl.col("recency_weight").sum(),
        repay_weighted_paid_due_ratio = (pl.col("amount_paid") * pl.col("recency_weight")).sum() / ((pl.col("amount_due") * pl.col("recency_weight")).sum() + 1e-5),
        
        # Max DPDs
        repay_max_dpd_all = pl.col("days_past_due").max(),
        repay_mean_dpd_all = pl.col("days_past_due").mean(),
        repay_max_max_dpd_all = pl.col("max_days_past_due").max(),
        
        # Threshold counts
        repay_count_dpd_ge_30 = (pl.col("days_past_due") >= 30).sum(),
        repay_count_dpd_ge_60 = (pl.col("days_past_due") >= 60).sum(),
        repay_count_dpd_ge_90 = (pl.col("days_past_due") >= 90).sum(),
        
        # Time since last delay
        repay_days_since_last_delay = pl.col("days_since_due").filter(pl.col("days_past_due") > 0).min(),
        repay_active_loans_60d = pl.col("loan_id").filter(pl.col("days_since_due") <= 60).n_unique(),
        repay_active_monthly_due_60d = pl.col("amount_due").filter(pl.col("days_since_due") <= 60).sum() / 2.0,
    ).with_columns(
        repay_ratio_underpaid_all = pl.col("repay_total_shortage") / (pl.col("repay_total_due") + 1e-5),
    )
    
    # Recent K=3 installments
    repay_desc = repay_valid.sort(["customer_id", "due_date"], descending=[False, True])
    r3 = repay_desc.group_by("customer_id").head(3).group_by("customer_id").agg(
        repay_r3_max_dpd = pl.col("days_past_due").max(),
        repay_r3_mean_dpd = pl.col("days_past_due").mean(),
        repay_r3_underpaid_ratio = pl.col("is_underpaid").mean(),
        repay_r3_ontime_ratio = pl.col("is_ontime").mean(),
    )
    
    # Recent K=6 installments
    r6 = repay_desc.group_by("customer_id").head(6).group_by("customer_id").agg(
        repay_r6_max_dpd = pl.col("days_past_due").max(),
        repay_r6_mean_dpd = pl.col("days_past_due").mean(),
        repay_r6_underpaid_ratio = pl.col("is_underpaid").mean(),
        repay_r6_ontime_ratio = pl.col("is_ontime").mean(),
    )
    
    # Per-loan delinquency severity: proportion of customer's loans that ever defaulted
    loan_max_dpd = repay_valid.group_by(["customer_id", "loan_id"]).agg(
        loan_max_dpd = pl.col("days_past_due").max()
    ).group_by("customer_id").agg(
        repay_prop_loans_ever_dpd30 = (pl.col("loan_max_dpd") >= 30).mean(),
        repay_prop_loans_ever_dpd60 = (pl.col("loan_max_dpd") >= 60).mean(),
        repay_prop_loans_ever_dpd90 = (pl.col("loan_max_dpd") >= 90).mean(),
    )
    
    res = repay_life.join(r3, on="customer_id", how="left")
    res = res.join(r6, on="customer_id", how="left")
    res = res.join(loan_max_dpd, on="customer_id", how="left")
    return res

def extract_b4_interest_rate_residual(df_train_features: pl.DataFrame, df_test_features: pl.DataFrame):
    print("Extracting B4: Interest Rate Risk-Based Pricing Residuals (Nested OOF fit)...")
    # Features used to predict annual_interest_rate (strictly observable demographic/loan request features, NO TARGET!)
    pred_rate_features = [
        "loan_amount", "term_months", "proposed_monthly_installment",
        "loan_purpose", "age", "region", "employment_type", "employment_tenure_months",
        "declared_monthly_income", "dependents", "housing_status"
    ]
    
    tr_df = df_train_features.select(["application_id"] + pred_rate_features + ["annual_interest_rate"]).to_pandas()
    te_df = df_test_features.select(["application_id"] + pred_rate_features + ["annual_interest_rate"]).to_pandas()
    
    cat_cols = ["loan_purpose", "region", "employment_type", "housing_status"]
    for c in cat_cols:
        tr_df[c] = tr_df[c].astype("category")
        te_df[c] = te_df[c].astype("category")
        
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_pred_rate = np.zeros(len(tr_df))
    test_pred_rate = np.zeros(len(te_df))
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "random_state": RANDOM_SEED,
        "n_jobs": 4
    }
    
    for trn_i, val_i in kf.split(tr_df):
        X_tr = tr_df.iloc[trn_i][pred_rate_features]
        y_tr = tr_df.iloc[trn_i]["annual_interest_rate"]
        X_va = tr_df.iloc[val_i][pred_rate_features]
        y_va = tr_df.iloc[val_i]["annual_interest_rate"]
        
        trn_d = lgb.Dataset(X_tr, label=y_tr)
        val_d = lgb.Dataset(X_va, label=y_va, reference=trn_d)
        
        bst = lgb.train(params, trn_d, num_boost_round=300, valid_sets=[trn_d, val_d], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_pred_rate[val_i] = bst.predict(X_va)
        test_pred_rate += bst.predict(te_df[pred_rate_features]) / 5.0
        
    oof_residual = tr_df["annual_interest_rate"].values - oof_pred_rate
    test_residual = te_df["annual_interest_rate"].values - test_pred_rate
    
    train_res_pl = pl.DataFrame({
        "application_id": tr_df["application_id"],
        "pricing_predicted_rate": oof_pred_rate,
        "pricing_rate_residual": oof_residual,
    })
    
    test_res_pl = pl.DataFrame({
        "application_id": te_df["application_id"],
        "pricing_predicted_rate": test_pred_rate,
        "pricing_rate_residual": test_residual,
    })
    
    return train_res_pl, test_res_pl

def extract_b5_b6_b7_features(df: pl.DataFrame) -> pl.DataFrame:
    print("Extracting B5 (Calendar & coverage), B6 (Composite ranks), B7 (Interactions & flags)...")
    # B5. Calendar features
    df = df.with_columns(
        cal_month_num = pl.col("decision_datetime").dt.month(),
        cal_hour = pl.col("decision_datetime").dt.hour(),
        cal_dow = pl.col("decision_datetime").dt.weekday(),
        cal_dom = pl.col("decision_datetime").dt.day(),
    ).with_columns(
        cal_sin_hour = (2.0 * np.pi * pl.col("cal_hour") / 24.0).sin(),
        cal_cos_hour = (2.0 * np.pi * pl.col("cal_hour") / 24.0).cos(),
        cal_sin_month = (2.0 * np.pi * pl.col("cal_month_num") / 12.0).sin(),
        cal_cos_month = (2.0 * np.pi * pl.col("cal_month_num") / 12.0).cos(),
        cal_is_weekend = (pl.col("cal_dow") >= 6).cast(pl.Int32),
        cal_is_payday_window = (pl.col("cal_dom") >= 25).cast(pl.Int32),
    )
    
    # B5. Coverage indicators
    df = df.with_columns(
        cov_has_prev_apps = pl.col("prev_app_count").is_not_null().cast(pl.Int32),
        cov_has_repayment = pl.col("repay_inst_total").is_not_null().cast(pl.Int32),
        cov_has_bureau_B = pl.col("bureau_latest_score_Bureau_B").is_not_null().cast(pl.Int32),
        cov_has_bureau_C = pl.col("bureau_latest_score_Bureau_C").is_not_null().cast(pl.Int32),
        cov_has_account = pl.col("acct_observed_months").is_not_null().cast(pl.Int32),
    ).with_columns(
        cov_source_count = pl.col("cov_has_prev_apps") + pl.col("cov_has_repayment") + pl.col("cov_has_bureau_B") + pl.col("cov_has_bureau_C") + pl.col("cov_has_account") + 2 # +2 for app & ecom which are universal
    )
    
    # B6. Latent composite ranks
    # Rank signals normalized to [0, 1]
    df = df.with_columns(
        rank_ecom_spend_30d = pl.col("ecom_spend_30d").fill_null(0).rank() / pl.len(),
        rank_ecom_orders_30d = pl.col("ecom_orders_30d").fill_null(0).rank() / pl.len(),
        rank_repay_max_dpd = pl.col("repay_max_max_dpd_all").fill_null(0).rank() / pl.len(),
        rank_bureau_score_inv = (-pl.col("bureau_life_score_latest").fill_null(650)).rank() / pl.len(),
        rank_interest_rate = pl.col("annual_interest_rate").rank() / pl.len(),
        rank_installment_income = pl.col("installment_income_ratio").rank() / pl.len(),
    ).with_columns(
        composite_risk_score = (
            pl.col("rank_ecom_spend_30d") * 0.25 +
            pl.col("rank_repay_max_dpd") * 0.25 +
            pl.col("rank_bureau_score_inv") * 0.20 +
            pl.col("rank_interest_rate") * 0.15 +
            pl.col("rank_installment_income") * 0.15
        )
    )
    
    # B7. Interaction ratios and business flags
    df = df.with_columns(
        # E-commerce spend vs declared income
        ratio_ecom_spend_30d_income = pl.col("ecom_spend_30d") / (pl.col("declared_monthly_income") + 1e-5),
        ratio_ecom_spend_30d_installment = pl.col("ecom_spend_30d") / (pl.col("proposed_monthly_installment") + 1e-5),
        
        # Total debt burden (bureau balance + active loan due + current loan amount) vs annual income
        ratio_total_debt_annual_income = (pl.col("bureau_life_out_bal_latest").fill_null(0) + pl.col("loan_amount")) / (pl.col("declared_monthly_income") * 12.0 + 1e-5),
        
        # Risk flags
        flag_high_cod_high_spend = ((pl.col("ecom_cod_share_30d") > 0.6) & (pl.col("ecom_spend_30d") > 1_500_000)).cast(pl.Int32),
        flag_recent_dpd_low_bureau = ((pl.col("repay_r6_max_dpd").fill_null(0) > 0) & (pl.col("bureau_life_score_latest").fill_null(800) < 650)).cast(pl.Int32),
        flag_credit_hunger = ((pl.col("ecom_ratio_spend_7d_30d") > 1.5) & (pl.col("ecom_orders_7d") >= 2)).cast(pl.Int32),
    )
    
    return df

def build_advanced_feature_dataset():
    print("=== BUILDING ADVANCED FEATURE DATASET (FASE 2) ===")
    
    # Load raw tables
    df_train_raw = pl.read_csv(f"{DATA_DIR}/train_applications.csv", try_parse_dates=True)
    df_test_raw = pl.read_csv(f"{DATA_DIR}/test_applications.csv", try_parse_dates=True)
    df_cust = pl.read_csv(f"{DATA_DIR}/customer_profiles.csv")
    df_prev = pl.read_csv(f"{DATA_DIR}/previous_applications.csv", try_parse_dates=True)
    df_repay = pl.read_csv(f"{DATA_DIR}/repayment_history.csv", try_parse_dates=True)
    df_acct = pl.read_csv(f"{DATA_DIR}/account_behavior.csv", try_parse_dates=True)
    df_bureau = pl.read_csv(f"{DATA_DIR}/bureau_observations.csv", try_parse_dates=True)
    df_ecom = pl.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv", try_parse_dates=True)
    
    # Load baseline 132 features as base
    train_base = pl.read_parquet("data/processed/train_features.parquet")
    test_base = pl.read_parquet("data/processed/test_features.parquet")
    
    # 1. Extract B1 E-commerce advanced
    t0 = time.time()
    ecom_tr = extract_b1_ecommerce_advanced(df_train_raw, df_ecom)
    ecom_te = extract_b1_ecommerce_advanced(df_test_raw, df_ecom)
    print(f"B1 E-commerce extracted in {time.time()-t0:.1f}s ({len(ecom_tr.columns)} cols)")
    
    # 2. Extract B2 Bureau multi-point
    t0 = time.time()
    bureau_tr = extract_b2_bureau_multi_point(df_train_raw, df_bureau)
    bureau_te = extract_b2_bureau_multi_point(df_test_raw, df_bureau)
    print(f"B2 Bureau extracted in {time.time()-t0:.1f}s ({len(bureau_tr.columns)} cols)")
    
    # 3. Extract B3 Repayment granular
    t0 = time.time()
    repay_tr = extract_b3_repayment_granular(df_train_raw, df_repay)
    repay_te = extract_b3_repayment_granular(df_test_raw, df_repay)
    print(f"B3 Repayment extracted in {time.time()-t0:.1f}s ({len(repay_tr.columns)} cols)")
    
    # 4. Extract B4 Residual Bunga
    t0 = time.time()
    b4_tr, b4_te = extract_b4_interest_rate_residual(train_base, test_base)
    print(f"B4 Residual Bunga extracted in {time.time()-t0:.1f}s ({len(b4_tr.columns)} cols)")
    
    # Join new features to baseline tables
    print("Joining new feature groups to base tables...")
    # Helper to join without duplicating columns
    def join_clean(base_df, new_df, on_key):
        existing = [c for c in new_df.columns if c in base_df.columns and c != on_key]
        clean_new = new_df.drop(existing)
        return base_df.join(clean_new, on=on_key, how="left")
    
    train_adv = join_clean(train_base, ecom_tr, "customer_id")
    test_adv = join_clean(test_base, ecom_te, "customer_id")
    
    train_adv = join_clean(train_adv, bureau_tr, "customer_id")
    test_adv = join_clean(test_adv, bureau_te, "customer_id")
    
    train_adv = join_clean(train_adv, repay_tr, "customer_id")
    test_adv = join_clean(test_adv, repay_te, "customer_id")
    
    train_adv = join_clean(train_adv, b4_tr, "application_id")
    test_adv = join_clean(test_adv, b4_te, "application_id")
    
    # 5. Extract B5, B6, B7 features
    train_adv = extract_b5_b6_b7_features(train_adv)
    test_adv = extract_b5_b6_b7_features(test_adv)
    
    # Validations
    print("Validating advanced feature tables...")
    assert len(train_adv) == 70000, f"Train row mismatch: {len(train_adv)}"
    assert len(test_adv) == 30000, f"Test row mismatch: {len(test_adv)}"
    assert train_adv["application_id"].n_unique() == 70000, "Train fanout!"
    assert test_adv["application_id"].n_unique() == 30000, "Test fanout!"
    
    diff_cols = set(train_adv.columns) - set(test_adv.columns)
    assert diff_cols == {"target"}, f"Column discrepancy: {diff_cols}"
    
    print(f"Final Advanced Feature Dataset: Train={train_adv.shape}, Test={test_adv.shape}")
    
    # Save to parquet
    train_path = "data/processed/train_features_v2.parquet"
    test_path = "data/processed/test_features_v2.parquet"
    train_adv.write_parquet(train_path)
    test_adv.write_parquet(test_path)
    print(f"Saved advanced feature tables to {train_path} and {test_path}!")

if __name__ == "__main__":
    build_advanced_feature_dataset()
