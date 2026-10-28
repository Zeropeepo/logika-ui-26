"""
Feature Factory v4: Massive Multi-Table & Longitudinal Time-Series Feature Engineering
Extracts:
1. Longitudinal slopes and second-order acceleration (Account Behavior & Bureau Observations)
2. Granular repayment delinquency streaks, recency (30/60/90 DPD), and early payment dynamics
3. Inter-event arrival time dynamics (burstiness & skewness) and discretionary spending in E-Commerce
4. Historical loan application haircuts, rejection recency, and approval velocity
5. Cross-silo domain ratios (DTI, Installment-to-Net-Cashflow, Leverage, Coverage)
Processes via disk-backed parquets to ensure RAM usage stays well below limits.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import gc
import time
import polars as pl
import numpy as np
from scipy.stats import skew
from src.utils_watchdog import cleanup_memory, log_resource_status, downcast_polars

DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_INTERIM = "data/interim"

def compute_slopes_vectorized(seqs):
    """Computes linear OLS slope for list of sequences."""
    slopes = np.zeros(len(seqs), dtype=np.float32)
    for i, y in enumerate(seqs):
        n = len(y)
        if n <= 1:
            slopes[i] = 0.0
            continue
        t = np.arange(n, dtype=np.float32)
        t_bar = (n - 1) / 2.0
        denom = n * (n**2 - 1) / 12.0
        numer = np.sum((t - t_bar) * y)
        slopes[i] = numer / denom
    return slopes

def extract_account_v4(apps: pl.DataFrame, is_train: bool) -> pl.DataFrame:
    print(f"--- Extracting Account Behavior v4 ({'Train' if is_train else 'Test'}) ---")
    log_resource_status("[ACCT]")
    
    # apps has customer_id and decision_datetime
    apps_cut = apps.select([
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff")
    ])
    
    # Calculate beginning of cutoff month
    # Complete months only: month < cutoff_month_start
    acct = pl.scan_csv(f"{DATA_RAW}/account_behavior.csv").with_columns(
        pl.col("month").str.to_datetime("%Y-%m-%d")
    ).join(
        apps_cut.lazy(),
        on="customer_id",
        how="inner"
    ).filter(
        # Complete calendar month: month + 1 month <= cutoff, or month.dt.truncate('1mo') < cutoff.dt.truncate('1mo')
        pl.col("month") < pl.col("cutoff").dt.truncate("1mo")
    ).sort(["customer_id", "month"]).collect()
    
    acct = acct.with_columns(
        net_cashflow = pl.col("inflows") - pl.col("outflows"),
        bal_change = pl.col("closing_balance") - pl.col("opening_balance")
    )
    
    grouped = acct.group_by("customer_id", maintain_order=True).agg(
        obs_count = pl.len(),
        closing_bal_seq = pl.col("closing_balance"),
        inflows_seq = pl.col("inflows"),
        outflows_seq = pl.col("outflows"),
        low_bal_seq = pl.col("low_balance_days"),
        net_cash_seq = pl.col("net_cashflow"),
        income_credits_sum = pl.col("income_like_credits").sum(),
        inflows_sum = pl.col("inflows").sum(),
        min_bal = pl.col("minimum_balance").min(),
        avg_bal = pl.col("average_balance").mean(),
        # Tail 3 and 6 months
        net_cash_3m = pl.col("net_cashflow").tail(3).mean(),
        net_cash_6m = pl.col("net_cashflow").tail(6).mean(),
        latest_bal = pl.col("closing_balance").last(),
        first_bal = pl.col("closing_balance").first(),
    )
    
    cust_ids = grouped["customer_id"].to_list()
    bal_seqs = grouped["closing_bal_seq"].to_list()
    inf_seqs = grouped["inflows_seq"].to_list()
    out_seqs = grouped["outflows_seq"].to_list()
    low_seqs = grouped["low_bal_seq"].to_list()
    net_seqs = grouped["net_cash_seq"].to_list()
    
    slope_bal = compute_slopes_vectorized(bal_seqs)
    slope_inf = compute_slopes_vectorized(inf_seqs)
    slope_out = compute_slopes_vectorized(out_seqs)
    slope_low = compute_slopes_vectorized(low_seqs)
    slope_net = compute_slopes_vectorized(net_seqs)
    
    # Second-order balance acceleration: (latest - prev) - (prev - prev2)
    bal_accel = np.zeros(len(cust_ids), dtype=np.float32)
    for i, seq in enumerate(bal_seqs):
        if len(seq) >= 3:
            diff1 = seq[-1] - seq[-2]
            diff0 = seq[-2] - seq[-3]
            bal_accel[i] = diff1 - diff0
            
    res = pl.DataFrame({
        "customer_id": cust_ids,
        "acct_slope_closing_balance": slope_bal,
        "acct_slope_inflows": slope_inf,
        "acct_slope_outflows": slope_out,
        "acct_slope_low_balance_days": slope_low,
        "acct_slope_net_cashflow": slope_net,
        "acct_balance_acceleration": bal_accel,
        "acct_net_cashflow_3m": grouped["net_cash_3m"].cast(pl.Float32),
        "acct_net_cashflow_6m": grouped["net_cash_6m"].cast(pl.Float32),
        "acct_income_credits_ratio": (grouped["income_credits_sum"] / (grouped["inflows_sum"] + 1e-5)).cast(pl.Float32),
        "acct_min_to_avg_bal_ratio": (grouped["min_bal"] / (grouped["avg_bal"] + 1e-5)).cast(pl.Float32),
    })
    
    del acct, grouped, cust_ids, bal_seqs, inf_seqs, out_seqs, low_seqs, net_seqs
    cleanup_memory()
    return downcast_polars(res)

def extract_bureau_v4(apps: pl.DataFrame, is_train: bool) -> pl.DataFrame:
    print(f"--- Extracting Bureau Observations v4 ({'Train' if is_train else 'Test'}) ---")
    log_resource_status("[BUREAU]")
    
    apps_cut = apps.select([
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff")
    ])
    
    bureau = pl.scan_csv(f"{DATA_RAW}/bureau_observations.csv").with_columns(
        pl.col("available_datetime").str.to_datetime()
    ).join(
        apps_cut.lazy(),
        on="customer_id",
        how="inner"
    ).filter(
        pl.col("available_datetime") <= pl.col("cutoff")
    ).sort(["customer_id", "available_datetime"]).collect()
    
    grouped = bureau.group_by("customer_id", maintain_order=True).agg(
        score_seq = pl.col("score").fill_null(strategy="forward").fill_null(650),
        out_bal_seq = pl.col("outstanding_balance"),
        util_seq = pl.col("utilization").fill_null(strategy="forward").fill_null(0.0),
        rev_lim_first = pl.col("revolving_credit_limit").first(),
        rev_lim_latest = pl.col("revolving_credit_limit").last(),
        inq_first = pl.col("inquiries_last_6m").first(),
        inq_latest = pl.col("inquiries_last_6m").last(),
        inq_mean = pl.col("inquiries_last_6m").mean(),
    )
    
    cust_ids = grouped["customer_id"].to_list()
    score_seqs = grouped["score_seq"].to_list()
    out_bal_seqs = grouped["out_bal_seq"].to_list()
    util_seqs = grouped["util_seq"].to_list()
    
    slope_score = compute_slopes_vectorized(score_seqs)
    slope_out_bal = compute_slopes_vectorized(out_bal_seqs)
    slope_util = compute_slopes_vectorized(util_seqs)
    
    score_accel = np.zeros(len(cust_ids), dtype=np.float32)
    for i, seq in enumerate(score_seqs):
        if len(seq) >= 3:
            d1 = seq[-1] - seq[-2]
            d0 = seq[-2] - seq[-3]
            score_accel[i] = d1 - d0
            
    limit_cut = (grouped["rev_lim_latest"] < grouped["rev_lim_first"]).cast(pl.Int32)
    inq_surge = (grouped["inq_latest"] - grouped["inq_first"]).cast(pl.Float32)
    inq_accel = (grouped["inq_latest"] - grouped["inq_mean"]).cast(pl.Float32)
    
    res = pl.DataFrame({
        "customer_id": cust_ids,
        "bureau_slope_score": slope_score,
        "bureau_slope_out_bal": slope_out_bal,
        "bureau_slope_util": slope_util,
        "bureau_score_acceleration": score_accel,
        "bureau_limit_cut_flag": limit_cut,
        "bureau_inquiries_surge_rate": inq_surge,
        "bureau_inquiries_accel": inq_accel,
    })
    
    del bureau, grouped, cust_ids, score_seqs, out_bal_seqs, util_seqs
    cleanup_memory()
    return downcast_polars(res)

def extract_repayment_v4(apps: pl.DataFrame, is_train: bool) -> pl.DataFrame:
    print(f"--- Extracting Repayment History v4 ({'Train' if is_train else 'Test'}) ---")
    log_resource_status("[REPAY]")
    
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
    ).sort(["customer_id", "due_date"]).collect()
    
    repay = repay.with_columns(
        days_since_due = (pl.col("cutoff") - pl.col("due_date")).dt.total_days(),
        days_early = (pl.col("due_date") - pl.col("last_payment_datetime")).dt.total_days(),
        part_ratio = pl.col("amount_paid") / (pl.col("amount_due") + 1e-5)
    )
    
    grouped = repay.group_by("customer_id", maintain_order=True).agg(
        dpd_seq = pl.col("days_past_due"),
        days_since_seq = pl.col("days_since_due"),
        early_seq = pl.col("days_early").filter(pl.col("days_past_due") == 0),
        part_ratio_min = pl.col("part_ratio").min(),
        part_ratio_mean = pl.col("part_ratio").mean(),
        recent_3_dpd = pl.col("days_past_due").tail(3).mean(),
        all_dpd = pl.col("days_past_due").mean(),
    )
    
    cust_ids = grouped["customer_id"].to_list()
    dpd_seqs = grouped["dpd_seq"].to_list()
    days_since_seqs = grouped["days_since_seq"].to_list()
    early_seqs = grouped["early_seq"].to_list()
    
    late_streak_cutoff = []
    max_late_streak = []
    days_since_30 = []
    days_since_60 = []
    days_since_90 = []
    mean_early = []
    
    for i, (dpd, ds) in enumerate(zip(dpd_seqs, days_since_seqs)):
        arr = np.array(dpd, dtype=np.int32)
        ds_arr = np.array(ds, dtype=np.float32)
        
        # Cutoff streak
        curr = 0
        for val in reversed(arr):
            if val > 0:
                curr += 1
            else:
                break
        late_streak_cutoff.append(curr)
        
        # Max streak
        max_s = 0
        c = 0
        for val in arr:
            if val > 0:
                c += 1
                if c > max_s: max_s = c
            else:
                c = 0
        max_late_streak.append(max_s)
        
        m30 = ds_arr[arr >= 30]
        days_since_30.append(float(np.min(m30)) if len(m30) > 0 else 999.0)
        m60 = ds_arr[arr >= 60]
        days_since_60.append(float(np.min(m60)) if len(m60) > 0 else 999.0)
        m90 = ds_arr[arr >= 90]
        days_since_90.append(float(np.min(m90)) if len(m90) > 0 else 999.0)
        
        e_arr = early_seqs[i]
        if len(e_arr) > 0:
            # Positive values mean early payment
            valid_e = [float(x) for x in e_arr if x is not None and x > 0]
            mean_early.append(float(np.mean(valid_e)) if len(valid_e) > 0 else 0.0)
        else:
            mean_early.append(0.0)
            
    dpd_slopes = compute_slopes_vectorized(dpd_seqs)
    dpd_ratio = (grouped["recent_3_dpd"] / (grouped["all_dpd"] + 1e-5)).cast(pl.Float32)
    
    res = pl.DataFrame({
        "customer_id": cust_ids,
        "repay_late_streak_cutoff": np.array(late_streak_cutoff, dtype=np.int32),
        "repay_max_late_streak_all": np.array(max_late_streak, dtype=np.int32),
        "repay_days_since_last_30_dpd": np.array(days_since_30, dtype=np.float32),
        "repay_days_since_last_60_dpd": np.array(days_since_60, dtype=np.float32),
        "repay_days_since_last_90_dpd": np.array(days_since_90, dtype=np.float32),
        "repay_mean_days_early": np.array(mean_early, dtype=np.float32),
        "repay_partial_payment_ratio_min": grouped["part_ratio_min"].cast(pl.Float32),
        "repay_partial_payment_ratio_mean": grouped["part_ratio_mean"].cast(pl.Float32),
        "repay_dpd_slope": dpd_slopes,
        "repay_dpd_recent_vs_all_ratio": dpd_ratio,
    })
    
    del repay, grouped, cust_ids, dpd_seqs, days_since_seqs, early_seqs
    cleanup_memory()
    return downcast_polars(res)

def extract_ecom_v4(apps: pl.DataFrame, is_train: bool) -> pl.DataFrame:
    print(f"--- Extracting E-commerce v4 ({'Train' if is_train else 'Test'}) ---")
    log_resource_status("[ECOM]")
    
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
    
    # Completed orders for gaps and spending
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
    )
    
    comp_orders = orders.filter(pl.col("order_status") == "completed").sort(["customer_id", "transaction_datetime"])
    comp_orders = comp_orders.with_columns(
        prev_dt = pl.col("transaction_datetime").shift(1).over("customer_id")
    ).with_columns(
        gap_days = (pl.col("transaction_datetime") - pl.col("prev_dt")).dt.total_days()
    )
    
    grouped_comp = comp_orders.group_by("customer_id", maintain_order=True).agg(
        gap_seq = pl.col("gap_days").drop_nulls(),
        night_spend = pl.col("order_total").filter((pl.col("hour") >= 0) & (pl.col("hour") <= 5)).sum(),
        total_spend = pl.col("order_total").sum(),
        cod_14d = pl.col("order_total").filter((pl.col("days_since") <= 14) & (pl.col("payment_method") == "COD")).sum(),
        spend_14d = pl.col("order_total").filter(pl.col("days_since") <= 14).sum(),
        cod_180d = pl.col("order_total").filter((pl.col("days_since") <= 180) & (pl.col("payment_method") == "COD")).sum(),
        spend_180d = pl.col("order_total").filter(pl.col("days_since") <= 180).sum(),
    )
    
    # Product categories: essential vs discretionary
    # essential: groceries, health, education
    # discretionary: electronics, fashion, home, personal_care
    cat_spend = ecom.filter(pl.col("order_status") == "completed").group_by("customer_id").agg(
        essential_spend = pl.col("item_total").filter(pl.col("product_category").is_in(["groceries", "health", "education"])).sum(),
        discretionary_spend = pl.col("item_total").filter(pl.col("product_category").is_in(["electronics", "fashion", "home", "personal_care"])).sum(),
    )
    
    cust_ids = grouped_comp["customer_id"].to_list()
    gap_seqs = grouped_comp["gap_seq"].to_list()
    
    skew_vals = []
    burst_vals = []
    for gaps in gap_seqs:
        if len(gaps) >= 3:
            arr = np.array(gaps, dtype=np.float32)
            m = float(np.mean(arr))
            s = float(np.std(arr))
            sk = float(skew(arr))
            b = (s - m) / (s + m + 1e-5)
            skew_vals.append(sk if np.isfinite(sk) else 0.0)
            burst_vals.append(b if np.isfinite(b) else 0.0)
        else:
            skew_vals.append(0.0)
            burst_vals.append(0.0)
            
    res_ecom = pl.DataFrame({
        "customer_id": cust_ids,
        "ecom_gap_skew_days": np.array(skew_vals, dtype=np.float32),
        "ecom_burstiness_metric": np.array(burst_vals, dtype=np.float32),
        "ecom_night_spend_share": (grouped_comp["night_spend"] / (grouped_comp["total_spend"] + 1e-5)).cast(pl.Float32),
        "ecom_cod_velocity_14d_vs_180d": (
            (grouped_comp["cod_14d"] / (grouped_comp["spend_14d"] + 1e-5)) -
            (grouped_comp["cod_180d"] / (grouped_comp["spend_180d"] + 1e-5))
        ).cast(pl.Float32),
    }).join(
        cat_spend.with_columns(
            ecom_discretionary_spend_ratio = (pl.col("discretionary_spend") / (pl.col("essential_spend") + 1.0)).cast(pl.Float32)
        ).select(["customer_id", "ecom_discretionary_spend_ratio"]),
        on="customer_id",
        how="left"
    )
    
    del ecom, orders, comp_orders, grouped_comp, cat_spend, cust_ids, gap_seqs
    cleanup_memory()
    return downcast_polars(res_ecom)

def extract_prev_apps_v4(apps: pl.DataFrame, is_train: bool) -> pl.DataFrame:
    print(f"--- Extracting Previous Applications v4 ({'Train' if is_train else 'Test'}) ---")
    log_resource_status("[PREV_APPS]")
    
    apps_cut = apps.select([
        pl.col("customer_id"),
        pl.col("decision_datetime").alias("cutoff")
    ])
    
    prev = pl.scan_csv(f"{DATA_RAW}/previous_applications.csv").with_columns(
        pl.col("decision_datetime").str.to_datetime()
    ).join(
        apps_cut.lazy(),
        on="customer_id",
        how="inner"
    ).filter(
        pl.col("decision_datetime") < pl.col("cutoff")
    ).collect()
    
    prev = prev.with_columns(
        days_since_app = (pl.col("cutoff") - pl.col("decision_datetime")).dt.total_days(),
        haircut = pl.col("approved_amount").fill_null(0) / (pl.col("requested_amount") + 1e-5)
    )
    
    grouped = prev.group_by("customer_id").agg(
        total_prev = pl.len(),
        appr_count = (pl.col("decision") == "approved").sum(),
        rej_count = (pl.col("decision") == "rejected").sum(),
        canc_count = (pl.col("decision") == "cancelled").sum(),
        mean_haircut = pl.col("haircut").filter(pl.col("decision") == "approved").mean(),
        days_since_last_app = pl.col("days_since_app").min(),
        days_since_last_rej = pl.col("days_since_app").filter(pl.col("decision") == "rejected").min(),
        history_days = pl.col("days_since_app").max() - pl.col("days_since_app").min(),
    ).with_columns(
        prev_apps_approval_rate = (pl.col("appr_count") / (pl.col("total_prev") + 1e-5)).cast(pl.Float32),
        prev_apps_rejection_rate = (pl.col("rej_count") / (pl.col("total_prev") + 1e-5)).cast(pl.Float32),
        prev_apps_cancellation_rate = (pl.col("canc_count") / (pl.col("total_prev") + 1e-5)).cast(pl.Float32),
        prev_apps_haircut_mean = pl.col("mean_haircut").fill_null(0.0).cast(pl.Float32),
        prev_days_since_latest_app = pl.col("days_since_last_app").cast(pl.Float32),
        prev_days_since_latest_rejection = pl.col("days_since_last_rej").fill_null(999.0).cast(pl.Float32),
        prev_apps_frequency_per_month = (pl.col("total_prev") / (pl.col("history_days") / 30.0 + 1.0)).cast(pl.Float32),
    ).select([
        "customer_id",
        "prev_apps_approval_rate",
        "prev_apps_rejection_rate",
        "prev_apps_cancellation_rate",
        "prev_apps_haircut_mean",
        "prev_days_since_latest_app",
        "prev_days_since_latest_rejection",
        "prev_apps_frequency_per_month",
    ])
    
    del prev
    cleanup_memory()
    return downcast_polars(grouped)

def add_cross_domain_ratios(df: pl.DataFrame) -> pl.DataFrame:
    print("Computing Cross-Table Non-Linear Domain Interactions...")
    # Safe fill_null
    out_bal = df["bureau_life_out_bal_latest"].fill_null(0)
    inst = df["proposed_monthly_installment"]
    income = df["declared_monthly_income"]
    loan_amt = df["loan_amount"]
    rate = df["annual_interest_rate"]
    bureau_score = df["bureau_life_score_latest"].fill_null(650)
    
    net_cf = (df["acct_mean_inflows_all"].fill_null(0) - df["acct_mean_outflows_all"].fill_null(0)).clip(lower_bound=1.0)
    avg_bal = df["acct_mean_balance_all"].fill_null(0)
    
    dti = (out_bal + inst) / (income + 1.0)
    inst_to_cashflow = inst / (net_cf + 1.0)
    bal_to_loan = avg_bal / (loan_amt + 1.0)
    tot_leverage = (out_bal + loan_amt) / (income * 12.0 + 1.0)
    cov_ratio = avg_bal / (inst + 1.0)
    rate_spread = rate * (850.0 - bureau_score) / 100.0
    
    res = df.with_columns(
        cross_dti_ratio = dti.cast(pl.Float32),
        cross_installment_to_net_cashflow = inst_to_cashflow.cast(pl.Float32),
        cross_balance_to_loan = bal_to_loan.cast(pl.Float32),
        cross_total_leverage_ratio = tot_leverage.cast(pl.Float32),
        cross_coverage_ratio = cov_ratio.cast(pl.Float32),
        cross_rate_bureau_spread = rate_spread.cast(pl.Float32),
    )
    return downcast_polars(res)

def build_v4_dataset():
    os.makedirs(DATA_INTERIM, exist_ok=True)
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    
    t0 = time.time()
    print("=== BUILDING DATASET V4 (TRAIN + TEST) ===")
    
    for split, is_train in [("train", True), ("test", False)]:
        print(f"\n==================== PROCESSING {split.upper()} ====================")
        v3_path = f"{DATA_PROCESSED}/{split}_features_v3.parquet"
        df_v3 = pl.read_parquet(v3_path)
        print(f"Loaded {split} v3: {df_v3.shape}")
        
        # 1. Account
        df_acct = extract_account_v4(df_v3, is_train)
        df_merged = df_v3.join(df_acct, on="customer_id", how="left")
        del df_acct; cleanup_memory()
        
        # 2. Bureau
        df_bureau = extract_bureau_v4(df_v3, is_train)
        df_merged = df_merged.join(df_bureau, on="customer_id", how="left")
        del df_bureau; cleanup_memory()
        
        # 3. Repayment
        df_repay = extract_repayment_v4(df_v3, is_train)
        df_merged = df_merged.join(df_repay, on="customer_id", how="left")
        del df_repay; cleanup_memory()
        
        # 4. E-commerce
        df_ecom = extract_ecom_v4(df_v3, is_train)
        df_merged = df_merged.join(df_ecom, on="customer_id", how="left")
        del df_ecom; cleanup_memory()
        
        # 5. Previous Applications
        df_prev = extract_prev_apps_v4(df_v3, is_train)
        df_merged = df_merged.join(df_prev, on="customer_id", how="left")
        del df_prev; cleanup_memory()
        
        # 6. Cross Domain Ratios
        df_merged = add_cross_domain_ratios(df_merged)
        
        # Final downcasting
        df_final = downcast_polars(df_merged)
        out_path = f"{DATA_PROCESSED}/{split}_features_v4.parquet"
        df_final.write_parquet(out_path)
        print(f"Saved {split} v4 to {out_path} with shape {df_final.shape} ({os.path.getsize(out_path)/1024**2:.1f} MB)")
        
        del df_v3, df_merged, df_final
        cleanup_memory()
        
    print(f"=== COMPLETED FEATURE EXTRACTION IN {time.time() - t0:.1f}s ===")

if __name__ == "__main__":
    build_v4_dataset()
