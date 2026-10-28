"""
Test script for v4 feature extraction logic on small sample
"""
import polars as pl
import numpy as np

def compute_linear_slopes(seqs):
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

def main():
    print("Testing v4 feature extractor on 500 sample applications...")
    apps = pl.read_csv("data/raw/train_applications.csv", n_rows=500).with_columns(
        pl.col("decision_datetime").str.to_datetime()
    )
    cust_set = set(apps["customer_id"].to_list())
    
    # 1. Repayment History
    print("Testing Repayment History extraction...")
    repay = pl.scan_csv("data/raw/repayment_history.csv").with_columns(
        pl.col("due_date").str.to_datetime(),
        pl.col("last_payment_datetime").str.to_datetime()
    ).join(
        apps.select(["customer_id", pl.col("decision_datetime").alias("cutoff")]).lazy(),
        on="customer_id",
        how="inner"
    ).filter(
        pl.col("due_date") < pl.col("cutoff")
    ).sort(["customer_id", "due_date"]).collect()
    
    print(f"Repayment filtered rows: {len(repay)}")
    
    # Test streak calculation
    repay_grouped = repay.group_by("customer_id", maintain_order=True).agg(
        dpd_list = pl.col("days_past_due"),
        days_since_due_list = (pl.col("cutoff") - pl.col("due_date")).dt.total_days(),
        amt_due_list = pl.col("amount_due"),
        amt_paid_list = pl.col("amount_paid"),
        last_pay_dt = pl.col("last_payment_datetime"),
        due_dt = pl.col("due_date")
    )
    
    cust_ids = repay_grouped["customer_id"].to_list()
    dpd_seqs = repay_grouped["dpd_list"].to_list()
    days_since_seqs = repay_grouped["days_since_due_list"].to_list()
    
    late_streak_cutoff = []
    max_late_streak = []
    days_since_last_30_dpd = []
    days_since_last_60_dpd = []
    days_since_last_90_dpd = []
    
    for dpd, ds in zip(dpd_seqs, days_since_seqs):
        arr = np.array(dpd, dtype=np.int32)
        ds_arr = np.array(ds, dtype=np.float32)
        
        # cutoff streak
        curr = 0
        for val in reversed(arr):
            if val > 0:
                curr += 1
            else:
                break
        late_streak_cutoff.append(curr)
        
        # max streak
        max_s = 0
        c = 0
        for val in arr:
            if val > 0:
                c += 1
                if c > max_s: max_s = c
            else:
                c = 0
        max_late_streak.append(max_s)
        
        # days since 30/60/90
        m30 = ds_arr[arr >= 30]
        days_since_last_30_dpd.append(float(np.min(m30)) if len(m30) > 0 else 999.0)
        m60 = ds_arr[arr >= 60]
        days_since_last_60_dpd.append(float(np.min(m60)) if len(m60) > 0 else 999.0)
        m90 = ds_arr[arr >= 90]
        days_since_last_90_dpd.append(float(np.min(m90)) if len(m90) > 0 else 999.0)
        
    df_repay_features = pl.DataFrame({
        "customer_id": cust_ids,
        "repay_late_streak_cutoff": np.array(late_streak_cutoff, dtype=np.int32),
        "repay_max_late_streak_all": np.array(max_late_streak, dtype=np.int32),
        "repay_days_since_last_30_dpd": np.array(days_since_last_30_dpd, dtype=np.float32),
        "repay_days_since_last_60_dpd": np.array(days_since_last_60_dpd, dtype=np.float32),
        "repay_days_since_last_90_dpd": np.array(days_since_last_90_dpd, dtype=np.float32),
    })
    print("Repayment features shape:", df_repay_features.shape)
    print(df_repay_features.head(3))
    print("Test passed successfully!")

if __name__ == "__main__":
    main()
