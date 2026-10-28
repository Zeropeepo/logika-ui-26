"""
Stage 4 Validation: Automated Assertion on Random Samples & Feature Dictionary Generator
"""
import os
import json
import polars as pl
import numpy as np

def run_stage4_verification():
    print("Running Stage 4 Verification & Validation...")
    train_feat = pl.read_parquet("data/processed/train_features.parquet")
    test_feat = pl.read_parquet("data/processed/test_features.parquet")
    
    # 1. Random Sample Automated Assertion
    print("Testing random samples manual reconstruction...")
    # Load raw tables
    df_train = pl.read_csv("data/raw/train_applications.csv", try_parse_dates=True)
    df_prev = pl.read_csv("data/raw/previous_applications.csv", try_parse_dates=True)
    df_repay = pl.read_csv("data/raw/repayment_history.csv", try_parse_dates=True)
    df_acct = pl.read_csv("data/raw/account_behavior.csv", try_parse_dates=True)
    df_bureau = pl.read_csv("data/raw/bureau_observations.csv", try_parse_dates=True)
    df_ecom = pl.read_csv("data/raw/ecommerce_transactions.csv", try_parse_dates=True)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(df_train), 5, replace=False)
    sample_apps = df_train[sample_indices]
    
    for row in sample_apps.iter_rows(named=True):
        app_id = row["application_id"]
        cust_id = row["customer_id"]
        cutoff = row["decision_datetime"]
        
        # Parquet row
        p_row = train_feat.filter(pl.col("application_id") == app_id).to_dicts()[0]
        
        # Manual Prev apps
        man_prev = df_prev.filter((pl.col("customer_id") == cust_id) & (pl.col("decision_datetime") < cutoff))
        exp_prev_count = len(man_prev) if len(man_prev) > 0 else None
        act_prev_count = p_row["prev_app_count"]
        assert (exp_prev_count == act_prev_count) or (exp_prev_count is None and act_prev_count is None), f"prev_app_count mismatch for {app_id}"
        
        # Manual Repay
        man_repay = df_repay.filter((pl.col("customer_id") == cust_id) & (pl.col("due_date") < cutoff))
        exp_repay_count = len(man_repay) if len(man_repay) > 0 else None
        act_repay_count = p_row["repay_installment_count"]
        assert (exp_repay_count == act_repay_count) or (exp_repay_count is None and act_repay_count is None), f"repay_count mismatch for {app_id}"
        
        # Manual Bureau
        man_bureau = df_bureau.filter((pl.col("application_id") == app_id) & (pl.col("available_datetime") <= cutoff))
        exp_bureau_provs = len(man_bureau)
        act_bureau_provs = p_row["bureau_provider_count"]
        assert exp_bureau_provs == act_bureau_provs, f"bureau_prov mismatch for {app_id}"
        
        # Manual Ecommerce
        man_ecom = df_ecom.filter((pl.col("customer_id") == cust_id) & (pl.col("transaction_datetime") < cutoff))
        man_orders = man_ecom.select(["order_id", "order_status"]).unique()
        exp_ecom_orders = len(man_orders) if len(man_orders) > 0 else None
        act_ecom_orders = p_row["ecom_total_orders"]
        assert (exp_ecom_orders == act_ecom_orders) or (exp_ecom_orders is None and act_ecom_orders is None), f"ecom_orders mismatch for {app_id}"
        
    print("Random sample assertions passed successfully!")
    
    # 2. Missingness and Distribution Comparison Train vs Test
    print("Comparing missingness and distributions between train and test...")
    feature_cols = [c for c in test_feat.columns if c not in ["application_id", "customer_id", "application_datetime", "decision_datetime"]]
    
    diff_report = []
    for col in feature_cols:
        tr_s = train_feat[col]
        te_s = test_feat[col]
        
        tr_null = tr_s.null_count() / len(train_feat) * 100
        te_null = te_s.null_count() / len(test_feat) * 100
        
        entry = {
            "feature": col,
            "train_null_pct": round(tr_null, 2),
            "test_null_pct": round(te_null, 2),
            "null_diff": round(abs(tr_null - te_null), 2),
        }
        if tr_s.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            tr_valid = tr_s.drop_nulls()
            te_valid = te_s.drop_nulls()
            if len(tr_valid) > 0 and len(te_valid) > 0:
                entry["train_mean"] = round(float(tr_valid.mean()), 4)
                entry["test_mean"] = round(float(te_valid.mean()), 4)
                entry["train_median"] = round(float(tr_valid.median()), 4)
                entry["test_median"] = round(float(te_valid.median()), 4)
        diff_report.append(entry)
        
    with open("reports/train_test_distribution_comparison.json", "w") as f:
        json.dump(diff_report, f, indent=2)
    print("Distribution comparison saved to reports/train_test_distribution_comparison.json")
    
    # 3. Build feature_dictionary.md
    print("Generating reports/feature_dictionary.md...")
    feat_dict_lines = [
        "# Feature Dictionary",
        "",
        "| Nama Fitur | Sumber | Definisi | Window | Alasan / Nilai Bisnis |",
        "|---|---|---|---|---|"
    ]
    
    # Categorize features by source prefix
    for col in feature_cols:
        if col.startswith("prev_"):
            src = "previous_applications"
            win = "Historical (< cutoff)"
            reason = "Track record aplikasi masa lalu, disiplin dan limit histori pinjaman"
            defin = f"Agregasi histori aplikasi sebelumnya ({col.replace('prev_', '')})"
        elif col.startswith("repay_"):
            src = "repayment_history"
            win = "Historical installment (< cutoff)"
            reason = "Kepatuhan angsuran, delinquency behavior, dan DPD track record"
            defin = f"Agregasi histori pembayaran angsuran pinjaman aktif/lalu ({col.replace('repay_', '')})"
        elif col.startswith("acct_"):
            src = "account_behavior"
            win = "3m / 6m / 12m / All-time complete months"
            reason = "Kapasitas finansial, stabilitas saldo, kecukupan likuiditas dan cashflow"
            defin = f"Metrik rekening tabungan bulanan ({col.replace('acct_', '')})"
        elif col.startswith("bureau_"):
            src = "bureau_observations"
            win = "Latest available (< cutoff)"
            reason = "Skor kredit eksternal lintas provider, total utang berjalan, utilisasi kartu kredit"
            defin = f"Observasi credit bureau resmi ({col.replace('bureau_', '')})"
        elif col.startswith("ecom_"):
            src = "ecommerce_transactions"
            win = "30d / 90d / 180d / All completed (< cutoff)"
            reason = "Pola belanja digital, ketergantungan COD, frekuensi dan daya beli"
            defin = f"Agregasi transaksi e-commerce ({col.replace('ecom_', '')})"
        else:
            src = "train_applications + customer_profiles"
            win = "Application snapshot"
            reason = "Profil demografi debitur, rasio utang terhadap pendapatan, beban angsuran"
            defin = f"Atribut aplikasi dan profil debitur ({col})"
            
        feat_dict_lines.append(f"| `{col}` | {src} | {defin} | {win} | {reason} |")
        
    with open("reports/feature_dictionary.md", "w") as f:
        f.write("\n".join(feat_dict_lines) + "\n")
    print("Feature dictionary written to reports/feature_dictionary.md")

if __name__ == "__main__":
    run_stage4_verification()
