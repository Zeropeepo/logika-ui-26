"""
Generate Final Production Submissions Integrating exp067 Kumo-Tabular Population Foundation Model:
1. submission_ULTRA_exp067_kumo_population_champion.csv:
   Population-decoupled blend injecting Kumo-Tabular and Ecom-GRU heavily into Thin-File (Pop 2) and Rejected (Pop 3).
2. submission_ULTRA_kumo_gru_0556_power_blend.csv:
   Global calibrated power blend anchoring on 0.5556 dual files + Kumo-Tabular + Ecom-GRU.
3. submission_PURE_exp067_kumo_foundation_expert.csv:
   Decoupled Foundation (Kumo) + Pop Expert GBDT + Ecom-GRU + Stacker.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import polars as pl
import numpy as np
from scipy.stats import rankdata

SUBMISSIONS_DIR = "submissions"
PREDS_DIR = "preds"

def normalize_rank(arr):
    return (rankdata(arr) - 0.5) / len(arr)

def main():
    print("=" * 85)
    print("GENERATING FINAL PRODUCTION SUBMISSIONS (TARGETING 0.5556 -> 0.57+)")
    print("=" * 85)
    
    # 1. Load predictions
    q4cb_df = pl.read_csv("/home/neva/Downloads/ens_q4cb.csv")
    q4cf_df = pl.read_csv("/home/neva/Downloads/ens_q4cf.csv")
    candA_df = pl.read_csv(f"{SUBMISSIONS_DIR}/submission_candidate_A_stacker.csv")
    
    test_kumo_df = pl.read_parquet(f"{PREDS_DIR}/test_kumo_population.parquet")
    test_gru_df  = pl.read_parquet(f"{PREDS_DIR}/test_ecom_gru.parquet")
    test_pop_df  = pl.read_parquet(f"{PREDS_DIR}/test_population_experts.parquet")
    test_v4_df   = pl.read_parquet(f"{PREDS_DIR}/test_v4_all_models.parquet")
    
    te_feat_df = pl.read_parquet("data/processed/test_features_v4.parquet").select([
        "application_id", "cov_has_prev_apps", "cov_has_repayment", "cov_has_account"
    ])
    
    app_ids = q4cb_df["application_id"].to_list()
    N = len(app_ids)
    
    # Normal rank arrays
    r_cb   = normalize_rank(q4cb_df["default_probability"].to_numpy())
    r_cf   = normalize_rank(q4cf_df["default_probability"].to_numpy())
    r_cand = normalize_rank(candA_df["default_probability"].to_numpy())
    r_kumo = normalize_rank(test_kumo_df["test_kumo"].to_numpy())
    r_gru  = normalize_rank(test_gru_df["test_gru"].to_numpy())
    r_pop  = normalize_rank(test_pop_df["test_pop_expert"].to_numpy())
    r_tab  = normalize_rank(test_v4_df["test_tab"].to_numpy())
    
    # 0.5556 combined anchor
    r_0556 = 0.5 * r_cb + 0.5 * r_cf
    
    # Populations
    has_prev = te_feat_df["cov_has_prev_apps"].to_numpy()
    has_repay = te_feat_df["cov_has_repayment"].to_numpy()
    has_acct = te_feat_df["cov_has_account"].to_numpy()
    
    p1 = (has_repay == 1) & (has_acct == 1)
    p3 = (has_prev == 1) & (has_repay == 0)
    p2 = (~p1) & (~p3)
    
    print(f"Test Populations: Pop1={p1.sum():,}, Pop2={p2.sum():,}, Pop3={p3.sum():,}")
    
    # -------------------------------------------------------------------------
    # SUBMISSION 1: ULTRA exp067 Kumo Population Champion (Target: > 0.5700)
    # -------------------------------------------------------------------------
    blend_1 = np.zeros(N, dtype=np.float64)
    # Pop 1 (Full 66%): Anchor 70% + 15% Kumo + 10% PopExpert + 5% TabResNet
    blend_1[p1] = 0.70 * r_0556[p1] + 0.15 * r_kumo[p1] + 0.10 * r_pop[p1] + 0.05 * r_tab[p1]
    
    # Pop 2 (Thin 30%): Anchor 55% + 22% Kumo + 13% Ecom-GRU + 10% PopExpert
    blend_1[p2] = 0.55 * r_0556[p2] + 0.22 * r_kumo[p2] + 0.13 * r_gru[p2] + 0.10 * r_pop[p2]
    
    # Pop 3 (Rejected 4%): Anchor 55% + 22% Kumo + 13% Ecom-GRU + 10% TabResNet
    blend_1[p3] = 0.55 * r_0556[p3] + 0.22 * r_kumo[p3] + 0.13 * r_gru[p3] + 0.10 * r_tab[p3]
    
    sub_1_probs = normalize_rank(blend_1)
    path_1 = f"{SUBMISSIONS_DIR}/submission_ULTRA_exp067_kumo_population_champion.csv"
    pl.DataFrame({"application_id": app_ids, "default_probability": sub_1_probs.astype(np.float64)}).write_csv(path_1)
    print(f"\n[1] Generated: {path_1}")
    
    # -------------------------------------------------------------------------
    # SUBMISSION 2: ULTRA Kumo + GRU + 0.5556 Global Power Blend
    # -------------------------------------------------------------------------
    # 42% cb + 42% cf + 8% kumo + 5% gru + 3% pop
    blend_2 = (
        0.42 * r_cb +
        0.42 * r_cf +
        0.08 * r_kumo +
        0.05 * r_gru +
        0.03 * r_pop
    )
    sub_2_probs = normalize_rank(blend_2)
    path_2 = f"{SUBMISSIONS_DIR}/submission_ULTRA_kumo_gru_0556_power_blend.csv"
    pl.DataFrame({"application_id": app_ids, "default_probability": sub_2_probs.astype(np.float64)}).write_csv(path_2)
    print(f"[2] Generated: {path_2}")
    
    # -------------------------------------------------------------------------
    # SUBMISSION 3: PURE exp067 Kumo Foundation + Pop Expert (Autonomous)
    # -------------------------------------------------------------------------
    # Using our CV-optimized weights per population:
    blend_3 = np.zeros(N, dtype=np.float64)
    blend_3[p1] = 0.36 * r_cand[p1] + 0.25 * r_pop[p1] + 0.19 * r_kumo[p1] + 0.10 * r_tab[p1] + 0.10 * r_gru[p1]
    blend_3[p2] = 0.35 * r_cand[p2] + 0.25 * r_pop[p2] + 0.20 * r_kumo[p2] + 0.10 * r_gru[p2] + 0.10 * r_tab[p2]
    blend_3[p3] = 0.32 * r_cand[p3] + 0.23 * r_kumo[p3] + 0.22 * r_pop[p3] + 0.13 * r_gru[p3] + 0.10 * r_tab[p3]
    
    sub_3_probs = normalize_rank(blend_3)
    path_3 = f"{SUBMISSIONS_DIR}/submission_PURE_exp067_kumo_foundation_expert.csv"
    pl.DataFrame({"application_id": app_ids, "default_probability": sub_3_probs.astype(np.float64)}).write_csv(path_3)
    print(f"[3] Generated: {path_3}")
    
    # -------------------------------------------------------------------------
    # RIGOROUS VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("SUBMISSION VERIFICATION & SANITY CHECKS:")
    print("=" * 85)
    for pth in [path_1, path_2, path_3]:
        df = pl.read_csv(pth)
        assert len(df) == 30000, f"Row count mismatch in {pth}: {len(df)}"
        assert df.columns == ["application_id", "default_probability"], f"Columns mismatch in {pth}"
        assert df["application_id"].to_list() == app_ids, f"ID order mismatch in {pth}"
        assert df["default_probability"].null_count() == 0, f"Nulls found in {pth}"
        assert df["default_probability"].is_finite().all(), f"Non-finite values found in {pth}"
        p_min = df["default_probability"].min()
        p_max = df["default_probability"].max()
        p_mean = df["default_probability"].mean()
        assert 0.0 <= p_min < p_max <= 1.0, f"Invalid probability bounds in {pth}"
        print(f"  [OK] {os.path.basename(pth):<55} | 30,000 rows | Range: [{p_min:.6f}, {p_max:.6f}] | Mean: {p_mean:.6f}")
    print("=" * 85)

if __name__ == "__main__":
    main()
