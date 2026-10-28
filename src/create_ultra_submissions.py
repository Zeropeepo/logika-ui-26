"""
Generate 3 High-Impact Submissions Targeting 0.5556 -> 0.57+:
1. submission_ULTRA_0556_gru_pop_blend.csv:
   Anchors 90% on the 0.5556 dual files (q4cb + q4cf), boosted by 5% Population Expert, 3% Ecom-GRU, and 2% Tab-ResNet.
2. submission_ULTRA_q4_balanced_champion.csv:
   Directly combines the two 0.5556 files (q4cb + q4cf) with orthogonal Ecom-GRU.
3. submission_POPULATION_DECOUPLED_EXPERT.csv:
   Population-segmented blend applying custom weights for Pop 1 (Full), Pop 2 (Thin), Pop 3 (Rejected).
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
    print("=" * 80)
    print("GENERATING 3 HIGH-IMPACT SUBMISSIONS TARGETING 0.5556 -> 0.57+")
    print("=" * 80)
    
    # 1. Load test data & predictions
    q4cb_df = pl.read_csv("/home/neva/Downloads/ens_q4cb.csv")
    q4cf_df = pl.read_csv("/home/neva/Downloads/ens_q4cf.csv")
    candA_df = pl.read_csv(f"{SUBMISSIONS_DIR}/submission_candidate_A_stacker.csv")
    
    test_gru_df = pl.read_parquet(f"{PREDS_DIR}/test_ecom_gru.parquet")
    test_pop_df = pl.read_parquet(f"{PREDS_DIR}/test_population_experts.parquet")
    test_v4_df  = pl.read_parquet(f"{PREDS_DIR}/test_v4_all_models.parquet")
    te_feat_df  = pl.read_parquet("data/processed/test_features_v4.parquet").select([
        "application_id", "cov_has_prev_apps", "cov_has_repayment", "cov_has_account"
    ])
    
    app_ids = q4cb_df["application_id"].to_list()
    N = len(app_ids)
    
    r_cb   = normalize_rank(q4cb_df["default_probability"].to_numpy())
    r_cf   = normalize_rank(q4cf_df["default_probability"].to_numpy())
    r_cand = normalize_rank(candA_df["default_probability"].to_numpy())
    r_gru  = normalize_rank(test_gru_df["test_gru"].to_numpy())
    r_pop  = normalize_rank(test_pop_df["test_pop_expert"].to_numpy())
    r_tab  = normalize_rank(test_v4_df["test_tab"].to_numpy())
    
    # -------------------------------------------------------------------------
    # VARIANT 1: ULTRA 0556 + GRU + POP EXPERT + TAB-FM BLEND
    # -------------------------------------------------------------------------
    # Anchor 0.5556 pair (45% cb + 43% cf) + 5% pop_expert + 4% gru + 3% tab
    blend_1 = (
        0.45 * r_cb +
        0.43 * r_cf +
        0.05 * r_pop +
        0.04 * r_gru +
        0.03 * r_tab
    )
    p1 = normalize_rank(blend_1)
    
    path_1 = f"{SUBMISSIONS_DIR}/submission_ULTRA_0556_gru_pop_blend.csv"
    pl.DataFrame({"application_id": app_ids, "default_probability": p1.astype(np.float64)}).write_csv(path_1)
    print(f"[1] Saved: {path_1}")
    
    # -------------------------------------------------------------------------
    # VARIANT 2: DUAL 0556 ANCHOR + GRU BOOST
    # -------------------------------------------------------------------------
    # 47% cb + 47% cf + 6% gru
    blend_2 = (
        0.47 * r_cb +
        0.47 * r_cf +
        0.06 * r_gru
    )
    p2 = normalize_rank(blend_2)
    
    path_2 = f"{SUBMISSIONS_DIR}/submission_ULTRA_q4_balanced_champion.csv"
    pl.DataFrame({"application_id": app_ids, "default_probability": p2.astype(np.float64)}).write_csv(path_2)
    print(f"[2] Saved: {path_2}")
    
    # -------------------------------------------------------------------------
    # VARIANT 3: POPULATION-DECOUPLED EXPERT BLEND
    # -------------------------------------------------------------------------
    has_prev = te_feat_df["cov_has_prev_apps"].to_numpy()
    has_repay = te_feat_df["cov_has_repayment"].to_numpy()
    has_acct = te_feat_df["cov_has_account"].to_numpy()
    
    te_pop1 = (has_repay == 1) & (has_acct == 1)
    te_pop2 = (has_prev == 0) & (has_acct == 0)
    te_pop3 = (has_prev == 1) & (has_repay == 0)
    te_pop2 = te_pop2 | ((~te_pop1) & (~te_pop2) & (~te_pop3))
    
    blend_3 = np.zeros(N, dtype=np.float64)
    # Pop 1 (Full 66%): Rely strongly on 0.5556 + Candidate A + Tab-ResNet
    blend_3[te_pop1] = 0.50 * r_cb[te_pop1] + 0.35 * r_cf[te_pop1] + 0.10 * r_cand[te_pop1] + 0.05 * r_tab[te_pop1]
    
    # Pop 2 (Thin 30%): Inject high weight of Pop Expert (clean bureau/ecom) + Ecom-GRU
    blend_3[te_pop2] = 0.50 * r_cb[te_pop2] + 0.30 * r_cf[te_pop2] + 0.12 * r_pop[te_pop2] + 0.08 * r_gru[te_pop2]
    
    # Pop 3 (Rejected 4%): Inject high weight of Ecom-GRU + Pop Expert
    blend_3[te_pop3] = 0.45 * r_cb[te_pop3] + 0.35 * r_cf[te_pop3] + 0.10 * r_pop[te_pop3] + 0.10 * r_gru[te_pop3]
    
    p3 = normalize_rank(blend_3)
    path_3 = f"{SUBMISSIONS_DIR}/submission_POPULATION_DECOUPLED_EXPERT.csv"
    pl.DataFrame({"application_id": app_ids, "default_probability": p3.astype(np.float64)}).write_csv(path_3)
    print(f"[3] Saved: {path_3}")
    
    # Rigorous verification
    for pth in [path_1, path_2, path_3]:
        df = pl.read_csv(pth)
        assert len(df) == 30000
        assert df.columns == ["application_id", "default_probability"]
        assert df["application_id"].to_list() == app_ids
        assert df["default_probability"].null_count() == 0
        assert df["default_probability"].is_finite().all()
        print(f"  Verified {os.path.basename(pth)}: 30,000 rows | Range: [{df['default_probability'].min():.5f}, {df['default_probability'].max():.5f}]")

if __name__ == "__main__":
    main()
