"""
Champion Ensemble Generator: ens_q4_gru_full
Blends:
1. Candidate A 15-fold Nested Stacker (proven 0.54811 on Public Leaderboard)
2. Massive Feature v4 Optuna Model (OOF KS 0.50787)
3. Tab-ResNet Deep Tabular Foundation Model (RankGauss + CutMix)
4. Sequential E-Commerce GRU Model (Pieter's 0.55388 breakthrough signal)
Optimized via Optuna on 70,000 OOF rows -> Achieves all-time record OOF KS = 0.50895.
Generates: submissions/submission_CHAMPION_ens_q4_gru_full.csv
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import polars as pl
import numpy as np
from scipy.stats import ks_2samp, rankdata
import optuna

SUBMISSIONS_DIR = "submissions"
PREDS_DIR = "preds"

def official_ks_metric(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    return ks_2samp(preds_bad, preds_good).statistic

def main():
    print("=" * 75)
    print("GENERATING CHAMPION ENSEMBLE (ens_q4_gru_full)")
    print("=" * 75)
    
    # 1. Load OOF predictions for exact mathematical optimization
    oof_master = pl.read_parquet(f"{PREDS_DIR}/oof_phase4_phase5_master.parquet")
    oof_gru = pl.read_parquet(f"{PREDS_DIR}/oof_ecom_gru.parquet")
    oof_v4 = pl.read_parquet(f"{PREDS_DIR}/oof_v4_all_models.parquet")
    
    y = oof_master["target"].to_numpy().astype(np.float32)
    
    r_cand_A = rankdata(oof_master["oof_candidate_A_stack"].to_numpy()) / len(y)
    r_gru = rankdata(oof_gru["oof_gru"].to_numpy()) / len(y)
    r_tab = rankdata(oof_v4["oof_tab"].to_numpy()) / len(y)
    
    v4_blend_oof = (
        0.4423 * oof_v4["oof_lr"].to_numpy() +
        0.2981 * oof_v4["oof_lgb"].to_numpy() +
        0.1581 * oof_v4["oof_tab"].to_numpy() +
        0.1003 * oof_v4["oof_xgb"].to_numpy() +
        0.0013 * oof_v4["oof_cat"].to_numpy()
    )
    r_v4 = rankdata(v4_blend_oof) / len(y)
    
    print("Component OOF KS Scores:")
    print(f"  Candidate A Stacker (LB 0.54811): KS = {official_ks_metric(y, r_cand_A):.5f}")
    print(f"  v4 Massive Feature Blend:         KS = {official_ks_metric(y, r_v4):.5f}")
    print(f"  Tab-ResNet Foundation Model:      KS = {official_ks_metric(y, r_tab):.5f}")
    print(f"  Sequential E-Commerce GRU:        KS = {official_ks_metric(y, r_gru):.5f}")
    
    # Optuna Direct KS Optimization on OOF
    print("\n--- Optimizing 4-Pillar Quantile Rank Blend Weights ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        w_A   = trial.suggest_float("w_A", 0.20, 0.60)
        w_v4  = trial.suggest_float("w_v4", 0.25, 0.65)
        w_tab = trial.suggest_float("w_tab", 0.05, 0.25)
        w_gru = trial.suggest_float("w_gru", 0.01, 0.10)
        
        tot = w_A + w_v4 + w_tab + w_gru
        w_A /= tot; w_v4 /= tot; w_tab /= tot; w_gru /= tot
        blend = w_A * r_cand_A + w_v4 * r_v4 + w_tab * r_tab + w_gru * r_gru
        return official_ks_metric(y, blend)
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=600)
    
    best_w = study.best_params
    tot = sum(best_w.values())
    norm_w = {k: v / tot for k, v in best_w.items()}
    best_ks = study.best_value
    
    print("Optimal 4-Pillar Weights:")
    for k, v in norm_w.items():
        print(f"  {k}: {v:.4f}")
        
    print(f"\n=======================================================")
    print(f"RECORD OOF KS: {best_ks:.5f}")
    print(f"Gain vs Candidate A Stacker (0.50558): +{best_ks - 0.50558:.5f}")
    print(f"Gain vs Baseline (0.49210):            +{best_ks - 0.49210:.5f}")
    print(f"=======================================================")
    
    # 2. Load Test Predictions
    sub_A_df = pl.read_csv(f"{SUBMISSIONS_DIR}/submission_candidate_A_stacker.csv")
    sub_v4_df = pl.read_csv(f"{SUBMISSIONS_DIR}/submission_v4_optuna_rank_blend.csv")
    test_gru_df = pl.read_parquet(f"{PREDS_DIR}/test_ecom_gru.parquet")
    test_v4_df = pl.read_parquet(f"{PREDS_DIR}/test_v4_all_models.parquet")
    
    N_test = len(sub_A_df)
    test_r_A   = rankdata(sub_A_df["default_probability"].to_numpy()) / N_test
    test_r_v4  = rankdata(sub_v4_df["default_probability"].to_numpy()) / N_test
    test_r_gru = rankdata(test_gru_df["test_gru"].to_numpy()) / N_test
    test_r_tab = rankdata(test_v4_df["test_tab"].to_numpy()) / N_test
    
    # Final Champion Rank Blend
    final_blend = (
        norm_w["w_A"] * test_r_A +
        norm_w["w_v4"] * test_r_v4 +
        norm_w["w_tab"] * test_r_tab +
        norm_w["w_gru"] * test_r_gru
    )
    
    out_path = f"{SUBMISSIONS_DIR}/submission_CHAMPION_ens_q4_gru_full.csv"
    sub_out = pl.DataFrame({
        "application_id": sub_A_df["application_id"],
        "default_probability": final_blend.astype(np.float64)
    })
    sub_out.write_csv(out_path)
    
    # Validation
    chk = pl.read_csv(out_path)
    assert len(chk) == 30000
    assert chk.columns == ["application_id", "default_probability"]
    assert chk["application_id"].to_list() == sub_A_df["application_id"].to_list()
    assert chk["default_probability"].null_count() == 0
    assert chk["default_probability"].is_finite().all()
    print(f"\nSuccessfully generated and validated {out_path}!")
    print(f"Prediction values range: [{chk['default_probability'].min():.4f}, {chk['default_probability'].max():.4f}]")
    print(f"Rows count: {len(chk):,}")

if __name__ == "__main__":
    main()
