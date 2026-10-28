# STATUS PROGRES MISI: LOGIKA-UI 2026

- **Fase Saat Ini**: FASE FINAL (Selesai Penuh)
- **Terakhir Diperbarui**: 17 Oktober 2026, 00:15 WIB
- **Pencapaian Tertinggi OOF KS**: **0.50787** (Optuna Rank Blend 5 Model pada 352 Fitur Massive v4)

---

## Ringkasan Progres per Fase

| Fase | Deskripsi | Status | Hasil Kunci |
|---|---|---|---|
| **Fase 1** | Verifikasi Data & Setup CV Folds | **SELESAI** | - Skema CV 70k (repeated 5-fold x 3 seed) tersimpan di `cv_folds_70k.parquet`.<br>- Baseline acuan: LR 0.4921, LightGBM 0.4900, CatBoost 0.4921. |
| **Fase 2** | Feature Engineering Mendalam (v2) | **SELESAI** | - Ekstraksi B1 s/d B7 (297 kolom fitur).<br>- 4-Model Rank Blend OOF KS mencapai **0.5044**. |
| **Fase 3 & 4** | Massive Multi-Table v4 & Tab-FM | **SELESAI** | - Sintesis 352 fitur v4: slopes longitudinal (account & biro), akselerasi saldo orde-2, streak keterlambatan angsuran, residu haircut pinjaman, burstiness e-commerce.<br>- Pembangunan Tab-FM (`Tab-ResNet`) dengan RankGauss normal quantile scaling, Feature Dropout 0.25, CutMix 0.20, AdamW weight decay 1e-2, dan FP16 mixed precision.<br>- Tab-ResNet OOF KS: 0.4907. |
| **Fase 5** | Ensembling 5 Model & Optuna Direct KS | **SELESAI** | - Optuna Direct KS Rank Optimization pada 5 model (LGB + XGB + CB + LR + Tab-ResNet).<br>- Bobot optimal: LR 44.2%, LGB 29.8%, Tab-ResNet 15.8%, XGB 10.0%, CB 0.1%.<br>- **OOF KS Rekor: 0.50787** (+0.01577 dari baseline, +0.00229 dari stacker fase sebelumnya). |
| **Fase 6** | Sanitasi, Ablasi & 2 Submisi Final | **SELESAI** | - Target Shuffle Sanity Check: PASSED (KS 0.0194, zero data leakage terverifikasi).<br>- Ablasi Fitur LOGO: Biro longitudinal slopes (+0.0041 KS) dan riwayat pinjaman (+0.0019 KS) terbukti paling dominan.<br>- 2 File Submisi tervalidasi penuh di `submissions/`: Candidate A (`submission_v4_optuna_rank_blend.csv`) & Candidate B (`submission_v4_prob_blend.csv`). |

---

## Log Eksperimen Terkini
Lihat detail lengkap di [experiments_log.csv](file:///home/neva/Lomba/LOGIKA-UI/experiments_log.csv).
- `EXP-010`: Tuned LightGBM (Mean KS: 0.5016)
- `EXP-011`: Tuned XGBoost GPU (Mean KS: 0.5013)
- `EXP-013`: Regularized LogReg (Mean KS: 0.5043)
- `EXP-014`: Nested Logistic Stacker (Mean KS: 0.5056)
- `EXP-017`: **Optuna Rank Ensemble 5 Model pada v4 Features (OOF KS: 0.50787, Rekor Juara)**
