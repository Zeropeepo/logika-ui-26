# LAPORAN FASE 2: ADVANCED FEATURE ENGINEERING & MULTI-MODEL BENCHMARK

**Tanggal/Waktu**: 2026-10-12  
**Dataset**: `data/processed/train_features_v2.parquet` (70.000 baris x 297 kolom), `test_features_v2.parquet` (30.000 baris x 296 kolom)  
**Protokol CV**: Repeated Stratified 5-Fold x 3 Seed (`[42, 2026, 999]`), 15 fold identik pada seluruh 70.000 baris train.  
**Metrik Utama**: Two-sample Kolmogorov-Smirnov (`scipy.stats.ks_2samp(y_pred[y==1], y_pred[y==0]).statistic`).  
**Constraint**: 100% lolos uji point-in-time (`<= decision_datetime`).

---

## 1. Ringkasan Eksekutif

Pada Fase 2, kelompok fitur tingkat lanjut diekstrak dari seluruh sumber data yang sah:
1. **B1 (E-commerce mendalam)**: Multi-window (7, 14, 30, 60, 90, 180, 365 hari), rasio momentum (7d/30d, 14d/30d, 30d/90d), jeda antar order (mean, std, max), share COD per window, pembelanjaan kategori produk (groceries, electronics, fashion).
2. **B2 (Bureau multi-titik)**: Memanfaatkan seluruh **797.563 observasi** biro kredit historis (termasuk 533.475 dari `previous_applications`), delta skor latest vs earliest, penurunan skor dari titik puncak, pertumbuhan utang, lonjakan inquiry, disparitas antar biro (A-B, A-C).
3. **B3 (Repayment granular)**: Pelunasan tertimbang recency decay (EWMA DPD), underpaid shortage total, performa K=3 dan K=6 installment terakhir, proporsi pinjaman yang pernah default (DPD 30/60/90).
4. **B4 (Residual suku bunga)**: Nested 5-fold OOF regression memprediksi `annual_interest_rate` dari demografi tanpa target. Residual merepresentasikan *unobservable risk markup* dari risk-based pricing lender.
5. **B5 (Kalender siklikal & cakupan)**: Sin/cos transformasi jam & bulan keputusan, window tanggal gajian (>=25), total sumber data aktif per nasabah.
6. **B6 (Skor risiko komposit laten)**: Rank-average normalisasi [0, 1] dari indikator risiko univariat terkuat dari sumber independen.
7. **B7 (Rasio interaksi & business flags)**: Rasio belanja e-commerce 30d terhadap pendapatan bulanan, rasio utang biro terhadap pendapatan tahunan, flag hunger belanja COD mendadak.

---

## 2. Hasil Evaluasi 15-Fold CV (Fase 2 vs Baseline Fase 1)

Semua model dievaluasi pada folds yang sama persis (`cv_folds_70k.parquet`).

| ID Eksperimen | Model | Fitur | KS CV Mean ± Std | Pairwise Δ KS vs Baseline | Overall OOF KS | Overall OOF AUC | Waktu (detik) | Status |
|---|---|---|---|---|---|---|---|---|
| **EXP-001** | Logistic Regression | 132 (Base) | 0.4921 ± 0.0104 | 0.0000 ± 0.0000 | 0.4893 | 0.8223 | 36.1s | Acuan |
| **EXP-002** | LightGBM | 132 (Base) | 0.4900 ± 0.0074 | 0.0000 ± 0.0000 | 0.4907 | 0.8243 | 69.1s | Acuan |
| **EXP-003** | CatBoost | 132 (Base) | 0.4921 ± 0.0087 | 0.0000 ± 0.0000 | 0.4905 | 0.8249 | 321.2s | Acuan |
| **EXP-004** | LightGBM v2 | 292 fitur | **0.4990 ± 0.0099** | **+0.0089 ± 0.0045** | 0.4991 | 0.8303 | 77.7s | **ADOPTED** |
| **EXP-005** | CatBoost v2 (GPU) | 292 fitur | **0.4982 ± 0.0104** | **+0.0061 ± 0.0039** | 0.4965 | 0.8299 | 247.3s | **ADOPTED** |
| **EXP-006** | XGBoost v2 (GPU) | 292 fitur | **0.4998 ± 0.0088** | **+0.0098 ± 0.0054** | 0.5003 | 0.8306 | 151.6s | **ADOPTED** |
| **EXP-007** | Logistic Regression v2 | 292 fitur | **0.5042 ± 0.0100** | **+0.0121 ± 0.0063** | 0.5011 | 0.8321 | 122.4s | **ADOPTED** |
| **ENSEMBLE** | 4-Model Rank Blend | 292 fitur | **0.5044** | **+0.0144** | **0.5044** | **0.8345** | - | **TOP BENCHMARK** |

### Analisis Pairwise Delta
- **Selisih Positif di Luar Noise**: Semua model menunjukkan pairwise improvement yang konsisten di 15 fold ($p < 0.0001$ via one-sample t-test pada selisih).
- **Linear Dominance**: Logistic Regression mengungguli pohon keputusan tunggal (Mean KS 0.5042 vs LightGBM 0.4990), menegaskan hipotesis bahwa struktur risiko laten pada dataset ini bersifat aditif luas.
- **Ensemble Gain**: Weighted Rank Blend (0.35 LR + 0.25 LGB + 0.20 CB + 0.20 XGB) mencapai OOF KS **0.5044** dan AUC **0.8345**.

---

## 3. Fitur Paling Berpengaruh (Top Feature Importance by LightGBM Gain)

1. `composite_risk_score` (Gain: 55.304,89) — Rank gabungan tertimbang sinyal independen terkuat.
2. `repay_max_max_dpd` (Gain: 15.182,55) — Keterlambatan terparah sepanjang sejarah nasabah.
3. `repay_weighted_underpaid` (Gain: 7.416,21) — Rasio angsuran kurang bayar berbobot peluruhan eksponensial waktu.
4. `ecom_median_order_val` (Gain: 4.856,83) — Nilai median transaksi belanja.
5. `bureau_life_score_mean` (Gain: 4.611,82) — Rata-rata skor biro lintas seluruh laporan biro sepanjang hidup.
6. `ecom_orders_14d` (Gain: 4.516,83) — Frekuensi belanja dalam 14 hari sebelum pengajuan pinjaman.
7. `repay_max_max_dpd_all` (Gain: 3.881,49) — Maksimum DPD lintas seluruh cicilan historis.
8. `ecom_spend_groceries` (Gain: 3.477,26) — Pembelanjaan bahan pokok/sembako.
9. `bureau_life_score_max` (Gain: 3.366,08) — Skor biro tertinggi yang pernah diraih nasabah.
10. `ecom_spend_60d` (Gain: 3.304,35) — Total belanja e-commerce 60 hari terakhir.

---

## 4. Matriks Korelasi Rank Prediksi OOF Antar Model

```
           LGB        CB       XGB        LR
LGB     1.0000    0.9814    0.9944    0.9458
CB      0.9814    1.0000    0.9815    0.9477
XGB     0.9944    0.9815    1.0000    0.9447
LR      0.9458    0.9477    0.9447    1.0000
```
- Hubungan antar tree models (LGB, CB, XGB) berkorelasi tinggi (0.981 - 0.994).
- Model linear (LR) memberikan keragaman nyata (korelasi ~0.945 terhadap pohon), menjadi sumber diversifikasi utama pada ensemble.

---

## 5. Keputusan & Langkah Berikutnya (Fase 3 & 4)

- **Status Fase 2**: Selesai dan sukses (OOF KS naik dari 0.490 ke 0.5044).
- **Peluang Lonjakan ke Target KS > 0.56**:
  1. **Fase 3 (Feature Selection & Denoising)**: Banyak fitur memiliki korelasi Spearman > 0.95. Pruning fitur redundan dan membangun representasi klaster non-redundant (< 100 fitur).
  2. **Fase 4 (Higher-Order Interactions & Hyperparameter Tuning)**:
     - Nonlinear risk ratios (e.g. `ecom_spend_30d / (declared_income - proposed_installment)`).
     - Target Encoding bernois / Out-of-Fold Bayesian encoding.
     - Tuning arsitektur LightGBM, CatBoost, dan Tabular Neural Network / MLP.
     - Stacking non-linear dengan logistic / spline meta-learner.
