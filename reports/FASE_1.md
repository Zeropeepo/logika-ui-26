# LAPORAN FASE 1: VERIFIKASI DAN INFRASTRUKTUR

**Kompetisi**: Data Science Competition LOGIKA UI 2026 (Credit Risk Scoring)  
**Tanggal**: 12 Oktober 2026  
**Status**: SELESAI (Menunggu instruksi untuk FASE 2)

---

## 1. TEMUAN KUNCI & HASIL VERIFIKASI (A1 – A6)

### A1. Rekalkulasi AUC OOF & Penyelesaian Misteri Nilai Identik 0.8204
- **Investigasi**: Nilai 0.8204 pada B1, B2, B3 di laporan baseline bukan bug array/shadowing, melainkan **artefak pembulatan 4 desimal**.
- **Hasil Presisi Tinggi (Floating Point)**:
  * **Logistic Regression**: AUC = `0.8204097` (round 4: 0.8204)
  * **LightGBM**: AUC = `0.8204325` (round 4: 0.8204)
  * **CatBoost**: AUC = `0.8204381` (round 4: 0.8204)
- **Koreksi**: Mulai Fase 1, metrik dilaporkan dengan presisi penuh dan fold statistics (mean ± std).

---

### A2. Audit Biro Kredit dari Aplikasi Historis (Peluang Emas Fase 2)
- **Fakta Data**:
  * Total observasi di `bureau_observations.csv`: **797.563 baris**.
  * Observasi yang matched langsung ke target `application_id`: **264.088 baris**.
  * Observasi milik `previous_applications`: **533.475 baris** (belum pernah dipakai di 132 fitur baseline!).
  * Pengujian Cutoff Temporal pada 533.475 observasi ini: **0 baris melanggar cutoff** (`available_datetime <= decision_datetime` 100% aman).
- **Keputusan**: Pada Fase 2, 533.475 observasi biro lama ini akan di-pool ke nasabah untuk membuat fitur **multi-titik** (tren skor biro dari waktu ke waktu, delta baki debet historis, laju penambahan inquiry).

---

### A3. Tabel Cakupan Sumber Data (Coverage Table)
| Sumber Data | Cakupan Train (%) | Cakupan Test (%) | Default Rate (Ada Data) | Default Rate (Tanpa Data) | $\Delta$ Default Rate | Status / Tipe Missing |
|---|---|---|---|---|---|---|
| `previous_applications` | **70.00%** | **70.00%** | 14.38% | 11.72% | +2.66% | Nasabah baru vs lama |
| `repayment_history` | **66.16%** | **66.15%** | 14.31% | 12.15% | +2.16% | Pernah punya pinjaman disetujui |
| `account_behavior` | **70.00%** | **70.00%** | 14.38% | 11.72% | +2.66% | Persis nasabah yang punya rekening |
| `bureau_Bureau_A` | **100.00%** | **100.00%** | 13.58% | - | - | Universal provider |
| `bureau_Bureau_B` | **94.10%** | **94.27%** | 13.74% | 11.01% | +2.73% | Missing coverage antar provider |
| `bureau_Bureau_C` | **94.21%** | **94.18%** | 13.66% | 12.35% | +1.31% | Missing coverage antar provider |
| `ecommerce_transactions` | **100.00%** | **100.00%** | 13.58% | - | - | Universal footprint |

*Catatan penting*: Distribusi cakupan data train vs test presisi identik (70.0% vs 70.0%, 66.16% vs 66.15%), membuktikan tidak ada bias seleksi data antara train dan test.

---

### A4. Pola Kalender Keputusan Kredit
- **Bulan Origination**: Rentang default rate 11.98% (Mei 2025) s/d 14.66% (Februari 2025).
- **Jam Keputusan**: Jam 14:00 (14.81%) & Jam 18:00 (14.46%) vs Jam 20:00 (12.90%).
- **Hari dalam Minggu**: Senin 13.16% vs Selasa 13.87% (variasi sangat kecil ~0.7%).
- **Tanggal dalam Bulan**: Rentang 11.71% (Tgl 6) s/d 15.11% (Tgl 3). Fitur kalender memiliki sinyal tipis, dapat dijadikan fitur sekunder.

---

### A5. Audit Mendalam `account_behavior` & Keputusan
- **Sinyal Univariat**:
  * `months_count`: KS = 0.0476, AUC = 0.5223
  * `mean_balance`: KS = 0.0477, AUC = 0.5214
  * `mean_income_credits`: KS = 0.0489, AUC = 0.5269
  * `total_low_balance`: KS = 0.0190, AUC = 0.5090
  * `income_like_credits / declared_income`: KS = 0.0477, AUC = 0.5165
- **Hasil Ablasi**: Penghapusan seluruh tabel `account_behavior` hanya menurunkan KS sebesar **-0.0009**.
- **Keputusan Tegas**: `account_behavior` **bukan pendorong utama prediktif**. Jangan buang waktu membuat puluhan varian fitur rumit dari tabel ini. Cukup pertahankan metrik likuiditas dasar dan alihkan fokus ke E-Commerce dan Biro Kredit.

---

### A6. Arah Hubungan E-Commerce & Mekanisme Bisnis
Binning kuantil membuktikan hubungan monotonik yang sangat kuat terhadap target:
1. **`ecom_spending_30d`**:
   - Bin 1 (Rp 0 – 237rb): Default Rate = **6.08%**
   - Bin 2 (Rp 237rb – 823rb): Default Rate = **11.57%**
   - Bin 3 (Rp 823rb – 2,19jt): Default Rate = **16.59%**
   - Bin 4 (Rp 2,19jt – 46,2jt): Default Rate = **27.60%** (Lonjakan 4.5x lipat!)
2. **`ecom_orders_30d`**:
   - $\le 1$ order: Default Rate = **6.66%**
   - $> 4$ order: Default Rate = **28.10%**
3. **`ecom_recency_days`**:
   - Recency 1–2 hari (baru saja belanja): Default Rate = **25.41%**
   - Recency > 40 hari (tidak belanja): Default Rate = **4.49%**
4. **`ecom_cod_share`**:
   - $\le 33\%$ COD: Default Rate = **11.52%**
   - $> 71\%$ COD: Default Rate = **16.00%**

**Penjelasan Mekanisme Bisnis (Credit Hunger & Liquidity Strain)**:
Dalam fintech lending konsumer Indonesia, nasabah yang berbelanja intensif sesaat menjelang mengajukan pinjaman tunai tanpa agunan mengalami fenomena **"Credit Hunger / Consumption Distress"**. Mereka menggunakan pinjaman untuk menutupi defisit kas akibat lonjakan belanja jangka pendek. Sebaliknya, nasabah yang belanja e-commerce-nya dorman (> 40 hari) memiliki disiplin pengeluaran yang jauh lebih stabil (default rate hanya 4.5%). Ketergantungan tinggi pada COD (> 70%) mencerminkan nasabah *unbanked/cash-dependent* dengan likuiditas digital minim.

---

## 2. SKEMA EVALUASI BARU & BASELINE ACUAN 70.000 BARIS (A7)

### Definisi Skema CV Resmi
- **Skema**: Repeated Stratified 5-Fold Cross Validation x 3 Seed (`[42, 2026, 999]`) pada **seluruh 70.000 baris train**.
- **Folds Disimpan**: [cv_folds_70k.parquet](file:///home/neva/Lomba/LOGIKA-UI/data/processed/cv_folds_70k.parquet) (seluruh model selanjutnya wajib memakai partisi ini).
- **Holdout 14.000**: Hanya digunakan di akhir (Fase 6) untuk konfirmasi ensemble final.
- **Prediksi OOF**: Disimpan di [oof_phase1_baselines.parquet](file:///home/neva/Lomba/LOGIKA-UI/preds/oof_phase1_baselines.parquet).

### Tabel Hasil Acuan Baseline Resmi (Fase 1)
| Model | Fold KS Mean ± Std (15 Folds) | OOF KS (Pooled 3 Seeds) | Seed-to-Seed Stability (Seed 42 / 2026 / 999) | OOF AUC | Selisih Berpasangan vs LR (Mean ± Std) | Waktu Komputasi |
|---|---|---|---|---|---|---|
| **Logistic Regression (B1)** | **0.4921 ± 0.0104** | 0.4893 | 0.4893 / 0.4897 / 0.4895 | 0.8223 | Baseline referensi | 36.1s |
| **LightGBM (B2)** | **0.4900 ± 0.0074** | 0.4907 | 0.4889 / 0.4885 / 0.4863 | 0.8243 | $-0.0021 \pm 0.0059$ | 69.1s |
| **CatBoost (B3)** | **0.4921 ± 0.0087** | 0.4905 | 0.4886 / 0.4886 / 0.4888 | 0.8249 | $+0.0000 \pm 0.0064$ | 321.2s |

*Pairwise CatBoost vs LightGBM*: $+0.0021 \pm 0.0031$ KS per fold.

---

## 3. EXPERIMENT LOG & ARTIFAK
- Log eksperimen: [experiments_log.csv](file:///home/neva/Lomba/LOGIKA-UI/experiments_log.csv) (EXP-001, EXP-002, EXP-003 tercatat).
- Summary JSON: [phase1_investigation_results.json](file:///home/neva/Lomba/LOGIKA-UI/reports/phase1_investigation_results.json) & [phase1_a7_eval_results.json](file:///home/neva/Lomba/LOGIKA-UI/reports/phase1_a7_eval_results.json).
- Lingkungan & hardware: [ENV.md](file:///home/neva/Lomba/LOGIKA-UI/reports/ENV.md).

---

## 4. KEPUTUSAN UNTUK FASE 2 (FEATURE ENGINEERING)
1. **Fokus Utama**:
   - **E-Commerce Multi-Window (B1)**: Window 7d, 14d, 60d, rasio 7d/30d dan 14d/90d, interaksi COD x belanja, jeda antar order.
   - **Bureau Multi-Titik (B2)**: Ekstraksi 533.475 observasi lama untuk menghitung delta skor biro (skor terbaru vs awal), tren penambahan inquiry, dan disparitas skor antar provider.
   - **Repayment Granular (B3)**: EWMA DPD, K installment terakhir, rasio cicilan tertunggak berbobot recency.
   - **Residual Bunga (B4)**: Prediksi suku bunga dari profil nasabah, ambil residual sebagai proxy skor internal lender.
2. **Deprioritaskan**:
   - `account_behavior` fitur rumit (terbukti sinyal lemah KS < 0.05).
   - Kalender tanggal/hari yang berdiri sendiri (variasi < 1%).
