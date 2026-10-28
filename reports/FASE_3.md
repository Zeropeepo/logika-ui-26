# LAPORAN FASE 3: SELEKSI FITUR, ABLASI KELOMPOK & KLASTER KORELASI

**Tanggal/Waktu**: 2026-10-12  
**Dataset**: `data/processed/train_features_v2.parquet` (297 kolom) vs `train_features_selected.parquet` (134 kolom)  
**Protokol CV**: Repeated Stratified 5-Fold x 3 Seed (`[42, 2026, 999]`), 15 fold identik pada 70.000 baris.  
**Metrik Utama**: Two-sample Kolmogorov-Smirnov (`scipy.stats.ks_2samp(y_pred[y==1], y_pred[y==0]).statistic`).

---

## 1. Analisis Ablasi Kelompok Konstruk (Group Drop Ablation)

Untuk membedah kontribusi tiap pilar informasi tanpa terdistorsi multikolinearitas antar fitur dalam kelompok yang sama, dilakukan eksperimen drop-group secara terisolasi pada model LightGBM:

| Kelompok Fitur | Jumlah Fitur | OOF KS Saat Grup Dibuang | Δ KS vs Model Penuh (0.4940) | Dampak & Signifikansi Bisnis |
|---|---|---|---|---|
| **E-Commerce** | 97 kolom | **0.4629** | **-0.0311** | **Pilar Utama #1**: Pembelanjaan, frekuensi order, tren recency, dan rasio COD menyumbang lebih dari 3,1% pemisahan KS mutlak. |
| **Repayment History** | 52 kolom | **0.4762** | **-0.0178** | **Pilar Utama #2**: Maksimum DPD, underpayment shortage, dan on-time ratio menyumbang 1,78% pemisahan KS. |
| **Bureau Observations** | 58 kolom | **0.4859** | **-0.0081** | **Pilar #3**: Skor biro kredit multi-titik dan disparitas antar provider memberikan kontribusi 0,81% KS. |
| **Customer Profile** | 7 kolom | **0.4928** | **-0.0012** | Demografi (usia, income, pekerjaan, status rumah) memiliki efek marginal minor. |
| **Pricing & Loan Request** | 9 kolom | **0.4940** | **0.0000** | Karakteristik pinjaman dan suku bunga sebagian besar telah tercakup dalam residual bunga. |
| **Composite & Flags** | 19 kolom | **0.4948** | **+0.0008** | Interaksi rasio dan flag tanpa komponen dasar sedikit netral/redundant bagi pohon keputusan. |

---

## 2. Hierarchical Clustering Korelasi Spearman & Reduksi Fitur

1. **Klasterisasi Hierarki**:
   - Menghitung matriks korelasi rank Spearman (|r|) pada seluruh 288 fitur numerik.
   - Menggunakan jarak $d = 1 - |r|$ dengan average linkage pada threshold $d = 0.20$ ($|r| \ge 0.80$).
   - Terbentuk **124 klaster independen**.
2. **Pemilihan Perwakilan (Representative Feature)**:
   - Dari setiap klaster, dipilih 1 fitur dengan univariat KS tertinggi dan integritas data terbaik.
   - Ditambah 4 fitur kategorikal dan fitur vital (`composite_risk_score`, `repay_max_max_dpd`, `annual_interest_rate`, `ecom_spend_30d`, `ecom_orders_30d`, `bureau_life_score_latest`, `pricing_rate_residual`).
   - Menghasilkan set fitur ringkas: **132 fitur** (tersimpan di `data/processed/train_features_selected.parquet`).

---

## 3. Perbandingan Model: Set Penuh (292 Fitur) vs Set Terseleksi (132 Fitur)

| Model | Set Fitur | Mean KS 15-Fold | Overall OOF KS | Overall OOF AUC | Waktu (detik) | Kesimpulan |
|---|---|---|---|---|---|---|
| **LightGBM** | 292 Fitur (Full v2) | **0.4990 ± 0.0099** | **0.4991** | **0.8303** | 77.7s | **UNGGUL**: Gradient boosting memanfaatkan variasi multi-window halus. |
| **LightGBM** | 132 Fitur (Selected) | 0.4925 ± 0.0095 | 0.4939 | 0.8270 | 46.4s | Cepat namun kehilangan ~0.005 KS dari granularitas multi-window. |
| **Logistic Regression** | 292 Fitur (Full v2) | **0.5042 ± 0.0100** | **0.5011** | **0.8321** | 122.4s | **TOP LINEAR**: Regularisasi L2 menangkap pooling aditif seluruh fitur. |
| **Logistic Regression** | 128 Fitur Numerik (Selected) | 0.4864 (OOF) | 0.4864 | 0.8250 | 42.3s | Kehilangan informasi pooling dari variasi multi-titik. |

---

## 4. Keputusan Arsitektur untuk Fase 4 & 5

1. **Retensi Fitur untuk Tree Models & Linear Stacking**:
   - Mempertahankan set penuh 292 fitur untuk training final XGBoost, CatBoost, LightGBM, dan Logistic Regression, karena terbukti memberikan performa tertinggi (Mean KS 0.5042 vs 0.4925).
2. **Utilitas Set 132 Fitur**:
   - Dipakai sebagai input representatif efisien untuk model fondasi tabular / neural networks (TabM / MLP / TabPFN) yang sensitif terhadap dimensionality dan komputasi memori O(N x D).
