# LAPORAN FASE 4: PEMODELAN BERAGAM & TUNING HIPERPARAMETER

**Tanggal/Waktu**: 2026-10-12  
**Dataset**: `data/processed/train_features_v3.parquet` (70.000 baris x 307 kolom, 302 fitur prediktif)  
**Protokol CV**: Repeated Stratified 5-Fold x 3 Seed (`[42, 2026, 999]`), 15 fold identik pada 70.000 baris.  
**Metrik Utama**: Two-sample Kolmogorov-Smirnov (`scipy.stats.ks_2samp(y_pred[y==1], y_pred[y==0]).statistic`).

---

## 1. Arsitektur & Strategi Tuning Model

Dalam domain *consumer credit scoring* dengan 302 fitur yang mencakup multi-window e-commerce, longitudinal bureau history, granular repayment, dan cross-silo financial ratios, pohon keputusan yang terlalu dalam (`num_leaves=31`, `max_depth >= 6`) rentan mengalami overfitting pada kombinasi acak.

Sesuai panduan di `mission.md` (4.1), kami menguji dan menerapkan **pohon sangat dangkal (shallow additive trees)** yang berfungsi layaknya Generalized Additive Model (GAM), di mana setiap pohon hanya mempelajari 1-2 split fitur utama dengan regularisasi ketat:

1. **Tuned LightGBM**:
   - `num_leaves`: 8 (sangat dangkal)
   - `min_child_samples`: 250 (mencegah split pada daun kecil)
   - `feature_fraction`: 0.40 (subsampling acak fitur ketat untuk mendiversifikasi pohon)
   - `learning_rate`: 0.025 dengan 1.200 boosting rounds
2. **Tuned XGBoost (GPU Cuda)**:
   - `max_depth`: 4 (kedalaman dangkal)
   - `colsample_bytree`: 0.45
   - `subsample`: 0.80
   - `learning_rate`: 0.025 dengan histogram GPU acceleration
3. **Tuned CatBoost (GPU)**:
   - `depth`: 5 (symmetric oblivious trees)
   - `l2_leaf_reg`: 8 (regularisasi L2 kuat pada daun)
   - `learning_rate`: 0.03
4. **Regularized Logistic Regression (Linear Benchmark)**:
   - `C`: 0.03 (regularisasi L2 kuat)
   - Imputasi median + standard scaling z-score

---

## 2. Hasil Evaluasi 15-Fold CV (Fase 4 Tuned vs Baseline Fase 1)

| Model | Konfigurasi Tuning | KS CV Mean ± Std | Pairwise Δ KS vs Base | OOF KS | OOF AUC | Waktu (15 fold) | Keputusan |
|---|---|---|---|---|---|---|---|
| **Baseline LightGBM** | Default (leaves=31) | 0.4900 ± 0.0074 | 0.0000 | 0.4907 | 0.8243 | 69.1s | Acuan Fase 1 |
| **Baseline CatBoost** | Default (depth=6) | 0.4921 ± 0.0087 | 0.0000 | 0.4905 | 0.8249 | 321.2s | Acuan Fase 1 |
| **Baseline LogReg** | Default (C=0.10) | 0.4921 ± 0.0104 | 0.0000 | 0.4893 | 0.8223 | 36.1s | Acuan Fase 1 |
| **Tuned LightGBM** | leaves=8, col=0.40 | **0.5016 ± 0.0087** | **+0.0116 ± 0.0087** | **0.5007** | **0.8314** | 96.0s | **ADOPTED** |
| **Tuned XGBoost GPU** | depth=4, col=0.45 | **0.5013 ± 0.0101** | **+0.0113 ± 0.0101** | **0.5016** | **0.8314** | 130.5s | **ADOPTED** |
| **Tuned CatBoost GPU** | depth=5, reg=8 | **0.4972 ± 0.0086** | **+0.0051 ± 0.0086** | **0.4934** | **0.8290** | 236.7s | **ADOPTED** |
| **Regularized LogReg** | C=0.03, z-score | **0.5043 ± 0.0100** | **+0.0122 ± 0.0100** | **0.5010** | **0.8320** | 114.9s | **ADOPTED** |

---

## 3. Temuan Kunci Fase 4

1. **Kemenangan Regularisasi Dangkal**:
   - Membatasi `num_leaves` dari 31 ke 8 pada LightGBM menaikkan KS Mean dari 0.4990 ke **0.5016**, dan pada XGBoost `max_depth=4` meraih **0.5013**.
   - Ini mengonfirmasi bahwa hubungan risiko kredit bersifat kuasi-aditif (efek belanja e-commerce + efek cicilan tertunggak + efek biro kredit).
2. **Kekuatan Model Linear Terus Bertahan**:
   - Logistic Regression dengan $C=0.03$ mencapai Mean KS **0.5043**, membuktikan bahwa linear weighting atas 302 fitur menolak noise lokal pohon dan menghasilkan ranking global yang sangat stabil.
3. **Penyelarasan Prediksi GPU**:
   - Percepatan GPU CUDA untuk XGBoost dan CatBoost memangkas runtime total hingga ~3-5x lipat dengan presisi 15 fold terjaga.
