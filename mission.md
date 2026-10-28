# MISI: MAKSIMALKAN KS DI LOMBA KREDIT (INDONESIA), TANPA PELANGGARAN

Aturan permanen di `.kaggle-credit-risk-rules.md` berlaku penuh. Jika ada konflik, aturan itu menang.

## 0. PERAN DAN GOAL

Kamu adalah senior data scientist credit risk. Pipeline fitur dan baseline sudah ada. Misimu sekarang: menaikkan KS secara nyata dan tervalidasi, lewat eksperimen yang luas (termasuk ide tidak lazim) dengan sumber informasi tetap bersih.

**Goal utama**
Naikkan KS OOF (repeated stratified CV pada seluruh 70.000 baris train) di atas baseline dengan selisih berpasangan positif di luar noise. Hasil akhirnya dua file submission kandidat yang sah dan laporan lengkap.

**Kriteria sukses**
1. Semua eksperimen tercatat, bisa direproduksi, dan lolos guard point-in-time.
2. Setiap klaim perbaikan didukung selisih berpasangan (mean dan std per fold) di luar noise.
3. Ada kandidat final (ensemble) dengan KS CV mean lebih tinggi dari baseline dan stabil antar fold.
4. Ada daftar fitur paling berpengaruh dan paling representatif, beserta penjelasan bisnis/mekanismenya.
5. Semua celah data yang ditemukan tetapi sengaja tidak dipakai tercatat.

**Ekspektasi realistis (bukan target keras)**
Fitur lebih penting daripada model. Ensemble LR+GBM+CatBoost biasanya hanya menambah sekitar +0.005 sampai +0.015 KS. Kenaikan lebih besar kemungkinan datang dari fitur baru (e-commerce, bureau multi-titik, repayment granular) dan keragaman model (model fondasi tabular, NN). Jangan memaksakan angka tertentu. Laporkan apa adanya.

## 1. KONTEKS

**Data**: dataset sintetis kredit konsumer Indonesia (tenor 12 bulan; tujuan: education, home_improvement, household, working_capital, medical). Sumber: train/test applications, customer_profiles, previous_applications, repayment_history, account_behavior, bureau_observations (3 provider, ada delay available_datetime 60-120 hari), ecommerce_transactions. Nilai IDR, timestamp Asia/Jakarta.

**Fakta terverifikasi dari baseline**
- Train 70.000 baris (target rate 13.58%, 9.507 positif), test 30.000. Satu aplikasi per customer, tidak ada customer yang muncul di train dan test. Split acak sepanjang 2025 (adversarial AUC 0.505, tanpa covariate shift).
- Tidak ada pelanggaran cutoff (0 baris) di semua sumber pada audit awal. Rekonstruksi target tidak mungkin (loan target tidak ada di repayment_history).
- Installment cocok dengan formula amortisasi (selisih <= 1 IDR).
- Fitur: 132 (feature table sudah ada di data/processed/train_features.parquet dan test_features.parquet).
- Baseline OOF KS (5-fold, 56.000 baris + holdout 14.000): LR 0.4916 ± 0.0104, LightGBM 0.4879, CatBoost 0.4855, holdout sekitar 0.50. Single feature annual_interest_rate 0.2289. Application+Profile saja 0.2296. Acak 0.013. Shuffle test 0.007.
- Ablasi (drop sumber dari LightGBM): e-commerce -0.0804, repayment -0.0543, bureau -0.0155, previous_apps -0.0057, account_behavior -0.0009.
- Univariat teratas: ecom_spending_30d (KS 0.292), ecom_orders_30d, ecom_spending_90d, ecom_recency_days, repay_max_max_dpd, bureau_score_mean, annual_interest_rate.

**Temuan yang perlu dicurigai atau diselidiki**
- AUC OOF identik 0.8204 untuk LR, LightGBM, dan CatBoost padahal KS berbeda, kemungkinan bug pelaporan.
- LR ≈ GBM, artinya sinyal kebanyakan aditif. Dugaan: data dibangkitkan dari satu risiko laten yang tercermin berisik di banyak sumber (bunga, skor bureau, DPD lama, perilaku belanja). Strategi: denoising lewat pooling banyak observasi, bukan model lebih rumit.
- Bureau dari aplikasi LAMA (201.984 dari 301.984 application_id) kemungkinan belum dipakai. Ini sumber multi-titik tetapi risiko leakage: observasi aplikasi lama bisa baru available setelah cutoff aplikasi saat ini.
- account_behavior hampir tanpa kontribusi dan hanya mencakup sebagian customer (belum jelas siapa).
- application_datetime di-blacklist seluruhnya, padahal fitur kalender (bulan, jam, hari, tanggal) sah dipakai.

## 2. INFRASTRUKTUR KERJA

- Struktur: `data/interim`, `data/processed`, `src/`, `reports/`, `preds/`, `submissions/`, `experiments_log.csv`.
- Periksa resource: `nvidia-smi`, RAM, jumlah core, versi Python. Laporkan di `reports/ENV.md`. Instal paket hanya dari registry standar (pip/conda): polars, scikit-learn, lightgbm, xgboost, catboost, optuna, shap, scipy, pyarrow, torch (bila ada GPU atau perlu NN).
- Screening cepat: untuk menyaring ide fitur gunakan model kecil (LR dan LightGBM kecil) pada subsample stratified; konfirmasi hanya yang menjanjikan pada CV penuh (5-fold x 3 seed, seluruh 70.000 baris).
- `experiments_log.csv`: id, fase, hipotesis, perubahan, set fitur, model, KS CV mean ± std, selisih berpasangan vs baseline (mean ± std), AUC, waktu, keputusan (adopsi/buang), catatan.
- Fungsi inti di `src/`: skor KS resmi, pembuat folds (disimpan ke disk agar semua model memakai folds yang sama), bootstrap berpasangan, assert point-in-time, registry eval-only.

**Mode eksekusi**: setelah selesai satu FASE, tulis `reports/FASE_n.md` (tabel angka, keputusan, risiko) lalu BERHENTI dan tunggu instruksi "lanjut FASE n+1". Jika pengguna menulis kata AUTONOMOUS, lanjut ke fase berikutnya otomatis tetapi tetap menulis laporan tiap fase.

---

## FASE 1: VERIFIKASI DAN INFRASTRUKTUR (kerjakan dulu)

A1. Hitung ulang AUC OOF tiap model (B1, B2, B3), temukan sumber bug identik 0.8204, koreksi laporan.
A2. Audit ulang bureau untuk observasi dari previous applications: join via customer_id lalu previous_applications, hitung jumlah observasi dengan `available_datetime > decision_datetime` aplikasi saat ini, pastikan pipeline hanya memakai `available_datetime <= cutoff`. Laporkan apakah observasi bureau aplikasi lama sudah dipakai di 132 fitur.
A3. Tabel cakupan: per sumber (previous_apps, repayment, account_behavior, bureau per provider, ecommerce), persentase customer yang punya data di train vs test, dan default rate untuk yang punya vs tidak punya. Bedakan "nol karena tidak ada aktivitas" dari "NaN karena tidak ada data".
A4. Default rate per bulan keputusan, jam, hari dalam minggu, tanggal dalam bulan.
A5. Cek account_behavior: panjang histori per customer, AUC univariat income_like_credits/declared_income, saldo, low_balance_days, tren. Putuskan tegas apakah sumber ini punya sinyal atau dibuang.
A6. Arah hubungan ecom_spending_30d, ecom_orders_30d, ecom_recency_days dengan target (binning vs default rate), dan jelaskan mekanisme yang masuk akal.
A7. Bangun skema evaluasi baru: repeated stratified 5-fold x 3 seed pada seluruh 70.000 baris, folds disimpan. Ukur noise antar-seed. Holdout 14.000 hanya konfirmasi akhir. Hitung ulang baseline (LR, LightGBM, CatBoost) pada skema ini sebagai acuan baru, termasuk selisih berpasangannya.

## FASE 2: FEATURE ENGINEERING (sumber kenaikan terbesar)

Semua fitur wajib lolos assert point-in-time. Urutan prioritas (catat tiap hipotesis di log):

**B1. E-commerce mendalam** (dedup order, completed saja untuk spending)
- Window 7/14/30/60/90/180/365 hari; rasio window pendek vs panjang; slope; jeda antar order (mean/std/max); umur histori; jumlah hari aktif.
- Per product_category dan payment_method x window (terutama COD, electronics); rasio cancel per window; persentil nilai order; spending dibanding declared income dan installment.
- Pola jam, hari, tanggal belanja (kedekatan dengan tanggal gajian).
- Pola awal vs akhir: order pertama vs terakhir, perubahan gaya belanja.

**B2. Bureau multi-titik**
- Semua observasi `available <= cutoff` milik customer (termasuk dari aplikasi lama) per provider: jumlah, pertama/terakhir, delta, slope, rata-rata tertimbang recency.
- Observasi dengan report_datetime paling baru; disparitas skor antar provider; perubahan utilization/inquiries/delinquent; utilization dihitung ulang dari total.

**B3. Repayment granular**
- Per loan: proporsi loan yang pernah >=30/60/90 DPD; DPD EWMA; waktu sejak keterlambatan terakhir; K installment terakhir (K=3,6,12); pola pembayaran parsial; rasio paid/due berbobot recency; jumlah loan aktif dan beban cicilan bulanan saat ini.

**B4. Residual bunga**
- Fit `annual_interest_rate ~ fitur non-bunga` (tanpa target, prediksi OOF) dan jadikan residual dan prediksi sebagai fitur. Bunga kemungkinan hasil risk-based pricing lender (proxy skor internal), residualnya bisa memuat informasi yang dilihat lender. Uji juga installment dan loan_amount.

**B5. Kalender dan cakupan**
- Bulan, jam, hari dalam minggu, tanggal dalam bulan dari decision_datetime. Indikator cakupan/missingness antar sumber, jumlah sumber yang punya data.

**B6. Pooling observasi berisik (hipotesis risiko laten)**
- Rata-rata/median/EWMA lintas semua observasi sebuah customer; shrinkage empirical Bayes berdasarkan jumlah observasi.
- Skor komposit: rata-rata rank beberapa fitur kuat dari sumber berbeda; uji seberapa dekat KS-nya dengan model penuh.

**B7. Ide tidak lazim (boleh dicoba, tetap dalam batas sumber informasi)**
- Eksperimen "hanya titik awal dan akhir" (bureau, account, repayment) vs fitur lengkap: seberapa banyak sinyal yang tersisa.
- Pembentukan rasio berpasangan otomatis antar fitur uang lalu seleksi; interaksi; rank/quantile transform.
- Konsistensi lintas tabel: declared income vs income_like_credits, installment vs inflows, utilization hitung ulang vs bureau.
- Pola kebulatan angka, count/frequency encoding, WOE (di dalam fold), kNN atau cluster di dalam fold.

Setelah B1-B7: tampilkan ablasi per sumber sebelum vs sesudah, daftar fitur yang diadopsi vs dibuang, dan KS CV berpasangan.

## FASE 3: FITUR PALING BERPENGARUH DAN PALING REPRESENTATIF

Seleksi dan ranking fitur dilakukan DI DALAM fold latih (nested).

**3.1 Paling berpengaruh**
- Permutation importance pada OOF (KS dan AUC), SHAP mean |value|, drop-column untuk 30 fitur teratas. Laporkan stabilitas rank antar fold dan seed.
- Importance per GRUP (wajib, karena fitur berkorelasi memecah importance): grup per sumber, per konstruk (aktivitas e-commerce, disiplin bayar, skor bureau, beban utang, arus kas, harga/bunga), dan per klaster korelasi. Ukur penurunan KS saat grup diacak atau dibuang (selisih berpasangan).
- Forward selection per grup; kurva KS vs jumlah fitur/grup; titik jenuh.
- Interaksi: apakah GBM unggul pada interaksi (bandingkan LR vs LR+interaksi, SHAP interaction). Laporkan seberapa aditif sinyalnya.
- Bentuk dan arah hubungan fitur teratas (binning vs default rate), dengan penjelasan bisnis.

**3.2 Paling representatif**
- Hierarchical clustering pada |Spearman| (jarak 1-|rho|). Perwakilan tiap klaster: fitur dengan importance tertinggi, paling stabil, paling sedikit missing.
- Bandingkan tiga set: (a) lengkap, (b) satu perwakilan per klaster, (c) perwakilan + faktor laten per klaster (PCA/factor analysis di-fit di dalam fold). Targetkan set (b)/(c) di bawah 100 fitur untuk model fondasi.
- Set akhir: yang paling ringkas dengan KS tidak lebih rendah di luar noise dari set lengkap. Perbarui `feature_dictionary.md`.

## FASE 4: MODEL BERAGAM

Semua pada folds dan set fitur yang sama. Simpan prediksi OOF dan test tiap model di `preds/`.

**4.1 Model klasik**: LR dengan WOE/scorecard, LightGBM sangat teregulasi (num_leaves 4-15, min_child_samples besar, colsample rendah, monotone constraints bila masuk akal), CatBoost, XGBoost. Tuning Optuna dengan budget terbatas pada CV yang sama. Objective ramah ranking (binary logloss, lambdarank/pairwise bila perlu), class weighting sebagai perbandingan.

**4.2 NN tabular**: TabM dan/atau RealMLP bila compute cukup.

**4.3 Model fondasi tabular (TFM)**
- Verifikasi dulu aturan lomba soal model pretrained dan lisensi tiap model. Bila perlu token/akses (mis. gated weights), BERHENTI dan tanya pengguna, jangan mengakali. Catat di `reports/OPEN_QUESTIONS.md`.
- Urutan: TabPFN (versi terbaru yang tersedia; TabPFN-3 mendukung hingga 1 juta baris x 200 fitur; TabPFN-2.6 untuk hingga 100 ribu baris x 2.000 fitur bila fitur >200; cek batas versi dengan `pip index versions tabpfn` dan dokumentasi), TabICL v2, lalu TabDPT/Mitra bila waktu cukup. Verifikasi ketersediaan paket dan versi sebelum memakainya.
- Input: set fitur perwakilan (< 100 fitur) dari Fase 3, bandingkan dengan 132 fitur penuh. Model fondasi cenderung lebih baik pada fitur sedikit; GBM unggul pada fitur banyak.
- Protokol: tiap fold, data latih fold = konteks; prediksi pakai probabilitas kelas 1. Jika tanpa GPU dan terlalu lambat, gunakan 5 fold x 1 seed atau subsampling konteks (catat jelas), dan prioritaskan TabICL v2.
- Laporkan: KS OOF per fold, selisih berpasangan vs LR dan LightGBM, AUC, korelasi rank OOF antar model, waktu, memori, kebutuhan GPU. Uji ukuran konteks (25/50/100% fold latih) untuk melihat apakah data lebih banyak masih membantu.

**4.4 Model per segmen (hipotesis, bukan asumsi)**: default rate antar region/purpose/employment hampir datar, jadi expert per segmen berprioritas rendah. Satu-satunya segmentasi yang layak diuji: berdasarkan ketersediaan data (ada/tidak: repayment, bureau per provider, ecommerce, account) dan jumlah observasi. Bandingkan model global + fitur segmen vs expert per segmen. Adopsi hanya jika selisih berpasangan di luar noise dan ukuran segmen cukup (>= beberapa ribu baris, >= 500 positif).

**Deprioritas**: target encoding region x employment x housing (kecuali eksperimen kecil terbukti), tuning berat sebelum Fase 2 selesai, deteksi payday dari account_behavior (kecuali A5 menunjukkan account_behavior bersinyal).

## FASE 5: ENSEMBLE

- Hitung korelasi rank antar prediksi OOF dan KS tiap anggota. Buang anggota dengan korelasi > 0.98 terhadap anggota lain tanpa nilai tambah.
- Metode (dari paling sederhana): rank average bobot sama; greedy forward selection berbasis AUC/logloss OOF (BUKAN KS); regresi logistik non-negatif sebagai stacker (nested CV).
- Tambahkan keragaman lewat jenis model (LR/WOE, GBM, CatBoost, NN, TFM), tampilan fitur (lengkap, perwakilan, WOE), dan seed berbeda.
- DILARANG mengoptimasi bobot langsung ke KS. Adopsi ensemble hanya jika KS berpasangan naik di luar noise.
- Prediksi test = rata-rata RANK antar model dan seed; skalakan ke [0,1] tanpa mengubah urutan.

**Opsional, hanya jika pengguna mengonfirmasi bahwa diizinkan aturan lomba atau tim**: file prediksi test eksternal (`blend_lr_lgb5_cat.csv`, rank-normalisasi, tanpa OOF). File ini TIDAK BISA divalidasi lewat CV dan TIDAK BOLEH jadi fitur atau input stacking. Lakukan hanya: cek ID, hitung korelasi rank dengan prediksi test sendiri, dan jika diizinkan dan korelasi < 0.98 siapkan SATU varian blend dengan bobot kecil tetap (mis. 0.2, tanpa tuning). Laporkan sebagai eksperimen terpisah yang tidak tervalidasi lokal. Jangan menjadikannya kandidat akhir hanya karena naik tipis di public leaderboard.

## FASE 6: KANDIDAT FINAL DAN SANITY CHECK

- Shuffle test pada pipeline final (KS ~ 0). Ablasi per sumber pada model final. Cek distribusi prediksi test vs OOF.
- Konfirmasi hold-out 14.000 sekali saja pada kandidat final.
- Siapkan DUA kandidat: (A) KS CV mean tertinggi, (B) paling stabil antar fold atau paling beragam. Latih ulang pada seluruh train, prediksi test.
- Validasi otomatis submission: jumlah baris = test, setiap application_id tepat sekali, tidak ada ID di luar test, tidak ada NaN, nilai finite di [0,1].
- Simpan di `submissions/`. Catat KS CV yang diharapkan agar bisa dibandingkan dengan leaderboard setelah pengguna submit.

## 3. OUTPUT AKHIR

1. `experiments_log.csv` lengkap dan `reports/PROGRESS.md`.
2. `reports/FASE_1.md` sampai `FASE_6.md` dengan tabel angka dan keputusan.
3. Tabel akhir: model/ensemble | set fitur | KS CV mean ± std | selisih berpasangan vs baseline | KS hold-out | korelasi OOF | waktu.
4. Ranking grup fitur (penurunan KS), klaster beserta perwakilan, kurva KS vs jumlah fitur.
5. `feature_dictionary.md` terbaru, `eval_only_sources.md`, kode pipeline reproducible (src/ + skrip berurutan).
6. `submissions/` berisi dua kandidat final yang lolos validasi format.
7. `reports/UNUSED_GAPS.md` (celah/anomali yang ditemukan tetapi sengaja tidak dipakai) dan `reports/OPEN_QUESTIONS.md`.
8. Daftar hipotesis berikutnya, risiko, dan hal yang tidak berhasil.

## 4. KAPAN HARUS BERHENTI DAN BERTANYA

- Butuh token/akses/akun untuk mengunduh model.
- Resource tidak cukup untuk menjalankan eksperimen yang direncanakan.