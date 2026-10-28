# LAPORAN EDA, AUDIT LEAKAGE, DAN BASELINE MODELING
**Kompetisi**: Data Science LOGIKA-UI (Credit Risk Scoring — Consumer Lending)  
**Metrik Utama**: KS Statistic (two-sample Kolmogorov-Smirnov pada kelas 1 vs kelas 0)  
**Author**: Senior Credit Risk Data Scientist  

---

## 1. TAHAP 0: RINGKASAN DATA, GRAIN, DAN RELASI

### Grain & Relasi Antar Tabel
| Tabel | Grain (1 Baris) | Key Utama / Foreign Key | Catatan Bisnis & Relasi |
|---|---|---|---|
| `train_applications` | 1 aplikasi pinjaman diajukan (train) | `application_id` (PK), `customer_id` (FK) | 70.000 baris. 1 customer = tepat 1 aplikasi train. Tenor 12 bulan. Target: binary (1 = DPD >= 90 dalam 12 bulan). |
| `test_applications` | 1 aplikasi pinjaman diajukan (test) | `application_id` (PK), `customer_id` (FK) | 30.000 baris. 1 customer = tepat 1 aplikasi test. Disjoint customer dari train. |
| `customer_profiles` | 1 nasabah unik | `customer_id` (PK) | 100.000 baris. Relasi 1-to-1 dengan total nasabah di (train + test). |
| `previous_applications` | 1 riwayat aplikasi historis | `application_id` (PK), `customer_id` (FK), `loan_id` (FK jika approved) | 201.984 baris (70.000 customer unik). Semua aplikasi terjadi strictly sebelum target `decision_datetime`. |
| `repayment_history` | 1 jadwal/cicilan pinjaman historis | `installment_id` (PK), `loan_id` (FK), `customer_id` (FK) | 1.361.341 baris. 155.901 loan_id unik (semua berasal dari `previous_applications`). Tidak ada cicilan dari target loan itu sendiri. |
| `account_behavior` | 1 ringkasan rekening per bulan kalender lengkap | `customer_id`, `month` (Composite PK) | 1.084.418 baris. 70.000 customer unik. Bulan lengkap diobservasi strictly sebelum target `decision_datetime`. |
| `bureau_observations` | 1 laporan credit bureau per provider | `bureau_observation_id` (PK), `application_id` (FK), `customer_id` (FK) | 797.563 baris (301.984 aplikasi unik = 70k train + 30k test + 201.984 prev apps). 3 provider (Bureau_A, B, C). |
| `ecommerce_transactions` | 1 item keranjang belanja e-commerce | `order_item_id` (PK), `order_id` (FK), `customer_id` (FK) | 4.359.817 baris. 2.986.855 order unik. Semua transaksi strictly sebelum `decision_datetime`. |

### Shape & Memory Usage
- `train_applications`: 70.000 baris x 10 kolom, 6.89 MB
- `test_applications`: 30.000 baris x 9 kolom, 2.72 MB
- `customer_profiles`: 100.000 baris x 8 kolom, 7.13 MB
- `previous_applications`: 201.984 baris x 11 kolom, 22.09 MB
- `repayment_history`: 1.361.341 baris x 9 kolom, 132.59 MB
- `account_behavior`: 1.084.418 baris x 10 kolom, 88.94 MB
- `bureau_observations`: 797.563 baris x 14 kolom, 108.20 MB
- `ecommerce_transactions`: 4.359.817 baris x 10 kolom, 460.04 MB
- `sample_submission`: 30.000 baris x 2 kolom, 0.74 MB

---

## 2. TAHAP 1: PROFILING & VERIFIKASI KLAIM KAMUS DATA

1. **`customer_profiles`**: 100.000 baris tepat 100.000 `customer_id` unik (0 duplikat). Klaim kamus data diverifikasi valid.
2. **Distribusi aplikasi per customer di train & test**:
   - `train_applications`: 70.000 aplikasi berasal dari 70.000 nasabah unik (1 aplikasi/nasabah).
   - `test_applications`: 30.000 aplikasi berasal dari 30.000 nasabah unik (1 aplikasi/nasabah).
3. **Overlap train vs test**:
   - Overlap `customer_id`: 0 nasabah.
   - Overlap `application_id`: 0 aplikasi.
   - Partisi bersifat customer-disjoint murni.
4. **Overlap `previous_applications` dengan train/test**:
   - `application_id` di `previous_applications` tidak ada yang overlap dengan train (0) maupun test (0).
5. **Observasi Bureau**:
   - 301.984 aplikasi unik memiliki data bureau (70k train, 30k test, 201.984 previous applications).
   - Rata-rata 2.64 observasi per aplikasi (Provider A: 301.984, Provider B: 247.657, Provider C: 247.922).
   - Delay ketersediaan report bureau: selisih `available_datetime` - `report_datetime` memiliki median 1.440 jam (60 hari), max 2.880 jam (120 hari). Ini membuktikan pentingnya menggunakan `available_datetime` sebagai filter cutoff temporal.
6. **Repayment History**:
   - 155.901 `loan_id` unik. Seluruh 155.901 loan terdaftar pada `previous_applications`.
   - Tidak ada satupun loan dari aplikasi train/test yang masuk ke tabel cicilan ini (loan saat ini belum memiliki installment historis).

---

## 3. TAHAP 2: PEMAHAMAN TARGET & POPULASI

1. **Target Rate**:
   - Train target rate = **13.58%** (9.507 default dari 70.000 aplikasi).
   - Distribusi per tujuan pinjaman: Education (13.65%), Home Improvement (13.52%), Household (13.82%), Medical (13.63%), Working Capital (13.30%).
   - Distribusi per wilayah: Region A (13.57%), Region B (13.63%), Region C (13.55%), Region D (14.25%), Region E (12.91%).
   - Distribusi per pekerjaan: Salaried (13.24%), Self Employed (13.79%), Variable Income (14.03%).
2. **Rentang Waktu & Adversarial Validation**:
   - Rentang aplikasi: Train [2024-12-27 s/d 2025-12-28], Test [2024-12-27 s/d 2025-12-28].
   - Rentang keputusan: Train [2025-01-01 s/d 2025-12-28], Test [2025-01-01 s/d 2025-12-28].
   - Split train/test **BUKAN temporal split**, melainkan random customer split sepanjang tahun 2025.
   - **Adversarial Validation AUC (train vs test)**: **0.5052**. Tidak ada perbedaan fitur sama sekali antara train dan test.
3. **Selisih Waktu Keputusan vs Aplikasi**:
   - Median selisih: 9.0 jam di train, 10.0 jam di test (min 0 jam, max 119 jam ~ 5 hari). 0 anomali negatif.
4. **Verifikasi Amortisasi**:
   - Seluruh 70.000 baris pinjaman train mematuhi formula amortisasi standar dengan selisih <= 1 IDR (akibat pembulatan integer).

---

## 4. TAHAP 3: AUDIT DATA LEAKAGE (POINT-IN-TIME AUDIT)

Audit dilakukan dengan menguji relasi setiap baris di tabel sekunder terhadap `decision_datetime` (cutoff) aplikasi yang bersangkutan.

| Sumber Tabel | Jumlah Baris Kandidat | Baris Melanggar Cutoff | % Melanggar | Aksi / Kebijakan Pipeline |
|---|---|---|---|---|
| `previous_applications` | 201.984 | 0 | 0.0% | Strict filter: `decision_datetime < cutoff` |
| `bureau_observations` | 264.088 (train+test) | 0 | 0.0% | Strict filter: `available_datetime <= cutoff` |
| `account_behavior` | 1.084.418 | 0 | 0.0% | Strict filter: `month + 1 bulan <= cutoff` (hanya bulan lengkap) |
| `repayment_history` | 1.361.341 | 0 | 0.0% | Strict filter: `due_date < cutoff` & `last_payment_datetime < cutoff` |
| `ecommerce_transactions` | 4.359.817 | 0 | 0.0% | Strict filter: `transaction_datetime < cutoff`, buang `cancelled` untuk spending, dedup order_id |
| `customer_profiles` | 100.000 | 0 | 0.0% | Data statis saat aplikasi diproses |

**Kesimpulan Audit**: Panitia lomba telah membersihkan data mentah sehingga tidak ada data observasi masa depan (future records) yang tercampur sebelum cutoff. Meskipun demikian, pipeline fitur dirancang dengan filter ketat untuk menjamin kepatuhan temporal.

---

## 5. TAHAP 4: ARSITEKTUR MERGED FEATURE TABLE

Pipeline dieksekusi secara identik untuk train dan test:
1. **Fitur Aplikasi & Profil Nasabah**: 23 fitur (rasio angsuran/income, DTI, suku bunga bulanan, rasio bunga/pokok, income/dependent, tenure/age).
2. **Fitur Previous Applications**: 13 fitur (volume aplikasi lalu, approval rate, rejection rate, rasio requested/approved, recency hari).
3. **Fitur Repayment History**: 18 fitur (installment count, ontime ratio, underpaid ratio, max DPD, dpd >= 30/60/90, recent 6 installment trend).
4. **Fitur Account Behavior**: 28 fitur (rata-rata saldo, inflow/outflow 3m/6m/12m, outflow/inflow ratio, tren saldo, low balance days).
5. **Fitur Bureau Observations**: 20 fitur (score min/mean/max, provider count, total baki debet, limit & balance revolving, utilisasi rekalkulasi, inquiries 6m, delinquent 12m).
6. **Fitur E-commerce Transactions**: 26 fitur (order count dedup, rasio cancel, spending 30d/90d/180d, frekuensi COD vs e-wallet/transfer, share kategori belanja).
7. **Cross-Source Interaction**: 2 fitur (income credit to declared ratio, total debt to annual income).

Total: **132 Fitur Model Bersih**. Disimpan ke:
- `data/processed/train_features.parquet` (70.000 baris x 137 kolom)
- `data/processed/test_features.parquet` (30.000 baris x 136 kolom)
- `reports/feature_dictionary.md`

Asersi otomatis pada 5 sampel acak berhasil 100% cocok dengan rekonstruksi manual.

---

## 6. TAHAP 5 & 6: EVALUASI, SKEMA VALIDASI & REGISTRY GROUND TRUTH

### Top Sinyal Univariat (KS & AUC)
1. `ecom_spending_30d`: KS = 0.2918, AUC = 0.6960
2. `ecom_orders_30d`: KS = 0.2888, AUC = 0.6928
3. `ecom_spending_90d`: KS = 0.2856, AUC = 0.6939
4. `ecom_recency_days`: KS = 0.2784, AUC = 0.6865
5. `repay_max_max_dpd`: KS = 0.2744, AUC = 0.6822
6. `bureau_score_mean`: KS = 0.2525, AUC = 0.6745
7. `annual_interest_rate`: KS = 0.2301, AUC = 0.6464

Tidak ada fitur yang memiliki KS/AUC mencurigakan (tidak ada AUC > 0.8 pada univariat).

### Registry Pemisahan Ketat (`eval_only_sources.md`)
Kolom berikut di-blacklist dari input model:
`target`, `application_id`, `customer_id`, `application_datetime`, `decision_datetime`, `cutoff`.

### Skema Validasi Lokal
Berdasarkan temuan:
- Customer train vs test tidak overlap.
- Rentang waktu train dan test identik (Januari - Desember 2025).
- Adversarial validation AUC = 0.5052 (tidak ada drift temporal/populasi).
- Skema yang ditetapkan: **Stratified 5-Fold Cross Validation (Seed 42)** pada 80% data latih (56.000 sampel), ditambah **20% Stratified Hold-out Set** (14.000 sampel) sebagai pseudo-test.

---

## 7. TAHAP 7: HASIL BASELINE MODELING

### Tabel Performa Model Baseline
| Model | OOF KS Mean ± Std | 95% Bootstrap CI | Holdout KS | OOF AUC | Catatan |
|---|---|---|---|---|---|
| **B0: Random Baseline** | 0.0131 ± 0.0000 | [0.0076, 0.0264] | 0.0118 | 0.4930 | Skor acak U(0,1), KS mendekati 0 |
| **B0: Single Feature (`annual_interest_rate`)** | 0.2289 ± 0.0000 | [0.2163, 0.2415] | 0.2351 | 0.6461 | Benchmark univariat terkuat di aplikasi |
| **B1: Logistic Regression** | 0.4916 ± 0.0104 | [0.4746, 0.5015] | **0.5025** | 0.8204 | Median impute + scaling di dalam fold |
| **B2: LightGBM (Conservative)** | 0.4879 ± 0.0099 | [0.4699, 0.4971] | **0.4991** | **0.8204** | lr=0.03, num_leaves=31, colsample=0.8 |
| **B3: CatBoost (Default)** | 0.4855 ± 0.0125 | [0.4667, 0.4950] | 0.4967 | 0.8204 | depth=6, lr=0.05, early stopping |

### Sanity Checks (Wajib)
1. **Target Shuffle Test**:
   - Target diacak secara random, evaluasi 3-fold CV.
   - Hasil KS: **0.0073** (lolos asersi KS < 0.03).
2. **Ablasi Per Sumber Tabel (Ablation Study)**:
   - Full Model (132 fitur): KS = **0.4836**
   - Tanpa `ecommerce_transactions` (106 fitur): KS = **0.4032** ($\Delta = -0.0804$, sinyal belanja sangat signifikan)
   - Tanpa `repayment_history` (113 fitur): KS = **0.4293** ($\Delta = -0.0543$, disiplin bayar krusial)
   - Tanpa `bureau_observations` (111 fitur): KS = **0.4681** ($\Delta = -0.0155$)
   - Tanpa `previous_applications` (119 fitur): KS = **0.4779** ($\Delta = -0.0057$)
   - Tanpa `account_behavior` (102 fitur): KS = **0.4827** ($\Delta = -0.0009$)
   - Hanya Aplikasi & Profil Nasabah (23 fitur): KS = **0.2296** ($\Delta = -0.2540$)
3. **Stabilitas Antar Segmen**:
   - `loan_purpose`: KS berkisar 0.4758 s/d 0.5047.
   - `region`: KS berkisar 0.4808 s/d 0.4938.
   - `employment_type`: KS berkisar 0.4822 s/d 0.4915. Model stabil di seluruh sub-populasi.

### Validasi Submission Pertama
- File: `submissions/submission_baseline.csv`
- Baris: 30.000 (tepat 1 per `application_id` test)
- Header: `application_id,default_probability`
- Nilai: Seluruh nilai berhingga, 0.0050 s/d 0.8848.
- Distribusi skor test vs OOF train:
  - Mean: Train 0.1345 vs Test 0.1353
  - Median: Train 0.0735 vs Test 0.0738
  - Std: Train 0.1544 vs Test 0.1535
- **Expected Leaderboard KS**: ~**0.485 - 0.500**.

---

## 8. HIPOTESIS FEATURE ENGINEERING & RISIKO BERIKUTNYA

### Hipotesis Pengembangan Lanjutan:
1. **E-commerce Basket & Price Interaction**: Agregasi rata-rata item per order dan tren belanja harian menjelang pengajuan (misal lonjakan belanja COD 7-14 hari sebelum pinjaman diajukan).
2. **Account Cashflow Volatility & Salary Detection**: Ekstraksi tanggal penerimaan gaji reguler dari `income_like_credits` dan buffer rasio saat akhir bulan.
3. **Bureau Delinquency Decay & Weighting**: Memberikan bobot lebih besar pada observasi Provider A yang paling lengkap dibandingkan provider yang unscored.
4. **Target Encoding Out-of-Fold**: Target encoding untuk kombinasi `region` x `employment_type` x `housing_status`.

### Risiko / Hal yang Dihindari:
1. Tidak melakukan data tuning berlebihan atau ensemble berulang sebelum validasi baseline terbukti selaras dengan public LB.
2. Mempertahankan pemisahan fold yang ketat tanpa data leakage.
