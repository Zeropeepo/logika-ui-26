# KuMo Small–Large: paket ablasi HPC

Jalankan dari **root repository** dengan Python 3.11. Paket ini terpisah dari
runner lama di `HPC/experiments`. Modelnya NVIDIA **Kumo Tabular**, library
`structured-data-models==0.2.1`, checkpoint `nvidia/Kumo-Tabular`, revision
`v1.0.1`. `fit()` menyimpan konteks untuk in-context learning; tidak melakukan
training bobot selama beberapa epoch.

## Isi dan urutan menjalankan

Ada **52 konfigurasi**, bukan perkalian semua kombinasi. Tahapan yang disarankan:

1. Lihat rencana tanpa memuat model.
2. Jalankan satu fold Small untuk memastikan lingkungan HPC bekerja.
3. Jalankan `screen`: enam konfigurasi × lima fold, tanpa inferensi test dulu.
4. Pilih kandidat berdasarkan KS per fold, AUC, waktu dan memori.
5. Jalankan ablasi relevan secara eksplisit dengan `--names`; hindari langsung
   menjalankan `all`. Konteks 32k, 16 estimator, dan delapan bags mahal.
6. Konfirmasi kandidat pada seed 2026 dan 999. `confirm` berisi sepuluh konfigurasi
   yang sudah ditentukan; untuk kandidat lain buat salinan konfigurasi di `plan.py`
   dengan seed berbeda **sebelum** evaluasi konfirmasi.
7. Ulangi kandidat final dengan `--predict-test`, lalu `summarize`. Checkpoint run
   tersebut terpisah dari run tanpa test karena kontrak prediksinya berbeda.

```bash
python -m HPC.kumo_suite plan --preset all
python -m HPC.kumo_suite plan --preset screen
```

### Lingkungan GPU

Gunakan virtual environment tersendiri. Contoh berikut memakai PyTorch 2.7.1 CUDA
12.6, sesuai syarat minimum SDM. Driver HPC harus kompatibel; lingkungan ini tidak
mengubah driver. Sesuaikan wheel bila administrator cluster mensyaratkan lainnya,
dan ubah pin torch di requirements agar tetap konsisten.

```bash
python3.11 -m venv .venv-kumo
source .venv-kumo/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r HPC/kumo_suite/requirements.txt
python -m pip check
python -c 'import torch,sdm; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())'
```

Model weights diunduh otomatis saat pertama membuat model pretrained. Node harus
bisa mengakses Hugging Face atau memakai cache yang sudah diisi sebelumnya. Untuk
node offline, jalankan unduhan di node yang diizinkan, lalu salin cache dan tetapkan
`HF_HOME` ke folder cache tersebut. Jangan menyimpan token di repository.

Alokasi awal yang disarankan: satu GPU, delapan CPU, RAM 48 GB. VRAM 80 GB berguna
untuk Large dan konteks panjang, tetapi belum merupakan jaminan semua konfigurasi
muat. Waktu antrean dan durasi aktual bergantung cluster.

### Input portable yang sudah disiapkan

Salin `data/kumo_inputs/` beserta seluruh isinya: train, test, folds dan manifest.
Arsip input terpisah dari source agar Git tidak dipenuhi data. Input ini berisi
70.000 train dan 30.000 test, v5 yang ada ditambah **39 fitur baru** dari raw.
Manifest menyimpan SHA256 input dan sumbernya.

Untuk mengirim tanpa seluruh repository, ekstrak `kumo_hpc_source.zip` dan
`kumo_hpc_inputs.zip` ke folder yang sama. Perintah dalam panduan tetap berlaku.
Arsip source hanya memuat suite ini; arsip input terpisah sekitar 70 MB.
Jika menambahkan suite ke repository yang sudah ada, gabungkan aturan `.gitignore`
dengan hati-hati; jangan menimpa aturan repository dari arsip.

Membuat ulang paket transfer:

```bash
python -m HPC.kumo_suite bundle --output transfer_baru
```

Jika ingin membangun ulang, dibutuhkan tiga file v5/folds di `data/processed` dan
empat CSV raw repayment, bureau, account, ecommerce di `data/raw`:

```bash
python -m HPC.kumo_suite prepare --output data/kumo_inputs_rebuilt
```

Folder output harus kosong. Paket transfer input siap pakai membuat pembangunan
ulang pipeline v5 lama tidak diperlukan di HPC. File data dan folder `transfer/`
diabaikan Git; kirim arsip input melalui kanal transfer file.

### Jalankan

Periksa lingkungan pada GPU yang benar. Smoke berikut benar-benar menggunakan
checkpoint Small, satu fold dan 128 baris validasi; bukan angka CV final:

```bash
python -m HPC.kumo_suite run --names small_base --folds 0 --validation-limit 128 --query-batch 64
python -m HPC.kumo_suite run --preset screen --query-batch 512
python -m HPC.kumo_suite summarize
```

Contoh ablasi tahap lanjut:

```bash
python -m HPC.kumo_suite run --names large_ctx8192 large_ctx16384 large_bag4 large_enhanced_rank
python -m HPC.kumo_suite run --names large_drop_repay large_drop_ecom large_drop_bureau
python -m HPC.kumo_suite run --preset confirm
```

Inferensi test kandidat dan refit dengan pool training penuh:

```bash
python -m HPC.kumo_suite run --names large_base large_enhanced --predict-test
python -m HPC.kumo_suite summarize
python -m HPC.kumo_suite run --names large_enhanced --refit
```

Refit menggunakan semua train sebagai **pool**; ukuran konteks tetap mengikuti
konfigurasi. Gunakan coverage bags/konteks lebih panjang bila ingin memperbesar
cakupan. Hasil refit tidak mempunyai skor OOF sendiri.

Setiap CV lengkap dengan test menghasilkan `submission_probability.csv` dan
`submission_rank.csv`. Refit menghasilkan `refit_probability.csv` dan
`refit_rank.csv`. Rank adalah nilai urutan, **bukan probabilitas terkalibrasi**.
Tidak ada upload submission otomatis.

Untuk Slurm, sesuaikan partition/account/time pada template lalu:

```bash
sbatch HPC/kumo_suite/submit_slurm.sh
```

Untuk membagi kerja, jalankan konfigurasi berbeda pada job berbeda memakai
`--names`. `--folds 0 1` juga didukung. Direktori run sama memakai lock sehingga
dua job tidak menulis bersamaan; job kedua akan berhenti, bukan merusak hasil.
Jalankan `summarize` setelah job selesai. Satu proses memakai satu GPU; tidak ada
DDP atau asumsi bahwa VRAM beberapa GPU bisa digabungkan.

## Definisi ablasi

| Faktor | Pilihan |
|---|---|
| Kapasitas | Small, Medium, Large; konteks/fitur/sampel sama pada baseline |
| Fitur | compact: 50 fitur; enhanced: 89; wide: v5 + fitur baru, dibersihkan |
| Konteks Large | 1.024, 2.048, 4.096, 8.192, 16.384, 32.768 |
| Estimator internal | 1, 2, 4, 8, 16 |
| Context bags | 1, 2, 4, 8 |
| Populasi | global; tiga experts; campuran 50% global + 50% expert |
| Sampling | stratified sesuai prevalensi; uniform; coverage per kelas |
| Transformasi | native; signed log; ECDF train; median + missing indicators |
| Penghapusan sumber | repayment, ecommerce, bureau, account, previous, pricing, calendar |
| Konfirmasi split | seed 42, 2026, 999, mengikuti fold yang disediakan |

Jumlah fitur aktual bisa berkurang per outer fold setelah menghapus konstanta dan
duplikat persis; daftar aktual disimpan di metadata setiap context bag. SDM juga
memakai ensemble preprocessing internal. `wide` adalah uji apakah lebar fitur
membantu; jangan menganggap lebih banyak fitur selalu lebih baik.

Populasi ditentukan dari ketersediaan sumber, tanpa label:

* `full`: ada repayment; train 46.312, test 19.845.
* `thin`: tidak ada previous; train 21.000, test 9.000.
* `prev_no_repay`: previous tersedia, repayment tidak; train 2.688, test 1.155.

Tidak semua `prev_no_repay` merupakan pemohon yang selalu ditolak. Konteks expert
dibatasi jumlah outer-train yang tersedia. Prevalensi kelas dipertahankan sedekat
mungkin; tidak ada oversampling positif atau ranking terpisah per populasi.
Coverage bags memakai permutasi kelas yang digeser sehingga cakupan bertambah;
setelah satu putaran penuh, konteks dapat saling tumpang tindih.

Query batch terpisah dari ukuran konteks. Default 512; OOM saat query akan mencoba
batch lebih kecil dengan konteks yang sama dan mencatat ukuran minimum. OOM saat
fit berhenti dengan error; ukuran model/konteks tidak diturunkan diam-diam. Pilih
konfigurasi konteks lebih kecil secara eksplisit. Waktu yang dicatat mencakup
inisialisasi model pada bag pertama. Peak VRAM/RSS merupakan high-water mark
proses, bukan konsumsi eksklusif bag tersebut.

## Pencegahan hasil evaluasi yang menyesatkan

* Fold digabung berdasarkan `application_id`; urutan file tidak diasumsikan sama.
* Tidak memakai ID/hash, prediksi pricing tersimpan, rank yang dihitung terpisah
  pada train/test, atau komposit turunannya. Tidak memakai label test/pseudo-label.
* NaN numerik dan categorical tetap native pada konfigurasi utama. Encoding dan
  transformasi tambahan hanya belajar dari outer training.
* Fitur temporal baru memfilter waktu ketersediaan terhadap decision cutoff;
  ecommerce diringkas per order; weekend memakai Sabtu–Minggu.
* Fitur repayment mempertahankan informasi yang tersedia pada snapshot. Ambang
  DPD censoring membatasi denominator pada installment yang sudah cukup tua.
  Ablasi tanpa repayment membantu mengukur ketergantungannya, bukan bukti kebocoran.
* Positive class diambil dari nama kelas `1` milik SDM, bukan ditebak dari label
  validasi. KS absolut dilaporkan bersama AUC agar arah skor juga terlihat.
* Hasil disimpan per fold/populasi/bag secara atomik. Input, source code, versi
  library dan konfigurasi membentuk identitas run; perubahan membuat run baru.
* `summary.csv/json` memuat jumlah fold lengkap, KS tiap fold, AUC/logloss OOF.
  `paired_comparisons.json` membandingkan dengan Large baseline jika identitas
  input/kode/lingkungan cocok. Gate screening: delta mean KS ≥0,0024 dan ≥4/5
  fold menang. Tetap konfirmasi pada split lain; 52 pencarian menambah selection bias.
* Skor OOF refit, leaderboard dan CV jangan dicampur. Prediksi Pieter hanya bisa
  menjadi pembanding test karena kita tidak memiliki OOF pasangannya.

Fitur compact berasal dari pipeline lokal yang telah dipakai sebelumnya, sehingga
CV suite ini mengukur perbandingan terkendali; bukan estimasi sepenuhnya bebas dari
riwayat pemilihan fitur. Wide mewarisi definisi fitur v5, termasuk keterbatasan
historis seperti slope biro yang mencampur provider. Fitur baru memakai perubahan
per provider dan per loan. Semua prediktor v5 belum direkonstruksi ulang dari raw.
Kalender yang dihapus pada enhanced adalah `hours_to_decision`; fitur kalender lain
ada pada wide. Penghapusan sumber merupakan uji dependensi fitur pada view tersebut,
bukan perbandingan kausal antar sumber data.

### Blend kandidat dengan bobot yang ditetapkan sebelumnya

Setelah semua fold dan test lengkap, gunakan direktori run dari output:

```bash
python -m HPC.kumo_suite blend --runs HPC/kumo_results/RUN_A HPC/kumo_results/RUN_B --weights 1 1 --output HPC/kumo_results/fixed_blend
```

Input/fold/ID wajib cocok. Blend memakai rank per fold untuk OOF dan rank global
untuk test. Bobot tidak dioptimasi terhadap OOF; konfirmasi kembali bila anggota
dipilih berdasarkan skor yang sama. Tidak tersedia stacking otomatis yang memakai
fitur OOF lintas fold tanpa nesting.

## Status pemeriksaan lokal

Persiapan data nyata berhasil: 70k/30k baris dan 39 fitur baru. Lima pemeriksaan
CPU pada input sintetis lulus saat pengembangan: metrik, sampling/coverage,
transformasi fold, penggabungan fold berdasarkan ID, dan alur soft experts + bags +
resume + ekspor + refit. Pemeriksaan CPU memakai backend `diagnostic` (logistic
regression) dan **tidak membuktikan performa KuMo**.

API SDM yang terpasang sudah diperiksa untuk class mapping, fit, predict dan clear.
KuMo pretrained Small/Medium/Large **belum dijalankan dengan paket ini di HPC**.
Mulai dari smoke GPU di atas. Tidak ada janji skor lebih tinggi dari 0,55922.
Lingkungan lokal saat pemeriksaan memakai Python 3.11.16, torch 2.14.1, SDM 0.2.1,
numpy 2.4.6, pandas 3.0.6, polars 2.0.0, scipy 1.17.1, sklearn 1.9.1.
Pin instalasi HPC menggunakan versi publik konservatif dan belum diinstal ulang
di mesin teman; `pip check` dan smoke GPU merupakan langkah awal wajib.

Detail reverse engineering dan interpretasi sinyal: `REFERENCE_FINDINGS.md`.
