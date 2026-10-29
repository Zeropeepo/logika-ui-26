# LOGIKA UI 2026 — credit risk experiments

Source code analisis data, feature engineering, baseline model, dan eksperimen HPC.

## Instalasi dependensi

Dengan Python 3.11 atau 3.12 dan `uv`, siapkan dependensi utama KuMo:

```bash
uv sync
```

Tambahkan `--extra legacy` untuk eksperimen LightGBM, CatBoost, XGBoost, dan
Optuna; tambahkan `--extra tabpfn` untuk benchmark TabPFN lama. Pilihan wheel
PyTorch dan CUDA harus sesuai dengan node HPC: untuk instalasi CUDA 12.6 yang
ditargetkan suite KuMo, ikuti [panduan HPC](HPC/kumo_suite/README.md).

## KuMo Small–Large

Paket terbaru menyediakan **52 konfigurasi ablasi**, 39 fitur temporal tambahan,
validasi berdasarkan ID, checkpoint per context bag, dan ekspor prediksi.

Mulai dari [panduan KuMo HPC](HPC/kumo_suite/README.md).
Lihat [analisis referensi Pieter](HPC/kumo_suite/REFERENCE_FINDINGS.md) dan
[daftar konfigurasi](HPC/kumo_suite/experiments.json).

```bash
python -m HPC.kumo_suite plan --preset screen
```

Data disediakan terpisah: ekstrak arsip `kumo_hpc_inputs.zip` di root repository
agar tersedia `data/kumo_inputs/`. Instal dependensi dan jalankan smoke GPU sesuai
panduan sebelum menjalankan eksperimen besar.

### Jika muncul error `train_features_v5.parquet` tidak ditemukan

Error tersebut berasal dari suite lama `HPC/experiments/exp01_kumo_foundation_suite.py`
atau `HPC/run_all_experiments_nohup.sh`. File v5 adalah tabel fitur hasil olahan,
dan tidak disertakan dalam Git. Suite KuMo terbaru menggunakan paket input yang
disiapkan terpisah, bukan nama file v5 tersebut.

Minta `kumo_hpc_inputs.zip` dari pemilik proyek, letakkan di root repository, lalu
jalankan dari root repository:

```bash
unzip kumo_hpc_inputs.zip
source HPC/.venv_hpc/bin/activate  # jika sudah memakai environment HPC lama
python -m HPC.kumo_suite run --names small_base --folds 0 --validation-limit 128 --query-batch 64
```

Jika environment lama belum tersedia, siapkan environment sesuai panduan KuMo HPC.
Perintah di atas adalah smoke run Small untuk memeriksa instalasi dan GPU. Setelah
berhasil, jalankan `python -m HPC.kumo_suite run --preset screen` untuk enam
eksperimen awal. Hindari menjalankan suite lama bersamaan pada GPU yang sama.

Pengolahan data dan pemeriksaan alur CPU telah dijalankan selama pengembangan.
Eksekusi pretrained KuMo dengan runner baru di GPU HPC belum diverifikasi.

## Struktur

- `HPC/kumo_suite/`: runner KuMo terbaru, konfigurasi, panduan dan template Slurm.
- `src/`: pipeline analisis, feature engineering dan eksperimen sebelumnya.
- `reports/`: catatan dan hasil analisis sebelumnya.
- `HPC/experiments/`: suite HPC sebelumnya; bukan entry point KuMo terbaru.

Dataset, prediksi, model outputs, dokumen pendaftaran pribadi, virtual environment,
dan kredensial lokal tidak disertakan dalam Git.
