# Inference submission local rank-GRU

Paket ini untuk submission `submission_local_rank_gru_20261030.csv`, yang menurut
hasil leaderboard pengguna mendapat public score **0,56133**. Pipeline memakai
LightGBM, regresi logistik, KuMo Medium (80 fitur), dan GRU transaksi. Base
merupakan campuran probabilitas 40:40:20; hasil akhir adalah rank global base
90% dan rank global GRU 10%. Output adalah skor ranking, bukan probabilitas yang
terkalibrasi. Seluruh test harus diproses dalam satu batch untuk perhitungan rank.

## Input dan artefak

Jalankan dari root repository. File berikut harus disediakan terpisah:

* `data/kumo_inputs/test.parquet`: test yang sudah melalui pipeline v5 dan
  persiapan `HPC/kumo_suite`. Test mentah saja belum cukup.
* Untuk `--mode model`, lima CSV di `data/raw/`: `ecommerce_transactions.csv`,
  `repayment_history.csv`, `account_behavior.csv`, `bureau_observations.csv`,
  dan `previous_applications.csv`.
* Isi folder `inference_rank_gru/artifacts/` dari arsip artefak lokal yang
  ditransfer terpisah. Folder itu **tidak disertakan di Git**: konteks KuMo
  memuat 4.096 baris training dan label. Simpan dan kirim secara privat. CSV
  submission asli juga ada di arsip sebagai `submission_scored.csv`.

`predict.py` memeriksa checksum artefak dan data, sehingga mode ini sengaja
terbatas pada dataset yang sama dengan submission yang dilaporkan. Jangan
menjalankan file `joblib` atau checkpoint dari sumber yang tidak dipercaya.

## Lingkungan

Gunakan Python 3.11 dan environment yang berisi dependensi pada
`requirements.txt`. Daftar tersebut mencatat versi environment lokal yang
dipakai, bukan jaminan bahwa wheel yang sama tersedia di setiap HPC. Mode model
membutuhkan GPU CUDA dan pretrained checkpoint `nvidia/Kumo-Tabular` yang bisa
diakses library `structured-data-models`; siapkan cache Hugging Face untuk node
offline. Proses `fit` KuMo di mode model hanya mengatur konteks tetap untuk
in-context inference, tanpa melatih bobot model.

```bash
python -m pip install -r inference_rank_gru/requirements.txt
```

## Mendapatkan CSV asli

Mode `replay` menyalin prediksi yang sudah tersimpan, memeriksa checksum dan
urutan ID terhadap test parquet, dan menghasilkan CSV byte-for-byte sama. Ini
adalah pemutaran hasil tersimpan, bukan inferensi baru; mode ini tidak memerlukan
GPU atau CSV raw.

```bash
python inference_rank_gru/predict.py --mode replay --output submission_final.csv
```

SHA-256 CSV asli: `c5fada389fbaa6a068b3eccd4da73fcc4b8993f862644c27a5503852690aca47`.
Script menolak menimpa output yang sudah ada.

## Inferensi model

```bash
python inference_rank_gru/predict.py --mode model --output submission_inference.csv
```

Checkpoint LightGBM, regresi logistik dan GRU asli tidak disimpan saat CSV
leaderboard dibuat. Checkpoint pada arsip direkonstruksi setelahnya memakai
konfigurasi, data dan seed yang sama. Pada uji lokal penuh, fitur jendela dan
sekuens test cocok persis dengan input sebelumnya, tetapi CSV inferensi ulang
tidak identik: perbedaan absolut rata-rata skornya 0,000587 dan korelasi Spearman
0,999995. Skor **0,56133** hanya diketahui untuk CSV asli di mode replay; skor
submission hasil rekonstruksi belum diuji di leaderboard.

Angka OOF yang sebelumnya dilaporkan juga tidak mengevaluasi pipeline final
secara persis: regresi logistik pada OOF menggunakan fitur wide tanpa fitur
jendela, sementara CSV final memakai wide **dan** fitur jendela. Karena itu,
jangan gunakan angka OOF tersebut sebagai validasi langsung CSV ini.
