# LOGIKA UI 2026 — credit risk experiments

Source code analisis data, feature engineering, baseline model, dan eksperimen HPC.

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

Pengolahan data dan pemeriksaan alur CPU telah dijalankan selama pengembangan.
Eksekusi pretrained KuMo dengan runner baru di GPU HPC belum diverifikasi.

## Struktur

- `HPC/kumo_suite/`: runner KuMo terbaru, konfigurasi, panduan dan template Slurm.
- `src/`: pipeline analisis, feature engineering dan eksperimen sebelumnya.
- `reports/`: catatan dan hasil analisis sebelumnya.
- `HPC/experiments/`: suite HPC sebelumnya; bukan entry point KuMo terbaru.

Dataset, prediksi, model outputs, dokumen pendaftaran pribadi, virtual environment,
dan kredensial lokal tidak disertakan dalam Git.
