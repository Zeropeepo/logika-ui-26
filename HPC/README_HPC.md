# PANDUAN EKSEKUSI HPC AI CENTER UB (A100-SXM4 80GB) - SUITE NOHUP

> **Paket KuMo terbaru (52 konfigurasi):** gunakan [panduan kumo_suite](kumo_suite/README.md)
> dan `python -m HPC.kumo_suite`. Paket ini memiliki Small/Medium/Large eksplisit,
> fitur temporal baru, resume per context bag, dan arsip transfer input tersendiri.
> Perintah suite NOHUP lama di bawah tidak menjalankan paket KuMo terbaru.

> **Kondisi Node:** NVIDIA A100-SXM4 80GB, Driver 580.126.20, CUDA 13.0, Shared GPU.  
> **Mode Eksekusi:** `nohup` (unattended background run, aman ditinggal semalaman). Auto-logging per eksperimen, auto-recovery bila satu eksperimen error.

---

## 1. DAFTAR EKSPERIMEN YANG DISIAPKAN

Di dalam folder `HPC/experiments/`:

| No | Script | Model / Metodologi | Peran & Target |
|:---|:---|:---|:---|
| **01** | `exp01_kumo_foundation_suite.py` | **NVIDIA Kumo-Tabular** (Medium / High Context, 4 Bags/Fold) | Maksimalisasi Tab-FM in-context learning per populasi |
| **02** | `exp02_smart_pseudolabel_rounds.py` | **Multi-Round Distillation** (Teacher = 0.55677 Anchor) | R1 ($P \ge 0.88$) + R2 Consensus soft pseudo-labeling |
| **03** | `exp03_population_expert_dart.py` | **Decoupled Pop Experts** (LGBM DART + High-MinChild) | Atasi drop fold terlemah pada Pop 2 (thin) & Pop 3 (rejected) |
| **04** | `exp04_catboost_gpu_suite.py` | **CatBoost GPU** (Native Categorical Interactions) | Ekstraksi non-linear pohon simetris di A100 GPU |
| **05** | `exp05_tab_resnet_rankgauss.py` | **Deep Tab-ResNet** + RankGauss Normal Transform | Neural manifold ortogonal (Spearman $r \approx 0.92$ vs pohon) |
| **06** | `exp06_adversarial_reweighted.py` | **Adversarial Sample Re-weighting** ($w_i = \frac{P_{test}}{1-P_{test}}$) | Kompensasi pergeseran domain covariate shift train vs test |
| **07** | `exp07_ensemble_optimizer.py` | **Master Nelder-Mead Ensemble Optimizer** | Optimasi bobot global + decoupled per segmen untuk proteksi min fold |

---

## 2. CARA MENJALANKAN DENGAN `NOHUP` (DI-TINGGAL)

### Langkah 1: Masuk ke Folder HPC
```bash
cd ~/LOGIKA-UI/HPC
```

### Langkah 2: Jalankan Master Runner di Background
```bash
nohup bash run_all_experiments_nohup.sh > master.out 2>&1 &
```
*Job akan langsung berjalan di background (detach dari terminal). Anda bisa menutup SSH/laptop.*

---

## 3. CARA MONITORING PROGRESS KAPAN SAJA

Di terminal HPC, cukup jalankan:
```bash
cd ~/LOGIKA-UI/HPC
bash monitor_nohup.sh
```

Script ini akan otomatis menampilkan:
1. Status proses Python yang sedang berjalan.
2. Kondisi VRAM GPU A100 saat ini.
3. Tabel ringkasan hasil eksperimen yang sudah selesai (`hpc_experiments_summary.csv`).
4. 25 baris log terakhir dari eksperimen aktif.

Untuk memantau log secara live (real-time stream):
```bash
tail -f logs/master_nohup_*.log
```

---

## 4. HASIL OUTPUT YANG DIHASILKAN

Semua output tersimpan terstruktur:
- **Prediksi OOF & Test Tiap Model:** di `HPC/preds/` (`oof_exp*.parquet`, `test_exp*.parquet`).
- **Tabel Rekap Metrik:** di `HPC/hpc_experiments_summary.csv` (berisi Overall KS, Min Fold KS, Mean Fold KS, Pop 1/2/3 KS).
- **Submissions Siap Submit:** di `HPC/submissions/`:
  1. `submission_HPC_CHAMPION_057_PLUS.csv` *(Prioritas Utama: Ensemble Kumo + Distilasi + Pop Expert + Anchor 0.55677)*
  2. `submission_HPC_BALANCED_POWER_BLEND.csv` *(Global power blend)*
  3. `submission_HPC_PURE_AUTONOMOUS_ENSEMBLE.csv` *(Ensemble murni tanpa anchor external)*
  4. Dan sub-file per eksperimen (`sub_exp01_*.csv` s/d `sub_exp06_*.csv`).
