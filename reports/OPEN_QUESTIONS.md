# OPEN QUESTIONS & EXTERNAL DEPENDENCY REGISTRY

## 1. Akses Token / Lisensi TabPFN 3 (Tabular Foundation Model)

- **Temuan**: Paket `tabpfn==9.1.0` telah terpasang di `.venv`. Namun, saat inisialisasi bobot untuk inferensi lokal (`hf_repo_id=tabpfn_3_5`), TabPFN mewajibkan autentikasi akun dan penerimaan lisensi Prior-Labs:
  ```
  TabPFN requires a one-time license acceptance to download model weights for local inference.
  https://ux.priorlabs.ai/login?callback=http://localhost:56739&hf_repo_id=tabpfn_3_5
  ```
- **Aturan Misi (mission.md 4.3)**:
  > *"Verifikasi dulu aturan lomba soal model pretrained dan lisensi tiap model. Bila perlu token/akses (mis. gated weights), BERHENTI dan tanya pengguna, jangan mengakali. Catat di reports/OPEN_QUESTIONS.md."*
- **Tindakan**:
  - Jika pengguna ingin menjalankan TabPFN: Pengguna dapat mendaftar di `https://ux.priorlabs.ai/account`, menerima lisensi, dan memberikan API key atau mengekspor `export PRIORLABS_API_KEY="..."`.
  - Sementara itu, pipeline kami telah menghasilkan model teruji offline yang tidak memerlukan token eksternal (LightGBM, XGBoost GPU, CatBoost GPU, dan Logistic Regression) yang telah terbukti mencetak OOF KS **0.5056** dan AUC **0.8348**.
