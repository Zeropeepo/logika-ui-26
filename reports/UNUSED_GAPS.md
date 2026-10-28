# LAPORAN CELAH DATA YANG DIIDENTIFIKASI & SENGAJA TIDAK DIPAKAI (UNUSED GAPS)

Sesuai kriteria sukses nomor 5 pada `mission.md`:
> *"Semua celah data yang ditemukan tetapi sengaja tidak dipakai tercatat."*

---

## 1. Daftar Celah Data & Alasan Pengabaian

| Sumber / Celah Data | Karakteristik Sinyal | Alasan Sengaja Dibuang / Tidak Dipakai | Risiko Jika Dipaksakan |
|---|---|---|---|
| **Kink / Step Threshold Binned Indicators** | 8 indikator biner titik patah (e.g., `spend > 2.19M`, `recency <= 2d`) | Gain penting (max 40.3) berada jauh di bawah noise kontrol acak (555.4). Pohon keputusan sudah memotong threshold kontinu secara optimal. | *Overfitting* & memecah kontinuitas gradien pohon (redundansi biner). |
| **Inkonsistensi Saldo Rekening (Flow vs Discrepancy)** | Selisih closing balance vs (opening + inflows - outflows) | Gain hanya 18.2 (di bawah noise kontrol). Akun bank disintesis dengan noise perbankan acak tanpa korelasi langsung ke target default. | Polusi dimensi fitur tanpa perbaikan pemisahan KS ($p > 0.05$). |
| **Entropi Kategori Belanja & HHI** | Herfindahl index konsentrasi kategori & Shannon entropy | Univariate KS < 0.010, gain 241 (di bawah noise kontrol 555.4). Jenis keragaman kategori tidak membedakan nasabah berisiko. | Menambah beban dimensionality ($D$) tanpa daya angkat pemisahan. |
| **Embedding Pretrained / TabPFN 3 Online Weights** | Model fondasi tabular prior-labs | Memerlukan autentikasi API token eksternal & lisensi gated (`ux.priorlabs.ai`), melanggar aturan perlombaan offline independen. | Ketergantungan layanan eksternal & risiko diskualifikasi regulasi kompetisi. |
| **Raw Timestamps & Datetime Exact Second** | `transaction_datetime`, `available_datetime` presisi detik | Mengandung noise temporal mikro yang tidak memiliki generalisasi ke data unlabelled di masa mendatang. | *Data leakage* semu / overfitting temporal. |
| **Kelipatan 100k pada Loan Amount** | Indikator `loan_amount % 100_000 == 0` | 100% sampel pada train dan test adalah kelipatan 100.000 (varians nol / konstanta). | Fitur konstan tanpa varians informasi. |
