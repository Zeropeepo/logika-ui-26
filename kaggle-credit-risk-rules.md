# ATURAN PERMANEN PROYEK: Kaggle Credit Risk (metrik KS)



## Metrik dan target
- Target = 1 jika end-of-day DPD >= 90 dalam 12 bulan kalender setelah origination.
- Metrik lomba = KS two-sample (scipy.stats.ks_2samp antara prediksi kelas 1 vs kelas 0). Hanya ranking yang penting, kalibrasi tidak.
- Submission: `application_id,default_probability`, tiap application_id test tepat sekali, nilai finite di [0,1].

## Aturan point-in-time (paling penting)
- `decision_datetime` adalah information cutoff. Dilarang memakai informasi dengan timestamp >= cutoff.
- bureau_observations: pakai hanya `available_datetime <= decision_datetime` (bukan report_datetime).
- account_behavior: hanya bulan lengkap sebelum cutoff.
- ecommerce: hanya `transaction_datetime < cutoff`, dedup order sebelum agregasi, order cancelled bukan spending.
- repayment_history: hanya installment/pembayaran sebelum cutoff.
- Window waktu selalu relatif terhadap decision_datetime masing-masing aplikasi, bukan tanggal global.
- Pipeline fitur HARUS identik untuk train dan test.

## Boleh Dilakukan
1. Mencari label test untuk digunakan sebagai evaluasi model dan mengoverfit ke data test karena itu memang objektif
2. Memakai ID atau urutan sebagai fitur
3. Memakai model pretrained utk tabular foundatio model seperti TabPFN, dll.
4. Boleh menggunakan expert models, jadi contoh ada TabFPN khusus A, tabFPN khusus B, dan seterusnya.
5. Memilih model berdasarkan skor leadeboard 
6. Boleh banget dan wajib optimasi bobot ensemble langsung ke KS
7. Pilih model berdasarkan leaderboard (KARENA TUJUAN UTAMA MEMANG SKOR KS SETINGGI MUNGKIN, kalau bisa diatas 0.6)

## LARANGAN KERAS (walau menaikkan skor)
1. Memakai data setelah cutoff.
2. Melakukan submit ke Kaggle sendiri. Pengguna yang submit manual.
3. Menjalankan perintah destruktif (rm -rf di luar folder interim/cache, menimpa data mentah) atau mengirim data ke layanan luar.

## Validasi
- Skema acuan: repeated stratified 5-fold x 3 seed pada SELURUH 70.000 baris train. Split train/test acak (adversarial AUC 0.505), jadi CV ini proxy leaderboard yang sah.
- Bandingkan eksperimen dengan selisih BERPASANGAN pada folds yang sama (mean dan std selisih). Selisih di bawah noise bukan perbaikan.
- Semua transformasi bergantung-target (target encoding, WOE, seleksi fitur, stacking) wajib di-fit di dalam fold latih (nested). Tidak boleh bocor ke fold validasi.
- Hold-out hanya untuk konfirmasi akhir.
- Lonjakan KS > 0.03 dari satu eksperimen wajib diinvestigasi (shuffle test + ablasi per sumber) sebelum diadopsi.

## Cara kerja
- Selalu buat rencana (Planning) sebelum eksekusi tiap fase, dan hasilkan walkthrough berisi tabel angka di akhir fase.
- Catat SETIAP eksperimen di `experiments_log.csv` dan ringkas status di `reports/PROGRESS.md` segera setelah selesai, karena konteks agen bisa hilang.
- Simpan hasil antara ke parquet (data/interim, data/processed). Simpan prediksi OOF dan test tiap model di `preds/` dengan nama jelas dan folds yang sama.
- Seed tetap di semua langkah. Skrip harus bisa dijalankan ulang (resumable) tanpa mengulang semuanya.
- Tunjukkan bukti (angka, tabel), bukan hanya kesimpulan. Laporkan juga apa yang gagal dan apa yang diragukan.
- Jika menemukan ambiguitas, tulis asumsimu eksplisit, uji dengan data bila bisa, dan catat di `reports/OPEN_QUESTIONS.md`.