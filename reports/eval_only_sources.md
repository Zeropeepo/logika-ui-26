# Eval-Only Sources & Strict Separation Registry

Kolom-kolom berikut terdaftar dalam `EVAL_ONLY_COLUMNS` dan DILARANG KERAS masuk sebagai fitur numerik atau kategorikal dalam model:

1. `target`: Label ground truth (end-of-day loan DPD >= 90 dalam 12 bulan kalender). Hanya untuk kalkulasi loss/loss gradient dan evaluasi metrik (KS/AUC).
2. `application_id`: Identifier aplikasi (opaque string). Tidak boleh dijadikan fitur model.
3. `customer_id`: Identifier nasabah (opaque string). Tidak boleh dijadikan fitur model.
4. `application_datetime`: Timestamp pengajuan aplikasi. Hanya untuk audit audit & window relative calculation.
5. `decision_datetime`: Cutoff timestamp keputusan kredit.
6. `cutoff`: Helper timestamp cutoff.

Pipeline fitur dan training script memiliki asersi ketat untuk memastikan tidak ada kolom di atas yang masuk ke feature matrix X.
