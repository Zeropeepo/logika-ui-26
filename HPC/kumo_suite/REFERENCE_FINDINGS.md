# Temuan referensi Pieter dan implikasi untuk KuMo

Sumber: arsip `logika-ui-main.zip` yang diberikan pengguna, terutama
`docs/experiments/EXP-053-ens-q4cb-full-refit.md`,
`EXP-054-bag-plus-factory.md`, `EXP-059-repayment-quarantine-strict-eco.md`,
`docs/changelog/2026-10-22-ens-q4cbf-full-lb.md`, `src/features.py`,
`src/ensemble.py`, dan `src/common.py`.
Angka leaderboard berikut merupakan laporan di arsip, bukan hasil run baru.
Instruksi workflow pribadi dalam arsip tidak diadopsi.

## Apa sebenarnya baseline terbaik tersebut?

`ens_q4cbf_full` dilaporkan mencapai public KS **0,55922**, dibanding
`ens_q4cb_full` 0,55751 dan `ens_q4_full` 0,55388.
Rantai pembentukannya:

1. Empat keluarga model: LR, LightGBM, CatBoost, TabM, memakai fitur tabular dan
   skor sekuensial dari riwayat.
2. Representasi CoLES dalam tiga pengulangan pretraining. Prediksi per keluarga
   dirata-ratakan untuk mengurangi variasi pengulangan tersebut.
3. Feature factory ditambahkan hanya pada LightGBM, karena LR/CatBoost/TabM
   tidak terbantu dalam eksperimen referensi.
4. Empat keluarga digabung dengan bobot rank setara.
5. Prediksi fold dan refit full-data digabung pada test. OOF yang dilaporkan
   tetap OOF dari model CV, bukan bukti evaluasi refit yang baru.

CV champion dilaporkan 0,5253 / 0,5282 pada seed 42 / 7. Factory di atas bag
memberi +0,0008 / +0,0024 dan menang 8 dari 10 fold. Penulis sendiri menyebut
selisih public kecil tersebut masih berada dalam noise public. Tidak ada bukti
bahwa `ens_q4cbf_full` berasal dari KuMo.

CSV lampiran pengguna cocok dengan format rank: nilai sekitar `rank / 30000`,
sehingga kolom `default_probability` bukan probabilitas terkalibrasi. Prediksi
berkorelasi sangat tinggi dengan `ens_q4cb`, sesuai lineage tersebut, tetapi CSV
sendiri tidak dapat mengidentifikasi arsitektur maupun membuktikan skornya.
Tidak ada OOF pasangan CSV tersebut dalam paket input suite.

## Mengapa 0,55 tidak berarti hampir random?

Metrik kompetisi adalah **KS dua sampel**, bukan accuracy. Random biasanya
mendekati KS 0, dengan deviasi sampling; KS 0,55 merupakan pemisahan distribusi
yang cukup kuat. Proporsi positif train 13,58%, sehingga dummy mayoritas pun
bisa mempunyai accuracy 86,42% sambil tetap mempunyai KS 0. Jadi persoalannya
bukan terbukti tidak ada sinyal; persoalannya mencari tambahan sinyal yang
stabil di atas pemisahan yang sudah ada.

Analisis lokal sebelumnya menemukan sinyal tunggal yang bermakna pada belanja
30 hari, keterlambatan repayment, skor biro dan tier bunga. Dataset sintetis
bisa mengandung komponen risiko laten, noise label dan fitur proxy yang saling
berulang. Batas Bayes/noise generator **belum diketahui**; tidak dapat disimpulkan
hanya dari plateau leaderboard.

Public referensi sekitar 0,03 lebih tinggi dari CV. Jangan menambahkan offset
tetap pada prediksi atau menganggapnya jaminan skor private. Mungkin ada variasi
sampel public, komposisi, atau hal lain yang belum teridentifikasi.

## Struktur yang bisa dimanfaatkan

* Semua customer hanya mempunyai satu aplikasi terpilih; sumber histori harus
  diringkas terhadap cutoff aplikasi itu, bukan diperlakukan sebagai contoh baru
  dengan label yang sama pada random split.
* Tiga populasi sumber mempunyai jumlah konteks dan prevalensi berbeda.
  Populasi kecil memerlukan sharing informasi melalui model global atau soft mix.
* Bunga diskret menjadi proxy keputusan underwriting. Term 12 bulan konstan;
  cicilan sangat deterministik dari jumlah pinjaman dan bunga.
* Order ecommerce mempunyai banyak item. Menghitung jumlah baris item sebagai
  jumlah pesanan menimbulkan penggandaan; belanja jangka pendek, perubahan
  frekuensi dan COD layak diuji.
* Biro berbeda memiliki skala yang berbeda. Perubahan score/debt/inquiry dalam
  provider lebih masuk akal daripada mencampur urutan semua provider.
* DPD memerlukan denominator yang sudah cukup tua untuk menunjukkan ambang
  keterlambatan. Perubahan per loan menghindari mencampur trajectory pinjaman.
* Native missingness menunjukkan ketersediaan sumber dan harus dipertahankan.

Pieter juga menguji karantina repayment. Menghapusnya menurunkan KS CV sekitar
0,04; versi strict eco mencapai public 0,51127. Audit referensi melaporkan tidak
menemukan snapshot repayment yang tidak kompatibel. Itu mendukung mempertahankan
repayment sebagai baseline sambil tetap mengukur ablasi, bukan menganggapnya
pasti leakage atau pasti aman pada dataset lain.

## Apa yang diubah pada suite KuMo

Suite memprioritaskan uji terkendali untuk tiga kapasitas, fitur ringkas, fitur
baru temporal, lebar fitur, panjang konteks, preprocessing, context bags,
coverage sampling, dan global versus population experts. **Bukan reproduksi
CoLES/GRU Pieter**: melatih representasi sekuensial yang baru dengan nesting
benar merupakan pekerjaan tambahan. Memasukkan skor tersimpan dari pipeline
lama tanpa provenance/nesting tidak dibenarkan.

Tiga view fitur memisahkan pertanyaan apakah KuMo lebih terbantu representasi
ringkas atau banyak ringkasan manual. Penghapusan sumber menunjukkan
ketergantungan pada view terpilih. Transformasi ECDF fit pada outer-train,
sehingga tidak mengulang masalah rank train/test yang dihitung secara terpisah.

Bukan semua 52 konfigurasi perlu dijalankan. Ukur skala biaya pada screen,
bandingkan delta berpasangan, konfirmasi kandidat pada seed lain, lalu jalankan
test/refit. Kenaikan besar belum dapat dijanjikan; dasar pemilihannya adalah
CV yang konsisten dan efisiensi HPC, bukan public leaderboard semata.

Dokumentasi model resmi:

* [Model card NVIDIA Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular)
* [API KumoTabular](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoTabular)
* [In-context learning](https://nvidia.github.io/structured-data-models/icl.html)
