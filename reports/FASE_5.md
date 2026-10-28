# LAPORAN FASE 5: ENSEMBLE COMBINATION & NESTED LOGISTIC STACKING

**Tanggal/Waktu**: 2026-10-12  
**Protokol CV**: Repeated Stratified 5-Fold x 3 Seed (`[42, 2026, 999]`), 15 fold identik pada 70.000 baris.  
**Metrik Utama**: Two-sample Kolmogorov-Smirnov (`scipy.stats.ks_2samp(y_pred[y==1], y_pred[y==0]).statistic`).

---

## 1. Audit Korelasi Rank Antar Model OOF

Sebelum melakukan ensembling, korelasi Spearman ($r$) antar probabilitas Out-of-Fold diukur untuk memastikan adanya diversitas fungsional:

```
              LightGBM      XGBoost     CatBoost       LogReg
LightGBM        1.0000       0.9964       0.9830       0.9522
XGBoost         0.9964       1.0000       0.9831       0.9502
CatBoost        0.9830       0.9831       1.0000       0.9485
LogReg          0.9522       0.9502       0.9485       1.0000
```

### Observasi Diversitas:
- Pohon keputusan (LGB, XGB, CatBoost) berkorelasi sangat tinggi ($r \approx 0.98 - 0.99$).
- Namun model linear (LogReg) memiliki korelasi yang jauh lebih longgar ($r \approx 0.948 - 0.952$), menyajikan sudut pandang induktif yang berbeda dan melengkapi pohon keputusan.

---

## 2. Eksperimen Kombinasi Ensemble

| Metode Ensemble | Formula / Struktur | OOF KS | OOF AUC | Peningkatan vs Baseline | Status |
|---|---|---|---|---|---|
| **Baseline Acuan** | LightGBM Fase 1 | 0.4907 | 0.8243 | +0.0000 | Acuan |
| **Ensemble 1 (Simple Rank)** | 0.25 (LGB + XGB + CB + LR) | 0.5030 | 0.8339 | +0.0123 | Pembanding |
| **Ensemble 2 (Weighted Rank)** | 0.40 LR + 0.25 LGB + 0.20 XGB + 0.15 CB | 0.5034 | 0.8346 | +0.0127 | Pembanding |
| **Ensemble 3 (Prob Blend)** | 0.40 LR + 0.25 LGB + 0.20 XGB + 0.15 CB | **0.5033** | **0.8347** | **+0.0126** | **KANDIDAT B** (Paling Stabil) |
| **Ensemble 4 (Logistic Stacker)** | Meta Logit Stacker (Nested 5-Fold) | **0.5056** | **0.8348** | **+0.0149** | **KANDIDAT A** (Skor Tertinggi) |

---

## 3. Analisis Bobot Logistic Stacker

Meta-learner dilatih dengan regresi logistik terregularisasi pada skala log-odds ($\text{logit}(p)$) menggunakan nested cross-validation untuk mencegah meta-leakage:
$$\text{logit}(P(y=1)) = \beta_0 + \beta_1 \text{logit}(p_{\text{LR}}) + \beta_2 \text{logit}(p_{\text{XGB}}) + \beta_3 \text{logit}(p_{\text{LGB}}) + \beta_4 \text{logit}(p_{\text{CB}})$$

Rata-rata koefisien bobot yang dipelajari:
- **$\beta_{\text{LR}} \approx 0.51$ (51%)**: Memegang porsi terbesar karena kestabilan global terhadap 302 fitur.
- **$\beta_{\text{XGB}} \approx 0.31$ (31%)**: Memberikan koreksi non-linear utama dari interaksi pohon histogram dangkal.
- **$\beta_{\text{LGB}} \approx 0.12$ (12%)**: Memberikan kontribusi split fitur komplementer.
- **$\beta_{\text{CB}} \approx 0.06$ (6%)**: Memberikan regularisasi simetris.

Hasil kombinasi ini berhasil mendorong OOF KS ke **0.5056** dan AUC ke **0.8348**.
