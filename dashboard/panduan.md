  AI Cycle - Credit Risk Prediction (Ringkasan Jelas)

  1. INFORMASI DATASET
   - Data: Lending Club Loan Data (2007-2014)
   - Jumlah: 466,285 baris dengan 75 kolom awal
   - Tujuan: Memprediksi apakah pinjaman akan Default (Charged Off) atau Lunas (Fully Paid)

  2. VARIABEL TARGET (Dependent Variable)
   - Nama: target (dibuat dari loan_status)
   - Nilai: 0 = "Fully Paid", 1 = "Charged Off"
   - Proporsi: ~82% Lunas, ~18% Default (dataset tidak seimbang/seimbang)

  3. VARIABEL INDEPENDEN (Fitur/Factors)
  Kategori Utama:
   - Info Pinjaman: loan_amnt, int_rate, term, installment
   - Info Peminjam: annual_inc, emp_length, home_ownership, dti
   - Riwayat Kredit: fico_range_low, fico_range_high, revol_bal, delinq_2yrs
   - Tujuan Pinjaman: purpose, grade, sub_grade

  4. PROSES PREPROCESSING

    1 Langkah 1: Filtering
    2 - Ambil hanya pinjaman yang sudah selesai: "Fully Paid" & "Charged Off"
    3 - Buang pinjaman yang masih "Current" (hasil belum diketahui)
    4 
    5 Langkah 2: Cleaning  
    6 - Hapus 47 kolom yang lebih dari 50% datanya kosong
    7 - Imputasi data kosong: median untuk angka, mode untuk teks
    8 
    9 Langkah 3: Transformasi
   10 - Ubah "3 years" → 3.0 (lama kerja jadi angka)
   11 - Ubah "10.5%" → 10.5 (suku bunga jadi angka)
   12 - Gabung FICO low & high → FICO rata-rata

  5. FEATURE ENGINEERING

   1 - loan_to_income = loan_amnt / annual_inc (beban pinjaman relatif terhadap pendapatan)
   2 - fico_avg = (fico_range_low + fico_range_high) / 2 (rata-rata skor kredit)
   3 - payment_to_income = (installment * 12) / annual_inc (cicilan tahunan terhadap pendapatan)
   4 - One-hot encoding untuk variabel kategorikal (grade_A, grade_B, home_MORTGAGE, etc.)
   5 - Standardisasi semua angka (mean=0, std=1)

  6. MODEL TRAINING
    Algoritma yang Digunakan:
   1. Logistic Regression - Model dasar untuk klasifikasi biner
   2. Random Forest - Ensemble dari pohon keputusan
   3. Gradient Boosting - Ensemble yang memperbaiki kesalahan sebelumnya
   4. Neural Network - Model non-linier kompleks

  7. EVALUASI MODEL
  Metrics yang Digunakan:
   - Accuracy: Berapa % prediksi benar secara keseluruhan
   - Precision: Dari yang diprediksi default, berapa % benar-benar default
   - Recall: Dari yang benar-benar default, berapa % berhasil diprediksi
   - ROC-AUC: Kemampuan membedakan kelas (terbaik jika mendekati 1.0)

  8. PEMILIHAN MODEL TERBAIK
   - Bandingkan semua model berdasarkan ROC-AUC
   - Pilih model dengan skor tertinggi
   - Simpan sebagai best_model.pkl

  9. INTEGRASI DENGAN DASHBOARD
   - Model disimpan dalam format .pkl
   - Dashboard membaca model saat startup
   - Saat user masukkan data → prediksi risiko default langsung muncul

  10. ALUR LENGKAP

   1 Raw Data (466k records) → Preprocessing → Features Engineering → Model Training
   2        ↓
   3 Model Evaluation → Best Model Selection → Model Saving → Dashboard Integration
   4        ↓
   5 Real-time Risk Prediction for Loan Applications

  Poin Penting:
   - Anda berhasil membuat pipeline ML lengkap: dari data mentah → model → dashboard
   - Menangani dataset yang tidak seimbang (imbalanced class)
   - Menggunakan 4 algoritma berbeda untuk perbandingan
   - Membuat dashboard interaktif untuk pengambilan keputusan bisnis