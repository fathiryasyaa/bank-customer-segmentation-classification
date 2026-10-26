# Bank Customer Segmentation and Classification Pipeline

Proyek ini merupakan submission akhir untuk kelas **Belajar Machine Learning untuk Pemula** di **Dicoding Academy**. Proyek ini mengintegrasikan pendekatan *unsupervised learning* (clustering) dan *supervised learning* (klasifikasi) dalam satu alur kerja terstruktur untuk menganalisis perilaku transaksi perbankan.

Tahap pertama menggunakan algoritma **K-Means Clustering** untuk mengidentifikasi segmen nasabah berdasarkan karakteristik transaksi dan demografi. Label segmen yang dihasilkan kemudian diintegrasikan kembali ke dataset dan digunakan sebagai target prediksi pada tahap kedua, yaitu pemodelan klasifikasi menggunakan **Decision Tree** dan **Random Forest** dengan optimasi parameter melalui **GridSearchCV**.

---

## Daftar Isi
- [Latar Belakang dan Dataset](#latar-belakang-dan-dataset)
- [Alur Pengerjaan](#alur-pengerjaan)
  - [1. Eksplorasi dan Pra-Pemrosesan Data](#1-eksplorasi-dan-pra-pemrosesan-data)
  - [2. Pemodelan Clustering (Unsupervised Learning)](#2-pemodelan-clustering-unsupervised-learning)
  - [3. Interpretasi dan Profiling Cluster](#3-interpretasi-dan-profiling-cluster)
  - [4. Pemodelan Klasifikasi (Supervised Learning)](#4-pemodelan-klasifikasi-supervised-learning)
- [Hasil dan Evaluasi](#hasil-dan-evaluasi)
- [Struktur Repositori](#struktur-repositori)
- [Panduan Penggunaan](#panduan-penggunaan)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Atribusi](#atribusi)

---

## Latar Belakang dan Dataset

Dataset yang digunakan merekam 2.512 sampel data transaksi nasabah perbankan dengan berbagai atribut demografis, finansial, dan aktivitas akun. Dataset ini disediakan oleh Dicoding Academy (adaptasi dari *Bank Transaction Dataset for Fraud Detection*).

Fitur-fitur utama meliputi:
- **Demografi & Akun:** `CustomerAge`, `CustomerOccupation`, `AccountBalance`.
- **Karakteristik Transaksi:** `TransactionAmount`, `TransactionType`, `Channel`, `Location`, `TransactionDuration`, `LoginAttempts`.
- **Metadata Identitas:** `TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID`, `TransactionDate`, `PreviousTransactionDate` (dihapus pada tahap pembersihan).

---

## Alur Pengerjaan

### 1. Eksplorasi dan Pra-Pemrosesan Data
- **Eksplorasi Data (EDA):** Analisis statistik ringkas (`describe`, `info`), korelasi antar fitur numerik menggunakan matriks korelasi heatmap, distribusi data dengan histogram, serta sebaran nominal transaksi per profesi melalui visualisasi boxplot tanpa label yang tumpang tindih.
- **Pembersihan Data:**
  - Penanganan missing values (403 entri) menggunakan `dropna()`.
  - Penghapusan data duplikat (21 baris) menggunakan `drop_duplicates()`.
  - Eliminasi kolom pengenal/metadata (`ID`, `Date`, `IP Address`) yang tidak relevan untuk pemodelan perilaku.
- **Transformasi Fitur:**
  - Encoding fitur kategorikal menggunakan `LabelEncoder`.
  - Penanganan outlier numerik menggunakan metode batas *Interquartile Range* (IQR).
  - Standarisasi skala fitur numerik menggunakan `StandardScaler`.
  - *Feature binning* pada kolom `CustomerAge` menjadi 3 kategori usia (`Muda`, `Dewasa`, `Lansia`) menggunakan `pd.qcut` yang kemudian di-encode kembali.

### 2. Pemodelan Clustering (Unsupervised Learning)
- **Penentuan Cluster Optimal:** Menggunakan **Elbow Method** yang dievaluasi dengan metrik **Silhouette Score** via `KElbowVisualizer`, menghasilkan jumlah kluster optimal $k = 2$.
- **Pelatihan Model:** Model dilatih menggunakan algoritma **K-Means Clustering** ($k = 2$, `random_state=42`) menghasilkan nilai *Silhouette Score* sebesar **0.572**.
- **Reduksi Dimensi & Visualisasi:** Menerapkan **Principal Component Analysis (PCA)** 2-komponen untuk memproyeksikan sebaran titik data cluster dan posisi centroid ke dalam ruang dua dimensi.
- **Pemodelan Alternatif PCA:** Membangun model K-Means sekunder yang dilatih langsung pada representasi data hasil reduksi PCA sebagai model komparasi.

### 3. Interpretasi dan Profiling Cluster
Untuk memahami arti bisnis dari kluster yang terbentuk, data yang sebelumnya berada dalam skala z-score dikembalikan ke satuan aslinya menggunakan `inverse_transform` (nominal dalam Dollar, usia dalam Tahun, durasi dalam Detik, dan label teks kategori).

Profil dua segmen nasabah yang terbentuk:

| Karakteristik | Cluster 0 (Nasabah Dewasa Mapan) | Cluster 1 (Nasabah Pelajar / Usia Muda) |
| :--- | :--- | :--- |
| **Rata-rata Usia** | 45.06 tahun (Modus: Dewasa) | 44.33 tahun (Modus: Muda) |
| **Profesi Terbanyak** | Doctor (Dokter) | Student (Mahasiswa/Pelajar) |
| **Rata-rata Saldo Akun** | $5,142.17 | $5,058.81 |
| **Rata-rata Nominal Transaksi** | $255.55 | $258.15 |
| **Rata-rata Durasi Transaksi** | 121.12 detik | 117.30 detik |
| **Kanal Transaksi Utama** | Branch (Kantor Cabang) | Branch (Kantor Cabang) |
| **Wilayah Dominan** | Kota alfabetik awal (mis. Charlotte) | Kota alfabetik akhir (mis. Tucson) |

### 4. Pemodelan Klasifikasi (Supervised Learning)
Label cluster yang telah terpetakan dijadikan sebagai kolom `Target` pada dataset `data_clustering_inverse.csv` untuk melatih model prediktif.
- **One-Hot Encoding:** Mengubah fitur kategorikal hasil inverse menggunakan `pd.get_dummies(..., drop_first=True)` sehingga siap diproses model pohon keputusan.
- **Data Splitting:** Pembagian 80% data latih (1.556 sampel) dan 20% data uji (389 sampel) secara terstratifikasi (`stratify=y`, `random_state=42`).
- **Algoritma yang Diterapkan:**
  1. **Decision Tree Classifier:** Model dasar klasifikasi pohon biner.
  2. **Random Forest Classifier:** Model ensemble sebagai pembanding performa.
  3. **Hyperparameter Tuning:** Optimasi parameter pada Random Forest menggunakan `GridSearchCV` 5-fold cross validation (`n_estimators`, `max_depth`, `min_samples_split`).

---

## Hasil dan Evaluasi

Kedua algoritma klasifikasi dievaluasi pada data uji (test set) yang belum pernah dilihat sebelumnya:

| Model | Parameter | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | Default (`random_state=42`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Random Forest** | Default (`random_state=42`) | 1.00 | 1.00 | 1.00 | 1.00 |
| **Random Forest (Tuned)** | Best params: `n_estimators=50`, `max_depth=None`, `min_samples_split=2` | 1.00 | 1.00 | 1.00 | 1.00 |

Label kluster terpisah secara tegas pada fitur geografis dan kombinasi variabel demografi, sehingga model klasifikasi mampu memprediksi segmen target dengan presisi dan recall sempurna pada pengujian data uji.

---

## Struktur Repositori

```text
.
├── [Clustering]_Submission_Akhir_BMLP_Fathiryasyaa Achmad Zumar.ipynb   # Notebook tahap clustering (EDA, preprocessing, K-Means, PCA, inverse)
├── [Klasifikasi]_Submission_Akhir_BMLP_Fathiryasyaa Achmad Zumar.ipynb  # Notebook tahap klasifikasi (data split, Decision Tree, Random Forest, tuning)
├── model_clustering.h5                                                # Model K-Means Clustering utama
├── PCA_model_clustering.h5                                            # Model K-Means berbasis fitur reduksi PCA
├── decision_tree_model.h5                                             # Model Decision Tree Classifier
├── explore_RandomForest_classification.h5                             # Model Random Forest Classifier
├── tuning_classification.h5                                           # Model Random Forest hasil GridSearchCV
├── data_clustering.csv                                                # Dataset hasil clustering (skala terskala + kolom Target)
├── data_clustering_inverse.csv                                        # Dataset hasil clustering (skala asli + kolom Target)
├── requirements.txt                                                   # Daftar dependensi pustaka Python
├── .gitignore                                                         # Konfigurasi file yang diabaikan Git
└── README.md                                                          # Dokumentasi proyek
```

---

## Panduan Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/fathiryasyaa/bank-customer-segmentation-classification.git
cd bank-customer-segmentation-classification
```

### 2. Membuat dan Mengaktifkan Virtual Environment
- **Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Notebook
Jalankan notebook secara berurutan:
1. Buka dan jalankan **`[Clustering]_Submission_Akhir_BMLP_Fathiryasyaa Achmad Zumar.ipynb`** hingga selesai untuk memproses data dan menghasilkan berkas model serta `data_clustering_inverse.csv`.
2. Buka dan jalankan **`[Klasifikasi]_Submission_Akhir_BMLP_Fathiryasyaa Achmad Zumar.ipynb`** untuk melatih dan mengevaluasi model klasifikasi.

---

## Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python 3.10+
- **Manipulasi & Analisis Data:** Pandas, NumPy
- **Visualisasi Data:** Matplotlib, Seaborn, Yellowbrick
- **Machine Learning:** Scikit-Learn
- **Serialisasi Model:** Joblib

---

## Atribusi
Proyek ini dikembangkan oleh **Fathiryasyaa Achmad Zumar** sebagai tugas akhir kelulusan kelas **Belajar Machine Learning untuk Pemula** pada kurikulum **Dicoding Academy Indonesia**. Seluruh kriteria pengerjaan memenuhi standar penilaian tingkat lanjut (*Advanced*).
