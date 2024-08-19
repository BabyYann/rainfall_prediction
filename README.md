# Proyek Prediksi Curah Hujan

Proyek ini bertujuan untuk memprediksi curah hujan menggunakan model LSTM berbasis Attention dan mengintegrasikannya ke dalam aplikasi Streamlit. Aplikasi ini menyediakan antarmuka pengguna untuk memproses dan memvisualisasikan prediksi curah hujan berdasarkan data historis.

## Struktur Proyek

- `data/`: Berisi file data mentah dan data yang telah diproses.
  - `raw/`: Data asli yang belum diproses.
  - `processed/`: Data yang telah dibersihkan dan dipersiapkan untuk pemodelan.

- `models/`: Berisi model-model akhir yang telah dilatih.
  - **`model_1.h5`**, **`model_2.h5`**, dan seterusnya: Model yang telah dilatih dan siap digunakan.

- `src/`:
  - **`models/`**: Salinan dari model-model akhir yang berada di luar direktori `src/`. Salinan ini digunakan oleh aplikasi Streamlit.
  - `app1.py`: Script utama untuk aplikasi Streamlit. Digunakan untuk menjalankan antarmuka pengguna dan memproses input data.
  - `test_data_pasti.csv`: File data uji yang digunakan dalam aplikasi Streamlit untuk menguji model dengan data baru.

- `.streamlit/`: Folder yang berisi tema untuk Streamlit, termasuk tema gelap dan terang.

- `requirements.txt`: Daftar paket Python yang diperlukan untuk menjalankan proyek.

- `README.md`: Dokumentasi proyek ini.

## Instalasi

1. Clone repository:
   ```bash
   git clone <URL_REPOSITORY>
   cd rainfall_prediction_project
