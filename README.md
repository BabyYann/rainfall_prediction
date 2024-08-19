# Proyek Prediksi Curah Hujan

Proyek ini bertujuan untuk memprediksi curah hujan menggunakan model LSTM berbasis Attention dan mengintegrasikannya ke dalam aplikasi Streamlit. Aplikasi ini menyediakan antarmuka pengguna untuk memproses dan memvisualisasikan prediksi curah hujan berdasarkan data historis. 

Model LSTM berbasis Attention ini dirancang untuk menangkap pola temporal dalam data curah hujan dan meningkatkan akurasi prediksi dengan memfokuskan perhatian pada fitur-fitur penting dalam data historis. Dengan integrasi ke dalam Streamlit, pengguna dapat dengan mudah mengunggah data, melihat hasil prediksi, dan menganalisis informasi visual melalui antarmuka yang interaktif.

Data yang digunakan dalam proyek ini diperoleh dari situs resmi BMKG untuk memastikan keakuratan dan relevansi prediksi. Proyek ini bertujuan untuk memberikan alat yang berguna bagi para peneliti, pengambil keputusan, dan pihak lain yang tertarik dalam analisis curah hujan.


## Struktur Proyek

- `data/`: Berisi file data mentah dan data yang telah diproses.
  - `raw/`: Data asli yang belum diproses.
  - `processed/`: Data yang telah dibersihkan dan dipersiapkan untuk pemodelan.

- `models/`: Berisi model-model akhir yang telah dilatih.
  - **`model_1.keras`**, **`model_2.keras`**, dan seterusnya: Model yang telah dilatih dan siap digunakan.

- `src/`:
  - **`models/`**: Salinan dari model-model akhir yang berada di luar direktori `src/`. Salinan ini digunakan oleh aplikasi Streamlit.
  - `app1.py`: Script utama untuk aplikasi Streamlit. Digunakan untuk menjalankan antarmuka pengguna dan memproses input data.
  - `test_data_pasti.csv`: File data uji yang digunakan dalam aplikasi Streamlit untuk menguji model dengan data baru.

- `notebooks/`: Jupyter notebooks untuk eksplorasi data, eksperimen model, dan analisis. Proses pelatihan dan eksperimen model dilakukan menggunakan Google Colab untuk memanfaatkan sumber daya komputasi yang lebih besar dan kemudahan akses ke alat-alat analisis.


- `.streamlit/`: Folder yang berisi tema untuk Streamlit, termasuk tema gelap dan terang.

- `requirements.txt`: Daftar paket Python yang diperlukan untuk menjalankan proyek.

- `README.md`: Dokumentasi proyek ini.

## Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/BabyYann/rainfall_prediction
   cd rainfall_prediction_project
