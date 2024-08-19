import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import os

# Fungsi untuk mengatur tema menggunakan config.toml
def set_theme(theme):
    config_path = os.path.join(os.path.dirname(__file__), '.streamlit/config.toml')
    with open(config_path, 'w') as config_file:
        if theme == "Gelap":
            config_file.write("""
            [theme]
            base="dark"
            primaryColor="#d33682"
            backgroundColor="#002b36"
            secondaryBackgroundColor="#586e75"
            textColor="#ffffff"
            """)
        else:
            config_file.write("""
            [theme]
            base="light"
            primaryColor="#d33682"
            backgroundColor="#f0f2f6"
            secondaryBackgroundColor="#ffffff"
            textColor="#000000"
            """)

# Fungsi untuk memuat model
def load_best_model(model_path):
    try:
        model = load_model(model_path)
        st.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Fungsi untuk menangani missing values
def handle_missing_values(data):
    return data.ffill().bfill()

# Fungsi untuk menggantikan outlier menggunakan metode Z-score
def replace_outliers_zscore(df, columns):
    for column in columns:
        if df[column].std() == 0:
            st.warning(f"Kolom {column} memiliki standar deviasi 0, outlier tidak dapat dihitung.")
            continue
        df['zscore'] = zscore(df[column])
        median = df[column].median()
        df[column] = np.where(np.abs(df['zscore']) > 3, median, df[column])
        df.drop(columns='zscore', inplace=True)
    return df

# Fungsi untuk normalisasi data
def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# Fungsi untuk membuat dataset baru dengan lagging berdasarkan timestep
def create_lagged_features(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + timesteps), :])
        y.append(data[i + timesteps, -1])
    return np.array(X), np.array(y)

# Fungsi untuk prediksi
def predict_rainfall(model, X_test):
    try:
        return model.predict(X_test)
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
        return None

# Sidebar untuk panduan pengguna dan pemilihan tema
with st.sidebar:
    with st.expander("Panduan Pengguna"):
        st.markdown("""
        **Langkah-langkah:**
        1. Upload file data mentah dengan kolom `Tanggal`, `Tavg (°C)`, `RH_avg (%)`, dan `RR (mm)`.
        2. Masukkan path model LSTM yang telah dilatih.
        3. Aplikasi akan menampilkan hasil prediksi dan grafik perbandingan.
        4. Anda dapat mengunduh hasil prediksi dalam format CSV.
        """)

    theme = st.selectbox("Pilih Tema", ["Terang", "Gelap"])

# Terapkan tema yang dipilih
set_theme(theme)

st.title("Rainfall Prediction with Deep Learning (LSTM)")

# Load Model
model_path = st.sidebar.text_input("Path Model", "E:rainfall_prediction_project/src/models/best_model_tanh_bs16_ep100.keras")
best_model = load_best_model(model_path)

# Upload data mentah di sidebar
uploaded_file = st.sidebar.file_uploader("Upload file data mentah", type=["csv", "xlsx"])

if uploaded_file is not None:
    with st.spinner("Sedang memproses data..."):
        # Baca data mentah dengan encoding 'latin-1'
        if uploaded_file.name.endswith('.csv'):
            raw_data = pd.read_csv(uploaded_file, encoding='latin-1')
        elif uploaded_file.name.endswith('.xlsx'):
            raw_data = pd.read_excel(uploaded_file)

        # Memisahkan kolom "Tanggal" dan mengonversinya ke format datetime
        if 'Tanggal' in raw_data.columns:
            tanggal = pd.to_datetime(raw_data['Tanggal'], format='%d/%m/%Y', errors='coerce')
            data = raw_data.drop(columns=['Tanggal'])
        else:
            st.error("Kolom 'Tanggal' tidak ditemukan. Harap pastikan file memiliki kolom 'Tanggal'.")
            st.stop()

        st.subheader("Data Mentah")
        st.dataframe(raw_data.style.set_properties(**{'text-align': 'center'}), height=150)  # Tampilkan hanya 5 baris pertama

        # Preprocessing data
        data = handle_missing_values(data)
        columns_to_check = ['Tavg (°C)', 'RH_avg (%)', 'RR (mm)']
        data = replace_outliers_zscore(data, columns_to_check)
        feature_columns = ['Tavg (°C)', 'RH_avg (%)', 'RR (mm)']
        data, scaler = normalize_data(data, feature_columns)
        timesteps = 7
        features = data[feature_columns].values
        X, y = create_lagged_features(features, timesteps)

        if best_model is not None:
            # Prediksi menggunakan model
            y_pred = predict_rainfall(best_model, X)
            
            if y_pred is not None:
                tanggal_pred = tanggal.iloc[timesteps:].reset_index(drop=True)
                hasil_df = pd.DataFrame({
                    'Tanggal': tanggal_pred,
                    'Actual': y,
                    'Predicted': y_pred.flatten()
                })

                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi")
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.dataframe(hasil_df.style.set_properties(**{'text-align': 'center'}), height=200)  # Tampilkan 10 baris pertama
                st.markdown("</div>", unsafe_allow_html=True)

                # Tombol unduh
                st.subheader("Unduh Hasil Prediksi")
                csv = hasil_df.to_csv(index=False)
                st.download_button(label="Unduh sebagai CSV", data=csv, file_name='hasil_prediksi.csv', mime='text/csv')

                # Plotly Visualisasi
                st.subheader("Grafik Actual vs Predicted")
                fig = go.Figure()

                # Plot actual vs predicted
                fig.add_trace(go.Scatter(x=tanggal_pred, y=y, mode='lines', name='Actual', hoverinfo='x+y'))
                fig.add_trace(go.Scatter(x=tanggal_pred, y=y_pred.flatten(), mode='lines', name='Predicted', line=dict(color='red'), hoverinfo='x+y'))

                fig.update_layout(
                    title='Actual vs Predicted Rainfall',
                    xaxis_title='Tanggal',
                    yaxis_title='Rainfall (mm)',
                    xaxis=dict(tickformat='%d-%m-%Y'),
                    template=theme == "Gelap" and 'plotly_dark' or 'plotly_white'
                )
                st.plotly_chart(fig)

                # Menampilkan pesan sukses
                st.success("Prediksi selesai dan grafik berhasil ditampilkan!")
else:
                st.info("Silakan upload file data mentah untuk memulai prediksi.")
