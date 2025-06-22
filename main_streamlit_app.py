# main_streamlit_app.py

import streamlit as st
import os
import numpy as np
import gdown # Digunakan untuk mengunduh model dari Google Drive
from tensorflow.keras.models import load_model # Untuk memuat model Keras/TensorFlow
from PIL import Image # Digunakan untuk manipulasi gambar (misalnya, mengubah ukuran)

# Impor fungsi pra-pemrosesan spesifik dari aplikasi TensorFlow Keras
# Ini penting untuk memberi makan gambar dengan benar ke model pra-terlatih
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

# --- Konfigurasi Model ---
# Direktori untuk menyimpan model yang diunduh
MODEL_DIR = "models"

# URL Google Drive untuk model Anda (menggunakan format 'uc?id=')
# PENTING: Pastikan URL ini benar dan file dapat diakses publik
MODEL_URLS = {
    "resnet": "https://drive.google.com/uc?id=1jVmr1kHY8cDSYgJEnIQ-OhqcUV8cj-qM",
    "vgg": "https://drive.google.com/uc?id=1kKUN75slUQtEsqv8tULqAC-HIVy_OBU8",
    "inception": "https://drive.google.com/uc?id=12YT-eiq09i3B8gY60KBjkOnJFHEKBIob"
}

# Nama file lokal untuk model
MODEL_FILENAMES = {
    "resnet": "resnet_best_model.h5",
    "vgg": "vgg_best_model.h5",
    "inception": "inception_best_model.h5"
}

# Ukuran gambar input yang dibutuhkan oleh setiap model
IMAGE_TARGET_SIZES = {
    "resnet": (224, 224),
    "vgg": (224, 224),
    "inception": (299, 299) # InceptionV3 biasanya membutuhkan 299x299
}

# Nama kelas untuk hasil prediksi
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- Fungsi Pembantu untuk Pra-pemrosesan Gambar ---
def preprocess_image_for_model(image_path, model_name):
    """
    Memuat dan pra-memproses gambar sesuai dengan persyaratan model yang ditentukan.
    """
    target_size = IMAGE_TARGET_SIZES.get(model_name)
    if not target_size:
        raise ValueError(f"Ukuran gambar tidak dikenal untuk model: {model_name}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch

    # Terapkan pra-pemrosesan spesifik model
    if model_name == "resnet":
        return resnet_preprocess(img_array)
    elif model_name == "vgg":
        return vgg_preprocess(img_array)
    elif model_name == "inception":
        return inception_preprocess(img_array)
    else:
        # Fallback untuk model tanpa pra-pemrosesan spesifik (misalnya, normalisasi sederhana)
        return img_array / 255.0

# --- Pemuatan Model dengan Caching Streamlit ---
# @st.cache_resource sangat penting di sini: memastikan model diunduh dan dimuat
# hanya sekali di seluruh sesi pengguna, mencegah operasi berat berulang.
@st.cache_resource(show_spinner=False) # show_spinner=False untuk menggunakan pesan kustom di bawah
def load_all_models_cached():
    """
    Mengunduh dan memuat semua model ensemble. Fungsi ini di-cache oleh Streamlit.
    """
    loaded_models = {}
    
    # Buat direktori model jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    st.markdown("---")
    st.write("Memulai proses memuat model di latar belakang...")
    st.write("Ini mungkin memakan waktu beberapa menit, terutama pada deployment pertama.")
    
    # Gunakan placeholder untuk pesan dinamis
    status_placeholder = st.empty()

    model_names_to_load = ["resnet", "vgg", "inception"]
    
    for i, name in enumerate(model_names_to_load):
        model_filepath = os.path.join(MODEL_DIR, MODEL_FILENAMES[name])
        
        # Unduh model jika belum ada
        if not os.path.exists(model_filepath):
            status_placeholder.info(f"⬇️ Mengunduh model {name} ({i+1}/{len(model_names_to_load)})...")
            try:
                # Gunakan fuzzy=True untuk gdown agar lebih tangguh dalam menangani potensi redirect
                gdown.download(url=MODEL_URLS[name], output=model_filepath, quiet=True, fuzzy=True)
                status_placeholder.success(f"✅ Model {name} berhasil diunduh.")
            except Exception as e:
                status_placeholder.error(f"❌ Gagal mengunduh model {name}: {e}")
                st.exception(e) # Tampilkan pengecualian lengkap di Streamlit
                return None # Menunjukkan kegagalan
        
        # Muat model
        status_placeholder.info(f"🧠 Memuat model {name} ({i+1}/{len(model_names_to_load)})...")
        try:
            model = load_model(model_filepath)
            loaded_models[name] = model
            status_placeholder.success(f"✅ Model {name} berhasil dimuat.")
        except Exception as e:
            status_placeholder.error(f"❌ Gagal memuat model {name}: {e}")
            st.exception(e) # Tampilkan pengecualian lengkap di Streamlit
            return None # Menunjukkan kegagalan
            
    status_placeholder.success("🎉 Semua model berhasil dimuat dan siap digunakan!")
    st.markdown("---")
    return loaded_models

# --- UI Aplikasi Streamlit ---

# Atur konfigurasi dasar halaman
st.set_page_config(
    page_title="Deteksi Penyakit Daun Jagung (Ensemble CNN)",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Judul dan Deskripsi Aplikasi
st.title("🌽 Deteksi Penyakit Daun Jagung")
st.markdown("""
Aplikasi ini menggunakan teknik *ensemble Convolutional Neural Network* (CNN)
(gabungan ResNet50, VGG16, dan Inception V3) untuk mendeteksi penyakit
pada daun jagung.
""")

# Panggil fungsi pemuatan model yang di-cache
ensemble_models = load_all_models_cached()

# Periksa apakah model berhasil dimuat
if ensemble_models is None or len(ensemble_models) != len(MODEL_FILENAMES):
    st.error("Aplikasi tidak dapat berfungsi penuh karena ada masalah dalam memuat model. Silakan periksa log deployment.")
    st.stop() # Hentikan eksekusi lebih lanjut jika model tidak dimuat

# Pengunggah File untuk input pengguna
uploaded_file = st.file_uploader(
    "Unggah gambar daun jagung Anda di sini:",
    type=["jpg", "jpeg", "png"],
    help="Hanya file gambar JPG, JPEG, atau PNG yang diizinkan."
)

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    st.image(uploaded_file, caption='Gambar yang Diunggah', use_column_width=True)
    st.write("")
    st.write("---")
    st.write("Menganalisis gambar...")

    # Buat path file sementara
    temp_image_path = os.path.join(MODEL_DIR, "uploaded_temp_image.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Placeholder untuk status prediksi
    prediction_status_placeholder = st.empty()
    prediction_status_placeholder.info("Melakukan prediksi...")

    try:
        predictions = []
        for model_name, model in ensemble_models.items():
            # Pra-proses gambar menggunakan fungsi pembantu
            processed_image = preprocess_image_for_model(temp_image_path, model_name)
            
            # Prediksi dengan model saat ini
            pred = model.predict(processed_image)
            predictions.append(pred)

        # Ensemble (soft voting: rata-rata probabilitas)
        ensemble_prediction = np.mean(predictions, axis=0)
        
        # Dapatkan indeks kelas yang diprediksi dan tingkat keyakinan
        predicted_class_index = np.argmax(ensemble_prediction)
        result = CLASS_NAMES[predicted_class_index]
        confidence = np.max(ensemble_prediction) * 100 # Konversi ke persentase

        prediction_status_placeholder.success("✅ Prediksi Selesai!")
        st.write("---")
        st.subheader(f"Hasil Deteksi: **{result}**")
        st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")
        st.markdown("---")
        
    except Exception as e:
        prediction_status_placeholder.error("❌ Terjadi kesalahan saat memproses gambar atau melakukan prediksi.")
        st.exception(e) # Tampilkan pengecualian lengkap untuk debugging
        
    finally:
        # Bersihkan file sementara
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)