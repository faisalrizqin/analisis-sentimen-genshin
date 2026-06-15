import streamlit as st
import joblib
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Konfigurasi Halaman (Menggunakan Layout Wide agar seperti Dashboard Professional)
st.set_page_config(
    page_title="Genshin Sentimen AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# 2. Fungsi Caching untuk Memuat Model & Vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    best_svm_model, vectorizer = load_model()
    model_loaded = True
except:
    model_loaded = False

# 3. Header Dashboard
st.title("✨ Genshin Impact Sentiment Analytics Dashboard")
st.markdown("Analisis Sentimen Twitter Menggunakan Perpaduan **AI Lokal (DistilBERT)** untuk Pelabelan dan **Support Vector Machine (SVM)** untuk Klasifikasi.")
st.write("---")

if not model_loaded:
    st.error("⚠️ Gagal memuat model! Pastikan file 'svm_model.pkl' dan 'tfidf_vectorizer.pkl' berada di folder yang sama dengan file app.py ini.")
else:
    # 4. MEMBUAT TABS UNTUK NAVIGASI DASHBOARD
    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Real-Time", "📊 Statistik Dataset", "📈 Performa Model SVM"])

    # =========================================================
    # TAB 1: PREDIKSI REAL-TIME
    # =========================================================
    with tab1:
        st.subheader("Pengecekan Sentimen Tweet Baru")
        user_input = st.text_area(
            "Masukkan tweet atau opini tentang Genshin Impact di sini:",
            placeholder="Contoh: Seneng banget akhirnya dapet Arlecchino di hard pity! atau Gacha ampas banget selalu rate off...",
            height=100
        )

        if st.button("Analisis Sentimen", type="primary"):
            if user_input.strip() == "":
                st.warning("Silakan masukkan teks terlebih dahulu.")
            else:
                with st.spinner("Model sedang menganalisis..."):
                    time.sleep(0.4)
                    transformed_text = vectorizer.transform([user_input])
                    prediction = best_svm_model.predict(transformed_text)[0]
                
                # Tampilan Hasil Analisis
                st.write("#### Hasil Analisis:")
                if prediction == 'positif':
                    st.success(f"### 🎉 SENTIMEN: POSITIF")
                    st.info("💡 *Karakteristik Teks:* Mengandung kata-kata apresiasi, kepuasan gacha, kesenangan bermain, atau emosi bahagia.")
                else:
                    st.error(f"### 😡 SENTIMEN: NEGATIF")
                    st.info("💡 *Karakteristik Teks:* Mengandung keluhan sistem game, kekecewaan gacha (rate-off/hard pity), kritik, atau emosi kesal.")

    # =========================================================
    # TAB 2: STATISTIK DATASET (VISUALISASI TAMBAHAN)
    # =========================================================
    with tab2:
        st.subheader("Analisis Distribusi Data Twitter")
        
        csv_path = "tweets-data/genshin_tweets_labeled.csv"
        if os.path.exists(csv_path):
            df_clean = pd.read_csv(csv_path)
            total_data = len(df_clean)
            counts = df_clean['label'].value_counts()
            
            # Membuat KPI Metrics di bagian atas
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Tweet Berlabel", f"{total_data} Tweet")
            col2.metric("Sentimen Positif (🎉)", f"{counts.get('positif', 0)} data", 
                        f"{round(counts.get('positif', 0)/total_data*100, 1)}%")
            col3.metric("Sentimen Negatif (😡)", f"{counts.get('negatif', 0)} data", 
                        f"-{round(counts.get('negatif', 0)/total_data*100, 1)}%", delta_color="inverse")
            
            st.write("---")
            
            # Membuat Visualisasi Samping-Sampingan (Bar Chart & Pie Chart)
            plot_col1, plot_col2 = st.columns(2)
            
            with plot_col1:
                st.markdown("##### **Grafik Batang Jumlah Sentimen**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=counts.index, y=counts.values, palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_ylabel("Jumlah Tweet")
                ax.set_xlabel("Kelas Sentimen")
                # Tambah angka di atas bar
                for i, v in enumerate(counts.values):
                    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
                st.pyplot(fig)
                
            with plot_col2:
                st.markdown("##### **Persentase Kontribusi Sentimen**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                colors = ['#5dade2', '#ec7063']
                ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', 
                        startangle=90, colors=colors, explode=(0.05, 0))
                ax2.axis('equal')  
                st.pyplot(fig2)
                
            # Menampilkan sampel data mentah di bawah grafik
            with st.expander("👀 Lihat Sampel Data Hasil Pelabelan AI (Top 10 Data)"):
                st.dataframe(df_clean[['username', 'clean_text', 'label']].head(10), use_container_width=True)
        else:
            st.warning("File 'genshin_tweets_labeled.csv' tidak ditemukan di folder 'tweets-data/'. Silakan jalankan Cell 1 di Notebook Anda terlebih dahulu.")

    # =========================================================
    # TAB 3: PERFORMA MODEL SVM
    # =========================================================
    with tab3:
        st.subheader("Evaluasi & Akurasi Model Klasifikasi")
        
        # Ringkasan Parameter & Akurasi
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        perf_col1.metric("Akurasi Pengujian (Accuracy)", "65.0%", "Good Performance")
        perf_col2.metric("Hyperparameter Terpilih", "C: 10 | Kernel: Linear")
        perf_col3.metric("Metode Pembagian Data", "80 Train : 20 Test")
        
        st.write("---")
        
        perf_plot1, perf_plot2 = st.columns([1, 1.2])
        
        with perf_plot1:
            st.markdown("##### **Tabel Laporan Klasifikasi (MOCK)**")
            # Membuat dataframe manual berdasarkan hasil classification report Anda sebelumnya
            report_data = {
                'Kelas Sentimen': ['Negatif', 'Positif', 'Macro Avg', 'Weighted Avg'],
                'Precision': [0.67, 0.63, 0.65, 0.65],
                'Recall': [0.55, 0.74, 0.64, 0.65],
                'F1-Score': [0.60, 0.68, 0.64, 0.64]
            }
            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report.set_index('Kelas Sentimen'), use_container_width=True)
            st.caption("💡 *Insight:* Model memiliki kemampuan yang sangat baik dalam mengenali tweet **Positif** dengan nilai Recall mencapai **0.74 (74%)**.")
            
        with perf_plot2:
            st.markdown("##### **Matriks Kebingungan (Confusion Matrix)**")
            # Kita buat ulang visualisasi confusion matrix dari hasil uji 113 data Anda kemarin
            # negatif aktual = 55 (30 terjawab benar, 25 salah tebak positif)
            # positif aktual = 58 (43 terjawab benar, 15 salah tebak negatif)
            cm_data = [[30, 25], 
                       [15, 43]]
            
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['negatif', 'positif'], 
                        yticklabels=['negatif', 'positif'], ax=ax3)
            ax3.set_xlabel('Label Prediksi Model SVM')
            ax3.set_ylabel('Label Aktual (Ground Truth)')
            st.pyplot(fig3)

st.write("---")
st.caption("Genshin Impact Sentiment Dashboard © 2026 | Dibuat Menggunakan Streamlit, Scikit-Learn, dan Seaborn")