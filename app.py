import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# --- JUDUL & INFO ---
st.title("🚀 Telco Customer Analytics Dashboard")
st.markdown("""
Aplikasi ini menggunakan teknik Data Mining untuk memprediksi perilaku pelanggan.
**Metode:** Logistic Regression, Ensemble (Random Forest), & Clustering (K-Means).
""")
st.markdown("---")

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file CSV ini nanti diupload bareng ke GitHub!
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = df.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("⚠️ File 'WA_Fn-UseC_-Telco-Customer-Churn.csv' tidak ditemukan! Pastikan file ini ada di Repository GitHub.")
    st.stop()

# --- SIDEBAR MENU ---
menu = st.sidebar.selectbox("Pilih Menu:", ["🔮 Prediksi Churn (Supervised)", "👥 Segmentasi Pelanggan (Clustering)"])

# --- PREPROCESSING UMUM ---
df_proc = df.copy()
le = LabelEncoder()
for col in df_proc.columns:
    if df_proc[col].dtype == 'object':
        df_proc[col] = le.fit_transform(df_proc[col])

# --- MENU 1: PREDIKSI CHURN (KLASIFIKASI) ---
if menu == "🔮 Prediksi Churn (Supervised)":
    st.header("🔮 Prediksi Churn: Logistic Regression vs Ensemble")
    
    # Split Data
    X = df_proc.drop('Churn', axis=1)
    y = df_proc['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sidebar Pilihan Model
    model_choice = st.sidebar.radio("Pilih Algoritma:", ("Logistic Regression", "Ensemble (Random Forest)"))

    # Training Model (On the fly)
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        st.info("Menggunakan Algoritma: **Logistic Regression**")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        st.success("Menggunakan Algoritma: **Ensemble (Random Forest)**")
    
    model.fit(X_train, y_train)
    
    # Evaluasi Cepat
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Akurasi Model", f"{acc*100:.2f}%")
    col_metric2.metric("Recall (Daya Deteksi Churn)", f"{rec*100:.2f}%")

    # --- INPUT USER ---
    st.write("### 📝 Simulasi Data Pelanggan Baru")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tenure = st.slider("Lama Langganan (Bulan)", 0, 72, 12)
        monthly = st.number_input("Tagihan Bulanan ($)", 0.0, 150.0, 70.0)
    with col2:
        contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col3:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    if st.button("🔍 Prediksi Sekarang"):
        # Mapping Manual Input User ke Angka
        # Kita gunakan data rata-rata dummy, lalu timpa dengan input user
        input_data = X.mean().values.reshape(1, -1)
        
        # Update nilai penting (Index kolom harus sesuai dataset)
        # Tenure=4, Internet=7, Tech=11, Contract=14, Payment=16, Monthly=17
        input_data[0][4] = tenure
        input_data[0][17] = monthly
        
        # Logic Mapping Sederhana
        input_data[0][14] = 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)
        input_data[0][7] = 0 if internet == "DSL" else (1 if internet == "Fiber optic" else 2)
        input_data[0][11] = 2 if tech_support == "Yes" else (0 if tech_support == "No" else 1)
        
        # Prediksi
        hasil = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        st.write("---")
        if hasil == 1:
            st.error(f"🚨 **HASIL: CHURN (BERISIKO)!**")
            st.write(f"Probabilitas Kabur: **{prob[1]*100:.1f}%**")
            st.warning("Rekomendasi: Berikan penawaran khusus segera.")
        else:
            st.success(f"✅ **HASIL: SETIA (AMAN)**")
            st.write(f"Probabilitas Bertahan: **{prob[0]*100:.1f}%**")

# --- MENU 2: CLUSTERING ---
elif menu == "👥 Segmentasi Pelanggan (Clustering)":
    st.header("👥 Segmentasi Pelanggan (K-Means)")
    st.info("Mengelompokkan pelanggan berdasarkan 'Tenure' dan 'Monthly Charges'.")

    # Persiapan Data Clustering
    X_cluster = df[['tenure', 'MonthlyCharges']].copy()
    
    # Scaling (Wajib buat K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Slider Jumlah Cluster
    k = st.sidebar.slider("Jumlah Cluster (K)", 2, 5, 3)
    
    # Modelling
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    X_cluster['Cluster'] = clusters

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Cluster', data=X_cluster, palette='viridis', s=100, ax=ax)
    plt.title(f"Visualisasi {k} Cluster Pelanggan")
    plt.xlabel("Lama Langganan (Bulan)")
    plt.ylabel("Tagihan Bulanan ($)")
    st.pyplot(fig)

    # Interpretasi
    st.write("### 📊 Profil Rata-Rata per Cluster")
    summary = X_cluster.groupby('Cluster').mean().reset_index()
    st.dataframe(summary)
    
    st.write("""
    **Tips Membaca:**
    * **Tenure Tinggi + Tagihan Tinggi:** Pelanggan Setia Premium (VIP).
    * **Tenure Rendah + Tagihan Tinggi:** Pelanggan Baru Potensial (Rawan Churn).
    * **Tagihan Rendah:** Pelanggan Hemat.
    """)