import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Churn Prediction & Clustering", layout="wide")

# --- JUDUL ---
st.title("📊 Telco Analytics: Logistic Regression & Clustering")
st.markdown("""
Aplikasi ini menggabungkan dua pendekatan Data Mining:
1.  **Logistic Regression:** Untuk memprediksi probabilitas pelanggan Churn.
2.  **K-Means Clustering:** Untuk segmentasi profil pelanggan.
""")
st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Pastikan file csv ada di GitHub
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = df.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    except:
        return None

df = load_data()

if df is None:
    st.error("⚠️ File CSV tidak ditemukan! Upload 'WA_Fn-UseC_-Telco-Customer-Churn.csv' ke GitHub.")
    st.stop()

# --- SIDEBAR ---
menu = st.sidebar.selectbox("Pilih Analisis:", ["🔮 Prediksi Churn (Logistic Regression)", "👥 Segmentasi (Clustering)"])

# --- PREPROCESSING (Label Encoding) ---
df_proc = df.copy()
le = LabelEncoder()
encoders = {} # Simpan encoder biar rapi
for col in df_proc.columns:
    if df_proc[col].dtype == 'object':
        le_col = LabelEncoder()
        df_proc[col] = le_col.fit_transform(df_proc[col])
        encoders[col] = le_col

# --- MENU 1: LOGISTIC REGRESSION ---
if menu == "🔮 Prediksi Churn (Logistic Regression)":
    st.header("🔮 Prediksi Risiko Churn Pelanggan")
    
    # Split Data
    X = df_proc.drop('Churn', axis=1)
    y = df_proc['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    # Tampilkan Metrik
    col1, col2 = st.columns(2)
    col1.metric("Akurasi Model", f"{acc*100:.2f}%", help="Seberapa sering model benar menebak")
    col2.metric("Recall (Sensitivitas Churn)", f"{rec*100:.2f}%", help="Kemampuan mendeteksi pelanggan yang mau kabur")

    # --- FITUR SPESIAL LOGREG: KOEFISIEN ---
    with st.expander("🔍 Lihat Faktor Penyebab Churn (Koefisien LogReg)"):
        st.write("Grafik ini menunjukkan variabel yang paling mempengaruhi keputusan pelanggan:")
        coef_df = pd.DataFrame({
            'Fitur': X.columns,
            'Pengaruh (Koefisien)': model.coef_[0]
        }).sort_values(by='Pengaruh (Koefisien)', ascending=False)
        
        # Plot Bar Chart
        fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Pengaruh (Koefisien)', y='Fitur', data=coef_df.head(10), palette='coolwarm', ax=ax_coef)
        plt.title("Top 10 Faktor Paling Berpengaruh (+ Bikin Kabur, - Bikin Setia)")
        st.pyplot(fig_coef)
        st.caption("Nilai Positif (+) = Mendorong Churn. Nilai Negatif (-) = Mencegah Churn.")

    # --- SIMULASI INPUT ---
    st.write("---")
    st.subheader("📝 Cek Pelanggan Baru")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Lama Langganan (Bulan)", 0, 72, 12)
        monthly = st.number_input("Tagihan Bulanan ($)", 0.0, 150.0, 70.0)
    with c2:
        contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c3:
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("Analisis Risiko"):
        # Buat data dummy rata-rata
        input_data = X.mean().values.reshape(1, -1)
        
        # Mapping Input User (Manual logic biar cepat)
        # Tenure=4, Internet=7, Tech=11, Contract=14, Paperless=15, Monthly=17
        input_data[0][4] = tenure
        input_data[0][17] = monthly
        
        # Logic Mapping (Sesuai urutan abjad LabelEncoder biasanya)
        # Contract
        input_data[0][14] = 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)
        # Internet
        input_data[0][7] = 0 if internet == "DSL" else (1 if internet == "Fiber optic" else 2)
        # Tech Support
        input_data[0][11] = 2 if tech == "Yes" else (0 if tech == "No" else 1)
        # Paperless
        input_data[0][15] = 1 if paperless == "Yes" else 0

        # Prediksi
        prob = model.predict_proba(input_data)[0]
        churn_risk = prob[1]
        
        st.write("### Hasil Analisis:")
        if churn_risk > 0.5:
            st.error(f"🚨 **BERISIKO TINGGI (CHURN)**")
            st.write(f"Probabilitas Kabur: **{churn_risk*100:.1f}%**")
            st.progress(int(churn_risk*100))
            st.warning("Saran: Tawarkan kontrak jangka panjang segera!")
        else:
            st.success(f"✅ **AMAN (SETIA)**")
            st.write(f"Probabilitas Kabur: **{churn_risk*100:.1f}%**")
            st.progress(int(churn_risk*100))

# --- MENU 2: CLUSTERING ---
elif menu == "👥 Segmentasi (Clustering)":
    st.header("👥 Segmentasi Pelanggan (K-Means)")
    
    # Data Clustering
    X_cl = df[['tenure', 'MonthlyCharges']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cl)
    
    # Slider K
    k = st.sidebar.slider("Jumlah Cluster (K)", 2, 5, 3)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    X_cl['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualisasi
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Cluster', data=X_cl, palette='viridis', s=100, ax=ax)
    plt.xlabel("Lama Langganan (Tenure)")
    plt.ylabel("Tagihan Bulanan ($)")
    plt.title(f"Visualisasi {k} Cluster")
    st.pyplot(fig)
    
    # Penjelasan
    st.write("#### 💡 Interpretasi Profil:")
    cluster_stats = X_cl.groupby('Cluster').mean().reset_index()
    st.dataframe(cluster_stats)
    st.info("Gunakan menu ini untuk menentukan strategi marketing (misal: Diskon untuk pelanggan baru tagihan tinggi).")
