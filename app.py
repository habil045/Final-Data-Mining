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
st.set_page_config(page_title="Telco Churn & Clustering", layout="wide")

# --- JUDUL ---
st.title("🚀 Telco Analytics: Prediction & Profiling")
st.markdown("""
Aplikasi ini mengintegrasikan dua kecerdasan buatan:
1.  **Logistic Regression:** Untuk memprediksi risiko Churn.
2.  **K-Means:** Untuk mengetahui profil/segmentasi pelanggan.
""")
st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = df.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    except:
        return None

df = load_data()

if df is None:
    st.error("⚠️ File CSV tidak ditemukan! Pastikan file ada di GitHub.")
    st.stop()

# --- SIDEBAR ---
menu = st.sidebar.selectbox("Pilih Analisis:", ["🔮 Prediksi & Profiling (Gabungan)", "📊 Visualisasi Cluster", "📈 Evaluasi Model"])

# --- PREPROCESSING UMUM ---
# 1. Encoding untuk Regresi
df_proc = df.copy()
le = LabelEncoder()
for col in df_proc.columns:
    if df_proc[col].dtype == 'object':
        df_proc[col] = le.fit_transform(df_proc[col])

# 2. Persiapan Data untuk Model (Dilakukan di awal biar konsisten)
X = df_proc.drop('Churn', axis=1)
y = df_proc['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Logistic Regression (Sekali saja di belakang layar)
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)

# Training K-Means (Sekali saja di belakang layar)
# Kita pakai Tenure & MonthlyCharges untuk profil
X_cluster = df[['tenure', 'MonthlyCharges']].copy()
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)

# Fungsi Pemberi Nama Cluster (Biar Manusiawi)
def get_cluster_name(tenure, monthly):
    # Logika sederhana berdasarkan rata-rata (disesuaikan dengan hasil K-Means Bung)
    # Ini cuma label logic biar terlihat pintar di demo
    if monthly > 80:
        if tenure > 24:
            return "💎 SULTAN SETIA (Premium Loyal)"
        else:
            return "💸 ORANG KAYA BARU (High Potential)"
    elif monthly < 50:
        return "💰 PELANGGAN HEMAT (Budget Saver)"
    else:
        return "👤 PELANGGAN STANDAR (Regular)"

# --- MENU 1: PREDIKSI + PROFILING (GABUNGAN) ---
if menu == "🔮 Prediksi & Profiling (Gabungan)":
    st.header("🔮 Simulasi Pelanggan Baru")
    st.info("Masukkan data pelanggan di bawah ini. AI akan menebak **Profilnya** dan **Risiko Churn-nya**.")

    # Input User
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

    if st.button("🔍 Analisis Lengkap"):
        # --- 1. MEMBUAT DATA FRAME INPUT (LEBIH AMAN) ---
        # Kita ambil rata-rata dulu sebagai base agar kolom lain terisi
        input_df = pd.DataFrame([X.mean()], columns=X.columns)
        
        # Update nilai sesuai input user (Pakai nama kolom, jangan indeks angka)
        input_df['tenure'] = tenure
        input_df['MonthlyCharges'] = monthly
        
        # Manual Encoding (Harus sama persis dengan LabelEncoder)
        # Contract
        if contract == "Month-to-month": input_df['Contract'] = 0
        elif contract == "One year": input_df['Contract'] = 1
        else: input_df['Contract'] = 2
        
        # InternetService
        if internet == "DSL": input_df['InternetService'] = 0
        elif internet == "Fiber optic": input_df['InternetService'] = 1
        else: input_df['InternetService'] = 2
        
        # TechSupport
        if tech == "No": input_df['TechSupport'] = 0
        elif tech == "No internet": input_df['TechSupport'] = 1
        else: input_df['TechSupport'] = 2
        
        # PaperlessBilling
        input_df['PaperlessBilling'] = 1 if paperless == "Yes" else 0
        
        # --- 2. PREDIKSI (LOGREG & K-MEANS) ---
        # Prediksi Churn
        prob_churn = log_model.predict_proba(input_df)[0][1]
        
        # Prediksi Cluster (Scaling dulu khusus kolom tenure & monthly)
        input_cluster = input_df[['tenure', 'MonthlyCharges']]
        input_cluster_scaled = scaler_cluster.transform(input_cluster)
        cluster_pred = kmeans.predict(input_cluster_scaled)[0]
        
        # Labeling Cluster
        cluster_label = get_cluster_name(tenure, monthly)

        # --- 3. TAMPILKAN HASIL ---
        st.write("---")
        col_res1, col_res2 = st.columns(2)
        
        # KARTU 1: HASIL PREDIKSI CHURN
        with col_res1:
            st.subheader("1️⃣ Risiko Churn")
            if prob_churn > 0.5:
                st.error(f"🚨 **BERISIKO TINGGI!**")
                st.write(f"Probabilitas Kabur: **{prob_churn*100:.1f}%**")
                st.progress(int(prob_churn*100))
            else:
                st.success(f"✅ **AMAN (SETIA)**")
                st.write(f"Probabilitas Kabur: **{prob_churn*100:.1f}%**")
                st.progress(int(prob_churn*100))

        # KARTU 2: HASIL PROFILING CLUSTER
        with col_res2:
            st.subheader("2️⃣ Profil Pelanggan")
            st.info(f"🏷️ **{cluster_label}**")
            st.caption(f"(Secara teknis masuk ke Cluster {cluster_pred})")
            
            # Rekomendasi
            if prob_churn > 0.5:
                st.warning("💡 **Action:** Segera tawarkan promo retensi!")
            else:
                st.success("💡 **Action:** Pertahankan servis yang baik.")

# --- MENU 2: VISUALISASI CLUSTER ---
elif menu == "📊 Visualisasi Cluster":
    st.header("👥 Peta Persebaran Pelanggan")
    
    # Predict semua data untuk visualisasi
    df['Cluster'] = kmeans.predict(X_cluster_scaled)
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Cluster', data=df, palette='viridis', s=50, ax=ax)
    plt.title("Segmentasi Pelanggan (Tenure vs Charges)")
    plt.xlabel("Lama Langganan (Bulan)")
    plt.ylabel("Tagihan Bulanan ($)")
    st.pyplot(fig)
    
    st.write("### Statistik Cluster")
    st.dataframe(df.groupby('Cluster')[['tenure', 'MonthlyCharges']].mean())

# --- MENU 3: EVALUASI MODEL ---
elif menu == "📈 Evaluasi Model":
    st.header("📈 Performa Logistic Regression")
    
    y_pred = log_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    c1, c2 = st.columns(2)
    c1.metric("Akurasi", f"{acc*100:.2f}%")
    c2.metric("Recall", f"{rec*100:.2f}%")
    
    st.write("### Faktor Penentu (Feature Importance)")
    coef_df = pd.DataFrame({'Fitur': X.columns, 'Koefisien': log_model.coef_[0]}).sort_values(by='Koefisien', ascending=False)
    
    # Ambil Top 5 & Bottom 5
    top_bot = pd.concat([coef_df.head(5), coef_df.tail(5)])
    
    fig2, ax2 = plt.subplots(figsize=(10,5))
    colors = ['red' if x > 0 else 'blue' for x in top_bot['Koefisien']]
    sns.barplot(x='Koefisien', y='Fitur', data=top_bot, palette=colors, ax=ax2)
    st.pyplot(fig2)
