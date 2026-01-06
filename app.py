import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "best_churn_model.joblib"

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

st.title("📉 Telco Customer Churn Prediction")
st.caption("Prediksi apakah pelanggan berpotensi churn (berhenti berlangganan) berdasarkan data layanan & billing.")

with st.spinner("Memuat model..."):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

st.success("✅ Model berhasil dimuat dan siap digunakan.")

# =========================
# SIDEBAR - INFO
# =========================
with st.sidebar:
    st.header("ℹ️ Info Aplikasi")
    st.write(
        "Aplikasi ini menggunakan model machine learning untuk memprediksi churn pelanggan.\n\n"
        "**Output:**\n"
        "- 0 = Tidak churn\n"
        "- 1 = Churn\n"
    )
    st.divider()
    st.subheader("📌 Tips Pengisian")
    st.write(
        "- Isi data sesuai kondisi pelanggan.\n"
        "- Nilai **TotalCharges** biasanya berkaitan dengan lama berlangganan (*tenure*).\n"
        "- Setelah klik **Prediksi**, lihat label & probabilitas churn."
    )
    st.divider()
    st.caption("Deployment: Streamlit Cloud • Model: .joblib")

# =========================
# OPTIONS
# =========================
YN = ["Yes", "No"]
GENDER = ["Male", "Female"]
INTERNET = ["DSL", "Fiber optic", "No"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
MULTILINES = ["No phone service", "No", "Yes"]
INTERNET_DEP = ["No internet service", "No", "Yes"]

# =========================
# LAYOUT: TABS
# =========================
tab_input, tab_about = st.tabs(["🧾 Input & Prediksi", "📘 Tentang Model"])

with tab_about:
    st.subheader("📘 Tentang Model")
    st.write(
        "Model yang digunakan adalah model terbaik dari proses pemodelan dan evaluasi.\n\n"
        "**Catatan:** Untuk model sklearn, environment (Python & scikit-learn) harus sesuai agar file `.joblib` bisa dimuat."
    )
    st.info("Jika aplikasi sudah berjalan dan bisa prediksi, artinya environment sudah kompatibel ✅")

with tab_input:
    st.subheader("🧾 Masukkan Data Pelanggan")

    # ===== Form =====
    with st.form("input_form", clear_on_submit=False):
        colA, colB, colC = st.columns([1, 1, 1])

        # --- Kolom A: Demografi ---
        with colA:
            st.markdown("### 👤 Demografi")
            gender = st.selectbox("gender", GENDER, index=0)
            SeniorCitizen = st.selectbox("SeniorCitizen (0=No, 1=Yes)", [0, 1], index=0)
            Partner = st.selectbox("Partner", YN, index=1)          # default No
            Dependents = st.selectbox("Dependents", YN, index=1)    # default No
            tenure = st.number_input("tenure (bulan)", min_value=0, max_value=100, value=12, step=1)

        # --- Kolom B: Layanan ---
        with colB:
            st.markdown("### 📡 Layanan")
            PhoneService = st.selectbox("PhoneService", YN, index=0)
            MultipleLines = st.selectbox("MultipleLines", MULTILINES, index=1)
            InternetService = st.selectbox("InternetService", INTERNET, index=0)

            OnlineSecurity = st.selectbox("OnlineSecurity", INTERNET_DEP, index=1)
            OnlineBackup = st.selectbox("OnlineBackup", INTERNET_DEP, index=1)
            DeviceProtection = st.selectbox("DeviceProtection", INTERNET_DEP, index=1)
            TechSupport = st.selectbox("TechSupport", INTERNET_DEP, index=1)
            StreamingTV = st.selectbox("StreamingTV", INTERNET_DEP, index=1)
            StreamingMovies = st.selectbox("StreamingMovies", INTERNET_DEP, index=1)

        # --- Kolom C: Kontrak & Biaya ---
        with colC:
            st.markdown("### 🧾 Kontrak & Billing")
            Contract = st.selectbox("Contract", CONTRACT, index=0)
            PaperlessBilling = st.selectbox("PaperlessBilling", YN, index=0)
            PaymentMethod = st.selectbox("PaymentMethod", PAYMENT, index=0)

            MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
            TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=2000.0, step=10.0)

        st.divider()

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            submitted = st.form_submit_button("🔮 Prediksi Churn", use_container_width=True)
        with btn_col2:
            reset = st.form_submit_button("🔄 Reset", use_container_width=True)

    if reset:
        st.rerun()

    # =========================
    # PREDICTION
    # =========================
    if submitted:
        input_dict = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }

        X_input = pd.DataFrame([input_dict])

        st.markdown("## ✅ Ringkasan Data Input")
        st.dataframe(X_input, use_container_width=True)

        try:
            pred = int(model.predict(X_input)[0])
            label = "CHURN (Yes)" if pred == 1 else "TIDAK CHURN (No)"

            st.markdown("## 📌 Hasil Prediksi")

            # Card-like layout
            left, right = st.columns([1, 1])

            with left:
                if pred == 1:
                    st.error(f"⚠️ Prediksi: **{label}**")
                    st.write("Interpretasi: Pelanggan berisiko tinggi untuk berhenti berlangganan.")
                else:
                    st.success(f"✅ Prediksi: **{label}**")
                    st.write("Interpretasi: Pelanggan diprediksi tetap berlangganan.")

            with right:
                if hasattr(model, "predict_proba"):
                    proba = float(model.predict_proba(X_input)[0][1])  # probability churn
                    st.metric("Probabilitas Churn", f"{proba:.2%}")
                    st.progress(min(max(proba, 0.0), 1.0))
                else:
                    st.info("Model tidak menyediakan probabilitas (predict_proba tidak tersedia).")

            st.caption("Catatan: hasil prediksi bergantung pada model dan data input yang diberikan.")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("© Telco Churn Prediction • Streamlit App • IBRAHIM AKBARA ARGA DEWANGGA A11.2022.14417")

