import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📉", layout="centered")

st.title("📉 Telco Customer Churn Prediction")
st.write(
    "Aplikasi ini memprediksi apakah pelanggan berpotensi **churn** (berhenti berlangganan) "
    "berdasarkan data layanan & billing."
)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "best_churn_model.joblib"

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

st.success("✅ Model berhasil dimuat.")

# =========================
# FORM INPUT
# =========================
st.subheader("🧾 Masukkan Data Pelanggan")

# Helper: opsi umum (menyesuaikan dataset Telco)
YN = ["Yes", "No"]
INTERNET = ["DSL", "Fiber optic", "No"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
GENDER = ["Male", "Female"]

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("gender", GENDER)
        SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
        Partner = st.selectbox("Partner", YN)
        Dependents = st.selectbox("Dependents", YN)
        tenure = st.number_input("tenure (bulan)", min_value=0, max_value=100, value=12, step=1)

    with col2:
        PhoneService = st.selectbox("PhoneService", YN)
        MultipleLines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("InternetService", INTERNET)
        OnlineSecurity = st.selectbox("OnlineSecurity", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("OnlineBackup", ["No internet service", "No", "Yes"])

    col3, col4 = st.columns(2)

    with col3:
        DeviceProtection = st.selectbox("DeviceProtection", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("TechSupport", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("StreamingTV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("StreamingMovies", ["No internet service", "No", "Yes"])

    with col4:
        Contract = st.selectbox("Contract", CONTRACT)
        PaperlessBilling = st.selectbox("PaperlessBilling", YN)
        PaymentMethod = st.selectbox("PaymentMethod", PAYMENT)
        MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
        TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=2000.0, step=10.0)

    submitted = st.form_submit_button("🔮 Prediksi Churn")

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

    try:
        pred = model.predict(X_input)[0]
        # pred: 0 = No churn, 1 = Churn (sesuai mapping di training)
        label = "CHURN (Yes)" if pred == 1 else "TIDAK CHURN (No)"

        st.subheader("📌 Hasil Prediksi")
        if pred == 1:
            st.error(f"⚠️ Prediksi: **{label}**")
        else:
            st.success(f"✅ Prediksi: **{label}**")

        # Jika model punya predict_proba
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_input)[0][1])
            st.write(f"Probabilitas churn: **{proba:.2%}**")
            st.progress(min(max(proba, 0.0), 1.0))

        st.caption("Catatan: hasil prediksi bergantung pada model terbaik yang sudah dilatih & disimpan.")
        with st.expander("Lihat data input"):
            st.dataframe(X_input)

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
