import streamlit as st
import joblib
model = joblib.load("bug_severity_model.pkl")
module_encoder = joblib.load("module_encoder.pkl")
bug_encoder = joblib.load("bug_encoder.pkl")
crash_encoder = joblib.load("crash_encoder.pkl")
repro_encoder = joblib.load("repro_encoder.pkl")
device_encoder = joblib.load("device_encoder.pkl")
severity_encoder = joblib.load("severity_encoder.pkl")
st.title("🐞 AI-Powered Software Bug Severity Prediction")
module = st.selectbox(
    "Select Module",
    module_encoder.classes_
)
bug_type = st.selectbox(
    "Select Bug Type",
    bug_encoder.classes_
)
crash = st.selectbox(
    "Crash?",
    crash_encoder.classes_
)
affected_users = st.number_input(
    "Affected Users",
    min_value=0,
    step=1
)
reports_per_day = st.number_input(
    "Reports Per Day",
    min_value=0,
    step=1
)
reproducible = st.selectbox(
    "Reproducible",
    repro_encoder.classes_
)
device = st.selectbox(
    "Device",
    device_encoder.classes_
)
if st.button("Predict Severity"):
    module = module_encoder.transform([module])[0]
    bug_type = bug_encoder.transform([bug_type])[0]
    crash = crash_encoder.transform([crash])[0]
    reproducible = repro_encoder.transform([reproducible])[0]
    device = device_encoder.transform([device])[0]
    user_data = [[
        module,
        bug_type,
        crash,
        affected_users,
        reports_per_day,
        reproducible,
        device
    ]]
    prediction = model.predict(user_data)
    severity = severity_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Severity: {severity}")