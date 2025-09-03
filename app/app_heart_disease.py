import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ⬇ بارگذاری مدل، اسکیلر و ترتیب ستون‌ها
model = joblib.load('app/heart_disease_model.pkl')
scaler = joblib.load('app/scaler.pkl')
feature_order = joblib.load('app/feature_order.pkl')  # لیست نام ستون‌ها با ترتیب درست

st.title(" Predicting Heart Disease ❤️")
st.write("Enter the following values:")

# دیکشنری‌ها برای گزینه‌های متنی
cp_dict = {"Ordinary pain": 0, "Angular pain": 1, "Non-cardiac": 2, "Painless": 3}
fbs_dict = {" less than 120": 0, " more than 120": 1}
restecg_dict = {"Normal": 0, "ST-T Abnormal": 1, "Left ventricular hypertrophy": 2}
exang_dict = {"No": 0, "Yes": 1}
slope_dict = {"Ascending": 0, "Flat": 1, "Descending": 2}
thal_dict = {"Normal": 1, "Suspicious": 2, "Definite": 3}

# ورودی کاربر
age = st.slider("age", 20, 100, 50)
sex = st.radio("gender", ["man", "woman"])
cp = cp_dict[st.selectbox("Type of chest pain", list(cp_dict.keys()))]
trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = fbs_dict[st.selectbox("Fasting blood sugar", list(fbs_dict.keys()))]
restecg = restecg_dict[st.selectbox("ECG result", list(restecg_dict.keys()))]
thalach = st.number_input("Maximum heart rate", 70, 250, 150)
exang = exang_dict[st.selectbox("Pain during exercise", list(exang_dict.keys()))]
oldpeak = st.number_input("Decrease in ST", 0.0, 6.0, 1.0, step=0.1)
slope = slope_dict[st.selectbox("ST Slope", list(slope_dict.keys()))]
ca = st.selectbox("Number of blocked veins", [0, 1, 2, 3, 4])
thal = thal_dict[st.selectbox("Thalassemia", list(thal_dict.keys()))]

# تبدیل جنسیت به عدد
sex = 1 if sex == "man" else 0

# تبدیل به dict و DataFrame با ترتیب دقیق
input_dict = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}
input_df = pd.DataFrame([input_dict])[feature_order]

# اسکیل و پیش‌بینی
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.write("🧪Raw Result:", prediction[0])  # for testing
    if prediction[0] == 1:
        st.error("❗ There is a possibility of heart disease.")
    else:
        st.success("✅ No signs of heart disease are seen.")
