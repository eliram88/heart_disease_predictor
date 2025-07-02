import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ⬇ بارگذاری مدل، اسکیلر و ترتیب ستون‌ها
model = joblib.load('app/heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_order = joblib.load('feature_order.pkl')  # لیست نام ستون‌ها با ترتیب درست

st.title("پیش‌بینی بیماری قلبی ❤️")
st.write("مقادیر زیر را وارد کنید:")

# دیکشنری‌ها برای گزینه‌های متنی
cp_dict = {"درد معمولی": 0, "درد زاویه‌ای": 1, "غیرقلبی": 2, "بدون درد": 3}
fbs_dict = {"کمتر از 120": 0, "بیشتر از 120": 1}
restecg_dict = {"نرمال": 0, "ST-T غیرعادی": 1, "هیپرتروفی بطن چپ": 2}
exang_dict = {"ندارد": 0, "دارد": 1}
slope_dict = {"صعودی": 0, "صاف": 1, "نزولی": 2}
thal_dict = {"نرمال": 1, "مشکوک": 2, "قطعی": 3}

# ورودی کاربر
age = st.slider("سن", 20, 100, 50)
sex = st.radio("جنسیت", ["مرد", "زن"])
cp = cp_dict[st.selectbox("نوع درد قفسه‌سینه", list(cp_dict.keys()))]
trestbps = st.number_input("فشار خون استراحت", 90, 200, 120)
chol = st.number_input("کلسترول", 100, 600, 200)
fbs = fbs_dict[st.selectbox("قند خون ناشتا", list(fbs_dict.keys()))]
restecg = restecg_dict[st.selectbox("نتیجه نوار قلب", list(restecg_dict.keys()))]
thalach = st.number_input("حداکثر ضربان قلب", 70, 250, 150)
exang = exang_dict[st.selectbox("درد حین ورزش", list(exang_dict.keys()))]
oldpeak = st.number_input("کاهش ST", 0.0, 6.0, 1.0, step=0.1)
slope = slope_dict[st.selectbox("شیب ST", list(slope_dict.keys()))]
ca = st.selectbox("تعداد رگ‌های مسدود", [0, 1, 2, 3, 4])
thal = thal_dict[st.selectbox("تالاسمی", list(thal_dict.keys()))]

# تبدیل جنسیت به عدد
sex = 1 if sex == "مرد" else 0

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

if st.button("پیش‌بینی"):
    prediction = model.predict(input_scaled)
    st.write("🧪 نتیجه خام:", prediction[0])  # برای تست
    if prediction[0] == 1:
        st.error("❗ احتمال بیماری قلبی وجود دارد")
    else:
        st.success("✅ نشانه‌ای از بیماری قلبی دیده نمی‌شود")
