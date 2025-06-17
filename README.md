![cover](cover.png)

# Heart Disease Predictor

🎯 هدف پروژه: پیش‌بینی احتمال بیماری قلبی با استفاده از ویژگی‌های پزشکی بیماران

دیتاست: https://archive.ics.uci.edu/dataset/45/heart+disease

خروجی: https://hdpredictor.streamlit.app/


## 🔧 ابزارهای استفاده‌شده

- Python (Pandas, Scikit-learn, Streamlit, XGBoost)
- Excel
- streamlit Cloud
- GitHub for version control


## 📊 مراحل پروژه

1. **پیش‌پردازش و پاک‌سازی داده‌ها**
2. **تحلیل آماری و بصری**
3. **مهندسی ویژگی‌ها (Feature Engineering)**
4. **مدل‌سازی با XGBoost و Logistic Regression**
5. **تحلیل دقیق با SHAP برای تفسیر مدل**
6. **داشبورد تعاملی با Streamlit**
7. **آماده‌سازی برای انتشار در GitHub و Streamlit Cloud**


## 🚀 خروجی‌ها

- 🌐 اپ Streamlit (`app/app_heart_disease.py`)
- 📋 تحلیل عددی، مهندسی ویژگی و بررسی مدل (Jupyter Notebook)


## 💡 ویژگی‌های کلیدی

✅ قابلیت پیش‌بینی بیماری بر اساس ۱۳ ویژگی پزشکی  
✅ تحلیل دقیق دلیل تصمیم‌گیری مدل با استفاده از SHAP  
✅ داشبورد کاملاً تعاملی برای استفاده توسط کاربران غیرتکنیکی  


## 🎯 برای اجرای اپ Streamlit:

```bash
pip install -r app/requirements.txt
streamlit run app/app_heart_disease.py
 ```

### 🌐 اجرای اپ آنلاین در Streamlit Cloud:
https://hdpredictor.streamlit.app/


## 📁 ساختار فایل‌ها
```bash
heart-disease-predictor/
│
├── 📁 data/
│   └── heart_disease.csv            # دیتای اصلی
│
├── 📁 notebook/
│   └── HDpredictor.ipynb            # تحلیل داده و مدل‌سازی (Jupyter Notebook)
│
├── 📁 app/
│   └── app_heart_disease.py         # اپ Streamlit
│   └── requirements.txt             # لیست کتابخانه‌ها
│   └── heart_disease_model.pkl      # مدل ذخیره‌شده
│   └── scaler.pkl                   # نرمال‌ساز داده
│   └── feature_order.pkl            # ترتیب صحیح ویژگی‌ها
│
├── 📁 dashboard/
│   └── dashboard-screenshot.png     # تصویر داشبورد نهایی sreamlit
│
├── 📄 README.md                     # توضیح پروژه (همین فایل)
```

## 🧑‍💻 توسعه‌دهنده

این پروژه توسط یک علاقه‌مند به تحلیل داده و یادگیری ماشین طراحی و اجرا شده  
با هدف شرکت در موقعیت "کارآموز تحلیلگر داده / دیتا ساینتیست".

✨ هدف: توسعه نمونه کار قابل ارائه، تمرین تحلیل واقعی، یادگیری مدل‌سازی حرفه‌ای و تفسیر مدل
