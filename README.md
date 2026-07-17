# 🐞 AI-Powered Software Bug Severity Prediction

## 📌 Project Overview

This project predicts the severity of software bugs using Machine Learning.

The model analyzes bug-related information such as module, bug type, crash status, affected users, reports per day, reproducibility, and device to predict the bug severity.

---

## 🚀 Features

- Data Preprocessing
- Label Encoding
- Random Forest Classifier
- GridSearchCV Hyperparameter Tuning
- Model Evaluation
- Streamlit Web Application
- Model & Encoder Serialization using Joblib

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Preprocessing
3. Label Encoding
4. Train-Test Split
5. Random Forest
6. GridSearchCV
7. Model Evaluation
8. Save Model
9. Streamlit Deployment

---

## 📂 Project Structure

```
├── app.py
├── train_model.py
├── csvbyme.csv
├── bug_severity_model.pkl
├── module_encoder.pkl
├── bug_encoder.pkl
├── crash_encoder.pkl
├── repro_encoder.pkl
├── device_encoder.pkl
├── severity_encoder.pkl
├── requirements.txt
└── README.md
```

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## ▶️ Run the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

Developed by **Lucky**