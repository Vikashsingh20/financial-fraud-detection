# 💳 Fraud Detection System using Machine Learning & Streamlit

## 📌 Project Overview

This project builds a Machine Learning-based Fraud Detection System trained on over **6.3 million financial transactions**.  

The model detects whether a transaction is **fraudulent or legitimate** and is deployed using a **Streamlit web application** for real-time prediction.

The dataset is highly imbalanced (only ~0.13% fraud cases), making this a real-world fraud detection problem.

---

## 📊 Dataset Information

- Total Transactions: **6,362,620**
- Fraudulent Transactions: **8,213**
- Fraud Rate: **0.13%**
- Features Used:
  - Transaction Type
  - Amount
  - Old Balance (Sender)
  - New Balance (Sender)
  - Old Balance (Receiver)
  - New Balance (Receiver)

---

## 🔎 Exploratory Data Analysis (EDA)

Performed detailed EDA including:

- Transaction type distribution
- Fraud percentage calculation
- Fraud rate by transaction type
- Log transformation for skewed amount distribution
- Boxplot comparison of fraud vs non-fraud
- Correlation matrix
- Business pattern detection (zero balance after transfer)

### 📌 Correlation Matrix

![Correlation Matrix]<img src="images/correlation_matrix.png" width="600">

### 📌 Fraud Distribution by Transaction Type

![Fraud Count]<img src="images/fraud_count.png" width="600">

---

## ⚙️ Machine Learning Pipeline

Used **Scikit-learn Pipeline** with:

- `ColumnTransformer`
- `StandardScaler` (for numerical features)
- `OneHotEncoder` (for categorical features)
- `LogisticRegression (class_weight="balanced")`

### Why class_weight="balanced"?
Because the dataset is highly imbalanced and fraud cases are rare.

---

## 📈 Model Performance
Accuracy: 94.67%



### Classification Report:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Non-Fraud (0) | 1.00 | 0.95 | 0.97 |
| Fraud (1) | 0.02 | 0.94 | 0.04 |

### Confusion Matrix:
[[1804823 101499]
[ 151 2313]]



## 🖥️ Streamlit Web Application

An interactive web app where users can:

- Select transaction type
- Enter transaction amount
- Provide sender & receiver balances
- Click predict to check fraud probability

### App Screenshot

![App Screenshot]<img src="images/app_screenshot.png" width="600">

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name



2️⃣ Install Dependencies
pip install -r requirements.txt


3️⃣ Run the Streamlit App
streamlit run fraud_detection.py


##Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Streamlit
Joblib


📂 Project Structure
├── EDA.ipynb
├── fraud_detection.py
├── fraud_detection_model.pkl
├── requirements.txt
└── README.md
🔥 Key Highlights

✔ Large-scale dataset (6.3M+ rows)
✔ Real-world imbalanced classification problem
✔ End-to-end ML pipeline
✔ Model serialization using joblib
✔ Deployed interactive web app
✔ Clean modular workflow


📌 Future Improvements
✔ Add advanced models (XGBoost, Random Forest)
✔ Add probability score display
✔ Deploy on Streamlit Cloud
✔ Add SHAP explainability
✔ Improve fraud precision




## 📂 Dataset

The dataset contains over 6.3 million financial transactions.

Due to its large size, it is not included in this repository.

You can download the dataset from the original source (https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset).




👨‍💻 Author
git
Vikash Singh
Machine Learning Enthusiast
India 🇮🇳
