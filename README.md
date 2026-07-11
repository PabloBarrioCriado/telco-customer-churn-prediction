# 📊 Telco Churn Predictor & Retention Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-customer-churn-prediction-pablobarriocriado.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)

## 📌 Overview
This project goes beyond standard predictive modeling by implementing a **Prescriptive Analytics System**. It is a full-stack Machine Learning application designed to predict customer churn in a telecommunications company and, more importantly, **automatically recommend targeted retention strategies** for the sales team.

## 🚀 Live Demo
You can try the fully deployed application here: **https://telco-customer-churn-prediction-pablobarriocriado.streamlit.app/**

## 💡 Business Value (The Retention Engine)
Predicting a customer leaving is only half the battle. This project includes a custom business logic engine that prescribes **Next Best Actions (NBA)** based on the customer's specific profile:
* **Contract Upgrade Tactics:** Offers targeted discounts for high-risk month-to-month customers.
* **Price Optimization:** Flags high-billing users for service audits to prevent price-based churn.
* **Value-Added Loyalty:** Automatically suggests tech support promotions, driven by the statistical insight that customers with active technical assistance have significantly higher retention rates.

## 🧠 Technical Architecture & Machine Learning
This project was built with a strict focus on statistical rigor and software engineering best practices:

* **Nested Cross-Validation:** Implemented to rigorously select the best algorithm without data leakage, avoiding the common pitfalls of a simple `train_test_split`.
* **Optimal Decision Threshold:** Instead of defaulting to a 0.5 probability, the decision threshold was mathematically optimized using **Youden's J Statistic (0.48)** to balance Sensitivity and Specificity.
* **Custom Scikit-Learn Transformers:** Built object-oriented transformers (`BaseEstimator`, `TransformerMixin`) integrated into a robust pipeline (`ServiceAggregator`) to automate data preprocessing.
* **Safe Memory Management:** Implemented runtime monkey-patching to handle specific serialization incompatibilities between strict `numpy` versions and `scikit-learn` categorical encoders.

## 🛠️ Technology Stack
* **Modeling & Machine Learning:** `scikit-learn`, `scipy`, `xgboost`, `lightgbm`
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Deployment & UI:** `streamlit`
* **Serialization:** `joblib`

## 📁 Project Structure
```text
telco-Costumer/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw data source
├── models/
│   └── churn_logistic_regression.joblib      # Trained model ready for inference
├── notebooks/
│   └── EDA_and_ML.ipynb                      # Exploratory Data Analysis and ML training
├── src/
│   ├── __pycache__/                          # Compiled Python files (ignored in Git)
│   ├── app.py                                # Streamlit web application & Retention Engine
│   └── transformer.py                        # Custom OOP transformers for the pipeline
├── .gitignore                                # Git ignore rules
├── Descripción de variables.docx             # Variable description (in spanish)
├── README.md                                 # Project documentation
└── requirements.txt                          # Strict dependencies for Cloud Deployment
```

## 💻 How to Run Locally

If you want to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/PabloBarrioCriado/telco-customer-churn-prediction.git](https://github.com/PabloBarrioCriado/telco-customer-churn-prediction.git)
   cd telco-Costumer
   ```
Install the required dependencies:
(It is recommended to use a virtual environment)

```bash
pip install -r requirements.txt
```
Launch the application:

```bash
cd src
streamlit run app.py
```
