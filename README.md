# 🏦 Explainable Credit Risk Scoring Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Imbalanced](https://img.shields.io/badge/Technique-SMOTE-purple)

A high-impact Data Science project aimed at solving the **"Black Box"** problem in financial AI. This application predicts loan default risk using an **XGBoost** classifier and uses **SHAP (SHapley Additive exPlanations)** values to explain *why* a specific applicant was approved or rejected. It also addresses the real-world challenge of class imbalance using **SMOTE**.

## 🚀 Overview

In the banking sector, accurate risk prediction is crucial, but regulatory compliance requires interpretability. You cannot simply tell a customer "Computer says No."

This project:
1.  **Simulates Real Banking Data:** Generates a synthetic dataset with realistic correlations (e.g., lower income + renting = higher risk).
2.  **Handles Class Imbalance:** Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the fact that loan defaulters are rare (only ~10-15% of the data).
3.  **Explains Decisions:** Integrates **SHAP** values to visualize feature importance for every single prediction, making the AI transparent.

## ✨ Key Features

* **⚡ Gradient Boosting:** specific implementation of XGBoost for state-of-the-art tabular data performance.
* **⚖️ Imbalance Handling:** Implementation of SMOTE to rebalance the training dataset, preventing the model from being biased toward the majority class (non-defaulters).
* **🔍 XAI (Explainable AI):** Generates individual force plots for each applicant to show which factors (Income, Age, Job History) pushed the risk score up or down.
* **📊 Interactive Dashboard:** A Streamlit-based UI for Loan Officers to input details and view real-time risk assessments.

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
* **Explainability:** SHAP
* **Web Framework:** Streamlit

## 📂 Project Structure

```text
CreditRisk/
├── src/
│   ├── data_gen.py        # Generates synthetic imbalance loan data
│   ├── model_engine.py    # Training pipeline (SMOTE + XGBoost + Serialization)
├── app.py                 # Loan Officer Dashboard (Streamlit)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation


```
💻 Installation & Usage
```
1. Clone the Repository
Bash

git clone [https://github.com/](https://github.com/)[YourUsername]/CreditRisk.git
cd CreditRisk
2. Install Dependencies
Bash

pip install -r requirements.txt
3. Run the Application
Bash

streamlit run app.py
Note: On the first run, the app will automatically generate the synthetic dataset, apply SMOTE, train the XGBoost model, and save it. This may take 30-60 seconds.
```
📊 How It Works (The Data Science Pipeline)
Data Generation: We create 5,000 applicant records. The target variable Risk_Flag is skewed (mostly 0s), mimicking real fraud/default datasets.

Preprocessing: One-Hot Encoding is avoided in favor of label encoding for tree-based efficiency.

SMOTE: We synthetically generate new examples of "Defaulters" in the training set so the model learns their patterns effectively.

Training: An XGBoost Classifier is trained on the balanced data.

Inference & Explanation: When a user inputs data, the model predicts a probability. Simultaneously, the TreeExplainer calculates SHAP values to quantify the marginal contribution of each feature to that specific prediction.

🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements.

📝 License
Distributed under the MIT License.
