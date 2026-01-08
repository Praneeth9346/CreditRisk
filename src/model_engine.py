import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from src.data_gen import generate_loan_data

class CreditModel:
    def __init__(self):
        self.model = None
        self.explainer = None
        
    def train(self):
        print("⏳ Generating Data...")
        df = generate_loan_data(5000)
        
        X = df.drop('Risk_Flag', axis=1)
        y = df['Risk_Flag']
        
        # 1. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Handle Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
        # This creates fake examples of "Defaulters" so the model learns better
        print("⚖️ Balancing Data with SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # 3. Train XGBoost (Industry Standard for Tabular Data)
        print("🚀 Training XGBoost...")
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train_res, y_train_res)
        
        # 4. Evaluate
        preds = self.model.predict(X_test)
        print(f"✅ Accuracy: {accuracy_score(y_test, preds):.2%}")
        print(classification_report(y_test, preds))
        
        # 5. Save Model & Train Data (for SHAP background)
        joblib.dump(self.model, 'xgb_model.pkl')
        # We save X_train to use as a baseline for SHAP explainability later
        joblib.dump(X_train, 'X_train.pkl') 
        print("💾 Model Saved!")

    def predict_explain(self, input_data):
        # Load Resources
        if self.model is None:
            self.model = joblib.load('xgb_model.pkl')
            self.X_train = joblib.load('X_train.pkl')
            
        # Prediction
        prob = self.model.predict_proba(input_data)[:, 1][0]
        prediction = 1 if prob > 0.5 else 0
        
        # Explanation (SHAP)
        # This calculates HOW MUCH each feature contributed to the "Yes/No" decision
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(input_data)
        
        return prediction, prob, shap_values, explainer

if __name__ == "__main__":
    cm = CreditModel()
    cm.train()
