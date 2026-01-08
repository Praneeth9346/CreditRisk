import pandas as pd
import numpy as np
import random

def generate_loan_data(n_samples=5000):
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        # Feature Generation
        income = np.random.randint(20000, 150000)
        age = np.random.randint(21, 70)
        experience = np.random.randint(0, 40)
        married = np.random.choice([0, 1])
        house_ownership = np.random.choice([0, 1, 2]) # 0=Rent, 1=Own, 2=Mortgage
        car_ownership = np.random.choice([0, 1])
        profession = np.random.randint(0, 50) # Encoded profession ID
        current_job_years = np.random.randint(0, experience + 1)
        house_years = np.random.randint(0, 20)
        
        # Risk Logic (Simulating Real World)
        # Lower income, renting, less experience = Higher Risk
        risk_score = (150000 - income) * 0.0005 + \
                     (70 - age) * 0.5 + \
                     (1 if house_ownership == 0 else 0) * 20 + \
                     (1 if current_job_years < 2 else 0) * 15
        
        # Add noise
        risk_score += np.random.normal(0, 10)
        
        # Threshold for default (Target)
        # We want roughly 10-15% defaults to simulate imbalance
        default = 1 if risk_score > 110 else 0
        
        data.append([income, age, experience, married, house_ownership, car_ownership, profession, current_job_years, house_years, default])
        
    cols = ['Income', 'Age', 'Experience', 'Married', 'House_Ownership', 'Car_Ownership', 'Profession', 'Current_Job_Years', 'House_Years', 'Risk_Flag']
    df = pd.DataFrame(data, columns=cols)
    return df

if __name__ == "__main__":
    df = generate_loan_data()
    df.to_csv("loan_data.csv", index=False)
    print("✅ Synthetic Loan Data Created!")
