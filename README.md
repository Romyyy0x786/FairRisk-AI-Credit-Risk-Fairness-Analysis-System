🏦 FairRisk-AI: Credit Risk & Fairness Analysis System
FairRisk-AI is an AI-driven credit risk assessment system designed to predict the likelihood of loan defaults using the UCI German Credit Dataset. This project goes beyond simple prediction by integrating Fairness Metrics and Explainable AI (XAI) to ensure financial decisions are both accurate and ethical.

🚀 Key Features
End-to-End ML Pipeline: Includes data cleaning, label encoding, feature scaling, and model training.

Class Imbalance Handling: Implemented SMOTE (Synthetic Minority Over-sampling Technique) to address the minority class (Risk) and improve model sensitivity.

Interactive Dashboard: A real-time Streamlit frontend where users can input applicant details and get instant risk scores.

Model Transparency: Leverages SHAP values to explain individual predictions, identifying which factors (e.g., Savings, Employment) contributed most to the risk score.

Fairness Analysis: Evaluates model bias across sensitive attributes (Personal Status/Gender) using Demographic Parity and Equal Opportunity metrics.

📊 Model Performance
The system utilizes a Random Forest Classifier optimized via GridSearchCV.

ROC-AUC Score: ~0.80

Accuracy: 77%

Key Risk Drivers: Checking Account Status, Credit Duration, and Savings Account Balance.

🛠️ Tech Stack
Language: Python 3.x

Machine Learning: Scikit-learn, Imbalanced-learn

Data Processing: Pandas, NumPy

Explainability: SHAP

Deployment: Streamlit

Visualization: Matplotlib, Seaborn
