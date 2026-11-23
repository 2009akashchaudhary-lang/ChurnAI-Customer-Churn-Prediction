
import pandas as pd
import joblib

# Load saved model files
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Sample customer details
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0
}

# Convert customer details into DataFrame
customer_df = pd.DataFrame([customer])

# Convert text values into numerical columns
customer_df = pd.get_dummies(
    customer_df,
    drop_first=True,
    dtype=int
)

# Match training dataset columns
customer_df = customer_df.reindex(
    columns=feature_names,
    fill_value=0
)

# Scale customer data
customer_scaled = scaler.transform(customer_df)

# Make prediction
prediction = model.predict(customer_scaled)
probability = model.predict_proba(customer_scaled)

# Display result
print("\n===== CUSTOMER CHURN PREDICTION =====")

if prediction[0] == 1:
    print("Prediction: Customer may LEAVE the company.")
else:
    print("Prediction: Customer may STAY with the company.")

print(
    "Churn Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)

print("=====================================")
