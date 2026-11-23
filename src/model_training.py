
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# 2. Remove customer ID
df = df.drop("customerID", axis=1)

# 3. Convert categorical columns into numerical columns
df = pd.get_dummies(df, drop_first=True, dtype=int)

# 4. Separate input features and target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# 5. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("MODEL TRAINING COMPLETED!")
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Save model files
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("\nModel, scaler and feature names saved successfully!")
