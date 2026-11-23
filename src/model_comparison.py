
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_customer_churn.csv")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical columns into numeric columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []
best_model = None
best_scaler = None
best_features = None
best_score = -1
best_name = ""

for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        current_scaler = scaler
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        current_scaler = None

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2),
        "ROC AUC": round(roc_auc * 100, 2)
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"ROC AUC: {roc_auc * 100:.2f}%")

    # Select model using churn-class F1 score
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_scaler = current_scaler
        best_features = X.columns.tolist()
        best_name = name

# Save comparison report
results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
results_df.to_csv(REPORT_PATH, index=False)

# Save best model separately; existing model files remain untouched
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_churn_model.pkl"))
joblib.dump(best_scaler, os.path.join(MODEL_DIR, "best_scaler.pkl"))
joblib.dump(best_features, os.path.join(MODEL_DIR, "best_feature_names.pkl"))

print("\n" + "=" * 45)
print("MODEL COMPARISON COMPLETED!")
print("=" * 45)
print(results_df.to_string(index=False))
print(f"\nBest Model: {best_name}")
print(f"Best F1 Score: {best_score * 100:.2f}%")
print("\nComparison report saved in reports/model_comparison.csv")
print("Best model saved separately in models folder.")
