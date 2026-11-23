
import pandas as pd

# Load original dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("ORIGINAL DATASET:", df.shape)

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Clean TotalCharges column
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

# Check missing values after conversion
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])

# Save cleaned dataset
df.to_csv("data/cleaned_customer_churn.csv", index=False)

print("\nCLEANING COMPLETED!")
print("Final dataset shape:", df.shape)
print("Cleaned dataset saved successfully!")
