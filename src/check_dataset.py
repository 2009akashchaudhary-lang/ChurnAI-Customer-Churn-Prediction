import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nDATASET LOADED SUCCESSFULLY!")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET INFORMATION:")
print(df.info())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())