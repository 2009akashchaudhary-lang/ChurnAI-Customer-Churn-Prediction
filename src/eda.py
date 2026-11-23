
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# Create reports folder if it does not exist
os.makedirs("reports", exist_ok=True)

print("CLEANED DATASET SHAPE:", df.shape)

print("\nCHURN COUNTS:")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGE:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# Set graph style
sns.set_theme(style="whitegrid")

# Graph 1: Customer Churn Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("reports/churn_count.png")
plt.show()
plt.close()

# Graph 2: Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("reports/churn_by_contract.png")
plt.show()
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in reports folder.")
