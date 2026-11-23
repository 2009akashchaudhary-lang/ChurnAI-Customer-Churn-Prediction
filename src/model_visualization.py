
import pandas as pd
import matplotlib.pyplot as plt
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports")

# Load comparison results
df = pd.read_csv(INPUT_PATH)

# Create reports folder if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Premium black and orange theme
plt.rcParams["figure.facecolor"] = "#080808"
plt.rcParams["axes.facecolor"] = "#111111"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.titlecolor"] = "#FF6B24"

# Metrics to visualize
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

for metric in metrics:
    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        df["Model"],
        df[metric],
        color="#FF6B24",
        width=0.55
    )

    plt.title(f"Model Comparison - {metric}", fontsize=16, pad=20)
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha="right")

    # Display values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            fontsize=10
        )

    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    filename = f"model_{metric.lower().replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")

print("\nMODEL VISUALIZATION COMPLETED!")
print("All comparison graphs saved in reports folder.")
