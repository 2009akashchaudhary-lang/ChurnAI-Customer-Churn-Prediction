
# ChurnAI – Customer Churn Prediction & Intelligence

An end-to-end Machine Learning project that predicts customer churn, explains model predictions, analyses customer segments, and supports data-driven retention strategies.

## Project Overview

ChurnAI is a customer churn prediction application built using Python, Machine Learning, and Streamlit.

The project analyses telecom customer information and estimates the probability that a customer may leave a service provider.

It combines predictive modelling, Explainable AI, customer analytics, and retention recommendations in one interactive application.

## Key Features

- **AI Customer Prediction:** Predict customer churn using a trained Logistic Regression model.
- **Churn Probability:** Estimate the probability of customer churn.
- **Risk Classification:** Categorise customers into Low, Medium, and High risk.
- **Explainable AI:** Use SHAP to understand individual model predictions.
- **Customer Analytics:** Explore churn patterns, customer segments, and risk distribution.
- **Retention Strategy:** Generate rule-based retention recommendations.
- **Model Performance:** Compare model metrics and view evaluation results.
- **Confusion Matrix:** Analyse true and false predictions.
- **Reports Center:** Export customer intelligence data and reports.
- **PDF Report Generator:** Generate a project analysis report.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Streamlit | Web application |
| Plotly | Interactive visualisations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| SHAP | Explainable AI |
| Joblib | Model loading |
| ReportLab | PDF report generation |

## Machine Learning Model

The application uses a trained Logistic Regression model for customer churn prediction.

The model uses customer-related features such as:

- Customer tenure
- Contract type
- Internet service
- Monthly charges
- Payment method
- Customer service subscriptions

The trained model, scaler, and feature list are stored in the `models/` directory.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── app_backup.py
├── generate_report.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── cleaned_customer_churn.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── best_churn_model.pkl
│   ├── best_scaler.pkl
│   ├── best_feature_names.pkl
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── reports/
│   ├── model_comparison.csv
│   └── ChurnAI_Project_Report.pdf
│
└── src/
    ├── check_dataset.py
    ├── data_cleaning.py
    ├── eda.py
    ├── model_comparison.py
    ├── model_training.py
    ├── model_visualization.py
    └── prediction.py
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Open the project folder

```bash
cd Customer-Churn-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Generate PDF Report

To generate the project analysis report, run:

```bash
python generate_report.py
```

The PDF will be saved at:

```text
reports/ChurnAI_Project_Report.pdf
```

## Model Evaluation

The project uses the following evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

Evaluation results should be interpreted carefully. Metrics calculated on records used during training are not independent test-set results and may be optimistic.

## Explainable AI

ChurnAI uses SHAP (SHapley Additive exPlanations) to help interpret model predictions.

SHAP explanations describe how input features contribute to the model output. They do not establish causal relationships.

## Important Limitations

- Predictions are estimates, not guarantees of future customer behaviour.
- Model quality depends on the quality and representativeness of the dataset.
- Retention recommendations are rule-based suggestions.
- Independent test-set evaluation is needed for a stronger assessment of generalisation.
- This is an educational portfolio project, not a production-validated enterprise system.

## Future Improvements

- Independent model testing and validation
- Advanced machine learning model experimentation
- Model monitoring and drift detection
- Improved deployment and security
- More advanced customer retention analysis

## Project Purpose

This project demonstrates practical skills in:

- Python programming
- Data analysis
- Machine Learning
- Explainable AI
- Data visualisation
- Streamlit application development
- Model evaluation
- Report generation

---

**ChurnAI – Customer Churn Prediction & Intelligence**

Built as a Machine Learning portfolio project.
