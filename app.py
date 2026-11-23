
import os
import joblib
import shap
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_customer_churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(MODEL_DIR, "best_churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "best_scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "best_feature_names.pkl")
COMPARISON_PATH = os.path.join(REPORT_DIR, "model_comparison.csv")

LOW_RISK_LIMIT = 30
HIGH_RISK_LIMIT = 70

# ==================================================
# 2. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ChurnAI | Customer Intelligence",
    page_icon="⚡",
    layout="wide"
)

# ==================================================
# 3. PREMIUM DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background: #090B0F;
    color: #F5F5F5;
    font-family: "Inter", "Segoe UI", sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(9, 11, 15, 0.95);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1600px;
}
[data-testid="stSidebar"] {
    background: #111419;
    border-right: 1px solid #292E36;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] h2 {
    color: #FF762E !important;
    font-size: 27px !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] p {
    color: #A6ABB4;
}
[data-testid="stSidebar"] hr {
    border-color: #303640;
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 7px;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #171B21;
    border: 1px solid #272D36;
    border-radius: 9px;
    padding: 10px 12px;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    border-color: #FF762E;
    background: #20242B;
}
h1, h2, h3, h4 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
p, label {
    color: #C4C8D0;
}
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #FF762E;
    margin-bottom: 7px;
    letter-spacing: -0.8px;
    line-height: 1.25;
}
.subtitle {
    color: #9CA3AF;
    font-size: 14px;
    margin-bottom: 28px;
    line-height: 1.6;
}
.metric-card {
    background: linear-gradient(145deg, #1A1F26, #12161C);
    border: 1px solid #303741;
    border-radius: 15px;
    padding: 22px;
    min-height: 125px;
    transition: all 0.25s ease;
}
.metric-card:hover {
    border-color: #FF762E;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(255, 118, 46, 0.08);
}
.metric-label {
    color: #A9AFB9;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.6px;
}
.metric-value {
    color: #FF762E;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
    line-height: 1.2;
}
.section-card {
    background: #13171D;
    border: 1px solid #2D333D;
    border-radius: 15px;
    padding: 20px;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #FF762E, #E95D20);
    color: #FFFFFF;
    border: 1px solid #FF762E;
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #FF8B4D;
    color: #FFFFFF;
    border-color: #FF8B4D;
}
.stDownloadButton > button {
    background: #191E25;
    color: #FF8B4D;
    border: 1px solid #FF762E;
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}
div[data-testid="stForm"] {
    background: #13171D;
    border: 1px solid #303741;
    padding: 22px;
    border-radius: 15px;
}
.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #191E25;
    border-radius: 8px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #303741;
    border-radius: 10px;
    overflow: hidden;
}
hr {
    border-color: #2A3038;
}
.stProgress > div > div > div > div {
    background-color: #FF762E;
}
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .main-title {
        font-size: 26px;
    }
    .metric-card {
        padding: 16px;
        min-height: 105px;
    }
    .metric-value {
        font-size: 24px;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 4. LOAD PROJECT FILES
# ==================================================

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, scaler, features

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)

try:
    model, scaler, feature_names = load_model_files()
    df = load_dataset()
    comparison = load_comparison()
except Exception as e:
    st.error(f"Project files could not be loaded: {e}")
    st.info("Check that your model, dataset and report files exist.")
    st.stop()

# ==================================================
# 5. SHAP EXPLAINER
# ==================================================

@st.cache_resource
def load_shap_explainer():
    background_df = pd.read_csv(DATA_PATH)
    background_features = background_df.drop(
        columns=["customerID", "Churn"]
    )
    background_encoded = pd.get_dummies(
        background_features,
        drop_first=True
    )
    background_encoded = background_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )
    background_sample = background_encoded.sample(
        n=min(100, len(background_encoded)),
        random_state=42
    )
    background_scaled = scaler.transform(background_sample)
    return shap.LinearExplainer(model, background_scaled)

# ==================================================
# 6. REUSABLE UI FUNCTIONS
# ==================================================

def page_header(title, description):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="subtitle">{description}</div>',
        unsafe_allow_html=True
    )

def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def get_risk_level(probability):
    if probability < LOW_RISK_LIMIT:
        return "Low"
    if probability < HIGH_RISK_LIMIT:
        return "Medium"
    return "High"

def get_feature_label(feature):
    labels = {
        "SeniorCitizen": "Senior Citizen",
        "tenure": "Customer Tenure",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "gender_Male": "Gender: Male",
        "Partner_Yes": "Partner: Yes",
        "Dependents_Yes": "Dependents: Yes",
        "PhoneService_Yes": "Phone Service: Yes",
        "MultipleLines_No phone service": "No Phone Service",
        "MultipleLines_Yes": "Multiple Lines: Yes",
        "InternetService_Fiber optic": "Fiber Optic Internet",
        "InternetService_No": "No Internet Service",
        "OnlineSecurity_No internet service": "No Internet: Online Security",
        "OnlineSecurity_Yes": "Online Security: Yes",
        "OnlineBackup_No internet service": "No Internet: Online Backup",
        "OnlineBackup_Yes": "Online Backup: Yes",
        "DeviceProtection_No internet service": "No Internet: Device Protection",
        "DeviceProtection_Yes": "Device Protection: Yes",
        "TechSupport_No internet service": "No Internet: Tech Support",
        "TechSupport_Yes": "Tech Support: Yes",
        "StreamingTV_No internet service": "No Internet: Streaming TV",
        "StreamingTV_Yes": "Streaming TV: Yes",
        "StreamingMovies_No internet service": "No Internet: Streaming Movies",
        "StreamingMovies_Yes": "Streaming Movies: Yes",
        "Contract_One year": "One-Year Contract",
        "Contract_Two year": "Two-Year Contract",
        "PaperlessBilling_Yes": "Paperless Billing: Yes",
        "PaymentMethod_Credit card (automatic)": "Automatic Credit Card",
        "PaymentMethod_Electronic check": "Electronic Check",
        "PaymentMethod_Mailed check": "Mailed Check"
    }
    return labels.get(feature, feature.replace("_", " "))

# ==================================================
# 7. CUSTOMER INPUT VALIDATION
# ==================================================

def validate_customer_input(customer):
    errors = []

    phone = customer["PhoneService"].iloc[0]
    multiple = customer["MultipleLines"].iloc[0]
    internet = customer["InternetService"].iloc[0]

    internet_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    if phone == "No" and multiple != "No phone service":
        errors.append(
            "Phone Service is No, so Multiple Lines must be "
            "'No phone service'."
        )

    if phone == "Yes" and multiple == "No phone service":
        errors.append(
            "Phone Service is Yes, so Multiple Lines cannot be "
            "'No phone service'."
        )

    for field in internet_services:
        value = customer[field].iloc[0]
        if internet == "No" and value != "No internet service":
            errors.append(
                f"{field} must be 'No internet service' "
                "when Internet Service is No."
            )
        elif internet != "No" and value == "No internet service":
            errors.append(
                f"{field} cannot be 'No internet service' "
                "when Internet Service is DSL or Fiber optic."
            )

    tenure = customer["tenure"].iloc[0]
    total = customer["TotalCharges"].iloc[0]

    if tenure == 0 and total > 0:
        errors.append(
            "Total Charges should be 0 when customer tenure is 0 months."
        )

    return errors

# ==================================================
# 8. MODEL COEFFICIENT EXPLANATION
# ==================================================

def get_model_signals(model, input_data, features, class_index):
    if not hasattr(model, "coef_"):
        return None, None

    coefficients = model.coef_
    coefficient_row = 0 if coefficients.shape[0] == 1 else class_index
    contributions = input_data[0] * coefficients[coefficient_row]

    signals = pd.DataFrame({
        "Feature": [get_feature_label(feature) for feature in features],
        "Contribution": contributions
    })

    positive = (
        signals[signals["Contribution"] > 0]
        .sort_values("Contribution", ascending=False)
        .head(5)
    )
    negative = (
        signals[signals["Contribution"] < 0]
        .sort_values("Contribution", ascending=True)
        .head(5)
    )
    return positive, negative

# ==================================================
# 9. SHAP EXPLANATION
# ==================================================

def get_shap_explanation(input_data):
    explainer = load_shap_explainer()
    explanation = explainer(input_data)
    values = explanation.values

    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    return pd.DataFrame({
        "Feature": [get_feature_label(feature) for feature in feature_names],
        "SHAP Contribution": values[0]
    })

# ==================================================
# 10. CUSTOMER INTELLIGENCE ENGINE
# ==================================================

def assign_customer_segment(row):
    risk = row["Predicted Risk"]
    tenure = row["tenure"]
    monthly = row["MonthlyCharges"]

    if risk == "High":
        return "High Risk - Immediate Attention"
    if risk == "Medium":
        return "Medium Risk - Engagement"
    if tenure <= 12:
        return "New Customer - Onboarding"
    if tenure >= 48:
        return "Loyal Customer"
    if monthly >= df["MonthlyCharges"].median():
        return "High Value - Stable"
    return "Stable Customer"

def get_retention_recommendation(segment):
    recommendations = {
        "High Risk - Immediate Attention":
            "Prioritize customer outreach. Understand service concerns, "
            "review billing issues and offer a suitable retention solution.",
        "Medium Risk - Engagement":
            "Schedule a proactive check-in. Explain available plans and "
            "review whether the current service meets customer needs.",
        "New Customer - Onboarding":
            "Provide a helpful onboarding experience. Explain services, "
            "billing and support channels.",
        "Loyal Customer":
            "Recognize customer loyalty. Offer relevant loyalty benefits "
            "and maintain service quality.",
        "High Value - Stable":
            "Maintain service quality and monitor satisfaction. "
            "Offer relevant upgrades only when useful.",
        "Stable Customer":
            "Continue regular engagement and monitor changes in "
            "customer behaviour."
    }
    return recommendations.get(
        segment,
        "Review customer needs and provide appropriate support."
    )

@st.cache_data(show_spinner="Analyzing customer records...")
def build_customer_intelligence(data, feature_list):
    model_input = data.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )
    encoded = pd.get_dummies(
        model_input,
        drop_first=True,
        dtype=int
    )
    encoded = encoded.reindex(
        columns=list(feature_list),
        fill_value=0
    )
    scaled_data = scaler.transform(encoded)

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Churn class 1 is not available in the model.")

    churn_index = classes.index(1)
    probabilities = model.predict_proba(scaled_data)

    result = data.copy()
    result["Predicted Churn Probability (%)"] = (
        probabilities[:, churn_index] * 100
    )
    result["Predicted Risk"] = (
        result["Predicted Churn Probability (%)"].apply(get_risk_level)
    )
    result["Customer Segment"] = result.apply(
        assign_customer_segment, axis=1
    )
    result["Recommended Action"] = (
        result["Customer Segment"].apply(get_retention_recommendation)
    )
    return result

# ==================================================
# 11. CUSTOMER INTELLIGENCE DATA
# ==================================================

try:
    scored_df = build_customer_intelligence(df, tuple(feature_names))
    intelligence_error = None
except Exception as e:
    scored_df = None
    intelligence_error = str(e)

# ==================================================
# 12. BASIC DATA PREPARATION
# ==================================================

total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = total_customers - churned
churn_rate = (
    (churned / total_customers) * 100
    if total_customers > 0 else 0
)
total_monthly_revenue = df["MonthlyCharges"].sum()
churned_monthly_revenue = df.loc[
    df["Churn"] == "Yes", "MonthlyCharges"
].sum()

# ==================================================
# 13. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown("## ⚡ ChurnAI")
st.sidebar.caption("CUSTOMER INTELLIGENCE PLATFORM")
st.sidebar.divider()
st.sidebar.markdown("### WORKSPACE")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "AI Prediction",
        "Model Performance",
        "Customer Analytics",
        "Retention Strategy",
        "Reports Center"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown("### PROJECT INFORMATION")
st.sidebar.write("**Machine Learning**")
st.sidebar.caption(type(model).__name__)
st.sidebar.write("**Dataset**")
st.sidebar.caption("Telco Customer Churn")
st.sidebar.write("**Input Features**")
st.sidebar.caption(f"{len(feature_names)} model features")
st.sidebar.divider()
st.sidebar.caption("CHURNAI | STUDENT ML PROJECT")
st.sidebar.caption("Advanced Prediction Engine")
st.sidebar.caption("SHAP Explainable AI Enabled")
st.sidebar.caption("Customer Intelligence Enabled")

# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    page_header(
        "Customer Intelligence Overview",
        "Monitor customer retention, churn behaviour and business performance."
    )

    st.markdown("### BUSINESS OVERVIEW")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TOTAL CUSTOMERS", f"{total_customers:,}")
    with c2:
        metric_card("CHURNED CUSTOMERS", f"{churned:,}")
    with c3:
        metric_card("CHURN RATE", f"{churn_rate:.2f}%")
    with c4:
        metric_card("RETAINED CUSTOMERS", f"{retained:,}")

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Customer Retention")
        fig = px.pie(
            names=["Retained", "Churned"],
            values=[retained, churned],
            hole=0.62,
            color_discrete_sequence=["#FF762E", "#555B66"]
        )
        fig.update_layout(
            paper_bgcolor="#090B0F",
            plot_bgcolor="#090B0F",
            font_color="white",
            margin=dict(t=25, b=20, l=10, r=10)
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("### Churn by Contract")
        contract_data = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )
        fig = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            color_discrete_sequence=["#FF762E", "#555B66"]
        )
        fig.update_layout(
            paper_bgcolor="#090B0F",
            plot_bgcolor="#13171D",
            font_color="white",
            margin=dict(t=25, b=20, l=10, r=10)
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("### REVENUE INSIGHTS")
    r1, r2 = st.columns(2)

    with r1:
        metric_card("TOTAL MONTHLY CHARGES", f"${total_monthly_revenue:,.2f}")
    with r2:
        metric_card(
            "MONTHLY CHARGES FROM CHURNED CUSTOMERS",
            f"${churned_monthly_revenue:,.2f}"
        )

    st.caption(
        "These figures represent monthly charges in the dataset, "
        "not verified future revenue loss."
    )

    if scored_df is not None:
        st.divider()
        st.markdown("### AI CUSTOMER RISK SNAPSHOT")

        high_count = int((scored_df["Predicted Risk"] == "High").sum())
        medium_count = int((scored_df["Predicted Risk"] == "Medium").sum())
        low_count = int((scored_df["Predicted Risk"] == "Low").sum())

        a, b, c = st.columns(3)
        with a:
            metric_card("HIGH MODEL RISK", f"{high_count:,}")
        with b:
            metric_card("MEDIUM MODEL RISK", f"{medium_count:,}")
        with c:
            metric_card("LOW MODEL RISK", f"{low_count:,}")

        st.caption(
            "Risk snapshot is generated using the trained model "
            "on the available dataset records."
        )

# ==================================================
# PAGE 2: AI PREDICTION
# ==================================================

elif page == "AI Prediction":

    page_header(
        "Advanced AI Prediction Engine",
        "Estimate customer churn probability, risk level and explainable AI signals."
    )

    st.info(
        "Fill in customer information. The existing trained "
        "Logistic Regression model will be used."
    )

    with st.form("prediction_form"):

        st.markdown("### PERSONAL INFORMATION")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            senior = st.selectbox("Senior Citizen", [0, 1])
        with c3:
            partner = st.selectbox("Partner", ["Yes", "No"])

        c1, c2 = st.columns(2)
        with c1:
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        with c2:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        st.markdown("### SERVICE INFORMATION")
        c1, c2, c3 = st.columns(3)

        with c1:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
        with c2:
            multiple = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )
        with c3:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        service_options = ["Yes", "No", "No internet service"]
        c1, c2, c3 = st.columns(3)

        with c1:
            security = st.selectbox("Online Security", service_options)
        with c2:
            backup = st.selectbox("Online Backup", service_options)
        with c3:
            protection = st.selectbox("Device Protection", service_options)

        c1, c2, c3 = st.columns(3)

        with c1:
            support = st.selectbox("Tech Support", service_options)
        with c2:
            tv = st.selectbox("Streaming TV", service_options)
        with c3:
            movies = st.selectbox("Streaming Movies", service_options)

        st.markdown("### BILLING INFORMATION")
        c1, c2 = st.columns(2)

        with c1:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        with c2:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=1000.0,
                value=75.07
            )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=900.06
        )

        submitted = st.form_submit_button(
            "⚡ ANALYZE CUSTOMER CHURN",
            width="stretch"
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        validation_errors = validate_customer_input(customer)

        if validation_errors:
            st.error("Customer information needs correction.")
            for error in validation_errors:
                st.write(f"- {error}")

        else:
            try:
                encoded = pd.get_dummies(
                    customer,
                    drop_first=True,
                    dtype=int
                )
                encoded = encoded.reindex(
                    columns=feature_names,
                    fill_value=0
                )
                input_data = scaler.transform(encoded)

                classes = list(model.classes_)
                if 1 not in classes:
                    raise ValueError(
                        "The trained model does not contain churn class 1."
                    )

                churn_class_index = classes.index(1)
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                probability = float(
                    probabilities[churn_class_index] * 100
                )
                risk_level = get_risk_level(probability)

                st.divider()
                st.markdown("## CUSTOMER PREDICTION RESULT")

                if prediction == 1:
                    st.error("⚠️ MODEL PREDICTION: CUSTOMER MAY CHURN")
                else:
                    st.success("✅ MODEL PREDICTION: CUSTOMER MAY STAY")

                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_card("CHURN PROBABILITY", f"{probability:.2f}%")
                with c2:
                    metric_card("CHURN RISK SCORE", f"{probability:.0f}/100")
                with c3:
                    metric_card("RISK CLASSIFICATION", risk_level.upper())

                st.write("")
                st.markdown("### CHURN RISK METER")
                st.progress(probability / 100)

                if risk_level == "Low":
                    st.success(
                        f"Low Risk — estimated churn probability is {probability:.2f}%."
                    )
                elif risk_level == "Medium":
                    st.warning(
                        f"Medium Risk — estimated churn probability is {probability:.2f}%."
                    )
                else:
                    st.error(
                        f"High Risk — estimated churn probability is {probability:.2f}%."
                    )

                st.caption(
                    f"Risk bands: Low < {LOW_RISK_LIMIT}%, "
                    f"Medium {LOW_RISK_LIMIT}% to below {HIGH_RISK_LIMIT}%, "
                    f"High ≥ {HIGH_RISK_LIMIT}%."
                )

                st.divider()
                st.markdown("## MODEL COEFFICIENT EXPLANATION")
                st.write(
                    "These signals represent feature contributions "
                    "to the Logistic Regression model's linear churn score."
                )

                positive, negative = get_model_signals(
                    model, input_data, feature_names, churn_class_index
                )

                if positive is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Signals Increasing Churn Score")
                        if not positive.empty:
                            st.dataframe(
                                positive.round(4),
                                hide_index=True,
                                width="stretch"
                            )
                        else:
                            st.write("No positive contributions found.")

                    with c2:
                        st.markdown("### Signals Decreasing Churn Score")
                        if not negative.empty:
                            st.dataframe(
                                negative.round(4),
                                hide_index=True,
                                width="stretch"
                            )
                        else:
                            st.write("No negative contributions found.")

                st.caption(
                    "Coefficient contributions are model-based associations, "
                    "not causal effects or SHAP values."
                )

                st.divider()
                st.markdown("## SHAP EXPLAINABLE AI")
                st.write(
                    "SHAP explains how each customer feature contributes "
                    "to the model output compared with a background group."
                )

                try:
                    shap_df = get_shap_explanation(input_data)

                    shap_positive = (
                        shap_df[shap_df["SHAP Contribution"] > 0]
                        .sort_values(
                            "SHAP Contribution", ascending=False
                        )
                        .head(5)
                    )
                    shap_negative = (
                        shap_df[shap_df["SHAP Contribution"] < 0]
                        .sort_values(
                            "SHAP Contribution", ascending=True
                        )
                        .head(5)
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Increasing Churn Contribution")
                        if not shap_positive.empty:
                            st.dataframe(
                                shap_positive.round(4),
                                hide_index=True,
                                width="stretch"
                            )
                        else:
                            st.info("No positive SHAP contributions found.")

                    with c2:
                        st.markdown("### Decreasing Churn Contribution")
                        if not shap_negative.empty:
                            st.dataframe(
                                shap_negative.round(4),
                                hide_index=True,
                                width="stretch"
                            )
                        else:
                            st.info("No negative SHAP contributions found.")

                    st.markdown("### TOP CUSTOMER CHURN DRIVERS")
                    top_features = shap_df.copy()
                    top_features["Absolute Contribution"] = (
                        top_features["SHAP Contribution"].abs()
                    )
                    top_features = (
                        top_features
                        .sort_values(
                            "Absolute Contribution", ascending=False
                        )
                        .head(10)
                        .sort_values("SHAP Contribution")
                    )

                    fig = px.bar(
                        top_features,
                        x="SHAP Contribution",
                        y="Feature",
                        orientation="h",
                        color="SHAP Contribution",
                        color_continuous_scale=["#777777", "#FF762E"]
                    )
                    fig.update_layout(
                        paper_bgcolor="#090B0F",
                        plot_bgcolor="#13171D",
                        font_color="white",
                        xaxis_title="SHAP Contribution (Log-Odds)",
                        yaxis_title="Customer Features",
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.success("SHAP explanation generated successfully!")
                    st.caption(
                        "Positive SHAP values move the model output toward "
                        "churn. Negative values move it away. They are "
                        "log-odds contributions, not percentage-point changes."
                    )

                except Exception as shap_error:
                    st.warning(
                        f"SHAP explanation could not be generated: {shap_error}"
                    )
                    st.info("The customer prediction is still available.")

                st.divider()
                st.markdown("### CUSTOMER INPUT SUMMARY")
                st.dataframe(
                    customer,
                    hide_index=True,
                    width="stretch"
                )
                st.caption(
                    "This is a machine-learning estimate, not a guarantee "
                    "of future customer behaviour."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ==================================================
# PAGE 3: MODEL PERFORMANCE + CONFUSION MATRIX
# ==================================================

elif page == "Model Performance":

    page_header(
        "Machine Learning Performance",
        "Compare trained machine learning models and evaluate customer churn predictions."
    )

    st.markdown("### MODEL COMPARISON")
    st.dataframe(comparison, width="stretch", hide_index=True)

    selected_metric = st.selectbox(
        "Select Performance Metric",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    )

    fig = px.bar(
        comparison,
        x="Model",
        y=selected_metric,
        text=selected_metric,
        color="Model",
        color_discrete_sequence=[
            "#FF762E", "#A64B20", "#666666", "#D98B5F"
        ]
    )
    fig.update_layout(
        paper_bgcolor="#090B0F",
        plot_bgcolor="#13171D",
        font_color="white",
        showlegend=False,
        yaxis_title="Score (%)",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # CONFUSION MATRIX

    st.markdown("### CONFUSION MATRIX")
    st.write(
        "Compare actual customer churn labels with predictions "
        "from the selected trained Logistic Regression model."
    )

    try:
        evaluation_input = df.drop(
            columns=["customerID", "Churn"],
            errors="ignore"
        )

        evaluation_encoded = pd.get_dummies(
            evaluation_input,
            drop_first=True,
            dtype=int
        )
        evaluation_encoded = evaluation_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        evaluation_scaled = scaler.transform(evaluation_encoded)

        actual_labels = (
            df["Churn"]
            .astype(str)
            .str.strip()
            .map({"No": 0, "Yes": 1})
        )

        if actual_labels.isna().any():
            raise ValueError(
                "Dataset contains unknown or missing Churn labels."
            )

        actual_labels = actual_labels.astype(int)
        predicted_labels = model.predict(evaluation_scaled)

        cm = confusion_matrix(
            actual_labels,
            predicted_labels,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Churn", "Actual: Churn"],
            columns=["Predicted: No Churn", "Predicted: Churn"]
        )

        fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[
                "#171B21", "#A64B20", "#FF762E"
            ],
            labels={
                "x": "Predicted Class",
                "y": "Actual Class",
                "color": "Customers"
            }
        )
        fig.update_layout(
            title="Customer Churn Confusion Matrix",
            paper_bgcolor="#090B0F",
            plot_bgcolor="#13171D",
            font_color="white",
            coloraxis_showscale=False,
            height=450
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Rows represent actual classes and columns represent predicted "
            "classes. Class 0 = No Churn; Class 1 = Churn."
        )

        st.markdown("### CONFUSION MATRIX BREAKDOWN")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("TRUE NEGATIVE (TN)", f"{tn:,}")
        with c2:
            metric_card("FALSE POSITIVE (FP)", f"{fp:,}")
        with c3:
            metric_card("FALSE NEGATIVE (FN)", f"{fn:,}")
        with c4:
            metric_card("TRUE POSITIVE (TP)", f"{tp:,}")

        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total * 100 if total else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) else 0

        st.divider()
        st.markdown("### MODEL EVALUATION METRICS")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("ACCURACY", f"{accuracy:.2f}%")
        with m2:
            metric_card("PRECISION", f"{precision:.2f}%")
        with m3:
            metric_card("RECALL", f"{recall:.2f}%")
        with m4:
            metric_card("SPECIFICITY", f"{specificity:.2f}%")

        st.caption(
            "These metrics evaluate predictions on the available dataset. "
            "They are not an independent test-set evaluation and may be "
            "optimistic because the model may have been trained using "
            "some of these same records."
        )

    except Exception as e:
        st.error(f"Confusion Matrix could not be generated: {e}")

    st.divider()
    st.markdown("### GENERATED COMPARISON GRAPHS")

    graph_files = [
        ("Accuracy", "model_accuracy.png"),
        ("Precision", "model_precision.png"),
        ("Recall", "model_recall.png"),
        ("F1 Score", "model_f1_score.png"),
        ("ROC AUC", "model_roc_auc.png")
    ]

    for label, filename in graph_files:
        path = os.path.join(REPORT_DIR, filename)
        if os.path.exists(path):
            st.markdown(f"#### {label}")
            st.image(path, width="stretch")

# ==================================================
# PAGE 4: ADVANCED CUSTOMER ANALYTICS
# ==================================================

elif page == "Customer Analytics":

    page_header(
        "Customer Analytics",
        "Explore customer behaviour, predicted risk and churn patterns."
    )

    if scored_df is None:
        st.error(
            "Customer Intelligence could not be generated: "
            f"{intelligence_error}"
        )
    else:
        st.markdown("### CUSTOMER RISK OVERVIEW")

        high_count = int((scored_df["Predicted Risk"] == "High").sum())
        medium_count = int((scored_df["Predicted Risk"] == "Medium").sum())
        low_count = int((scored_df["Predicted Risk"] == "Low").sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("HIGH RISK CUSTOMERS", f"{high_count:,}")
        with c2:
            metric_card("MEDIUM RISK CUSTOMERS", f"{medium_count:,}")
        with c3:
            metric_card("LOW RISK CUSTOMERS", f"{low_count:,}")

        st.divider()
        st.markdown("### RISK DISTRIBUTION")

        risk_counts = (
            scored_df["Predicted Risk"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk Level")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Customers",
            hole=0.55,
            color="Risk Level",
            color_discrete_map={
                "Low": "#4C9B78",
                "Medium": "#E0A34A",
                "High": "#E95D4F"
            }
        )
        fig.update_layout(
            paper_bgcolor="#090B0F",
            font_color="white"
        )
        st.plotly_chart(fig, width="stretch")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### PREDICTED RISK BY CONTRACT")
            contract_risk = (
                scored_df.groupby(["Contract", "Predicted Risk"])
                .size()
                .reset_index(name="Customers")
            )
            fig = px.bar(
                contract_risk,
                x="Contract",
                y="Customers",
                color="Predicted Risk",
                barmode="group",
                color_discrete_map={
                    "Low": "#4C9B78",
                    "Medium": "#E0A34A",
                    "High": "#E95D4F"
                }
            )
            fig.update_layout(
                paper_bgcolor="#090B0F",
                plot_bgcolor="#13171D",
                font_color="white"
            )
            st.plotly_chart(fig, width="stretch")

        with c2:
            st.markdown("### PREDICTED RISK BY INTERNET SERVICE")
            internet_risk = (
                scored_df.groupby(["InternetService", "Predicted Risk"])
                .size()
                .reset_index(name="Customers")
            )
            fig = px.bar(
                internet_risk,
                x="InternetService",
                y="Customers",
                color="Predicted Risk",
                barmode="group",
                color_discrete_map={
                    "Low": "#4C9B78",
                    "Medium": "#E0A34A",
                    "High": "#E95D4F"
                }
            )
            fig.update_layout(
                paper_bgcolor="#090B0F",
                plot_bgcolor="#13171D",
                font_color="white"
            )
            st.plotly_chart(fig, width="stretch")

        st.divider()
        st.markdown("### CUSTOMER SEGMENTATION")

        segment_counts = (
            scored_df["Customer Segment"]
            .value_counts()
            .reset_index()
        )
        segment_counts.columns = ["Customer Segment", "Customers"]

        fig = px.bar(
            segment_counts,
            x="Customer Segment",
            y="Customers",
            color="Customers",
            color_continuous_scale=["#A64B20", "#FF762E"]
        )
        fig.update_layout(
            paper_bgcolor="#090B0F",
            plot_bgcolor="#13171D",
            font_color="white",
            xaxis_tickangle=-25,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, width="stretch")

        st.divider()
        st.markdown("### CUSTOMER FILTERS")

        f1, f2, f3 = st.columns(3)
        with f1:
            selected_risk = st.selectbox(
                "Filter by Risk",
                ["All", "High", "Medium", "Low"]
            )
        with f2:
            selected_contract = st.selectbox(
                "Filter by Contract",
                ["All"] + sorted(
                    scored_df["Contract"].dropna().unique().tolist()
                )
            )
        with f3:
            selected_segment = st.selectbox(
                "Filter by Customer Segment",
                ["All"] + sorted(
                    scored_df["Customer Segment"].unique().tolist()
                )
            )

        filtered = scored_df.copy()

        if selected_risk != "All":
            filtered = filtered[
                filtered["Predicted Risk"] == selected_risk
            ]
        if selected_contract != "All":
            filtered = filtered[
                filtered["Contract"] == selected_contract
            ]
        if selected_segment != "All":
            filtered = filtered[
                filtered["Customer Segment"] == selected_segment
            ]

        st.write(f"Customers found: **{len(filtered):,}**")

        display_columns = [
            col for col in [
                "customerID", "tenure", "Contract", "InternetService",
                "MonthlyCharges", "Churn",
                "Predicted Churn Probability (%)", "Predicted Risk",
                "Customer Segment"
            ] if col in filtered.columns
        ]

        st.dataframe(
            filtered[display_columns].round(2),
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Predicted risk is generated by applying the trained model "
            "to these dataset records. It is not an independently "
            "validated future-risk assessment."
        )

        st.divider()
        st.markdown("### TENURE DISTRIBUTION")

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=30,
            barmode="overlay",
            color_discrete_sequence=["#FF762E", "#777777"]
        )
        fig.update_layout(
            paper_bgcolor="#090B0F",
            plot_bgcolor="#13171D",
            font_color="white"
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("### ORIGINAL CUSTOMER DATA PREVIEW")
        st.dataframe(
            df.head(20),
            width="stretch",
            hide_index=True
        )

# ==================================================
# PAGE 5: RETENTION STRATEGY
# ==================================================

elif page == "Retention Strategy":

    page_header(
        "Customer Retention Strategy",
        "Identify customers needing attention and explore suggested retention actions."
    )

    if scored_df is None:
        st.error(
            "Customer Intelligence could not be generated: "
            f"{intelligence_error}"
        )
    else:
        high_risk = scored_df[
            scored_df["Predicted Risk"] == "High"
        ].copy()
        medium_risk = scored_df[
            scored_df["Predicted Risk"] == "Medium"
        ].copy()
        at_risk = scored_df[
            scored_df["Predicted Risk"].isin(["High", "Medium"])
        ].copy()

        st.markdown("### RETENTION PRIORITY")
        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("HIGH PRIORITY", f"{len(high_risk):,}")
        with c2:
            metric_card("MEDIUM PRIORITY", f"{len(medium_risk):,}")
        with c3:
            metric_card("TOTAL AT-RISK", f"{len(at_risk):,}")

        st.divider()
        st.markdown("### RETENTION ACTION PLAN")

        strategy = pd.DataFrame([
            {
                "Customer Segment": "High Risk - Immediate Attention",
                "Priority": "Urgent",
                "Suggested Action": (
                    "Contact the customer, understand service concerns "
                    "and review suitable retention options."
                )
            },
            {
                "Customer Segment": "Medium Risk - Engagement",
                "Priority": "High",
                "Suggested Action": (
                    "Schedule a proactive check-in and explain "
                    "available plans and support options."
                )
            },
            {
                "Customer Segment": "New Customer - Onboarding",
                "Priority": "Medium",
                "Suggested Action": (
                    "Provide onboarding guidance and explain billing "
                    "and service features."
                )
            },
            {
                "Customer Segment": "Loyal Customer",
                "Priority": "Normal",
                "Suggested Action": (
                    "Recognize loyalty and maintain consistent service quality."
                )
            },
            {
                "Customer Segment": "High Value - Stable",
                "Priority": "Normal",
                "Suggested Action": (
                    "Monitor satisfaction and offer relevant upgrades "
                    "when appropriate."
                )
            },
            {
                "Customer Segment": "Stable Customer",
                "Priority": "Normal",
                "Suggested Action": (
                    "Continue regular engagement and monitor behaviour."
                )
            }
        ])

        st.dataframe(strategy, width="stretch", hide_index=True)

        st.divider()
        st.markdown("### HIGH-RISK CUSTOMER LIST")

        if high_risk.empty:
            st.success("No customers fall into the High Risk category.")
        else:
            high_display = high_risk[
                [
                    "customerID", "tenure", "Contract", "MonthlyCharges",
                    "Predicted Churn Probability (%)", "Customer Segment",
                    "Recommended Action"
                ]
            ].sort_values(
                "Predicted Churn Probability (%)",
                ascending=False
            )
            st.dataframe(
                high_display.round(2),
                width="stretch",
                hide_index=True
            )

        st.divider()
        st.markdown("### MEDIUM-RISK CUSTOMER LIST")

        if medium_risk.empty:
            st.success("No customers fall into the Medium Risk category.")
        else:
            medium_display = medium_risk[
                [
                    "customerID", "tenure", "Contract", "MonthlyCharges",
                    "Predicted Churn Probability (%)", "Customer Segment",
                    "Recommended Action"
                ]
            ].sort_values(
                "Predicted Churn Probability (%)",
                ascending=False
            )
            st.dataframe(
                medium_display.round(2),
                width="stretch",
                hide_index=True
            )

        st.divider()
        st.markdown("### RETENTION CAMPAIGN EXPORT")

        campaign_data = at_risk[
            [
                "customerID", "tenure", "Contract", "MonthlyCharges",
                "Predicted Churn Probability (%)", "Predicted Risk",
                "Customer Segment", "Recommended Action"
            ]
        ].sort_values(
            "Predicted Churn Probability (%)",
            ascending=False
        )

        st.write(
            f"Customers included in this retention list: "
            f"**{len(campaign_data):,}**"
        )

        campaign_csv = campaign_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download Retention Campaign CSV",
            data=campaign_csv,
            file_name="churnai_retention_campaign.csv",
            mime="text/csv"
        )

        st.info(
            "Retention recommendations are rule-based suggestions for "
            "this student project. They are not automated customer "
            "communications or guaranteed churn-prevention actions."
        )

# ==================================================
# PAGE 6: REPORTS CENTER
# ==================================================

elif page == "Reports Center":

    page_header(
        "Reports Center",
        "Download customer data, model reports and retention analysis."
    )

    st.markdown("### MODEL COMPARISON REPORT")
    st.dataframe(comparison, width="stretch", hide_index=True)

    csv_data = comparison.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Model Comparison CSV",
        data=csv_data,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    st.divider()
    st.markdown("### CLEANED CUSTOMER DATASET")

    dataset_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Customer Dataset",
        data=dataset_csv,
        file_name="cleaned_customer_churn.csv",
        mime="text/csv"
    )

    st.divider()
    st.markdown("### CUSTOMER INTELLIGENCE REPORT")

    if scored_df is not None:
        intelligence_csv = scored_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download Customer Intelligence CSV",
            data=intelligence_csv,
            file_name="churnai_customer_intelligence.csv",
            mime="text/csv"
        )

        retention_df = scored_df[
            scored_df["Predicted Risk"].isin(["High", "Medium"])
        ].copy()

        retention_csv = retention_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download At-Risk Customer CSV",
            data=retention_csv,
            file_name="churnai_at_risk_customers.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            f"Customer Intelligence report unavailable: {intelligence_error}"
        )

    st.divider()
    st.markdown("### PROJECT SUMMARY")

    st.write(f"Total Customers Analysed: {total_customers:,}")
    st.write(f"Churned Customers: {churned:,}")
    st.write(f"Customer Churn Rate: {churn_rate:.2f}%")
    st.write("Models Evaluated: 4")
    st.write("Best Model: Logistic Regression")
    st.write("Explainable AI: SHAP")
    st.write("Customer Intelligence: Enabled")

# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#858B95;
        padding:12px 0;
        font-size:12px;
    ">
        <strong style="color:#FF762E;">⚡ ChurnAI</strong>
        &nbsp; | &nbsp; Customer Churn Prediction
        &nbsp; | &nbsp; Machine Learning Portfolio Project
        &nbsp; | &nbsp; SHAP Explainable AI
        &nbsp; | &nbsp; Customer Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
