
import os
import traceback
from datetime import datetime
from html import escape

import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    LongTable,
    TableStyle,
    HRFlowable,
    PageBreak
)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_customer_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "best_scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "best_feature_names.pkl")
COMPARISON_PATH = os.path.join(BASE_DIR, "reports", "model_comparison.csv")

OUTPUT_PATH = os.path.join(
    BASE_DIR, "reports", "ChurnAI_Project_Report.pdf"
)

# ==================================================
# 2. REPORT DESIGN
# ==================================================

ORANGE = colors.HexColor("#FF762E")
DARK = colors.HexColor("#171B21")
LIGHT = colors.HexColor("#F2F4F7")
GREY = colors.HexColor("#666666")
WHITE = colors.white

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="CoverTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=27,
    leading=34,
    textColor=DARK,
    alignment=TA_CENTER,
    spaceAfter=12
))

styles.add(ParagraphStyle(
    name="CoverSubtitle",
    parent=styles["Normal"],
    fontSize=11,
    leading=17,
    textColor=GREY,
    alignment=TA_CENTER,
    spaceAfter=12
))

styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=20,
    textColor=ORANGE,
    spaceBefore=16,
    spaceAfter=9
))

styles.add(ParagraphStyle(
    name="BodyTextCustom",
    parent=styles["BodyText"],
    fontSize=9,
    leading=14,
    textColor=DARK,
    spaceAfter=7
))

styles.add(ParagraphStyle(
    name="SmallNote",
    parent=styles["Normal"],
    fontSize=8,
    leading=11,
    textColor=GREY
))

styles.add(ParagraphStyle(
    name="TableCell",
    parent=styles["Normal"],
    fontSize=7,
    leading=9,
    textColor=DARK,
    wordWrap="CJK"
))

styles.add(ParagraphStyle(
    name="TableHeader",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7,
    leading=9,
    textColor=WHITE,
    wordWrap="CJK"
))

# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def make_table(data, widths=None):
    table = LongTable(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DCE2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    return table


def make_comparison_table(comparison):
    if comparison.empty or len(comparison.columns) == 0:
        return Paragraph(
            "No model comparison records were found.",
            styles["BodyTextCustom"]
        )

    available_width = A4[0] - 90
    column_count = len(comparison.columns)
    column_width = available_width / column_count

    table_data = []

    header = [
        Paragraph(escape(str(col)), styles["TableHeader"])
        for col in comparison.columns
    ]
    table_data.append(header)

    for _, row in comparison.iterrows():
        table_data.append([
            Paragraph(escape(str(value)), styles["TableCell"])
            for value in row.tolist()
        ])

    return make_table(
        table_data,
        [column_width] * column_count
    )


def add_page_number(canvas, document):
    canvas.saveState()

    page_width, _ = A4

    canvas.setStrokeColor(ORANGE)
    canvas.setLineWidth(1)
    canvas.line(40, 38, page_width - 40, 38)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)

    canvas.drawString(
        40, 25,
        "ChurnAI | Customer Churn Intelligence"
    )

    canvas.drawRightString(
        page_width - 40, 25,
        f"Page {document.page}"
    )

    canvas.restoreState()


def safe_percentage(value):
    return f"{value * 100:.2f}%"


# ==================================================
# 4. GENERATE REPORT
# ==================================================

def generate_report():

    print("\nStarting ChurnAI report generation...")

    required_files = [
        DATA_PATH,
        MODEL_PATH,
        SCALER_PATH,
        FEATURE_PATH,
        COMPARISON_PATH
    ]

    for file_path in required_files:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Required project file not found: {file_path}"
            )

    # ------------------------------------------------
    # LOAD DATA AND MODEL
    # ------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = list(joblib.load(FEATURE_PATH))

    comparison = pd.read_csv(COMPARISON_PATH)

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    if "Churn" not in df.columns:
        raise ValueError("Churn column is missing from the dataset.")

    if len(feature_names) == 0:
        raise ValueError("Model feature list is empty.")

    # ------------------------------------------------
    # DATASET ANALYSIS
    # ------------------------------------------------

    total_customers = len(df)

    churn_labels = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .map({"No": 0, "Yes": 1})
    )

    if churn_labels.isna().any():
        raise ValueError("Dataset contains invalid Churn labels.")

    actual = churn_labels.astype(int)

    churned = int((actual == 1).sum())
    retained = int((actual == 0).sum())

    churn_rate = (
        churned / total_customers * 100
        if total_customers else 0
    )

    # ------------------------------------------------
    # MODEL EVALUATION
    # ------------------------------------------------

    model_input = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    encoded = pd.get_dummies(
        model_input,
        drop_first=True,
        dtype=int
    )

    encoded = encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    scaled = scaler.transform(encoded)

    predicted = model.predict(scaled)

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain churn class 1.")

    churn_index = classes.index(1)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")

    probabilities = model.predict_proba(scaled)[:, churn_index]

    roc_auc = roc_auc_score(actual, probabilities)

    # ------------------------------------------------
    # CONTRACT ANALYSIS
    # ------------------------------------------------

    contract_rows = [
        ["Contract", "Total Customers", "Churned", "Churn Rate"]
    ]

    if "Contract" in df.columns:

        contract_analysis = (
            df.assign(ActualChurn=actual.values)
            .groupby("Contract")
            .agg(
                TotalCustomers=("ActualChurn", "size"),
                Churned=("ActualChurn", "sum")
            )
            .reset_index()
        )

        for _, row in contract_analysis.iterrows():

            rate = (
                row["Churned"] / row["TotalCustomers"] * 100
                if row["TotalCustomers"] else 0
            )

            contract_rows.append([
                str(row["Contract"]),
                str(int(row["TotalCustomers"])),
                str(int(row["Churned"])),
                f"{rate:.2f}%"
            ])

    # ------------------------------------------------
    # CREATE OUTPUT FOLDER
    # ------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    document = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=48,
        bottomMargin=55,
        title="ChurnAI Customer Churn Intelligence Report",
        author="ChurnAI Project"
    )

    story = []

    # ==================================================
    # 5. COVER PAGE
    # ==================================================

    story.append(Spacer(1, 0.7 * inch))

    story.append(Paragraph(
        "CHURNAI",
        ParagraphStyle(
            "Brand",
            parent=styles["CoverTitle"],
            fontSize=17,
            textColor=ORANGE
        )
    ))

    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "Customer Churn<br/>Intelligence Report",
        styles["CoverTitle"]
    ))

    story.append(HRFlowable(
        width="55%",
        thickness=3,
        color=ORANGE,
        hAlign="CENTER",
        spaceBefore=10,
        spaceAfter=18
    ))

    story.append(Paragraph(
        "Machine Learning | Explainable AI | Customer Analytics",
        styles["CoverSubtitle"]
    ))

    story.append(Spacer(1, 0.4 * inch))

    cover_data = [
        ["REPORT INFORMATION", "DETAILS"],
        ["Project", "ChurnAI - Customer Churn Prediction"],
        ["Model", type(model).__name__],
        ["Dataset Records", f"{total_customers:,}"],
        ["Model Input Features", str(len(feature_names))],
        ["Report Generated", datetime.now().strftime("%d %B %Y, %I:%M %p")]
    ]

    story.append(make_table(
        cover_data,
        [2.0 * inch, 3.8 * inch]
    ))

    story.append(Spacer(1, 0.45 * inch))

    story.append(Paragraph(
        "A student machine learning portfolio project demonstrating "
        "customer churn prediction, model evaluation, explainable AI "
        "and customer intelligence.",
        styles["CoverSubtitle"]
    ))

    story.append(Spacer(1, 0.5 * inch))

    story.append(Paragraph(
        "Prepared as a project analysis report. This report is based "
        "on the dataset and trained model available in the project.",
        styles["SmallNote"]
    ))

    story.append(PageBreak())

    # ==================================================
    # 6. PROJECT OVERVIEW
    # ==================================================

    story.append(Paragraph(
        "1. Project Overview",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "ChurnAI is a machine learning project designed to estimate "
        "whether a telecom customer may leave a service provider. "
        "The application combines customer data analysis, churn "
        "prediction, model evaluation and explainable AI.",
        styles["BodyTextCustom"]
    ))

    story.append(Paragraph(
        "Project objectives:",
        styles["BodyTextCustom"]
    ))

    objectives = [
        "Analyse customer churn behaviour.",
        "Predict customer churn probability using a trained ML model.",
        "Evaluate model performance using classification metrics.",
        "Explain model predictions using SHAP.",
        "Identify customer groups for retention analysis."
    ]

    for objective in objectives:
        story.append(Paragraph(
            f"&bull; {objective}",
            styles["BodyTextCustom"]
        ))

    # ==================================================
    # 7. DATASET SUMMARY
    # ==================================================

    story.append(Paragraph(
        "2. Dataset Summary",
        styles["SectionHeading"]
    ))

    dataset_table = [
        ["Dataset Metric", "Value"],
        ["Total Customers", f"{total_customers:,}"],
        ["Churned Customers", f"{churned:,}"],
        ["Retained Customers", f"{retained:,}"],
        ["Observed Churn Rate", f"{churn_rate:.2f}%"],
        ["Model Input Features", str(len(feature_names))]
    ]

    story.append(make_table(
        dataset_table,
        [3.0 * inch, 2.8 * inch]
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "The observed churn rate describes the labels in the supplied "
        "dataset. It is not a prediction of future market churn.",
        styles["SmallNote"]
    ))

    # ==================================================
    # 8. CONTRACT ANALYSIS
    # ==================================================

    story.append(Paragraph(
        "3. Churn by Contract",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The following table summarises the number of customers and "
        "observed churn rate for each contract type.",
        styles["BodyTextCustom"]
    ))

    story.append(make_table(
        contract_rows,
        [1.7 * inch, 1.3 * inch, 1.1 * inch, 1.3 * inch]
    ))

    # ==================================================
    # 9. MODEL COMPARISON
    # ==================================================

    story.append(Paragraph(
        "4. Machine Learning Model Comparison",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The project includes a model comparison report. The table "
        "below contains the available model comparison results.",
        styles["BodyTextCustom"]
    ))

    story.append(make_comparison_table(comparison))

    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"The selected model loaded for this report is "
        f"{escape(type(model).__name__)}.",
        styles["BodyTextCustom"]
    ))

    # ==================================================
    # 10. CONFUSION MATRIX
    # ==================================================

    story.append(Paragraph(
        "5. Confusion Matrix",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The confusion matrix compares actual customer labels with "
        "the predictions generated by the loaded model.",
        styles["BodyTextCustom"]
    ))

    matrix_data = [
        ["Actual / Predicted", "No Churn", "Churn"],
        ["No Churn", str(tn), str(fp)],
        ["Churn", str(fn), str(tp)]
    ]

    story.append(make_table(
        matrix_data,
        [2.2 * inch, 1.8 * inch, 1.8 * inch]
    ))

    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"True Negative: {tn:,} | "
        f"False Positive: {fp:,} | "
        f"False Negative: {fn:,} | "
        f"True Positive: {tp:,}",
        styles["BodyTextCustom"]
    ))

    # ==================================================
    # 11. MODEL METRICS
    # ==================================================

    story.append(Paragraph(
        "6. Model Evaluation Metrics",
        styles["SectionHeading"]
    ))

    metric_data = [
        ["Metric", "Score"],
        ["Accuracy", safe_percentage(accuracy)],
        ["Precision", safe_percentage(precision)],
        ["Recall", safe_percentage(recall)],
        ["F1 Score", safe_percentage(f1)],
        ["ROC AUC", f"{roc_auc:.4f}"]
    ]

    story.append(make_table(
        metric_data,
        [3.0 * inch, 2.8 * inch]
    ))

    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Important evaluation note: These metrics are calculated on "
        "the available cleaned dataset using the loaded model. This "
        "is not an independent test-set evaluation. Results may be "
        "optimistic if these records were also used during model training.",
        styles["SmallNote"]
    ))

    # ==================================================
    # 12. EXPLAINABLE AI
    # ==================================================

    story.append(Paragraph(
        "7. Explainable AI",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The project uses SHAP (SHapley Additive exPlanations) to "
        "interpret the contribution of customer features to model "
        "predictions. SHAP explanations help understand model behaviour "
        "but do not establish that a feature causes customer churn.",
        styles["BodyTextCustom"]
    ))

    # ==================================================
    # 13. CUSTOMER INTELLIGENCE
    # ==================================================

    story.append(Paragraph(
        "8. Customer Intelligence and Retention",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "The application estimates churn probability for customer "
        "records and groups them into Low, Medium and High model-risk "
        "categories. It also provides customer segments and "
        "rule-based retention suggestions.",
        styles["BodyTextCustom"]
    ))

    story.append(Paragraph(
        "Retention suggestions are decision-support recommendations. "
        "They are not automated customer communications and do not "
        "guarantee that churn will be prevented.",
        styles["BodyTextCustom"]
    ))

    # ==================================================
    # 14. LIMITATIONS
    # ==================================================

    story.append(Paragraph(
        "9. Project Limitations",
        styles["SectionHeading"]
    ))

    limitations = [
        "Predictions depend on the quality and representativeness of the dataset.",
        "Model estimates are not guarantees of future customer behaviour.",
        "Evaluation on training or previously seen records can overstate performance.",
        "SHAP explanations describe model behaviour, not causal relationships.",
        "Retention recommendations are rule-based and require human review."
    ]

    for item in limitations:
        story.append(Paragraph(
            f"&bull; {item}",
            styles["BodyTextCustom"]
        ))

    # ==================================================
    # 15. CONCLUSION
    # ==================================================

    story.append(Paragraph(
        "10. Conclusion",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "ChurnAI demonstrates an end-to-end student machine learning "
        "workflow covering data analysis, churn prediction, model "
        "evaluation, explainable AI and customer intelligence. "
        "The project provides a foundation for further improvements "
        "such as independent test-set evaluation, model monitoring "
        "and deployment.",
        styles["BodyTextCustom"]
    ))

    story.append(Spacer(1, 15))

    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=ORANGE
    ))

    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "CHURNAI | CUSTOMER CHURN INTELLIGENCE",
        ParagraphStyle(
            "FooterBrand",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=ORANGE,
            alignment=TA_CENTER
        )
    ))

    # ==================================================
    # 16. GENERATE PDF
    # ==================================================

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    if not os.path.isfile(OUTPUT_PATH):
        raise RuntimeError("PDF file was not created.")

    file_size = os.path.getsize(OUTPUT_PATH)

    if file_size == 0:
        raise RuntimeError("Generated PDF file is empty.")

    print("\n" + "=" * 55)
    print("CHURNAI PDF REPORT GENERATED SUCCESSFULLY!")
    print("=" * 55)
    print(f"Customers analysed: {total_customers:,}")
    print(f"Observed churn rate: {churn_rate:.2f}%")
    print(f"Model: {type(model).__name__}")
    print(f"PDF size: {file_size / 1024:.2f} KB")
    print(f"PDF location: {OUTPUT_PATH}")
    print("=" * 55)


# ==================================================
# 17. RUN REPORT GENERATOR
# ==================================================

if __name__ == "__main__":

    try:
        generate_report()

    except Exception as error:
        print("\nREPORT GENERATION FAILED")
        print(f"Reason: {error}")
        traceback.print_exc()
