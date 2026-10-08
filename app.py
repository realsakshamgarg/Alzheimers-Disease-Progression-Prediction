import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Alzheimer's Disease Progression Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

RF_PATH = BASE_DIR / "alzheimer_rf_model.pkl"
SCALER_PATH = BASE_DIR / "kmeans_scaler.pkl"
KMEANS_PATH = BASE_DIR / "kmeans_model.pkl"
LABELS_PATH = BASE_DIR / "severity_labels.pkl"


# ============================================================
# UI STYLING
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #0b0f17;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.hero {
    background: linear-gradient(135deg, #151d31, #101522);
    border: 1px solid #293650;
    border-radius: 18px;

    padding: 36px 32px;
    margin-top: 10px;
    margin-bottom: 25px;

    width: 100%;
    box-sizing: border-box;

    overflow: visible;
}

.hero-title {
    font-size: 38px;
    font-weight: 750;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #a9b3c4;
    font-size: 16px;
}

.section {
    font-size: 26px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 18px;
    line-height: 1.3;
}

.input-box {
    background: #111722;
    border: 1px solid #252f43;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
}

.result-box {
    background: #111722;
    border: 1px solid #2b3850;
    border-radius: 14px;
    padding: 25px;
    min-height: 180px;
}

.result-label {
    color: #9ca8ba;
    font-size: 14px;
    margin-bottom: 8px;
}

.result-value {
    font-size: 34px;
    font-weight: 750;
}

.result-stage {
    font-size: 19px;
    font-weight: 650;
    margin-top: 8px;
}

.report-box {
    background: #111722;
    border: 1px solid #293650;
    border-radius: 14px;
    padding: 25px;
    margin-top: 15px;
}

.disclaimer {
    background: #151515;
    border: 1px solid #55451d;
    border-radius: 12px;
    padding: 20px;
    margin-top: 28px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    random_forest_model = joblib.load(RF_PATH)
    kmeans_scaler = joblib.load(SCALER_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)
    severity_labels = joblib.load(LABELS_PATH)

    return (
        random_forest_model,
        kmeans_scaler,
        kmeans_model,
        severity_labels
    )


try:

    (
        random_forest_model,
        kmeans_scaler,
        kmeans_model,
        severity_labels
    ) = load_models()

except Exception as error:

    st.error("The trained model files could not be loaded.")
    st.code(str(error))
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_severity_from_score(score):

    if score < 25:
        return "Early Phase / Baseline"

    elif score < 50:
        return "Mild Cognitive Impairment"

    elif score < 75:
        return "Symptomatic Alzheimer's"

    else:
        return "Severe Progression"


def get_cluster_label(cluster_number):

    try:

        if isinstance(severity_labels, dict):

            return severity_labels.get(
                cluster_number,
                f"Cluster {cluster_number + 1}"
            )

        return severity_labels[cluster_number]

    except Exception:

        return f"Cluster {cluster_number + 1}"


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🧠 Alzheimer's Disease Progression Predictor
</div>

<div class="hero-subtitle">
Machine Learning based clinical severity prediction and patient risk clustering.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PATIENT DETAILS
# ============================================================

st.markdown(
    '<div class="section">Patient Clinical Details</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the patient's clinical information below and generate "
    "a machine learning based progression report."
)


st.markdown(
    '<div class="input-box">',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    age = st.slider(
        "Age",
        min_value=60,
        max_value=95,
        value=74
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=15.0,
        max_value=45.0,
        value=27.0,
        step=0.1
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        ["No", "Yes"]
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    alcohol_consumption = st.selectbox(
        "Alcohol Consumption",
        ["No", "Yes"]
    )

    mmse_score = st.slider(
        "Mini-Mental State Examination (MMSE) Score",
        min_value=0.0,
        max_value=30.0,
        value=20.8,
        step=0.1
    )

    functional_skill = st.slider(
        "Functional Skill Score",
        min_value=0.0,
        max_value=10.0,
        value=6.3,
        step=0.1
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown(
    '<div class="section">Generate Patient Report</div>',
    unsafe_allow_html=True
)


generate_report = st.button(
    "🔍 Analyze Patient and Generate Report",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if generate_report:

    smoking_value = 1 if smoking_status == "Yes" else 0

    alcohol_value = 1 if alcohol_consumption == "Yes" else 0


    # IMPORTANT:
    # These columns must match the feature order
    # used while training the model.

    patient_data = pd.DataFrame(
        [[
            age,
            bmi,
            smoking_value,
            alcohol_value,
            mmse_score,
            functional_skill
        ]],
        columns=[
            "Age",
            "BMI",
            "Smoking_Status",
            "Alcohol_Consumption",
            "MMSE_Score",
            "Functional_Skill"
        ]
    )


    try:

        # ----------------------------------------------------
        # RANDOM FOREST PREDICTION
        # ----------------------------------------------------

        predicted_score = random_forest_model.predict(
            patient_data
        )[0]


        predicted_score = float(
            np.clip(
                predicted_score,
                0,
                100
            )
        )


        predicted_severity = get_severity_from_score(
            predicted_score
        )


        # ----------------------------------------------------
        # K-MEANS PREDICTION
        # ----------------------------------------------------

        scaled_patient_data = kmeans_scaler.transform(
            patient_data
        )


        cluster_number = int(
            kmeans_model.predict(
                scaled_patient_data
            )[0]
        )


        cluster_label = get_cluster_label(
            cluster_number
        )


        # ----------------------------------------------------
        # SAVE REPORT
        # ----------------------------------------------------

        st.session_state["patient_report"] = {

            "age": age,

            "bmi": bmi,

            "smoking": smoking_status,

            "alcohol": alcohol_consumption,

            "mmse": mmse_score,

            "functional_skill": functional_skill,

            "predicted_score": predicted_score,

            "predicted_severity": predicted_severity,

            "cluster_number": cluster_number + 1,

            "cluster_label": cluster_label
        }


    except Exception as error:

        st.error(
            "The prediction could not be generated."
        )

        st.code(str(error))


# ============================================================
# REPORT
# ============================================================

if "patient_report" in st.session_state:

    report = st.session_state["patient_report"]


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.markdown(
        '<div class="section">Prediction Result</div>',
        unsafe_allow_html=True
    )


    result_col1, result_col2 = st.columns(2)


    # --------------------------------------------------------
    # PREDICTION SCORE
    # --------------------------------------------------------

    with result_col1:

        st.markdown("""
        <div class="result-box">

        <div class="result-label">
        Predicted Alzheimer's Disease Progression Severity
        </div>

        """, unsafe_allow_html=True)


        st.markdown(
            f"""
            <div class="result-value">
            {report["predicted_score"]:.2f} / 100
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="result-stage">
            {report["predicted_severity"]}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # K-MEANS CLUSTER
    # --------------------------------------------------------

    with result_col2:

        st.markdown("""
        <div class="result-box">

        <div class="result-label">
        K-Means Patient Risk Cluster
        </div>

        """, unsafe_allow_html=True)


        st.markdown(
            f"""
            <div class="result-value">
            Cluster {report["cluster_number"]}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="result-stage">
            {report["cluster_label"]}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section">Patient Information Summary</div>',
        unsafe_allow_html=True
    )


    summary_col1, summary_col2, summary_col3 = st.columns(3)


    with summary_col1:

        st.metric(
            "Age",
            f'{report["age"]} years'
        )

        st.metric(
            "Body Mass Index",
            f'{report["bmi"]:.1f}'
        )


    with summary_col2:

        st.metric(
            "Mini-Mental State Examination Score",
            f'{report["mmse"]:.1f} / 30'
        )

        st.metric(
            "Functional Skill Score",
            f'{report["functional_skill"]:.1f} / 10'
        )


    with summary_col3:

        st.metric(
            "Smoking Status",
            report["smoking"]
        )

        st.metric(
            "Alcohol Consumption",
            report["alcohol"]
        )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section">Model Performance</div>',
        unsafe_allow_html=True
    )


    performance_col1, performance_col2, performance_col3 = st.columns(3)


    with performance_col1:

        st.metric(
            "Coefficient of Determination (R²)",
            "98.22%"
        )


    with performance_col2:

        st.metric(
            "Root Mean Squared Error (RMSE)",
            "2.14"
        )


    with performance_col3:

        st.metric(
            "Mean Absolute Error (MAE)",
            "1.70"
        )


    # ========================================================
    # SEVERITY SCALE
    # ========================================================

    st.markdown(
        '<div class="section">Disease Progression Severity Scale</div>',
        unsafe_allow_html=True
    )


    severity_table = pd.DataFrame({

        "Disease Progression Stage": [

            "Early Phase / Baseline",

            "Mild Cognitive Impairment",

            "Symptomatic Alzheimer's",

            "Severe Progression"
        ],

        "Progression Score Range": [

            "0 – 24.99",

            "25 – 49.99",

            "50 – 74.99",

            "75 – 100"
        ]
    })


    st.table(
        severity_table
    )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown(
        '<div class="section">Patient Report</div>',
        unsafe_allow_html=True
    )


    report_text = f"""
ALZHEIMER'S DISEASE PROGRESSION PREDICTION REPORT
=================================================

PATIENT INFORMATION
-------------------

Age:
{report["age"]} years

Body Mass Index:
{report["bmi"]:.1f}

Smoking Status:
{report["smoking"]}

Alcohol Consumption:
{report["alcohol"]}

Mini-Mental State Examination Score:
{report["mmse"]:.1f} / 30

Functional Skill Score:
{report["functional_skill"]:.1f} / 10


PREDICTION RESULT
-----------------

Predicted Alzheimer's Disease Progression Severity:
{report["predicted_score"]:.2f} / 100

Predicted Disease Progression Stage:
{report["predicted_severity"]}


K-MEANS CLUSTERING RESULT
-------------------------

Patient Risk Cluster:
Cluster {report["cluster_number"]}

Cluster Severity Label:
{report["cluster_label"]}


MODEL PERFORMANCE
-----------------

Coefficient of Determination (R²):
98.22%

Root Mean Squared Error (RMSE):
2.14

Mean Absolute Error (MAE):
1.70


DISEASE PROGRESSION SEVERITY SCALE
-----------------------------------

Early Phase / Baseline:
0 – 24.99

Mild Cognitive Impairment:
25 – 49.99

Symptomatic Alzheimer's:
50 – 74.99

Severe Progression:
75 – 100


DISCLAIMER
----------

This application is a machine-learning demonstration and
research/educational project.

It is not a medical diagnostic system and should not be used
to make clinical decisions.

Predictions should not replace evaluation by a qualified
healthcare professional.
"""


    st.markdown(
        """
        <div class="report-box">

        <b>Patient report generated successfully.</b>

        <br><br>

        The report contains the patient's clinical details,
        prediction result, K-Means clustering result,
        model performance and disease progression scale.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.download_button(

        "📄 Download Patient Report",

        data=report_text,

        file_name="alzheimer_patient_report.txt",

        mime="text/plain",

        use_container_width=True
    )


