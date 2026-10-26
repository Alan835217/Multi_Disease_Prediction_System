from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

# Dataset file, model file, target column, and display metadata.
DISEASES = {
    "Diabetes": {
        "dataset": "diabetes.csv",
        "model": "diabetes_model.pkl",
        "target": "Outcome",
        "description": "Estimate a model prediction from common clinical and demographic measurements.",
        "positive": "Model predicted diabetes",
        "negative": "Model did not predict diabetes",
        "positive_values": {1, "1", "yes", "positive", "diabetes"},
    },
    "Heart Disease": {
        "dataset": "heart.csv",
        "model": "heart_disease_model.pkl",
        "target": "target",
        "description": "Estimate a model prediction from cardiovascular measurements and coded clinical attributes.",
        "positive": "Model predicted heart disease",
        "negative": "Model did not predict heart disease",
        "positive_values": {1, "1", "yes", "positive", "disease"},
    },
    "Chronic Kidney Disease": {
        "dataset": "kidney_disease.csv",
        "model": "kidney_disease_model.pkl",
        "target": "classification",
        "description": "Estimate a model prediction from kidney-related measurements and clinical indicators.",
        "positive": "Model predicted chronic kidney disease",
        "negative": "Model did not predict chronic kidney disease",
        "positive_values": {1, "1", "ckd", "yes", "positive", "disease"},
    },
    "Breast Cancer": {
        "dataset": "breast-cancer.csv",
        "model": "breast_cancer_model.pkl",
        "target": "diagnosis",
        "description": "Use the dataset's tumour measurement features to generate a model classification.",
        "positive": "Model predicted malignant class",
        "negative": "Model predicted benign class",
        "positive_values": {1, "1", "m", "malignant", "positive"},
    },
    "Thyroid Cancer Recurrence": {
        "dataset": "thyroid.csv",
        "model": "thyroid_recurrence_model.pkl",
        "target": "Recurred",
        "description": "Estimate whether recurrence is predicted for a patient represented by thyroid cancer dataset features.",
        "positive": "Model predicted thyroid cancer recurrence",
        "negative": "Model did not predict thyroid cancer recurrence",
        "positive_values": {"yes", "recurred", "1", 1},
    },
}

# Some source CSVs use an identifier column that must not be passed as a feature.
ID_COLUMNS = {"id", "unnamed: 0"}

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #18232F;
        color: #E5EDF5;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4, p, li, label {
        color: #E5EDF5;
    }
    [data-testid="stSidebar"] {
        background-color: #101923;
        border-right: 1px solid #2B3948;
    }
    [data-testid="stSidebar"] > div {
        background-color: #101923;
    }
    [data-testid="stSidebar"] * {
        color: #DCE6EF;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        border-radius: 8px;
        padding: 6px 9px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #263746;
    }
    .hero {
        background: linear-gradient(125deg, #101923, #174D54);
        border: 1px solid #31545D;
        padding: 36px;
        border-radius: 18px;
        margin-bottom: 26px;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }
    .hero h1 {
        color: #FFFFFF !important;
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 10px;
    }
    .hero p {
        color: #CBD9E5 !important;
        font-size: 15px;
        line-height: 1.7;
    }
    .section-title {
        color: #F0F5FA;
        font-size: 23px;
        font-weight: 700;
        margin: 25px 0 15px;
    }
    .info-card {
        background: #222F3D;
        border: 1px solid #354657;
        border-radius: 14px;
        padding: 22px;
        min-height: 145px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,.12);
        transition: border-color .2s ease;
    }
    .info-card:hover {
        border-color: #4EAAA5;
    }
    .info-card h3 {
        color: #76D2C5 !important;
        font-size: 17px;
        margin-bottom: 10px;
    }
    .info-card p {
        color: #B8C7D6 !important;
        font-size: 13px;
        line-height: 1.7;
    }
    [data-testid="stMetric"] {
        background: #222F3D;
        border: 1px solid #354657;
        padding: 18px;
        border-radius: 13px;
    }
    [data-testid="stMetricLabel"] {
        color: #B8C7D6 !important;
    }
    [data-testid="stMetricValue"] {
        color: #76D2C5 !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        background-color: #176B70;
        color: #FFFFFF;
        border: 1px solid #328B88;
        border-radius: 9px;
        min-height: 42px;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #21817D;
        color: #FFFFFF;
        border-color: #65C6B9;
    }
    [data-testid="stExpander"] {
        background: #222F3D;
        border: 1px solid #354657;
        border-radius: 12px;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #E5EDF5 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #354657;
        border-radius: 10px;
    }
    [data-baseweb="select"] > div {
        background-color: #222F3D;
        border-color: #45576A;
        color: #E5EDF5;
    }
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        color: #E5EDF5;
    }
    .footer {
        color: #93A5B7;
        text-align: center;
        font-size: 12px;
        padding: 20px 0;
    }
    .notice {
        background: #2B3037;
        border-left: 4px solid #E0B15A;
        padding: 14px 16px;
        border-radius: 8px;
        color: #E5EDF5;
        margin: 12px 0 20px;
    }
    @media (max-width: 768px) {
        .hero { padding: 24px; }
        .hero h1 { font-size: 29px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model and dataset loading
# ---------------------------------------------------------
def model_file_signature():
    """Build a cache key that changes whenever a saved model file changes."""
    signature = []
    for disease, info in DISEASES.items():
        path = MODEL_DIR / info["model"]
        if path.exists():
            stat = path.stat()
            signature.append((disease, str(path), stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((disease, str(path), None, None))
    return tuple(signature)


@st.cache_resource
def load_models(model_signature):
    """Load models once per model-file version; changed files trigger a reload."""
    loaded = {}
    errors = {}
    for disease, info in DISEASES.items():
        path = MODEL_DIR / info["model"]
        if not path.exists():
            errors[disease] = f"Model file not found: {path}"
            continue
        try:
            loaded[disease] = joblib.load(path)
        except Exception as exc:
            errors[disease] = f"Could not load model: {exc}"
    return loaded, errors


@st.cache_data
def load_dataset(disease):
    info = DISEASES[disease]
    path = DATA_DIR / info["dataset"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    # Standardise whitespace around column names without changing case.
    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalise_label(value):
    """Create a comparable label for common target encodings."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value.strip().lower()
    return value


def get_target_value(disease, raw_value):
    """Convert common target encodings to the app's positive/negative meaning."""
    value = normalise_label(raw_value)
    positive_values = DISEASES[disease]["positive_values"]

    # Handle numeric target labels.
    if value in positive_values:
        return True
    if disease == "Chronic Kidney Disease" and value in {"notckd", "not ckd", "0", "no"}:
        return False
    if disease == "Breast Cancer" and value in {"b", "benign", "0"}:
        return False
    if value in {0, "0", "no", "negative", "benign", "notckd", "not ckd", "no disease"}:
        return False
    return None


def feature_columns(disease, df):
    """Return feature columns while excluding targets and identifier columns."""
    info = DISEASES[disease]
    target_names = info.get("target_aliases", [info["target"]])
    target_names = {str(name).strip().lower() for name in target_names}

    excluded = set(ID_COLUMNS) | target_names
    # Exclude common target aliases as an extra safeguard against target leakage.
    if disease == "Diabetes":
        excluded.update({"outcome", "target"})
    elif disease == "Heart Disease":
        excluded.update({"target", "num"})
    elif disease == "Chronic Kidney Disease":
        excluded.update({"classification", "class", "target"})
    elif disease == "Breast Cancer":
        excluded.update({"diagnosis", "target"})
    elif disease == "Thyroid Cancer Recurrence":
        excluded.update({"recurred", "target", "label"})

    return [col for col in df.columns if str(col).strip().lower() not in excluded]


def is_categorical_series(series):
    """Treat text columns as categorical unless most non-null values are numeric."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return True

    # Kidney datasets often store numeric measurements (for example PCV/WC/RC)
    # as strings. Detect these so the form uses number inputs instead of select boxes.
    converted = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
    numeric_fraction = float(converted.notna().mean())
    return numeric_fraction < 0.80


FEATURE_LABELS = {
    "tot_bilirubin": "Total Bilirubin",
    "direct_bilirubin": "Direct Bilirubin",
    "alkphos": "Alkaline Phosphatase",
    "sgpt": "SGPT (ALT)",
    "sgot": "SGOT (AST)",
    "tot_proteins": "Total Proteins",
    "ag_ratio": "Albumin/Globulin Ratio",
    "pregnancies": "Pregnancies",
    "bmi": "Body Mass Index (BMI)",
    "bp": "Blood Pressure",
    "thalach": "Maximum Heart Rate",
    "oldpeak": "ST Depression (Oldpeak)",
}


def make_prediction_form(disease, model, df):
    features = feature_columns(disease, df)
    if not features:
        st.error("Could not determine the input features from this dataset.")
        return

    st.markdown(
        "<div class='notice'>Enter values for the available fields. Defaults are based on this dataset for demonstration only. Use appropriate, verified measurements when testing the model.</div>",
        unsafe_allow_html=True,
    )

    with st.form(key=f"form_{disease}"):
        st.subheader("Patient / sample inputs")
        values = {}
        columns = st.columns(2)

        for index, feature in enumerate(features):
            series = df[feature]
            column = columns[index % 2]
            with column:
                if is_categorical_series(series):
                    choices = [
                        value for value in series.dropna().unique().tolist()
                        if str(value).strip() != ""
                    ]
                    choices = sorted(choices, key=lambda x: str(x))
                    if not choices:
                        choices = ["unknown"]
                    mode = series.mode(dropna=True)
                    default_value = mode.iloc[0] if not mode.empty else choices[0]
                    default_index = choices.index(default_value) if default_value in choices else 0
                    values[feature] = st.selectbox(
                        FEATURE_LABELS.get(
                            feature, feature.replace("_", " ").replace(".", " ").title()
                        ),
                        options=choices,
                        index=default_index,
                        key=f"{disease}_{feature}",
                        help=f"Categorical values observed in {DISEASES[disease]['dataset']}.",
                    )
                else:
                    numeric = pd.to_numeric(series, errors="coerce").replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()
                    default_value = float(numeric.median()) if not numeric.empty else 0.0
                    min_value = float(numeric.min()) if not numeric.empty else None
                    max_value = float(numeric.max()) if not numeric.empty else None
                    # Keep a small margin around the dataset range so users can enter values
                    # slightly outside the observed minimum/maximum.
                    if min_value is not None and max_value is not None and min_value == max_value:
                        min_value -= 1.0
                        max_value += 1.0
                    step = 1.0 if pd.api.types.is_integer_dtype(series) else 0.1
                    values[feature] = st.number_input(
                        FEATURE_LABELS.get(
                            feature, feature.replace("_", " ").replace(".", " ").title()
                        ),
                        value=default_value,
                        min_value=min_value,
                        max_value=max_value,
                        step=step,
                        format="%.4f" if step < 1 else "%.2f",
                        key=f"{disease}_{feature}",
                        help=f"Typical range in dataset: {min_value:.3f} to {max_value:.3f}"
                        if min_value is not None and max_value is not None
                        else None,
                    )

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([values], columns=features)

            # Some models were trained using the exact source feature names. Preserve them.
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                missing = [col for col in expected if col not in input_df.columns]
                if missing:
                    st.error(
                        "The model expects input columns that were not found in the dataset: "
                        + ", ".join(missing)
                    )
                    return
                input_df = input_df[expected]

            prediction = model.predict(input_df)[0]
            result = get_target_value(disease, prediction)

            st.divider()
            st.subheader("Prediction result")

            if result is True:
                st.warning(f"**{DISEASES[disease]['positive']}**")
            elif result is False:
                st.success(f"**{DISEASES[disease]['negative']}**")
            else:
                st.info(f"The model returned class: {normalise_label(prediction)}")

            # Display a class score only when the estimator exposes probabilities
            # and its class labels can be mapped reliably to the positive class.
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    classes = getattr(model, "classes_", None)
                    positive_probability = None
                    if classes is not None and len(classes) == len(probabilities):
                        for class_label, probability in zip(classes, probabilities):
                            if get_target_value(disease, class_label) is True:
                                positive_probability = float(probability)
                                break
                    if positive_probability is not None:
                        st.metric(
                            "Model score for positive class",
                            f"{positive_probability:.1%}",
                        )
                        st.caption(
                            "This score is model-estimated and may not be calibrated; "
                            "it is not a person's confirmed disease probability."
                        )
                except Exception:
                    # A probability display is optional; the class prediction remains valid.
                    pass

            st.caption(
                "Academic demonstration only — this result is not a medical diagnosis. "
                "Consult a qualified healthcare professional for medical concerns."
            )

            with st.expander("View submitted input values"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

            with st.expander("Technical details"):
                st.write("Predicted class:", repr(prediction))
                st.write("Feature order used by the model:", input_df.columns.tolist())
        except Exception as exc:
            st.error(
                "Prediction failed. Confirm that this model was trained with the feature "
                f"columns and preprocessing expected by this app. Details: {exc}"
            )


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
def navigate_to(disease_name):
    st.session_state["page_selector"] = disease_name


models, model_errors = load_models(model_file_signature())

with st.sidebar:
    st.markdown("## 🩺 PRKR HealthAI")
    st.caption("Multi-Disease Prediction System")
    st.divider()
    navigation_options = [
        "Home",
        "Diabetes",
        "Heart Disease",
        "Chronic Kidney Disease",
        "Breast Cancer",
        "Thyroid Cancer Recurrence",
        "About Project",
    ]
    if "page_selector" not in st.session_state:
        st.session_state["page_selector"] = "Home"
    page = st.radio(
        "NAVIGATION",
        navigation_options,
        key="page_selector",
    )
    st.divider()
    st.caption("MCA Machine Learning Theory/Lab Project")
    st.caption("Developed by PRKR teammates")
    

# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Multi-Disease Prediction System</h1>
            <p>
                A machine-learning academic prototype that demonstrates how trained
                classification models can process structured health-related data and
                return model predictions for four disease categories and thyroid cancer recurrence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction Modules", "5")
    m2.metric("ML Approach", "Classification")
    m3.metric("Interface", "Streamlit")
    m4.metric("Model Storage", "Joblib")

    st.markdown("<div class='section-title'>Choose a prediction module</div>", unsafe_allow_html=True)
    cards = [
        ("🩸 Diabetes", "Diabetes", "Uses demographic and clinical measurements from the diabetes dataset."),
        ("❤️ Heart Disease", "Heart Disease", "Uses cardiovascular measurements and coded clinical attributes."),
        ("🧪 Chronic Kidney Disease", "Chronic Kidney Disease", "Uses laboratory values and clinical indicators."),
        ("🎗️ Breast Cancer", "Breast Cancer", "Uses tumour measurement features for benign/malignant classification."),
        ("🦋 Thyroid Cancer Recurrence", "Thyroid Cancer Recurrence", "Predicts recurrence among patients represented in the thyroid cancer dataset."),
    ]
    card_cols = st.columns(3)
    for index, (heading, disease, description) in enumerate(cards):
        with card_cols[index % 3]:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{heading}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                f"Open {disease}",
                key=f"open_{disease}",
                use_container_width=True,
                on_click=navigate_to,
                args=(disease,),
            )

    st.markdown("<div class='section-title'>Technology stack</div>", unsafe_allow_html=True)
    tech_cols = st.columns(4)
    tech = [
        ("Python", "Application and model integration"),
        ("Pandas / NumPy", "Tabular data processing"),
        ("Scikit-learn", "Machine-learning pipelines"),
        ("Streamlit", "Interactive web interface"),
    ]
    for col, (name, detail) in zip(tech_cols, tech):
        with col:
            st.markdown(
                f"<div class='info-card'><h3>{name}</h3><p>{detail}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Dataset overview</div>", unsafe_allow_html=True)
    dataset_rows = []
    for disease, info in DISEASES.items():
        path = DATA_DIR / info["dataset"]
        if path.exists():
            try:
                df = load_dataset(disease)
                dataset_rows.append(
                    {
                        "Disease module": disease,
                        "Dataset file": info["dataset"],
                        "Rows": int(df.shape[0]),
                        "Columns": int(df.shape[1]),
                        "Model loaded": "Yes" if disease in models else "No",
                    }
                )
            except Exception:
                dataset_rows.append(
                    {
                        "Disease module": disease,
                        "Dataset file": info["dataset"],
                        "Rows": "—",
                        "Columns": "—",
                        "Model loaded": "Yes" if disease in models else "No",
                    }
                )
        else:
            dataset_rows.append(
                {
                    "Disease module": disease,
                    "Dataset file": info["dataset"],
                    "Rows": "Not found",
                    "Columns": "Not found",
                    "Model loaded": "Yes" if disease in models else "No",
                }
            )
    st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True, hide_index=True)

    with st.expander("Methodology"):
        st.markdown(
            """
            1. **Data collection:** load a separate public dataset for each disease.
            2. **Preprocessing:** handle missing values, clean target labels, and prepare features.
            3. **Train/test split:** evaluate models on data held out from training.
            4. **Model selection:** compare suitable classification algorithms using metrics such as
               accuracy, precision, recall, F1-score, confusion matrix, and cross-validation.
            5. **Model persistence:** save the fitted model/pipeline with Joblib.
            6. **Inference:** collect input values in the interface and pass them to the saved model.
            """
        )

    st.markdown(
        """
        <div class="notice">
            <strong>Important:</strong> This project is for learning and demonstration.
            Public datasets may be limited, imbalanced, or not representative of real patients.
            Model outputs must not be interpreted as diagnoses or a substitute for qualified
            medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Disease prediction pages
# ---------------------------------------------------------
elif page in DISEASES:
    info = DISEASES[page]
    st.title(page)
    st.write(info["description"])
    st.caption(f"Dataset: `{info['dataset']}`  ·  Model: `{info['model']}`")

    if page in model_errors:
        st.error(model_errors[page])
        st.info("Check that the model file exists in the project's `models` folder.")
    elif page not in models:
        st.error("The trained model could not be loaded.")
    else:
        try:
            dataset = load_dataset(page)
            make_prediction_form(page, models[page], dataset)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Place the required CSV file in the project's `datasets` folder.")
        except Exception as exc:
            st.error(f"Could not prepare this page: {exc}")

# ---------------------------------------------------------
# About page
# ---------------------------------------------------------
elif page == "About Project":
    st.title("About the Project")
    st.write(
        "The Multi-Disease Prediction System is an MCA-level machine-learning project "
        "that demonstrates model loading, structured input collection, and classification "
        "for four disease datasets plus a thyroid cancer recurrence dataset."
    )

    st.markdown("---")
    st.subheader("Project modules")
    st.dataframe(
        pd.DataFrame(
            [
                {"Module": disease, "Dataset": info["dataset"], "Saved model": info["model"]}
                for disease, info in DISEASES.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("---")
    st.subheader("Expected folder structure")
    st.code(
        """Multi_Disease_Prediction/
├── app.py
├── requirements.txt
├── datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── kidney_disease.csv
│   ├── breast-cancer.csv
│   └── thyroid.csv
└── models/
    ├── diabetes_model.pkl
    ├── heart_disease_model.pkl
    ├── kidney_disease_model.pkl
    ├── breast_cancer_model.pkl
    └── thyroid_recurrence_model.pkl
""",
        language="text",
    )

    st.markdown("---")
    st.subheader("Libraries")
    st.markdown(
        """
        - **Streamlit:** web interface and navigation
        - **Pandas:** CSV loading and tabular data handling
        - **NumPy:** numerical support
        - **Scikit-learn:** model and preprocessing compatibility
        - **Joblib:** loading saved model objects
        """
    )
    st.markdown("---")
    st.subheader("Model availability")
    for disease in DISEASES:
        if disease in models:
            st.success(f"{disease}: model loaded")
        else:
            st.error(f"{disease}: {model_errors.get(disease, 'model unavailable')}")

    st.markdown(
        """
        <div class="notice">
            <strong>Disclaimer:</strong> This project is still in development and not perfect so use this wisely 😉.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='footer'>PRKR Team · Multi-Disease Prediction System · MCA Machine Learning Project</div>",
    unsafe_allow_html=True,
)

st.markdown("""
## 🤝 Development Team

<div style="
padding:20px;
border-radius:12px;
background-color:rgba(70, 130, 180, 0.15);
border:1px solid rgba(255,255,255,0.15);
">

<h3>👥 PRKR Team</h3>

<p>
This project was collaboratively developed by the <b>PRKR Team</b> as part of the
Master of Computer Applications (MCA) Machine Learning Theory and Lab Project.
</p>

<b>Team Members</b>

<ul>
<li>Parmeshwar Kumar Mahto (MCA/40050/25)</li>
<li>Ritul Raj Pandey (MCA/40077/25)</li>
<li>Rakesh Kumar (MCA/40076/25)</li>
<li>Md.Kaif (MCA/40080/25)</li>
</ul>

</div>
""", unsafe_allow_html=True)
