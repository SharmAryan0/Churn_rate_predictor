from pathlib import Path

import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "Churn_Modelling.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "ann_model.keras"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0D17;
        color: #F5F7FF;
    }

    [data-testid="stSidebar"] {
        background-color: #171D38;
    }

    div[data-testid="stMetric"] {
        background-color: #171D38;
        border: 1px solid #343B67;
        padding: 18px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path, modified_time):
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_preprocessor(path, modified_time):
    return joblib.load(path)


st.sidebar.title("🧠 Churn Intelligence")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Customer Analytics",
        "Predict Churn",
        "Model Performance"
    ]
)

st.sidebar.caption(
    "ANN-powered customer retention analytics"
)

st.title("Customer Churn Intelligence")

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place Churn_Modelling.csv "
        "inside the data folder."
    )
    st.stop()

df = load_data(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime
)

if page == "Overview":
    total_customers = len(df)
    churned_customers = int((df["Exited"] == 1).sum())

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers
        else 0
    )

    retention_rate = 100 - churn_rate

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%"
    )

    st.subheader("Customer Churn Distribution")

    churn_counts = (
        df["Exited"]
        .value_counts()
        .rename(index={
            0: "Retained",
            1: "Churned"
        })
    )

    st.bar_chart(churn_counts)

elif page == "Customer Analytics":
    st.subheader("Customer Analytics")

    st.write("### Age and Churn")

    st.bar_chart(
        df.groupby("Age")["Exited"].mean()
    )

    st.write("### Geography and Churn")

    st.bar_chart(
        df.groupby("Geography")["Exited"].mean()
    )

    st.write("### Customer Data")

    st.dataframe(
        df.drop(
            columns=["Surname"],
            errors="ignore"
        ),
        use_container_width=True
    )

elif page == "Predict Churn":
    st.subheader("Predict Customer Churn")

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.warning(
            "Model artifacts are missing. Run the notebook "
            "training and saving steps first."
        )
        st.stop()

    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )

    preprocessor = load_preprocessor(
        str(PREPROCESSOR_PATH),
        PREPROCESSOR_PATH.stat().st_mtime
    )

    with st.form("customer_prediction_form"):
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        geography = st.selectbox(
            "Geography",
            sorted(df["Geography"].dropna().unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].dropna().unique())
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=10,
            value=5
        )

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=75000.0
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2
        )

        has_credit_card = st.selectbox(
            "Has Credit Card",
            [1, 0]
        )

        is_active_member = st.selectbox(
            "Is Active Member",
            [1, 0]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=80000.0
        )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "CreditScore": credit_score,
                    "Geography": geography,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": num_products,
                    "HasCrCard": has_credit_card,
                    "IsActiveMember": is_active_member,
                    "EstimatedSalary": estimated_salary
                }
            ]
        )

        try:
            customer = customer[
                preprocessor.feature_names_in_
            ]

            processed_customer = preprocessor.transform(
                customer
            )

            probability = float(
                model.predict(
                    processed_customer,
                    verbose=0
                )[0, 0]
            )

            retention_probability = 1 - probability

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )

            st.metric(
                "Retention Probability",
                f"{retention_probability:.2%}"
            )

            if probability < 0.30:
                risk = "Low Risk"
            elif probability < 0.60:
                risk = "Medium Risk"
            else:
                risk = "High Risk"

            st.subheader(f"Risk Category: {risk}")

            st.progress(probability)

            if probability >= 0.5:
                st.warning(
                    "The model predicts that this customer "
                    "is likely to churn at the selected threshold."
                )
            else:
                st.success(
                    "The model predicts that this customer "
                    "is likely to stay at the selected threshold."
                )

        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}"
            )

elif page == "Model Performance":
    st.subheader("Model Performance")

    metrics_path = ARTIFACT_DIR / "metrics.json"
    history_path = ARTIFACT_DIR / "training_history.json"

    if not metrics_path.exists():
        st.warning(
            "Evaluation metrics are missing. "
            "Run the notebook evaluation and saving steps."
        )
        st.stop()

    import json

    with open(metrics_path) as file:
        metrics = json.load(file)

    columns = st.columns(len(metrics))

    for column, (name, value) in zip(
        columns,
        metrics.items()
    ):
        column.metric(
            name.replace("_", " ").title(),
            f"{value:.3f}"
        )

    if history_path.exists():
        with open(history_path) as file:
            history = json.load(file)

        st.subheader("Training and Validation Loss")

        loss_data = pd.DataFrame(
            {
                "Training Loss": history.get("loss", []),
                "Validation Loss": history.get("val_loss", [])
            }
        )

        st.line_chart(loss_data)

st.caption(
    "Predictions are estimates and should support, "
    "not replace, informed business decisions."
)