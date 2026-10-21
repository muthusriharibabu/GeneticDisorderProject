

import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Genetic Disorder Screening",
    page_icon="🧬"
)

st.title("Genetic Disorder Subclass Screening System")
st.write("Enter the patient details below.")

API_URL = "http://127.0.0.1:5000"

try:
    response = requests.get(API_URL + "/metadata", timeout=5)
    response.raise_for_status()
    fields = response.json()

except requests.RequestException:
    st.error("Flask API is not running. Start model_api.py first.")
    st.stop()

with st.form("patient_form"):
    patient = {}

    for col, info in fields.items():
        if info["type"] == "number":
            patient[col] = st.number_input(
                col,
                value=float(info["default"])
            )
        else:
            options = info["options"]

            if options:
                default = info["default"]
                index = options.index(default) if default in options else 0

                patient[col] = st.selectbox(
                    col,
                    options,
                    index=index
                )
            else:
                patient[col] = st.text_input(
                    col,
                    value=info["default"]
                )

    submitted = st.form_submit_button("Predict Subclass")

if submitted:
    try:
        response = requests.post(
            API_URL + "/predict",
            json=patient,
            timeout=30
        )

        result = response.json()

        if response.ok:
            st.success(
                "Predicted Subclass: "
                + result["predicted_subclass"]
            )
            st.write("Selected Kernel:", result["kernel"])
        else:
            st.error(result.get("error", "Prediction failed"))

    except requests.RequestException:
        st.error("Could not connect to Flask API.")

st.header("SVM Kernel Performance Comparison")

try:
    comparison = pd.read_csv("kernel_comparison.csv")

    comparison["Accuracy (%)"] = (
        comparison["Accuracy"] * 100
    ).round(2)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        comparison.set_index("Kernel")["Accuracy (%)"]
    )

    best = comparison.loc[
        comparison["Accuracy"].idxmax()
    ]

    st.success(
        f"Highest recorded accuracy: {best['Kernel']} "
        f"({best['Accuracy (%)']:.2f}%)"
    )

except Exception as e:
    st.warning(f"Comparison could not be loaded: {e}")

st.header("Confusion Matrix")

try:
    st.image(
        "confusion_matrix.png",
        caption="SVM Classification Confusion Matrix",
        use_container_width=True
    )
except Exception:
    st.warning("Confusion matrix image not found.")

st.caption(
    "Academic screening prototype only. "
    "This prediction is not a medical diagnosis."
)
