

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "genetic_disorder_model.pkl"
)

CSV_PATH = os.path.join(
    BASE_DIR, "kernel_comparison.csv"
)

# Load trained SVM model
model = joblib.load(MODEL_PATH)


# Find preprocessing component
def find_preprocessor(estimator):

    if isinstance(estimator, ColumnTransformer):
        return estimator

    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            result = find_preprocessor(step)

            if result is not None:
                return result

    return None


# Find SVM classifier
def find_svm(estimator):

    if isinstance(estimator, SVC):
        return estimator

    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            result = find_svm(step)

            if result is not None:
                return result

    return None


# Get input fields from model
def get_metadata():

    preprocessing = find_preprocessor(model)

    if preprocessing is None:
        raise ValueError("Preprocessing component not found")

    fields = {}

    for name, transformer, columns in preprocessing.transformers_:

        if transformer == "drop" or transformer == "passthrough":
            continue

        steps = (
            list(transformer.named_steps.values())
            if isinstance(transformer, Pipeline)
            else [transformer]
        )

        imputer = next(
            (step for step in steps
             if isinstance(step, SimpleImputer)),
            None
        )

        encoder = next(
            (step for step in steps
             if isinstance(step, OneHotEncoder)),
            None
        )

        if imputer is None:
            continue

        for i, col in enumerate(columns):

            default = imputer.statistics_[i]

            if encoder is not None:

                options = [
                    str(value)
                    for value in encoder.categories_[i]
                    if not pd.isna(value)
                ]

                fields[str(col)] = {
                    "type": "category",
                    "default": str(default),
                    "options": options
                }

            else:

                fields[str(col)] = {
                    "type": "number",
                    "default": float(default)
                }

    return fields


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Patient input fields
@app.route("/metadata")
def metadata():

    try:
        return jsonify(get_metadata())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Predict disorder subclass
@app.route("/predict", methods=["POST"])
def predict():

    try:

        patient = request.get_json(silent=True)

        if not isinstance(patient, dict):
            return jsonify({
                "error": "Patient details are required"
            }), 400

        fields = get_metadata()
        row = {}

        for col, info in fields.items():

            value = patient.get(col, info["default"])

            if info["type"] == "number":
                row[col] = float(value)
            else:
                row[col] = str(value)

        input_data = pd.DataFrame([row])

        prediction = model.predict(input_data)[0]

        svm = find_svm(model)

        kernel = svm.kernel if svm else "Unknown"

        return jsonify({
            "predicted_subclass": str(prediction),
            "kernel": str(kernel)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# Compare SVM kernels
@app.route("/comparison")
def comparison():

    try:

        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()

        kernel_col = next(
            (c for c in df.columns
             if c.lower() in ["kernel", "model", "svm kernel"]),
            None
        )

        accuracy_col = next(
            (c for c in df.columns
             if "accuracy" in c.lower()),
            None
        )

        if kernel_col is None or accuracy_col is None:
            return jsonify({
                "error": "CSV needs Kernel and Accuracy columns"
            }), 400

        df[accuracy_col] = pd.to_numeric(
            df[accuracy_col].astype(str).str.replace("%", ""),
            errors="coerce"
        )

        df = df.dropna(subset=[accuracy_col])

        if df.empty:
            return jsonify({
                "error": "No accuracy results found"
            }), 400

        if df[accuracy_col].max() <= 1:
            df[accuracy_col] *= 100

        results = []

        for _, row in df.iterrows():

            results.append({
                "kernel": str(row[kernel_col]),
                "accuracy": round(float(row[accuracy_col]), 2)
            })

        best = max(results, key=lambda x: x["accuracy"])

        return jsonify({
            "results": results,
            "best_kernel": best["kernel"],
            "best_accuracy": best["accuracy"]
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
