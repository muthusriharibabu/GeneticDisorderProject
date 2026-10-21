
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("genetic_disorder_model.pkl")


def get_metadata():
    prep = model.named_steps["preprocessing"]

    fields = {}

    for name, transformer, columns in prep.transformers_:
        if name == "num":
            imputer = transformer.named_steps["imputer"]

            for col, value in zip(columns, imputer.statistics_):
                fields[col] = {
                    "type": "number",
                    "default": float(value)
                }

        elif name == "cat":
            imputer = transformer.named_steps["imputer"]
            encoder = transformer.named_steps["encoder"]

            for col, value, categories in zip(
                columns,
                imputer.statistics_,
                encoder.categories_
            ):
                fields[col] = {
                    "type": "category",
                    "default": str(value),
                    "options": [str(x) for x in categories]
                }

    return fields


@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify(get_metadata())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient = request.get_json()

        if not patient:
            return jsonify({"error": "Patient details are required"}), 400

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

        return jsonify({
            "predicted_subclass": str(prediction),
            "kernel": model.named_steps["svm"].kernel
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
