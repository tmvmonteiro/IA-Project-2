import pickle
import re
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = APP_DIR / "model_artifacts.pkl"

app = Flask(__name__, static_folder="static", static_url_path="")


def load_artifact():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run `python web_app/train_model.py` first."
        )

    with ARTIFACT_PATH.open("rb") as f:
        return pickle.load(f)


artifact = load_artifact()


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def encode_payload(payload):
    row = {column: 0.0 for column in artifact["featureColumns"]}
    active_categoricals = {}

    for feature in artifact["numericalFeatures"]:
        value = payload.get(feature)
        if value is None or value == "":
            value = artifact["options"]["numerical"][feature]["median"]
        if feature in row:
            row[feature] = float(value)

    for feature in artifact["categoricalFeatures"]:
        value = payload.get(feature)
        if value is None or value == "":
            values = artifact["options"]["categorical"][feature]
            value = values[0] if values else "Unknown"
        dummy_column = re.sub(r"[^A-Za-z0-9_]+", "_", f"{feature}_{value}")
        if dummy_column in row:
            row[dummy_column] = 1.0
        active_categoricals[feature] = {
            "value": str(value),
            "encodedColumn": dummy_column if dummy_column in row else None,
        }

    return pd.DataFrame([row], columns=artifact["featureColumns"]), active_categoricals


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/options")
def options():
    return jsonify(
        {
            "trainingRows": artifact["trainingRows"],
            "numericalFeatures": artifact["numericalFeatures"],
            "categoricalFeatures": artifact["categoricalFeatures"],
            "options": artifact["options"],
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(force=True)
    encoded, active_categoricals = encode_payload(payload)
    prediction = float(artifact["model"].predict(encoded)[0])

    return jsonify(
        {
            "prediction": round(prediction, 2),
            "formattedPrediction": f"EUR {prediction:,.2f}",
            "received": {
                "Municipality": payload.get("Municipality"),
                "Parish": payload.get("Parish"),
            },
            "activeCategoricals": active_categoricals,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
