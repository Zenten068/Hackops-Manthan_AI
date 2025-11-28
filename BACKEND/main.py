from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path("fraud_model.pkl")

MODEL_AVAILABLE = False
PIPELINE = None
BEST_THRESHOLD = 0.5
NUM_COLS = []
CAT_COLS = []

# ---------- Try loading trained model ----------
if MODEL_PATH.exists():
    try:
        artifact = joblib.load(MODEL_PATH)
        PIPELINE = artifact["pipeline"]
        NUM_COLS = artifact.get("numeric_cols", [])
        CAT_COLS = artifact.get("categorical_cols", [])
        BEST_THRESHOLD = artifact.get("best_threshold", 0.5)
        MODEL_AVAILABLE = True
        print(f"[INFO] Loaded model from {MODEL_PATH}, threshold={BEST_THRESHOLD}")
    except Exception as e:
        print("[ERROR] Failed to load model:", e)
else:
    print(f"[WARN] Model file {MODEL_PATH} not found. Using heuristic only.")


# ---------- Heuristic fallback ----------
def calculate_dummy_risk(payload: dict) -> float:
    def to_float(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    amount = to_float(payload.get("tx_amount", 0))
    hour = to_float(payload.get("hour_of_day", 0))
    account_age = to_float(payload.get("account_age_days", 0))
    velocity = to_float(payload.get("tx_velocity", 0))
    failed = to_float(payload.get("num_failed_tx", 0))
    credit = to_float(payload.get("credit_score", 0))
    vpn = to_float(payload.get("vpn_detected", 0))

    risk = 0.0

    # High-value + night + new account
    if amount > 5000 and 0 <= hour <= 6 and account_age < 30:
        risk += 0.45

    # High velocity + VPN
    if velocity > 10 and vpn == 1:
        risk += 0.25

    # Many failed tx
    if failed >= 3:
        risk += 0.15

    # Low credit + high amount
    if 0 < credit < 550 and amount > 3000:
        risk += 0.25

    # Base scaling by amount
    risk += min(amount / 20000.0, 0.15)

    risk = max(0.0, min(1.0, risk))
    return float(risk)


# ---------- ML prediction ----------
def predict_with_model(payload: dict) -> float:
    if not MODEL_AVAILABLE or PIPELINE is None:
        return calculate_dummy_risk(payload)

    row = {}

    # numeric features used in training
    for col in NUM_COLS:
        row[col] = payload.get(col)

    # categorical features
    for col in CAT_COLS:
        row[col] = payload.get(col)

    # original fields used for engineered features
    for col in ["tx_amount", "hour_of_day", "day_of_week", "month"]:
        if col not in row:
            row[col] = payload.get(col)

    df = pd.DataFrame([row])

    proba = PIPELINE.predict_proba(df)[:, 1][0]
    return float(proba)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}

    if MODEL_AVAILABLE:
        risk_score = predict_with_model(data)
        threshold = BEST_THRESHOLD
        source = "ml_model"
    else:
        risk_score = calculate_dummy_risk(data)
        threshold = 0.5
        source = "heuristic"

    label = int(risk_score >= threshold)

    return jsonify(
        {
            "fraud_probability": risk_score,
            "label": label,
            "threshold": threshold,
            "source": source,
        }
    )


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL_AVAILABLE,
            "threshold": BEST_THRESHOLD if MODEL_AVAILABLE else 0.5,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
