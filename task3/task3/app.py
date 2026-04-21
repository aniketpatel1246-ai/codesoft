"""
Customer Churn Prediction API
Flask Backend — serves ML model predictions and stats
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─── Load artifacts ───────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "Churn_Modelling.csv")

model    = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le_geo   = joblib.load(os.path.join(MODEL_DIR, "le_geo.pkl"))
le_gen   = joblib.load(os.path.join(MODEL_DIR, "le_gen.pkl"))

with open(os.path.join(MODEL_DIR, "summary.json")) as f:
    summary = json.load(f)

# CORS helper (no flask-cors dependency)
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.after_request
def after_request(response):
    return add_cors(response)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "best_model": summary["best_model"]})


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return model comparison stats and dataset overview."""
    return jsonify({
        "dataset_size":       summary["dataset_size"],
        "churn_rate":         summary["churn_rate"],
        "best_model":         summary["best_model"],
        "best_auc":           summary["best_auc"],
        "feature_importances": summary["feature_importances"],
        "model_results": {
            name: {
                "accuracy": v["accuracy"],
                "auc":      v["auc"],
                "confusion_matrix": v["confusion_matrix"],
            }
            for name, v in summary["model_results"].items()
        }
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Predict churn for a single customer.
    Body JSON fields:
        CreditScore, Geography, Gender, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(force=True)
    required = ["CreditScore","Geography","Gender","Age","Tenure",
                "Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        geo = le_geo.transform([data["Geography"]])[0]
        gen = le_gen.transform([data["Gender"]])[0]
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    row = np.array([[
        float(data["CreditScore"]),
        float(geo),
        float(gen),
        float(data["Age"]),
        float(data["Tenure"]),
        float(data["Balance"]),
        float(data["NumOfProducts"]),
        float(data["HasCrCard"]),
        float(data["IsActiveMember"]),
        float(data["EstimatedSalary"]),
    ]])

    prob  = float(model.predict_proba(row)[0][1])
    pred  = int(model.predict(row)[0])
    risk  = "High" if prob >= 0.6 else "Medium" if prob >= 0.35 else "Low"

    return jsonify({
        "churn_probability": round(prob * 100, 2),
        "prediction":        pred,
        "risk_level":        risk,
        "model_used":        summary["best_model"],
    })


@app.route("/api/batch_predict", methods=["POST", "OPTIONS"])
def batch_predict():
    """Accept JSON array of customers; return predictions for each."""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    customers = request.get_json(force=True)
    if not isinstance(customers, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    predictions = []
    for c in customers:
        try:
            geo = le_geo.transform([c["Geography"]])[0]
            gen = le_gen.transform([c["Gender"]])[0]
            row = np.array([[
                float(c.get("CreditScore", 600)),
                float(geo), float(gen),
                float(c.get("Age", 35)),
                float(c.get("Tenure", 3)),
                float(c.get("Balance", 0)),
                float(c.get("NumOfProducts", 1)),
                float(c.get("HasCrCard", 1)),
                float(c.get("IsActiveMember", 1)),
                float(c.get("EstimatedSalary", 50000)),
            ]])
            prob = float(model.predict_proba(row)[0][1])
            predictions.append({
                "id":               c.get("id", None),
                "churn_probability": round(prob * 100, 2),
                "prediction":       int(model.predict(row)[0]),
                "risk_level":       "High" if prob >= 0.6 else "Medium" if prob >= 0.35 else "Low",
            })
        except Exception as e:
            predictions.append({"id": c.get("id"), "error": str(e)})

    return jsonify(predictions)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
