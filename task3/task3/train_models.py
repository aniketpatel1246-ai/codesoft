"""
train_models.py
Run this once to train all models and save artifacts.
Usage: python train_models.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "Churn_Modelling.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Load & clean ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

le_geo = LabelEncoder()
le_gen = LabelEncoder()
df["Geography"] = le_geo.fit_transform(df["Geography"])
df["Gender"]    = le_gen.fit_transform(df["Gender"])

X = df.drop("Exited", axis=1)
y = df["Exited"]
FEATURE_NAMES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_s   = scaler.transform(X_test)

# ─── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression":  (LogisticRegression(max_iter=1000, random_state=42), True),
    "Random Forest":        (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "Gradient Boosting":    (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
}

results      = {}
best_auc     = 0
best_name    = None
best_model   = None

for name, (mdl, use_scaled) in MODELS.items():
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s  if use_scaled else X_test

    mdl.fit(Xtr, y_train)
    y_pred = mdl.predict(Xte)
    y_prob = mdl.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "accuracy":              round(float(acc) * 100, 2),
        "auc":                   round(float(auc), 4),
        "confusion_matrix":      cm.tolist(),
        "classification_report": rep,
    }
    print(f"  {name:25s}  ACC={acc:.4f}  AUC={auc:.4f}")

    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, mdl

# ─── Feature importances (Random Forest) ─────────────────────────────────────
rf_model     = MODELS["Random Forest"][0]
importances  = rf_model.feature_importances_
feat_imp     = sorted(zip(FEATURE_NAMES, importances.tolist()), key=lambda x: -x[1])

# ─── Save artifacts ───────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le_geo,     os.path.join(MODEL_DIR, "le_geo.pkl"))
joblib.dump(le_gen,     os.path.join(MODEL_DIR, "le_gen.pkl"))

summary = {
    "best_model":          best_name,
    "best_auc":            round(best_auc, 4),
    "feature_names":       FEATURE_NAMES,
    "feature_importances": feat_imp,
    "model_results":       results,
    "churn_rate":          round(float(y.mean()) * 100, 2),
    "dataset_size":        len(df),
    "test_size":           len(y_test),
}
with open(os.path.join(MODEL_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅  Best model: {best_name}  (AUC {best_auc:.4f})")
print(f"   Artifacts saved to: {MODEL_DIR}")
