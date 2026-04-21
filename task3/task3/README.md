# 🎯 ChurnSight — Customer Churn Prediction

A full-stack machine learning application that predicts customer churn for subscription-based services using **Logistic Regression**, **Random Forest**, and **Gradient Boosting** models.

---

## 📁 Repository Structure

```
churn-predictor/
├── backend/
│   ├── app.py               # Flask REST API
│   ├── train_models.py      # Model training script
│   ├── requirements.txt     # Python dependencies
│   ├── data/
│   │   └── Churn_Modelling.csv
│   └── models/              # Generated after training
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── le_geo.pkl
│       ├── le_gen.pkl
│       └── summary.json
└── frontend/
    └── index.html           # Single-file React-free dashboard
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd churn-predictor
```

### 2. Backend — Train & Run

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train all models (saves artifacts to models/)
python train_models.py

# Start the API server
python app.py
# → Running on http://localhost:5000
```

### 3. Frontend

Open `frontend/index.html` in any browser.

> **Note:** The dashboard works in demo mode even without the backend running. Connect to the live API for real predictions.

---

## 📊 Dataset

**Source:** Bank Customer Churn Dataset (`Churn_Modelling.csv`)

| Column | Type | Description |
|---|---|---|
| CreditScore | Numeric | Credit score (300–850) |
| Geography | Category | France / Germany / Spain |
| Gender | Category | Male / Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years with bank (0–10) |
| Balance | Numeric | Account balance ($) |
| NumOfProducts | Numeric | Number of products held |
| HasCrCard | Binary | Has credit card? |
| IsActiveMember | Binary | Active in last period? |
| EstimatedSalary | Numeric | Annual salary estimate |
| **Exited** | **Target** | **1 = Churned, 0 = Retained** |

**Stats:** 10,000 rows · 20.4% churn rate

---

## 🤖 Model Results

| Model | Accuracy | AUC | Notes |
|---|---|---|---|
| Logistic Regression | 80.50% | 0.7710 | Baseline |
| Random Forest | 86.40% | 0.8464 | Good |
| **Gradient Boosting** | **86.75%** | **0.8673** | ⭐ Best |

---

## 🔌 API Reference

### `GET /api/health`
Returns server status.

### `GET /api/stats`
Returns model comparison stats, feature importances, and dataset overview.

### `POST /api/predict`
Predict churn for a single customer.

**Request body:**
```json
{
  "CreditScore": 620,
  "Geography": "France",
  "Gender": "Male",
  "Age": 38,
  "Tenure": 5,
  "Balance": 80000,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 55000
}
```

**Response:**
```json
{
  "churn_probability": 18.4,
  "prediction": 0,
  "risk_level": "Low",
  "model_used": "Gradient Boosting"
}
```

### `POST /api/batch_predict`
Accepts a JSON array of customers and returns predictions for each.

---

## 🖥️ Frontend Features

- **Dashboard** — dataset stats, churn distribution chart, model AUC comparison
- **Predict Customer** — fill in customer details to get instant churn risk
- **Model Comparison** — accuracy/AUC charts + confusion matrix
- **Feature Insights** — feature importance bars + field descriptions

---

## 🧠 Key Findings

1. **Age** is the strongest predictor — older customers churn more
2. **Balance** matters — high-balance customers with few products are at risk
3. **Number of Products** has a non-linear effect (3+ products = high risk)
4. **Active Members** churn significantly less
5. **Germany** has higher churn than France/Spain

---

## 🔧 Extending the Project

- Add XGBoost or LightGBM for potentially higher AUC
- Implement SHAP values for per-customer explainability
- Add a `/api/batch_predict` CSV upload endpoint
- Deploy backend to Render / Railway / AWS Lambda
- Deploy frontend to Vercel / Netlify

---

## 📝 License

MIT
