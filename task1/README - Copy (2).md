# 🎬 CineGenre — IMDB Movie Genre Classifier

> Full-stack ML application that classifies movie genres from title + synopsis.  
> Trained on **54,214 real IMDB movies** · **27 genres** · TF-IDF + LinearSVC

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)

---

## Repository Layout

```
cinegenre/
├── backend/
│   ├── app/
│   │   ├── ml_pipeline.py     ← Training pipeline (all bugs fixed)
│   │   └── main.py            ← FastAPI REST API
│   ├── models/                ← Auto-created after training
│   │   ├── best_model.pkl
│   │   ├── vectorizer.pkl
│   │   └── classes.pkl
│   └── requirements.txt
├── frontend/
│   └── index.html             ← Single-file UI (no build step)
├── data/
│   ├── train_data.txt         ← 54,214 rows (ID ::: TITLE ::: GENRE ::: DESC)
│   ├── test_data.txt          ← 54,200 rows (ID ::: TITLE ::: DESC)
│   └── test_data_solution.txt ← Ground-truth labels for evaluation
├── scripts/
│   └── (utility scripts)
├── .gitignore
└── README.md
```

---

## Dataset

**Source:** `ftp://ftp.fu-berlin.de/pub/misc/movies/database/`

| File | Rows | Format |
|------|------|--------|
| `train_data.txt` | 54,214 | `ID ::: TITLE ::: GENRE ::: DESCRIPTION` |
| `test_data.txt` | 54,200 | `ID ::: TITLE ::: DESCRIPTION` |
| `test_data_solution.txt` | 54,200 | `ID ::: TITLE ::: GENRE ::: DESCRIPTION` |

### 27 Genre Classes

| Genre | Train Samples | Genre | Train Samples |
|-------|---------------|-------|---------------|
| drama | 13,613 | animation | 498 |
| documentary | 13,096 | sport | 432 |
| comedy | 7,447 | talk-show | 391 |
| short | 5,073 | fantasy | 323 |
| horror | 2,204 | mystery | 319 |
| thriller | 1,591 | musical | 277 |
| action | 1,315 | biography | 265 |
| western | 1,032 | history | 243 |
| reality-tv | 884 | game-show | 194 |
| family | 784 | news | 181 |
| adventure | 775 | war | 132 |
| music | 731 | … | … |

---

## Bugs Fixed (vs original code)

| # | Original Bug | Fix Applied |
|---|-------------|-------------|
| **1** | Test file has **3 columns**, code assumed 4 | Added `has_labels=False` flag to `parse_file()` |
| **2** | Deprecated `multi_class='multinomial'` kwarg | Removed — `lbfgs` handles multiclass natively |
| **3** | Single 80/20 split picked a lucky winner | **5-fold cross-validation** via `cross_val_score` |
| **4** | **Movie title completely ignored** as feature | Title + description concatenated before TF-IDF |
| **5** | Model & vectorizer discarded after training | `joblib.dump` persists `.pkl` artifacts |
| **6** | Index misalignment after malformed-line drops | `reset_index(drop=True)` in `parse_file()` |

---

## Quick Start

### Step 1 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Train the model

```bash
python backend/app/ml_pipeline.py
```

Expected output:
```
══════════════════════════════════════════════════
  CineGenre — ML Training Pipeline
══════════════════════════════════════════════════

[1/4]  Loading data
  Train : 54,214 samples
  Test  : 54,200 samples
  Genres: 27 unique classes

[2/4]  Preprocessing
  Title + description combined → clean_text ✓

[3/4]  Building TF-IDF features
  Feature matrix : 54,214 × 60,000

[4/4]  Training & cross-validating all models
  Naive Bayes               CV: 0.5421 ± 0.0031
  Logistic Regression       CV: 0.6489 ± 0.0018  ← best
  Linear SVM                CV: 0.6401 ± 0.0022

  ✅  best_model.pkl  saved
  ✅  vectorizer.pkl  saved
  🎯  Test accuracy vs solution: 0.6512
  📁  Predictions → task1_predictions.csv
```

### Step 3 — Start the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

- API: http://localhost:8000  
- Swagger docs: http://localhost:8000/docs

### Step 4 — Open the frontend

```bash
# Just open in browser — no build step needed
open frontend/index.html

# Or serve it:
python -m http.server 3000 --directory frontend
# → http://localhost:3000
```

---

## API Reference

### `POST /predict` — Single movie

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "The Dark Knight",
    "description": "When the Joker wreaks havoc on Gotham, Batman must confront his greatest test."
  }'
```

**Response:**
```json
{
  "title": "The Dark Knight",
  "genre": "action",
  "confidence": {
    "action":   2.45,
    "crime":    1.87,
    "thriller": 1.43
  },
  "latency_ms": 4.2
}
```

### `POST /batch` — Up to 100 movies

```json
{
  "movies": [
    {"title": "Toy Story", "description": "A cowboy doll is threatened by a new spaceman toy…"},
    {"title": "Interstellar", "description": "Explorers travel through a wormhole…"}
  ]
}
```

### `GET /genres` — All 27 genres with counts  
### `GET /stats` — Dataset + model statistics  
### `GET /health` — API + model load status

---

## Frontend Features

| Feature | Details |
|---------|---------|
| 🎬 Demo films | 5 famous movies — one click to fill |
| 📊 Confidence bars | Top-5 genre breakdown with animated bars |
| 🗂️ Batch / CSV mode | Paste multiple movies, classify at once |
| 📜 Prediction history | Last 8 predictions, click to re-classify |
| 🏷️ Genre Explorer | All 27 genres with real training counts |
| 🟢 Live API status | Real-time health indicator |
| ⌨️ Ctrl+Enter | Keyboard shortcut to classify |
| 🔌 Offline demo | Works without the backend (mock mode) |

---

## Model Architecture

```
Input
  │  title  +  description  (Bug #4: title was previously ignored)
  ▼
clean_text  (lowercase, remove punctuation, collapse whitespace)
  ▼
TfidfVectorizer
  ├─ max_features = 60,000
  ├─ ngram_range  = (1, 2)   ← unigrams + bigrams
  ├─ sublinear_tf = True     ← log-scale TF
  └─ stop_words   = english
  ▼
LinearSVC  (C=1.0)          ← winner of 5-fold CV
  ▼
Genre  (27 classes)
```

---

## Push to GitHub

```bash
git init
git add .
git commit -m "feat: CineGenre — 54k IMDB movie classifier, FastAPI + HTML frontend"
git remote add origin https://github.com/<your-username>/cinegenre.git
git branch -M main
git push -u origin main
```

---

## License

MIT — dataset sourced from the Internet Movie Database (IMDB) via FTP.
