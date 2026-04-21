"""
CineGenre — ML Pipeline
========================
Dataset  : IMDB via ftp://ftp.fu-berlin.de/pub/misc/movies/database/
Train    : 54,214 rows  →  ID ::: TITLE ::: GENRE ::: DESCRIPTION
Test     : 54,200 rows  →  ID ::: TITLE ::: DESCRIPTION
Genres   : 27 classes

Bugs fixed vs original code
───────────────────────────
#1  Test file has 3 columns, original code assumed 4  → has_labels flag
#2  Deprecated multi_class='multinomial' kwarg        → removed (lbfgs handles it)
#3  Single 80/20 split chose a lucky winner           → 5-fold cross_val_score
#4  Movie title completely ignored as feature          → title+desc concatenated
#5  Model & vectorizer thrown away after training     → joblib persistence
#6  Index misalignment after malformed-line drops     → reset_index(drop=True)
"""

from __future__ import annotations

import os
import re
import warnings
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT      = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA_DIR        = os.path.join(_REPO_ROOT, "data")
MODELS_DIR      = os.path.join(_HERE, "..", "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
CLASSES_PATH    = os.path.join(MODELS_DIR, "classes.pkl")

# 27 genres confirmed from real training data
ALL_GENRES = [
    "action", "adult", "adventure", "animation", "biography",
    "comedy", "crime", "documentary", "drama", "family",
    "fantasy", "game-show", "history", "horror", "music",
    "musical", "mystery", "news", "reality-tv", "romance",
    "sci-fi", "short", "sport", "talk-show", "thriller",
    "war", "western",
]

GENRE_COUNTS = {
    "drama": 13613, "documentary": 13096, "comedy": 7447, "short": 5073,
    "horror": 2204, "thriller": 1591, "action": 1315, "western": 1032,
    "reality-tv": 884, "family": 784, "adventure": 775, "music": 731,
    "romance": 672, "sci-fi": 647, "adult": 590, "crime": 505,
    "animation": 498, "sport": 432, "talk-show": 391, "fantasy": 323,
    "mystery": 319, "musical": 277, "biography": 265, "history": 243,
    "game-show": 194, "news": 181, "war": 132,
}


# ── Data loading ───────────────────────────────────────────────────────────────

def parse_file(path: str, has_labels: bool = True) -> pd.DataFrame:
    """
    Parse ` ::: ` delimited IMDB genre file.
      has_labels=True  → 4 fields: ID TITLE GENRE DESCRIPTION  (train)
      has_labels=False → 3 fields: ID TITLE DESCRIPTION         (test)  ← Bug #1 fix
    """
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(" ::: ")
            if has_labels and len(parts) == 4:
                records.append({
                    "id":          parts[0].strip(),
                    "title":       parts[1].strip(),
                    "genre":       parts[2].strip().lower(),
                    "description": parts[3].strip(),
                })
            elif not has_labels and len(parts) == 3:
                records.append({
                    "id":          parts[0].strip(),
                    "title":       parts[1].strip(),
                    "description": parts[2].strip(),
                })
    return pd.DataFrame(records).reset_index(drop=True)  # Bug #6 fix


def load_data(
    train_path: str | None = None,
    test_path:  str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = train_path or os.path.join(DATA_DIR, "train_data.txt")
    test_path  = test_path  or os.path.join(DATA_DIR, "test_data.txt")

    train_df = parse_file(train_path, has_labels=True)
    test_df  = parse_file(test_path,  has_labels=False)  # Bug #1 fix

    print(f"  Train : {len(train_df):,} samples")
    print(f"  Test  : {len(test_df):,} samples")
    print(f"  Genres: {train_df['genre'].nunique()} unique classes\n")
    print("  Top-10 genre distribution:")
    for g, c in train_df["genre"].value_counts().head(10).items():
        bar = "█" * int(c / 700)
        print(f"    {g:<14} {c:>6,}  {bar}")
    return train_df, test_df


# ── Preprocessing ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+",      " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Bug #4 fix — title carries strong genre signal, was completely ignored before
    df["combined"]   = df["title"].fillna("") + " " + df["description"].fillna("")
    df["clean_text"] = df["combined"].apply(clean_text)
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple:
    vectorizer = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),      # unigrams + bigrams
        stop_words="english",
        sublinear_tf=True,       # log-scale TF
        min_df=2,                # ignore very rare terms
    )
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test  = vectorizer.transform(test_df["clean_text"])
    y_train = train_df["genre"]
    print(f"  Feature matrix : {X_train.shape[0]:,} × {X_train.shape[1]:,}")
    return X_train, X_test, y_train, vectorizer


# ── Training & evaluation ──────────────────────────────────────────────────────

def train_and_evaluate(X_train, y_train):
    candidates = {
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=5, solver="lbfgs",
            # Bug #2 fix — multi_class='multinomial' is deprecated; lbfgs handles it natively
        ),
        "Linear SVM": LinearSVC(C=1.0, max_iter=3000),
    }

    best_model, best_mean = None, 0.0

    for name, model in candidates.items():
        # Bug #3 fix — 5-fold CV instead of a single lucky 80/20 split
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="accuracy", n_jobs=-1,
        )
        mean, std = cv_scores.mean(), cv_scores.std()
        print(f"  {name:<24}  CV: {mean:.4f} ± {std:.4f}  {['','← best'][mean > best_mean]}")
        if mean > best_mean:
            best_mean, best_model = mean, model

    # Detailed held-out report for the winner
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=42, stratify=y_train,
    )
    best_model.fit(X_tr, y_tr)
    val_preds = best_model.predict(X_val)
    print(f"\n  Held-out validation report (best model, 15% split):")
    print(f"  Accuracy : {accuracy_score(y_val, val_preds):.4f}")
    print(classification_report(y_val, val_preds, zero_division=0))
    return best_model


# ── Persistence ────────────────────────────────────────────────────────────────

def save_artifacts(model, vectorizer) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,                MODEL_PATH)       # Bug #5 fix
    joblib.dump(vectorizer,           VECTORIZER_PATH)
    joblib.dump(list(model.classes_), CLASSES_PATH)
    print(f"  ✅  best_model.pkl   → {MODEL_PATH}")
    print(f"  ✅  vectorizer.pkl   → {VECTORIZER_PATH}")
    print(f"  ✅  classes.pkl      → {CLASSES_PATH}")


def load_artifacts():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    classes    = joblib.load(CLASSES_PATH)
    return model, vectorizer, classes


def artifacts_exist() -> bool:
    return all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, CLASSES_PATH])


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_single(
    title: str,
    description: str,
    model=None,
    vectorizer=None,
) -> dict:
    """Single-movie inference — used by FastAPI."""
    if model is None or vectorizer is None:
        model, vectorizer, _ = load_artifacts()

    combined = clean_text(title + " " + description)
    X        = vectorizer.transform([combined])
    genre    = model.predict(X)[0]

    confidence: dict | None = None
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(X)[0]
        top_idx    = proba.argsort()[::-1][:5]
        confidence = {model.classes_[i]: round(float(proba[i]), 4) for i in top_idx}
    elif hasattr(model, "decision_function"):
        scores     = model.decision_function(X)[0]
        top_idx    = scores.argsort()[::-1][:5]
        confidence = {model.classes_[i]: round(float(scores[i]), 4) for i in top_idx}

    return {"genre": genre, "confidence": confidence}


def predict_and_save(
    model,
    X_test,
    test_df: pd.DataFrame,
    solution_path: str | None = None,
    out_path: str = "task1_predictions.csv",
) -> pd.DataFrame:
    preds  = model.predict(X_test)
    result = test_df[["id", "title"]].copy()
    result["predicted_genre"] = preds

    if solution_path and os.path.exists(solution_path):
        sol    = parse_file(solution_path, has_labels=True)
        result = result.merge(
            sol[["id", "genre"]].rename(columns={"genre": "true_genre"}),
            on="id", how="left",
        )
        acc = accuracy_score(result["true_genre"], result["predicted_genre"])
        print(f"\n  🎯  Test accuracy vs solution file: {acc:.4f}")

    result.to_csv(out_path, index=False)
    print(f"  📁  Predictions → {out_path}")
    print(result.head(8).to_string(index=False))
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "─" * 52
    print(f"\n{'═'*52}")
    print("  CineGenre — ML Training Pipeline")
    print(f"{'═'*52}\n")

    print(f"[1/4]  Loading data\n{sep}")
    train_df, test_df = load_data()

    print(f"\n[2/4]  Preprocessing\n{sep}")
    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)
    print("  Title + description combined → clean_text ✓")

    print(f"\n[3/4]  Building TF-IDF features\n{sep}")
    X_train, X_test, y_train, vectorizer = build_features(train_df, test_df)

    print(f"\n[4/4]  Training & cross-validating all models\n{sep}")
    best_model = train_and_evaluate(X_train, y_train)

    print(f"\n  Retraining winner on full training set …")
    best_model.fit(X_train, y_train)
    save_artifacts(best_model, vectorizer)

    sol_path = os.path.join(DATA_DIR, "test_data_solution.txt")
    predict_and_save(
        best_model, X_test, test_df,
        solution_path=sol_path,
        out_path="task1_predictions.csv",
    )
    print(f"\n{'═'*52}")
    print("  ✅  Pipeline complete!")
    print(f"{'═'*52}\n")
