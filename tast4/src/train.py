"""
train.py
--------
Trains three classifiers on the SMS spam dataset and saves each model
(along with the fitted TF-IDF vectorizer) to the models/ directory.

Usage:
    python src/train.py

Output files:
    models/vectorizer.pkl         -- Fitted TF-IDF vectorizer
    models/naive_bayes.pkl        -- Trained Multinomial Naive Bayes
    models/logistic_regression.pkl-- Trained Logistic Regression
    models/svm.pkl                -- Trained Linear SVM
"""

import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Make sure the src package is importable when this script is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_dataset, build_vectorizer

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "spam.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def get_classifiers() -> dict:
    """
    Return a dictionary of classifier name -> unfitted classifier.

    Naive Bayes:
        alpha=0.1 is a mild Laplace smoothing value.  The default (1.0)
        tends to over-smooth on short text like SMS.

    Logistic Regression:
        C=1.0 is the inverse regularisation strength.  'lbfgs' is
        memory-efficient and converges well on sparse TF-IDF features.
        max_iter=1000 avoids ConvergenceWarnings on larger datasets.

    LinearSVC:
        C=0.5 keeps the margin wide, which works well when the class
        imbalance is moderate (spam ~13 %).  dual=False is preferred
        when n_samples > n_features, which is true after TF-IDF.
    """
    return {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        "svm": LinearSVC(C=0.5, dual=False, max_iter=2000),
    }


def print_cross_val_scores(name: str, clf, X, y) -> None:
    """Run 5-fold stratified CV and print mean ± std F1 score."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  {name:<25} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")


def train():
    print("=" * 60)
    print("  SMS Spam Classifier — Training Pipeline")
    print("=" * 60)

    # ── 1. Load and preprocess the dataset ──────────────────────
    print("\n[1/4] Loading dataset ...")
    df = load_dataset(DATA_PATH)
    print(f"      {len(df)} messages loaded  "
          f"(spam: {df['label'].sum()}, ham: {(df['label'] == 0).sum()})")

    # ── 2. Vectorise with TF-IDF ─────────────────────────────────
    print("\n[2/4] Vectorising text with TF-IDF ...")
    vectorizer = build_vectorizer(max_features=5000, ngram_range=(1, 2))

    # Fit on the entire dataset so the vocabulary covers all messages;
    # we hold out a test split for final evaluation in evaluate.py.
    X = vectorizer.fit_transform(df["message"])
    y = df["label"].values
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,} tokens")

    # Save the fitted vectorizer — it must be the same object used at
    # inference time so the token indices stay consistent.
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    print("      Vectorizer saved → models/vectorizer.pkl")

    # ── 3. Cross-validation summary ─────────────────────────────
    print("\n[3/4] 5-fold cross-validation (F1 on full dataset) ...")
    classifiers = get_classifiers()
    for name, clf in classifiers.items():
        print_cross_val_scores(name, clf, X, y)

    # ── 4. Train on 80 % and save each model ────────────────────
    print("\n[4/4] Training on 80 / 20 split and saving models ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Persist the split indices so evaluate.py uses the same test set
    np.save(os.path.join(MODELS_DIR, "X_test.npy"), X_test.toarray())
    np.save(os.path.join(MODELS_DIR, "y_test.npy"), y_test)

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(clf, model_path)
        print(f"      {name:<25} saved → models/{name}.pkl")

    print("\n✓ All models trained and saved successfully.\n")
    print("  Next step: run  python src/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
