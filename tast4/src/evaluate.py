"""
evaluate.py
-----------
Loads all trained models and the held-out test set, prints a detailed
classification report for each classifier, and saves:

    results/confusion_matrix_<model>.png   -- heatmap per model
    results/model_comparison.txt           -- side-by-side accuracy table

Usage:
    python src/evaluate.py
"""

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Friendly display names for the three models
MODEL_FILES = {
    "Naive Bayes":          "naive_bayes.pkl",
    "Logistic Regression":  "logistic_regression.pkl",
    "SVM (Linear)":         "svm.pkl",
}


def load_test_data():
    """Load the pre-saved test split from the models/ directory."""
    X_test = np.load(os.path.join(MODELS_DIR, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(MODELS_DIR, "y_test.npy"))
    return X_test, y_test


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str) -> None:
    """Draw a labelled heatmap and save it to disk."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate():
    print("=" * 65)
    print("  SMS Spam Classifier — Evaluation Report")
    print("=" * 65)

    # ── Load test data ───────────────────────────────────────────
    X_test, y_test = load_test_data()
    print(f"\nTest set: {len(y_test)} messages  "
          f"(spam: {y_test.sum()}, ham: {(y_test == 0).sum()})\n")

    # Summary table rows are collected here and written to a file later
    summary_rows = []
    header = f"{'Model':<22}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}"
    divider = "-" * len(header)

    print(header)
    print(divider)

    for display_name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(model_path):
            print(f"  [SKIP] {display_name} — model file not found. Run train.py first.")
            continue

        clf = joblib.load(model_path)
        y_pred = clf.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        row = f"{display_name:<22}  {acc:>9.4f}  {prec:>9.4f}  {rec:>9.4f}  {f1:>9.4f}"
        print(row)
        summary_rows.append(row)

        # Detailed report
        print(f"\n  Classification Report — {display_name}")
        print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        safe_name = filename.replace(".pkl", "")
        img_path  = os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png")
        plot_confusion_matrix(cm, display_name, img_path)
        print(f"  Confusion matrix saved → results/confusion_matrix_{safe_name}.png\n")
        print(divider)

    # ── Write comparison file ────────────────────────────────────
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.txt")
    with open(comparison_path, "w") as f:
        f.write("SMS Spam Classifier — Model Comparison\n")
        f.write("=" * len(header) + "\n")
        f.write(header + "\n")
        f.write(divider + "\n")
        for row in summary_rows:
            f.write(row + "\n")

    print(f"\n✓ Comparison table saved → results/model_comparison.txt")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
