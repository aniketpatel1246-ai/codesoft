"""
predict.py
----------
Use any trained model to classify one or more SMS messages as spam or ham.

Usage — command line:
    python src/predict.py --message "Win a free iPhone now!"
    python src/predict.py --message "Hey, are you free tonight?" --model svm

Usage — as a module:
    from src.predict import predict_message
    result = predict_message("Claim your prize now!", model_name="logistic_regression")
    print(result)   # {'message': ..., 'prediction': 'spam', 'confidence': 0.97}
"""

import os
import sys
import argparse
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

AVAILABLE_MODELS = ["naive_bayes", "logistic_regression", "svm"]


def load_artifacts(model_name: str):
    """
    Load the TF-IDF vectorizer and the requested classifier from disk.

    Args:
        model_name: One of 'naive_bayes', 'logistic_regression', 'svm'.

    Returns:
        Tuple of (vectorizer, classifier).

    Raises:
        FileNotFoundError: If either .pkl file is missing.
    """
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    model_path      = os.path.join(MODELS_DIR, f"{model_name}.pkl")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "Vectorizer not found. Run  python src/train.py  first."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found. Run  python src/train.py  first."
        )

    vectorizer = joblib.load(vectorizer_path)
    clf        = joblib.load(model_path)
    return vectorizer, clf


def predict_message(message: str, model_name: str = "svm") -> dict:
    """
    Classify a single SMS message.

    Args:
        message:    Raw SMS text (as the user would type it).
        model_name: Which trained model to use.

    Returns:
        Dictionary with keys:
            'message'    -- the original input text
            'cleaned'    -- the preprocessed version
            'prediction' -- 'spam' or 'ham'
            'confidence' -- probability or decision score (float)
    """
    vectorizer, clf = load_artifacts(model_name)

    cleaned = clean_text(message)
    X       = vectorizer.transform([cleaned])

    label_index = clf.predict(X)[0]
    prediction  = "spam" if label_index == 1 else "ham"

    # Not all classifiers expose predict_proba; LinearSVC uses decision_function
    if hasattr(clf, "predict_proba"):
        proba      = clf.predict_proba(X)[0]
        confidence = float(proba[label_index])
    elif hasattr(clf, "decision_function"):
        score      = clf.decision_function(X)[0]
        # Squash the raw margin into [0, 1] using a sigmoid for a rough confidence
        confidence = float(1 / (1 + np.exp(-score))) if label_index == 1 else float(1 / (1 + np.exp(score)))
    else:
        confidence = 1.0

    return {
        "message":    message,
        "cleaned":    cleaned,
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }


def predict_batch(messages: list, model_name: str = "svm") -> list:
    """
    Classify a list of SMS messages in one vectorizer call (more efficient
    than calling predict_message in a loop for large batches).

    Args:
        messages:   List of raw SMS strings.
        model_name: Which trained model to use.

    Returns:
        List of result dictionaries (same structure as predict_message).
    """
    vectorizer, clf = load_artifacts(model_name)

    cleaned_messages = [clean_text(m) for m in messages]
    X                = vectorizer.transform(cleaned_messages)
    label_indices    = clf.predict(X)

    results = []
    for i, (original, cleaned, label_idx) in enumerate(
        zip(messages, cleaned_messages, label_indices)
    ):
        prediction = "spam" if label_idx == 1 else "ham"

        if hasattr(clf, "predict_proba"):
            proba      = clf.predict_proba(X[i])[0]
            confidence = float(proba[label_idx])
        elif hasattr(clf, "decision_function"):
            score      = clf.decision_function(X[i])[0]
            confidence = float(1 / (1 + np.exp(-score))) if label_idx == 1 else float(1 / (1 + np.exp(score)))
        else:
            confidence = 1.0

        results.append({
            "message":    original,
            "cleaned":    cleaned,
            "prediction": prediction,
            "confidence": round(confidence, 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Classify an SMS message as spam or ham."
    )
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        help='The SMS text to classify.  E.g. --message "You won a prize!"',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=AVAILABLE_MODELS,
        help="Which model to use for prediction (default: svm).",
    )
    args = parser.parse_args()

    result = predict_message(args.message, model_name=args.model)

    print("\n" + "=" * 55)
    print(f"  Model      : {args.model}")
    print(f"  Message    : {result['message']}")
    print(f"  Prediction : {result['prediction'].upper()}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
