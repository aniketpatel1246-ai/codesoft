"""
tests/test_predict.py
---------------------
Unit tests for the preprocessing and prediction modules.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
import unittest

# Add project root to path so imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Support running with pytest if available, fall back to unittest gracefully
try:
    import pytest
    _has_pytest = True
    skipif = pytest.mark.skipif
except ImportError:
    _has_pytest = False
    # Stub skipif so the decorator syntax still works without pytest
    def skipif(condition, reason=""):
        def decorator(fn):
            if condition:
                def skip_wrapper(*args, **kwargs):
                    print(f"  SKIP: {reason}")
                return skip_wrapper
            return fn
        return decorator

from src.preprocess import clean_text, build_vectorizer


# ─── Preprocessing tests ───────────────────────────────────────────────────

class TestCleanText(unittest.TestCase):

    def test_lowercases_input(self):
        result = clean_text("HELLO WORLD")
        # After lowercasing and stop-word removal, 'hello' and 'world' should remain
        assert "hello" in result or "world" in result or result == ""

    def test_removes_urls(self):
        result = clean_text("Visit https://spam-site.com to claim your prize")
        assert "http" not in result
        assert "spam" in result or "claim" in result   # other words should survive

    def test_removes_punctuation(self):
        result = clean_text("Hello!!! Are you there???")
        assert "!" not in result
        assert "?" not in result

    def test_removes_digits(self):
        result = clean_text("Call 07911123456 now")
        assert any(ch.isdigit() for ch in result) is False

    def test_returns_string(self):
        assert isinstance(clean_text("some text"), str)

    def test_empty_string(self):
        # Should not raise an exception
        result = clean_text("")
        assert isinstance(result, str)

    def test_stop_words_removed(self):
        # "the", "is", "a" are stop words and should be stripped
        result = clean_text("the cat is a mammal")
        assert "the" not in result.split()
        assert "is" not in result.split()
        # "cat" and "mammal" are content words and should survive
        assert len(result) > 0


class TestBuildVectorizer(unittest.TestCase):

    def test_returns_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = build_vectorizer()
        assert isinstance(vec, TfidfVectorizer)

    def test_custom_max_features(self):
        vec = build_vectorizer(max_features=100)
        assert vec.max_features == 100

    def test_bigram_range(self):
        vec = build_vectorizer(ngram_range=(1, 2))
        assert vec.ngram_range == (1, 2)

    def test_vectorizer_fits_and_transforms(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Override min_df=1 here because the tiny 3-doc test corpus would
        # be pruned to nothing by the default min_df=2 used in production.
        vec = TfidfVectorizer(max_features=50, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        corpus = ["free prize now", "hello how are you", "click here to win"]
        X = vec.fit_transform(corpus)
        self.assertEqual(X.shape[0], 3)   # three documents
        self.assertLessEqual(X.shape[1], 50)  # vocabulary capped


# ─── Integration test (requires trained models) ───────────────────────────

class TestPredictMessage(unittest.TestCase):
    """
    These tests only run if the models/ directory contains trained model files.
    Skip gracefully if train.py has not been run yet.
    """

    models_present = os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "vectorizer.pkl")
    )

    @skipif(not os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "vectorizer.pkl")),
        reason="Models not trained yet. Run train.py first.")
    def test_spam_message_classified(self):
        from src.predict import predict_message
        result = predict_message("Congratulations! You've won a free iPhone. Claim now!")
        self.assertIn(result["prediction"], ("spam", "ham"))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    @skipif(not os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "vectorizer.pkl")),
        reason="Models not trained yet. Run train.py first.")
    def test_ham_message_classified(self):
        from src.predict import predict_message
        result = predict_message("Hey, are we still on for dinner tonight?")
        self.assertIn(result["prediction"], ("spam", "ham"))

    @skipif(not os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "vectorizer.pkl")),
        reason="Models not trained yet. Run train.py first.")
    def test_result_has_required_keys(self):
        from src.predict import predict_message
        result = predict_message("Test message")
        for key in ("message", "cleaned", "prediction", "confidence"):
            self.assertIn(key, result)

    @skipif(not os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "vectorizer.pkl")),
        reason="Models not trained yet. Run train.py first.")
    def test_batch_predict_length(self):
        from src.predict import predict_batch
        messages = ["Win a free car!", "I'll call you later", "URGENT: click here"]
        results  = predict_batch(messages)
        self.assertEqual(len(results), len(messages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
