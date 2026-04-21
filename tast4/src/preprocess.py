"""
preprocess.py
-------------
Handles all text cleaning and feature extraction for the SMS spam classifier.
This module is fully self-contained — no external NLP libraries required.

Steps applied to each message:
    1. Convert to lowercase
    2. Remove URLs, email addresses, and phone numbers
    3. Strip punctuation and digits
    4. Tokenize on whitespace
    5. Remove English stop words (built-in list below)
    6. Apply a lightweight Porter-style stemmer (built-in, no NLTK)
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Stop words ───────────────────────────────────────────────────────────────
# Standard English stop-word list kept inline so the project works in any
# environment without downloading NLTK corpora.
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "ll", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "us", "ve", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "you", "your", "yours", "yourself", "yourselves",
}



# ─── Minimal Porter-style stemmer (no external dependencies) ─────────────────

def _stem(word: str) -> str:
    """
    Apply simplified Porter stemming rules to reduce a word to its
    approximate root.  Covers the most common English suffixes found
    in SMS text without requiring any external library or download.

    Examples:
        "running"  -> "run"
        "claimed"  -> "claim"
        "prizes"   -> "prize"
        "easily"   -> "easi"
    """
    if len(word) <= 3:
        return word

    # Longer / more specific suffixes are checked first to avoid partial matches
    suffixes = [
        ("ational", "ate"),  ("tional", "tion"),  ("enci", "ence"),
        ("anci", "ance"),    ("izer", "ize"),      ("ising", "ise"),
        ("izing", "ize"),    ("ated", "ate"),      ("ating", "ate"),
        ("ation", "ate"),    ("ness", ""),         ("ment", ""),
        ("ings", ""),        ("ing", ""),          ("edly", ""),
        ("tion", "t"),       ("ions", "ion"),      ("ies", "i"),
        ("ers", "er"),       ("est", ""),          ("eed", "ee"),
        ("ed", ""),          ("ly", ""),           ("es", ""),
        ("er", ""),          ("s", ""),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement

    return word


# ─── Public API ───────────────────────────────────────────────────────────────

def clean_text(message: str) -> str:
    """
    Clean a single SMS message string.

    Args:
        message: Raw SMS text.

    Returns:
        A cleaned, stemmed string ready for TF-IDF vectorization.
    """
    # 1. Lowercase so "FREE" and "free" are the same token
    text = message.lower()

    # 2. Remove URLs (http/https and bare www. addresses)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Remove phone numbers (e.g. 07123456789 or +44 7911 123456)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", " ", text)

    # 5. Strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 6. Remove leftover standalone digits
    text = re.sub(r"\b\d+\b", " ", text)

    # 7. Tokenize
    tokens = text.split()

    # 8. Drop stop words and very short tokens
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]

    # 9. Stem to approximate root form
    tokens = [_stem(w) for w in tokens]

    return " ".join(tokens)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the SMS spam CSV and return a clean DataFrame with two columns:
    'label' (0 = ham, 1 = spam) and 'message' (cleaned text).

    The raw file has five columns; only the first two are used:
        v1 -- label ('ham' or 'spam')
        v2 -- raw message text

    Args:
        filepath: Path to spam.csv.

    Returns:
        DataFrame with columns ['label', 'message'].
    """
    df = pd.read_csv(filepath, encoding="latin-1", usecols=[0, 1], header=0)
    df.columns = ["label", "message"]

    df.dropna(subset=["message"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Encode: ham -> 0, spam -> 1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    print("  Cleaning text ...", end=" ", flush=True)
    df["message"] = df["message"].apply(clean_text)
    print("done.")

    return df


def build_vectorizer(max_features: int = 5000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with sensible defaults for SMS spam detection.

    Args:
        max_features:  Vocabulary size cap.  5000 covers the vast majority
                       of meaningful SMS words without overfitting.
        ngram_range:   (1, 2) captures single words and two-word phrases
                       such as "free entry" and "click here".

    Returns:
        An unfitted TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,   # Use 1 + log(TF) to dampen very frequent terms
        min_df=2,            # Ignore tokens in fewer than 2 docs (noise/typos)
    )
