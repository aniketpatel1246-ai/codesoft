# SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham (legitimate)** using classic NLP techniques. This project compares three classifiers — Naive Bayes, Logistic Regression, and Support Vector Machine — combined with TF-IDF feature extraction.

---

## Dataset

The dataset used is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), which contains **5,572 SMS messages** labeled as spam or ham.

| Label | Count |
|-------|-------|
| Ham   | 4,825 |
| Spam  | 747   |

Place the dataset at `data/spam.csv` before running any scripts.

---

## Project Structure

```
sms-spam-classifier/
│
├── data/
│   └── spam.csv                  # Raw SMS dataset
│
├── src/
│   ├── preprocess.py             # Text cleaning and preprocessing
│   ├── train.py                  # Train and save all models
│   ├── evaluate.py               # Evaluate models and generate reports
│   └── predict.py                # Predict on new SMS messages
│
├── models/                       # Saved model files (.pkl)
│
├── results/                      # Classification reports, confusion matrices
│
├── notebooks/
│   └── exploration.ipynb         # EDA and experimentation notebook
│
├── tests/
│   └── test_predict.py           # Unit tests
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Models

```bash
python src/train.py
```

This will train Naive Bayes, Logistic Regression, and SVM models and save them in the `models/` folder.

### 5. Evaluate the Models

```bash
python src/evaluate.py
```

This prints a full classification report for each model and saves confusion matrix plots to `results/`.

### 6. Predict on a New Message

```bash
python src/predict.py --message "Congratulations! You've won a free iPhone. Click here to claim."
```

---

## Models and Approach

### Feature Extraction
- **TF-IDF (Term Frequency–Inverse Document Frequency)** with unigrams and bigrams
- Custom text preprocessing: lowercasing, punctuation removal, stop word removal, stemming

### Classifiers

| Model               | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| Naive Bayes         | Fast probabilistic classifier, great baseline for text data       |
| Logistic Regression | Linear classifier with strong performance on sparse features      |
| SVM (Linear Kernel) | Maximum margin classifier, excellent for high-dimensional text    |

---

## Results

After training, a comparison table is printed to the console and saved to `results/model_comparison.txt`.

Sample output:

```
Model                  Accuracy    Precision   Recall      F1 Score
--------------------------------------------------------------------
Naive Bayes            97.85%      96.72%      93.40%      95.03%
Logistic Regression    98.74%      97.89%      96.23%      97.05%
SVM (Linear)           98.92%      98.11%      96.70%      97.40%
```

---

## Running Tests

```bash
python -m pytest tests/
```

---

## License

This project is open-source and available under the MIT License.
