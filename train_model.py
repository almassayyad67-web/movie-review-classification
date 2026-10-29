"""
Train a movie-review sentiment classifier using the NLTK movie_reviews corpus.
Run: python train_model.py
This downloads the public NLTK corpus on first run, trains a TF-IDF + Logistic
Regression pipeline, evaluates it on a held-out test set, and saves artifacts.
"""
import json
import os
import nltk
import joblib
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_model.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    try:
        movie_reviews.categories()
    except LookupError:
        nltk.download("movie_reviews", quiet=False)
        nltk.download("punkt", quiet=True)

    documents, labels = [], []
    for category in movie_reviews.categories():
        for file_id in movie_reviews.fileids(category):
            documents.append(" ".join(movie_reviews.words(file_id)))
            labels.append(category)  # pos or neg

    X_train, X_test, y_train, y_test = train_test_split(
        documents, labels, test_size=0.20, random_state=42, stratify=labels
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=50000, sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "dataset": "NLTK movie_reviews corpus",
        "total_reviews": len(documents),
        "train_reviews": len(X_train),
        "test_reviews": len(X_test),
        "accuracy": round(float(accuracy), 4),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": ["neg", "pos"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["neg", "pos"]
        ).tolist()
    }
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("Confusion matrix (rows=true, columns=predicted; labels neg, pos):")
    print(metrics["confusion_matrix"])
    print(f"Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
