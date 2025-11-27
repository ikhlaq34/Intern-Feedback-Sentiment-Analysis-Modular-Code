"""
modeling.py
Contains model training, evaluation wrappers, saving/loading utilities.
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Any, Tuple
import numpy as np


def train_logistic_regression(X, y, random_state: int = 42, max_iter: int = 1000) -> LogisticRegression:
    """Train and return a LogisticRegression model."""
    model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    model.fit(X, y)
    return model


def evaluate_classifier(model: Any, X_test, y_test, target_names=None) -> dict:
    """Run basic evaluation and return results in a dict."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    return {
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm,
        'predictions': preds
    }


def save_model(model: Any, path: str) -> None:
    """Persist model or vectorizer using joblib."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load a persisted object saved by joblib."""
    return joblib.load(path)


def predict_with_confidence(model: Any, X) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, confidences) where confidence is max probability per row."""
    probs = model.predict_proba(X)
    preds = model.predict(X)
    confidences = probs.max(axis=1)
    return preds, confidences


if __name__ == '__main__':
    # tiny smoke test — requires sklearn objects in scope, so only illustrative
    print('Run model functions from main script')