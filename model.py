"""
model.py — Heart Disease Prediction ML Pipeline
================================================
Trains Naive Bayes, KNN, SVM, and Random Forest on the Cleveland
Heart Disease dataset, selects the best model, and persists it with joblib.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "heart.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# 13 Cleveland heart-disease feature names (in order)
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]


# ─── 1. Data Loading & Preprocessing ─────────────────────────────────────────
def load_and_preprocess(path: str):
    """Load CSV, handle missing values, split features/target, scale."""
    log.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    # Replace common missing-value placeholders
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing = df.isnull().sum().sum()
    if missing:
        log.warning("Found %d missing values — filling with column medians", missing)
        df.fillna(df.median(numeric_only=True), inplace=True)

    # Ensure 'target' is binary (some variants use 0–4)
    df["target"] = (df["target"] > 0).astype(int)

    X = df[FEATURE_NAMES].values
    y = df["target"].values

    log.info("Dataset shape: %s  |  Positive cases: %d / %d", df.shape, y.sum(), len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─── 2. Model Definitions ────────────────────────────────────────────────────
def build_models():
    """Return a dict of candidate models."""
    return {
        "Naive Bayes":     GaussianNB(),
        "KNN":             KNeighborsClassifier(n_neighbors=5, metric="minkowski"),
        "SVM":             SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest":   RandomForestClassifier(
                               n_estimators=200, max_depth=8,
                               random_state=42, n_jobs=-1
                           ),
    }


# ─── 3. Training & Evaluation ────────────────────────────────────────────────
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Fit every model, print metrics, return (best_name, best_model, best_acc)."""
    results = {}

    for name, model in models.items():
        log.info("Training  %-16s …", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cv  = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results[name] = {"model": model, "accuracy": acc, "cv_mean": cv.mean()}

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  CV (5-fold)   : {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Disease','Heart Disease'])}")

    # Pick the model with the highest test accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best      = results[best_name]
    log.info("Best model → %s  (accuracy=%.4f)", best_name, best["accuracy"])
    return best_name, best["model"], best["accuracy"]


# ─── 4. Persistence ──────────────────────────────────────────────────────────
def save_artifacts(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    log.info("Model  saved → %s", MODEL_PATH)
    log.info("Scaler saved → %s", SCALER_PATH)


def load_artifacts():
    """Load persisted model and scaler (used by app.py)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run  python model.py  first to train."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    log.info("Loaded model and scaler from disk")
    return model, scaler


# ─── 5. Prediction Helper ────────────────────────────────────────────────────
def predict(features: list, model=None, scaler=None):
    """
    Accept a list/array of 13 feature values and return:
        { "prediction": 0 or 1,
          "label":      "Heart Disease" | "No Heart Disease",
          "confidence": float (0–1) }
    """
    if model is None or scaler is None:
        model, scaler = load_artifacts()

    if len(features) != 13:
        raise ValueError(f"Expected 13 features, got {len(features)}")

    arr = np.array(features, dtype=float).reshape(1, -1)
    arr_sc = scaler.transform(arr)

    pred  = int(model.predict(arr_sc)[0])
    proba = model.predict_proba(arr_sc)[0] if hasattr(model, "predict_proba") else None

    confidence = float(proba[pred]) if proba is not None else None
    label = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    return {"prediction": pred, "label": label, "confidence": confidence}


# ─── Entry-point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(DATA_PATH)
    models = build_models()
    best_name, best_model, best_acc = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )
    save_artifacts(best_model, scaler)

    print(f"\n✅  Training complete — best model: {best_name}  ({best_acc:.2%})")
    print(f"   Artifacts saved to: {MODEL_DIR}/")
