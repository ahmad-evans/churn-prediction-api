"""
Trains the churn model and saves it for the API.

    python train.py

Writes:
    models/churn_model.joblib   the fitted scikit-learn pipeline
    models/metadata.json        version, threshold, test metrics, training date
"""
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features import CATEGORICAL, NUMERIC, add_features

MODEL_VERSION = "1.0.0"
SEED = 42
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def build_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "telco_customer_churn.csv")
    X = add_features(raw)
    y = (raw["Churn"] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Pick the decision threshold from out-of-fold predictions on TRAINING data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build_pipeline(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    threshold = float(thr[f1s.argmax()])

    # Evaluate once on the held-out test set
    pipe = build_pipeline().fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {"roc_auc": round(roc_auc_score(y_test, proba), 3),
               "recall": round(recall_score(y_test, pred), 3),
               "precision": round(precision_score(y_test, pred), 3),
               "f1": round(f1_score(y_test, pred), 3)}

    # Final model: refit on all data for serving
    final = build_pipeline().fit(X, y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "churn_model.joblib")
    meta = {"model_version": MODEL_VERSION, "algorithm": "LogisticRegression",
            "decision_threshold": round(threshold, 3), "test_metrics": metrics,
            "training_rows": len(X),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
