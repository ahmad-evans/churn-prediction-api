"""Loads the trained model and turns customer records into predictions."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.features import add_features
from app.schemas import Customer, Prediction

MODELS = Path(__file__).resolve().parents[1] / "models"


class ChurnModel:
    def __init__(self, model_dir: Path = MODELS):
        self.pipeline = joblib.load(model_dir / "churn_model.joblib")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.threshold = self.metadata["decision_threshold"]
        prep = self.pipeline.named_steps["prep"]
        self.feature_names = prep.get_feature_names_out()
        self.coefs = self.pipeline.named_steps["model"].coef_[0]

    def risk_tier(self, p: float) -> str:
        if p >= 0.5:
            return "High"
        return "Medium" if p >= self.threshold else "Low"

    # Only report a factor if it moves the log-odds by at least this much (filters out noise)
    MIN_CONTRIBUTION = 0.25

    def top_factors(self, row: np.ndarray, n: int = 3) -> list[str]:
        """
        For logistic regression, each feature pushes the score up or down by
        (coefficient x feature value). Return up to n features pushing risk UP the most.
        """
        contrib = row * self.coefs
        order = np.argsort(contrib)[::-1]
        factors = []
        for i in order[:n]:
            if contrib[i] < self.MIN_CONTRIBUTION:
                break
            kind, name = self.feature_names[i].split("__", 1)  # e.g. "cat", "Contract_Month-to-month"
            if kind == "num":
                # Scaled numeric value: above 0 = higher than the average customer
                factors.append(f"{name} is {'high' if row[i] > 0 else 'low'}")
            else:
                factors.append(name.replace("_", " = ", 1))
        return factors

    def predict(self, customers: list[Customer]) -> list[Prediction]:
        raw = pd.DataFrame([c.model_dump(exclude={"customer_id"}) for c in customers])
        X = add_features(raw)
        proba = self.pipeline.predict_proba(X)[:, 1]
        transformed = self.pipeline.named_steps["prep"].transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return [
            Prediction(customer_id=c.customer_id,
                       churn_probability=round(float(p), 3),
                       will_churn=bool(p >= self.threshold),
                       risk_tier=self.risk_tier(float(p)),
                       top_risk_factors=self.top_factors(row))
            for c, p, row in zip(customers, proba, transformed, strict=True)
        ]
