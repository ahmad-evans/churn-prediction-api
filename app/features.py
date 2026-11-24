"""
Feature engineering shared by training (train.py) and serving (app/model.py).

Keeping this in ONE place means the API builds features exactly the same way
the model saw them during training (avoids "training/serving skew").
"""
import numpy as np
import pandas as pd

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddons",
           "AvgMonthlySpend", "SeniorCitizen", "AutoPay", "NewCustomer"]

CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered columns added."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["NumAddons"] = (out[ADDONS] == "Yes").sum(axis=1)
    out["AutoPay"] = out["PaymentMethod"].str.contains("automatic", regex=False).astype(int)
    out["AvgMonthlySpend"] = np.where(out["tenure"] > 0,
                                      out["TotalCharges"] / out["tenure"].clip(lower=1),
                                      out["MonthlyCharges"])
    out["NewCustomer"] = (out["tenure"] < 12).astype(int)
    return out[NUMERIC + CATEGORICAL]
