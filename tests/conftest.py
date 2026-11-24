import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app

HIGH_RISK = {
    "customer_id": "TEST-HIGH", "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0, "TotalCharges": 160.0,
}

LOW_RISK = {
    "customer_id": "TEST-LOW", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "Yes", "tenure": 65, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 70.0, "TotalCharges": 4550.0,
}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # "with" runs the startup code that loads the model
        yield c


@pytest.fixture
def high_risk():
    return copy.deepcopy(HIGH_RISK)


@pytest.fixture
def low_risk():
    return copy.deepcopy(LOW_RISK)
