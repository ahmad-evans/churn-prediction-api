"""Request and response shapes. Pydantic rejects bad input with a clear 422 error."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 256.5}})

    customer_id: str | None = Field(None, max_length=50, description="Optional ID, echoed back in the response")
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0, le=1000)
    TotalCharges: float = Field(ge=0)

    @model_validator(mode="after")
    def check_consistency(self):
        # Business rules the type checks alone can't catch
        addons = [self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection,
                  self.TechSupport, self.StreamingTV, self.StreamingMovies]
        if self.InternetService == "No" and any(a != "No internet service" for a in addons):
            raise ValueError("Customers without internet must have 'No internet service' for all add-ons")
        if self.InternetService != "No" and any(a == "No internet service" for a in addons):
            raise ValueError("'No internet service' is only valid when InternetService is 'No'")
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("Customers without phone service must have MultipleLines = 'No phone service'")
        return self


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float
    will_churn: bool
    risk_tier: Literal["Low", "Medium", "High"]
    top_risk_factors: list[str]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    high_risk_count: int
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_version: str
    algorithm: str
    decision_threshold: float
    test_metrics: dict[str, float]
    training_rows: int
    trained_at: str
