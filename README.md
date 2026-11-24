# Churn Prediction API

![CI](https://github.com/ahmad-evans/churn-prediction-api/actions/workflows/ci.yml/badge.svg)

A REST API that predicts which telecom customers are likely to cancel, and explains why, so a retention team can use the model from their own tools.

It serves the model from my [Customer Churn Analytics](https://github.com/ahmad-evans/customer-churn-analytics) project.

## Highlights

- **FastAPI** service with 4 endpoints, auto-generated interactive docs at `/docs`
- **Explainable predictions:** each response lists the top factors pushing that customer's risk up
- **Input validation:** types, ranges and business rules, with clear `422` errors
- **API-key authentication**, request IDs and timing logs on every call
- **30 automated tests, 97% coverage**, linted with ruff
- **CI/CD:** GitHub Actions tests on Python 3.11 and 3.12, then builds and smoke-tests the Docker image
- Documented [development process (SDLC)](docs/SDLC.md) and [changelog](CHANGELOG.md)

## Quick Start

```bash
pip install -r requirements.txt
python train.py                     # trains the model -> models/
uvicorn app.main:app --reload       # open http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t churn-api .
docker run -p 8000:8000 -e API_KEY=my-secret churn-api
```

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Liveness check |
| GET | `/model/info` | No | Model version, decision threshold, test metrics |
| POST | `/predict` | Yes* | Score one customer |
| POST | `/predict/batch` | Yes* | Score up to 500 customers |

\*Only when the `API_KEY` environment variable is set. Send it as an `X-API-Key` header.

### Example

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" -H "X-API-Key: my-secret" \
  -d '{"customer_id":"C-1001","gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No",
       "tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":85.5,"TotalCharges":256.5}'
```

```json
{
  "customer_id": "C-1001",
  "churn_probability": 0.742,
  "will_churn": true,
  "risk_tier": "High",
  "top_risk_factors": ["tenure is low", "Contract = Month-to-month", "InternetService = Fiber optic"]
}
```

Bad input gets a clear error:

```json
// "Contract": "Three year"
{"detail": [{"loc": ["body", "Contract"], "msg": "Input should be 'Month-to-month', 'One year' or 'Two year'"}]}
```

## Model

Logistic regression trained on 7,043 customers ([IBM Telco dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)).

| Metric (held-out test set) | Value |
|---|---|
| ROC-AUC | 0.84 |
| Recall (churners caught) | 75% |
| Precision | 53% |
| Decision threshold | 0.309 (tuned for best F1, instead of the default 0.5) |

Risk tiers: **High** ≥ 50% · **Medium** 30.9-50% · **Low** < 30.9%

## Project Structure

```
app/
  main.py        API routes, auth, logging middleware
  schemas.py     request/response models + validation rules
  model.py       loads the model, makes predictions and explanations
  features.py    feature engineering shared by training and serving
tests/           30 tests: unit, API, validation, security, model behavior
train.py         trains the model and saves metadata
docs/SDLC.md     requirements, design, testing, deployment, maintenance
.github/workflows/ci.yml
Dockerfile
```

## Run the Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest --cov=app
```

---
Ahmad Evans · M.S. Information Systems & Business Analytics, Park University
