# Development Process (SDLC)

How this project moved through each stage of the software development life cycle.

## 1. Requirements

**Problem:** A retention team needs churn predictions inside their own tools (CRM, dashboards), not in a notebook.

| ID | Requirement | Type |
|----|-------------|------|
| R1 | Score one customer and return a churn probability, yes/no prediction and risk tier | Functional |
| R2 | Score up to 500 customers in one request | Functional |
| R3 | Explain *why* a customer is high risk (top factors) | Functional |
| R4 | Reject invalid or contradictory input with a clear error message | Functional |
| R5 | Expose model version and accuracy so users know what they're getting | Functional |
| R6 | Protect prediction endpoints with an API key | Security |
| R7 | Respond in under 100 ms for a single customer | Performance |
| R8 | Every request is logged with an ID and duration so problems can be traced | Operability |
| R9 | Runs the same on any machine (containerized) | Deployment |

## 2. Design

```
Client ──HTTP/JSON──▶ FastAPI (app/main.py)
                        │  middleware: request ID + timing log        (R8)
                        │  auth: X-API-Key header                     (R6)
                        │  validation: Pydantic schemas               (R4)
                        ▼
                    ChurnModel (app/model.py)
                        │  features: app/features.py  ◀── shared with train.py
                        │  scikit-learn pipeline (scaler + one-hot + logistic regression)
                        ▼
                    Prediction + top risk factors                     (R1, R3)
```

**Key decisions**

- **FastAPI**: automatic input validation, auto-generated docs at `/docs`, and fast async serving.
- **One shared feature module.** Training and serving both call `add_features()`, so the API can never build features differently from how the model was trained (avoids training/serving skew).
- **Logistic regression.** It matched Random Forest and Gradient Boosting on accuracy (see the companion analysis project), and its coefficients make per-customer explanations (R3) simple and exact.
- **Model loaded once at startup**, not per request, to meet R7.
- **Validation beyond types.** Business rules (e.g. a customer with no internet can't have online backup) are checked in the schema, so bad data never reaches the model.

## 3. Implementation

- Python 3.11+, type hints throughout
- `ruff` linting with bug-catching rules enabled (`B`, `RUF`)
- Model metadata (version, threshold, metrics, training date) is saved next to the model file

## 4. Testing

30 automated tests, 97% code coverage (`pytest --cov=app`).

| Test type | Examples |
|-----------|----------|
| Unit | Feature engineering: add-on counts, autopay flag, divide-by-zero on new customers, input not mutated |
| API / integration | Every endpoint, happy paths, batch ordering, request-ID header |
| Validation | 6 invalid-value cases, missing fields, contradictory internet/phone fields, batch size limits |
| Security | Missing key -> 401, wrong key -> 401, correct key -> 200, health check stays public |
| Model behavior | Longer contract lowers risk; predictions are deterministic; deployed model must have ROC-AUC > 0.8 |

## 5. Deployment

- **CI (GitHub Actions):** on every push and pull request: lint -> train -> test on Python 3.11 and 3.12 (fails under 90% coverage) -> build Docker image -> start the container and check `/health`.
- **Docker:** slim Python image, model trained at build time so it matches the image's library versions, runs as a non-root user, built-in health check.

## 6. Maintenance

- **Versioning:** semantic versioning (`MAJOR.MINOR.PATCH`); changes recorded in [CHANGELOG.md](../CHANGELOG.md).
- **Branching:** `main` is always deployable. Work happens on feature branches (`feature/batch-endpoint`, `fix/factor-labels`) merged through pull requests that must pass CI.
- **Retraining:** rerun `train.py` on fresh data, bump the model version, and check `/model/info` metrics before deploying.
- **Monitoring:** request logs include duration and status, so slow or failing requests can be found by request ID.

## Definition of Done

- [ ] Requirement has at least one test
- [ ] `ruff check .` passes
- [ ] All tests pass, coverage ≥ 90%
- [ ] README / docs updated
- [ ] CHANGELOG entry added
