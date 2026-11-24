"""
Churn Prediction REST API.

    uvicorn app.main:app --reload
    Interactive docs: http://localhost:8000/docs
"""
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.model import ChurnModel
from app.schemas import BatchRequest, BatchResponse, Customer, ModelInfo, Prediction

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churn_api")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup instead of on every request
    try:
        state["model"] = ChurnModel()
        log.info("model loaded version=%s", state["model"].metadata["model_version"])
    except FileNotFoundError:
        log.error("model files not found - run `python train.py` first")
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan,
              description="Predicts which telecom customers are likely to cancel, and why.")

# ---------------------------------------------------------------- auth
# If the API_KEY environment variable is set, every /predict call needs a matching X-API-Key header.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("API_KEY")
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key")


def get_model() -> ChurnModel:
    model = state.get("model")
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return model


# ---------------------------------------------------------------- logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    log.info("request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
             request_id, request.method, request.url.path, response.status_code, ms)
    return response


# ---------------------------------------------------------------- endpoints
@app.get("/health", tags=["ops"])
def health():
    """Liveness check used by Docker and load balancers."""
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model/info", response_model=ModelInfo, tags=["model"])
def model_info(model: ChurnModel = Depends(get_model)):
    """Version, decision threshold and test-set metrics of the model being served."""
    return model.metadata


@app.post("/predict", response_model=Prediction, tags=["predict"],
          dependencies=[Depends(require_api_key)])
def predict(customer: Customer, model: ChurnModel = Depends(get_model)):
    """Score one customer."""
    return model.predict([customer])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"],
          dependencies=[Depends(require_api_key)])
def predict_batch(req: BatchRequest, model: ChurnModel = Depends(get_model)):
    """Score up to 500 customers in one call."""
    preds = model.predict(req.customers)
    return BatchResponse(count=len(preds),
                         high_risk_count=sum(p.risk_tier == "High" for p in preds),
                         predictions=preds)
