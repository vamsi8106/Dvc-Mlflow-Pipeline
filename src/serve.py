from typing import List, Dict, Any
import time, uuid
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from src.config import load_config
from src.logger import get_logger

logger = get_logger("serve")
cfg = load_config()
app = FastAPI(title="Inference Service", version="1.1.0")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[Any]

@app.middleware("http")
async def request_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.time()
    resp = await call_next(request)
    ms = (time.time() - start) * 1000.0
    logger.info(f"rid={rid} {request.method} {request.url.path} status={resp.status_code} ms={ms:.2f}")
    resp.headers["x-request-id"] = rid
    return resp

@app.get("/healthz", status_code=status.HTTP_200_OK)
def healthz():
    return {"status": "ok"}

def load_production_model():
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    uri = f"models:/{cfg.mlflow.model_name}@{cfg.mlflow.production_alias}"
    logger.info(f"Loading model: {uri}")
    return mlflow.pyfunc.load_model(uri)

MODEL = None
try:
    MODEL = load_production_model()
except Exception as e:
    logger.error(f"Initial model load failed: {e}")

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(req: PredictRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        df = pd.DataFrame.from_records(req.records)
        preds = MODEL.predict(df)
        preds = [p.item() if hasattr(p, "item") else p for p in preds]
        return PredictResponse(predictions=preds)
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/reload", status_code=status.HTTP_202_ACCEPTED)
def reload_model():
    global MODEL
    MODEL = load_production_model()
    return {"status": "reloaded"}
