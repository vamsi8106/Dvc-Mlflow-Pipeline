from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from src.config import load_config
from src.logger import get_logger

logger = get_logger("serve")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[Any]

cfg = load_config()
app = FastAPI(title="Inference Service", version="1.0.0")

def load_production_model():
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    uri = f"models:/{cfg.mlflow.model_name}/Production"
    logger.info(f"Loading Production model: {uri}")
    return mlflow.pyfunc.load_model(uri)

MODEL = load_production_model()

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame.from_records(req.records)
        preds = MODEL.predict(df)
        preds = [p.item() if hasattr(p, "item") else p for p in preds]
        return PredictResponse(predictions=preds)
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reload")
def reload_model():
    global MODEL
    MODEL = load_production_model()
    return {"status": "reloaded"}
