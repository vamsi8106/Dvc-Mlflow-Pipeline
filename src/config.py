import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict
import yaml
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env") if Path(".env").exists() else None)

def _yaml_params() -> Dict[str, Any]:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f) or {}

def _get_env(key: str) -> Optional[str]:
    v = os.getenv(key)
    return v if (v is not None and v != "") else None

@dataclass(frozen=True)
class DataConfig:
    url: str

@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int

@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    max_depth: int

@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment: str
    model_name: str

@dataclass(frozen=True)
class PromotionGates:
    min_f1: float
    min_accuracy: float

@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    mlflow: MLflowConfig
    gates: PromotionGates

def load_config() -> AppConfig:
    p = _yaml_params()

    data_url = _get_env("DATA_URL") or p["data"]["url"]
    test_size = float(_get_env("TEST_SIZE") or p["split"]["test_size"])
    random_state = int(_get_env("RANDOM_STATE") or p["split"]["random_state"])
    n_estimators = int(_get_env("N_ESTIMATORS") or p["model"]["n_estimators"])
    max_depth = int(_get_env("MAX_DEPTH") or p["model"]["max_depth"])

    tracking_uri = _get_env("MLFLOW_TRACKING_URI") or p["mlflow"]["tracking_uri"]
    experiment = p["mlflow"]["experiment"]
    model_name = _get_env("MLFLOW_MODEL_NAME") or p["mlflow"]["model_name"]

    min_f1 = float(_get_env("PROMOTE_MIN_F1") or 0.0)
    min_accuracy = float(_get_env("PROMOTE_MIN_ACCURACY") or 0.0)

    return AppConfig(
        data=DataConfig(url=data_url),
        split=SplitConfig(test_size=test_size, random_state=random_state),
        model=ModelConfig(n_estimators=n_estimators, max_depth=max_depth),
        mlflow=MLflowConfig(tracking_uri=tracking_uri, experiment=experiment, model_name=model_name),
        gates=PromotionGates(min_f1=min_f1, min_accuracy=min_accuracy),
    )
