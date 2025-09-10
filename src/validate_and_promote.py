from pathlib import Path
import os
import pandas as pd
import numpy as np
from typing import Tuple
import requests
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from src.config import load_config
from src.logger import get_logger

logger = get_logger("validate_and_promote")

def _load_test_df() -> Tuple[pd.DataFrame, pd.Series]:
    p = Path("data/test.csv")
    if not p.exists():
        raise FileNotFoundError("data/test.csv not found. Run pipeline (dvc repro) first.")
    df = pd.read_csv(p)
    return df.drop(columns=["target"]), df["target"]

def _score(model, X: pd.DataFrame, y: pd.Series) -> dict:
    if model is None:
        return {}
    preds = model.predict(X)
    y_pred = preds.argmax(axis=1) if hasattr(preds, "ndim") and preds.ndim > 1 else np.asarray(preds).reshape(-1)
    probas = None
    try:
        sk = getattr(getattr(model, "_model_impl", None), "sklearn_model", None)
        if sk is not None and hasattr(sk, "predict_proba"):
            probas = sk.predict_proba(X)
    except Exception:
        pass
    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro")
    m = {"accuracy": float(acc), "precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1)}
    if probas is not None:
        try:
            auc = roc_auc_score(y, probas, multi_class="ovr", average="macro")
            m["roc_auc_macro"] = float(auc)
        except Exception:
            pass
    return m

def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    client = MlflowClient()

    model_name = cfg.mlflow.model_name
    alias = cfg.mlflow.production_alias

    X, y = _load_test_df()

    # Latest candidate (numerically max version)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for '{model_name}'")
    cand = max(versions, key=lambda v: int(v.version))

    # Current champion via alias (preferred). Fallback to stage if alias unavailable.
    champ = None
    try:
        champ = client.get_model_version_by_alias(name=model_name, alias=alias)
    except Exception:
        # fallback for older servers
        prods = [v for v in versions if v.current_stage == "Production"]
        champ = max(prods, key=lambda v: int(v.version)) if prods else None

    cand_uri = f"models:/{model_name}/{cand.version}"
    champ_uri = f"models:/{model_name}@{alias}" if champ else None

    logger.info(f"Candidate: {model_name} v{cand.version} (current_stage={cand.current_stage})")
    if champ:
        logger.info(f"Champion via alias '{alias}': {model_name} v{champ.version}")
    else:
        logger.info(f"No current champion for alias '{alias}'.")

    cand_model = mlflow.pyfunc.load_model(cand_uri)
    champ_model = mlflow.pyfunc.load_model(champ_uri) if champ_uri else None

    cand_m = _score(cand_model, X, y)
    champ_m = _score(champ_model, X, y) if champ_model else None

    logger.info(f"Candidate metrics: {cand_m}")
    if champ_m:
        logger.info(f"Champion metrics:  {champ_m}")

    gates_ok = (cand_m.get("accuracy", 0.0) >= cfg.gates.min_accuracy) and \
               (cand_m.get("f1_macro", 1.0) >= cfg.gates.min_f1)

    better = True
    if champ_m:
        better = (cand_m.get("accuracy", 0.0) >= champ_m.get("accuracy", 0.0)) and \
                 (cand_m.get("f1_macro", 0.0) >= champ_m.get("f1_macro", 0.0))

    decision = "promote" if (gates_ok and better) or (not champ_m and gates_ok) else "reject"

    # Audit run
    with mlflow.start_run(run_name="validation_compare_champion_challenger"):
        mlflow.set_tags({
            "model_name": model_name,
            "candidate_version": str(cand.version),
            "champion_version": str(getattr(champ, "version", "None")),
            "decision": decision,
        })
        for k, v in cand_m.items(): mlflow.log_metric(f"candidate_{k}", v)
        if champ_m:
            for k, v in champ_m.items(): mlflow.log_metric(f"champion_{k}", v)

    # If candidate is already the one on alias, skip
    if champ and int(cand.version) == int(champ.version):
        logger.info("Candidate is already the production champion; skipping.")
        return

    if decision == "promote":
        # Set alias -> candidate
        client.set_registered_model_alias(name=model_name, alias=alias, version=cand.version)
        logger.info(f"✅ Set alias '{alias}' -> {model_name} v{cand.version}")

        # Optional: tag as winner
        client.set_model_version_tag(name=model_name, version=cand.version, key="decision", value="promoted")

        # Auto-reload inference if configured
        if cfg.api.reload_url:
            headers = {"Content-Type": "application/json"}
            if cfg.api.reload_token:
                headers["Authorization"] = f"Bearer {cfg.api.reload_token}"
            try:
                r = requests.post(cfg.api.reload_url, headers=headers, timeout=5)
                logger.info(f"Reload request -> {cfg.api.reload_url} status={r.status_code}")
            except Exception as e:
                logger.error(f"Reload request failed: {e}")
    else:
        # Tag as rejected (leave stage None)
        client.set_model_version_tag(name=model_name, version=cand.version, key="decision", value="rejected")
        reason_bits = []
        if not gates_ok: reason_bits.append("failed_gates")
        if champ_m and not better: reason_bits.append("not_better_than_champion")
        client.set_model_version_tag(name=model_name, version=cand.version, key="rejected_reason",
                                     value=",".join(reason_bits) or "unspecified")
        logger.info("❌ Candidate rejected; production alias unchanged.")

if __name__ == "__main__":
    main()
