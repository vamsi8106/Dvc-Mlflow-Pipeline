from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os

from src.config import load_config
from src.logger import get_logger

logger = get_logger("validate_and_promote")

def _load_test_df() -> Tuple[pd.DataFrame, pd.Series]:
    test_path = Path("data/test.csv")
    if not test_path.exists():
        raise FileNotFoundError("data/test.csv not found. Run pipeline (dvc repro) first.")
    df = pd.read_csv(test_path)
    return df.drop(columns=["target"]), df["target"]

def _score(model, X: pd.DataFrame, y: pd.Series) -> dict:
    if model is None:
        return {}
    preds = model.predict(X)
    if hasattr(preds, "ndim") and getattr(preds, "ndim", 1) > 1:
        y_pred = preds.argmax(axis=1)
    else:
        y_pred = np.asarray(preds).reshape(-1)

    probas = None
    try:
        sk = getattr(getattr(model, "_model_impl", None), "sklearn_model", None)
        if sk is not None and hasattr(sk, "predict_proba"):
            probas = sk.predict_proba(X)
    except Exception:
        pass

    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro")
    m = {"accuracy": float(acc), "precision_macro": float(precision),
         "recall_macro": float(recall), "f1_macro": float(f1)}
    if probas is not None:
        try:
            auc = roc_auc_score(y, probas, multi_class="ovr", average="macro")
            m["roc_auc_macro"] = float(auc)
        except Exception:
            pass
    return m

def _latest_candidate(client: MlflowClient, name: str):
    vs = client.search_model_versions(f"name='{name}'")
    return max(vs, key=lambda v: int(v.version)) if vs else None

def _production(client: MlflowClient, name: str):
    vs = client.search_model_versions(f"name='{name}'")
    prods = [v for v in vs if v.current_stage == "Production"]
    return max(prods, key=lambda v: int(v.version)) if prods else None

def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    mlflow.set_experiment(cfg.mlflow.experiment)
    
    client = MlflowClient()

    model_name = cfg.mlflow.model_name
    X, y = _load_test_df()

    cand = _latest_candidate(client, model_name)
    if cand is None:
        raise RuntimeError(f"No versions found for '{model_name}'")

    champ = _production(client, model_name)

    cand_uri = f"models:/{model_name}/{cand.version}"
    champ_uri = f"models:/{model_name}/Production" if champ else None

    logger.info(f"Candidate: {model_name} v{cand.version} (stage={cand.current_stage})")
    if champ:
        logger.info(f"Champion:  {model_name} v{champ.version} (Production)")
    else:
        logger.info("No current champion in Production.")

    cand_model = mlflow.pyfunc.load_model(cand_uri)
    champ_model = mlflow.pyfunc.load_model(champ_uri) if champ_uri else None

    cand_m = _score(cand_model, X, y)
    champ_m = _score(champ_model, X, y) if champ_model else None

    logger.info(f"Candidate metrics: {cand_m}")
    if champ_m:
        logger.info(f"Champion metrics:  {champ_m}")

    # Absolute gates
    gates_ok = (cand_m.get("accuracy", 0.0) >= cfg.gates.min_accuracy) and \
               (cand_m.get("f1_macro", 1.0) >= cfg.gates.min_f1)

    # Relative comparison
    better = True
    if champ_m:
        better = (cand_m.get("accuracy", 0.0) >= champ_m.get("accuracy", 0.0)) and \
                 (cand_m.get("f1_macro", 0.0) >= champ_m.get("f1_macro", 0.0))

    decision = "promote" if (gates_ok and better) or (not champ_m and gates_ok) else "reject"

    # Validation run for auditability
    with mlflow.start_run(run_name="validation_compare_champion_challenger"):
        mlflow.set_tags({
            "model_name": model_name,
            "candidate_version": str(cand.version),
            "champion_version": str(champ.version) if champ else "None",
            "decision": decision,
        })
        for k, v in cand_m.items():
            mlflow.log_metric(f"candidate_{k}", v)
        if champ_m:
            for k, v in champ_m.items():
                mlflow.log_metric(f"champion_{k}", v)

    # --- New guard: avoid transitioning if candidate is already the Production champion
    if champ and int(cand.version) == int(champ.version):
        logger.info("Candidate version is already the current Production champion; skipping transition.")
        return

    if decision == "promote":
        # Promote candidate → Staging → Production (archive old Production)
        client.transition_model_version_stage(
            name=model_name, version=cand.version, stage="Staging", archive_existing_versions=False
        )
        client.transition_model_version_stage(
            name=model_name, version=cand.version, stage="Production", archive_existing_versions=True
        )
        logger.info(f"✅ Promoted {model_name} v{cand.version} → Production")
    else:
        # Leave stage as 'None'; tag it as rejected for clarity
        client.set_model_version_tag(name=model_name, version=cand.version, key="decision", value="rejected")
        reason_bits = []
        if not gates_ok:
            reason_bits.append("failed_gates")
        if champ_m and not better:
            reason_bits.append("not_better_than_champion")
        client.set_model_version_tag(
            name=model_name, version=cand.version, key="rejected_reason",
            value=",".join(reason_bits) or "unspecified"
        )
        logger.info("❌ Candidate failed gates and/or did not beat champion — left in stage 'None' with tags.")
        logger.info("   (Optional) Archive losers instead of tagging if you prefer.")
        # Example to archive instead:
        # client.transition_model_version_stage(name=model_name, version=cand.version, stage="Archived")

if __name__ == "__main__":
    main()
