import json
from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import mlflow
from src.config import load_config
from src.logger import get_logger

logger = get_logger("evaluate")

def main():
    cfg = load_config()

    test_path = Path("data/test.csv")
    model_path = Path("artifacts/model.joblib")
    if not test_path.exists() or not model_path.exists():
        raise FileNotFoundError("Missing test data or model. Run train/evaluate via dvc repro.")

    df = pd.read_csv(test_path)
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    clf = load(model_path)
    probas = clf.predict_proba(X_test)
    y_pred = probas.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }

    try:
        auc = roc_auc_score(y_test, probas, multi_class="ovr", average="macro")
        metrics["roc_auc_macro"] = float(auc)
    except Exception:
        pass

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name="evaluate_model"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    logger.info(f"Evaluation done. Metrics: {metrics}")

if __name__ == "__main__":
    main()
