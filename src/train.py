from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.config import load_config
from src.logger import get_logger

logger = get_logger("train")

def main():
    cfg = load_config()

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    train_path = Path("data/train.csv")
    if not train_path.exists():
        raise FileNotFoundError("data/train.csv not found. Run prepare stage (dvc repro).")

    df = pd.read_csv(train_path)
    X_train = df.drop(columns=["target"])
    y_train = df["target"]

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name="train_random_forest") as run:
        mlflow.log_param("n_estimators", cfg.model.n_estimators)
        mlflow.log_param("max_depth", cfg.model.max_depth)
        mlflow.log_param("random_state", cfg.split.random_state)

        clf = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            random_state=cfg.split.random_state,
        ).fit(X_train, y_train)

        acc_cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()
        mlflow.log_metric("cv_accuracy", float(acc_cv))

        # Save local artifact for evaluate stage
        model_path = Path("artifacts/model.joblib")
        dump(clf, model_path)

        # Register candidate in Model Registry
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=cfg.mlflow.model_name,
            signature=signature,
            input_example=X_train.head(2),
        )

        logger.info(f"Model trained (CV accuracy={acc_cv:.4f}). "
                    f"Run_id={run.info.run_id}; registered under '{cfg.mlflow.model_name}'.")

if __name__ == "__main__":
    main()
