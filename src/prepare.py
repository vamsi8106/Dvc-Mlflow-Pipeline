from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import load_config
from src.logger import get_logger

logger = get_logger("prepare")

def main():
    cfg = load_config()
    Path("data").mkdir(parents=True, exist_ok=True)

    raw_path = Path("data/raw.csv")
    if not raw_path.exists():
        raise FileNotFoundError("data/raw.csv not found. Run get_data first (dvc repro).")

    df = pd.read_csv(raw_path)

    # Normalize target column -> 'target' (Iris has 'species')
    if "species" in df.columns:
        df = df.rename(columns={"species": "target"})
    if df["target"].dtype == object:
        df["target"] = df["target"].astype("category").cat.codes

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=y
    )

    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)


    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    logger.info(f"Prepared train={train.shape}, test={test.shape}")

if __name__ == "__main__":
    main()
