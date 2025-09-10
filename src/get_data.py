import requests
from pathlib import Path
from src.config import load_config
from src.logger import get_logger

logger = get_logger("get_data")

def main():
    cfg = load_config()
    url = cfg.data.url
    Path("data").mkdir(parents=True, exist_ok=True)
    out_path = Path("data/raw.csv")

    logger.info(f"Downloading dataset from {url} -> {out_path}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"Downloaded {out_path.resolve()} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
