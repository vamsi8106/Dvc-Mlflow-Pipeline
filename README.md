
# DVC + MLflow (Aliases) + FastAPI

**End-to-end local pipeline** with champion–challenger and **CI/CD via GitHub Actions** (self-hosted runner).  
**Ports:** MLflow **5000**, API **8000**.

---

## ✨ Features
- Reproducible **DVC** pipeline: `get_data → prepare → train → evaluate`
- **MLflow Tracking + Model Registry** using **aliases** (`production`)
- **Validator** promotes the best model and can **/reload** your API automatically
- **FastAPI** serves `models:/<name>@production`
- Single rotating **log** (`logs/app.log`)
- Config via **.env** with fallback to **params.yaml**
- GitHub Actions workflow using **secrets only** and a **self-hosted runner**

---

## 📁 Repository Layout
```
.
├─ data/                      # raw/train/test CSVs
├─ artifacts/                 # local model file for evaluate stage
├─ logs/                      # app.log (rotating)
├─ src/
│  ├─ config.py               # loads .env + params.yaml
│  ├─ logger.py
│  ├─ get_data.py
│  ├─ prepare.py
│  ├─ train.py                # logs run + registers candidate
│  ├─ evaluate.py             # logs metrics
│  ├─ validate_and_promote.py # alias-based promotion + optional API reload
│  └─ serve.py                # FastAPI serving @production alias
├─ params.yaml
├─ dvc.yaml
├─ dvc.lock
├─ requirements.txt
├─ .env.example
└─ .github/workflows/train-validate-promote.yml
```

---

## 🔧 Prerequisites
- Python **3.11+**
- Git + **DVC** (`pip install dvc`)
- **MLflow** (`pip install mlflow`)
- (Prod later) Postgres + S3/GCS/Azure (local uses SQLite + `./mlruns`)

---

## 🚀 Local Setup

### 1) Virtualenv & deps
```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) .env (copy from .env.example)
```dotenv
# Data / Model
DATA_URL=https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
TEST_SIZE=0.2
RANDOM_STATE=42
N_ESTIMATORS=200
MAX_DEPTH=8

# MLflow (LOCAL)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_MODEL_NAME=iris-rf-classifier
MLFLOW_PRODUCTION_ALIAS=production

# Gates
PROMOTE_MIN_ACCURACY=0.92
PROMOTE_MIN_F1=0.90

# API reload hook (LOCAL)
MODEL_API_RELOAD_URL=http://127.0.0.1:8000/reload
MODEL_API_TOKEN=
```
Export:
```bash
set -a; source .env; set +a
```

### 3) Start MLflow (port 5000)
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 --port 5000
```
UI: http://127.0.0.1:5000

### 4) Train pipeline (register candidate)
```bash
dvc repro -f
```

### 5) Validate & maybe promote (alias)
```bash
python -m src.validate_and_promote
```
If first time & gates too strict, set alias once:
```bash
python - <<'PY'
from mlflow.tracking import MlflowClient
c = MlflowClient(tracking_uri="http://127.0.0.1:5000")
name="iris-rf-classifier"; alias="production"
vers=c.search_model_versions(f"name='{name}'")
assert vers, "No versions. Run: dvc repro -f"
v=max(vers, key=lambda x:int(x.version))
c.set_registered_model_alias(name=name, alias=alias, version=v.version)
print(f"production -> {name} v{v.version}")
PY
```

### 6) Serve API (port 8000)
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
curl -s http://127.0.0.1:8000/healthz
```

### 7) Predict (Test API)
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "records": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.6, "petal_width": 0.2}
  ]
}'
```

### 8) Next cycle
```bash
dvc repro -f
python -m src.validate_and_promote
curl -X POST http://127.0.0.1:8000/reload   # if validator didn't call it
```

---

## 🧪 Champion–Challenger
- **Candidate** = latest registered version (vN)
- **Champion** = model under alias **production**
- Promotion in `validate_and_promote.py`:
  - check absolute gates (accuracy, F1) and relative ≥ champion
  - on win → `set_registered_model_alias(name, "production", vN)`
  - else → tag `decision=rejected`
- **Serving** always loads: `models:/<MLFLOW_MODEL_NAME>@<MLFLOW_PRODUCTION_ALIAS>`

---

## 🧹 Reset (Dev Only)
```bash
# stop MLflow first
rm -rf mlflow.db mlruns
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 127.0.0.1 --port 5000
dvc repro -f
python -m src.validate_and_promote
```
If API says “Model not loaded”, set `@production` then `/reload`.

---

## 🟢 GitHub Actions (Self-Hosted Runner + Secrets)

We run CI **on local machine** so `127.0.0.1:5000` (MLflow) and `127.0.0.1:8000` (API) are reachable.

### 1) GitHub **Secrets** (Repo → Settings → *Secrets and variables* → *Actions*)
- `MLFLOW_TRACKING_URI` = `http://127.0.0.1:5000`
- `MLFLOW_MODEL_NAME` = `iris-rf-classifier`
- `MLFLOW_PRODUCTION_ALIAS` = `production`
- `PROMOTE_MIN_ACCURACY` = `0.92`
- `PROMOTE_MIN_F1` = `0.90`
- `MODEL_API_RELOAD_URL` = `http://127.0.0.1:8000/reload`
- `MODEL_API_TOKEN` = *(optional)*

### 2) Self-hosted runner on your laptop
Repo → **Settings → Actions → Runners → New self-hosted runner** (Linux example): Run the commands given over in the system where mlflow and api and running.
```bash
mkdir ~/actions-runner && cd ~/actions-runner
curl -o actions-runner.tar.gz -L https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz
tar xzf actions-runner.tar.gz
./config.sh --url https://github.com/<ORG_OR_USER>/<REPO> --token <RUNNER_TOKEN>
# optional label: mlops
./run.sh
# or as a service:
# ./svc.sh install && ./svc.sh start
```

### 3) Ensure local services are running
```bash
# MLflow (5000)
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 127.0.0.1 --port 5000
# API (8000)
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

### 4) Workflow file
`.github/workflows/train-validate-promote.yml`:
```yaml
name: train-validate-promote
on:
  push: { branches: [ main ] }
  workflow_dispatch:

jobs:
  mlops:
    runs-on: [self-hosted]     # or [self-hosted, mlops] if you labeled it
    timeout-minutes: 30
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_MODEL_NAME: ${{ secrets.MLFLOW_MODEL_NAME }}
      MLFLOW_PRODUCTION_ALIAS: ${{ secrets.MLFLOW_PRODUCTION_ALIAS }}
      PROMOTE_MIN_ACCURACY: ${{ secrets.PROMOTE_MIN_ACCURACY }}
      PROMOTE_MIN_F1: ${{ secrets.PROMOTE_MIN_F1 }}
      MODEL_API_RELOAD_URL: ${{ secrets.MODEL_API_RELOAD_URL }}
      MODEL_API_TOKEN: ${{ secrets.MODEL_API_TOKEN }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }

      - name: Install deps
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Check local services reachable (no secrets printed)
        run: |
          code=$(curl -sS -o /dev/null -w "%{http_code}" "$MLFLOW_TRACKING_URI" || true)
          if [ "$code" = "000" ]; then
            echo "MLflow not reachable at $MLFLOW_TRACKING_URI"; exit 1
          else
            echo "MLflow HTTP: $code"
          fi
          if [ -n "$MODEL_API_RELOAD_URL" ]; then
            base="${MODEL_API_RELOAD_URL%/reload}"
            code=$(curl -sS -o /dev/null -w "%{http_code}" "$base/healthz" || true)
            echo "API /healthz HTTP: ${code:-000}"
          fi

      - name: Reproduce DVC pipeline (force run)
        run: |
          source .venv/bin/activate
          dvc repro -f

      - name: Validate & maybe promote (alias) and auto-/reload
        run: |
          source .venv/bin/activate
          python -m src.validate_and_promote

      - name: Upload logs (optional)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: pipeline-logs
          path: |
            logs
            dvc.lock
            metrics.json
```

---

## 🧰 Troubleshooting
- **“MLflow not reachable” (Actions)** → runner not running, or MLflow not on 5000
- **“Model not loaded” (API)** → no `@production` alias yet; promote or set once, then `/reload`
- **“Not inside a DVC repo” (Actions)** → commit `.dvc/` and `dvc.yaml`; add `working-directory` if using subfolders
- **No versions found** → force fresh candidate: `dvc repro -f`
- **Port mismatch** → keep MLflow **5000** and API **8000** consistent across `.env`, servers, and secrets

---

## 📌 Suggested `.gitignore`
```
.venv/
__pycache__/
*.pyc
.env
mlruns/
mlflow.db
logs/*.log
artifacts/*.joblib
.dvc/cache/
.dvc/tmp/
```

---

## 📦 Production Notes (Later)
- Run MLflow on **PostgreSQL + S3/GCS/Azure** behind HTTPS; keep using **aliases** (`production`)
- Autoscale API, add `/healthz`, request timing, and monitoring
- Add data quality checks (e.g., Great Expectations) and online evaluation


## 📚 References
- DVC Documentation: https://dvc.org/doc
- GitHub Actions Documentation: https://docs.github.com/actions
- MLflow Documentation: https://mlflow.org/docs/latest/index.html

## 📝 License (Apache 2.0)

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
