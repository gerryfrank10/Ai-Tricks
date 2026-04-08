# MLOps & Model Deployment

MLOps applies DevOps principles to machine learning. It covers the full lifecycle from experiment tracking and model versioning to CI/CD pipelines, deployment, monitoring, and drift detection in production.

---

## 📖 **Sections**

- [MLOps Maturity Model](#mlops-maturity-model)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Model Registry & Versioning](#model-registry--versioning)
- [Containerization with Docker](#containerization-with-docker)
- [Serving Models](#serving-models)
- [CI/CD for ML](#cicd-for-ml)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Feature Stores](#feature-stores)

---

## 📈 **MLOps Maturity Model**

| Level | Description | Automation |
|-------|-------------|------------|
| 0 | Manual, Jupyter notebooks | None |
| 1 | ML pipeline automation | Training pipelines |
| 2 | CI/CD pipeline automation | Everything automated |

---

## 🧪 **Experiment Tracking with MLflow**

```bash
pip install mlflow
mlflow ui  # Opens UI at http://localhost:5000
```

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Set experiment
mlflow.set_experiment("customer-churn-prediction")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Track experiment run
with mlflow.start_run(run_name="rf-baseline"):
    # Log parameters
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }
    mlflow.log_metrics(metrics)

    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="churn-predictor",
        signature=mlflow.models.infer_signature(X_train, y_pred),
    )

    # Log artifacts (plots, feature importance, etc.)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(feature_names, model.feature_importances_)
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Comparing Runs Programmatically

```python
import mlflow

client = mlflow.MlflowClient()

# Get all runs for an experiment
experiment = client.get_experiment_by_name("customer-churn-prediction")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=5
)

print("Top 5 runs by F1 Score:")
for run in runs:
    print(f"  Run {run.info.run_id[:8]} | F1: {run.data.metrics.get('f1_score', 0):.4f} | "
          f"n_estimators: {run.data.params.get('n_estimators')}")
```

---

## 📦 **Model Registry & Versioning**

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to staging
client.transition_model_version_stage(
    name="churn-predictor",
    version=3,
    stage="Staging",
    archive_existing_versions=False
)

# Load model from registry
import mlflow.pyfunc

# Load latest production model
model = mlflow.pyfunc.load_model("models:/churn-predictor/Production")
predictions = model.predict(X_test)

# Load specific version
model_v2 = mlflow.pyfunc.load_model("models:/churn-predictor/2")

# Promote to production after validation
client.transition_model_version_stage(
    name="churn-predictor",
    version=3,
    stage="Production",
    archive_existing_versions=True  # Archive current production version
)
```

---

## 🐳 **Containerization with Docker**

### `Dockerfile` for a FastAPI ML Service

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `app/main.py` — FastAPI Prediction Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = mlflow.pyfunc.load_model("models:/churn-predictor/Production")
    logger.info("Model loaded successfully")

class PredictionRequest(BaseModel):
    tenure_months: float
    monthly_charges: float
    total_charges: float
    contract_type: str  # "month-to-month", "one-year", "two-year"
    payment_method: str

class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    confidence: str

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        features = pd.DataFrame([request.dict()])

        # Predict
        probability = float(model.predict(features)[0])
        will_churn = probability > 0.5
        confidence = "high" if abs(probability - 0.5) > 0.3 else "medium" if abs(probability - 0.5) > 0.1 else "low"

        logger.info(f"Prediction: {probability:.3f} (churn={will_churn})")

        return PredictionResponse(
            churn_probability=probability,
            will_churn=will_churn,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Return model performance metrics."""
    return {
        "model_name": "churn-predictor",
        "model_version": "Production",
        "last_evaluated": "2025-01-15",
        "test_accuracy": 0.892,
        "test_f1": 0.871
    }
```

### Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: "3.8"

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root /mlflow/artifacts

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  mlflow-data:
```

---

## 🚀 **Serving Models**

### BentoML (Production-Ready Serving)

```python
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, JSON

# Save model to BentoML
bentoml.sklearn.save_model("churn_predictor", trained_model)

# Create service
svc = bentoml.Service("churn_prediction", runners=[
    bentoml.sklearn.get("churn_predictor:latest").to_runner()
])

churn_runner = bentoml.sklearn.get("churn_predictor:latest").to_runner()

@svc.api(input=NumpyNdarray(), output=JSON())
async def predict(input_data: np.ndarray):
    result = await churn_runner.predict.async_run(input_data)
    return {"predictions": result.tolist()}
```

```bash
# Serve
bentoml serve service:svc --reload

# Build and containerize
bentoml build
bentoml containerize churn_prediction:latest
```

---

## 🔄 **CI/CD for ML**

### GitHub Actions Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths: ["src/**", "data/**", "requirements.txt"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Data validation
        run: python scripts/validate_data.py

  train-and-evaluate:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Train model
        run: python scripts/train.py --experiment-name "ci-${{ github.sha[:8] }}"

      - name: Evaluate model
        run: python scripts/evaluate.py --min-f1 0.85

      - name: Register model if metrics pass
        run: python scripts/register_model.py --stage Staging
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

  deploy:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Build Docker image
        run: docker build -t ml-api:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USER }} --password-stdin
          docker push ml-api:${{ github.sha }}

      - name: Deploy to production
        run: kubectl set image deployment/ml-api ml-api=ml-api:${{ github.sha }}
```

---

## 📡 **Monitoring & Drift Detection**

### Data Drift Detection with Evidently

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric
import pandas as pd

# Reference data (training distribution)
reference_data = pd.read_csv("training_data.csv")

# Current production data
current_data = pd.read_csv("production_data_last_week.csv")

# Generate drift report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    ColumnDriftMetric(column_name="monthly_charges"),
    ColumnDriftMetric(column_name="tenure_months"),
])

report.run(reference_data=reference_data, current_data=current_data)
report.save_html("drift_report.html")

# Check if retraining is needed
report_dict = report.as_dict()
drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]
drift_share = report_dict["metrics"][0]["result"]["drift_share"]

print(f"Drift detected: {drift_detected}")
print(f"Drift share: {drift_share:.2%} of features drifted")

if drift_share > 0.3:
    print("ALERT: Significant drift detected. Retraining recommended.")
```

### Prediction Monitoring

```python
import sqlite3
from datetime import datetime
import json

class PredictionMonitor:
    def __init__(self, db_path: str = "predictions.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_features TEXT,
                prediction REAL,
                actual REAL,
                model_version TEXT
            )
        """)
        self.conn.commit()

    def log_prediction(self, features: dict, prediction: float,
                       model_version: str, actual: float = None):
        self.conn.execute("""
            INSERT INTO predictions (timestamp, input_features, prediction, actual, model_version)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            json.dumps(features),
            prediction,
            actual,
            model_version
        ))
        self.conn.commit()

    def get_accuracy_last_n_days(self, n_days: int = 7) -> float:
        cursor = self.conn.execute(f"""
            SELECT AVG(CASE WHEN (prediction > 0.5) = (actual > 0.5) THEN 1 ELSE 0 END)
            FROM predictions
            WHERE actual IS NOT NULL
              AND timestamp > datetime('now', '-{n_days} days')
        """)
        return cursor.fetchone()[0] or 0.0

monitor = PredictionMonitor()
```

---

## 🏪 **Feature Stores**

### Simple Feature Store with Redis

```python
import redis
import json
from datetime import timedelta

class FeatureStore:
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)

    def write_features(self, entity_id: str, features: dict, ttl_hours: int = 24):
        """Write features for an entity (e.g., user, product)."""
        key = f"features:{entity_id}"
        self.redis.setex(
            key,
            timedelta(hours=ttl_hours),
            json.dumps(features)
        )

    def read_features(self, entity_id: str) -> dict | None:
        """Read features for an entity."""
        key = f"features:{entity_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def read_batch(self, entity_ids: list[str]) -> dict:
        """Read features for multiple entities efficiently."""
        pipeline = self.redis.pipeline()
        for eid in entity_ids:
            pipeline.get(f"features:{eid}")

        results = pipeline.execute()
        return {
            eid: json.loads(r) if r else None
            for eid, r in zip(entity_ids, results)
        }

# Usage
store = FeatureStore()

# Precompute and store features (offline job)
store.write_features("user_123", {
    "avg_session_duration": 8.5,
    "purchase_count_30d": 3,
    "days_since_last_login": 2,
    "lifetime_value": 245.50
})

# Read features at inference time (< 1ms)
features = store.read_features("user_123")
prediction = model.predict([list(features.values())])
```

---

## 💡 **Tips & Tricks**

1. **Log everything**: Parameters, metrics, artifacts, data versions — you can't debug what you didn't log
2. **Reproducibility**: Pin random seeds, library versions, and data snapshots
3. **Shadow deployment**: Run new model in parallel with old one, compare predictions before cutover
4. **Canary releases**: Route 5% of traffic to new model first
5. **Model cards**: Document model limitations, training data, and fairness considerations
6. **Set drift thresholds**: Alert at 10% drift, trigger retraining at 30% drift
7. **Separate concerns**: Training pipeline, serving infrastructure, and monitoring should be independent

---

## 🔗 **Related Topics**

- [Machine Learning Lifecycle](../Machine%20Learning/Generic-Methodology.md)
- [Cloud Computing for AI](../Cloud%20Computing%20For%20AI/)
- [AI Ethics](../AI%20Ethics/)
- [Fine-Tuning LLMs](../Fine-Tuning/README.md)
