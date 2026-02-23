# 01 — Deployment

Production-ready FastAPI backend for AI/ML model serving with MLOps and AIOps pipelines.

## Structure

```
01-deployment/
├── config.py              # Pydantic Settings (app, model, LLM providers, tracking, monitoring)
├── .env.example           # Environment variable template
├── main.py                # FastAPI app with lifespan, CORS, middleware
├── models/
│   ├── base.py            # Abstract BaseModel contract
│   └── registry.py        # ModelRegistry — auto-loads TorchScript & HF models
├── routers/
│   ├── predict.py         # POST /v1/predict, POST /v1/predict/batch, GET /v1/models
│   ├── health.py          # GET /health, GET /health/ready (k8s probes)
│   └── metrics.py         # GET /metrics — GPU memory, latency percentiles
├── middleware/
│   ├── auth.py            # API key middleware (X-API-Key header)
│   └── logging.py         # Structured access logging with request IDs
├── pipelines/
│   └── mlops_pipeline.py  # MLOpsPipeline + AIOPsPipeline with step engine
└── examples/
    ├── hf_model_deploy.py # Load & query a HuggingFace pipeline directly
    └── run_pipeline.py    # Run full MLOps + AIOps pipelines
```

## Quick Start

```bash
cp .env.example .env
# Edit .env with your keys

uvicorn main:app --reload
```

## API Endpoints

| Method | Path                | Description                                         |
| ------ | ------------------- | --------------------------------------------------- |
| GET    | `/health`           | Liveness — returns uptime, CUDA info, loaded models |
| GET    | `/health/ready`     | Readiness probe for k8s                             |
| GET    | `/metrics`          | GPU memory usage + per-model p50/p95/p99 latency    |
| GET    | `/v1/models`        | List all loaded models                              |
| POST   | `/v1/predict`       | Single inference                                    |
| POST   | `/v1/predict/batch` | Batch inference (max 32)                            |

## Pipelines

### MLOps

```
data_validation → preprocessing → training → evaluation → model_registry → deployment → monitoring_setup
```

Swap each async step function with your real logic (PyTorch Trainer, MLflow, etc.).

### AIOps

```
ingest_live_data → drift_detection → retraining_trigger → retraining → re_deployment → notification
```

Schedule this pipeline with Airflow / Prefect / APScheduler for continuous monitoring.

## Configuration Keys

| Variable              | Description                                    |
| --------------------- | ---------------------------------------------- |
| `API_KEY`             | Shared secret — sent as `X-API-Key` header     |
| `DEFAULT_MODEL`       | Model name used when none specified in request |
| `MODEL_DIR`           | Directory scanned for model weights on startup |
| `MLFLOW_TRACKING_URI` | MLflow server URI for experiment tracking      |
| `HF_TOKEN`            | HuggingFace access token for private models    |
