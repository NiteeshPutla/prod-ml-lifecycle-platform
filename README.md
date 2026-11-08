# Production ML Lifecycle Platform

> Work in progress: the initial publication has 14 passing and 20 failing tests. See [test status](docs/TEST_STATUS.md) for known issues.

Full ML lifecycle from feature engineering → training → deployment → monitoring → A/B iteration for Ranking, Recommendation, Search & Ads models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION ML LIFECYCLE PLATFORM                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ FEATURE  │───▶│ TRAINING │───▶│  SERVING │───▶│   A/B    │              │
│  │ ENGINEER │    │ PIPELINE │    │  LAYER   │    │ EXPERIM. │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │              │              │              │                         │
│       ▼              ▼              ▼              ▼                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  OFFLINE │    │  MODEL   │    │ ONLINE   │    │  REAL-   │              │
│  │ FEATURE  │    │ REGISTRY │    │ FEATURE  │    │  TIME    │              │
│  │  STORE   │    │ (MLflow) │    │  STORE   │    │ METRICS  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MONITORING & DRIFT DETECTION                       │   │
│  │  Data Drift │ Concept Drift │ Feature Drift │ Prediction Drift      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    PRODUCTION TRAFFIC SIMULATOR                       │   │
│  │  Realistic user sessions, click models, conversion funnels           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

| Module | Purpose |
|--------|---------|
| `core/` | Shared schemas, configs, utilities |
| `features/` | Offline & online feature store (SQLite + Redis) |
| `training/` | Training pipelines, model registry, feature lineage |
| `serving/` | Low-latency model serving with TensorRT/ONNX support |
| `experimentation/` | Real-time A/B testing framework with statistical rigor |
| `monitoring/` | Data/concept/feature/prediction drift detection |
| `simulator/` | Production traffic simulator for shadow testing |
| `cli/` | Typer CLI for end-to-end workflows |
| `apps/dashboard/` | Next.js real-time monitoring dashboard |

## Quickstart

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run feature engineering pipeline
mlops features build --config configs/features.yaml

# Train a ranking model
mlops training run --model-type ranker --experiment ranking_v1

# Start serving
mlops serving start --model ranking_v1 --port 8080

# Run A/B experiment
mlops experiment create --config configs/experiment_ranking.yaml

# Start monitoring
mlops monitoring start --model ranking_v1
```

## Learning Path

1. **Feature Engineering** - Build features with proper lineage & freshness
2. **Training** - Train ranking/recommendation/search/ads models with tracking
3. **Serving** - Deploy with low latency, batch + online inference
4. **A/B Testing** - Run experiments with statistical validity
5. **Monitoring** - Detect drift, alert, auto-retrain triggers
6. **Shadow Testing** - Validate with production traffic before launch