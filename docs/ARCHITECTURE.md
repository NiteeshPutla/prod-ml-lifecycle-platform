# Production ML Lifecycle Platform - Architecture

## Overview

This platform provides a complete end-to-end ML lifecycle management system covering:
- Feature Engineering (Offline/Online Feature Stores)
- Model Training (XGBoost with MLflow tracking)
- Model Serving (FastAPI with batching/caching)
- A/B Testing (Statistical rigor with Frequentist/Bayesian/Sequential analysis)
- Drift Detection (PSI, KS, Chi2, JS, Wasserstein, KL, Hellinger, Anderson-Darling)
- Alert Management (Slack, Email, PagerDuty, Webhook)
- Traffic Simulation (7 behavior profiles, shadow testing, load testing)

## Directory Structure

```
prod-ml-lifecycle-platform/
├── core/                    # Core schemas and configuration
│   ├── schemas.py          # All Pydantic v2 models
│   └── config.py           # Configuration management
├── features/               # Feature engineering
│   ├── offline_store.py    # SQLite-based offline feature store
│   ├── online_store.py     # Redis/Mock online feature store
│   ├── registry.py         # Feature registry with lineage
│   ├── transformers.py     # Feature transformers (sklearn-compatible)
│   └── lineage.py          # Lineage tracking
├── training/               # Model training
│   ├── pipeline.py         # Training pipeline with MLflow
│   ├── registry.py         # Model registry with versioning
│   ├── tracker.py          # Experiment tracking
│   └── models.py           # 7 model types (Ranking, Recommendation, etc.)
├── serving/                # Model serving
│   └── __init__.py         # FastAPI server, batching, caching
├── experimentation/        # A/B testing
│   ├── framework.py        # Experiment framework, traffic allocation
│   ├── analysis.py         # Statistical analysis (frequentist/bayesian)
│   ├── sequential.py       # Sequential testing (Pocock, OBF, Lan-DeMets, SPRT)
│   └── guardrails.py       # Guardrail monitoring
├── monitoring/             # Monitoring & alerting
│   ├── drift.py            # Data/Concept/Feature/Prediction drift
│   ├── alerts.py           # Multi-channel alert manager
│   ├── metrics.py          # Metrics collection & aggregation
│   └── health.py           # Health checks
├── simulator/              # Traffic simulation
│   └── __init__.py         # 7 behavior profiles, click/conversion models
├── cli/                    # Typer CLI
│   └── main.py             # All subcommands
├── apps/
│   └── dashboard/          # Dashboard (placeholder)
├── tests/                  # Tests (placeholder)
└── scripts/                # Utility scripts (placeholder)
```

## Component Details

### 1. Feature Engineering

#### Offline Feature Store (`features/offline_store.py`)
- SQLite-based storage for feature tables
- Materialization from raw data
- Statistics computation (mean, std, min, max, null_rate, unique_count)
- Point-in-time correct queries
- Feature ingestion API

#### Online Feature Store (`features/online_store.py`)
- Redis backend for low-latency serving
- `MockOnlineFeatureStore` for testing without Redis
- Batch get/set operations
- TTL support

#### Feature Registry (`features/registry.py`)
- Feature specifications with metadata
- Feature groups (user, item, interaction, context)
- Dependency tracking
- Lineage integration
- Search and filtering

#### Transformers (`features/transformers.py`)
- `NumericalTransformer`: StandardScaler, MinMaxScaler, RobustScaler, Log, BoxCox, Quantile
- `CategoricalTransformer`: OneHot, Label, Target, Frequency, Hashing, Embedding
- `TextTransformer`: TF-IDF, CountVectorizer, HashingVectorizer
- `EmbeddingTransformer`: Learned embeddings for high-cardinality categorical
- `CrossFeatureTransformer`: Polynomial, Interaction, Pairwise cross features
- `FeaturePipeline`: ColumnTransformer-compatible pipeline

#### Lineage Tracker (`features/lineage.py`)
- Node types: DataSource, Feature, Transformation, Model, Experiment
- Edge types: DERIVES_FROM, DEPENDS_ON, USED_BY, TRIGGERS
- Graph visualization (GraphViz)
- Impact analysis (upstream/downstream)

### 2. Model Training

#### Training Pipeline (`training/pipeline.py`)
- Data preparation from feature store
- Feature engineering with transformers
- XGBoost training with early stopping
- MLflow experiment tracking
- Model artifact logging

#### Model Registry (`training/registry.py`)
- Model versioning
- Stages: None, Staging, Production, Archived
- Deployments with rollback
- Model metadata and metrics

#### Experiment Tracker (`training/tracker.py`)
- MLflow integration
- Run management
- Artifact logging
- Metric/parameter tracking

#### Model Types (`training/models.py`)
1. `RankingModel` - Learning to Rank (LambdaMART)
2. `RecommendationModel` - Collaborative/Content-based filtering
3. `SearchRerankerModel` - Search result reranking
4. `CTRPredictorModel` - Click-through rate prediction
5. `CVRPredictorModel` - Conversion rate prediction
6. `AdRankerModel` - Ad ranking with pacing/budget
7. `PersonalizationModel` - User personalization

### 3. Model Serving

#### Model Server (`serving/__init__.py`)
- FastAPI-based REST API
- `BatchProcessor`: Async request batching with configurable batch size/wait time
- `PredictionCache`: LRU cache with TTL
- `ModelLoader`: Model loading with version management
- Health endpoints
- Metrics endpoint

### 4. Experimentation

#### Framework (`experimentation/framework.py`)
- `ABTestFramework`: Experiment lifecycle management
- `TrafficAllocator`: Deterministic consistent hashing for variant assignment
- Event recording: impressions, clicks, conversions, custom events
- Sequential analysis integration

#### Analysis (`experimentation/analysis.py`)
- Frequentist: t-test, confidence intervals, power calculation
- Bayesian: Beta-Binomial model for binary metrics
- Sample size calculator
- Sequential analysis (Pocock, O'Brien-Fleming)

#### Sequential Testing (`experimentation/sequential.py`)
- `SequentialTester`: Group sequential designs
  - Pocock boundaries (constant alpha)
  - O'Brien-Fleming boundaries (conservative early)
  - Lan-DeMets alpha spending (flexible)
- `SPRTTester`: Sequential Probability Ratio Test

#### Guardrails (`experimentation/guardrails.py`)
- Business metric thresholds
- Automatic violation detection
- Alert integration

### 5. Monitoring

#### Drift Detection (`monitoring/drift.py`)
- `DataDriftDetector`: Feature distribution drift
  - PSI, KS, Chi2, Jensen-Shannon, Wasserstein, KL, Hellinger, Anderson-Darling
- `ConceptDriftDetector`: Target distribution shift
  - ADWIN, DDM, EDDM, KS on residuals
- `FeatureDriftDetector`: Per-feature importance-weighted drift
- `PredictionDriftDetector`: Prediction distribution drift

#### Alert Manager (`monitoring/alerts.py`)
- Channels: Slack, Email, PagerDuty, Webhook, Log
- Severity levels: Info, Warning, Critical, Emergency
- Alert routing by severity
- Cooldown periods
- Retry logic with exponential backoff

#### Metrics (`monitoring/metrics.py`)
- `MetricsCollector`: Prometheus-compatible metrics
- `MetricAggregator`: Time-windowed aggregations
- `ModelMetricsCollector`: Latency, throughput, error rates
- `SystemMetricsCollector`: CPU, memory, disk, network

#### Health Checks (`monitoring/health.py`)
- HTTP/TCP/Custom checks
- Composite health status
- Dependency tracking
- Readiness/liveness probes

### 6. Traffic Simulation

#### Simulator (`simulator/__init__.py`)
- 7 behavior profiles:
  - Casual Browser
  - Intentional Buyer
  - Bargain Hunter
  - Brand Loyalist
  - Impulse Buyer
  - Researcher
  - Window Shopper
- `ClickModel`: Position-biased click simulation
- `ConversionModel`: Price/category-based conversion
- `ShadowTestRunner`: Shadow testing against production
- `run_load_test`: Concurrent load testing

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │────▶│  Offline    │────▶│  Training   │
│  Ingestion  │     │  Feature    │     │  Pipeline   │
└─────────────┘     │  Store      │     └──────┬──────┘
                    └──────┬──────┘            │
                           │                   ▼
                    ┌──────▼──────┐     ┌─────────────┐
                    │  Online     │◀───▶│   Model     │
                    │  Feature    │     │  Registry   │
                    │  Store      │     └──────┬──────┘
                    └──────┬──────┘            │
                           │                   ▼
                    ┌──────▼──────┐     ┌─────────────┐
                    │  Model      │────▶│ Experiment  │
                    │  Serving    │     │  Framework  │
                    └──────┬──────┘     └──────┬──────┘
                           │                   │
                    ┌──────▼──────┐     ┌──────▼──────┐
                    │  Monitoring │     │   Simulator │
                    │  (Drift/    │     │  (Shadow/   │
                    │   Alerts)   │     │  Load Test) │
                    └─────────────┘     └─────────────┘
```

## CLI Commands

```bash
# Initialize project
prod-ml init --name my-project

# Feature store
prod-ml features build --sample
prod-ml features list
prod-ml features materialize
prod-ml features stats --name user_age

# Training
prod-ml training run --config configs/training.yaml

# Model registry
prod-ml training register --model-path model.json --name my_model --type ranker
prod-ml training list-models
prod-ml training deploy --model-id <id> --version <v> --stage production

# Serving
prod-ml serving start --model-path model.json --port 8000

# Experimentation
prod-ml experiment create --name test --control control --treatment treatment --metric ctr
prod-ml experiment start --experiment-id <id>
prod-ml experiment assign --experiment-id <id> --user-id user_123
prod-ml experiment record --experiment-id <id> --user-id user_123 --event click
prod-ml experiment analyze --experiment-id <id>

# Monitoring
prod-ml monitoring drift --baseline data/baseline.parquet --current data/current.parquet
prod-ml monitoring alert --message "High drift detected" --severity warning

# Simulation
prod-ml simulator run --users 1000 --duration 60
prod-ml simulator shadow --model-path model.json --traffic-fraction 0.1
prod-ml simulator load-test --model-path model.json --rps 100 --duration 60
```

## Configuration

All configuration via YAML files in `configs/`:
- `training.yaml`: Training parameters
- `serving.yaml`: Serving configuration
- `experiment.yaml`: Experiment defaults
- `monitoring.yaml`: Alert channels, thresholds
- `simulator.yaml`: Behavior profiles, traffic patterns

## Requirements

- Python 3.10+
- XGBoost (with libomp on macOS)
- MLflow
- Redis (optional, has mock)
- scikit-learn
- pandas, numpy
- FastAPI, uvicorn
- pyyaml, typer
- scipy, prometheus-client

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Initialize project
prod-ml init --name my-project
cd my-project

# 3. Build features with sample data
prod-ml features build --sample

# 4. Train a model
prod-ml training run --config configs/training.yaml

# 5. Start experiment
prod-ml experiment create --name my_test --control control --treatment treatment --metric ctr
prod-ml experiment start --experiment-id <id>

# 6. Run traffic simulation
prod-ml simulator run --users 100 --duration 30

# 7. Check drift
prod-ml monitoring drift --baseline data/train.parquet --current data/test.parquet
```