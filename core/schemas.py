"""Core Pydantic schemas for the ML lifecycle platform."""

from enum import Enum
from typing import Any, Optional, Dict, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class ModelType(str, Enum):
    RANKER = "ranker"
    RECOMMENDER = "recommender"
    SEARCH_RERANKER = "search_reranker"
    CTR_PREDICTOR = "ctr_predictor"
    CVR_PREDICTOR = "cvr_predictor"
    AD_RANKER = "ad_ranker"
    PERSONALIZER = "personalizer"


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    TEXT = "text"
    SEQUENCE = "sequence"
    MULTI_HOT = "multi_hot"


class FeatureSource(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    REALTIME = "realtime"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServingStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    STARTING = "starting"
    STOPPING = "stopping"


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class DriftType(str, Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeatureSpec(BaseModel):
    """Specification for a single feature."""
    model_config = ConfigDict(use_enum_values=True)

    name: str
    feature_type: FeatureType
    source: FeatureSource
    description: str = ""
    default_value: Any = None
    embedding_dim: Optional[int] = None
    vocabulary_size: Optional[int] = None
    is_key: bool = False
    is_target: bool = False
    freshness_seconds: Optional[int] = None
    transformation: Optional[str] = None  # e.g., "log", "sqrt", "normalize", "bucketize"
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelSpec(BaseModel):
    """Model specification with metadata."""
    model_config = ConfigDict(use_enum_values=True)

    model_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    model_type: ModelType
    version: str = "1.0.0"
    description: str = ""
    framework: str = "xgboost"  # xgboost, pytorch, sklearn, onnx
    features: List[FeatureSpec] = Field(default_factory=list)
    target_feature: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    trained_at: Optional[datetime] = None
    trained_by: Optional[str] = None
    git_commit: Optional[str] = None
    lineage: Dict[str, Any] = Field(default_factory=dict)  # feature versions, data versions


class ExperimentSpec(BaseModel):
    """A/B experiment specification."""
    model_config = ConfigDict(use_enum_values=True)

    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    hypothesis: str = ""
    control_model: ModelSpec
    treatment_models: List[ModelSpec] = Field(default_factory=list)
    traffic_split: Dict[str, float] = Field(default_factory=dict)  # model_id -> percentage
    targeting_rules: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[str] = Field(default_factory=list)
    primary_metric: str = ""
    min_sample_size: int = 1000
    max_duration_days: int = 14
    significance_level: float = 0.05
    power: float = 0.8
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    results: Dict[str, Any] = Field(default_factory=dict)


class TrainingJob(BaseModel):
    """Training job tracking."""
    model_config = ConfigDict(use_enum_values=True)

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    model_spec: ModelSpec
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    metrics_history: List[Dict[str, float]] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    resource_usage: Dict[str, Any] = Field(default_factory=dict)


class ServingEndpoint(BaseModel):
    """Model serving endpoint configuration."""
    model_config = ConfigDict(use_enum_values=True)

    endpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    model_spec: ModelSpec
    model_version: str = "latest"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    batch_size: int = 32
    timeout_ms: int = 100
    enable_batching: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    rate_limit_rps: int = 1000
    status: ServingStatus = ServingStatus.STARTING
    health_check_interval_seconds: int = 30
    metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DriftAlert(BaseModel):
    """Drift detection alert."""
    model_config = ConfigDict(use_enum_values=True)

    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    drift_type: DriftType
    severity: DriftSeverity
    feature_name: Optional[str] = None
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    detection_method: str
    description: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    auto_retrain_triggered: bool = False


class ABTestConfig(BaseModel):
    """A/B test runtime configuration."""
    model_config = ConfigDict(use_enum_values=True)

    experiment_id: str
    variant_assignments: Dict[str, str] = Field(default_factory=dict)  # user_id -> variant
    sticky_sessions: bool = True
    ramp_up_schedule: Optional[Dict[str, float]] = None  # day -> traffic %
    holdout_percentage: float = 0.0
    min_sessions_per_variant: int = 100
    guardrail_metrics: Dict[str, float] = Field(default_factory=dict)  # metric -> max degradation


class UserSession(BaseModel):
    """Simulated/user session for traffic generation."""
    model_config = ConfigDict(use_enum_values=True)

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    user_features: Dict[str, Any] = Field(default_factory=dict)
    context_features: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    device_type: str = "mobile"
    platform: str = "ios"
    geo_country: str = "US"
    referrer: Optional[str] = None
    ab_variant: Optional[str] = None


class ClickEvent(BaseModel):
    """Click/impression event."""
    model_config = ConfigDict(use_enum_values=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    item_id: str
    position: int
    score: float
    model_version: str
    variant: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    features: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversionEvent(BaseModel):
    """Conversion event (purchase, signup, etc.)."""
    model_config = ConfigDict(use_enum_values=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    item_id: str
    conversion_type: str  # purchase, signup, download, etc.
    value: float = 0.0
    currency: str = "USD"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    attribution_window_hours: int = 24
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureVector(BaseModel):
    """Feature vector for online serving."""
    model_config = ConfigDict(use_enum_values=True)

    entity_id: str
    entity_type: str  # user, item, user_item
    features: Dict[str, Any]
    feature_timestamp: datetime
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    freshness_seconds: Optional[float] = None


class PredictionRequest(BaseModel):
    """Online prediction request."""
    model_config = ConfigDict(use_enum_values=True)

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    model_version: str = "latest"
    entities: List[FeatureVector]
    context: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionResponse(BaseModel):
    """Online prediction response."""
    model_config = ConfigDict(use_enum_values=True)

    request_id: str
    model_id: str
    model_version: str
    predictions: List[Dict[str, Any]]  # entity_id -> {score, rank, ...}
    latency_ms: float
    cache_hit: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricSnapshot(BaseModel):
    """Point-in-time metric snapshot."""
    model_config = ConfigDict(use_enum_values=True)

    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    model_version: str
    variant: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, float]
    sample_size: int
    window_start: datetime
    window_end: datetime


# Configuration schemas
class FeatureStoreConfig(BaseModel):
    """Feature store configuration."""
    offline_store: Dict[str, Any] = Field(default_factory=dict)  # SQLite, Parquet, BigQuery
    online_store: Dict[str, Any] = Field(default_factory=dict)   # Redis, DynamoDB, Cassandra
    registry: Dict[str, Any] = Field(default_factory=dict)       # SQL database for metadata
    entity_types: List[str] = Field(default_factory=list)
    default_ttl_seconds: int = 86400
    enable_feature_versioning: bool = True


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""
    experiment_name: str
    model_type: ModelType
    data_source: Dict[str, Any]
    feature_list: List[str]
    target: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    early_stopping_rounds: int = 50
    num_boost_round: int = 1000
    eval_metric: str = "auc"
    objective: str = "binary:logistic"
    random_seed: int = 42
    gpu_enabled: bool = False
    mlflow_tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None


class ServingConfig(BaseModel):
    """Model serving configuration."""
    model_id: str
    model_version: str = "latest"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    batch_size: int = 32
    timeout_ms: int = 100
    enable_batching: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    rate_limit_rps: int = 1000
    health_check_interval_seconds: int = 30
    log_predictions: bool = True
    prediction_log_path: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    name: str
    hypothesis: str
    control_model_id: str
    treatment_model_ids: List[str]
    traffic_split: Dict[str, float]
    primary_metric: str
    guardrail_metrics: Dict[str, float] = Field(default_factory=dict)
    min_sample_size: int = 1000
    max_duration_days: int = 14
    significance_level: float = 0.05
    power: float = 0.8
    targeting_rules: Dict[str, Any] = Field(default_factory=dict)
    ramp_up_days: int = 0
    holdout_percentage: float = 0.0


class MonitoringConfig(BaseModel):
    """Monitoring and drift detection configuration."""
    model_id: str
    model_version: str = "latest"
    check_interval_minutes: int = 60
    drift_detection_methods: List[str] = Field(default_factory=lambda: ["psi", "ks", "chi2"])
    psi_threshold: float = 0.2
    ks_threshold: float = 0.05
    chi2_threshold: float = 0.05
    prediction_drift_threshold: float = 0.1
    min_samples_for_drift: int = 100
    alert_channels: List[str] = Field(default_factory=list)  # slack, email, pagerduty
    auto_retrain_on_drift: bool = False
    retrain_config: Optional[TrainingConfig] = None
    feature_importance_threshold: float = 0.01


def load_config(config_path: str, config_type: type) -> BaseModel:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return config_type(**data)