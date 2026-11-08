"""Main CLI application for the ML lifecycle platform."""

import typer
import logging
import os
import json
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from core.config import load_config, get_config_dir, save_config
from core.schemas import (
    FeatureStoreConfig, TrainingConfig, ServingConfig, ExperimentConfig, MonitoringConfig,
    ModelType, ModelSpec, FeatureSpec, FeatureType, FeatureSource
)

# Sub-commands
from features.offline_store import OfflineFeatureStore, create_sample_features
from features.online_store import OnlineFeatureStore, create_sample_online_data, MockOnlineFeatureStore
from training.pipeline import TrainingPipeline, create_training_config_example
from training.registry import ModelRegistry, ModelStage
from serving import ModelServer, create_serving_config
from experimentation.framework import ABTestFramework
from monitoring.drift import DataDriftDetector, PredictionDriftDetector
from monitoring.alerts import AlertManager, AlertChannel
from simulator import TrafficSimulator, SimulationConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = typer.Typer(name="mlops", help="MLOps CLI for managing the entire ML lifecycle.")


# --- Init Command ---
@app.command()
def init(project_name: str = typer.Argument(..., help="Name of the MLOps project.")):
    """Initialize a new MLOps project configuration."""
    project_dir = Path.cwd() / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Initializing MLOps project at {project_dir}")

    # Create default config files
    config_dir = project_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    # Feature Store Config
    feature_store_config = FeatureStoreConfig(
        offline_store={"path": str(project_dir / "data/offline_store.db")},
        online_store={"redis_url": "redis://localhost:6379/0"}
    )
    save_config(feature_store_config, str(config_dir / "feature_store.yaml"))

    # Training Config
    training_config = create_training_config_example()
    save_config(training_config, str(config_dir / "training.yaml"))

    # Serving Config (placeholder, requires model_id)
    serving_config = ServingConfig(model_id="", model_version="", host="0.0.0.0", port=8080)
    save_config(serving_config, str(config_dir / "serving.yaml"))

    # Experiment Config (placeholder)
    exp_config = ExperimentConfig(
        name="default_exp",
        hypothesis="",
        control_model_id="",
        treatment_model_ids=[],
        traffic_split={},
        primary_metric="ctr"
    )
    save_config(exp_config, str(config_dir / "experiment.yaml"))

    # Monitoring Config (placeholder, requires model_id)
    monitoring_config = MonitoringConfig(model_id="")
    save_config(monitoring_config, str(config_dir / "monitoring.yaml"))

    typer.echo(f"Project {project_name} initialized successfully.")
    typer.echo(f"Configuration files created in {config_dir}")


# --- Features Group ---
features_app = typer.Typer(name="features", help="Manage feature engineering and feature stores.")
app.add_typer(features_app)

@features_app.command("build")
def build_features(
    config_path: Path = typer.Option(
        "configs/feature_store.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to feature store configuration YAML."
    ),
    populate_sample: bool = typer.Option(False, "--sample", "-s", help="Populate with sample features.")
):
    """Build and manage offline feature store (materialization)."""
    typer.echo(f"Loading feature store config from {config_path}")
    config = load_config(str(config_path), FeatureStoreConfig)

    offline_store = OfflineFeatureStore(
        db_path=config.offline_store.get("path", "data/offline_store.db")
    )

    if populate_sample:
        create_sample_features(offline_store)
        typer.echo("Offline feature store populated with sample data.")

    typer.echo("Feature store build command completed (currently placeholder for materialization logic).")

@features_app.command("online-populate")
def populate_online_features(
    config_path: Path = typer.Option(
        "configs/feature_store.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to feature store configuration YAML."
    ),
    mock_redis: bool = typer.Option(False, "--mock-redis", help="Use mock Redis store (in-memory).")
):
    """Populate online feature store with sample data."""
    typer.echo(f"Loading feature store config from {config_path}")
    config = load_config(str(config_path), FeatureStoreConfig)

    if mock_redis:
        online_store = MockOnlineFeatureStore()
        typer.echo("Using MockOnlineFeatureStore (in-memory).")
    else:
        redis_url = config.online_store.get("redis_url", "redis://localhost:6379/0")
        online_store = OnlineFeatureStore(redis_url=redis_url)
        typer.echo(f"Connecting to Redis at {redis_url}")

    create_sample_online_data(online_store)
    typer.echo("Online feature store populated with sample data.")


# --- Training Group ---
training_app = typer.Typer(name="training", help="Manage model training pipelines.")
app.add_typer(training_app)

@training_app.command("run")
def run_training(
    config_path: Path = typer.Option(
        "configs/training.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to training configuration YAML."
    ),
    start_time_str: str = typer.Option(
        (datetime.utcnow() - timedelta(days=7)).isoformat(),
        "--start-time",
        help="Training data start time (ISO format, e.g., YYYY-MM-DDTHH:MM:SSZ)."
    ),
    end_time_str: str = typer.Option(
        datetime.utcnow().isoformat(),
        "--end-time",
        help="Training data end time (ISO format, e.g., YYYY-MM-DDTHH:MM:SSZ)."
    ),
    experiment_name: Optional[str] = typer.Option(None, "--experiment", "-e", help="MLflow experiment name."),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Name for the trained model.")
):
    """Run a model training pipeline."""
    typer.echo(f"Loading training config from {config_path}")
    config = load_config(str(config_path), TrainingConfig)

    if experiment_name:
        config.experiment_name = experiment_name

    # Ensure model_name is set either from option or a default based on model_type
    if model_name:
        model_spec_name = model_name
    else:
        model_spec_name = f"{config.model_type.value}_model_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # Prepare a ModelSpec for the training job
    model_spec = ModelSpec(
        name=model_spec_name,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        training_config=config.model_dump()
    )

    start_time = datetime.fromisoformat(start_time_str)
    end_time = datetime.fromisoformat(end_time_str)

    pipeline = TrainingPipeline(config)
    # Get entity_type from config data_source
    entity_type = config.data_source.get("entity_type", "user_item")
    job = pipeline.run(start_time, end_time, entity_type=entity_type, model_spec=model_spec)

    typer.echo(f"Training job {job.job_id} {job.status.value}.")
    typer.echo(f"Model {job.model_spec.name} v{job.model_spec.version} trained and logged.")
    if job.model_spec.metrics:
        for metric, value in job.model_spec.metrics.items():
            typer.echo(f"  {metric}: {value:.4f}")

@training_app.command("register")
def register_model(
    model_path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, help="Path to the trained model artifact directory."),
    model_name: str = typer.Option(..., help="Name to register the model under."),
    model_version: str = typer.Option("1.0.0", help="Version of the model."),
    model_type: ModelType = typer.Option(ModelType.RANKER, help="Type of the model (e.g., ranker, recommender)."),
    framework: str = typer.Option("xgboost", help="Model framework (e.g., xgboost, pytorch)."),
    description: Optional[str] = typer.Option(None, help="Description of the model."),
    trained_by: Optional[str] = typer.Option("cli-user", help="User who trained the model."),
    git_commit: Optional[str] = typer.Option(None, help="Git commit hash at training time.")
):
    """Register a trained model in the model registry."""
    registry = ModelRegistry()

    # Load model_spec.json from model_path if exists
    spec_file = model_path / "model_spec.json"
    if spec_file.exists():
        with open(spec_file, 'r') as f:
            spec_data = json.load(f)
        model_spec = ModelSpec(**spec_data)
    else:
        # Create a basic ModelSpec if no file exists
        model_spec = ModelSpec(
            name=model_name,
            model_id=f"{model_name}-{model_version}", # A unique ID if not coming from a full spec
            model_type=model_type,
            version=model_version,
            description=description or "",
            framework=framework,
            trained_by=trained_by,
            git_commit=git_commit,
            trained_at=datetime.utcnow(),
            status=TrainingStatus.COMPLETED # Assume completed for manual registration
        )

    # Ensure artifact_path is set to the directory for copying
    model_spec.artifact_path = str(model_path)

    registry.register_model(model_spec, artifact_path=str(model_path))
    typer.echo(f"Model {model_name} v{model_version} registered successfully.")


@training_app.command("list-models")
def list_models(
    model_type: Optional[ModelType] = typer.Option(None, help="Filter by model type."),
    stage: Optional[ModelStage] = typer.Option(None, help="Filter by model stage (e.g., production, staging).")
):
    """List registered models in the model registry."""
    registry = ModelRegistry()
    models = registry.list_models(model_type=model_type, stage=stage)

    if not models:
        typer.echo("No models found.")
        return

    for model_info in models:
        typer.echo(f"- {model_info['name']} (ID: {model_info['model_id']}) - Type: {model_info['model_type']}")
        versions = registry.list_versions(model_info['model_id'])
        for ver in versions:
            typer.echo(f"    v{ver['version']} - Stage: {ver['stage']} - Created: {ver['created_at'][:10]}")


# --- Serving Group ---
serving_app = typer.Typer(name="serving", help="Manage model serving endpoints.")
app.add_typer(serving_app)

@serving_app.command("start")
def start_serving(
    config_path: Path = typer.Option(
        "configs/serving.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to serving configuration YAML."
    ),
    model_id: str = typer.Option(..., help="ID of the model to serve from the registry."),
    model_version: str = typer.Option("latest", help="Version of the model to serve."),
    port: int = typer.Option(8080, "--port", "-p", help="Port to run the serving API on.")
):
    """Start a model serving endpoint."""
    typer.echo(f"Loading serving config from {config_path}")
    config = load_config(str(config_path), ServingConfig)

    config.model_id = model_id
    config.model_version = model_version
    config.port = port

    # Attempt to load model from registry (optional, for metadata)
    registry = ModelRegistry()
    model_info = registry.get_version(model_id, model_version)
    if not model_info:
        typer.echo(f"Warning: Model {model_id} v{model_version} not found in registry. Serving raw artifact.")
        # If not found, create a dummy ModelSpec based on CLI inputs for the server to load
        dummy_model_spec = ModelSpec(
            model_id=model_id,
            name=model_id,
            model_type=ModelType.RANKER, # Default to ranker if not specified
            version=model_version,
            framework="xgboost", # Default to xgboost
            description="Served via CLI, not explicitly in registry."
        )
        # Save dummy spec for server to pick up
        serving_model_dir = Path("artifacts/models") / f"{model_id}_{model_version}"
        serving_model_dir.mkdir(parents=True, exist_ok=True)
        with open(serving_model_dir / "model_spec.json", 'w') as f:
            json.dump(dummy_model_spec.model_dump(), f, default=str, indent=2)
        typer.echo(f"Created dummy model spec for {model_id} v{model_version} at {serving_model_dir}")

        # Also try to copy model.xgb and feature_pipeline.pkl if they exist from a default location
        # This assumes a standard training output structure
        src_model_path = Path(f"artifacts/models/{model_id}_{model_version}.pkl") # Example model artifact path
        if src_model_path.exists():
            typer.echo(f"Attempting to copy model artifact from {src_model_path}")
            try:
                import shutil
                shutil.copy2(src_model_path, serving_model_dir / "model.xgb") # Assuming XGBoost model
                typer.echo(f"Copied {src_model_path.name} to {serving_model_dir}")
            except Exception as e:
                typer.echo(f"Error copying model artifact: {e}")

        src_pipeline_path = Path(f"artifacts/models/{model_id}_{model_version}_pipeline.pkl") # Example pipeline artifact path
        if src_pipeline_path.exists():
            typer.echo(f"Attempting to copy feature pipeline from {src_pipeline_path}")
            try:
                import shutil
                shutil.copy2(src_pipeline_path, serving_model_dir / "feature_pipeline.pkl")
                typer.echo(f"Copied {src_pipeline_path.name} to {serving_model_dir}")
            except Exception as e:
                typer.echo(f"Error copying feature pipeline: {e}")

    # Instantiate the server
    server = ModelServer(config)
    typer.echo(f"Starting model serving for {config.model_id} v{config.model_version} on port {config.port}...")
    server.run()


# --- Experimentation Group ---
experiment_app = typer.Typer(name="experiment", help="Manage A/B experiments.")
app.add_typer(experiment_app)

@experiment_app.command("create")
def create_experiment(
    config_path: Path = typer.Option(
        "configs/experiment.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to experiment configuration YAML."
    )
):
    """Create a new A/B experiment."""
    typer.echo(f"Loading experiment config from {config_path}")
    config = load_config(str(config_path), ExperimentConfig)

    # Fetch model specs for control and treatments from registry
    registry = ModelRegistry()
    control_model_spec = registry.get_version(config.control_model_id, "latest")
    if not control_model_spec:
        raise typer.BadParameter(f"Control model {config.control_model_id} not found in registry.")

    treatment_model_specs = []
    for model_id in config.treatment_model_ids:
        treatment_spec = registry.get_version(model_id, "latest")
        if not treatment_spec:
            raise typer.BadParameter(f"Treatment model {model_id} not found in registry.")
        treatment_model_specs.append(ModelSpec(**treatment_spec))

    # Reconstruct ExperimentSpec with full model specs
    exp_spec = ExperimentSpec(
        name=config.name,
        hypothesis=config.hypothesis,
        control_model=ModelSpec(**control_model_spec),
        treatment_models=treatment_model_specs,
        traffic_split=config.traffic_split,
        primary_metric=config.primary_metric,
        metrics=config.metrics, # Assuming this is handled or defaults
        min_sample_size=config.min_sample_size,
        max_duration_days=config.max_duration_days,
        significance_level=config.significance_level,
        power=config.power,
        targeting_rules=config.targeting_rules
    )

    framework = ABTestFramework()
    exp_id = framework.create_experiment(exp_spec)
    typer.echo(f"Experiment {config.name} (ID: {exp_id}) created successfully.")
    typer.echo(f"To start the experiment, run: mlops experiment start {exp_id}")


@experiment_app.command("start")
def start_experiment(
    experiment_id: str = typer.Argument(..., help="ID of the experiment to start.")
):
    """Start a created A/B experiment."""
    framework = ABTestFramework()
    if framework.start_experiment(experiment_id):
        typer.echo(f"Experiment {experiment_id} started.")
    else:
        typer.echo(f"Failed to start experiment {experiment_id}.")


@experiment_app.command("stop")
def stop_experiment(
    experiment_id: str = typer.Argument(..., help="ID of the experiment to stop."),
    reason: str = typer.Option("Manual stop via CLI", "--reason", "-r", help="Reason for stopping the experiment.")
):
    """Stop a running A/B experiment."""
    framework = ABTestFramework()
    if framework.stop_experiment(experiment_id, reason):
        typer.echo(f"Experiment {experiment_id} stopped.")
    else:
        typer.echo(f"Failed to stop experiment {experiment_id}.")


@experiment_app.command("status")
def experiment_status(
    experiment_id: str = typer.Argument(..., help="ID of the experiment to check status for.")
):
    """Get the current status and metrics of an experiment."""
    framework = ABTestFramework()
    exp = framework.get_experiment(experiment_id)
    if not exp:
        typer.echo(f"Experiment {experiment_id} not found.")
        raise typer.Exit(code=1)

    typer.echo(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
    typer.echo(f"Status: {exp.status.value}")
    typer.echo(f"Started: {exp.started_at.isoformat() if exp.started_at else 'N/A'}")
    typer.echo(f"Ended: {exp.ended_at.isoformat() if exp.ended_at else 'N/A'}")
    typer.echo(f"Primary Metric: {exp.primary_metric}")

    if exp.status == ExperimentStatus.RUNNING:
        stats = framework.get_variant_stats(experiment_id)
        typer.echo("\n--- Current Variant Stats ---")
        for variant, data in stats.items():
            typer.echo(f"Variant: {variant}")
            typer.echo(f"  Users: {data.get('users', 0)}")
            for metric_name, value in data.items():
                if metric_name.endswith("_mean"):
                    typer.echo(f"  {metric_name.replace('_mean', '').upper()} Mean: {value:.4f}")

        analysis = framework.analyze_experiment(experiment_id)
        typer.echo("\n--- Analysis (Frequentist) ---")
        typer.echo(f"Decision: {analysis.overall_decision}")
        typer.echo(f"Recommendation: {analysis.recommendation}")
        typer.echo(f"Primary Metric ({analysis.primary_metric.metric_name.upper()}):")
        typer.echo(f"  Control Mean: {analysis.primary_metric.control_mean:.4f}")
        typer.echo(f"  Treatment Mean: {analysis.primary_metric.treatment_mean:.4f}")
        typer.echo(f"  Relative Diff: {analysis.primary_metric.relative_diff:.2f}%")
        typer.echo(f"  P-value: {analysis.primary_metric.p_value:.4f}")
        typer.echo(f"  Significant: {analysis.primary_metric.is_significant}")
        typer.echo(f"  Confidence Interval: {analysis.primary_metric.confidence_interval}")

        guardrails = framework.check_guardrails(experiment_id)
        if guardrails["violations"]:
            typer.echo(f"\n--- Guardrail Violations ({guardrails['overall_status']}) ---")
            for g_res in guardrails["guardrails"]:
                if g_res["status"] == "violation":
                    typer.echo(f"  - {g_res['metric_name']}: {g_res['message']}")
        elif guardrails["warnings"]:
            typer.echo(f"\n--- Guardrail Warnings ({guardrails['overall_status']}) ---")
            for g_res in guardrails["guardrails"]:
                if g_res["status"] == "warning":
                    typer.echo(f"  - {g_res['metric_name']}: {g_res['message']}")
        else:
            typer.echo("\n--- Guardrails: PASS ---")


# --- Monitoring Group ---
monitoring_app = typer.Typer(name="monitoring", help="Monitor models and detect drift.")
app.add_typer(monitoring_app)

@monitoring_app.command("start")
def start_monitoring(
    config_path: Path = typer.Option(
        "configs/monitoring.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to monitoring configuration YAML."
    ),
    model_id: str = typer.Option(..., help="ID of the model to monitor.")
):
    """Start continuous monitoring for a model."""
    typer.echo(f"Loading monitoring config from {config_path}")
    config = load_config(str(config_path), MonitoringConfig)
    config.model_id = model_id

    # Initialize components
    offline_store = OfflineFeatureStore()
    alert_manager = AlertManager()

    # For real-time monitoring, we'd connect to a stream of predictions/feedback
    # For now, this is a placeholder. A background task would run drift detection.
    typer.echo(f"Starting monitoring for model {config.model_id}. (Background process not yet implemented in CLI)")
    typer.echo("You would typically set up a cron job or a dedicated service for continuous monitoring.")


@monitoring_app.command("check-drift")
def check_drift(
    model_id: str = typer.Argument(..., help="ID of the model to check drift for."),
    config_path: Path = typer.Option(
        "configs/monitoring.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to monitoring configuration YAML."
    ),
    data_path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, help="Path to current data (e.g., CSV, Parquet) for drift detection."),
    baseline_path: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, help="Path to baseline data (e.g., CSV, Parquet) for comparison. If not provided, will attempt to load from feature store.")
):
    """Perform an on-demand drift check for a model."""
    typer.echo(f"Loading monitoring config from {config_path}")
    config = load_config(str(config_path), MonitoringConfig)
    config.model_id = model_id

    # Load current data
    current_df = pd.read_csv(data_path) if data_path.suffix == ".csv" else pd.read_parquet(data_path)
    typer.echo(f"Loaded current data from {data_path} ({len(current_df)} rows).")

    # Load baseline data
    baseline_df = None
    if baseline_path:
        baseline_df = pd.read_csv(baseline_path) if baseline_path.suffix == ".csv" else pd.read_parquet(baseline_path)
        typer.echo(f"Loaded baseline data from {baseline_path} ({len(baseline_df)} rows).")
    else:
        typer.echo("No baseline path provided. Attempting to get baseline from OfflineFeatureStore.")
        offline_store = OfflineFeatureStore()
        # This assumes a model_id is linked to a training job with specific data and features
        # For simplicity, we'll need to mock/simplify this or require more explicit config
        typer.echo("Note: Automatic baseline retrieval from feature store needs more context (e.g., training job details).")
        # Placeholder: for now, if no baseline_path, we'll simulate/error
        raise typer.BadParameter("Baseline data path is required for drift detection.")

    # Initialize Data Drift Detector
    data_drift_detector = DataDriftDetector(
        thresholds={
            "psi": config.psi_threshold,
            "ks": config.ks_threshold,
            "chi2": config.chi2_threshold
        }
    )
    data_drift_detector.fit(baseline_df) # Fit on baseline
    drift_results = data_drift_detector.detect_drift(current_df)
    drift_summary = data_drift_detector.get_drift_summary(drift_results)

    typer.echo("\n--- Data Drift Detection Results ---")
    typer.echo(f"Overall Drift Detected: {drift_summary['features_with_drift'] > 0}")
    typer.echo(f"Features with Drift: {drift_summary['features_with_drift']}/{drift_summary['total_features_checked']}")
    for feature_info in drift_summary["drifted_features"]:
        typer.echo(f"  - Feature: {feature_info['feature']} (Method: {feature_info['method']}, Score: {feature_info['score']:.4f}, Threshold: {feature_info['threshold']:.4f})")

    if config.auto_retrain_on_drift and drift_summary['features_with_drift'] > 0:
        typer.echo("Auto-retrain is enabled and drift detected. Triggering retraining... (Not implemented yet)")

    alert_manager = AlertManager()
    alerts = data_drift_detector.create_alerts(model_id, drift_results, drift_type=DriftType.DATA_DRIFT)
    for alert in alerts:
        alert_manager.create_drift_alert(alert)
        typer.echo(f"Created alert: {alert.title} (Severity: {alert.severity.value})")


# --- Simulator Group ---
simulator_app = typer.Typer(name="simulator", help="Run traffic simulations and load tests.")
app.add_typer(simulator_app)

@simulator_app.command("run-traffic")
def run_traffic_simulation(
    config_path: Path = typer.Option(
        "configs/simulation.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to simulation configuration YAML."
    ),
    duration_seconds: int = typer.Option(60, "--duration", "-d", help="Duration of the simulation in seconds."),
    concurrent_sessions: int = typer.Option(10, "--concurrent", "-j", help="Number of concurrent user sessions."),
    model_version: str = typer.Option("latest", help="Model version to use for predictions in simulation.")
):
    """Run a traffic simulation to generate user interactions and predictions."""
    typer.echo(f"Loading simulation config from {config_path}")
    sim_config = load_config(str(config_path), SimulationConfig)

    # Create a mock online feature store for simulation if not explicitly provided
    if not sim_config.feature_store:
        sim_config.feature_store = MockOnlineFeatureStore()
        create_sample_online_data(sim_config.feature_store, num_users=sim_config.num_users, num_items=1000)
        typer.echo("Populated mock online feature store for simulation.")

    simulator = TrafficSimulator(sim_config)

    typer.echo(f"Starting traffic simulation for {duration_seconds} seconds with {concurrent_sessions} concurrent sessions...")
    asyncio.run(simulator.run_simulation(duration_seconds, concurrent_sessions, model_version=model_version))

    results = simulator.get_results()
    typer.echo("\n--- Simulation Results ---")
    typer.echo(f"Total Sessions: {results['summary']['total_sessions']}")
    typer.echo(f"Total Clicks: {results['summary']['total_clicks']}")
    typer.echo(f"Total Conversions: {results['summary']['total_conversions']}")
    typer.echo(f"Avg Latency (ms): {results['summary']['avg_latency_ms']:.2f}")

    typer.echo("\n--- Metrics By Variant ---")
    for variant, metrics in results['by_variant'].items():
        typer.echo(f"Variant: {variant}")
        typer.echo(f"  Impressions: {metrics['impressions']}")
        typer.echo(f"  Clicks: {metrics['clicks']}")
        typer.echo(f"  CTR: {metrics['ctr']:.4f}")
        typer.echo(f"  Conversions: {metrics['conversions']}")
        typer.echo(f"  CVR: {metrics['cvr']:.4f}")


@simulator_app.command("run-load-test")
def run_load_test(
    config_path: Path = typer.Option(
        "configs/simulation.yaml",
        "--config", "-c",
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True,
        resolve_path=True, help="Path to simulation configuration YAML."
    ),
    num_requests: int = typer.Option(1000, "--num-requests", "-n", help="Total number of prediction requests to send."),
    concurrent_requests: int = typer.Option(50, "--concurrent", "-j", help="Maximum number of concurrent requests."),
    model_id: str = typer.Option("ranker_v1", help="Model ID to target for load test predictions."),
    model_version: str = typer.Option("latest", help="Model version to target.")
):
    """Run a load test against a model serving endpoint."""
    typer.echo(f"Loading simulation config from {config_path}")
    sim_config = load_config(str(config_path), SimulationConfig)

    # Create a mock online feature store for simulation if not explicitly provided
    if not sim_config.feature_store:
        sim_config.feature_store = MockOnlineFeatureStore()
        create_sample_online_data(sim_config.feature_store, num_users=sim_config.num_users, num_items=1000)
        typer.echo("Populated mock online feature store for load test.")

    # The TrafficSimulator itself is used, but for load testing it doesn't call an external endpoint for predictions
    # It simulates a local prediction based on its internal logic.
    # For a real load test against an external endpoint, the simulator's _simulate_prediction would be replaced
    # by an actual HTTP call to config.model_endpoint.

    simulator = TrafficSimulator(sim_config)

    typer.echo(f"Starting load test with {num_requests} requests, {concurrent_requests} concurrent...")
    load_results = asyncio.run(simulator.run_load_test(num_requests, concurrent_requests, model_id, model_version))

    typer.echo("\n--- Load Test Results ---")
    typer.echo(f"Total Requests: {load_results['total_requests']}")
    typer.echo(f"Successful Requests: {load_results['successful_requests']}")
    typer.echo(f"Errors: {load_results['errors']}")
    typer.echo(f"Error Rate: {load_results['error_rate']:.2%}")
    typer.echo(f"Duration (s): {load_results['duration_seconds']:.2f}")
    typer.echo(f"Throughput (rps): {load_results['throughput_rps']:.2f}")
    typer.echo(f"Latency (ms):")
    for percentile, value in load_results['latency_stats'].items():
        typer.echo(f"  {percentile}: {value:.2f}")

# --- Main entry point ---
if __name__ == "__main__":
    app()