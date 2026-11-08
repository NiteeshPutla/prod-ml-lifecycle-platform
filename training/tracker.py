"""Experiment tracking with MLflow integration."""

import os
import logging
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import json

from core.schemas import ModelType, ExperimentSpec, ExperimentStatus

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """Container for experiment run information."""
    run_id: str
    experiment_id: str
    experiment_name: str
    run_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    params: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    artifacts: List[str]


class ExperimentTracker:
    """Experiment tracking with MLflow backend."""

    def __init__(
        self,
        tracking_uri: str = None,
        artifact_location: str = None,
        default_experiment: str = "default"
    ):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.artifact_location = artifact_location
        self.default_experiment = default_experiment

        mlflow.set_tracking_uri(self.tracking_uri)

        # Create default experiment if needed
        exp = mlflow.get_experiment_by_name(default_experiment)
        if exp is None:
            mlflow.create_experiment(default_experiment, artifact_location=artifact_location)

        mlflow.set_experiment(default_experiment)

    def create_experiment(
        self,
        name: str,
        artifact_location: str = None,
        tags: Dict[str, str] = None
    ) -> str:
        """Create a new experiment."""
        exp_id = mlflow.create_experiment(
            name,
            artifact_location=artifact_location or self.artifact_location,
            tags=tags
        )
        logger.info(f"Created experiment: {name} (id: {exp_id})")
        return exp_id

    def set_experiment(self, name: str) -> None:
        """Set active experiment."""
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            self.create_experiment(name)
        mlflow.set_experiment(name)
        self.default_experiment = name

    def start_run(
        self,
        run_name: str = None,
        experiment_name: str = None,
        tags: Dict[str, str] = None,
        nested: bool = False
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if experiment_name:
            self.set_experiment(experiment_name)

        run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        logger.info(f"Started run: {run.info.run_id} ({run_name})")
        return run

    def end_run(self, status: str = "FINISHED") -> None:
        """End current run."""
        mlflow.end_run(status=status)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int = None
    ) -> None:
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int = None) -> None:
        """Log single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log artifact."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: str = None) -> None:
        """Log all artifacts in directory."""
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        flavor: str = "xgboost",
        **kwargs
    ) -> None:
        """Log model with appropriate flavor."""
        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path, **kwargs)
        elif flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        elif flavor == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        elif flavor == "pyfunc":
            mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported flavor: {flavor}")

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log matplotlib figure."""
        mlflow.log_figure(figure, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log text content."""
        mlflow.log_text(text, artifact_file)

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        """Log dictionary as JSON artifact."""
        mlflow.log_dict(data, artifact_file)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set run tags."""
        mlflow.set_tags(tags)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get run information."""
        run = mlflow.get_run(run_id)
        if not run:
            return None

        return ExperimentRun(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            experiment_name=mlflow.get_experiment(run.info.experiment_id).name,
            run_name=run.data.tags.get("mlflow.runName", ""),
            status=run.info.status,
            start_time=datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
            end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            params=run.data.params,
            metrics=run.data.metrics,
            tags=run.data.tags,
            artifacts=[f.path for f in mlflow.list_artifacts(run_id)]
        )

    def search_runs(
        self,
        experiment_names: List[str] = None,
        filter_string: str = "",
        max_results: int = 100,
        order_by: List[str] = None
    ) -> List[ExperimentRun]:
        """Search runs across experiments."""
        if experiment_names is None:
            experiment_names = [self.default_experiment]

        experiment_ids = []
        for name in experiment_names:
            exp = mlflow.get_experiment_by_name(name)
            if exp:
                experiment_ids.append(exp.experiment_id)

        if not experiment_ids:
            return []

        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or ["metrics.test_auc DESC"]
        )

        results = []
        for _, row in runs.iterrows():
            run = mlflow.get_run(row["run_id"])
            if run:
                results.append(ExperimentRun(
                    run_id=run.info.run_id,
                    experiment_id=run.info.experiment_id,
                    experiment_name=mlflow.get_experiment(run.info.experiment_id).name,
                    run_name=run.data.tags.get("mlflow.runName", ""),
                    status=run.info.status,
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
                    end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    params=run.data.params,
                    metrics=run.data.metrics,
                    tags=run.data.tags,
                    artifacts=[f.path for f in mlflow.list_artifacts(run.info.run_id)]
                ))

        return results

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "test_auc",
        ascending: bool = False
    ) -> Optional[ExperimentRun]:
        """Get best run by metric."""
        runs = self.search_runs(
            experiment_names=[experiment_name],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        return runs[0] if runs else None

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        runs = [self.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        if not runs:
            return {}

        # Collect all params and metrics
        all_params = set()
        all_metrics = set()
        for run in runs:
            all_params.update(run.params.keys())
            all_metrics.update(run.metrics.keys())

        comparison = {
            "runs": [
                {
                    "run_id": r.run_id,
                    "run_name": r.run_name,
                    "status": r.status,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "params": r.params,
                    "metrics": r.metrics
                }
                for r in runs
            ],
            "params_comparison": {},
            "metrics_comparison": {}
        }

        for param in sorted(all_params):
            values = {r.run_id: r.params.get(param) for r in runs}
            if len(set(values.values())) > 1:
                comparison["params_comparison"][param] = values

        for metric in sorted(all_metrics):
            values = {r.run_id: r.metrics.get(metric) for r in runs}
            if any(v is not None for v in values.values()):
                comparison["metrics_comparison"][metric] = values

        return comparison

    def delete_run(self, run_id: str) -> None:
        """Delete a run."""
        mlflow.delete_run(run_id)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment."""
        mlflow.delete_experiment(experiment_id)

    def get_experiment_info(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get experiment information."""
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return None

        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "creation_time": datetime.fromtimestamp(exp.creation_time / 1000) if exp.creation_time else None,
            "tags": dict(exp.tags) if exp.tags else {}
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = mlflow.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": datetime.fromtimestamp(exp.creation_time / 1000) if exp.creation_time else None
            }
            for exp in experiments
        ]


class ExperimentManager:
    """High-level experiment management for ML lifecycle."""

    def __init__(self, tracker: ExperimentTracker = None):
        self.tracker = tracker or ExperimentTracker()

    def run_experiment(
        self,
        experiment_spec: ExperimentSpec,
        train_func: callable,
        train_args: Dict[str, Any] = None
    ) -> ExperimentRun:
        """Run a complete experiment with tracking."""
        train_args = train_args or {}

        self.tracker.set_experiment(experiment_spec.name)

        with self.tracker.start_run(
            run_name=f"{experiment_spec.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "model_type": experiment_spec.control_model.model_type.value,
                "experiment_id": experiment_spec.experiment_id
            }
        ) as run:
            # Log experiment config
            self.tracker.log_params({
                "experiment": experiment_spec.model_dump(),
                **train_args
            })

            # Run training
            result = train_func(**train_args)

            # Log results
            if hasattr(result, 'model_spec') and result.model_spec.metrics:
                self.tracker.log_metrics(result.model_spec.metrics)

            # Log model
            if hasattr(result, 'model'):
                self.tracker.log_model(result.model, "model", flavor="xgboost")

            # Log feature pipeline
            if hasattr(result, 'feature_pipeline'):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    result.feature_pipeline.save(f.name)
                    self.tracker.log_artifact(f.name, "feature_pipeline")

            return self.tracker.get_run(run.info.run_id)

    def run_ab_experiment(
        self,
        experiment_spec: ExperimentSpec,
        train_func: callable,
        train_args_list: List[Dict[str, Any]]
    ) -> List[ExperimentRun]:
        """Run A/B experiment with multiple variants."""
        runs = []

        # Run control
        control_args = train_args_list[0] if train_args_list else {}
        control_args["variant"] = "control"
        control_args["model_spec"] = experiment_spec.control_model

        run = self.run_experiment(experiment_spec, train_func, control_args)
        runs.append(run)

        # Run treatments
        for i, treatment_model in enumerate(experiment_spec.treatment_models):
            treatment_args = train_args_list[i + 1] if i + 1 < len(train_args_list) else {}
            treatment_args["variant"] = f"treatment_{i}"
            treatment_args["model_spec"] = treatment_model

            run = self.run_experiment(experiment_spec, train_func, treatment_args)
            runs.append(run)

        return runs

    def promote_best_model(
        self,
        experiment_name: str,
        metric: str = "test_auc",
        model_registry = None
    ) -> Optional[ExperimentRun]:
        """Find and promote best model from experiment."""
        best_run = self.tracker.get_best_run(experiment_name, metric=metric)

        if best_run and model_registry:
            # Register best model
            from training.registry import ModelRegistry, ModelStage
            registry = model_registry
            # This would need the actual model artifact
            logger.info(f"Best run: {best_run.run_id} with {metric}={best_run.metrics.get(metric)}")

        return best_run


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tracker = ExperimentTracker()

    # Test
    tracker.set_experiment("test_experiment")

    with tracker.start_run(run_name="test_run") as run:
        tracker.log_params({"learning_rate": 0.1, "max_depth": 6})
        tracker.log_metrics({"train_auc": 0.85, "val_auc": 0.82})
        tracker.set_tags({"model_type": "ranker", "dataset": "v1"})

    print(f"Run completed: {run.info.run_id}")

    # Search runs
    runs = tracker.search_runs(max_results=5)
    print(f"Found {len(runs)} runs")

    for r in runs:
        print(f"  {r.run_name}: {r.metrics}")