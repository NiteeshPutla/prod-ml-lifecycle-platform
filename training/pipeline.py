"""Training pipeline for ranking, recommendation, search, and ads models."""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, ndcg_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.sklearn

from core.schemas import (
    ModelSpec, ModelType, TrainingJob, TrainingStatus, FeatureSpec,
    FeatureType, FeatureSource, TrainingConfig
)
from features.offline_store import OfflineFeatureStore
from features.transformers import FeaturePipeline, create_ranking_feature_pipeline

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    train_auc: float
    val_auc: float
    test_auc: float
    train_logloss: float
    val_logloss: float
    test_logloss: float
    train_samples: int
    val_samples: int
    test_samples: int
    feature_importance: Dict[str, float]
    training_time_seconds: float
    epochs_trained: int


class TrainingPipeline:
    """End-to-end training pipeline with MLflow tracking."""

    def __init__(
        self,
        config: TrainingConfig,
        feature_store: OfflineFeatureStore = None,
        mlflow_tracking_uri: str = None,
        artifact_dir: str = "artifacts/models"
    ):
        self.config = config
        feature_store_db_path = config.data_source.get("db_path", "data/offline_store.db")
        self.feature_store = feature_store or OfflineFeatureStore(db_path=feature_store_db_path)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        elif config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)

        mlflow.set_experiment(config.experiment_name)

        self.feature_pipeline = None
        self.model = None
        self.model_spec = None

    def prepare_data(
        self,
        start_time: datetime,
        end_time: datetime,
        entity_type: str = "user_item"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from feature store."""
        # Get feature names from config
        feature_names = self.config.feature_list.copy()
        target = self.config.target

        # Fetch training data
        df = self.feature_store.get_training_data(
            entity_type=entity_type,
            feature_names=feature_names,
            target_feature=target,
            start_time=start_time,
            end_time=end_time
        )

        if df.empty:
            raise ValueError("No training data found")

        logger.info(f"Loaded training data: {df.shape}")

        # Split features and target
        X = df[feature_names]
        y = df[target]

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train/val/test."""
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
            stratify=y if len(y.unique()) == 2 else None
        )

        # Second split: train vs val
        val_ratio = self.config.val_split / (self.config.train_split + self.config.val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.config.random_seed,
            stratify=y_trainval if len(y_trainval.unique()) == 2 else None
        )

        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_feature_pipeline(self, X_train: pd.DataFrame) -> FeaturePipeline:
        """Build feature engineering pipeline from training data."""
        # Infer feature types from data
        feature_specs = {}

        for col in X_train.columns:
            dtype = X_train[col].dtype
            if dtype in ['int64', 'float64', 'int32', 'float32']:
                # Check if it's likely categorical (low cardinality)
                if X_train[col].nunique() < 20 and X_train[col].min() >= 0:
                    feature_specs[col] = {"type": "categorical", "method": "label"}
                else:
                    feature_specs[col] = {"type": "numerical", "method": "standard"}
            elif dtype == 'object':
                # Check if it's embedding (list/array)
                sample = X_train[col].dropna().iloc[0] if not X_train[col].dropna().empty else None
                if isinstance(sample, (list, np.ndarray)):
                    feature_specs[col] = {"type": "embedding", "embedding_dim": len(sample) if hasattr(sample, '__len__') else 64}
                else:
                    feature_specs[col] = {"type": "categorical", "method": "label"}

        # Override with config if provided
        if hasattr(self.config, 'feature_specs') and self.config.feature_specs:
            feature_specs.update(self.config.feature_specs)

        pipeline = FeaturePipeline(self.config.__dict__ if hasattr(self.config, '__dict__') else {})
        pipeline.build(feature_specs)
        pipeline.fit(X_train)

        return pipeline

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Dict[str, Any] = None
    ) -> Tuple[xgb.Booster, TrainingMetrics]:
        """Train XGBoost model."""
        # Default parameters
        default_params = {
            "objective": self.config.objective,
            "eval_metric": self.config.eval_metric,
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "lambda": 1,
            "alpha": 0,
            "seed": self.config.random_seed,
            "verbosity": 1,
            "nthread": -1,
        }

        if params:
            default_params.update(params)

        if self.config.gpu_enabled:
            default_params.update({
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor"
            })

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train with early stopping
        evals = [(dtrain, "train"), (dval, "val")]
        evals_result = {}

        start_time = datetime.utcnow()

        model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50
        )

        training_time = (datetime.utcnow() - start_time).total_seconds()
        epochs_trained = model.best_iteration + 1

        # Compute metrics
        train_pred = model.predict(dtrain)
        val_pred = model.predict(dval)

        if self.config.objective == "binary:logistic":
            train_auc = roc_auc_score(y_train, train_pred)
            val_auc = roc_auc_score(y_val, val_pred)
            train_ll = log_loss(y_train, train_pred)
            val_ll = log_loss(y_val, val_pred)
        else:
            train_auc = val_auc = 0.0
            train_ll = val_ll = 0.0

        # Feature importance
        importance = model.get_score(importance_type="gain")
        total = sum(importance.values())
        importance = {k: v/total for k, v in importance.items()}

        metrics = TrainingMetrics(
            train_auc=train_auc,
            val_auc=val_auc,
            test_auc=0.0,  # Will be filled later
            train_logloss=train_ll,
            val_logloss=val_ll,
            test_logloss=0.0,
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=0,
            feature_importance=importance,
            training_time_seconds=training_time,
            epochs_trained=epochs_trained
        )

        return model, metrics

    def evaluate_model(
        self,
        model: xgb.Booster,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        dtest = xgb.DMatrix(X_test, label=y_test)
        test_pred = model.predict(dtest)

        if self.config.objective == "binary:logistic":
            test_auc = roc_auc_score(y_test, test_pred)
            test_ll = log_loss(y_test, test_pred)
            test_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
        else:
            test_auc = test_ll = test_acc = 0.0

        return {
            "test_auc": test_auc,
            "test_logloss": test_ll,
            "test_accuracy": test_acc
        }

    def run(
        self,
        start_time: datetime,
        end_time: datetime,
        entity_type: str = "user_item",
        model_spec: ModelSpec = None
    ) -> TrainingJob:
        """Run the complete training pipeline."""
        job = TrainingJob(
            model_spec=model_spec or ModelSpec(
                name=f"{self.config.model_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                model_type=self.config.model_type,
                hyperparameters=self.config.hyperparameters,
                training_config=self.config.__dict__
            )
        )

        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()

        with mlflow.start_run(run_name=job.model_spec.name) as run:
            try:
                # Log parameters
                mlflow.log_params({
                    "model_type": self.config.model_type.value,
                    "objective": self.config.objective,
                    "eval_metric": self.config.eval_metric,
                    "num_boost_round": self.config.num_boost_round,
                    "early_stopping_rounds": self.config.early_stopping_rounds,
                    "train_split": self.config.train_split,
                    "val_split": self.config.val_split,
                    "test_split": self.config.test_split,
                    "random_seed": self.config.random_seed,
                    **self.config.hyperparameters
                })

                # Prepare data
                logger.info("Preparing training data...")
                X, y = self.prepare_data(start_time, end_time, entity_type)

                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

                # Build feature pipeline
                logger.info("Building feature pipeline...")
                self.feature_pipeline = self.build_feature_pipeline(X_train)

                # Transform data
                logger.info("Transforming features...")
                X_train_transformed = self.feature_pipeline.transform(X_train)
                X_val_transformed = self.feature_pipeline.transform(X_val)
                X_test_transformed = self.feature_pipeline.transform(X_test)

                # Train model
                logger.info("Training model...")
                self.model, metrics = self.train_xgboost(
                    X_train_transformed, y_train,
                    X_val_transformed, y_val,
                    self.config.hyperparameters
                )

                # Evaluate on test set
                logger.info("Evaluating on test set...")
                test_metrics = self.evaluate_model(self.model, X_test_transformed, y_test)
                metrics.test_auc = test_metrics["test_auc"]
                metrics.test_logloss = test_metrics["test_logloss"]
                metrics.test_samples = len(X_test)

                # Log metrics
                mlflow.log_metrics({
                    "train_auc": metrics.train_auc,
                    "val_auc": metrics.val_auc,
                    "test_auc": metrics.test_auc,
                    "train_logloss": metrics.train_logloss,
                    "val_logloss": metrics.val_logloss,
                    "test_logloss": metrics.test_logloss,
                    "training_time_seconds": metrics.training_time_seconds,
                    "epochs_trained": metrics.epochs_trained
                })

                # Save model and pipeline
                model_path = self.artifact_dir / f"{job.model_spec.model_id}.pkl"
                pipeline_path = self.artifact_dir / f"{job.model_spec.model_id}_pipeline.pkl"

                self.model.save_model(str(model_path))
                self.feature_pipeline.save(str(pipeline_path))

                # Log artifacts
                mlflow.log_artifact(str(model_path))
                mlflow.log_artifact(str(pipeline_path))

                # Update job
                job.model_spec.metrics = {
                    "train_auc": metrics.train_auc,
                    "val_auc": metrics.val_auc,
                    "test_auc": metrics.test_auc,
                    "train_logloss": metrics.train_logloss,
                    "val_logloss": metrics.val_logloss,
                    "test_logloss": metrics.test_logloss
                }
                job.model_spec.artifact_path = str(model_path)
                job.model_spec.status = TrainingStatus.COMPLETED
                job.model_spec.trained_at = datetime.utcnow()
                job.model_spec.lineage = {
                    "feature_pipeline_path": str(pipeline_path),
                    "mlflow_run_id": run.info.run_id,
                    "training_data_period": f"{start_time.isoformat()}_{end_time.isoformat()}",
                    "feature_count": len(X_train_transformed.columns)
                }

                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.metrics_history.append(job.model_spec.metrics)

                logger.info(f"Training completed! Test AUC: {metrics.test_auc:.4f}")

            except Exception as e:
                logger.error(f"Training failed: {e}")
                job.status = TrainingStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.utcnow()
                raise

        return job

    def save_model_package(self, job: TrainingJob, output_path: str) -> str:
        """Save complete model package for deployment."""
        package_path = Path(output_path)
        package_path.mkdir(parents=True, exist_ok=True)

        # Save model spec
        with open(package_path / "model_spec.json", 'w') as f:
            json.dump(job.model_spec.model_dump(), f, default=str, indent=2)

        # Copy model artifact
        import shutil
        if job.model_spec.artifact_path:
            shutil.copy2(job.model_spec.artifact_path, package_path / "model.xgb")

        # Copy pipeline
        pipeline_path = Path(job.model_spec.lineage.get("feature_pipeline_path", ""))
        if pipeline_path.exists():
            shutil.copy2(pipeline_path, package_path / "feature_pipeline.pkl")

        # Create serving config
        serving_config = {
            "model_id": job.model_spec.model_id,
            "model_version": job.model_spec.version,
            "model_type": job.model_spec.model_type.value,
            "input_features": job.model_spec.lineage.get("feature_count", 0),
            "expected_latency_ms": 10
        }
        with open(package_path / "serving_config.json", 'w') as f:
            json.dump(serving_config, f, indent=2)

        logger.info(f"Model package saved to {package_path}")
        return str(package_path)


def create_training_config_example() -> TrainingConfig:
    """Create example training configuration."""
    return TrainingConfig(
        experiment_name="ranking_models",
        model_type=ModelType.RANKER,
        data_source={"type": "offline_store", "entity_type": "user_item"},
        feature_list=[
            "user_age", "user_gender", "user_country", "user_past_ctr",
            "user_session_count_7d", "item_category", "item_price",
            "item_ctr_7d", "item_cvr_7d", "position", "user_item_affinity"
        ],
        target="clicked",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        hyperparameters={
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        early_stopping_rounds=50,
        num_boost_round=1000,
        eval_metric="auc",
        objective="binary:logistic",
        random_seed=42,
        gpu_enabled=False
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # This would run with actual data
    print("Training pipeline module ready!")
    print("Example config:")
    config = create_training_config_example()
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Model type: {config.model_type}")
    print(f"  Features: {len(config.feature_list)}")