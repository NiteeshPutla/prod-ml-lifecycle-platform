"""Model registry for versioning, staging, and deployment."""

import sqlite3
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
from enum import Enum
from dataclasses import dataclass, asdict

from core.schemas import ModelSpec, ModelType, TrainingStatus

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry:
    """Model registry for versioning, staging, and deployment management."""

    def __init__(self, db_path: str = "data/model_registry.db", artifact_root: str = "artifacts/registry"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize registry database."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS registered_models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    framework TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    git_commit TEXT,
                    tags TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    stage TEXT DEFAULT 'none',
                    artifact_path TEXT,
                    metrics TEXT,
                    hyperparameters TEXT,
                    feature_pipeline_path TEXT,
                    lineage TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    UNIQUE(model_id, version)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_stage_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    from_stage TEXT,
                    to_stage TEXT,
                    changed_by TEXT,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_deployments (
                    deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    endpoint_name TEXT,
                    environment TEXT,
                    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployed_by TEXT,
                    status TEXT,
                    config TEXT
                )
            """)

            conn.commit()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def register_model(self, model_spec: ModelSpec, artifact_path: str = None) -> str:
        """Register a new model or version."""
        with self._get_conn() as conn:
            # Check if model exists
            existing = conn.execute(
                "SELECT model_id FROM registered_models WHERE name = ?",
                (model_spec.name,)
            ).fetchone()

            if existing:
                model_id = existing["model_id"]
                model_type_val = model_spec.model_type.value if hasattr(model_spec.model_type, 'value') else model_spec.model_type
                # Update model metadata
                conn.execute("""
                    UPDATE registered_models SET
                        model_type = ?, version = ?, description = ?, framework = ?,
                        updated_at = ?, git_commit = ?, tags = ?
                    WHERE model_id = ?
                """, (
                    model_type_val,
                    model_spec.version,
                    model_spec.description,
                    model_spec.framework,
                    datetime.utcnow().isoformat(),
                    model_spec.git_commit,
                    json.dumps(getattr(model_spec, 'tags', {})),
                    model_id
                ))
            else:
                model_id = model_spec.model_id
                model_type_val = model_spec.model_type.value if hasattr(model_spec.model_type, 'value') else model_spec.model_type
                conn.execute("""
                    INSERT INTO registered_models
                    (model_id, name, model_type, version, description, framework, created_by, git_commit, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id,
                    model_spec.name,
                    model_type_val,
                    model_spec.version,
                    model_spec.description,
                    model_spec.framework,
                    model_spec.trained_by or "system",
                    model_spec.git_commit,
                    json.dumps(getattr(model_spec, 'tags', {}))
                ))

            # Add version
            conn.execute("""
                INSERT OR REPLACE INTO model_versions
                (model_id, version, stage, artifact_path, metrics, hyperparameters,
                 feature_pipeline_path, lineage, status, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                model_spec.version,
                ModelStage.NONE.value,
                artifact_path or model_spec.artifact_path,
                json.dumps(model_spec.metrics),
                json.dumps(model_spec.hyperparameters),
                model_spec.lineage.get("feature_pipeline_path") if model_spec.lineage else None,
                json.dumps(model_spec.lineage) if model_spec.lineage else None,
                model_spec.status.value,
                model_spec.trained_by or "system"
            ))

            conn.commit()

        logger.info(f"Registered model: {model_spec.name} v{model_spec.version} (id: {model_id})")
        return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM registered_models WHERE model_id = ?",
                (model_id,)
            ).fetchone()

        if not row:
            return None

        return dict(row)

    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model by name."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM registered_models WHERE name = ?",
                (name,)
            ).fetchone()

        if not row:
            return None

        return dict(row)

    def get_latest_version(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get latest version of a model."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM model_versions WHERE model_id = ?
                ORDER BY version_id DESC LIMIT 1
            """, (model_id,)).fetchone()

        if not row:
            return None

        return dict(row)

    def get_version(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get specific model version."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM model_versions WHERE model_id = ? AND version = ?
            """, (model_id, version)).fetchone()

        if not row:
            return None

        return dict(row)

    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM model_versions WHERE model_id = ?
                ORDER BY version_id DESC
            """, (model_id,)).fetchall()

        return [dict(row) for row in rows]

    def list_models(
        self,
        model_type: ModelType = None,
        stage: ModelStage = None,
        tags: List[str] = None
    ) -> List[Dict[str, Any]]:
        """List registered models with filters."""
        query = "SELECT * FROM registered_models WHERE 1=1"
        params = []

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type.value)

        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        models = [dict(row) for row in rows]

        # Filter by stage if requested
        if stage:
            filtered = []
            for model in models:
                latest = self.get_latest_version(model["model_id"])
                if latest and latest["stage"] == stage.value:
                    filtered.append(model)
            return filtered

        return models

    def transition_stage(
        self,
        model_id: str,
        version: str,
        to_stage: ModelStage,
        changed_by: str = "system",
        reason: str = ""
    ) -> bool:
        """Transition model version to a new stage."""
        # Validate transition
        valid_transitions = {
            ModelStage.NONE: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.NONE, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED, ModelStage.STAGING],
            ModelStage.ARCHIVED: [ModelStage.NONE]
        }

        current = self.get_version(model_id, version)
        if not current:
            raise ValueError(f"Model version not found: {model_id} v{version}")

        from_stage = ModelStage(current["stage"])

        if to_stage not in valid_transitions.get(from_stage, []):
            raise ValueError(f"Invalid stage transition: {from_stage} -> {to_stage}")

        # If moving to production, archive current production version
        if to_stage == ModelStage.PRODUCTION:
            with self._get_conn() as conn:
                prod_versions = conn.execute("""
                    SELECT version FROM model_versions
                    WHERE model_id = ? AND stage = ? AND version != ?
                """, (model_id, ModelStage.PRODUCTION.value, version)).fetchall()

                for pv in prod_versions:
                    conn.execute("""
                        UPDATE model_versions SET stage = ? WHERE model_id = ? AND version = ?
                    """, (ModelStage.ARCHIVED.value, model_id, pv["version"]))

                    conn.execute("""
                        INSERT INTO model_stage_history
                        (model_id, version, from_stage, to_stage, changed_by, reason)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (model_id, pv["version"], ModelStage.PRODUCTION.value, ModelStage.ARCHIVED.value, changed_by, "Auto-archived on new production deployment"))

        # Update version stage
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE model_versions SET stage = ? WHERE model_id = ? AND version = ?
            """, (to_stage.value, model_id, version))

            conn.execute("""
                INSERT INTO model_stage_history
                (model_id, version, from_stage, to_stage, changed_by, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_id, version, from_stage.value, to_stage.value, changed_by, reason))

            conn.commit()

        logger.info(f"Transitioned {model_id} v{version}: {from_stage} -> {to_stage}")
        return True

    def get_production_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get current production version of a model."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM model_versions WHERE model_id = ? AND stage = ?
            """, (model_id, ModelStage.PRODUCTION.value)).fetchone()

        if not row:
            return None

        return dict(row)

    def get_staging_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get current staging version of a model."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM model_versions WHERE model_id = ? AND stage = ?
            """, (model_id, ModelStage.STAGING.value)).fetchone()

        if not row:
            return None

        return dict(row)

    def record_deployment(
        self,
        model_id: str,
        version: str,
        endpoint_name: str,
        environment: str,
        deployed_by: str,
        config: Dict[str, Any]
    ) -> int:
        """Record a model deployment."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO model_deployments
                (model_id, version, endpoint_name, environment, deployed_by, status, config)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model_id, version, endpoint_name, environment, deployed_by, "active", json.dumps(config)))
            conn.commit()
            return cursor.lastrowid

    def get_deployments(self, model_id: str = None, environment: str = None) -> List[Dict[str, Any]]:
        """Get deployment history."""
        query = "SELECT * FROM model_deployments WHERE 1=1"
        params = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if environment:
            query += " AND environment = ?"
            params.append(environment)

        query += " ORDER BY deployed_at DESC"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def archive_model(self, model_id: str, version: str, reason: str = "") -> bool:
        """Archive a model version."""
        return self.transition_stage(model_id, version, ModelStage.ARCHIVED, reason=reason)

    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete a model (only if no production/staging versions)."""
        with self._get_conn() as conn:
            # Check for active versions
            active = conn.execute("""
                SELECT COUNT(*) as cnt FROM model_versions
                WHERE model_id = ? AND stage IN (?, ?)
            """, (model_id, ModelStage.PRODUCTION.value, ModelStage.STAGING.value)).fetchone()

            if active["cnt"] > 0 and not force:
                raise ValueError(f"Model has active versions. Use force=True to delete.")

            # Delete versions
            conn.execute("DELETE FROM model_versions WHERE model_id = ?", (model_id,))
            conn.execute("DELETE FROM registered_models WHERE model_id = ?", (model_id,))
            conn.commit()

        # Delete artifacts
        model_dir = self.artifact_root / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)

        logger.info(f"Deleted model: {model_id}")
        return True

    def get_stage_history(self, model_id: str, version: str = None) -> List[Dict[str, Any]]:
        """Get stage transition history."""
        query = "SELECT * FROM model_stage_history WHERE model_id = ?"
        params = [model_id]

        if version:
            query += " AND version = ?"
            params.append(version)

        query += " ORDER BY changed_at DESC"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def compare_versions(self, model_id: str, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare two model versions."""
        ver_a = self.get_version(model_id, version_a)
        ver_b = self.get_version(model_id, version_b)

        if not ver_a or not ver_b:
            raise ValueError("One or both versions not found")

        metrics_a = json.loads(ver_a["metrics"]) if ver_a["metrics"] else {}
        metrics_b = json.loads(ver_b["metrics"]) if ver_b["metrics"] else {}

        hp_a = json.loads(ver_a["hyperparameters"]) if ver_a["hyperparameters"] else {}
        hp_b = json.loads(ver_b["hyperparameters"]) if ver_b["hyperparameters"] else {}

        # Compare metrics
        metric_diff = {}
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
        for m in all_metrics:
            a_val = metrics_a.get(m)
            b_val = metrics_b.get(m)
            if a_val is not None and b_val is not None:
                metric_diff[m] = {
                    "version_a": a_val,
                    "version_b": b_val,
                    "diff": b_val - a_val,
                    "pct_change": ((b_val - a_val) / a_val * 100) if a_val != 0 else None
                }

        # Compare hyperparameters
        hp_diff = {}
        all_hp = set(hp_a.keys()) | set(hp_b.keys())
        for h in all_hp:
            if hp_a.get(h) != hp_b.get(h):
                hp_diff[h] = {"version_a": hp_a.get(h), "version_b": hp_b.get(h)}

        return {
            "model_id": model_id,
            "version_a": version_a,
            "version_b": version_b,
            "metrics_comparison": metric_diff,
            "hyperparameters_comparison": hp_diff,
            "stage_a": ver_a["stage"],
            "stage_b": ver_b["stage"]
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    registry = ModelRegistry()

    # Test
    from core.schemas import ModelSpec, ModelType

    model = ModelSpec(
        name="ranker_v1",
        model_type=ModelType.RANKER,
        version="1.0.0",
        description="Initial ranking model",
        framework="xgboost",
        hyperparameters={"max_depth": 6, "learning_rate": 0.1},
        metrics={"auc": 0.82, "logloss": 0.35},
        trained_by="ml-engineer"
    )

    model_id = registry.register_model(model)
    print(f"Registered: {model_id}")

    # Transition to staging
    registry.transition_stage(model_id, "1.0.0", ModelStage.STAGING, reason="Ready for validation")

    # Transition to production
    registry.transition_stage(model_id, "1.0.0", ModelStage.PRODUCTION, reason="Validated in staging")

    print(f"Production model: {registry.get_production_model(model_id)}")
    print(f"Models: {registry.list_models()}")