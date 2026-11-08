"""A/B testing framework for real-time experimentation."""

import sqlite3
import json
import logging
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import numpy as np
from scipy import stats

from core.schemas import (
    ExperimentSpec, ExperimentStatus, ABTestConfig, ModelSpec, ModelType,
    UserSession, ClickEvent, ConversionEvent, MetricSnapshot
)

logger = logging.getLogger(__name__)


class TrafficAllocator:
    """Deterministic traffic allocation using consistent hashing."""

    @staticmethod
    def allocate(user_id: str, experiment_id: str, splits: Dict[str, float]) -> str:
        """Allocate user to variant using consistent hashing."""
        # Create hash from user_id + experiment_id for consistency
        hash_input = f"{user_id}:{experiment_id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        # Normalize to [0, 1)
        percentile = (hash_val % 10000) / 10000.0

        # Allocate based on cumulative splits
        cumulative = 0.0
        for variant, split in splits.items():
            cumulative += split
            if percentile < cumulative:
                return variant

        # Fallback to first variant
        return list(splits.keys())[0]

    @staticmethod
    def allocate_with_ramp(
        user_id: str,
        experiment_id: str,
        splits: Dict[str, float],
        ramp_schedule: Dict[int, float],
        current_day: int
    ) -> str:
        """Allocate with ramp-up schedule."""
        # Get current ramp percentage
        ramp_pct = 1.0
        for day, pct in sorted(ramp_schedule.items()):
            if current_day >= day:
                ramp_pct = pct

        # Apply ramp to treatment variants only
        control_split = splits.get("control", 0.0)
        treatment_split = sum(v for k, v in splits.items() if k != "control")

        # Scale treatment splits by ramp
        adjusted_splits = {"control": control_split + treatment_split * (1 - ramp_pct)}
        for variant, split in splits.items():
            if variant != "control":
                adjusted_splits[variant] = split * ramp_pct

        return TrafficAllocator.allocate(user_id, experiment_id, adjusted_splits)


@dataclass
class ExperimentData:
    """In-memory experiment data for real-time processing."""
    experiment_id: str
    config: ABTestConfig
    variant_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    variant_metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    user_assignments: Dict[str, str] = field(default_factory=dict)  # user_id -> variant
    session_events: Dict[str, List[Dict]] = field(default_factory=lambda: defaultdict(list))  # session_id -> events
    created_at: datetime = field(default_factory=datetime.utcnow)


class ABTestFramework:
    """Real-time A/B testing framework with statistical rigor."""

    def __init__(
        self,
        db_path: str = "data/experiments.db",
        assignment_ttl_days: int = 30
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.assignment_ttl_days = assignment_ttl_days
        self.experiments: Dict[str, ExperimentData] = {}
        self._init_db()

    def _init_db(self):
        """Initialize experiment database."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    hypothesis TEXT,
                    control_model_id TEXT,
                    treatment_model_ids TEXT,
                    traffic_split TEXT,
                    targeting_rules TEXT,
                    metrics TEXT,
                    primary_metric TEXT,
                    min_sample_size INTEGER,
                    max_duration_days INTEGER,
                    significance_level REAL,
                    power REAL,
                    status TEXT DEFAULT 'draft',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    results TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_assignments (
                    assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(experiment_id, user_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_assignments_exp ON experiment_assignments(experiment_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_exp ON experiment_events(experiment_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_user ON experiment_events(user_id)
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

    def create_experiment(self, spec: ExperimentSpec) -> str:
        """Create a new experiment."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO experiments
                (experiment_id, name, description, hypothesis, control_model_id,
                 treatment_model_ids, traffic_split, targeting_rules, metrics,
                 primary_metric, min_sample_size, max_duration_days,
                 significance_level, power, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                spec.experiment_id,
                spec.name,
                spec.description,
                spec.hypothesis,
                spec.control_model.model_id,
                json.dumps([m.model_id for m in spec.treatment_models]),
                json.dumps(spec.traffic_split),
                json.dumps(spec.targeting_rules),
                json.dumps(spec.metrics),
                spec.primary_metric,
                spec.min_sample_size,
                spec.max_duration_days,
                spec.significance_level,
                spec.power,
                spec.status.value
            ))
            conn.commit()

        # Initialize in-memory data
        config = ABTestConfig(
            experiment_id=spec.experiment_id,
            variant_assignments={},
            sticky_sessions=True,
            ramp_up_schedule=spec.ramp_up_schedule if hasattr(spec, 'ramp_up_schedule') else None,
            holdout_percentage=getattr(spec, 'holdout_percentage', 0.0),
            min_sessions_per_variant=getattr(spec, 'min_sessions_per_variant', 100),
            guardrail_metrics=getattr(spec, 'guardrail_metrics', {})
        )

        self.experiments[spec.experiment_id] = ExperimentData(
            experiment_id=spec.experiment_id,
            config=config
        )

        logger.info(f"Created experiment: {spec.name} ({spec.experiment_id})")
        return spec.experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE experiments SET status = ?, started_at = ?
                WHERE experiment_id = ?
            """, (ExperimentStatus.RUNNING.value, datetime.utcnow().isoformat(), experiment_id))
            conn.commit()

        if experiment_id in self.experiments:
            self.experiments[experiment_id].config = ABTestConfig(
                experiment_id=experiment_id
            )

        logger.info(f"Started experiment: {experiment_id}")
        return True

    def stop_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Stop an experiment."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE experiments SET status = ?, ended_at = ?, results = ?
                WHERE experiment_id = ?
            """, (
                ExperimentStatus.STOPPED.value,
                datetime.utcnow().isoformat(),
                json.dumps({"stopped_reason": reason}),
                experiment_id
            ))
            conn.commit()

        logger.info(f"Stopped experiment: {experiment_id} - {reason}")
        return True

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentSpec]:
        """Get experiment specification."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()

        if not row:
            return None

        # Reconstruct ExperimentSpec
        control_model = ModelSpec(
            model_id=row["control_model_id"],
            name=row["control_model_id"],
            model_type=ModelType.RANKER  # Would need to fetch actual
        )

        treatment_models = []
        for mid in json.loads(row["treatment_model_ids"]):
            treatment_models.append(ModelSpec(
                model_id=mid,
                name=mid,
                model_type=ModelType.RANKER
            ))

        return ExperimentSpec(
            experiment_id=row["experiment_id"],
            name=row["name"],
            description=row["description"] or "",
            hypothesis=row["hypothesis"] or "",
            control_model=control_model,
            treatment_models=treatment_models,
            traffic_split=json.loads(row["traffic_split"]),
            targeting_rules=json.loads(row["targeting_rules"]) if row["targeting_rules"] else {},
            metrics=json.loads(row["metrics"]) if row["metrics"] else [],
            primary_metric=row["primary_metric"] or "",
            min_sample_size=row["min_sample_size"],
            max_duration_days=row["max_duration_days"],
            significance_level=row["significance_level"],
            power=row["power"],
            status=ExperimentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            results=json.loads(row["results"]) if row["results"] else {}
        )

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Assign user to experiment variant."""
        context = context or {}

        # Check targeting rules
        exp = self.get_experiment(experiment_id)
        if not exp:
            return "control"

        if exp.status != ExperimentStatus.RUNNING:
            return "control"

        # Check targeting
        if not self._matches_targeting(user_id, context, exp.targeting_rules):
            return "control"

        # Check holdout
        config = self.experiments.get(experiment_id, ExperimentData(
            experiment_id=experiment_id,
            config=ABTestConfig(experiment_id=experiment_id)
        )).config

        if config.holdout_percentage > 0:
            hash_val = int(hashlib.md5(f"{user_id}:{experiment_id}:holdout".encode()).hexdigest(), 16)
            if (hash_val % 100) / 100.0 < config.holdout_percentage:
                return "holdout"

        # Check existing assignment (sticky sessions)
        if config.sticky_sessions:
            with self._get_conn() as conn:
                row = conn.execute("""
                    SELECT variant FROM experiment_assignments
                    WHERE experiment_id = ? AND user_id = ?
                """, (experiment_id, user_id)).fetchone()

            if row:
                return row["variant"]

        # Allocate variant
        splits = exp.traffic_split
        if config.ramp_up_schedule:
            current_day = (datetime.utcnow() - exp.started_at).days if exp.started_at else 0
            variant = TrafficAllocator.allocate_with_ramp(
                user_id, experiment_id, splits, config.ramp_up_schedule, current_day
            )
        else:
            variant = TrafficAllocator.allocate(user_id, experiment_id, splits)

        # Record assignment
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiment_assignments
                (experiment_id, user_id, variant)
                VALUES (?, ?, ?)
            """, (experiment_id, user_id, variant))
            conn.commit()

        # Update in-memory
        if experiment_id in self.experiments:
            self.experiments[experiment_id].variant_counts[variant] += 1
            self.experiments[experiment_id].user_assignments[user_id] = variant

        return variant

    def _matches_targeting(
        self,
        user_id: str,
        context: Dict[str, Any],
        targeting_rules: Dict[str, Any]
    ) -> bool:
        """Check if user matches targeting rules."""
        if not targeting_rules:
            return True

        for key, rule in targeting_rules.items():
            if key not in context:
                return False

            user_value = context[key]

            if isinstance(rule, dict):
                if "in" in rule:
                    if user_value not in rule["in"]:
                        return False
                if "not_in" in rule:
                    if user_value in rule["not_in"]:
                        return False
                if "gte" in rule:
                    if user_value < rule["gte"]:
                        return False
                if "lte" in rule:
                    if user_value > rule["lte"]:
                        return False
            elif isinstance(rule, list):
                if user_value not in rule:
                    return False
            else:
                if user_value != rule:
                    return False

        return True

    def record_event(
        self,
        experiment_id: str,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any] = None,
        session_id: str = None
    ) -> None:
        """Record experiment event (click, conversion, etc.)."""
        variant = self.assign_variant(experiment_id, user_id)

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO experiment_events
                (experiment_id, user_id, variant, session_id, event_type, event_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                user_id,
                variant,
                session_id,
                event_type,
                json.dumps(event_data or {})
            ))
            conn.commit()

        # Update in-memory metrics
        if experiment_id in self.experiments:
            exp_data = self.experiments[experiment_id]
            if isinstance(event_data, dict) and "metric" in event_data:
                metric_name = event_data["metric"]
                metric_value = event_data.get("value", 1.0)
                exp_data.variant_metrics[variant][metric_name].append(metric_value)
            # Note: click and conversion events with metric already handled above

    def record_click(
        self,
        experiment_id: str,
        user_id: str,
        item_id: str,
        position: int,
        score: float,
        session_id: str = None
    ) -> None:
        """Record click event. Uses binary metric: impression=0, click=1."""
        # Record the impression (0)
        self.record_event(
            experiment_id=experiment_id,
            user_id=user_id,
            event_type="impression",
            event_data={"metric": "ctr", "value": 0.0},
            session_id=session_id
        )
        # Record the click (1)
        self.record_event(
            experiment_id=experiment_id,
            user_id=user_id,
            event_type="click",
            event_data={
                "item_id": item_id,
                "position": position,
                "score": score,
                "metric": "ctr",
                "value": 1.0
            },
            session_id=session_id
        )

    def record_impression(
        self,
        experiment_id: str,
        user_id: str,
        session_id: str = None
    ) -> None:
        """Record an impression without click."""
        self.record_event(
            experiment_id=experiment_id,
            user_id=user_id,
            event_type="impression",
            event_data={"metric": "ctr", "value": 0.0},
            session_id=session_id
        )

    def record_conversion(
        self,
        experiment_id: str,
        user_id: str,
        item_id: str,
        conversion_type: str,
        value: float,
        session_id: str = None
    ) -> None:
        """Record conversion event."""
        self.record_event(
            experiment_id=experiment_id,
            user_id=user_id,
            event_type="conversion",
            event_data={
                "item_id": item_id,
                "conversion_type": conversion_type,
                "metric": "cvr",
                "value": value
            },
            session_id=session_id
        )

    def get_variant_stats(self, experiment_id: str) -> Dict[str, Dict[str, Any]]:
        """Get statistics per variant."""
        with self._get_conn() as conn:
            # Get assignments
            assign_rows = conn.execute("""
                SELECT variant, COUNT(*) as count
                FROM experiment_assignments
                WHERE experiment_id = ?
                GROUP BY variant
            """, (experiment_id,)).fetchall()

            # Get events
            event_rows = conn.execute("""
                SELECT variant, event_type, event_data
                FROM experiment_events
                WHERE experiment_id = ?
            """, (experiment_id,)).fetchall()

        stats = {}
        for row in assign_rows:
            variant = row["variant"]
            stats[variant] = {
                "users": row["count"],
                "events": defaultdict(int),
                "metrics": defaultdict(list)
            }

        for row in event_rows:
            variant = row["variant"]
            if variant in stats:
                event_data = json.loads(row["event_data"]) if row["event_data"] else {}
                stats[variant]["events"][row["event_type"]] += 1

                if "metric" in event_data:
                    metric_name = event_data["metric"]
                    metric_value = event_data.get("value", 1.0)
                    stats[variant]["metrics"][metric_name].append(metric_value)

        # Compute aggregate metrics
        for variant, data in stats.items():
            for metric_name, values in data["metrics"].items():
                if values:
                    data[f"{metric_name}_mean"] = np.mean(values)
                    data[f"{metric_name}_sum"] = np.sum(values)
                    data[f"{metric_name}_count"] = len(values)

        return stats

    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results with statistical tests."""
        from experimentation.analysis import ExperimentAnalyzer

        analyzer = ExperimentAnalyzer()
        return analyzer.analyze(self.get_experiment(experiment_id), self.get_variant_stats(experiment_id))

    def check_guardrails(self, experiment_id: str) -> Dict[str, Any]:
        """Check guardrail metrics."""
        from experimentation.guardrails import GuardrailMonitor

        monitor = GuardrailMonitor()
        exp = self.get_experiment(experiment_id)
        config = self.experiments.get(experiment_id, ExperimentData(
            experiment_id=experiment_id,
            config=ABTestConfig(experiment_id=experiment_id)
        )).config

        return monitor.check_guardrails(exp, self.get_variant_stats(experiment_id), config.guardrail_metrics)

    def should_stop_early(self, experiment_id: str) -> Tuple[bool, str]:
        """Check if experiment should stop early (sequential testing)."""
        from experimentation.sequential import SequentialTester

        tester = SequentialTester()
        exp = self.get_experiment(experiment_id)

        return tester.should_stop(exp, self.get_variant_stats(experiment_id))

    def list_experiments(
        self,
        status: ExperimentStatus = None
    ) -> List[Dict[str, Any]]:
        """List experiments."""
        query = "SELECT * FROM experiments"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def export_results(self, experiment_id: str) -> Dict[str, Any]:
        """Export complete experiment results."""
        exp = self.get_experiment(experiment_id)
        stats = self.get_variant_stats(experiment_id)
        analysis = self.analyze_experiment(experiment_id)
        guardrails = self.check_guardrails(experiment_id)

        return {
            "experiment": exp.model_dump() if exp else None,
            "variant_stats": stats,
            "analysis": analysis,
            "guardrails": guardrails,
            "exported_at": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    framework = ABTestFramework()

    # Create experiment
    from core.schemas import ModelSpec, ModelType

    control = ModelSpec(name="control", model_id="model_v1", model_type=ModelType.RANKER)
    treatment = ModelSpec(name="treatment", model_id="model_v2", model_type=ModelType.RANKER)

    exp = ExperimentSpec(
        name="ranking_test_v1",
        hypothesis="New ranker improves CTR by 2%",
        control_model=control,
        treatment_models=[treatment],
        traffic_split={"control": 0.5, "treatment": 0.5},
        primary_metric="ctr",
        metrics=["ctr", "cvr", "revenue"],
        min_sample_size=1000,
        max_duration_days=14,
        significance_level=0.05,
        power=0.8
    )

    exp_id = framework.create_experiment(exp)
    framework.start_experiment(exp_id)

    # Simulate assignments
    for i in range(1000):
        user_id = f"user_{i}"
        variant = framework.assign_variant(exp_id, user_id)

    # Simulate events
    for i in range(1000):
        user_id = f"user_{i}"
        if np.random.random() < 0.1:  # 10% click rate
            framework.record_click(exp_id, user_id, f"item_{i}", 1, 0.8)
        if np.random.random() < 0.02:  # 2% conversion rate
            framework.record_conversion(exp_id, user_id, f"item_{i}", "purchase", 10.0)

    stats = framework.get_variant_stats(exp_id)
    print(f"Variant stats: {stats}")

    analysis = framework.analyze_experiment(exp_id)
    print(f"Analysis: {analysis}")