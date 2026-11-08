"""Feature lineage tracking for reproducibility and governance."""

import sqlite3
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class LineageNodeType(str, Enum):
    DATA_SOURCE = "data_source"
    FEATURE = "feature"
    TRANSFORMATION = "transformation"
    MODEL = "model"
    EXPERIMENT = "experiment"


class LineageEdgeType(str, Enum):
    DERIVED_FROM = "derived_from"
    TRANSFORMED_BY = "transformed_by"
    USED_IN = "used_in"
    TRAINED_ON = "trained_on"
    EVALUATED_ON = "evaluated_on"


@dataclass
class LineageNode:
    """A node in the lineage graph."""
    node_id: str
    node_type: LineageNodeType
    name: str
    description: str = ""
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    created_by: str = "system"
    version: str = "1.0.0"
    checksum: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        node_type_val = self.node_type.value if hasattr(self.node_type, 'value') else self.node_type
        content = f"{node_type_val}:{self.name}:{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class LineageEdge:
    """An edge in the lineage graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: LineageEdgeType
    metadata: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class LineageTracker:
    """Track feature lineage for reproducibility and governance."""

    def __init__(self, db_path: str = "data/lineage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize lineage database."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    version TEXT DEFAULT '1.0.0',
                    checksum TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_node_id) REFERENCES lineage_nodes(node_id),
                    FOREIGN KEY (target_node_id) REFERENCES lineage_nodes(node_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_source ON lineage_edges(source_node_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_target ON lineage_edges(target_node_id)
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

    def add_node(self, node: LineageNode) -> str:
        """Add a lineage node."""
        node_type_val = node.node_type.value if hasattr(node.node_type, 'value') else node.node_type
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lineage_nodes
                (node_id, node_type, name, description, metadata, created_at, created_by, version, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.node_id,
                node_type_val,
                node.name,
                node.description,
                json.dumps(node.metadata),
                node.created_at.isoformat(),
                node.created_by,
                node.version,
                node.checksum
            ))
            conn.commit()
        return node.node_id

    def add_edge(self, edge: LineageEdge) -> str:
        """Add a lineage edge."""
        edge_type_val = edge.edge_type.value if hasattr(edge.edge_type, 'value') else edge.edge_type
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lineage_edges
                (edge_id, source_node_id, target_node_id, edge_type, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge.edge_id,
                edge.source_node_id,
                edge.target_node_id,
                edge_type_val,
                json.dumps(edge.metadata),
                edge.created_at.isoformat()
            ))
            conn.commit()
        return edge.edge_id

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get a lineage node by ID."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM lineage_nodes WHERE node_id = ?", (node_id,)).fetchone()

        if not row:
            return None

        return LineageNode(
            node_id=row["node_id"],
            node_type=LineageNodeType(row["node_type"]),
            name=row["name"],
            description=row["description"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=row["created_by"] or "system",
            version=row["version"] or "1.0.0",
            checksum=row["checksum"] or ""
        )

    def get_nodes_by_type(self, node_type: LineageNodeType) -> List[LineageNode]:
        """Get all nodes of a given type."""
        nt_val = node_type.value if hasattr(node_type, 'value') else node_type
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM lineage_nodes WHERE node_type = ?", (nt_val,)
            ).fetchall()

        return [
            LineageNode(
                node_id=row["node_id"],
                node_type=LineageNodeType(row["node_type"]),
                name=row["name"],
                description=row["description"] or "",
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
                created_by=row["created_by"] or "system",
                version=row["version"] or "1.0.0",
                checksum=row["checksum"] or ""
            )
            for row in rows
        ]

    def get_upstream(self, node_id: str, max_depth: int = 10) -> List[LineageNode]:
        """Get all upstream dependencies of a node."""
        visited = set()
        result = []

        def dfs(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)

            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT ln.* FROM lineage_nodes ln
                    JOIN lineage_edges le ON ln.node_id = le.source_node_id
                    WHERE le.target_node_id = ?
                """, (current_id,)).fetchall()

            for row in rows:
                node = LineageNode(
                    node_id=row["node_id"],
                    node_type=LineageNodeType(row["node_type"]),
                    name=row["name"],
                    description=row["description"] or "",
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=datetime.fromisoformat(row["created_at"]),
                    created_by=row["created_by"] or "system",
                    version=row["version"] or "1.0.0",
                    checksum=row["checksum"] or ""
                )
                result.append(node)
                dfs(node.node_id, depth + 1)

        dfs(node_id, 0)
        return result

    def get_downstream(self, node_id: str, max_depth: int = 10) -> List[LineageNode]:
        """Get all downstream dependents of a node."""
        visited = set()
        result = []

        def dfs(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)

            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT ln.* FROM lineage_nodes ln
                    JOIN lineage_edges le ON ln.node_id = le.target_node_id
                    WHERE le.source_node_id = ?
                """, (current_id,)).fetchall()

            for row in rows:
                node = LineageNode(
                    node_id=row["node_id"],
                    node_type=LineageNodeType(row["node_type"]),
                    name=row["name"],
                    description=row["description"] or "",
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=datetime.fromisoformat(row["created_at"]),
                    created_by=row["created_by"] or "system",
                    version=row["version"] or "1.0.0",
                    checksum=row["checksum"] or ""
                )
                result.append(node)
                dfs(node.node_id, depth + 1)

        dfs(node_id, 0)
        return result

    def get_full_lineage(self, node_id: str) -> Dict[str, List[LineageNode]]:
        """Get complete upstream and downstream lineage."""
        return {
            "upstream": self.get_upstream(node_id),
            "downstream": self.get_downstream(node_id)
        }

    def track_data_source(
        self,
        name: str,
        source_type: str,  # database, file, api, stream
        connection_info: Dict[str, Any],
        schema: Dict[str, Any],
        created_by: str = "system"
    ) -> str:
        """Track a data source."""
        node = LineageNode(
            node_id=f"datasource_{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            node_type=LineageNodeType.DATA_SOURCE,
            name=name,
            description=f"Data source: {source_type}",
            metadata={
                "source_type": source_type,
                "connection_info": connection_info,
                "schema": schema
            },
            created_by=created_by
        )
        return self.add_node(node)

    def track_feature(
        self,
        name: str,
        feature_type: str,
        source_node_id: str,
        transformation_logic: str = "",
        parameters: Dict[str, Any] = None,
        created_by: str = "system"
    ) -> str:
        """Track a derived feature."""
        # Create transformation node
        transform_node = LineageNode(
            node_id=f"transform_{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            node_type=LineageNodeType.TRANSFORMATION,
            name=f"transform_{name}",
            description=f"Transformation for feature {name}",
            metadata={
                "transformation_logic": transformation_logic,
                "parameters": parameters or {}
            },
            created_by=created_by
        )
        self.add_node(transform_node)

        # Link source to transformation
        self.add_edge(LineageEdge(
            edge_id=f"edge_{source_node_id}_{transform_node.node_id}",
            source_node_id=source_node_id,
            target_node_id=transform_node.node_id,
            edge_type=LineageEdgeType.TRANSFORMED_BY
        ))

        # Create feature node
        feature_node = LineageNode(
            node_id=f"feature_{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            node_type=LineageNodeType.FEATURE,
            name=name,
            description=f"Feature: {name} ({feature_type})",
            metadata={
                "feature_type": feature_type,
                "transformation_node": transform_node.node_id
            },
            created_by=created_by
        )
        self.add_node(feature_node)

        # Link transformation to feature
        self.add_edge(LineageEdge(
            edge_id=f"edge_{transform_node.node_id}_{feature_node.node_id}",
            source_node_id=transform_node.node_id,
            target_node_id=feature_node.node_id,
            edge_type=LineageEdgeType.DERIVED_FROM
        ))

        return feature_node.node_id

    def track_model(
        self,
        name: str,
        model_type: str,
        feature_node_ids: List[str],
        training_data_node_id: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        created_by: str = "system"
    ) -> str:
        """Track a trained model."""
        node = LineageNode(
            node_id=f"model_{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            node_type=LineageNodeType.MODEL,
            name=name,
            description=f"Model: {name} ({model_type})",
            metadata={
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "metrics": metrics,
                "feature_count": len(feature_node_ids)
            },
            created_by=created_by
        )
        self.add_node(node)

        # Link features to model
        for fid in feature_node_ids:
            self.add_edge(LineageEdge(
                edge_id=f"edge_{fid}_{node.node_id}",
                source_node_id=fid,
                target_node_id=node.node_id,
                edge_type=LineageEdgeType.USED_IN
            ))

        # Link training data to model
        self.add_edge(LineageEdge(
            edge_id=f"edge_{training_data_node_id}_{node.node_id}",
            source_node_id=training_data_node_id,
            target_node_id=node.node_id,
            edge_type=LineageEdgeType.TRAINED_ON
        ))

        return node.node_id

    def track_experiment(
        self,
        name: str,
        model_node_ids: List[str],
        experiment_config: Dict[str, Any],
        results: Dict[str, Any],
        created_by: str = "system"
    ) -> str:
        """Track an experiment."""
        node = LineageNode(
            node_id=f"experiment_{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            node_type=LineageNodeType.EXPERIMENT,
            name=name,
            description=f"Experiment: {name}",
            metadata={
                "config": experiment_config,
                "results": results,
                "model_count": len(model_node_ids)
            },
            created_by=created_by
        )
        self.add_node(node)

        # Link models to experiment
        for mid in model_node_ids:
            self.add_edge(LineageEdge(
                edge_id=f"edge_{mid}_{node.node_id}",
                source_node_id=mid,
                target_node_id=node.node_id,
                edge_type=LineageEdgeType.EVALUATED_ON
            ))

        return node.node_id

    def export_lineage_graph(self, node_id: str) -> Dict[str, Any]:
        """Export lineage as a graph for visualization."""
        lineage = self.get_full_lineage(node_id)

        nodes = [self.get_node(node_id)] + lineage["upstream"] + lineage["downstream"]
        node_ids = {n.node_id for n in nodes}

        edges = []
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM lineage_edges").fetchall()
            for row in rows:
                if row["source_node_id"] in node_ids and row["target_node_id"] in node_ids:
                    edges.append({
                        "edge_id": row["edge_id"],
                        "source": row["source_node_id"],
                        "target": row["target_node_id"],
                        "type": row["edge_type"]
                    })

        return {
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type.value if hasattr(n.node_type, 'value') else n.node_type,
                    "name": n.name,
                    "description": n.description,
                    "metadata": n.metadata,
                    "created_at": n.created_at.isoformat(),
                    "created_by": n.created_by,
                    "version": n.version
                }
                for n in nodes
            ],
            "edges": edges
        }

    def check_reproducibility(self, node_id: str) -> Dict[str, Any]:
        """Check if a node's lineage is fully reproducible."""
        lineage = self.get_full_lineage(node_id)
        all_nodes = [self.get_node(node_id)] + lineage["upstream"]

        issues = []
        for node in all_nodes:
            # Check if node has required metadata
            if node.node_type == LineageNodeType.DATA_SOURCE:
                if not node.metadata.get("schema"):
                    issues.append(f"Data source {node.name} missing schema")
                if not node.metadata.get("connection_info"):
                    issues.append(f"Data source {node.name} missing connection info")
            elif node.node_type == LineageNodeType.TRANSFORMATION:
                if not node.metadata.get("transformation_logic"):
                    issues.append(f"Transformation {node.name} missing logic")
            elif node.node_type == LineageNodeType.FEATURE:
                if not node.metadata.get("feature_type"):
                    issues.append(f"Feature {node.name} missing type")
            elif node.node_type == LineageNodeType.MODEL:
                if not node.metadata.get("hyperparameters"):
                    issues.append(f"Model {node.name} missing hyperparameters")

        return {
            "reproducible": len(issues) == 0,
            "issues": issues,
            "total_nodes": len(all_nodes),
            "node_types": {(n.node_type.value if hasattr(n.node_type, 'value') else n.node_type): sum(1 for n in all_nodes if n.node_type == n.node_type) for n in all_nodes}
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tracker = LineageTracker()

    # Track a data source
    ds_id = tracker.track_data_source(
        name="user_events",
        source_type="database",
        connection_info={"host": "localhost", "database": "events", "table": "user_clicks"},
        schema={"user_id": "string", "item_id": "string", "timestamp": "datetime", "clicked": "int"},
        created_by="data-engineer"
    )
    print(f"Data source: {ds_id}")

    # Track features derived from it
    feat1_id = tracker.track_feature(
        name="user_ctr_7d",
        feature_type="numerical",
        source_node_id=ds_id,
        transformation_logic="SELECT user_id, AVG(clicked) FROM user_clicks WHERE timestamp > NOW() - INTERVAL 7 DAY GROUP BY user_id",
        parameters={"window_days": 7},
        created_by="ml-engineer"
    )
    print(f"Feature 1: {feat1_id}")

    feat2_id = tracker.track_feature(
        name="item_ctr_7d",
        feature_type="numerical",
        source_node_id=ds_id,
        transformation_logic="SELECT item_id, AVG(clicked) FROM user_clicks WHERE timestamp > NOW() - INTERVAL 7 DAY GROUP BY item_id",
        parameters={"window_days": 7},
        created_by="ml-engineer"
    )
    print(f"Feature 2: {feat2_id}")

    # Track model
    model_id = tracker.track_model(
        name="ranker_v1",
        model_type="xgboost",
        feature_node_ids=[feat1_id, feat2_id],
        training_data_node_id=ds_id,
        hyperparameters={"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100},
        metrics={"auc": 0.82, "logloss": 0.35},
        created_by="ml-engineer"
    )
    print(f"Model: {model_id}")

    # Check reproducibility
    repro = tracker.check_reproducibility(model_id)
    print(f"Reproducibility: {repro}")

    # Export graph
    graph = tracker.export_lineage_graph(model_id)
    print(f"Graph nodes: {len(graph['nodes'])}, edges: {len(graph['edges'])}")