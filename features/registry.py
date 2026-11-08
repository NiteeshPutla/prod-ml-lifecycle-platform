"""Feature registry for metadata management and discovery."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from core.schemas import FeatureSpec, FeatureType, FeatureSource

logger = logging.getLogger(__name__)


@dataclass
class FeatureGroup:
    """A group of related features."""
    name: str
    description: str
    entity_type: str
    feature_names: List[str]
    owner: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime


class FeatureRegistry:
    """Central registry for feature metadata, discovery, and governance."""

    def __init__(self, db_path: str = "data/feature_registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize registry database."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    name TEXT PRIMARY KEY,
                    feature_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT,
                    default_value TEXT,
                    embedding_dim INTEGER,
                    vocabulary_size INTEGER,
                    is_key BOOLEAN DEFAULT 0,
                    is_target BOOLEAN DEFAULT 0,
                    freshness_seconds INTEGER,
                    transformation TEXT,
                    tags TEXT,
                    owner TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_groups (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    entity_type TEXT NOT NULL,
                    feature_names TEXT NOT NULL,
                    owner TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_dependencies (
                    feature_name TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    dependency_type TEXT DEFAULT 'requires',
                    PRIMARY KEY (feature_name, depends_on)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_lineage (
                    lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    upstream_features TEXT,
                    transformation_logic TEXT,
                    data_source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT
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

    def register_feature(self, feature: FeatureSpec, owner: str = "system") -> None:
        """Register a feature in the registry."""
        with self._get_conn() as conn:
            # Handle both enum objects and string values (Pydantic v2 use_enum_values=True stores strings)
            feature_type_val = feature.feature_type.value if hasattr(feature.feature_type, 'value') else feature.feature_type
            source_val = feature.source.value if hasattr(feature.source, 'value') else feature.source

            conn.execute("""
                INSERT OR REPLACE INTO features
                (name, feature_type, source, description, default_value, embedding_dim,
                 vocabulary_size, is_key, is_target, freshness_seconds, transformation,
                 tags, owner, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feature.name,
                feature_type_val,
                source_val,
                feature.description,
                json.dumps(feature.default_value) if feature.default_value is not None else None,
                feature.embedding_dim,
                feature.vocabulary_size,
                feature.is_key,
                feature.is_target,
                feature.freshness_seconds,
                feature.transformation,
                json.dumps(feature.tags),
                owner,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        logger.info(f"Registered feature in registry: {feature.name}")

    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """Get feature by name."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM features WHERE name = ?", (name,)).fetchone()

        if not row:
            return None

        return FeatureSpec(
            name=row["name"],
            feature_type=FeatureType(row["feature_type"]),
            source=FeatureSource(row["source"]),
            description=row["description"] or "",
            default_value=json.loads(row["default_value"]) if row["default_value"] else None,
            embedding_dim=row["embedding_dim"],
            vocabulary_size=row["vocabulary_size"],
            is_key=bool(row["is_key"]),
            is_target=bool(row["is_target"]),
            freshness_seconds=row["freshness_seconds"],
            transformation=row["transformation"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
        )

    def list_features(
        self,
        feature_type: FeatureType = None,
        source: FeatureSource = None,
        entity_type: str = None,
        tags: List[str] = None,
        status: str = "active"
    ) -> List[FeatureSpec]:
        """List features with filters."""
        query = "SELECT * FROM features WHERE status = ?"
        params = [status]

        if feature_type:
            query += " AND feature_type = ?"
            ft_val = feature_type.value if hasattr(feature_type, 'value') else feature_type
            params.append(ft_val)
        if source:
            query += " AND source = ?"
            src_val = source.value if hasattr(source, 'value') else source
            params.append(src_val)
        if tags:
            # Simple tag matching - in production use JSON functions
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self.get_feature(row["name"]) for row in rows]

    def create_feature_group(
        self,
        name: str,
        description: str,
        entity_type: str,
        feature_names: List[str],
        owner: str = "system",
        tags: List[str] = None
    ) -> FeatureGroup:
        """Create a feature group."""
        group = FeatureGroup(
            name=name,
            description=description,
            entity_type=entity_type,
            feature_names=feature_names,
            owner=owner,
            tags=tags or [],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_groups
                (name, description, entity_type, feature_names, owner, tags, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                group.name,
                group.description,
                group.entity_type,
                json.dumps(group.feature_names),
                group.owner,
                json.dumps(group.tags),
                group.updated_at.isoformat()
            ))
            conn.commit()

        return group

    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """Get feature group by name."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM feature_groups WHERE name = ?", (name,)).fetchone()

        if not row:
            return None

        return FeatureGroup(
            name=row["name"],
            description=row["description"] or "",
            entity_type=row["entity_type"],
            feature_names=json.loads(row["feature_names"]),
            owner=row["owner"] or "",
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    def add_dependency(self, feature_name: str, depends_on: str, dependency_type: str = "requires") -> None:
        """Add feature dependency."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_dependencies
                (feature_name, depends_on, dependency_type)
                VALUES (?, ?, ?)
            """, (feature_name, depends_on, dependency_type))
            conn.commit()

    def get_dependencies(self, feature_name: str) -> List[str]:
        """Get upstream dependencies for a feature."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT depends_on FROM feature_dependencies WHERE feature_name = ?",
                (feature_name,)
            ).fetchall()
        return [row["depends_on"] for row in rows]

    def get_dependents(self, feature_name: str) -> List[str]:
        """Get downstream features that depend on this feature."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT feature_name FROM feature_dependencies WHERE depends_on = ?",
                (feature_name,)
            ).fetchall()
        return [row["feature_name"] for row in rows]

    def record_lineage(
        self,
        feature_name: str,
        upstream_features: List[str],
        transformation_logic: str,
        data_source: str,
        created_by: str = "system"
    ) -> None:
        """Record feature lineage."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO feature_lineage
                (feature_name, upstream_features, transformation_logic, data_source, created_by)
                VALUES (?, ?, ?, ?, ?)
            """, (
                feature_name,
                json.dumps(upstream_features),
                transformation_logic,
                data_source,
                created_by
            ))
            conn.commit()

    def get_lineage(self, feature_name: str) -> List[Dict[str, Any]]:
        """Get feature lineage history."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feature_lineage WHERE feature_name = ? ORDER BY created_at DESC",
                (feature_name,)
            ).fetchall()

        return [
            {
                "lineage_id": row["lineage_id"],
                "upstream_features": json.loads(row["upstream_features"]),
                "transformation_logic": row["transformation_logic"],
                "data_source": row["data_source"],
                "created_at": row["created_at"],
                "created_by": row["created_by"]
            }
            for row in rows
        ]

    def search_features(self, query: str) -> List[FeatureSpec]:
        """Search features by name or description."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM features
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY name
            """, (f"%{query}%", f"%{query}%")).fetchall()

        return [self.get_feature(row["name"]) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM features").fetchone()["cnt"]
            by_type = conn.execute("""
                SELECT feature_type, COUNT(*) as cnt FROM features GROUP BY feature_type
            """).fetchall()
            by_source = conn.execute("""
                SELECT source, COUNT(*) as cnt FROM features GROUP BY source
            """).fetchall()
            groups = conn.execute("SELECT COUNT(*) as cnt FROM feature_groups").fetchone()["cnt"]

        return {
            "total_features": total,
            "by_type": {row["feature_type"]: row["cnt"] for row in by_type},
            "by_source": {row["source"]: row["cnt"] for row in by_source},
            "feature_groups": groups
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    registry = FeatureRegistry()

    # Test
    from core.schemas import FeatureSpec, FeatureType, FeatureSource

    feature = FeatureSpec(
        name="user_age",
        feature_type=FeatureType.NUMERICAL,
        source=FeatureSource.OFFLINE,
        description="User age in years",
        transformation="normalize",
        tags=["demographics", "user"]
    )

    registry.register_feature(feature, owner="ml-team")

    retrieved = registry.get_feature("user_age")
    print(f"Retrieved: {retrieved}")

    registry.create_feature_group(
        name="user_demographics",
        description="User demographic features",
        entity_type="user",
        feature_names=["user_age", "user_gender", "user_country"],
        owner="ml-team",
        tags=["demographics"]
    )

    print(f"Stats: {registry.get_stats()}")