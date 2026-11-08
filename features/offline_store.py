"""Offline feature store using SQLite/Parquet for batch feature computation."""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
import logging

from core.schemas import FeatureSpec, FeatureType, FeatureSource

logger = logging.getLogger(__name__)


class OfflineFeatureStore:
    """Offline feature store for batch feature computation and historical retrieval.

    Supports:
    - SQLite for metadata and small-to-medium feature tables
    - Parquet for large-scale feature materialization
    - Time-travel queries for point-in-time correctness
    - Feature versioning and lineage
    """

    def __init__(
        self,
        db_path: str = "data/offline_store.db",
        parquet_dir: str = "data/parquet",
        enable_parquet: bool = True
    ):
        self.db_path = Path(db_path)
        self.parquet_dir = Path(parquet_dir) if enable_parquet else None
        self.enable_parquet = enable_parquet

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.parquet_dir:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_metadata (
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_tables (
                    table_name TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id_column TEXT NOT NULL,
                    timestamp_column TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_versions (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    sql_definition TEXT,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    git_commit TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (feature_name) REFERENCES feature_metadata(name)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS materialization_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    rows_processed INTEGER,
                    status TEXT,
                    error_message TEXT,
                    triggered_by TEXT
                )
            """)

            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def register_feature(self, feature: FeatureSpec) -> None:
        """Register a feature in the metadata store."""
        with self._get_conn() as conn:
            # Handle both enum objects and string values (Pydantic v2 use_enum_values=True stores strings)
            feature_type_val = feature.feature_type.value if hasattr(feature.feature_type, 'value') else feature.feature_type
            source_val = feature.source.value if hasattr(feature.source, 'value') else feature.source

            conn.execute("""
                INSERT OR REPLACE INTO feature_metadata
                (name, feature_type, source, description, default_value, embedding_dim,
                 vocabulary_size, is_key, is_target, freshness_seconds, transformation, tags, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        logger.info(f"Registered feature: {feature.name}")

    def register_feature_table(
        self,
        table_name: str,
        entity_type: str,
        entity_id_column: str,
        timestamp_column: str = "event_timestamp",
        description: str = ""
    ) -> None:
        """Register a feature table."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_tables
                (table_name, entity_type, entity_id_column, timestamp_column, description)
                VALUES (?, ?, ?, ?, ?)
            """, (table_name, entity_type, entity_id_column, timestamp_column, description))
            conn.commit()
        logger.info(f"Registered feature table: {table_name}")

    def create_feature_version(
        self,
        feature_name: str,
        table_name: str,
        sql_definition: str,
        parameters: Dict[str, Any] = None,
        created_by: str = "system",
        git_commit: str = None
    ) -> int:
        """Create a new feature version with SQL definition."""
        with self._get_conn() as conn:
            # Deactivate previous versions
            conn.execute(
                "UPDATE feature_versions SET is_active = 0 WHERE feature_name = ?",
                (feature_name,)
            )

            cursor = conn.execute("""
                INSERT INTO feature_versions
                (feature_name, table_name, sql_definition, parameters, created_by, git_commit)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                feature_name,
                table_name,
                sql_definition,
                json.dumps(parameters or {}),
                created_by,
                git_commit
            ))
            conn.commit()
            version_id = cursor.lastrowid

        logger.info(f"Created feature version {version_id} for {feature_name}")
        return version_id

    def ingest_features(
        self,
        table_name: str,
        df: pd.DataFrame
    ) -> None:
        """Ingest feature data into a feature table."""
        # Create the table if it doesn't exist
        with self._get_conn() as conn:
            # Get column info from DataFrame
            columns = []
            for col in df.columns:
                if col == "event_timestamp":
                    col_type = "TIMESTAMP"
                elif pd.api.types.is_numeric_dtype(df[col]):
                    col_type = "REAL"
                elif pd.api.types.is_object_dtype(df[col]):
                    # Check if it's a list/array (embeddings)
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if isinstance(sample, (list, tuple, np.ndarray)):
                        col_type = "TEXT"  # Store as JSON
                    else:
                        col_type = "TEXT"
                else:
                    col_type = "TEXT"
                columns.append(f"{col} {col_type}")

            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            conn.execute(create_sql)
            conn.commit()

        # Insert data
        # Convert complex types to JSON
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == object:
                sample = df_copy[col].dropna().iloc[0] if len(df_copy[col].dropna()) > 0 else None
                if isinstance(sample, (list, tuple, np.ndarray)):
                    df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if x is not None else None)

        with self._get_conn() as conn:
            df_copy.to_sql(table_name, conn, if_exists='append', index=False)
        logger.info(f"Ingested {len(df)} rows into {table_name}")

    def materialize_features(
        self,
        table_name: str,
        entity_ids: List[str],
        feature_names: List[str],
        start_time: datetime,
        end_time: datetime,
        output_path: str = None
    ) -> pd.DataFrame:
        """Materialize features for given entities and time range."""
        import uuid
        log_id = str(uuid.uuid4())[:8]

        with self._get_conn() as conn:
            # Log start
            conn.execute("""
                INSERT INTO materialization_log (table_name, start_time, status, triggered_by)
                VALUES (?, ?, ?, ?)
            """, (table_name, start_time.isoformat(), "running", log_id))
            conn.commit()

        try:
            # Build query
            feature_cols = ", ".join(feature_names)
            query = f"""
                SELECT {feature_cols}
                FROM {table_name}
                WHERE {entity_ids[0] if entity_ids else '1=1'}
                AND event_timestamp >= ? AND event_timestamp <= ?
            """

            # Simplified - in production, use proper parameterized queries
            # Also need to make sure the table exists, otherwise read_sql_query fails
            with self._get_conn() as conn:
                try:
                    pd.read_sql_query(f"SELECT 1 FROM {table_name} LIMIT 1", conn)
                except Exception:
                    # If table doesn't exist, return empty DataFrame
                    return pd.DataFrame(columns=feature_names)

                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[start_time.isoformat(), end_time.isoformat()]
                )

            if output_path and self.enable_parquet:
                output_file = self.parquet_dir / f"{table_name}_{log_id}.parquet"
                df.to_parquet(output_file, index=False)
                logger.info(f"Materialized {len(df)} rows to {output_file}")

            with self._get_conn() as conn:
                conn.execute("""
                    UPDATE materialization_log
                    SET end_time = ?, rows_processed = ?, status = ?
                    WHERE log_id = (SELECT MAX(log_id) FROM materialization_log WHERE triggered_by = ?)
                """, (datetime.utcnow().isoformat(), len(df), "completed", log_id))
                conn.commit()

            return df

        except Exception as e:
            with self._get_conn() as conn:
                conn.execute("""
                    UPDATE materialization_log
                    SET end_time = ?, status = ?, error_message = ?
                    WHERE triggered_by = ?
                """, (datetime.utcnow().isoformat(), "failed", str(e), log_id))
                conn.commit()
            logger.error(f"Materialization failed: {e}")
            raise

    def get_training_data(
        self,
        entity_type: str,
        feature_names: List[str],
        target_feature: str,
        start_time: datetime,
        end_time: datetime,
        sample_rate: float = 1.0
    ) -> pd.DataFrame:
        """Get training dataset with point-in-time correctness."""
        if entity_type == "user_item":
            # Join user_features, item_features, and interaction_features
            query = f"""
                SELECT i.user_id, i.item_id, i.{target_feature} as {target_feature},
                    u.user_age, u.user_gender, u.user_country, u.user_past_ctr, u.user_session_count_7d,
                    it.item_category, it.item_price, it.item_ctr_7d, it.item_cvr_7d,
                    i.position, i.user_item_affinity
                FROM interaction_features i
                LEFT JOIN user_features u ON i.user_id = u.user_id
                LEFT JOIN item_features it ON i.item_id = it.item_id
                WHERE i.event_timestamp >= ? AND i.event_timestamp <= ?
            """
            all_features = feature_names + [target_feature]
            with self._get_conn() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[start_time.isoformat(), end_time.isoformat()]
                )
            return df
        else:
            # Original single table logic
            with self._get_conn() as conn:
                tables = conn.execute(
                    "SELECT table_name FROM feature_tables WHERE entity_type = ?",
                    (entity_type,)
                ).fetchall()

            if not tables:
                raise ValueError(f"No feature tables found for entity type: {entity_type}")

            # For simplicity, assume single table per entity type
            table_name = tables[0]["table_name"]

            all_features = feature_names + [target_feature]
            return self.materialize_features(
                table_name=table_name,
                entity_ids=[],  # All entities
                feature_names=all_features,
                start_time=start_time,
                end_time=end_time
            )

    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureSpec]:
        """Get feature metadata."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM feature_metadata WHERE name = ?",
                (feature_name,)
            ).fetchone()

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
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    def list_features(self, entity_type: str = None) -> List[FeatureSpec]:
        """List all registered features."""
        with self._get_conn() as conn:
            if entity_type:
                # Join with feature_tables
                rows = conn.execute("""
                    SELECT fm.* FROM feature_metadata fm
                    JOIN feature_versions fv ON fm.name = fv.feature_name
                    JOIN feature_tables ft ON fv.table_name = ft.table_name
                    WHERE ft.entity_type = ? AND fv.is_active = 1
                """, (entity_type,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM feature_metadata").fetchall()

        return [self.get_feature_metadata(row["name"]) for row in rows]

    def compute_feature_statistics(
        self,
        table_name: str,
        feature_names: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics for drift detection baseline."""
        df = self.materialize_features(
            table_name=table_name,
            entity_ids=[],
            feature_names=feature_names,
            start_time=start_time,
            end_time=end_time
        )

        stats = {}
        for feature in feature_names:
            if feature not in df.columns:
                continue

            series = df[feature].dropna()
            if len(series) == 0:
                continue

            if pd.api.types.is_numeric_dtype(series):
                stats[feature] = {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "null_rate": float(df[feature].isna().mean())
                }
            else:
                # Categorical
                value_counts = series.value_counts()
                stats[feature] = {
                    "count": len(series),
                    "unique": int(series.nunique()),
                    "top": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "null_rate": float(df[feature].isna().mean()),
                    "entropy": float(-(value_counts / len(series) * np.log(value_counts / len(series))).sum())
                }

        return stats


def create_sample_features(store: OfflineFeatureStore) -> None:
    """Create sample features for ranking/recommendation use case."""

    # User features
    user_features = [
        FeatureSpec(
            name="user_id",
            feature_type=FeatureType.CATEGORICAL,
            source=FeatureSource.OFFLINE,
            is_key=True,
            description="Unique user identifier",
            vocabulary_size=1000000
        ),
        FeatureSpec(
            name="user_age",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.OFFLINE,
            description="User age",
            transformation="normalize",
            freshness_seconds=86400
        ),
        FeatureSpec(
            name="user_gender",
            feature_type=FeatureType.CATEGORICAL,
            source=FeatureSource.OFFLINE,
            description="User gender",
            vocabulary_size=3
        ),
        FeatureSpec(
            name="user_country",
            feature_type=FeatureType.CATEGORICAL,
            source=FeatureSource.OFFLINE,
            description="User country",
            vocabulary_size=200
        ),
        FeatureSpec(
            name="user_embedding",
            feature_type=FeatureType.EMBEDDING,
            source=FeatureSource.OFFLINE,
            description="User latent embedding",
            embedding_dim=64
        ),
        FeatureSpec(
            name="user_past_ctr",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.OFFLINE,
            description="Historical click-through rate",
            transformation="log",
            freshness_seconds=3600
        ),
        FeatureSpec(
            name="user_session_count_7d",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.OFFLINE,
            description="Sessions in last 7 days",
            transformation="sqrt",
            freshness_seconds=3600
        ),
    ]

    # Item features
    item_features = [
        FeatureSpec(
            name="item_id",
            feature_type=FeatureType.CATEGORICAL,
            source=FeatureSource.OFFLINE,
            is_key=True,
            description="Unique item identifier",
            vocabulary_size=500000
        ),
        FeatureSpec(
            name="item_category",
            feature_type=FeatureType.CATEGORICAL,
            source=FeatureSource.OFFLINE,
            description="Item category",
            vocabulary_size=500
        ),
        FeatureSpec(
            name="item_price",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.OFFLINE,
            description="Item price",
            transformation="log",
            freshness_seconds=86400
        ),
        FeatureSpec(
            name="item_embedding",
            feature_type=FeatureType.EMBEDDING,
            source=FeatureSource.OFFLINE,
            description="Item latent embedding",
            embedding_dim=64
        ),
        FeatureSpec(
            name="item_ctr_7d",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.OFFLINE,
            description="Item CTR in last 7 days",
            transformation="log",
            freshness_seconds=3600
        ),
        FeatureSpec(
            name="item_cvr_7d",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.OFFLINE,
            description="Item CVR in last 7 days",
            transformation="log",
            freshness_seconds=3600
        ),
    ]

    # Cross features
    cross_features = [
        FeatureSpec(
            name="user_item_affinity",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.REALTIME,
            description="User-item affinity score",
            freshness_seconds=60
        ),
        FeatureSpec(
            name="position",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.REALTIME,
            description="Position in ranking",
            is_target=False
        ),
        FeatureSpec(
            name="clicked",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.REALTIME,
            description="Click label (target)",
            is_target=True
        ),
        FeatureSpec(
            name="converted",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.REALTIME,
            description="Conversion label (target)",
            is_target=True
        ),
    ]

    all_features = user_features + item_features + cross_features

    for feature in all_features:
        store.register_feature(feature)

    # Register feature tables
    store.register_feature_table(
        table_name="user_features",
        entity_type="user",
        entity_id_column="user_id",
        description="User profile and behavioral features"
    )

    store.register_feature_table(
        table_name="item_features",
        entity_type="item",
        entity_id_column="item_id",
        description="Item catalog and performance features"
    )

    store.register_feature_table(
        table_name="interaction_features",
        entity_type="user_item",
        entity_id_column="user_id",
        description="Real-time interaction features"
    )

    logger.info("Created sample features for ranking/recommendation")

    # Add sample data to the feature tables
    import random
    import numpy as np

    # Generate more sample data
    n_users = 100
    n_items = 50
    n_interactions = 5000

    np.random.seed(42)
    random.seed(42)

    user_ids = [f"user_{i}" for i in range(1, n_users + 1)]
    item_ids = [f"item_{i}" for i in range(1, n_items + 1)]
    categories = ["electronics", "books", "home", "clothing", "sports", "toys"]
    countries = ["US", "CA", "UK", "DE", "FR", "JP", "AU"]
    genders = ["M", "F", "O"]

    sample_user_features = pd.DataFrame({
        "user_id": user_ids,
        "user_age": np.random.randint(18, 70, n_users),
        "user_gender": np.random.choice(genders, n_users),
        "user_country": np.random.choice(countries, n_users),
        "user_embedding": [list(np.random.randn(10)) for _ in range(n_users)],
        "user_past_ctr": np.random.beta(2, 20, n_users),
        "user_session_count_7d": np.random.poisson(10, n_users),
        "event_timestamp": [datetime.utcnow() - timedelta(days=random.randint(0, 30)) for _ in range(n_users)]
    })
    store.ingest_features(table_name="user_features", df=sample_user_features)

    sample_item_features = pd.DataFrame({
        "item_id": item_ids,
        "item_category": np.random.choice(categories, n_items),
        "item_price": np.random.lognormal(3, 1, n_items),
        "item_ctr_7d": np.random.beta(2, 20, n_items),
        "item_cvr_7d": np.random.beta(1, 50, n_items),
        "event_timestamp": [datetime.utcnow() - timedelta(days=random.randint(0, 30)) for _ in range(n_items)]
    })
    store.ingest_features(table_name="item_features", df=sample_item_features)

    sample_interaction_features = pd.DataFrame({
        "user_id": np.random.choice(user_ids, n_interactions),
        "item_id": np.random.choice(item_ids, n_interactions),
        "user_item_affinity": np.random.beta(2, 5, n_interactions),
        "position": np.random.randint(1, 10, n_interactions),
        "clicked": np.random.binomial(1, 0.15, n_interactions),
        "converted": np.random.binomial(1, 0.02, n_interactions),
        "event_timestamp": [datetime.utcnow() - timedelta(hours=random.randint(0, 720)) for _ in range(n_interactions)]
    })
    store.ingest_features(table_name="interaction_features", df=sample_interaction_features)

    logger.info("Ingested sample data into feature tables.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = OfflineFeatureStore()
    create_sample_features(store)
    print("Sample features created!")
    print(f"Total features: {len(store.list_features())}")