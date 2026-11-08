"""Online feature store using Redis for low-latency feature serving."""

import redis
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle

from core.schemas import FeatureVector, FeatureSpec, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class FeatureValue:
    """A single feature value with metadata."""
    value: Any
    timestamp: datetime
    ttl_seconds: int

    def to_bytes(self) -> bytes:
        return pickle.dumps(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FeatureValue':
        d = pickle.loads(data)
        d['timestamp'] = datetime.fromisoformat(d['timestamp']) if isinstance(d['timestamp'], str) else d['timestamp']
        return cls(**d)


class OnlineFeatureStore:
    """Online feature store for real-time feature serving.

    Features:
    - Redis backend with connection pooling
    - TTL-based expiration for freshness
    - Batch get/set for efficiency
    - Feature vector retrieval for entities
    - Support for embeddings, categorical, numerical features
    - Fallback to default values
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl_seconds: int = 86400,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        enable_compression: bool = True
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl_seconds
        self.enable_compression = enable_compression

        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False  # We handle encoding ourselves
        )
        self.client = redis.Redis(connection_pool=self.pool)

        # Feature metadata cache
        self._feature_specs: Dict[str, FeatureSpec] = {}

    def register_feature_spec(self, feature: FeatureSpec) -> None:
        """Register feature specification for validation and defaults."""
        self._feature_specs[feature.name] = feature

    def _make_key(self, entity_type: str, entity_id: str, feature_name: str) -> str:
        """Create Redis key for feature."""
        return f"feat:{entity_type}:{entity_id}:{feature_name}"

    def _make_vector_key(self, entity_type: str, entity_id: str) -> str:
        """Create Redis key for feature vector."""
        return f"fvec:{entity_type}:{entity_id}"

    def set_feature(
        self,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        value: Any,
        timestamp: datetime = None,
        ttl_seconds: int = None
    ) -> bool:
        """Set a single feature value."""
        timestamp = timestamp or datetime.utcnow()
        ttl = ttl_seconds or self.default_ttl

        feature_value = FeatureValue(
            value=value,
            timestamp=timestamp,
            ttl_seconds=ttl
        )

        key = self._make_key(entity_type, entity_id, feature_name)
        try:
            self.client.setex(key, ttl, feature_value.to_bytes())
            return True
        except Exception as e:
            logger.error(f"Failed to set feature {key}: {e}")
            return False

    def get_feature(
        self,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        default: Any = None
    ) -> Any:
        """Get a single feature value with fallback to default."""
        key = self._make_key(entity_type, entity_id, feature_name)

        try:
            data = self.client.get(key)
            if data is None:
                # Check for default in spec
                if feature_name in self._feature_specs:
                    return self._feature_specs[feature_name].default_value
                return default

            feature_value = FeatureValue.from_bytes(data)

            # Check freshness
            if self._feature_specs.get(feature_name, None):
                spec = self._feature_specs[feature_name]
                if spec.freshness_seconds:
                    age = (datetime.utcnow() - feature_value.timestamp).total_seconds()
                    if age > spec.freshness_seconds:
                        logger.warning(f"Feature {feature_name} stale: {age}s old")
                        if spec.default_value is not None:
                            return spec.default_value

            return feature_value.value

        except Exception as e:
            logger.error(f"Failed to get feature {key}: {e}")
            if feature_name in self._feature_specs:
                return self._feature_specs[feature_name].default_value
            return default

    def get_features_batch(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Get multiple features in a single pipeline call."""
        keys = [self._make_key(entity_type, entity_id, fn) for fn in feature_names]

        try:
            pipe = self.client.mget(keys)
            results = {}

            for feature_name, data in zip(feature_names, pipe):
                if data is None:
                    if feature_name in self._feature_specs:
                        results[feature_name] = self._feature_specs[feature_name].default_value
                    else:
                        results[feature_name] = None
                else:
                    try:
                        feature_value = FeatureValue.from_bytes(data)
                        results[feature_name] = feature_value.value
                    except Exception:
                        if feature_name in self._feature_specs:
                            results[feature_name] = self._feature_specs[feature_name].default_value
                        else:
                            results[feature_name] = None

            return results

        except Exception as e:
            logger.error(f"Batch get failed: {e}")
            return {fn: self._feature_specs.get(fn, FeatureSpec(name=fn, feature_type=FeatureType.NUMERICAL, source=FeatureSource.ONLINE)).default_value
                    for fn in feature_names}

    def set_feature_vector(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: datetime = None,
        ttl_seconds: int = None
    ) -> bool:
        """Set entire feature vector for an entity."""
        timestamp = timestamp or datetime.utcnow()
        ttl = ttl_seconds or self.default_ttl

        feature_vector = FeatureVector(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            feature_timestamp=timestamp,
            freshness_seconds=ttl
        )

        key = self._make_vector_key(entity_type, entity_id)
        try:
            # Store as JSON for readability
            data = json.dumps({
                "entity_id": entity_id,
                "entity_type": entity_type,
                "features": features,
                "feature_timestamp": timestamp.isoformat(),
                "retrieved_at": datetime.utcnow().isoformat(),
                "freshness_seconds": ttl
            })
            self.client.setex(key, ttl, data)

            # Also store individual features for partial retrieval
            pipe = self.client.pipeline()
            for fname, fvalue in features.items():
                fkey = self._make_key(entity_type, entity_id, fname)
                fv = FeatureValue(value=fvalue, timestamp=timestamp, ttl_seconds=ttl)
                pipe.setex(fkey, ttl, fv.to_bytes())
            pipe.execute()

            return True

        except Exception as e:
            logger.error(f"Failed to set feature vector {key}: {e}")
            return False

    def get_feature_vector(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: List[str] = None
    ) -> Optional[FeatureVector]:
        """Get feature vector for an entity."""
        key = self._make_vector_key(entity_type, entity_id)

        try:
            data = self.client.get(key)
            if data is None:
                # Try to build from individual features
                if feature_names:
                    features = self.get_features_batch(entity_type, entity_id, feature_names)
                    return FeatureVector(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        features=features,
                        feature_timestamp=datetime.utcnow()
                    )
                return None

            parsed = json.loads(data)
            feature_vec = FeatureVector(
                entity_id=parsed["entity_id"],
                entity_type=parsed["entity_type"],
                features=parsed["features"],
                feature_timestamp=datetime.fromisoformat(parsed["feature_timestamp"]),
                retrieved_at=datetime.fromisoformat(parsed["retrieved_at"]),
                freshness_seconds=parsed.get("freshness_seconds")
            )

            # Filter to requested features
            if feature_names:
                feature_vec.features = {
                    k: v for k, v in feature_vec.features.items()
                    if k in feature_names
                }

            return feature_vec

        except Exception as e:
            logger.error(f"Failed to get feature vector {key}: {e}")
            return None

    def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete all features for an entity."""
        try:
            # Find and delete all keys for this entity
            pattern = f"feat:{entity_type}:{entity_id}:*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                self.client.delete(*keys)

            # Delete vector key
            vec_key = self._make_vector_key(entity_type, entity_id)
            self.client.delete(vec_key)

            return True
        except Exception as e:
            logger.error(f"Failed to delete entity {entity_type}:{entity_id}: {e}")
            return False

    def exists(self, entity_type: str, entity_id: str, feature_name: str) -> bool:
        """Check if feature exists."""
        key = self._make_key(entity_type, entity_id, feature_name)
        return self.client.exists(key) > 0

    def get_ttl(self, entity_type: str, entity_id: str, feature_name: str) -> int:
        """Get remaining TTL for a feature."""
        key = self._make_key(entity_type, entity_id, feature_name)
        return self.client.ttl(key)

    def refresh_ttl(self, entity_type: str, entity_id: str, feature_name: str, ttl_seconds: int = None) -> bool:
        """Refresh TTL for a feature."""
        key = self._make_key(entity_type, entity_id, feature_name)
        ttl = ttl_seconds or self.default_ttl
        return self.client.expire(key, ttl)

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis stats."""
        info = self.client.info()
        return {
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "total_commands_processed": info.get("total_commands_processed"),
            "keyspace_hits": info.get("keyspace_hits"),
            "keyspace_misses": info.get("keyspace_misses"),
            "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
        }

    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self.client.ping()
        except Exception:
            return False

    def close(self):
        """Close connections."""
        self.client.close()
        self.pool.disconnect()


class MockOnlineFeatureStore(OnlineFeatureStore):
    """In-memory mock for testing without Redis."""

    def __init__(self, *args, **kwargs):
        # Don't call parent init
        self._store: Dict[str, FeatureValue] = {}
        self._vectors: Dict[str, FeatureVector] = {}
        self._feature_specs: Dict[str, FeatureSpec] = {}
        self.default_ttl = kwargs.get('default_ttl_seconds', 86400)

    def _make_key(self, entity_type: str, entity_id: str, feature_name: str) -> str:
        return f"{entity_type}:{entity_id}:{feature_name}"

    def _make_vector_key(self, entity_type: str, entity_id: str) -> str:
        return f"vector:{entity_type}:{entity_id}"

    def set_feature(self, entity_type: str, entity_id: str, feature_name: str,
                    value: Any, timestamp: datetime = None, ttl_seconds: int = None) -> bool:
        timestamp = timestamp or datetime.utcnow()
        key = self._make_key(entity_type, entity_id, feature_name)
        self._store[key] = FeatureValue(
            value=value,
            timestamp=timestamp,
            ttl_seconds=ttl_seconds or self.default_ttl
        )
        return True

    def get_feature(self, entity_type: str, entity_id: str, feature_name: str, default: Any = None) -> Any:
        key = self._make_key(entity_type, entity_id, feature_name)
        if key not in self._store:
            if feature_name in self._feature_specs:
                return self._feature_specs[feature_name].default_value
            return default
        return self._store[key].value

    def get_features_batch(self, entity_type: str, entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
        return {fn: self.get_feature(entity_type, entity_id, fn) for fn in feature_names}

    def set_feature_vector(self, entity_type: str, entity_id: str, features: Dict[str, Any],
                           timestamp: datetime = None, ttl_seconds: int = None) -> bool:
        timestamp = timestamp or datetime.utcnow()
        key = self._make_vector_key(entity_type, entity_id)
        self._vectors[key] = FeatureVector(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            feature_timestamp=timestamp,
            freshness_seconds=ttl_seconds or self.default_ttl
        )
        # Also set individual features
        for fname, fvalue in features.items():
            self.set_feature(entity_type, entity_id, fname, fvalue, timestamp, ttl_seconds)
        return True

    def get_feature_vector(self, entity_type: str, entity_id: str,
                           feature_names: List[str] = None) -> Optional[FeatureVector]:
        key = self._make_vector_key(entity_type, entity_id)
        if key not in self._vectors:
            if feature_names:
                features = self.get_features_batch(entity_type, entity_id, feature_names)
                return FeatureVector(entity_id=entity_id, entity_type=entity_type, features=features,
                                    feature_timestamp=datetime.utcnow())
            return None

        vec = self._vectors[key]
        if feature_names:
            vec.features = {k: v for k, v in vec.features.items() if k in feature_names}
        return vec

    def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        prefix = f"{entity_type}:{entity_id}:"
        keys_to_delete = [k for k in self._store.keys() if k.startswith(prefix)]
        for k in keys_to_delete:
            del self._store[k]

        vec_key = self._make_vector_key(entity_type, entity_id)
        if vec_key in self._vectors:
            del self._vectors[vec_key]
        return True

    def exists(self, entity_type: str, entity_id: str, feature_name: str) -> bool:
        key = self._make_key(entity_type, entity_id, feature_name)
        return key in self._store

    def health_check(self) -> bool:
        return True

    def close(self):
        pass


def create_sample_online_data(store: OnlineFeatureStore, num_users: int = 100, num_items: int = 50) -> None:
    """Populate online store with sample data for testing."""
    import random
    import numpy as np

    # Register feature specs
    specs = [
        FeatureSpec(name="user_id", feature_type=FeatureType.CATEGORICAL, source=FeatureSource.ONLINE, is_key=True, vocabulary_size=1000000),
        FeatureSpec(name="user_age", feature_type=FeatureType.NUMERICAL, source=FeatureSource.ONLINE, default_value=25, transformation="normalize"),
        FeatureSpec(name="user_gender", feature_type=FeatureType.CATEGORICAL, source=FeatureSource.ONLINE, default_value="unknown", vocabulary_size=3),
        FeatureSpec(name="user_embedding", feature_type=FeatureType.EMBEDDING, source=FeatureSource.ONLINE, embedding_dim=64, default_value=[0.0]*64),
        FeatureSpec(name="user_past_ctr", feature_type=FeatureType.NUMERICAL, source=FeatureSource.ONLINE, default_value=0.01, transformation="log"),
        FeatureSpec(name="item_id", feature_type=FeatureType.CATEGORICAL, source=FeatureSource.ONLINE, is_key=True, vocabulary_size=500000),
        FeatureSpec(name="item_category", feature_type=FeatureType.CATEGORICAL, source=FeatureSource.ONLINE, default_value="general", vocabulary_size=500),
        FeatureSpec(name="item_price", feature_type=FeatureType.NUMERICAL, source=FeatureSource.ONLINE, default_value=10.0, transformation="log"),
        FeatureSpec(name="item_embedding", feature_type=FeatureType.EMBEDDING, source=FeatureSource.ONLINE, embedding_dim=64, default_value=[0.0]*64),
        FeatureSpec(name="item_ctr_7d", feature_type=FeatureType.NUMERICAL, source=FeatureSource.ONLINE, default_value=0.02, transformation="log"),
        FeatureSpec(name="user_item_affinity", feature_type=FeatureType.NUMERICAL, source=FeatureSource.REALTIME, default_value=0.5, freshness_seconds=60),
    ]

    for spec in specs:
        store.register_feature_spec(spec)

    # Populate user features
    for i in range(num_users):
        user_id = f"user_{i}"
        store.set_feature_vector("user", user_id, {
            "user_id": user_id,
            "user_age": random.randint(18, 70),
            "user_gender": random.choice(["M", "F", "O"]),
            "user_embedding": np.random.randn(64).tolist(),
            "user_past_ctr": random.random() * 0.1,
        })

    # Populate item features
    for i in range(num_items):
        item_id = f"item_{i}"
        store.set_feature_vector("item", item_id, {
            "item_id": item_id,
            "item_category": random.choice(["electronics", "clothing", "books", "home", "sports"]),
            "item_price": random.uniform(1, 500),
            "item_embedding": np.random.randn(64).tolist(),
            "item_ctr_7d": random.random() * 0.1,
        })

    # Populate some user-item affinity scores
    for i in range(min(num_users, 20)):
        for j in range(min(num_items, 10)):
            store.set_feature(
                "user_item", f"user_{i}_item_{j}",
                "user_item_affinity", random.random()
            )

    logger.info(f"Created sample data: {num_users} users, {num_items} items")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Use mock store for testing
    store = MockOnlineFeatureStore()
    create_sample_online_data(store)

    # Test retrieval
    vec = store.get_feature_vector("user", "user_0", ["user_age", "user_embedding"])
    print(f"User vector: {vec}")

    vec = store.get_feature_vector("item", "item_0")
    print(f"Item vector: {vec}")

    print("Online feature store test passed!")