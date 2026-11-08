"""Production traffic simulator for shadow testing and load testing."""

import asyncio
import logging
import random
import numpy as np
import time
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import uuid

from core.schemas import (
    UserSession, ClickEvent, ConversionEvent,
    FeatureVector, PredictionRequest, PredictionResponse
)
from features.online_store import OnlineFeatureStore, MockOnlineFeatureStore

logger = logging.getLogger(__name__)


class UserBehaviorType(str, Enum):
    """Types of user behavior patterns."""
    CASUAL_BROWSER = "casual_browser"
    INTENT_SEARCHER = "intent_searcher"
    DEAL_HUNTER = "deal_hunter"
    LOYAL_CUSTOMER = "loyal_customer"
    NEW_VISITOR = "new_visitor"
    MOBILE_USER = "mobile_user"
    DESKTOP_USER = "desktop_user"


@dataclass
class BehaviorProfile:
    """User behavior profile for simulation."""
    behavior_type: UserBehaviorType
    session_duration_mean: float = 300  # seconds
    session_duration_std: float = 120
    pages_per_session_mean: float = 5
    pages_per_session_std: float = 3
    click_probability: float = 0.15
    conversion_probability: float = 0.02
    price_sensitivity: float = 0.5  # 0-1
    exploration_factor: float = 0.3  # 0-1
    device_type: str = "mobile"
    platform: str = "ios"
    geo_country: str = "US"
    referrer: str = "direct"
    user_features: Dict[str, Any] = field(default_factory=dict)


# Default behavior profiles
DEFAULT_PROFILES = {
    UserBehaviorType.CASUAL_BROWSER: BehaviorProfile(
        behavior_type=UserBehaviorType.CASUAL_BROWSER,
        session_duration_mean=180,
        pages_per_session_mean=3,
        click_probability=0.08,
        conversion_probability=0.01,
        price_sensitivity=0.7,
        exploration_factor=0.6,
        user_features={"user_past_ctr": 0.05, "user_session_count_7d": 2}
    ),
    UserBehaviorType.INTENT_SEARCHER: BehaviorProfile(
        behavior_type=UserBehaviorType.INTENT_SEARCHER,
        session_duration_mean=120,
        pages_per_session_mean=2,
        click_probability=0.3,
        conversion_probability=0.05,
        price_sensitivity=0.3,
        exploration_factor=0.1,
        user_features={"user_past_ctr": 0.25, "user_session_count_7d": 10}
    ),
    UserBehaviorType.DEAL_HUNTER: BehaviorProfile(
        behavior_type=UserBehaviorType.DEAL_HUNTER,
        session_duration_mean=300,
        pages_per_session_mean=8,
        click_probability=0.12,
        conversion_probability=0.03,
        price_sensitivity=0.9,
        exploration_factor=0.7,
        user_features={"user_past_ctr": 0.15, "user_session_count_7d": 15}
    ),
    UserBehaviorType.LOYAL_CUSTOMER: BehaviorProfile(
        behavior_type=UserBehaviorType.LOYAL_CUSTOMER,
        session_duration_mean=240,
        pages_per_session_mean=4,
        click_probability=0.2,
        conversion_probability=0.08,
        price_sensitivity=0.2,
        exploration_factor=0.2,
        user_features={"user_past_ctr": 0.3, "user_session_count_7d": 25}
    ),
    UserBehaviorType.NEW_VISITOR: BehaviorProfile(
        behavior_type=UserBehaviorType.NEW_VISITOR,
        session_duration_mean=90,
        pages_per_session_mean=2,
        click_probability=0.05,
        conversion_probability=0.005,
        price_sensitivity=0.6,
        exploration_factor=0.8,
        user_features={"user_past_ctr": 0.02, "user_session_count_7d": 0}
    ),
    UserBehaviorType.MOBILE_USER: BehaviorProfile(
        behavior_type=UserBehaviorType.MOBILE_USER,
        session_duration_mean=150,
        pages_per_session_mean=3,
        click_probability=0.1,
        conversion_probability=0.015,
        price_sensitivity=0.5,
        exploration_factor=0.4,
        device_type="mobile",
        user_features={"user_past_ctr": 0.08, "user_session_count_7d": 5}
    ),
    UserBehaviorType.DESKTOP_USER: BehaviorProfile(
        behavior_type=UserBehaviorType.DESKTOP_USER,
        session_duration_mean=300,
        pages_per_session_mean=5,
        click_probability=0.15,
        conversion_probability=0.03,
        price_sensitivity=0.4,
        exploration_factor=0.3,
        device_type="desktop",
        platform="windows",
        user_features={"user_past_ctr": 0.12, "user_session_count_7d": 8}
    ),
}


class ClickModel:
    """Click probability model based on position and relevance."""

    def __init__(
        self,
        position_bias: Dict[int, float] = None,
        relevance_weight: float = 1.0,
        noise_std: float = 0.1
    ):
        # Position-based click probability (typical values)
        self.position_bias = position_bias or {
            1: 0.3, 2: 0.15, 3: 0.1, 4: 0.07, 5: 0.05,
            6: 0.04, 7: 0.03, 8: 0.025, 9: 0.02, 10: 0.015
        }
        self.relevance_weight = relevance_weight
        self.noise_std = noise_std

    def click_probability(
        self,
        position: int,
        relevance_score: float,
        user_profile: BehaviorProfile = None
    ) -> float:
        """Calculate click probability."""
        base_prob = self.position_bias.get(position, 0.01)

        # Adjust by relevance
        relevance_factor = 1 + self.relevance_weight * (relevance_score - 0.5)

        # Adjust by user behavior
        user_factor = 1.0
        if user_profile:
            user_factor = user_profile.click_probability / 0.1  # Normalize

        prob = base_prob * relevance_factor * user_factor

        # Add noise
        noise = np.random.normal(0, self.noise_std)
        prob = np.clip(prob + noise, 0.001, 0.99)

        return prob

    def simulate_click(
        self,
        position: int,
        relevance_score: float,
        user_profile: BehaviorProfile = None
    ) -> bool:
        """Simulate click event."""
        prob = self.click_probability(position, relevance_score, user_profile)
        return np.random.random() < prob


class ConversionModel:
    """Conversion probability model."""

    def __init__(
        self,
        base_conversion_rate: float = 0.02,
        price_elasticity: float = -1.5,
        relevance_weight: float = 0.5
    ):
        self.base_conversion_rate = base_conversion_rate
        self.price_elasticity = price_elasticity
        self.relevance_weight = relevance_weight

    def conversion_probability(
        self,
        price: float,
        relevance_score: float,
        user_profile: BehaviorProfile = None,
        item_category: str = "general"
    ) -> float:
        """Calculate conversion probability."""
        # Price effect (log-linear)
        price_factor = np.exp(self.price_elasticity * np.log(max(price, 1) / 50))

        # Relevance effect
        relevance_factor = 1 + self.relevance_weight * (relevance_score - 0.5)

        # User behavior effect
        user_factor = 1.0
        if user_profile:
            user_factor = user_profile.conversion_probability / self.base_conversion_rate

        # Category effect (simplified)
        category_factors = {
            "electronics": 0.8,
            "clothing": 1.2,
            "books": 1.5,
            "home": 1.0,
            "sports": 1.1
        }
        category_factor = category_factors.get(item_category, 1.0)

        prob = self.base_conversion_rate * price_factor * relevance_factor * user_factor * category_factor

        return np.clip(prob, 0.0001, 0.5)

    def simulate_conversion(
        self,
        price: float,
        relevance_score: float,
        user_profile: BehaviorProfile = None,
        item_category: str = "general"
    ) -> bool:
        """Simulate conversion event."""
        prob = self.conversion_probability(price, relevance_score, user_profile, item_category)
        return np.random.random() < prob


@dataclass
class SimulationConfig:
    """Configuration for traffic simulation."""
    num_users: int = 1000
    sessions_per_user: int = 5
    items_per_session: int = 10
    behavior_distribution: Dict[UserBehaviorType, float] = field(default_factory=dict)
    model_endpoint: str = "http://localhost:8080/predict"
    enable_feature_store: bool = True
    feature_store: OnlineFeatureStore = None
    click_model: ClickModel = None
    conversion_model: ConversionModel = None
    think_time_mean: float = 2.0  # seconds between actions
    think_time_std: float = 1.0
    session_gap_mean: float = 3600  # seconds between sessions
    session_gap_std: float = 1800


class TrafficSimulator:
    """Production traffic simulator for shadow testing and A/B validation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.click_model = config.click_model or ClickModel()
        self.conversion_model = config.conversion_model or ConversionModel()

        # Initialize feature store
        if config.enable_feature_store and config.feature_store:
            self.feature_store = config.feature_store
        else:
            self.feature_store = MockOnlineFeatureStore()
            self._populate_feature_store()

        # Behavior distribution
        self.behavior_distribution = config.behavior_distribution or {
            UserBehaviorType.CASUAL_BROWSER: 0.3,
            UserBehaviorType.INTENT_SEARCHER: 0.2,
            UserBehaviorType.DEAL_HUNTER: 0.15,
            UserBehaviorType.LOYAL_CUSTOMER: 0.15,
            UserBehaviorType.NEW_VISITOR: 0.1,
            UserBehaviorType.MOBILE_USER: 0.05,
            UserBehaviorType.DESKTOP_USER: 0.05
        }

        # Normalize distribution
        total = sum(self.behavior_distribution.values())
        self.behavior_distribution = {k: v/total for k, v in self.behavior_distribution.items()}

        # Simulation state
        self.active_sessions: Dict[str, UserSession] = {}
        self.events: List[ClickEvent] = []
        self.conversions: List[ConversionEvent] = []
        self.predictions: List[PredictionResponse] = []

        # Statistics
        self.stats = {
            "sessions_created": 0,
            "predictions_requested": 0,
            "clicks_simulated": 0,
            "conversions_simulated": 0,
            "total_latency_ms": 0.0,
            "errors": 0
        }

    def _populate_feature_store(self):
        """Populate mock feature store with sample data."""
        import numpy as np

        # Create sample users
        for i in range(self.config.num_users):
            user_id = f"user_{i}"
            self.feature_store.set_feature_vector("user", user_id, {
                "user_id": user_id,
                "user_age": np.random.randint(18, 70),
                "user_gender": np.random.choice(["M", "F", "O"]),
                "user_embedding": np.random.randn(64).tolist(),
                "user_past_ctr": np.random.beta(1, 50),
                "user_session_count_7d": np.random.poisson(5)
            })

        # Create sample items
        for i in range(1000):
            item_id = f"item_{i}"
            self.feature_store.set_feature_vector("item", item_id, {
                "item_id": item_id,
                "item_category": np.random.choice(["electronics", "clothing", "books", "home", "sports"]),
                "item_price": np.random.lognormal(3, 1),
                "item_embedding": np.random.randn(64).tolist(),
                "item_ctr_7d": np.random.beta(2, 100),
                "item_cvr_7d": np.random.beta(1, 200)
            })

    def _sample_behavior(self) -> BehaviorProfile:
        """Sample a behavior profile from distribution."""
        behaviors = list(self.behavior_distribution.keys())
        probs = list(self.behavior_distribution.values())
        # Convert enums to their string values for np.random.choice
        behavior_values = [b.value for b in behaviors]
        behavior_type_str = np.random.choice(behavior_values, p=probs)
        # Convert back to enum
        behavior_type = UserBehaviorType(behavior_type_str)
        return DEFAULT_PROFILES[behavior_type]

    def _generate_user_id(self) -> str:
        """Generate a user ID."""
        return f"user_{np.random.randint(0, self.config.num_users)}"

    def _generate_item_ids(self, count: int) -> List[str]:
        """Generate item IDs for a session."""
        return [f"item_{np.random.randint(0, 1000)}" for _ in range(count)]

    async def simulate_session(
        self,
        user_id: str = None,
        variant: str = "control",
        model_version: str = "latest"
    ) -> UserSession:
        """Simulate a single user session."""
        user_id = user_id or self._generate_user_id()
        profile = self._sample_behavior()

        # Create session
        session = UserSession(
            user_id=user_id,
            user_features=profile.user_features,
            context_features={},
            device_type=profile.device_type,
            platform=profile.platform,
            geo_country=profile.geo_country,
            referrer=profile.referrer,
            ab_variant=variant
        )

        self.active_sessions[session.session_id] = session
        self.stats["sessions_created"] += 1

        # Generate items for this session
        num_items = max(1, int(np.random.normal(
            profile.pages_per_session_mean,
            profile.pages_per_session_std
        )))
        item_ids = self._generate_item_ids(num_items)

        # Get user features from store
        user_vec = self.feature_store.get_feature_vector("user", user_id)
        user_features = user_vec.features if user_vec else {}

        # Simulate session steps
        for position, item_id in enumerate(item_ids, 1):
            # Get item features
            item_vec = self.feature_store.get_feature_vector("item", item_id)
            item_features = item_vec.features if item_vec else {}

            # Build context
            context = {
                "position": position,
                "device_type": profile.device_type,
                "platform": profile.platform,
                "geo_country": profile.geo_country
            }

            # Request prediction
            entities = [
                FeatureVector(
                    entity_id=user_id,
                    entity_type="user",
                    features=user_features,
                    feature_timestamp=datetime.utcnow()
                ),
                FeatureVector(
                    entity_id=item_id,
                    entity_type="item",
                    features=item_features,
                    feature_timestamp=datetime.utcnow()
                )
            ]

            request = PredictionRequest(
                model_id="ranker_v1",
                model_version=model_version,
                entities=entities,
                context=context
            )

            # Simulate prediction (in real use, call actual model)
            start_time = time.time()
            relevance_score = self._simulate_prediction(user_features, item_features, position)
            latency = (time.time() - start_time) * 1000

            response = PredictionResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                model_version=request.model_version,
                predictions=[{"score": relevance_score, "item_id": item_id}],
                latency_ms=latency
            )

            self.predictions.append(response)
            self.stats["predictions_requested"] += 1
            self.stats["total_latency_ms"] += latency

            # Simulate click
            clicked = self.click_model.simulate_click(
                position=position,
                relevance_score=relevance_score,
                user_profile=profile
            )

            if clicked:
                click_event = ClickEvent(
                    session_id=session.session_id,
                    user_id=user_id,
                    item_id=item_id,
                    position=position,
                    score=relevance_score,
                    model_version=model_version,
                    variant=variant,
                    features={**user_features, **item_features, "position": position}
                )
                self.events.append(click_event)
                self.stats["clicks_simulated"] += 1

                # Simulate conversion
                price = item_features.get("item_price", 50)
                category = item_features.get("item_category", "general")

                converted = self.conversion_model.simulate_conversion(
                    price=price,
                    relevance_score=relevance_score,
                    user_profile=profile,
                    item_category=category
                )

                if converted:
                    conversion = ConversionEvent(
                        session_id=session.session_id,
                        user_id=user_id,
                        item_id=item_id,
                        conversion_type="purchase",
                        value=price,
                        metadata={"category": category, "position": position}
                    )
                    self.conversions.append(conversion)
                    self.stats["conversions_simulated"] += 1

            # Think time
            think_time = max(0.1, np.random.normal(
                self.config.think_time_mean,
                self.config.think_time_std
            ))
            await asyncio.sleep(think_time)

        # End session
        session.ended_at = datetime.utcnow()
        del self.active_sessions[session.session_id]

        return session

    def _simulate_prediction(
        self,
        user_features: Dict[str, Any],
        item_features: Dict[str, Any],
        position: int
    ) -> float:
        """Simulate model prediction (replace with actual model call)."""
        # Simple heuristic scoring
        user_ctr = user_features.get("user_past_ctr", 0.1)
        item_ctr = item_features.get("item_ctr_7d", 0.05)
        affinity = user_features.get("user_item_affinity", 0.5)

        # Position decay
        position_factor = 1.0 / (position ** 0.5)

        # Combine signals
        score = 0.4 * user_ctr + 0.3 * item_ctr + 0.3 * affinity
        score *= position_factor

        # Add noise
        score += np.random.normal(0, 0.05)

        return np.clip(score, 0.0, 1.0)

    async def run_simulation(
        self,
        duration_seconds: int = 3600,
        concurrent_sessions: int = 10,
        variant_split: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Run traffic simulation for specified duration."""
        variant_split = variant_split or {"control": 0.5, "treatment": 0.5}
        variants = list(variant_split.keys())
        variant_probs = list(variant_split.values())

        start_time = time.time()
        end_time = start_time + duration_seconds

        # Session generator
        async def session_generator():
            while time.time() < end_time:
                # Check concurrent limit
                while len(self.active_sessions) >= concurrent_sessions:
                    await asyncio.sleep(0.1)

                if time.time() >= end_time:
                    break

                # Select variant
                variant = np.random.choice(variants, p=variant_probs)

                # Start session
                asyncio.create_task(self.simulate_session(variant=variant))

                # Gap between session starts
                gap = max(0.1, np.random.normal(
                    self.config.session_gap_mean / concurrent_sessions,
                    self.config.session_gap_std / concurrent_sessions
                ))
                await asyncio.sleep(gap)

        # Run generator
        await session_generator()

        # Wait for active sessions to complete
        while self.active_sessions:
            await asyncio.sleep(0.5)

        return self.get_results()

    async def run_load_test(
        self,
        num_requests: int = 1000,
        concurrent_requests: int = 50,
        model_id: str = "ranker_v1",
        model_version: str = "latest"
    ) -> Dict[str, Any]:
        """Run load test against model endpoint."""
        latencies = []
        errors = 0
        start_time = time.time()

        semaphore = asyncio.Semaphore(concurrent_requests)

        async def make_request():
            nonlocal errors
            async with semaphore:
                request_start = time.time()
                try:
                    # Simulate prediction
                    user_id = self._generate_user_id()
                    item_id = self._generate_item_ids(1)[0]

                    user_vec = self.feature_store.get_feature_vector("user", user_id)
                    item_vec = self.feature_store.get_feature_vector("item", item_id)

                    relevance = self._simulate_prediction(
                        user_vec.features if user_vec else {},
                        item_vec.features if item_vec else {},
                        1
                    )

                    latency = (time.time() - request_start) * 1000
                    latencies.append(latency)
                except Exception as e:
                    errors += 1
                    logger.error(f"Load test request failed: {e}")

        tasks = [make_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        return {
            "total_requests": num_requests,
            "successful_requests": num_requests - errors,
            "errors": errors,
            "error_rate": errors / num_requests,
            "duration_seconds": total_time,
            "throughput_rps": num_requests / total_time,
            "latency_stats": {
                "mean": np.mean(latencies) if latencies else 0,
                "p50": np.percentile(latencies, 50) if latencies else 0,
                "p90": np.percentile(latencies, 90) if latencies else 0,
                "p95": np.percentile(latencies, 95) if latencies else 0,
                "p99": np.percentile(latencies, 99) if latencies else 0,
                "max": max(latencies) if latencies else 0
            }
        }

    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        # Calculate CTR by variant
        variant_clicks = defaultdict(int)
        variant_impressions = defaultdict(int)
        variant_conversions = defaultdict(int)

        for event in self.events:
            variant_impressions[event.variant] += 1
            if event.position == 1:  # Simplified - count all as impressions
                pass
            variant_clicks[event.variant] += 1

        for conv in self.conversions:
            # Find variant from click event
            click = next((e for e in self.events if e.session_id == conv.session_id and e.item_id == conv.item_id), None)
            if click:
                variant_conversions[click.variant] += 1

        results = {
            "summary": {
                "total_sessions": self.stats["sessions_created"],
                "total_predictions": self.stats["predictions_requested"],
                "total_clicks": self.stats["clicks_simulated"],
                "total_conversions": self.stats["conversions_simulated"],
                "avg_latency_ms": self.stats["total_latency_ms"] / max(1, self.stats["predictions_requested"]),
                "errors": self.stats["errors"]
            },
            "by_variant": {}
        }

        for variant in variant_clicks.keys():
            impressions = variant_impressions.get(variant, 1)
            clicks = variant_clicks.get(variant, 0)
            conversions = variant_conversions.get(variant, 0)

            results["by_variant"][variant] = {
                "impressions": impressions,
                "clicks": clicks,
                "ctr": clicks / impressions if impressions > 0 else 0,
                "conversions": conversions,
                "cvr": conversions / clicks if clicks > 0 else 0,
                "revenue_per_session": 0  # Would calculate from conversion values
            }

        return results

    def export_events(self, format: str = "json") -> str:
        """Export events for analysis."""
        import json

        data = {
            "clicks": [
                {
                    "event_id": e.event_id,
                    "session_id": e.session_id,
                    "user_id": e.user_id,
                    "item_id": e.item_id,
                    "position": e.position,
                    "score": e.score,
                    "model_version": e.model_version,
                    "variant": e.variant,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in self.events
            ],
            "conversions": [
                {
                    "event_id": c.event_id,
                    "session_id": c.session_id,
                    "user_id": c.user_id,
                    "item_id": c.item_id,
                    "conversion_type": c.conversion_type,
                    "value": c.value,
                    "timestamp": c.timestamp.isoformat()
                }
                for c in self.conversions
            ]
        }

        return json.dumps(data, indent=2)


class ShadowTestRunner:
    """Run shadow tests comparing model variants on production traffic."""

    def __init__(
        self,
        simulator: TrafficSimulator,
        control_model: str = "control",
        treatment_models: List[str] = None
    ):
        self.simulator = simulator
        self.control_model = control_model
        self.treatment_models = treatment_models or ["treatment"]

    async def run_shadow_test(
        self,
        num_sessions: int = 1000,
        traffic_split: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Run shadow test where all variants score but only control serves."""
        traffic_split = traffic_split or {self.control_model: 1.0}
        for tm in self.treatment_models:
            traffic_split[tm] = 0.0  # Shadow only

        # Modify to score all variants for each request
        original_simulate = self.simulator._simulate_prediction

        variant_scores = defaultdict(list)

        async def shadow_simulate_session(user_id: str = None):
            user_id = user_id or self.simulator._generate_user_id()
            profile = self.simulator._sample_behavior()

            session = UserSession(
                user_id=user_id,
                user_features=profile.user_features,
                context_features={},
                device_type=profile.device_type,
                platform=profile.platform,
                geo_country=profile.geo_country,
                referrer=profile.referrer
            )

            num_items = max(1, int(np.random.normal(
                profile.pages_per_session_mean,
                profile.pages_per_session_std
            )))
            item_ids = self.simulator._generate_item_ids(num_items)

            user_vec = self.simulator.feature_store.get_feature_vector("user", user_id)
            user_features = user_vec.features if user_vec else {}

            for position, item_id in enumerate(item_ids, 1):
                item_vec = self.simulator.feature_store.get_feature_vector("item", item_id)
                item_features = item_vec.features if item_vec else {}

                # Score all variants
                for variant in [self.control_model] + self.treatment_models:
                    score = original_simulate(user_features, item_features, position)
                    variant_scores[variant].append(score)

                # Use control for click simulation
                clicked = self.simulator.click_model.simulate_click(
                    position, variant_scores[self.control_model][-1], profile
                )

                if clicked:
                    self.simulator.events.append(ClickEvent(
                        session_id=session.session_id,
                        user_id=user_id,
                        item_id=item_id,
                        position=position,
                        score=variant_scores[self.control_model][-1],
                        model_version="shadow",
                        variant=self.control_model
                    ))

            session.ended_at = datetime.utcnow()

        # Run sessions
        tasks = [shadow_simulate_session() for _ in range(num_sessions)]
        await asyncio.gather(*tasks)

        # Compare variants
        comparison = {}
        for variant, scores in variant_scores.items():
            comparison[variant] = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "score_distribution": np.histogram(scores, bins=10)[0].tolist()
            }

        return {
            "variant_comparison": comparison,
            "traffic_shadowed": True,
            "control_served": self.control_model
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        config = SimulationConfig(
            num_users=100,
            sessions_per_user=2,
            items_per_session=5
        )

        simulator = TrafficSimulator(config)

        # Run short simulation
        results = await simulator.run_simulation(
            duration_seconds=10,
            concurrent_sessions=5
        )

        print(f"Results: {results}")

        # Run load test
        load_results = await simulator.run_load_test(
            num_requests=100,
            concurrent_requests=10
        )
        print(f"Load test: {load_results}")

    asyncio.run(test())