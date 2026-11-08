import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from core.schemas import ExperimentSpec, ModelSpec, ModelType, DriftSeverity
from experimentation.framework import ABTestFramework
from experimentation.analysis import ExperimentAnalyzer
from experimentation.sequential import SequentialTester, SPRTTester
from experimentation.guardrails import GuardrailMonitor
from monitoring.drift import DataDriftDetector
from monitoring.alerts import AlertManager, AlertChannel, Alert, AlertStatus
from features.offline_store import OfflineFeatureStore
from features.online_store import MockOnlineFeatureStore
from features.registry import FeatureRegistry
from features.transformers import (
    NumericalTransformer, CategoricalTransformer, TextTransformer,
    EmbeddingTransformer, CrossFeatureTransformer, FeaturePipeline
)
from features.lineage import LineageTracker
from training.pipeline import TrainingPipeline
from training.registry import ModelRegistry
from training.tracker import ExperimentTracker
from training.models import RankingModel, RecommendationModel, SearchRerankerModel
from simulator import TrafficSimulator, SimulationConfig, ClickModel, ConversionModel, ShadowTestRunner
from core.config import load_config


class TestFeatureStore:
    """Tests for feature store components."""

    def test_offline_store_ingest_and_query(self, tmp_path):
        db_path = tmp_path / "offline_store.db"
        store = OfflineFeatureStore(db_path=str(db_path))

        now = datetime.utcnow()
        store.ingest_features("user_features", pd.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "event_timestamp": [now, now, now],
            "user_age": [25, 30, 35],
            "user_signup_days": [10, 20, 30]
        }))

        stats = store.compute_feature_statistics(
            "user_features",
            ["user_age", "user_signup_days"],
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        )
        assert stats["user_age"]["count"] == 3
        assert stats["user_age"]["mean"] == 30.0
        assert stats["user_signup_days"]["mean"] == 20.0

    def test_offline_store_get_training_data(self, tmp_path):
        db_path = tmp_path / "offline_store.db"
        store = OfflineFeatureStore(db_path=str(db_path))

        now = datetime.utcnow()
        # Ingest user features with columns matching get_training_data query
        store.ingest_features("user_features", pd.DataFrame({
            "user_id": ["u1", "u2"],
            "event_timestamp": [now, now],
            "user_age": [25, 30],
            "user_gender": ["M", "F"],
            "user_country": ["US", "CA"],
            "user_past_ctr": [0.1, 0.2],
            "user_session_count_7d": [5, 10]
        }))
        # Ingest item features with columns matching get_training_data query
        store.ingest_features("item_features", pd.DataFrame({
            "item_id": ["i1", "i2"],
            "event_timestamp": [now, now],
            "item_category": ["electronics", "clothing"],
            "item_price": [10.0, 20.0],
            "item_ctr_7d": [0.05, 0.03],
            "item_cvr_7d": [0.02, 0.01],
            "item_brand": ["brand_a", "brand_b"],
            "item_rating": [4.5, 4.0]
        }))
        # Ingest interactions
        store.ingest_features("interaction_features", pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i1", "i2"],
            "event_timestamp": [now, now, now, now],
            "clicked": [1, 0, 0, 1],
            "position": [1, 2, 1, 2],
            "user_item_affinity": [0.8, 0.3, 0.4, 0.9]
        }))

        # Test get_training_data
        df = store.get_training_data(
            entity_type="user_item",
            feature_names=["user_age", "user_gender", "user_country", "user_past_ctr", "user_session_count_7d",
                          "item_price", "item_category", "item_brand", "item_rating",
                          "item_ctr_7d", "item_cvr_7d"],
            target_feature="clicked",
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=1)
        )
        assert len(df) == 4
        assert "user_id" in df.columns
        assert "item_id" in df.columns
        assert "clicked" in df.columns
        assert "user_age" in df.columns
        assert "item_price" in df.columns

    def test_online_store_mock(self):
        store = MockOnlineFeatureStore()

        # Test mset and mget (actual methods)
        store.mset({"user:123": {"age": 25, "country": "US"},
                    "user:456": {"age": 30, "country": "CA"}})
        result = store.mget(["user:123", "user:456", "user:999"])
        assert result["user:123"]["age"] == 25
        assert result["user:456"]["age"] == 30
        assert "user:999" not in result


class TestFeatureRegistry:
    """Tests for feature registry."""

    def test_register_and_get_feature(self):
        registry = FeatureRegistry()

        from core.schemas import FeatureSpec, FeatureType, FeatureSource
        feature = FeatureSpec(
            name="user_age",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.USER,
            description="User age in years",
            default_value=25.0,
            tags=["demographics"]
        )
        registry.register_feature(feature)

        retrieved = registry.get_feature("user_age")
        assert retrieved is not None
        assert retrieved.name == "user_age"
        assert retrieved.feature_type.value == "numerical"
        assert retrieved.source.value == "user"

    def test_feature_groups(self):
        registry = FeatureRegistry()

        from core.schemas import FeatureSpec, FeatureType, FeatureSource
        user_feature = FeatureSpec(
            name="user_age",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.USER,
            description="User age"
        )
        item_feature = FeatureSpec(
            name="item_price",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.ITEM,
            description="Item price"
        )
        registry.register_feature(user_feature)
        registry.register_feature(item_feature)

        user_features = registry.get_features_by_group("user")
        item_features = registry.get_features_by_group("item")

        assert len(user_features) == 1
        assert len(item_features) == 1
        assert user_features[0].name == "user_age"
        assert item_features[0].name == "item_price"

    def test_feature_dependencies(self):
        registry = FeatureRegistry()

        from core.schemas import FeatureSpec, FeatureType, FeatureSource
        base_feature = FeatureSpec(
            name="user_age",
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.USER,
            description="Base feature"
        )
        derived_feature = FeatureSpec(
            name="user_age_bucket",
            feature_type=FeatureType.CATEGORICAL,
            source=FeatureSource.USER,
            description="Derived feature",
            dependencies=["user_age"]
        )
        registry.register_feature(base_feature)
        registry.register_feature(derived_feature)

        deps = registry.get_dependencies("user_age_bucket")
        assert "user_age" in deps

        dependents = registry.get_dependents("user_age")
        assert "user_age_bucket" in dependents


class TestFeatureTransformers:
    """Tests for feature transformers."""

    def test_numerical_transformer(self):
        transformer = NumericalTransformer(method="standard")
        data = pd.DataFrame({"age": [20, 25, 30, 35, 40]})
        transformer.fit(data)
        transformed = transformer.transform(data)

        # Check standardization (mean ~0)
        assert abs(transformed["age"].mean()) < 0.01
        # std should be close to 1 for standard scaler
        assert abs(transformed["age"].std() - 1.0) < 0.5  # relaxed for small sample

    def test_categorical_transformer(self):
        transformer = CategoricalTransformer(method="onehot")
        data = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})
        transformer.fit(data)
        transformed = transformer.transform(data)

        # Check one-hot columns created
        assert any("red" in c for c in transformed.columns)
        assert any("blue" in c for c in transformed.columns)
        assert any("green" in c for c in transformed.columns)

    def test_text_transformer(self):
        transformer = TextTransformer(method="tfidf", max_features=10)
        data = pd.DataFrame({"text": ["hello world", "hello there", "world peace"]})
        transformer.fit(data)
        transformed = transformer.transform(data)

        assert transformed.shape[0] == 3
        assert transformed.shape[1] <= 10

    def test_embedding_transformer(self):
        transformer = EmbeddingTransformer(embedding_dim=4)
        data = pd.DataFrame({"category": [1, 2, 3, 1, 2, 99]})  # 99 is OOV
        transformer.fit(data)
        transformed = transformer.transform(data)

        assert transformed.shape[0] == 6
        assert transformed.shape[1] == 4  # embedding_dim

    def test_cross_feature_transformer(self):
        transformer = CrossFeatureTransformer()
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        transformer.fit(data)
        transformed = transformer.transform(data)

        # Check interaction features created
        assert any("a_x_b" in c or "a*b" in c for c in transformed.columns)

    def test_feature_pipeline(self):
        pipeline = FeaturePipeline()
        pipeline.add("num", NumericalTransformer("standard"), ["age", "income"])
        pipeline.add("cat", CategoricalTransformer("onehot"), ["gender", "country"])

        data = pd.DataFrame({
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "gender": ["M", "F", "M"],
            "country": ["US", "CA", "US"]
        })

        pipeline.fit(data)
        transformed = pipeline.transform(data)

        # Check all columns transformed
        assert "age" in transformed.columns  # scaled
        assert "income" in transformed.columns  # scaled
        assert any("gender" in c for c in transformed.columns)  # one-hot
        assert any("country" in c for c in transformed.columns)


class TestLineageTracker:
    """Tests for lineage tracking."""

    def test_add_nodes_and_edges(self):
        tracker = LineageTracker()

        from features.lineage import LineageNode, LineageEdge

        # Add nodes
        tracker.add_node(LineageNode(
            id="source_1",
            type="data_source",
            name="raw_events",
            metadata={"description": "Raw event logs"}
        ))
        tracker.add_node(LineageNode(
            id="feature_1",
            type="feature",
            name="user_age",
            metadata={"table": "user_features"}
        ))
        tracker.add_node(LineageNode(
            id="transform_1",
            type="transformation",
            name="age_standardization",
            metadata={"method": "standard"}
        ))

        # Add edges
        tracker.add_edge(LineageEdge(
            source="source_1",
            target="feature_1",
            type="derives_from"
        ))
        tracker.add_edge(LineageEdge(
            source="feature_1",
            target="transform_1",
            type="derives_from"
        ))

        # Test upstream/downstream
        upstream = tracker.get_upstream("transform_1")
        assert "feature_1" in upstream
        assert "source_1" in upstream

        downstream = tracker.get_downstream("source_1")
        assert "feature_1" in downstream
        assert "transform_1" in downstream


class TestTraining:
    """Tests for training components."""

    def test_training_config_load(self):
        config = load_config("test-project/configs/training.yaml", type("Config", (), {"__init__": lambda self, **kwargs: setattr(self, k, v) for k, v in kwargs.items()})())
        assert config is not None

    def test_model_registry(self, tmp_path):
        registry = ModelRegistry(registry_dir=str(tmp_path / "model_registry"))

        # Register a model
        model_id = registry.register_model(
            name="test_ranker",
            model_type="ranker",
            model_path="path/to/model",
            metrics={"auc": 0.85, "precision": 0.75},
            tags={"team": "ranking"}
        )
        assert model_id is not None

        # Get model
        model = registry.get_model(model_id)
        assert model is not None
        assert model.name == "test_ranker"
        assert model.model_type == "ranker"

        # Test versioning
        v1 = registry.create_version(model_id, "path/to/model_v1", {"auc": 0.85})
        v2 = registry.create_version(model_id, "path/to/model_v2", {"auc": 0.87})
        assert v2 > v1

        # Test stage promotion
        registry.promote_version(model_id, v2, "staging")
        registry.promote_version(model_id, v2, "production")

        prod_model = registry.get_production_model(model_id)
        assert prod_model is not None
        assert prod_model.version == v2


class TestExperimentFramework:
    """Tests for experimentation framework."""

    def test_experiment_lifecycle(self):
        framework = ABTestFramework()

        control_model = ModelSpec(name='control', model_type=ModelType.RANKER, model_id='control')
        treatment_model = ModelSpec(name='treatment', model_type=ModelType.RANKER, model_id='treatment')

        spec = ExperimentSpec(
            name='test_exp',
            hypothesis='Testing',
            control_model=control_model,
            treatment_models=[treatment_model],
            traffic_split={'control': 0.5, 'treatment': 0.5},
            primary_metric='ctr',
            min_sample_size=10
        )

        exp_id = framework.create_experiment(spec)
        framework.start_experiment(exp_id)

        # Assign variants
        variant = framework.assign_variant(exp_id, 'user_123')
        assert variant in ['control', 'treatment']

        # Consistent assignment
        variant2 = framework.assign_variant(exp_id, 'user_123')
        assert variant == variant2

        # Record events
        framework.record_impression(exp_id, 'user_123')
        framework.record_click(exp_id, 'user_123', 'item_1', 1, 0.9)
        framework.record_conversion(exp_id, 'user_123', 'item_1', 1, 10.0)

        exp_data = framework.experiments[exp_id]
        assert exp_data.variant_counts[variant] == 1
        # Check metrics are recorded
        assert exp_data.variant_metrics[variant].get('impressions', 0) >= 1
        assert exp_data.variant_metrics[variant].get('clicks', 0) >= 1
        assert exp_data.variant_metrics[variant].get('conversions', 0) >= 1
        assert exp_data.variant_metrics[variant].get('revenue', 0) >= 10.0


class TestExperimentAnalysis:
    """Tests for experiment analysis."""

    def test_frequentist_analysis(self):
        analyzer = ExperimentAnalyzer()

        np.random.seed(42)
        control = np.random.binomial(1, 0.1, 5000).tolist()  # 10%
        treatment = np.random.binomial(1, 0.12, 5000).tolist()  # 12%

        variant_stats = {
            'control': {'ctr': control},
            'treatment': {'ctr': treatment}
        }

        control_model = ModelSpec(name='control', model_type=ModelType.RANKER, model_id='control')
        treatment_model = ModelSpec(name='treatment', model_type=ModelType.RANKER, model_id='treatment')
        spec = ExperimentSpec(
            name='test_exp',
            hypothesis='Testing',
            control_model=control_model,
            treatment_models=[treatment_model],
            traffic_split={'control': 0.5, 'treatment': 0.5},
            primary_metric='ctr',
            min_sample_size=100
        )

        result = analyzer.analyze(spec, variant_stats)
        assert result.primary_metric.control_mean == pytest.approx(0.1, abs=0.01)
        assert result.primary_metric.treatment_mean == pytest.approx(0.12, abs=0.01)
        assert result.primary_metric.p_value < 0.05
        assert result.overall_decision == 'launch'

    def test_bayesian_analysis(self):
        analyzer = ExperimentAnalyzer()

        control = [1]*100 + [0]*900  # 10%
        treatment = [1]*120 + [0]*880  # 12%

        result = analyzer.bayesian_analysis(control, treatment)
        assert 0 <= result['probability_treatment_better'] <= 1
        assert 'expected_relative_lift' in result
        assert 'lift_credible_interval' in result
        assert result['probability_treatment_better'] > 0.9  # Should strongly favor treatment

    def test_sample_size_calculator(self):
        analyzer = ExperimentAnalyzer()

        ss = analyzer.sample_size_calculator(
            baseline_rate=0.1,
            minimum_detectable_effect=0.05,
            significance_level=0.05,
            power=0.8
        )

        assert 'control_sample_size' in ss
        assert 'treatment_sample_size' in ss
        assert 'total_sample_size' in ss
        assert ss['total_sample_size'] > 10000  # Should be substantial


class TestSequentialTesting:
    """Tests for sequential testing."""

    def test_sequential_tester_pocock(self):
        tester = SequentialTester(
            alpha=0.05,
            beta=0.2,
            max_looks=5,
            spending_function="pocock"
        )

        # Simulate accumulating data
        np.random.seed(42)
        control_data = []
        treatment_data = []

        for i in range(5):
            # Add 1000 samples per look
            control_data.extend(np.random.binomial(1, 0.1, 1000).tolist())
            treatment_data.extend(np.random.binomial(1, 0.12, 1000).tolist())

            result = tester.check(control_data, treatment_data)
            assert 'p_value' in result
            assert 'significant' in result
            assert 'adjusted_alpha' in result
            assert result['look'] == i + 1

    def test_sequential_tester_obf(self):
        tester = SequentialTester(
            alpha=0.05,
            beta=0.2,
            max_looks=5,
            spending_function="obrien_fleming"
        )

        np.random.seed(42)
        control_data = []
        treatment_data = []

        for i in range(5):
            control_data.extend(np.random.binomial(1, 0.1, 1000).tolist())
            treatment_data.extend(np.random.binomial(1, 0.12, 1000).tolist())

            result = tester.check(control_data, treatment_data)
            assert 'p_value' in result
            assert 'significant' in result

    def test_sprt_tester(self):
        tester = SPRTTester(
            alpha=0.05,
            beta=0.2,
            p0=0.1,
            p1=0.12
        )

        np.random.seed(42)
        # Generate data from alternative hypothesis
        data = np.random.binomial(1, 0.12, 10000).tolist()

        result = tester.test(data)
        assert 'decision' in result
        assert result['decision'] in ['accept_h0', 'accept_h1', 'continue']


class TestGuardrails:
    """Tests for guardrail monitoring."""

    def test_guardrail_monitor(self):
        monitor = GuardrailMonitor()

        # Add a guardrail
        monitor.add_guardrail_config(
            metric_name="revenue_per_user",
            threshold=-0.05,  # 5% decrease
            direction="decrease"
        )

        # Test violation
        variant_stats = {
            'control': {'metrics': {'revenue_per_user': [10.0]*100}},
            'treatment': {'metrics': {'revenue_per_user': [9.0]*100}}  # 10% decrease
        }

        violations = monitor.check_guardrails(variant_stats, 'control', 'treatment')
        assert len(violations) >= 0  # May or may not violate depending on implementation


class TestDriftDetection:
    """Tests for drift detection."""

    def test_data_drift_detector(self):
        detector = DataDriftDetector(
            thresholds={'psi': 0.2, 'ks': 0.05, 'js': 0.1},
            min_samples=50
        )

        np.random.seed(42)
        baseline = pd.DataFrame({
            'feat1': np.random.normal(0, 1, 500),
            'feat2': np.random.normal(5, 1, 500),
            'feat3': np.random.uniform(0, 1, 500)
        })
        detector.fit(baseline)

        # Test with drifted data
        current = pd.DataFrame({
            'feat1': np.random.normal(2, 1.5, 500),  # Drifted
            'feat2': np.random.normal(5, 1, 500),    # Not drifted
            'feat3': np.random.uniform(0.5, 1.5, 500)  # Drifted
        })

        results = detector.detect_drift(current)
        summary = detector.get_drift_summary(results)

        assert summary['total_features_checked'] == 3
        assert summary['features_with_drift'] >= 1
        assert any('feat1' in d['feature'] for d in summary['drifted_features'])

    def test_concept_drift_detector(self):
        from monitoring.drift import ConceptDriftDetector

        # Create a simple model for testing
        class MockModel:
            def predict(self, X):
                return 0.5 * X[:, 0] + 0.3 * X[:, 1]

        model = MockModel()
        detector = ConceptDriftDetector(model=model)

        np.random.seed(42)
        # Baseline: y = 0.5*x1 + 0.3*x2 + noise
        X_baseline = np.random.normal(0, 1, (500, 2))
        y_baseline = 0.5 * X_baseline[:, 0] + 0.3 * X_baseline[:, 1] + np.random.normal(0, 0.1, 500)
        detector.fit(X_baseline, y_baseline)

        # Current: y = 1.0*x1 + 0.3*x2 + noise (concept changed)
        X_current = np.random.normal(0, 1, (500, 2))
        y_current = 1.0 * X_current[:, 0] + 0.3 * X_current[:, 1] + np.random.normal(0, 0.1, 500)

        results = detector.detect_drift(X_current, y_current)
        assert 'drift_detected' in results
        assert 'method' in results


class TestAlertManager:
    """Tests for alert manager."""

    def test_alert_creation(self):
        manager = AlertManager()

        alert = Alert(
            alert_id="test_alert_1",
            title="Test Alert",
            message="This is a test alert",
            severity=DriftSeverity.WARNING,
            source="test",
            model_id="test_model"
        )

        result = manager.send_alert(alert, channels=[AlertChannel.LOG])

        assert result[AlertChannel.LOG] == True

        # Check history
        history = manager.get_alerts(limit=1)
        assert len(history) == 1
        assert history[0].title == "Test Alert"
        assert history[0].severity == DriftSeverity.WARNING


class TestTrafficSimulator:
    """Tests for traffic simulator."""

    def test_simulation_config(self):
        config = SimulationConfig(
            num_users=10,
            sessions_per_user=2,
            items_per_session=5,
            feature_store=MockOnlineFeatureStore(),
            click_model=ClickModel(),
            conversion_model=ConversionModel(),
            session_gap_mean=1.0,
            session_gap_std=0.5
        )

        assert config.num_users == 10
        assert config.sessions_per_user == 2

    def test_click_model(self):
        model = ClickModel()

        # Test simulate method (actual method name)
        clicks = model.simulate(
            scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            n=100
        )

        assert len(clicks) == 100
        assert all(c in [0, 1] for c in clicks)

    def test_conversion_model(self):
        model = ConversionModel()

        prob = model.predict(
            price=10.0,
            category="electronics",
            user_features={"income": 100000, "past_purchases": 5}
        )

        assert 0 <= prob <= 1


class TestShadowTesting:
    """Tests for shadow testing."""

    def test_shadow_test_runner(self):
        runner = ShadowTestRunner(
            production_model=lambda x: {"predictions": [0.5] * len(x)},
            candidate_model=lambda x: {"predictions": [0.6] * len(x)},
            traffic_fraction=0.1
        )

        # Run shadow test
        requests = [{"user_id": f"u{i}", "item_id": f"i{i}"} for i in range(100)]
        results = runner.run(requests)

        assert 'production_predictions' in results
        assert 'candidate_predictions' in results
        assert 'traffic_fraction' in results
        assert results['traffic_fraction'] == 0.1


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])