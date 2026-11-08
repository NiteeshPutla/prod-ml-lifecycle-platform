import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.schemas import ExperimentSpec, ModelSpec, ModelType
from experimentation.framework import ABTestFramework
from experimentation.analysis import ExperimentAnalyzer, ExperimentAnalyzer as Analyzer
from monitoring.drift import DataDriftDetector
from features.offline_store import OfflineFeatureStore
from simulator import TrafficSimulator, SimulationConfig, ClickModel, ConversionModel
from features.online_store import MockOnlineFeatureStore


def test_offline_store(tmp_path):
    db_path = tmp_path / "offline_store.db"
    store = OfflineFeatureStore(db_path=str(db_path))

    # Test sample ingestion with event_timestamp
    now = datetime.utcnow()
    store.ingest_features("user_features", pd.DataFrame({
        "user_id": ["u1", "u2"],
        "event_timestamp": [now, now],
        "user_age": [25, 30],
        "user_signup_days": [10, 20]
    }))

    stats = store.compute_feature_statistics("user_features", ["user_age"], datetime.utcnow() - timedelta(days=1), datetime.utcnow())
    assert stats["user_age"]["count"] == 2
    assert stats["user_age"]["mean"] == 27.5


def test_experiment_framework():
    framework = ABTestFramework()
    control_model = ModelSpec(name='control', model_type=ModelType.RANKER, model_id='control')
    treatment_model = ModelSpec(name='treatment', model_type=ModelType.RANKER, model_id='treatment')

    spec = ExperimentSpec(
        name='test_exp',
        hypothesis='Testing framework',
        control_model=control_model,
        treatment_models=[treatment_model],
        traffic_split={'control': 0.5, 'treatment': 0.5},
        primary_metric='ctr',
        min_sample_size=10
    )

    exp_id = framework.create_experiment(spec)
    framework.start_experiment(exp_id)

    variant = framework.assign_variant(exp_id, 'user_123')
    assert variant in ['control', 'treatment']

    framework.record_impression(exp_id, 'user_123')
    framework.record_click(exp_id, 'user_123', 'item_1', 1, 0.9)

    exp_data = framework.experiments[exp_id]
    assert exp_data.variant_counts[variant] == 1


def test_experiment_analyzer():
    analyzer = Analyzer()

    np.random.seed(42)
    control = np.random.binomial(1, 0.1, 1000).tolist()
    treatment = np.random.binomial(1, 0.15, 1000).tolist()

    variant_stats = {
        'control': {'ctr': control},
        'treatment': {'ctr': treatment}
    }

    control_model = ModelSpec(name='control', model_type=ModelType.RANKER, model_id='control')
    treatment_model = ModelSpec(name='treatment', model_type=ModelType.RANKER, model_id='treatment')
    spec = ExperimentSpec(
        name='test_exp',
        hypothesis='Testing analysis',
        control_model=control_model,
        treatment_models=[treatment_model],
        traffic_split={'control': 0.5, 'treatment': 0.5},
        primary_metric='ctr',
        min_sample_size=100
    )

    result = analyzer.analyze(spec, variant_stats)
    assert result.primary_metric.control_mean == pytest.approx(0.1, abs=0.03)
    assert result.primary_metric.treatment_mean == pytest.approx(0.15, abs=0.03)
    assert result.primary_metric.p_value < 0.05
    assert result.overall_decision == 'launch'


def test_drift_detector():
    detector = DataDriftDetector(thresholds={'psi': 0.2, 'ks': 0.05}, min_samples=50)

    np.random.seed(42)
    baseline = pd.DataFrame({
        'feat1': np.random.normal(0, 1, 500),
        'feat2': np.random.normal(5, 1, 500)
    })
    detector.fit(baseline)

    current = pd.DataFrame({
        'feat1': np.random.normal(2, 1, 500),  # Drifted
        'feat2': np.random.normal(5, 1, 500)   # Not drifted
    })

    results = detector.detect_drift(current)
    summary = detector.get_drift_summary(results)

    assert summary["features_with_drift"] == 1
    assert summary["drifted_features"][0]["feature"] == 'feat1'
