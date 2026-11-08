"""Drift detection for data, concept, feature, and prediction drift."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import logging
import json

from core.schemas import DriftAlert, DriftType, DriftSeverity

logger = logging.getLogger(__name__)


class DriftMethod(str, Enum):
    PSI = "psi"                    # Population Stability Index
    KS = "ks"                      # Kolmogorov-Smirnov test
    CHI2 = "chi2"                  # Chi-square test
    JS = "js"                      # Jensen-Shannon divergence
    WASSERSTEIN = "wasserstein"    # Wasserstein distance
    KL = "kl"                      # KL divergence
    HELLINGER = "hellinger"        # Hellinger distance
    AD = "ad"                      # Anderson-Darling test
    CUSTOM = "custom"              # Custom function


@dataclass
class DriftResult:
    """Result of drift detection."""
    feature_name: str
    method: DriftMethod
    drift_score: float
    threshold: float
    is_drift: bool
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DriftDetector:
    """Base drift detector with multiple detection methods."""

    def __init__(
        self,
        thresholds: Dict[DriftMethod, float] = None,
        min_samples: int = 100
    ):
        self.thresholds = thresholds or {
            DriftMethod.PSI: 0.2,
            DriftMethod.KS: 0.05,
            DriftMethod.CHI2: 0.05,
            DriftMethod.JS: 0.1,
            DriftMethod.WASSERSTEIN: 0.1,
            DriftMethod.KL: 0.1,
            DriftMethod.HELLINGER: 0.1,
            DriftMethod.AD: 0.05
        }
        self.min_samples = min_samples
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}

    def set_baseline(self, feature_name: str, data: np.ndarray) -> None:
        """Set baseline distribution for a feature."""
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]

        if len(data) < self.min_samples:
            logger.warning(f"Baseline for {feature_name} has only {len(data)} samples")

        # Compute baseline statistics
        self.baseline_stats[feature_name] = {
            "data": data,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "histogram": self._compute_histogram(data),
            "ecdf": self._compute_ecdf(data),
            "sample_size": len(data)
        }

        logger.info(f"Set baseline for {feature_name}: {len(data)} samples")

    def _compute_histogram(self, data: np.ndarray, bins: int = 50) -> Dict[str, Any]:
        """Compute histogram for distribution comparison."""
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        return {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bins": bins
        }

    def _compute_ecdf(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute empirical CDF."""
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return {
            "x": sorted_data.tolist(),
            "y": y.tolist()
        }

    def detect(
        self,
        feature_name: str,
        current_data: np.ndarray,
        methods: List[DriftMethod] = None
    ) -> List[DriftResult]:
        """Detect drift for a feature."""
        if feature_name not in self.baseline_stats:
            raise ValueError(f"No baseline set for feature: {feature_name}")

        methods = methods or list(self.thresholds.keys())
        current_data = np.asarray(current_data).flatten()
        current_data = current_data[~np.isnan(current_data)]

        if len(current_data) < self.min_samples:
            logger.warning(f"Current data for {feature_name} has only {len(current_data)} samples")

        baseline = self.baseline_stats[feature_name]
        baseline_data = baseline["data"]

        results = []
        for method in methods:
            result = self._detect_method(feature_name, method, baseline_data, current_data)
            results.append(result)

        return results

    def _detect_method(
        self,
        feature_name: str,
        method: DriftMethod,
        baseline_data: np.ndarray,
        current_data: np.ndarray
    ) -> DriftResult:
        """Run specific drift detection method."""
        threshold = self.thresholds.get(method, 0.1)

        if method == DriftMethod.PSI:
            score, details = self._psi(baseline_data, current_data)
        elif method == DriftMethod.KS:
            score, p_value, details = self._ks_test(baseline_data, current_data)
        elif method == DriftMethod.CHI2:
            score, p_value, details = self._chi2_test(baseline_data, current_data)
        elif method == DriftMethod.JS:
            score, details = self._js_divergence(baseline_data, current_data)
        elif method == DriftMethod.WASSERSTEIN:
            score, details = self._wasserstein(baseline_data, current_data)
        elif method == DriftMethod.KL:
            score, details = self._kl_divergence(baseline_data, current_data)
        elif method == DriftMethod.HELLINGER:
            score, details = self._hellinger(baseline_data, current_data)
        elif method == DriftMethod.AD:
            score, p_value, details = self._anderson_darling(baseline_data, current_data)
        else:
            raise ValueError(f"Unknown method: {method}")

        is_drift = score > threshold

        return DriftResult(
            feature_name=feature_name,
            method=method,
            drift_score=score,
            threshold=threshold,
            is_drift=is_drift,
            p_value=p_value if 'p_value' in locals() else None,
            details=details
        )

    def _psi(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, Dict]:
        """Population Stability Index."""
        # Use quantile-based binning
        n_bins = 10
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(baseline, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)

        # Normalize
        baseline_pct = baseline_hist / max(1, len(baseline))
        current_pct = current_hist / max(1, len(current))

        # Avoid zeros
        epsilon = 1e-6
        baseline_pct = np.maximum(baseline_pct, epsilon)
        current_pct = np.maximum(current_pct, epsilon)

        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return float(psi), {
            "baseline_distribution": baseline_pct.tolist(),
            "current_distribution": current_pct.tolist(),
            "bins": bins.tolist()
        }

    def _ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float, Dict]:
        """Kolmogorov-Smirnov test."""
        stat, p_value = stats.ks_2samp(current, baseline, alternative='two-sided')
        return float(stat), float(p_value), {"statistic": float(stat)}

    def _chi2_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float, Dict]:
        """Chi-square test for categorical/binned data."""
        # Bin data
        n_bins = min(20, len(np.unique(baseline)))
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)

        # Chi-square test
        stat, p_value = stats.chisquare(current_hist, baseline_hist)
        return float(stat), float(p_value), {"statistic": float(stat)}

    def _js_divergence(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, Dict]:
        """Jensen-Shannon divergence."""
        # Bin data
        n_bins = 50
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
        current_hist, _ = np.histogram(current, bins=bins, density=True)

        # Normalize to probabilities
        baseline_p = baseline_hist / (np.sum(baseline_hist) + 1e-10)
        current_p = current_hist / (np.sum(current_hist) + 1e-10)

        # Add epsilon
        epsilon = 1e-10
        baseline_p = baseline_p + epsilon
        current_p = current_p + epsilon

        # Renormalize
        baseline_p = baseline_p / np.sum(baseline_p)
        current_p = current_p / np.sum(current_p)

        js = jensenshannon(baseline_p, current_p)
        return float(js**2), {"js_distance": float(js)}

    def _wasserstein(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, Dict]:
        """Wasserstein distance (Earth Mover's Distance)."""
        stat = stats.wasserstein_distance(current, baseline)
        return float(stat), {"distance": float(stat)}

    def _kl_divergence(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, Dict]:
        """KL divergence (baseline || current)."""
        n_bins = 50
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
        current_hist, _ = np.histogram(current, bins=bins, density=True)

        baseline_p = baseline_hist / (np.sum(baseline_hist) + 1e-10)
        current_p = current_hist / (np.sum(current_hist) + 1e-10)

        epsilon = 1e-10
        baseline_p = baseline_p + epsilon
        current_p = current_p + epsilon

        baseline_p = baseline_p / np.sum(baseline_p)
        current_p = current_p / np.sum(current_p)

        kl = np.sum(baseline_p * np.log(baseline_p / current_p))
        return float(kl), {"kl_divergence": float(kl)}

    def _hellinger(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, Dict]:
        """Hellinger distance."""
        n_bins = 50
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
        current_hist, _ = np.histogram(current, bins=bins, density=True)

        baseline_p = baseline_hist / (np.sum(baseline_hist) + 1e-10)
        current_p = current_hist / (np.sum(current_hist) + 1e-10)

        hellinger = np.sqrt(np.sum((np.sqrt(baseline_p) - np.sqrt(current_p))**2)) / np.sqrt(2)
        return float(hellinger), {"hellinger_distance": float(hellinger)}

    def _anderson_darling(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float, Dict]:
        """Anderson-Darling test."""
        # Use scipy's implementation
        result = stats.anderson_ksamp([baseline, current])
        stat = result.statistic
        # Approximate p-value
        p_value = result.p_value if hasattr(result, 'p_value') else 0.05
        return float(stat), float(p_value), {"statistic": float(stat)}

    def detect_all_features(
        self,
        current_data: Dict[str, np.ndarray],
        methods: List[DriftMethod] = None
    ) -> Dict[str, List[DriftResult]]:
        """Detect drift for all features with baselines."""
        results = {}
        for feature_name in self.baseline_stats:
            if feature_name in current_data:
                results[feature_name] = self.detect(feature_name, current_data[feature_name], methods)
        return results

    def create_alerts(
        self,
        model_id: str,
        results: Dict[str, List[DriftResult]],
        drift_type: DriftType = DriftType.DATA_DRIFT
    ) -> List[DriftAlert]:
        """Create drift alerts from detection results."""
        alerts = []

        for feature_name, feature_results in results.items():
            for result in feature_results:
                if result.is_drift:
                    # Determine severity
                    ratio = result.drift_score / result.threshold
                    if ratio > 3:
                        severity = DriftSeverity.CRITICAL
                    elif ratio > 2:
                        severity = DriftSeverity.HIGH
                    elif ratio > 1.5:
                        severity = DriftSeverity.MEDIUM
                    else:
                        severity = DriftSeverity.LOW

                    alert = DriftAlert(
                        model_id=model_id,
                        drift_type=drift_type,
                        severity=severity,
                        feature_name=feature_name,
                        metric_name=result.method.value,
                        current_value=result.drift_score,
                        baseline_value=0.0,  # Baseline is 0 for no drift
                        threshold=result.threshold,
                        detection_method=result.method.value,
                        description=(
                            f"Drift detected in {feature_name} using {result.method.value}: "
                            f"score={result.drift_score:.4f}, threshold={result.threshold:.4f}"
                        )
                    )
                    alerts.append(alert)

        return alerts


class DataDriftDetector(DriftDetector):
    """Specialized detector for input data drift."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> 'DataDriftDetector':
        """Fit detector on training data."""
        self.feature_names = X.columns.tolist()

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                data = X[col].dropna().values
                if len(data) >= self.min_samples:
                    self.set_baseline(col, data)

        return self

    def detect_drift(self, X: pd.DataFrame) -> Dict[str, List[DriftResult]]:
        """Detect drift in new data."""
        current_data = {}
        for col in self.feature_names:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                current_data[col] = X[col].dropna().values

        return self.detect_all_features(current_data)

    def get_drift_summary(self, results: Dict[str, List[DriftResult]]) -> Dict[str, Any]:
        """Get summary of drift detection."""
        summary = {
            "total_features_checked": len(results),
            "features_with_drift": 0,
            "drift_by_method": defaultdict(int),
            "max_drift_score": 0.0,
            "drifted_features": []
        }

        for feature_name, feature_results in results.items():
            for result in feature_results:
                if result.is_drift:
                    summary["features_with_drift"] += 1
                    method_val = result.method.value if hasattr(result.method, 'value') else result.method
                    summary["drift_by_method"][method_val] += 1
                    summary["max_drift_score"] = max(summary["max_drift_score"], result.drift_score)
                    summary["drifted_features"].append({
                        "feature": feature_name,
                        "method": method_val,
                        "score": result.drift_score,
                        "threshold": result.threshold
                    })
                    break  # Count each feature once

        return summary


class ConceptDriftDetector:
    """Detector for concept drift (P(y|x) changes)."""

    def __init__(
        self,
        model: Any,
        threshold: float = 0.1,
        window_size: int = 1000
    ):
        self.model = model
        self.threshold = threshold
        self.window_size = window_size
        self.prediction_history: List[Dict] = []
        self.baseline_performance: Dict[str, float] = {}

    def set_baseline_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str] = None
    ) -> None:
        """Set baseline model performance."""
        metrics = metrics or ["accuracy", "auc", "logloss"]

        preds = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else self.model.predict(X)

        if "accuracy" in metrics:
            self.baseline_performance["accuracy"] = (preds > 0.5).astype(int).mean() if preds.ndim == 1 else np.mean(preds == y)
        if "auc" in metrics:
            try:
                self.baseline_performance["auc"] = stats.roc_auc_score(y, preds)
            except:
                pass
        if "logloss" in metrics:
            try:
                eps = 1e-15
                preds_clipped = np.clip(preds, eps, 1 - eps)
                self.baseline_performance["logloss"] = -np.mean(y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped))
            except:
                pass

        logger.info(f"Baseline performance: {self.baseline_performance}")

    def update(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Update with new data and check for concept drift."""
        preds = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else self.model.predict(X)

        current_performance = {}
        if "accuracy" in self.baseline_performance:
            current_performance["accuracy"] = (preds > 0.5).astype(int).mean() if preds.ndim == 1 else np.mean(preds == y)
        if "auc" in self.baseline_performance:
            try:
                current_performance["auc"] = stats.roc_auc_score(y, preds)
            except:
                pass
        if "logloss" in self.baseline_performance:
            try:
                eps = 1e-15
                preds_clipped = np.clip(preds, eps, 1 - eps)
                current_performance["logloss"] = -np.mean(y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped))
            except:
                pass

        # Check for degradation
        drift_detected = False
        degradation = {}

        for metric, baseline in self.baseline_performance.items():
            if metric in current_performance:
                current = current_performance[metric]
                if metric in ["accuracy", "auc"]:
                    deg = (baseline - current) / baseline
                else:  # logloss - higher is worse
                    deg = (current - baseline) / baseline

                degradation[metric] = deg
                if deg > self.threshold:
                    drift_detected = True

        # Store in history
        self.prediction_history.append({
            "timestamp": datetime.utcnow(),
            "performance": current_performance,
            "degradation": degradation,
            "drift_detected": drift_detected,
            "sample_size": len(y)
        })

        # Trim history
        if len(self.prediction_history) > self.window_size:
            self.prediction_history = self.prediction_history[-self.window_size:]

        return {
            "drift_detected": drift_detected,
            "current_performance": current_performance,
            "baseline_performance": self.baseline_performance,
            "degradation": degradation
        }

    def get_performance_trend(self) -> pd.DataFrame:
        """Get performance trend over time."""
        if not self.prediction_history:
            return pd.DataFrame()

        return pd.DataFrame(self.prediction_history)


class FeatureDriftDetector(DriftDetector):
    """Detector for feature drift (input feature distribution changes)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_importance: Dict[str, float] = {}

    def set_feature_importance(self, importance: Dict[str, float]) -> None:
        """Set feature importance for weighted drift detection."""
        self.feature_importance = importance

    def detect_with_importance(
        self,
        current_data: Dict[str, np.ndarray],
        methods: List[DriftMethod] = None
    ) -> Dict[str, Any]:
        """Detect drift weighted by feature importance."""
        results = self.detect_all_features(current_data, methods)

        # Weight drift scores by importance
        weighted_scores = {}
        for feature_name, feature_results in results.items():
            importance = self.feature_importance.get(feature_name, 1.0)
            max_score = max(r.drift_score for r in feature_results) if feature_results else 0
            weighted_scores[feature_name] = max_score * importance

        # Overall weighted drift score
        total_importance = sum(self.feature_importance.values()) or 1
        overall_score = sum(weighted_scores.values()) / total_importance

        return {
            "feature_results": results,
            "weighted_scores": weighted_scores,
            "overall_drift_score": overall_score,
            "drift_detected": overall_score > np.mean(list(self.thresholds.values()))
        }


class PredictionDriftDetector:
    """Detector for prediction drift (output distribution changes)."""

    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 10000,
        methods: List[DriftMethod] = None
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.methods = methods or [DriftMethod.PSI, DriftMethod.KS, DriftMethod.JS]
        self.predictions_history: List[float] = []
        self.baseline_distribution: Optional[np.ndarray] = None
        self.detector = DriftDetector(thresholds={m: threshold for m in self.methods})

    def set_baseline(self, predictions: np.ndarray) -> None:
        """Set baseline prediction distribution."""
        self.baseline_distribution = predictions.flatten()
        self.detector.set_baseline("predictions", self.baseline_distribution)

    def update(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Update with new predictions and check for drift."""
        preds = predictions.flatten()
        self.predictions_history.extend(preds.tolist())

        # Trim history
        if len(self.predictions_history) > self.window_size:
            self.predictions_history = self.predictions_history[-self.window_size:]

        # Detect drift
        if self.baseline_distribution is not None:
            current_array = np.array(self.predictions_history)
            results = self.detector.detect("predictions", current_array, self.methods)

            drift_detected = any(r.is_drift for r in results)
            max_score = max(r.drift_score for r in results) if results else 0

            return {
                "drift_detected": drift_detected,
                "max_drift_score": max_score,
                "results": results,
                "current_stats": {
                    "mean": float(np.mean(current_array)),
                    "std": float(np.std(current_array)),
                    "min": float(np.min(current_array)),
                    "max": float(np.max(current_array)),
                    "sample_size": len(current_array)
                }
            }

        return {"drift_detected": False, "reason": "No baseline set"}

    def get_prediction_stats(self) -> Dict[str, float]:
        """Get current prediction statistics."""
        if not self.predictions_history:
            return {}

        arr = np.array(self.predictions_history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "sample_size": len(arr)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test data drift
    np.random.seed(42)
    baseline = np.random.normal(0, 1, 10000)
    current = np.random.normal(0.2, 1.1, 5000)  # Shifted distribution

    detector = DriftDetector()
    detector.set_baseline("feature_1", baseline)

    results = detector.detect("feature_1", current)
    for r in results:
        print(f"{r.method.value}: score={r.drift_score:.4f}, threshold={r.threshold:.4f}, drift={r.is_drift}")

    # Test concept drift
    from training.models import RankingModel
    from core.schemas import ModelSpec, ModelType

    spec = ModelSpec(name="test", model_type=ModelType.RANKER)
    model = RankingModel(spec)

    # Create dummy trained model
    X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    y = (X["f1"] + X["f2"] > 0).astype(int)
    model.fit(X, y)

    concept_detector = ConceptDriftDetector(model)
    concept_detector.set_baseline_performance(X, y)

    # Shifted data
    X_shifted = pd.DataFrame({"f1": np.random.randn(100) + 1, "f2": np.random.randn(100) + 1})
    y_shifted = (X_shifted["f1"] + X_shifted["f2"] > 1).astype(int)

    result = concept_detector.update(X_shifted, y_shifted)
    print(f"\nConcept drift: {result}")