"""Guardrail monitoring for A/B experiments."""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GuardrailStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    VIOLATION = "violation"


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    metric_name: str
    status: GuardrailStatus
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_diff: float
    threshold: float
    p_value: float
    message: str


class GuardrailMonitor:
    """Monitor guardrail metrics during experiments."""

    def __init__(
        self,
        significance_level: float = 0.05,
        warning_threshold: float = 0.5  # Fraction of violation threshold
    ):
        self.significance_level = significance_level
        self.warning_threshold = warning_threshold

    def check_guardrails(
        self,
        experiment: Any,
        variant_stats: Dict[str, Dict[str, Any]],
        guardrail_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check all guardrail metrics."""
        results = []
        violations = []
        warnings = []

        variants = list(variant_stats.keys())
        control_variant = "control" if "control" in variants else variants[0]
        treatment_variants = [v for v in variants if v != control_variant and v != "holdout"]

        if not treatment_variants:
            return {
                "guardrails": [],
                "violations": [],
                "warnings": [],
                "overall_status": GuardrailStatus.PASS.value
            }

        treatment_variant = treatment_variants[0]

        for metric_name, max_degradation in guardrail_metrics.items():
            result = self._check_metric(
                metric_name,
                variant_stats,
                control_variant,
                treatment_variant,
                max_degradation
            )
            results.append(result)

            if result.status == GuardrailStatus.VIOLATION:
                violations.append(result.metric_name)
            elif result.status == GuardrailStatus.WARNING:
                warnings.append(result.metric_name)

        overall_status = GuardrailStatus.PASS
        if violations:
            overall_status = GuardrailStatus.VIOLATION
        elif warnings:
            overall_status = GuardrailStatus.WARNING

        return {
            "guardrails": [
                {
                    "metric_name": r.metric_name,
                    "status": r.status.value,
                    "control_value": r.control_value,
                    "treatment_value": r.treatment_value,
                    "absolute_diff": r.absolute_diff,
                    "relative_diff": r.relative_diff,
                    "threshold": r.threshold,
                    "p_value": r.p_value,
                    "message": r.message
                }
                for r in results
            ],
            "violations": violations,
            "warnings": warnings,
            "overall_status": overall_status.value
        }

    def _check_metric(
        self,
        metric_name: str,
        variant_stats: Dict[str, Dict[str, Any]],
        control_variant: str,
        treatment_variant: str,
        max_degradation: float
    ) -> GuardrailResult:
        """Check a single guardrail metric."""
        control_data = variant_stats.get(control_variant, {}).get("metrics", {}).get(metric_name, [])
        treatment_data = variant_stats.get(treatment_variant, {}).get("metrics", {}).get(metric_name, [])

        if not control_data or not treatment_data:
            return GuardrailResult(
                metric_name=metric_name,
                status=GuardrailStatus.PASS,
                control_value=0.0,
                treatment_value=0.0,
                absolute_diff=0.0,
                relative_diff=0.0,
                threshold=max_degradation,
                p_value=1.0,
                message="Insufficient data for guardrail check"
            )

        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        absolute_diff = treatment_mean - control_mean
        relative_diff = (absolute_diff / control_mean) if control_mean != 0 else 0

        # Statistical test (one-sided: treatment < control * (1 - threshold))
        # We test if treatment is significantly worse than allowed
        threshold_value = control_mean * (1 - max_degradation)

        # One-sample t-test: treatment mean vs threshold
        stat, p_value = stats.ttest_1samp(treatment_data, threshold_value, alternative='less')

        # Determine status
        if p_value < self.significance_level and treatment_mean < threshold_value:
            status = GuardrailStatus.VIOLATION
            message = (
                f"VIOLATION: {metric_name} degraded by {abs(relative_diff)*100:.1f}% "
                f"(threshold: {max_degradation*100:.1f}%), p={p_value:.4f}"
            )
        elif treatment_mean < control_mean * (1 - max_degradation * self.warning_threshold):
            status = GuardrailStatus.WARNING
            message = (
                f"WARNING: {metric_name} degraded by {abs(relative_diff)*100:.1f}% "
                f"(approaching threshold: {max_degradation*100:.1f}%)"
            )
        else:
            status = GuardrailStatus.PASS
            message = f"PASS: {metric_name} within acceptable bounds"

        return GuardrailResult(
            metric_name=metric_name,
            status=status,
            control_value=control_mean,
            treatment_value=treatment_mean,
            absolute_diff=absolute_diff,
            relative_diff=relative_diff,
            threshold=max_degradation,
            p_value=p_value,
            message=message
        )

    def check_all_metrics(
        self,
        experiment: Any,
        variant_stats: Dict[str, Dict[str, Any]],
        metrics_to_check: List[str] = None
    ) -> Dict[str, GuardrailResult]:
        """Check all metrics for unexpected degradation."""
        if metrics_to_check is None:
            # Check all metrics present in data
            variants = list(variant_stats.keys())
            all_metrics = set()
            for v in variants:
                all_metrics.update(variant_stats[v].get("metrics", {}).keys())
            metrics_to_check = list(all_metrics)

        results = {}
        control_variant = "control" if "control" in variants else variants[0]
        treatment_variants = [v for v in variants if v != control_variant and v != "holdout"]

        if not treatment_variants:
            return results

        treatment_variant = treatment_variants[0]

        for metric_name in metrics_to_check:
            # Use a default threshold of 5% for general metrics
            default_threshold = 0.05
            result = self._check_metric(
                metric_name, variant_stats, control_variant,
                treatment_variant, default_threshold
            )
            results[metric_name] = result

        return results


class BusinessMetricGuardrails:
    """Guardrails for specific business metrics."""

    # Default guardrail thresholds by metric type
    DEFAULT_THRESHOLDS = {
        "revenue": 0.02,        # No more than 2% revenue decrease
        "ctr": 0.03,            # No more than 3% CTR decrease
        "cvr": 0.02,            # No more than 2% CVR decrease
        "engagement_time": 0.05, # No more than 5% engagement decrease
        "session_duration": 0.05,
        "bounce_rate": 0.10,    # No more than 10% bounce rate increase
        "error_rate": 0.0,      # Zero tolerance for error increase
        "latency_p99": 0.20,    # No more than 20% latency increase
        "cost_per_action": 0.10 # No more than 10% CPA increase
    }

    def __init__(self, custom_thresholds: Dict[str, float] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(custom_thresholds or {})}
        self.monitor = GuardrailMonitor()

    def check_business_guardrails(
        self,
        experiment: Any,
        variant_stats: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check all relevant business guardrails."""
        # Get metrics present in data
        variants = list(variant_stats.keys())
        all_metrics = set()
        for v in variants:
            all_metrics.update(variant_stats[v].get("metrics", {}).keys())

        # Filter to metrics we have thresholds for
        relevant_metrics = {
            m: self.thresholds[m] for m in all_metrics if m in self.thresholds
        }

        return self.monitor.check_guardrails(experiment, variant_stats, relevant_metrics)

    def add_threshold(self, metric: str, threshold: float):
        """Add or update a guardrail threshold."""
        self.thresholds[metric] = threshold

    def get_threshold(self, metric: str) -> float:
        """Get threshold for a metric."""
        return self.thresholds.get(metric, 0.05)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    monitor = GuardrailMonitor()

    # Simulate variant stats
    variant_stats = {
        "control": {
            "metrics": {
                "revenue": np.random.normal(10, 2, 1000).tolist(),
                "ctr": np.random.binomial(1, 0.1, 1000).tolist(),
                "latency_p99": np.random.normal(100, 10, 1000).tolist()
            }
        },
        "treatment": {
            "metrics": {
                "revenue": np.random.normal(9.5, 2, 1000).tolist(),  # 5% drop
                "ctr": np.random.binomial(1, 0.095, 1000).tolist(),  # 5% drop
                "latency_p99": np.random.normal(110, 10, 1000).tolist()  # 10% increase
            }
        }
    }

    guardrails = {
        "revenue": 0.02,
        "ctr": 0.03,
        "latency_p99": 0.20
    }

    class MockExp:
        pass

    result = monitor.check_guardrails(MockExp(), variant_stats, guardrails)
    print(f"Overall status: {result['overall_status']}")
    print(f"Violations: {result['violations']}")
    print(f"Warnings: {result['warnings']}")
    for g in result['guardrails']:
        print(f"  {g['metric_name']}: {g['status']} - {g['message']}")