"""Statistical analysis for A/B experiments."""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of metric analysis."""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_diff: float
    relative_diff: float  # percentage
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    power: float


@dataclass
class ExperimentAnalysisResult:
    """Complete experiment analysis result."""
    experiment_id: str
    primary_metric: MetricResult
    secondary_metrics: List[MetricResult]
    overall_decision: str  # "launch", "no_launch", "inconclusive"
    recommendation: str
    guardrail_violations: List[str]
    sample_size_adequate: bool


class ExperimentAnalyzer:
    """Statistical analyzer for A/B experiments."""

    def __init__(
        self,
        significance_level: float = 0.05,
        minimum_detectable_effect: float = 0.01,
        test_type: str = "two_sided"  # two_sided, one_sided
    ):
        self.significance_level = significance_level
        self.minimum_detectable_effect = minimum_detectable_effect
        self.test_type = test_type

    def analyze(
        self,
        experiment: Any,  # ExperimentSpec
        variant_stats: Dict[str, Dict[str, Any]]
    ) -> ExperimentAnalysisResult:
        """Analyze experiment results."""
        # Get variants
        variants = list(variant_stats.keys())
        if "control" not in variants:
            logger.warning("No control variant found")
            control_variant = variants[0] if variants else "control"
        else:
            control_variant = "control"

        treatment_variants = [v for v in variants if v != control_variant and v != "holdout"]

        if not treatment_variants:
            return ExperimentAnalysisResult(
                experiment_id=getattr(experiment, 'experiment_id', 'unknown'),
                primary_metric=self._empty_metric(),
                secondary_metrics=[],
                overall_decision="inconclusive",
                recommendation="No treatment variants found",
                guardrail_violations=[],
                sample_size_adequate=False
            )

        # Analyze primary metric
        primary_metric_name = getattr(experiment, 'primary_metric', 'ctr')
        primary_result = self._analyze_metric(
            primary_metric_name,
            variant_stats,
            control_variant,
            treatment_variants[0]
        )

        # Analyze secondary metrics
        secondary_metrics = getattr(experiment, 'metrics', [])
        secondary_results = []
        for metric in secondary_metrics:
            if metric != primary_metric_name:
                result = self._analyze_metric(
                    metric, variant_stats, control_variant, treatment_variants[0]
                )
                secondary_results.append(result)

        # Overall decision
        decision, recommendation = self._make_decision(
            primary_result, secondary_results
        )

        # Check sample size adequacy
        sample_size_adequate = (
            primary_result.sample_size_control >= getattr(experiment, 'min_sample_size', 1000) and
            primary_result.sample_size_treatment >= getattr(experiment, 'min_sample_size', 1000)
        )

        return ExperimentAnalysisResult(
            experiment_id=getattr(experiment, 'experiment_id', 'unknown'),
            primary_metric=primary_result,
            secondary_metrics=secondary_results,
            overall_decision=decision,
            recommendation=recommendation,
            guardrail_violations=[],  # Filled by GuardrailMonitor
            sample_size_adequate=sample_size_adequate
        )

    def _analyze_metric(
        self,
        metric_name: str,
        variant_stats: Dict[str, Dict[str, Any]],
        control_variant: str,
        treatment_variant: str
    ) -> MetricResult:
        """Analyze a single metric between control and treatment."""
        control_data = variant_stats.get(control_variant, {})
        treatment_data = variant_stats.get(treatment_variant, {})

        # Handle both structures: direct metric dict or nested under "metrics"
        control_values = control_data.get(metric_name, [])
        treatment_values = treatment_data.get(metric_name, [])
        if not control_values and "metrics" in control_data:
            control_values = control_data.get("metrics", {}).get(metric_name, [])
        if not treatment_values and "metrics" in treatment_data:
            treatment_values = treatment_data.get("metrics", {}).get(metric_name, [])

        if not control_values or not treatment_values:
            return self._empty_metric(metric_name)

        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        absolute_diff = treatment_mean - control_mean
        relative_diff = (absolute_diff / control_mean * 100) if control_mean != 0 else 0

        # Statistical test
        if self.test_type == "two_sided":
            stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        else:
            # One-sided: treatment > control
            stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False, alternative='greater')

        # Confidence interval
        se = np.sqrt(
            np.var(treatment_values, ddof=1) / len(treatment_values) +
            np.var(control_values, ddof=1) / len(control_values)
        )
        df = len(treatment_values) + len(control_values) - 2
        t_critical = stats.t.ppf(1 - self.significance_level / 2, df)
        margin = t_critical * se
        ci = (absolute_diff - margin, absolute_diff + margin)

        # Power calculation
        effect_size = absolute_diff / np.sqrt(
            (np.var(treatment_values, ddof=1) + np.var(control_values, ddof=1)) / 2
        ) if len(treatment_values) > 1 and len(control_values) > 1 else 0

        power = self._calculate_power(
            effect_size, len(control_values), len(treatment_values)
        )

        is_significant = p_value < self.significance_level

        return MetricResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_diff=absolute_diff,
            relative_diff=relative_diff,
            p_value=p_value,
            confidence_interval=ci,
            is_significant=is_significant,
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values),
            power=power
        )

    def _calculate_power(
        self,
        effect_size: float,
        n_control: int,
        n_treatment: int
    ) -> float:
        """Calculate statistical power."""
        if effect_size == 0 or n_control == 0 or n_treatment == 0:
            return 0.0

        # Using normal approximation
        from scipy.stats import norm
        alpha = self.significance_level
        if self.test_type == "one_sided":
            z_alpha = norm.ppf(1 - alpha)
        else:
            z_alpha = norm.ppf(1 - alpha / 2)

        # Pooled standard error
        n_total = n_control + n_treatment
        prop_control = n_control / n_total

        se = np.sqrt(1 / (prop_control * (1 - prop_control) * n_total))
        z_power = effect_size / se - z_alpha

        power = norm.cdf(z_power)
        return max(0.0, min(1.0, power))

    def _empty_metric(self, metric_name: str = "") -> MetricResult:
        """Create empty metric result."""
        return MetricResult(
            metric_name=metric_name,
            control_mean=0.0,
            treatment_mean=0.0,
            absolute_diff=0.0,
            relative_diff=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            sample_size_control=0,
            sample_size_treatment=0,
            power=0.0
        )

    def _make_decision(
        self,
        primary: MetricResult,
        secondary: List[MetricResult]
    ) -> Tuple[str, str]:
        """Make overall launch decision."""
        # Check primary metric significance and direction
        if not primary.is_significant:
            return "inconclusive", (
                f"Primary metric ({primary.metric_name}) not statistically significant "
                f"(p={primary.p_value:.4f}). Need more data or effect is too small."
            )

        if primary.absolute_diff <= 0:
            return "no_launch", (
                f"Primary metric ({primary.metric_name}) shows negative or zero effect "
                f"(diff={primary.absolute_diff:.4f}, p={primary.p_value:.4f}). Do not launch."
            )

        # Check minimum detectable effect
        if primary.relative_diff < self.minimum_detectable_effect * 100:
            return "inconclusive", (
                f"Effect size ({primary.relative_diff:.2f}%) below minimum detectable effect "
                f"({self.minimum_detectable_effect*100:.2f}%). Consider running longer."
            )

        # Check secondary metrics for concerning trends
        concerning_secondaries = [
            m for m in secondary
            if m.is_significant and m.absolute_diff < 0 and abs(m.relative_diff) > 1.0
        ]

        if concerning_secondaries:
            return "inconclusive", (
                f"Primary metric positive but concerning secondary metrics: "
                f"{', '.join([m.metric_name for m in concerning_secondaries])}. "
                f"Investigate before launch."
            )

        return "launch", (
            f"Primary metric ({primary.metric_name}) shows statistically significant "
            f"improvement of {primary.relative_diff:.2f}% (p={primary.p_value:.4f}). "
            f"Recommend launching treatment."
        )

    def bayesian_analysis(
        self,
        control_values: List[float],
        treatment_values: List[float],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ) -> Dict[str, Any]:
        """Bayesian A/B test analysis (for binary metrics)."""
        # Assume binary metrics (0/1) for Beta-Binomial model
        control_successes = sum(control_values)
        control_failures = len(control_values) - control_successes
        treatment_successes = sum(treatment_values)
        treatment_failures = len(treatment_values) - treatment_successes

        # Posterior parameters
        post_control_alpha = prior_alpha + control_successes
        post_control_beta = prior_beta + control_failures
        post_treatment_alpha = prior_alpha + treatment_successes
        post_treatment_beta = prior_beta + treatment_failures

        # Probability treatment > control (Monte Carlo)
        n_samples = 100000
        control_samples = np.random.beta(post_control_alpha, post_control_beta, n_samples)
        treatment_samples = np.random.beta(post_treatment_alpha, post_treatment_beta, n_samples)

        prob_treatment_better = np.mean(treatment_samples > control_samples)
        expected_lift = np.mean((treatment_samples - control_samples) / control_samples)

        # Credible interval for lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        ci_lower = np.percentile(lift_samples, 2.5)
        ci_upper = np.percentile(lift_samples, 97.5)

        return {
            "probability_treatment_better": prob_treatment_better,
            "expected_relative_lift": expected_lift,
            "lift_credible_interval": (ci_lower, ci_upper),
            "posterior_control": {"alpha": post_control_alpha, "beta": post_control_beta},
            "posterior_treatment": {"alpha": post_treatment_alpha, "beta": post_treatment_beta}
        }

    def sample_size_calculator(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        significance_level: float = None,
        power: float = 0.8,
        ratio: float = 1.0
    ) -> Dict[str, int]:
        """Calculate required sample size per variant."""
        from scipy.stats import norm

        alpha = significance_level or self.significance_level

        if self.test_type == "two_sided":
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)

        z_beta = norm.ppf(power)

        # For proportion metrics
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        # Pooled proportion
        p_pool = (p1 + ratio * p2) / (1 + ratio)

        # Sample size formula
        n1 = (z_alpha * np.sqrt(p_pool * (1 - p_pool) * (1 + 1/ratio)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio))**2 / (p2 - p1)**2

        n1 = int(np.ceil(n1))
        n2 = int(np.ceil(n1 * ratio))

        return {
            "control_sample_size": n1,
            "treatment_sample_size": n2,
            "total_sample_size": n1 + n2
        }

    def sequential_analysis(
        self,
        control_values: List[float],
        treatment_values: List[float],
        alpha_spending: str = "pocock"
    ) -> Dict[str, Any]:
        """Sequential analysis for interim looks."""
        # Simplified - in production use proper sequential testing (e.g., SPRT, Alpha spending)
        n_looks = len(control_values) // 100  # Every 100 samples

        if alpha_spending == "pocock":
            # Pocock boundary - constant alpha at each look
            adjusted_alpha = self.significance_level / max(1, n_looks)
        elif alpha_spending == "obrien_fleming":
            # O'Brien-Fleming - conservative early, liberal late
            adjusted_alpha = self.significance_level / (n_looks ** 2)
        else:
            adjusted_alpha = self.significance_level

        # Run test with adjusted alpha
        old_alpha = self.significance_level
        self.significance_level = adjusted_alpha

        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)

        stat, p_value = stats.ttest_ind(
            treatment_values, control_values, equal_var=False
        )

        self.significance_level = old_alpha

        return {
            "p_value": p_value,
            "adjusted_alpha": adjusted_alpha,
            "significant": p_value < adjusted_alpha,
            "n_looks": n_looks,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    analyzer = ExperimentAnalyzer()

    # Simulate data
    np.random.seed(42)
    control = np.random.binomial(1, 0.1, 10000).tolist()  # 10% CTR
    treatment = np.random.binomial(1, 0.105, 10000).tolist()  # 10.5% CTR

    # Frequentist
    stat, p = stats.ttest_ind(treatment, control)
    print(f"Frequentist: p={p:.4f}, significant={p<0.05}")

    # Bayesian
    bayes = analyzer.bayesian_analysis(control, treatment)
    print(f"Bayesian: P(treatment better)={bayes['probability_treatment_better']:.4f}")
    print(f"  Expected lift: {bayes['expected_relative_lift']:.4f}")
    print(f"  95% CI: {bayes['lift_credible_interval']}")

    # Sample size
    ss = analyzer.sample_size_calculator(0.1, 0.05)
    print(f"Sample size per variant: {ss}")