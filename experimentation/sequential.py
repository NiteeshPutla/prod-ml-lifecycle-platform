"""Sequential testing for early stopping of experiments."""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StoppingDecision(str, Enum):
    CONTINUE = "continue"
    STOP_EFFICACY = "stop_efficacy"  # Treatment significantly better
    STOP_FUTILITY = "stop_futility"  # Unlikely to detect effect
    STOP_HARM = "stop_harm"  # Treatment significantly worse


@dataclass
class SequentialResult:
    """Result of sequential test."""
    decision: StoppingDecision
    p_value: float
    test_statistic: float
    boundary: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    conditional_power: float
    recommendation: str


class SequentialTester:
    """Sequential testing for early stopping using alpha spending."""

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.2,
        spending_function: str = "pocock",  # pocock, obrien_fleming, lan_demets
        max_looks: int = 5,
        minimum_looks: int = 2
    ):
        self.alpha = alpha
        self.beta = beta
        self.spending_function = spending_function
        self.max_looks = max_looks
        self.minimum_looks = minimum_looks
        self.look_history: List[Dict[str, Any]] = []

    def should_stop(
        self,
        experiment: Any,
        variant_stats: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Check if experiment should stop early."""
        result = self.analyze_sequential(experiment, variant_stats)
        return result.decision != StoppingDecision.CONTINUE, result.recommendation

    def analyze_sequential(
        self,
        experiment: Any,
        variant_stats: Dict[str, Dict[str, Any]]
    ) -> SequentialResult:
        """Perform sequential analysis."""
        # Get variants
        variants = list(variant_stats.keys())
        control_variant = "control" if "control" in variants else variants[0]
        treatment_variants = [v for v in variants if v != control_variant and v != "holdout"]

        if not treatment_variants:
            return SequentialResult(
                decision=StoppingDecision.CONTINUE,
                p_value=1.0,
                test_statistic=0.0,
                boundary=0.0,
                sample_size=0,
                confidence_interval=(0.0, 0.0),
                conditional_power=0.0,
                recommendation="No treatment variants"
            )

        treatment_variant = treatment_variants[0]
        primary_metric = getattr(experiment, 'primary_metric', 'ctr')

        control_data = variant_stats.get(control_variant, {}).get("metrics", {}).get(primary_metric, [])
        treatment_data = variant_stats.get(treatment_variant, {}).get("metrics", {}).get(primary_metric, [])

        if len(control_data) < 10 or len(treatment_data) < 10:
            return SequentialResult(
                decision=StoppingDecision.CONTINUE,
                p_value=1.0,
                test_statistic=0.0,
                boundary=0.0,
                sample_size=len(control_data) + len(treatment_data),
                confidence_interval=(0.0, 0.0),
                conditional_power=0.0,
                recommendation="Insufficient sample size for sequential testing"
            )

        # Current look number
        total_samples = len(control_data) + len(treatment_data)
        current_look = min(self.max_looks, max(1, total_samples // 1000))

        # Alpha spending boundary
        spent_alpha = self._alpha_spent(current_look, self.max_looks)

        # Run test
        if isinstance(control_data[0], (int, float)) and control_data[0] in [0, 1]:
            # Binary metric - use proportion test
            stat, p_value = self._proportion_test(control_data, treatment_data)
        else:
            # Continuous metric - use t-test
            stat, p_value = stats.ttest_ind(
                treatment_data, control_data, equal_var=False, alternative='greater'
            )

        # Confidence interval
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        diff = treatment_mean - control_mean

        se = np.sqrt(
            np.var(treatment_data, ddof=1) / len(treatment_data) +
            np.var(control_data, ddof=1) / len(control_data)
        )

        z = stats.norm.ppf(1 - spent_alpha)
        margin = z * se
        ci = (diff - margin, diff + margin)

        # Conditional power
        conditional_power = self._conditional_power(
            control_data, treatment_data, spent_alpha
        )

        # Decision
        if p_value < spent_alpha:
            if diff > 0:
                decision = StoppingDecision.STOP_EFFICACY
                recommendation = (
                    f"Stop for efficacy: p={p_value:.4f} < boundary={spent_alpha:.4f}, "
                    f"positive effect detected"
                )
            else:
                decision = StoppingDecision.STOP_HARM
                recommendation = (
                    f"Stop for harm: p={p_value:.4f} < boundary={spent_alpha:.4f}, "
                    f"negative effect detected"
                )
        elif conditional_power < 0.1 and current_look >= self.minimum_looks:
            decision = StoppingDecision.STOP_FUTILITY
            recommendation = (
                f"Stop for futility: conditional power={conditional_power:.2f} < 0.1"
            )
        else:
            decision = StoppingDecision.CONTINUE
            recommendation = (
                f"Continue: p={p_value:.4f}, boundary={spent_alpha:.4f}, "
                f"conditional power={conditional_power:.2f}"
            )

        # Record look
        self.look_history.append({
            "look": current_look,
            "total_samples": total_samples,
            "p_value": p_value,
            "boundary": spent_alpha,
            "decision": decision.value,
            "effect_size": diff,
            "conditional_power": conditional_power
        })

        return SequentialResult(
            decision=decision,
            p_value=p_value,
            test_statistic=stat,
            boundary=spent_alpha,
            sample_size=total_samples,
            confidence_interval=ci,
            conditional_power=conditional_power,
            recommendation=recommendation
        )

    def _alpha_spent(self, look: int, max_looks: int) -> float:
        """Calculate alpha spent at current look."""
        t = look / max_looks  # Information fraction

        if self.spending_function == "pocock":
            # Pocock: constant boundary
            # Use approximation: alpha* = alpha / (1 + (max_looks-1)*rho)
            # where rho is correlation between looks
            rho = 0.5
            return self.alpha / (1 + (max_looks - 1) * rho)

        elif self.spending_function == "obrien_fleming":
            # O'Brien-Fleming: conservative early
            return 2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - self.alpha/2) / np.sqrt(t))

        elif self.spending_function == "lan_demets":
            # Lan-DeMets approximation to O'Brien-Fleming
            return self.alpha * t

        else:
            return self.alpha * t

    def _proportion_test(
        self,
        control: List[float],
        treatment: List[float]
    ) -> Tuple[float, float]:
        """Two-proportion z-test."""
        n1, n2 = len(control), len(treatment)
        x1, x2 = sum(control), sum(treatment)
        p1, p2 = x1 / n1, x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)

        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p2 - p1) / se if se > 0 else 0

        p_value = 1 - stats.norm.cdf(z)  # One-sided

        return z, p_value

    def _conditional_power(
        self,
        control: List[float],
        treatment: List[float],
        alpha_spent: float
    ) -> float:
        """Calculate conditional power given current data."""
        n1, n2 = len(control), len(treatment)
        x1, x2 = sum(control), sum(treatment) if all(v in [0, 1] for v in treatment) else None

        # For continuous metrics, estimate from current effect
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        observed_effect = treatment_mean - control_mean

        # Pooled variance
        var_pool = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2

        # Project to full sample size
        target_n = max(n1, n2) * 2  # Assume we'll double sample size

        # Expected z-score at completion
        expected_z = observed_effect / np.sqrt(2 * var_pool / target_n)

        # Critical value
        z_critical = stats.norm.ppf(1 - alpha_spent)

        # Conditional power
        cp = 1 - stats.norm.cdf(z_critical - expected_z)

        return max(0.0, min(1.0, cp))

    def get_look_history(self) -> List[Dict[str, Any]]:
        """Get history of sequential looks."""
        return self.look_history

    def reset(self):
        """Reset look history."""
        self.look_history = []


class SPRTTester:
    """Sequential Probability Ratio Test for simple hypotheses."""

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.2,
        null_effect: float = 0.0,
        alt_effect: float = 0.05
    ):
        self.alpha = alpha
        self.beta = beta
        self.null_effect = null_effect
        self.alt_effect = alt_effect

        # SPRT boundaries
        self.A = (1 - beta) / alpha
        self.B = beta / (1 - alpha)

        self.log_likelihood_ratio = 0.0
        self.n_observations = 0

    def update(self, control_outcome: float, treatment_outcome: float) -> StoppingDecision:
        """Update with new observation pair."""
        # For binary outcomes, log-likelihood ratio
        if control_outcome in [0, 1] and treatment_outcome in [0, 1]:
            # Bernoulli likelihood ratio
            p0 = 0.5  # Under null, both have same rate
            p1_treat = 0.5 + self.alt_effect
            p1_control = 0.5

            # Likelihood ratio for this observation pair
            if treatment_outcome == 1:
                lr_treat = p1_treat / p0
            else:
                lr_treat = (1 - p1_treat) / (1 - p0)

            if control_outcome == 1:
                lr_control = p1_control / p0
            else:
                lr_control = (1 - p1_control) / (1 - p0)

            self.log_likelihood_ratio += np.log(lr_treat) - np.log(lr_control)

        else:
            # Continuous - normal approximation
            # Simplified: assume known variance
            var = 1.0
            self.log_likelihood_ratio += (
                (treatment_outcome - control_outcome - self.null_effect) * self.alt_effect / var
                - self.alt_effect**2 / (2 * var)
            )

        self.n_observations += 1

        # Check boundaries
        if self.log_likelihood_ratio >= np.log(self.A):
            return StoppingDecision.STOP_EFFICACY
        elif self.log_likelihood_ratio <= np.log(self.B):
            return StoppingDecision.STOP_FUTILITY
        else:
            return StoppingDecision.CONTINUE

    def get_decision(self) -> Tuple[StoppingDecision, str]:
        """Get current decision."""
        if self.log_likelihood_ratio >= np.log(self.A):
            return StoppingDecision.STOP_EFFICACY, "Reject null: treatment effective"
        elif self.log_likelihood_ratio <= np.log(self.B):
            return StoppingDecision.STOP_FUTILITY, "Accept null: treatment not effective"
        else:
            return StoppingDecision.CONTINUE, "Continue sampling"

    def reset(self):
        """Reset test."""
        self.log_likelihood_ratio = 0.0
        self.n_observations = 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test sequential tester
    tester = SequentialTester(alpha=0.05, spending_function="pocock", max_looks=5)

    # Simulate data
    np.random.seed(42)
    for look in range(1, 6):
        n = look * 2000
        control = np.random.binomial(1, 0.1, n).tolist()
        treatment = np.random.binomial(1, 0.105, n).tolist()

        variant_stats = {
            "control": {"metrics": {"ctr": control}},
            "treatment": {"metrics": {"ctr": treatment}}
        }

        class MockExp:
            primary_metric = "ctr"

        result = tester.analyze_sequential(MockExp(), variant_stats)
        print(f"Look {look}: n={result.sample_size}, p={result.p_value:.4f}, "
              f"boundary={result.boundary:.4f}, decision={result.decision.value}")

    print(f"\nLook history: {tester.get_look_history()}")