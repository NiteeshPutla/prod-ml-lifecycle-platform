"""Health checking for ML model serving infrastructure."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import httpx

from core.schemas import ServingEndpoint, ServingStatus

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    MODEL_SERVER = "model_server"
    FEATURE_STORE = "feature_store"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    component_type: ComponentType
    check_fn: Callable[[], bool]
    timeout_seconds: float = 5.0
    interval_seconds: int = 30
    critical: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    component_type: ComponentType
    status: HealthStatus
    latency_ms: float
    timestamp: datetime
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Aggregated health for a component."""
    component_type: ComponentType
    status: HealthStatus
    checks: List[HealthCheckResult]
    last_check: datetime
    uptime_percentage: float


class HealthChecker:
    """Health checking for ML infrastructure components."""

    def __init__(
        self,
        check_interval_seconds: int = 30,
        timeout_seconds: float = 10.0
    ):
        self.check_interval = check_interval_seconds
        self.default_timeout = timeout_seconds
        self.checks: Dict[str, HealthCheck] = {}
        self.results_history: Dict[str, List[HealthCheckResult]] = {}
        self._running = False
        self._check_task = None

    def register_check(self, check: HealthCheck):
        """Register a health check."""
        self.checks[check.name] = check
        logger.info(f"Registered health check: {check.name} ({check.component_type.value})")

    def register_http_check(
        self,
        name: str,
        url: str,
        component_type: ComponentType = ComponentType.MODEL_SERVER,
        expected_status: int = 200,
        timeout: float = 5.0,
        critical: bool = True
    ):
        """Register an HTTP health check."""

        async def check_fn():
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url)
                    return response.status_code == expected_status
            except Exception:
                return False

        check = HealthCheck(
            name=name,
            component_type=component_type,
            check_fn=check_fn,
            timeout_seconds=timeout,
            critical=critical,
            metadata={"url": url, "expected_status": expected_status}
        )
        self.register_check(check)

    def register_tcp_check(
        self,
        name: str,
        host: str,
        port: int,
        component_type: ComponentType = ComponentType.DATABASE,
        timeout: float = 5.0,
        critical: bool = True
    ):
        """Register a TCP connectivity check."""

        async def check_fn():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout
                )
                writer.close()
                await writer.wait_closed()
                return True
            except Exception:
                return False

        check = HealthCheck(
            name=name,
            component_type=component_type,
            check_fn=check_fn,
            timeout_seconds=timeout,
            critical=critical,
            metadata={"host": host, "port": port}
        )
        self.register_check(check)

    def register_custom_check(
        self,
        name: str,
        check_fn: Callable[[], bool],
        component_type: ComponentType,
        timeout: float = 5.0,
        critical: bool = True,
        metadata: Dict[str, Any] = None
    ):
        """Register a custom health check function."""
        check = HealthCheck(
            name=name,
            component_type=component_type,
            check_fn=check_fn,
            timeout_seconds=timeout,
            critical=critical,
            metadata=metadata or {}
        )
        self.register_check(check)

    async def run_check(self, check: HealthCheck) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()

        try:
            # Run with timeout
            if asyncio.iscoroutinefunction(check.check_fn):
                result = await asyncio.wait_for(
                    check.check_fn(),
                    timeout=check.timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(check.check_fn),
                    timeout=check.timeout_seconds
                )

            latency = (time.time() - start_time) * 1000

            if result:
                status = HealthStatus.HEALTHY
                error = None
            else:
                status = HealthStatus.UNHEALTHY
                error = "Check returned False"

        except asyncio.TimeoutError:
            latency = check.timeout_seconds * 1000
            status = HealthStatus.UNHEALTHY
            error = f"Timeout after {check.timeout_seconds}s"
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            status = HealthStatus.UNHEALTHY
            error = str(e)

        check_result = HealthCheckResult(
            name=check.name,
            component_type=check.component_type,
            status=status,
            latency_ms=latency,
            timestamp=datetime.utcnow(),
            error=error,
            metadata=check.metadata
        )

        # Store in history
        if check.name not in self.results_history:
            self.results_history[check.name] = []
        self.results_history[check.name].append(check_result)

        # Trim history
        max_history = 1000
        if len(self.results_history[check.name]) > max_history:
            self.results_history[check.name] = self.results_history[check.name][-max_history:]

        return check_result

    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        tasks = [self.run_check(check) for check in self.checks.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        check_results = {}
        for check, result in zip(self.checks.values(), results):
            if isinstance(result, Exception):
                check_results[check.name] = HealthCheckResult(
                    name=check.name,
                    component_type=check.component_type,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    timestamp=datetime.utcnow(),
                    error=str(result)
                )
            else:
                check_results[check.name] = result

        return check_results

    def get_component_health(self, component_type: ComponentType) -> ComponentHealth:
        """Get aggregated health for a component type."""
        component_checks = [
            name for name, check in self.checks.items()
            if check.component_type == component_type
        ]

        if not component_checks:
            return ComponentHealth(
                component_type=component_type,
                status=HealthStatus.UNKNOWN,
                checks=[],
                last_check=datetime.utcnow(),
                uptime_percentage=0.0
            )

        # Get latest results
        latest_results = []
        for name in component_checks:
            if name in self.results_history and self.results_history[name]:
                latest_results.append(self.results_history[name][-1])

        if not latest_results:
            return ComponentHealth(
                component_type=component_type,
                status=HealthStatus.UNKNOWN,
                checks=[],
                last_check=datetime.utcnow(),
                uptime_percentage=0.0
            )

        # Determine overall status
        critical_failed = any(
            r.status == HealthStatus.UNHEALTHY and self.checks[r.name].critical
            for r in latest_results
        )
        any_failed = any(r.status == HealthStatus.UNHEALTHY for r in latest_results)

        if critical_failed:
            overall_status = HealthStatus.UNHEALTHY
        elif any_failed:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Calculate uptime
        all_results = []
        for name in component_checks:
            if name in self.results_history:
                all_results.extend(self.results_history[name])

        if all_results:
            healthy_count = sum(1 for r in all_results if r.status == HealthStatus.HEALTHY)
            uptime = healthy_count / len(all_results) * 100
        else:
            uptime = 0.0

        return ComponentHealth(
            component_type=component_type,
            status=overall_status,
            checks=latest_results,
            last_check=max(r.timestamp for r in latest_results),
            uptime_percentage=uptime
        )

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        component_types = set(check.component_type for check in self.checks.values())

        components = {}
        overall_healthy = True
        overall_degraded = False

        for comp_type in component_types:
            health = self.get_component_health(comp_type)
            components[comp_type.value] = {
                "status": health.status.value,
                "uptime_percentage": health.uptime_percentage,
                "last_check": health.last_check.isoformat(),
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status.value,
                        "latency_ms": c.latency_ms,
                        "error": c.error
                    }
                    for c in health.checks
                ]
            }

            if health.status == HealthStatus.UNHEALTHY:
                overall_healthy = False
            elif health.status == HealthStatus.DEGRADED:
                overall_degraded = True

        if not overall_healthy:
            system_status = HealthStatus.UNHEALTHY
        elif overall_degraded:
            system_status = HealthStatus.DEGRADED
        else:
            system_status = HealthStatus.HEALTHY

        return {
            "status": system_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components
        }

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._running:
            return

        self._running = True

        async def monitor_loop():
            while self._running:
                try:
                    await self.run_all_checks()
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")

                await asyncio.sleep(self.check_interval)

        self._check_task = asyncio.create_task(monitor_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    def get_check_history(
        self,
        check_name: str,
        since: datetime = None,
        limit: int = 100
    ) -> List[HealthCheckResult]:
        """Get history for a specific check."""
        if check_name not in self.results_history:
            return []

        results = self.results_history[check_name]

        if since:
            results = [r for r in results if r.timestamp >= since]

        return results[-limit:]

    def get_uptime_report(
        self,
        component_type: ComponentType = None,
        window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get uptime report for components."""
        cutoff = datetime.utcnow() - window

        if component_type:
            components = [component_type]
        else:
            components = set(check.component_type for check in self.checks.values())

        report = {}
        for comp_type in components:
            health = self.get_component_health(comp_type)

            # Filter by window
            recent_results = []
            for check in health.checks:
                if check.timestamp >= cutoff:
                    recent_results.append(check)

            # Get all history in window
            all_in_window = []
            for name, check in self.checks.items():
                if check.component_type == comp_type:
                    for result in self.results_history.get(name, []):
                        if result.timestamp >= cutoff:
                            all_in_window.append(result)

            if all_in_window:
                healthy = sum(1 for r in all_in_window if r.status == HealthStatus.HEALTHY)
                uptime = healthy / len(all_in_window) * 100
            else:
                uptime = 0.0

            report[comp_type.value] = {
                "uptime_percentage": uptime,
                "total_checks": len(all_in_window),
                "healthy_checks": healthy if all_in_window else 0,
                "current_status": health.status.value
            }

        return report


class ModelServerHealthChecker(HealthChecker):
    """Specialized health checker for model servers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_endpoints: Dict[str, str] = {}  # model_id -> health_url

    def register_model_endpoint(self, model_id: str, health_url: str):
        """Register a model server endpoint for health checking."""
        self.model_endpoints[model_id] = health_url
        self.register_http_check(
            name=f"model_{model_id}",
            url=health_url,
            component_type=ComponentType.MODEL_SERVER,
            critical=True
        )

    async def check_model_readiness(self, model_id: str) -> bool:
        """Check if a specific model is ready to serve."""
        if model_id not in self.model_endpoints:
            return False

        url = self.model_endpoints[model_id]
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    data = response.json()
                    # Check if model is loaded
                    models_loaded = data.get("models_loaded", 0)
                    return models_loaded > 0
                return False
        except Exception:
            return False

    async def check_model_performance(
        self,
        model_id: str,
        max_latency_ms: float = 100,
        max_error_rate: float = 0.01
    ) -> Dict[str, Any]:
        """Check model performance metrics."""
        if model_id not in self.model_endpoints:
            return {"healthy": False, "reason": "Model not registered"}

        url = self.model_endpoints[model_id]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/metrics")
                if response.status_code == 200:
                    metrics = response.json()

                    avg_latency = metrics.get("avg_latency_ms", 0)
                    error_rate = metrics.get("error_rate", 0)

                    healthy = (
                        avg_latency <= max_latency_ms and
                        error_rate <= max_error_rate
                    )

                    return {
                        "healthy": healthy,
                        "avg_latency_ms": avg_latency,
                        "error_rate": error_rate,
                        "max_latency_ms": max_latency_ms,
                        "max_error_rate": max_error_rate
                    }
                return {"healthy": False, "reason": "Metrics endpoint failed"}
        except Exception as e:
            return {"healthy": False, "reason": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    async def test():
        checker = HealthChecker()

        # Register checks
        checker.register_custom_check(
            "test_check",
            lambda: True,
            ComponentType.MODEL_SERVER
        )

        checker.register_custom_check(
            "failing_check",
            lambda: False,
            ComponentType.DATABASE,
            critical=True
        )

        # Run checks
        results = await checker.run_all_checks()
        for name, result in results.items():
            print(f"{name}: {result.status.value} ({result.latency_ms:.1f}ms)")

        # Overall health
        health = checker.get_overall_health()
        print(f"\nOverall: {health['status']}")
        for comp, info in health['components'].items():
            print(f"  {comp}: {info['status']}")

    asyncio.run(test())