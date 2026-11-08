"""Metrics collection and aggregation for ML monitoring."""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AggregatedMetric:
    """Aggregated metric over a time window."""
    name: str
    labels: Dict[str, str]
    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    window_start: datetime
    window_end: datetime


class MetricsCollector:
    """Collect and buffer metrics for aggregation."""

    def __init__(
        self,
        max_buffer_size: int = 100000,
        flush_interval_seconds: int = 60
    ):
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval_seconds
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_buffer_size))
        self.lock = Lock()
        self.last_flush = time.time()
        self.custom_aggregators: Dict[str, Callable] = {}

    def record(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a metric value."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            metric_type=metric_type
        )

        key = self._make_key(name, labels)

        with self.lock:
            self.buffers[key].append(point)

            # Check if flush needed
            if time.time() - self.last_flush > self.flush_interval:
                self._flush()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Dict[str, str] = None
    ):
        """Increment a counter."""
        self.record(name, value, labels, MetricType.COUNTER)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """Set a gauge value."""
        self.record(name, value, labels, MetricType.GAUGE)

    def histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """Record a histogram value."""
        self.record(name, value, labels, MetricType.HISTOGRAM)

    def timing(
        self,
        name: str,
        duration_ms: float,
        labels: Dict[str, str] = None
    ):
        """Record a timing (latency) metric."""
        self.histogram(name, duration_ms, labels)

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create buffer key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _flush(self):
        """Flush buffers (placeholder for persistence)."""
        self.last_flush = time.time()
        # In production, would write to time-series DB
        logger.debug(f"Flushed {sum(len(b) for b in self.buffers.values())} metric points")

    def get_buffer(self, name: str, labels: Dict[str, str] = None) -> List[MetricPoint]:
        """Get buffered points for a metric."""
        key = self._make_key(name, labels)
        with self.lock:
            return list(self.buffers[key])

    def clear_buffer(self, name: str, labels: Dict[str, str] = None):
        """Clear buffer for a metric."""
        key = self._make_key(name, labels)
        with self.lock:
            if key in self.buffers:
                self.buffers[key].clear()

    def register_aggregator(self, name: str, func: Callable[[List[float]], float]):
        """Register custom aggregation function."""
        self.custom_aggregators[name] = func

    def get_all_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Get all buffered metrics."""
        with self.lock:
            return {k: list(v) for k, v in self.buffers.items()}


class MetricAggregator:
    """Aggregate metrics over time windows."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.default_percentiles = [50, 90, 95, 99]

    def aggregate(
        self,
        name: str,
        labels: Dict[str, str] = None,
        window: timedelta = None,
        percentiles: List[int] = None
    ) -> Optional[AggregatedMetric]:
        """Aggregate metric over time window."""
        points = self.collector.get_buffer(name, labels)

        if not points:
            return None

        # Filter by time window
        if window:
            cutoff = datetime.utcnow() - window
            points = [p for p in points if p.timestamp >= cutoff]

        if not points:
            return None

        values = [p.value for p in points]
        percentiles = percentiles or self.default_percentiles

        return AggregatedMetric(
            name=name,
            labels=labels or {},
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=np.mean(values),
            std=np.std(values),
            p50=np.percentile(values, 50),
            p90=np.percentile(values, 90),
            p95=np.percentile(values, 95),
            p99=np.percentile(values, 99),
            window_start=min(p.timestamp for p in points),
            window_end=max(p.timestamp for p in points)
        )

    def aggregate_all(
        self,
        window: timedelta = None,
        percentiles: List[int] = None
    ) -> Dict[str, AggregatedMetric]:
        """Aggregate all metrics."""
        results = {}
        all_metrics = self.collector.get_all_metrics()

        for key, points in all_metrics.items():
            if not points:
                continue

            # Extract name and labels from key
            if "{" in key:
                name = key[:key.index("{")]
                label_str = key[key.index("{")+1:key.index("}")]
                labels = dict(pair.split("=") for pair in label_str.split(",") if "=" in pair)
            else:
                name = key
                labels = {}

            # Use first point's labels if available
            if points and not labels:
                labels = points[0].labels

            agg = self.aggregate(name, labels, window, percentiles)
            if agg:
                results[key] = agg

        return results

    def get_rate(
        self,
        name: str,
        labels: Dict[str, str] = None,
        window: timedelta = timedelta(minutes=1)
    ) -> float:
        """Calculate rate (count per second) for counter metrics."""
        points = self.collector.get_buffer(name, labels)

        if not points:
            return 0.0

        cutoff = datetime.utcnow() - window
        recent = [p for p in points if p.timestamp >= cutoff]

        if not recent:
            return 0.0

        # For counters, sum the increments
        total = sum(p.value for p in recent)
        seconds = window.total_seconds()

        return total / seconds if seconds > 0 else 0.0

    def get_latency_percentiles(
        self,
        name: str,
        labels: Dict[str, str] = None,
        window: timedelta = timedelta(minutes=5),
        percentiles: List[int] = None
    ) -> Dict[int, float]:
        """Get latency percentiles."""
        agg = self.aggregate(name, labels, window, percentiles)
        if not agg:
            return {}

        percentiles = percentiles or self.default_percentiles
        return {
            p: getattr(agg, f"p{p}")
            for p in percentiles
        }


class ModelMetricsCollector:
    """Specialized metrics collector for ML models."""

    def __init__(self, collector: MetricsCollector = None):
        self.collector = collector or MetricsCollector()
        self.aggregator = MetricAggregator(self.collector)

    def record_prediction(
        self,
        model_id: str,
        model_version: str,
        latency_ms: float,
        prediction: float,
        variant: str = "control",
        cache_hit: bool = False
    ):
        """Record prediction metrics."""
        labels = {
            "model_id": model_id,
            "model_version": model_version,
            "variant": variant
        }

        self.collector.timing("model_prediction_latency_ms", latency_ms, labels)
        self.collector.histogram("model_prediction_value", prediction, labels)
        self.collector.increment("model_predictions_total", 1.0, labels)

        if cache_hit:
            self.collector.increment("model_cache_hits_total", 1.0, labels)
        else:
            self.collector.increment("model_cache_misses_total", 1.0, labels)

    def record_feedback(
        self,
        model_id: str,
        model_version: str,
        label: float,
        prediction: float,
        variant: str = "control"
    ):
        """Record feedback/label for monitoring."""
        labels = {
            "model_id": model_id,
            "model_version": model_version,
            "variant": variant
        }

        self.collector.histogram("model_label", label, labels)
        self.collector.histogram("model_prediction_vs_label", prediction - label, labels)
        self.collector.increment("model_feedback_total", 1.0, labels)

    def record_drift(
        self,
        model_id: str,
        feature_name: str,
        drift_score: float,
        threshold: float,
        method: str,
        is_drift: bool
    ):
        """Record drift detection metrics."""
        labels = {
            "model_id": model_id,
            "feature_name": feature_name,
            "method": method,
            "is_drift": str(is_drift).lower()
        }

        self.collector.gauge("model_drift_score", drift_score, labels)
        self.collector.gauge("model_drift_threshold", threshold, labels)

        if is_drift:
            self.collector.increment("model_drift_alerts_total", 1.0, labels)

    def record_experiment(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
        value: float
    ):
        """Record experiment metric."""
        labels = {
            "experiment_id": experiment_id,
            "variant": variant,
            "metric": metric_name
        }

        self.collector.gauge("experiment_metric_value", value, labels)

    def get_model_health(
        self,
        model_id: str,
        model_version: str = "latest",
        window: timedelta = timedelta(minutes=5)
    ) -> Dict[str, Any]:
        """Get model health summary."""
        base_labels = {"model_id": model_id}
        if model_version != "latest":
            base_labels["model_version"] = model_version

        # Latency
        latency_pcts = self.aggregator.get_latency_percentiles(
            "model_prediction_latency_ms", base_labels, window
        )

        # Throughput
        throughput = self.aggregator.get_rate(
            "model_predictions_total", base_labels, window
        )

        # Cache hit rate
        hits = self.aggregator.get_rate(
            "model_cache_hits_total", base_labels, window
        )
        misses = self.aggregator.get_rate(
            "model_cache_misses_total", base_labels, window
        )
        cache_hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0

        # Error rate (if tracked)
        errors = self.aggregator.get_rate(
            "model_prediction_errors_total", base_labels, window
        )
        error_rate = errors / (throughput + errors) if (throughput + errors) > 0 else 0

        return {
            "model_id": model_id,
            "model_version": model_version,
            "window_minutes": window.total_seconds() / 60,
            "latency_percentiles": latency_pcts,
            "throughput_rps": throughput,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": error_rate,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_experiment_summary(
        self,
        experiment_id: str,
        window: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """Get experiment metrics summary."""
        # Get all metrics for this experiment
        all_metrics = self.collector.get_all_metrics()

        variant_metrics = defaultdict(lambda: defaultdict(list))

        for key, points in all_metrics.items():
            if f"experiment_id={experiment_id}" not in key:
                continue

            # Extract variant and metric from labels
            for point in points:
                variant = point.labels.get("variant", "unknown")
                metric = point.labels.get("metric", "unknown")
                variant_metrics[variant][metric].append(point.value)

        summary = {}
        for variant, metrics in variant_metrics.items():
            summary[variant] = {}
            for metric, values in metrics.items():
                if values:
                    summary[variant][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                        "latest": values[-1]
                    }

        return {
            "experiment_id": experiment_id,
            "variants": summary,
            "timestamp": datetime.utcnow().isoformat()
        }


class SystemMetricsCollector:
    """Collect system-level metrics."""

    def __init__(self, collector: MetricsCollector = None):
        self.collector = collector or MetricsCollector()
        self.aggregator = MetricAggregator(self.collector)
        self._collecting = False
        self._collection_thread = None

    def start_collection(self, interval_seconds: int = 10):
        """Start background system metrics collection."""
        import threading
        import psutil

        if self._collecting:
            return

        self._collecting = True

        def collect():
            while self._collecting:
                try:
                    # CPU
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.collector.gauge("system_cpu_percent", cpu_percent)

                    # Memory
                    mem = psutil.virtual_memory()
                    self.collector.gauge("system_memory_percent", mem.percent)
                    self.collector.gauge("system_memory_available_gb", mem.available / (1024**3))

                    # Disk
                    disk = psutil.disk_usage("/")
                    self.collector.gauge("system_disk_percent", disk.percent)
                    self.collector.gauge("system_disk_free_gb", disk.free / (1024**3))

                    # Network
                    net = psutil.net_io_counters()
                    self.collector.increment("system_network_bytes_sent", net.bytes_sent)
                    self.collector.increment("system_network_bytes_recv", net.bytes_recv)

                    # Process
                    proc = psutil.Process()
                    self.collector.gauge("process_cpu_percent", proc.cpu_percent())
                    self.collector.gauge("process_memory_mb", proc.memory_info().rss / (1024**2))
                    self.collector.gauge("process_threads", proc.num_threads())

                except Exception as e:
                    logger.error(f"System metrics collection error: {e}")

                time.sleep(interval_seconds)

        self._collection_thread = threading.Thread(target=collect, daemon=True)
        self._collection_thread.start()
        logger.info("System metrics collection started")

    def stop_collection(self):
        """Stop background collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("System metrics collection stopped")

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health."""
        return self.aggregator.aggregate_all(window=timedelta(minutes=5))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    collector = MetricsCollector()
    model_metrics = ModelMetricsCollector(collector)

    # Simulate predictions
    for i in range(100):
        model_metrics.record_prediction(
            model_id="ranker_v1",
            model_version="1.0.0",
            latency_ms=np.random.exponential(10),
            prediction=np.random.random(),
            variant="control" if i % 2 == 0 else "treatment"
        )

    health = model_metrics.get_model_health("ranker_v1")
    print(f"Model health: {health}")