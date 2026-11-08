"""Alert management for ML monitoring."""

import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import requests
import threading
import time

from core.schemas import DriftAlert, DriftSeverity

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


class AlertStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Unified alert structure."""
    alert_id: str
    title: str
    message: str
    severity: DriftSeverity
    source: str  # drift, performance, health, etc.
    model_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.PENDING
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "model_id": self.model_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "retry_count": self.retry_count
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    rule_id: str
    name: str
    source: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: DriftSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 60
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Centralized alert management with multiple channels."""

    def __init__(
        self,
        db_path: str = "data/alerts.db",
        default_channels: List[AlertChannel] = None
    ):
        self.db_path = db_path
        self.default_channels = default_channels or [AlertChannel.LOG]
        self.rules: Dict[str, AlertRule] = {}
        self.channel_handlers: Dict[AlertChannel, Callable] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        # Register default handlers
        self.register_handler(AlertChannel.LOG, self._send_log)
        self.register_handler(AlertChannel.WEBHOOK, self._send_webhook)

        # Start background processor
        self._processing = True
        self._processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processor_thread.start()

        self._init_db()

    def _init_db(self):
        """Initialize alerts database."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    message TEXT,
                    severity TEXT,
                    source TEXT,
                    model_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    status TEXT,
                    sent_at TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    acknowledged_by TEXT,
                    resolved_at TIMESTAMP,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_model ON alerts(model_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)
            """)
            conn.commit()

    def register_handler(self, channel: AlertChannel, handler: Callable[[Alert, Dict], bool]):
        """Register a channel handler."""
        self.channel_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel}")

    def register_rule(self, rule: AlertRule):
        """Register an alert rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Registered alert rule: {rule.name}")

    def create_alert(
        self,
        title: str,
        message: str,
        severity: DriftSeverity,
        source: str,
        model_id: str,
        metadata: Dict[str, Any] = None,
        channels: List[AlertChannel] = None
    ) -> Alert:
        """Create and queue an alert."""
        import uuid
        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            title=title,
            message=message,
            severity=severity,
            source=source,
            model_id=model_id,
            metadata=metadata or {},
            channels=channels or self.default_channels
        )

        with self._lock:
            self.alert_history.append(alert)

        # Save to DB
        self._save_alert(alert)

        logger.info(f"Created alert: {alert.alert_id} - {title} ({severity.value})")
        return alert

    def create_drift_alert(self, drift_alert: DriftAlert) -> Alert:
        """Create alert from drift detection."""
        return self.create_alert(
            title=f"Drift Detected: {drift_alert.feature_name or 'prediction'}",
            message=drift_alert.description,
            severity=drift_alert.severity,
            source="drift",
            model_id=drift_alert.model_id,
            metadata={
                "drift_type": drift_alert.drift_type.value,
                "feature_name": drift_alert.feature_name,
                "metric_name": drift_alert.metric_name,
                "current_value": drift_alert.current_value,
                "baseline_value": drift_alert.baseline_value,
                "threshold": drift_alert.threshold,
                "detection_method": drift_alert.detection_method,
                "drift_alert_id": drift_alert.alert_id
            }
        )

    def create_performance_alert(
        self,
        model_id: str,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        threshold: float,
        severity: DriftSeverity
    ) -> Alert:
        """Create alert for performance degradation."""
        return self.create_alert(
            title=f"Performance Degradation: {metric_name}",
            message=(
                f"Model {model_id} {metric_name} degraded from {baseline_value:.4f} "
                f"to {current_value:.4f} (threshold: {threshold:.4f})"
            ),
            severity=severity,
            source="performance",
            model_id=model_id,
            metadata={
                "metric_name": metric_name,
                "current_value": current_value,
                "baseline_value": baseline_value,
                "threshold": threshold,
                "relative_change": (current_value - baseline_value) / baseline_value if baseline_value != 0 else 0
            }
        )

    def create_health_alert(
        self,
        model_id: str,
        component: str,
        message: str,
        severity: DriftSeverity
    ) -> Alert:
        """Create alert for system health issues."""
        return self.create_alert(
            title=f"Health Issue: {component}",
            message=message,
            severity=severity,
            source="health",
            model_id=model_id,
            metadata={"component": component}
        )

    def _save_alert(self, alert: Alert):
        """Save alert to database."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts
                (alert_id, title, message, severity, source, model_id, metadata,
                 created_at, status, sent_at, acknowledged_at, acknowledged_by,
                 resolved_at, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.title,
                alert.message,
                alert.severity.value,
                alert.source,
                alert.model_id,
                json.dumps(alert.metadata),
                alert.created_at.isoformat(),
                alert.status.value,
                alert.sent_at.isoformat() if alert.sent_at else None,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.acknowledged_by,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.retry_count
            ))
            conn.commit()

    def send_alert(self, alert: Alert, channels: List[AlertChannel] = None) -> Dict[AlertChannel, bool]:
        """Send alert through specified channels."""
        channels = channels or alert.metadata.get("channels", self.default_channels)
        results = {}

        for channel in channels:
            handler = self.channel_handlers.get(channel)
            if not handler:
                logger.warning(f"No handler for channel: {channel}")
                results[channel] = False
                continue

            try:
                success = handler(alert, {})
                results[channel] = success
                if success:
                    alert.status = AlertStatus.SENT
                    alert.sent_at = datetime.utcnow()
                else:
                    alert.retry_count += 1
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
                results[channel] = False
                alert.retry_count += 1

        self._save_alert(alert)
        return results

    def _send_log(self, alert: Alert, config: Dict) -> bool:
        """Send alert to log."""
        log_level = {
            DriftSeverity.LOW: logging.INFO,
            DriftSeverity.MEDIUM: logging.WARNING,
            DriftSeverity.HIGH: logging.ERROR,
            DriftSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)

        logger.log(log_level, f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
        return True

    def _send_webhook(self, alert: Alert, config: Dict) -> bool:
        """Send alert to webhook."""
        webhook_url = config.get("webhook_url") or alert.metadata.get("webhook_url")
        if not webhook_url:
            return False

        try:
            response = requests.post(
                webhook_url,
                json=alert.to_dict(),
                timeout=10
            )
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False

    def configure_slack(self, webhook_url: str, channel: str = None, username: str = "MLOps Bot"):
        """Configure Slack notifications."""
        def send_slack(alert: Alert, config: Dict) -> bool:
            color = {
                DriftSeverity.LOW: "good",
                DriftSeverity.MEDIUM: "warning",
                DriftSeverity.HIGH: "danger",
                DriftSeverity.CRITICAL: "#8B0000"
            }.get(alert.severity, "warning")

            payload = {
                "username": username,
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Model", "value": alert.model_id, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Alert ID", "value": alert.alert_id, "short": True}
                    ],
                    "footer": "MLOps Platform",
                    "ts": int(alert.created_at.timestamp())
                }]
            }

            if channel:
                payload["channel"] = channel

            try:
                response = requests.post(webhook_url, json=payload, timeout=10)
                return response.status_code < 400
            except Exception as e:
                logger.error(f"Slack send failed: {e}")
                return False

        self.register_handler(AlertChannel.SLACK, send_slack)
        logger.info("Slack notifications configured")

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
        use_tls: bool = True
    ):
        """Configure email notifications."""
        def send_email(alert: Alert, config: Dict) -> bool:
            try:
                msg = MIMEMultipart()
                msg['From'] = from_email
                msg['To'] = ", ".join(to_emails)
                msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

                body = f"""
                Alert: {alert.title}
                Model: {alert.model_id}
                Severity: {alert.severity.value}
                Source: {alert.source}
                Time: {alert.created_at.isoformat()}
                Alert ID: {alert.alert_id}

                Message:
                {alert.message}

                Metadata:
                {json.dumps(alert.metadata, indent=2)}
                """

                msg.attach(MIMEText(body, 'plain'))

                server = smtplib.SMTP(smtp_host, smtp_port)
                if use_tls:
                    server.starttls()
                server.login(username, password)
                server.send_message(msg)
                server.quit()

                return True
            except Exception as e:
                logger.error(f"Email send failed: {e}")
                return False

        self.register_handler(AlertChannel.EMAIL, send_email)
        logger.info("Email notifications configured")

    def configure_pagerduty(self, integration_key: str):
        """Configure PagerDuty notifications."""
        def send_pagerduty(alert: Alert, config: Dict) -> bool:
            severity_map = {
                DriftSeverity.LOW: "info",
                DriftSeverity.MEDIUM: "warning",
                DriftSeverity.HIGH: "error",
                DriftSeverity.CRITICAL: "critical"
            }

            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "payload": {
                    "summary": alert.title,
                    "severity": severity_map.get(alert.severity, "warning"),
                    "source": f"mlops-{alert.model_id}",
                    "component": alert.source,
                    "group": "MLOps",
                    "class": alert.source,
                    "custom_details": alert.metadata
                }
            }

            try:
                response = requests.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    timeout=10
                )
                return response.status_code < 400
            except Exception as e:
                logger.error(f"PagerDuty send failed: {e}")
                return False

        self.register_handler(AlertChannel.PAGERDUTY, send_pagerduty)
        logger.info("PagerDuty notifications configured")

    def _process_queue(self):
        """Background processor for alert queue."""
        while self._processing:
            try:
                with self._lock:
                    pending = [a for a in self.alert_history if a.status == AlertStatus.PENDING]

                for alert in pending:
                    # Check cooldown
                    rule_key = f"{alert.model_id}:{alert.source}"
                    if rule_key in self.last_alert_time:
                        cooldown = timedelta(minutes=60)  # Default cooldown
                        if datetime.utcnow() - self.last_alert_time[rule_key] < cooldown:
                            continue

                    # Send alert
                    self.send_alert(alert)
                    self.last_alert_time[rule_key] = datetime.utcnow()

                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Alert processor error: {e}")
                time.sleep(30)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_at = datetime.utcnow()
                    alert.acknowledged_by = acknowledged_by
                    self._save_alert(alert)
                    return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.utcnow()
                    self._save_alert(alert)
                    return True
        return False

    def get_alerts(
        self,
        model_id: str = None,
        status: AlertStatus = None,
        severity: DriftSeverity = None,
        source: str = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts with filters."""
        with self._lock:
            alerts = self.alert_history

        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        if since:
            alerts = [a for a in alerts if a.created_at >= since]

        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[:limit]

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            alerts = self.alert_history

        stats = {
            "total": len(alerts),
            "by_status": defaultdict(int),
            "by_severity": defaultdict(int),
            "by_source": defaultdict(int),
            "by_model": defaultdict(int),
            "recent_24h": 0
        }

        cutoff = datetime.utcnow() - timedelta(hours=24)

        for alert in alerts:
            stats["by_status"][alert.status.value] += 1
            stats["by_severity"][alert.severity.value] += 1
            stats["by_source"][alert.source] += 1
            stats["by_model"][alert.model_id] += 1
            if alert.created_at >= cutoff:
                stats["recent_24h"] += 1

        return {k: dict(v) if isinstance(v, defaultdict) else v for k, v in stats.items()}

    def stop(self):
        """Stop the alert processor."""
        self._processing = False
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    manager = AlertManager()

    # Create alerts
    alert1 = manager.create_alert(
        title="Test Drift Alert",
        message="Feature user_age has drifted",
        severity=DriftSeverity.HIGH,
        source="drift",
        model_id="ranker_v1",
        metadata={"feature": "user_age", "psi": 0.35}
    )

    alert2 = manager.create_performance_alert(
        model_id="ranker_v1",
        metric_name="auc",
        current_value=0.78,
        baseline_value=0.82,
        threshold=0.02,
        severity=DriftSeverity.MEDIUM
    )

    print(f"Alert 1: {alert1.alert_id}")
    print(f"Alert 2: {alert2.alert_id}")

    # Check stats
    print(f"Stats: {manager.get_alert_stats()}")

    manager.stop()