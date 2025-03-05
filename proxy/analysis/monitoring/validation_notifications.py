"""Email notifications for validation exports."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import aiosmtplib
import json
import jinja2
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration."""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    from_addr: Optional[str] = None
    template_dir: Optional[Path] = None

class ValidationNotifier:
    """Send notifications for validation exports."""
    
    def __init__(
        self,
        config: EmailConfig
    ):
        self.config = config
        self.template_env = self._setup_templates()
    
    def _setup_templates(self) -> jinja2.Environment:
        """Setup Jinja2 template environment."""
        template_dir = (
            self.config.template_dir or
            Path(__file__).parent / "templates"
        )
        
        if not template_dir.exists():
            template_dir.mkdir(parents=True)
            self._create_default_templates(template_dir)
        
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True
        )
    
    def _create_default_templates(
        self,
        template_dir: Path
    ):
        """Create default email templates."""
        templates = {
            "metrics_export.html": """
                <h2>Validation Metrics Export</h2>
                <p>A new validation metrics export has been generated.</p>
                <h3>Summary</h3>
                <ul>
                    <li>Export Time: {{ timestamp }}</li>
                    <li>Format: {{ format }}</li>
                    <li>Window: {{ window }} hours</li>
                    <li>Metrics Count: {{ metric_count }}</li>
                </ul>
                <h3>Key Metrics</h3>
                <ul>
                {% for name, value in key_metrics.items() %}
                    <li>{{ name }}: {{ "%.3f"|format(value) }}</li>
                {% endfor %}
                </ul>
            """,
            
            "dashboard_export.html": """
                <h2>Validation Dashboard Export</h2>
                <p>A new validation dashboard has been generated.</p>
                <h3>Details</h3>
                <ul>
                    <li>Export Time: {{ timestamp }}</li>
                    <li>Format: {{ format }}</li>
                    <li>Components: {{ components|join(", ") }}</li>
                </ul>
                {% if issues %}
                <h3>Issues Detected</h3>
                <ul>
                {% for issue in issues %}
                    <li>{{ issue }}</li>
                {% endfor %}
                </ul>
                {% endif %}
            """,
            
            "full_report.html": """
                <h2>Validation Report</h2>
                <p>A new comprehensive validation report has been generated.</p>
                <h3>Report Contents</h3>
                <ul>
                {% for section, count in sections.items() %}
                    <li>{{ section }}: {{ count }} items</li>
                {% endfor %}
                </ul>
                <h3>Key Findings</h3>
                <ul>
                {% for finding in findings %}
                    <li>{{ finding }}</li>
                {% endfor %}
                </ul>
                <h3>Recommendations</h3>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            """
        }
        
        for name, content in templates.items():
            with open(template_dir / name, "w") as f:
                f.write(content.strip())
    
    async def send_export_notification(
        self,
        recipients: List[str],
        export_type: str,
        export_path: Path,
        metadata: Dict[str, Any],
        template_name: Optional[str] = None,
        attach_export: bool = False
    ):
        """Send notification about completed export."""
        try:
            # Prepare email content
            template = self.template_env.get_template(
                template_name or f"{export_type}_export.html"
            )
            
            # Add common template variables
            template_vars = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "export_type": export_type,
                "export_path": str(export_path),
                **metadata
            }
            
            html_content = template.render(**template_vars)
            
            # Create email message
            msg = MIMEMultipart()
            msg["Subject"] = f"Validation {export_type.title()} Export"
            msg["From"] = self.config.from_addr or self.config.username
            msg["To"] = ", ".join(recipients)
            
            msg.attach(MIMEText(html_content, "html"))
            
            # Attach export file if requested
            if attach_export and export_path.exists():
                with open(export_path, "rb") as f:
                    attachment = MIMEApplication(f.read())
                    attachment.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=export_path.name
                    )
                    msg.attach(attachment)
            
            # Send email
            if self.config.use_tls:
                await aiosmtplib.send(
                    msg,
                    hostname=self.config.smtp_host,
                    port=self.config.smtp_port,
                    username=self.config.username,
                    password=self.config.password,
                    use_tls=True
                )
            else:
                await aiosmtplib.send(
                    msg,
                    hostname=self.config.smtp_host,
                    port=self.config.smtp_port,
                    username=self.config.username,
                    password=self.config.password
                )
            
            logger.info(
                f"Sent {export_type} export notification to {len(recipients)} recipients"
            )
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise
    
    async def send_metrics_notification(
        self,
        recipients: List[str],
        export_path: Path,
        metrics: Dict[str, Any],
        window: Optional[int] = None,
        attach_export: bool = False
    ):
        """Send notification about metrics export."""
        # Extract key metrics
        key_metrics = {
            name: values[-1] if isinstance(values, list) else values
            for name, values in metrics["trends"].items()
        }
        
        metadata = {
            "format": export_path.suffix[1:],
            "window": window or "all",
            "metric_count": len(metrics["trends"]),
            "key_metrics": key_metrics
        }
        
        await self.send_export_notification(
            recipients,
            "metrics",
            export_path,
            metadata,
            attach_export=attach_export
        )
    
    async def send_dashboard_notification(
        self,
        recipients: List[str],
        export_path: Path,
        components: List[str],
        issues: Optional[List[str]] = None,
        attach_export: bool = False
    ):
        """Send notification about dashboard export."""
        metadata = {
            "format": export_path.suffix[1:],
            "components": components,
            "issues": issues or []
        }
        
        await self.send_export_notification(
            recipients,
            "dashboard",
            export_path,
            metadata,
            attach_export=attach_export
        )
    
    async def send_report_notification(
        self,
        recipients: List[str],
        report_dir: Path,
        sections: Dict[str, int],
        findings: List[str],
        recommendations: List[str],
        attach_summary: bool = True
    ):
        """Send notification about full report."""
        metadata = {
            "sections": sections,
            "findings": findings,
            "recommendations": recommendations
        }
        
        # Find summary file
        summary_path = report_dir / "report_summary.json"
        
        await self.send_export_notification(
            recipients,
            "full_report",
            report_dir,
            metadata,
            attach_export=attach_summary and summary_path.exists()
        )

def create_validation_notifier(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    use_tls: bool = True,
    from_addr: Optional[str] = None,
    template_dir: Optional[Path] = None
) -> ValidationNotifier:
    """Create validation notifier."""
    config = EmailConfig(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        username=username,
        password=password,
        use_tls=use_tls,
        from_addr=from_addr,
        template_dir=template_dir
    )
    return ValidationNotifier(config)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create notifier
        notifier = create_validation_notifier(
            "smtp.example.com",
            587,
            "user@example.com",
            "password"
        )
        
        # Example metrics notification
        await notifier.send_metrics_notification(
            ["admin@example.com"],
            Path("exports/metrics.csv"),
            {
                "trends": {
                    "silhouette": [0.8, 0.85, 0.82],
                    "calinski_harabasz": [120, 125, 118]
                }
            },
            window=24
        )
        
        # Example dashboard notification
        await notifier.send_dashboard_notification(
            ["admin@example.com"],
            Path("exports/dashboard.html"),
            ["metrics", "clusters", "patterns"],
            issues=["Low cluster cohesion detected"]
        )
        
        # Example report notification
        await notifier.send_report_notification(
            ["admin@example.com"],
            Path("exports/report"),
            {
                "Metrics": 10,
                "Visualizations": 5,
                "Analysis": 3
            },
            findings=["Cluster stability improved", "New pattern detected"],
            recommendations=["Increase min_cluster_size"]
        )
    
    asyncio.run(main())
