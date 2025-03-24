"""Reports for rule validation results."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from .rule_validation import (
    RuleValidator, ValidationResult, ValidationError,
    ValidationConfig
)

@dataclass
class ReportConfig:
    """Configuration for validation reports."""
    output_dir: Path = Path("validation_reports")
    formats: Set[str] = field(default_factory=lambda: {"html", "json", "md"})
    max_history: int = 10
    enable_trends: bool = True
    enable_summaries: bool = True
    trend_window: timedelta = timedelta(days=7)
    chart_theme: str = "plotly"
    group_by_severity: bool = True
    include_suggestions: bool = True

@dataclass
class ReportSummary:
    """Summary of validation results."""
    total_rules: int
    valid_rules: int
    invalid_rules: int
    total_errors: int
    total_warnings: int
    error_types: Dict[str, int]
    warning_types: Dict[str, int]
    stats: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class ValidationReporter:
    """Generate reports from validation results."""
    
    def __init__(
        self,
        validator: RuleValidator,
        config: ReportConfig = None
    ):
        self.validator = validator
        self.config = config or ReportConfig()
        
        # Report storage
        self.report_history: List[ReportSummary] = []
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(
        self,
        results: Dict[str, ValidationResult]
    ) -> ReportSummary:
        """Generate validation report."""
        # Calculate summary
        summary = self._create_summary(results)
        
        # Store in history
        self.report_history.append(summary)
        if len(self.report_history) > self.config.max_history:
            self.report_history.pop(0)
        
        # Generate report files
        await self._write_reports(results, summary)
        
        return summary
    
    def _create_summary(
        self,
        results: Dict[str, ValidationResult]
    ) -> ReportSummary:
        """Create report summary."""
        error_types: Dict[str, int] = {}
        warning_types: Dict[str, int] = {}
        total_errors = 0
        total_warnings = 0
        
        for result in results.values():
            total_errors += len(result.errors)
            total_warnings += len(result.warnings)
            
            for error in result.errors:
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            for warning in result.warnings:
                warning_types[warning.error_type] = warning_types.get(warning.error_type, 0) + 1
        
        return ReportSummary(
            total_rules=len(results),
            valid_rules=sum(1 for r in results.values() if r.is_valid),
            invalid_rules=sum(1 for r in results.values() if not r.is_valid),
            total_errors=total_errors,
            total_warnings=total_warnings,
            error_types=error_types,
            warning_types=warning_types,
            stats=self._aggregate_stats(results)
        )
    
    def _aggregate_stats(
        self,
        results: Dict[str, ValidationResult]
    ) -> Dict[str, Any]:
        """Aggregate validation statistics."""
        stats = {
            "avg_conditions": 0.0,
            "avg_actions": 0.0,
            "avg_size": 0,
            "max_size": 0,
            "total_fields": 0
        }
        
        if not results:
            return stats
        
        for result in results.values():
            stats["avg_conditions"] += result.stats["condition_count"]
            stats["avg_actions"] += result.stats["action_count"]
            stats["avg_size"] += result.stats["total_size"]
            stats["max_size"] = max(stats["max_size"], result.stats["total_size"])
            stats["total_fields"] += result.stats["field_count"]
        
        stats["avg_conditions"] /= len(results)
        stats["avg_actions"] /= len(results)
        stats["avg_size"] /= len(results)
        
        return stats
    
    async def _write_reports(
        self,
        results: Dict[str, ValidationResult],
        summary: ReportSummary
    ):
        """Write report files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if "html" in self.config.formats:
            await self._write_html_report(results, summary, timestamp)
        
        if "json" in self.config.formats:
            await self._write_json_report(results, summary, timestamp)
        
        if "md" in self.config.formats:
            await self._write_markdown_report(results, summary, timestamp)
    
    async def _write_html_report(
        self,
        results: Dict[str, ValidationResult],
        summary: ReportSummary,
        timestamp: str
    ):
        """Generate HTML report."""
        report_file = self.config.output_dir / f"validation_report_{timestamp}.html"
        
        # Create plots
        plots = await self._create_report_plots(results, summary)
        
        # Generate HTML
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Validation Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".summary { margin-bottom: 30px; }",
            ".error { color: red; }",
            ".warning { color: orange; }",
            ".suggestion { color: blue; font-style: italic; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Validation Report - {datetime.now().isoformat()}</h1>",
            "<div class='summary'>",
            "<h2>Summary</h2>",
            f"<p>Total Rules: {summary.total_rules}</p>",
            f"<p>Valid Rules: {summary.valid_rules}</p>",
            f"<p>Invalid Rules: {summary.invalid_rules}</p>",
            f"<p>Total Errors: {summary.total_errors}</p>",
            f"<p>Total Warnings: {summary.total_warnings}</p>",
            "</div>"
        ]
        
        # Add plots
        for name, plot in plots.items():
            html.append(f"<div id='plot_{name}'>{plot.to_html()}</div>")
        
        # Add detailed results
        html.extend([
            "<h2>Detailed Results</h2>",
            "<table>",
            "<tr><th>Rule</th><th>Status</th><th>Errors</th><th>Warnings</th></tr>"
        ])
        
        for rule_name, result in results.items():
            status = "Valid" if result.is_valid else "Invalid"
            html.append(f"<tr><td>{rule_name}</td><td>{status}</td>")
            
            # Add errors
            html.append("<td><ul>")
            for error in result.errors:
                html.append(
                    f"<li class='error'>{error.message}"
                    f"{f' <span class=\"suggestion\">Suggestion: {error.suggestion}</span>' if error.suggestion else ''}"
                    "</li>"
                )
            html.append("</ul></td>")
            
            # Add warnings
            html.append("<td><ul>")
            for warning in result.warnings:
                html.append(
                    f"<li class='warning'>{warning.message}"
                    f"{f' <span class=\"suggestion\">Suggestion: {warning.suggestion}</span>' if warning.suggestion else ''}"
                    "</li>"
                )
            html.append("</ul></td></tr>")
        
        html.extend([
            "</table>",
            "</body>",
            "</html>"
        ])
        
        with open(report_file, "w") as f:
            f.write("\n".join(html))
    
    async def _write_json_report(
        self,
        results: Dict[str, ValidationResult],
        summary: ReportSummary,
        timestamp: str
    ):
        """Generate JSON report."""
        report_file = self.config.output_dir / f"validation_report_{timestamp}.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_rules": summary.total_rules,
                "valid_rules": summary.valid_rules,
                "invalid_rules": summary.invalid_rules,
                "total_errors": summary.total_errors,
                "total_warnings": summary.total_warnings,
                "error_types": summary.error_types,
                "warning_types": summary.warning_types,
                "stats": summary.stats
            },
            "results": {
                name: {
                    "is_valid": result.is_valid,
                    "errors": [
                        {
                            "type": error.error_type,
                            "message": error.message,
                            "path": error.path,
                            "severity": error.severity,
                            "suggestion": error.suggestion
                        }
                        for error in result.errors
                    ],
                    "warnings": [
                        {
                            "type": warning.error_type,
                            "message": warning.message,
                            "path": warning.path,
                            "severity": warning.severity,
                            "suggestion": warning.suggestion
                        }
                        for warning in result.warnings
                    ],
                    "stats": result.stats
                }
                for name, result in results.items()
            }
        }
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
    
    async def _write_markdown_report(
        self,
        results: Dict[str, ValidationResult],
        summary: ReportSummary,
        timestamp: str
    ):
        """Generate Markdown report."""
        report_file = self.config.output_dir / f"validation_report_{timestamp}.md"
        
        md = [
            f"# Validation Report - {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"- Total Rules: {summary.total_rules}",
            f"- Valid Rules: {summary.valid_rules}",
            f"- Invalid Rules: {summary.invalid_rules}",
            f"- Total Errors: {summary.total_errors}",
            f"- Total Warnings: {summary.total_warnings}",
            "",
            "### Error Types",
            ""
        ]
        
        for error_type, count in summary.error_types.items():
            md.append(f"- {error_type}: {count}")
        
        md.extend([
            "",
            "### Warning Types",
            ""
        ])
        
        for warning_type, count in summary.warning_types.items():
            md.append(f"- {warning_type}: {count}")
        
        md.extend([
            "",
            "## Detailed Results",
            ""
        ])
        
        for rule_name, result in results.items():
            md.extend([
                f"### {rule_name}",
                "",
                f"Status: {'Valid' if result.is_valid else 'Invalid'}",
                "",
                "#### Errors",
                ""
            ])
            
            for error in result.errors:
                md.extend([
                    f"- **{error.error_type}**: {error.message}",
                    f"  - Path: `{error.path}`" if error.path else "",
                    f"  - Suggestion: _{error.suggestion}_" if error.suggestion else ""
                ])
            
            md.extend([
                "",
                "#### Warnings",
                ""
            ])
            
            for warning in result.warnings:
                md.extend([
                    f"- **{warning.error_type}**: {warning.message}",
                    f"  - Path: `{warning.path}`" if warning.path else "",
                    f"  - Suggestion: _{warning.suggestion}_" if warning.suggestion else ""
                ])
            
            md.extend([
                "",
                "#### Stats",
                "",
                "```json",
                json.dumps(result.stats, indent=2),
                "```",
                ""
            ])
        
        with open(report_file, "w") as f:
            f.write("\n".join(md))
    
    async def _create_report_plots(
        self,
        results: Dict[str, ValidationResult],
        summary: ReportSummary
    ) -> Dict[str, go.Figure]:
        """Create visualization plots."""
        plots = {}
        
        # Status distribution
        status_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Valid", "Invalid"],
                    values=[summary.valid_rules, summary.invalid_rules],
                    hole=0.3
                )
            ]
        )
        status_fig.update_layout(title="Rule Status Distribution")
        plots["status"] = status_fig
        
        # Error and warning types
        types_fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Error Types", "Warning Types"]
        )
        
        types_fig.add_trace(
            go.Bar(
                x=list(summary.error_types.keys()),
                y=list(summary.error_types.values()),
                name="Errors"
            ),
            row=1,
            col=1
        )
        
        types_fig.add_trace(
            go.Bar(
                x=list(summary.warning_types.keys()),
                y=list(summary.warning_types.values()),
                name="Warnings"
            ),
            row=2,
            col=1
        )
        
        types_fig.update_layout(
            height=600,
            showlegend=False,
            title="Error and Warning Types"
        )
        plots["types"] = types_fig
        
        # Trends over time
        if self.config.enable_trends and len(self.report_history) > 1:
            trend_fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Valid vs Invalid Rules",
                    "Total Errors and Warnings",
                    "Average Conditions per Rule",
                    "Average Actions per Rule"
                ]
            )
            
            # Convert history to dataframe
            history_df = pd.DataFrame([
                {
                    "timestamp": r.timestamp,
                    "valid_rules": r.valid_rules,
                    "invalid_rules": r.invalid_rules,
                    "total_errors": r.total_errors,
                    "total_warnings": r.total_warnings,
                    "avg_conditions": r.stats["avg_conditions"],
                    "avg_actions": r.stats["avg_actions"]
                }
                for r in self.report_history
            ])
            
            # Plot trends
            trend_fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["valid_rules"],
                    name="Valid Rules",
                    line=dict(color="green")
                ),
                row=1,
                col=1
            )
            
            trend_fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["invalid_rules"],
                    name="Invalid Rules",
                    line=dict(color="red")
                ),
                row=1,
                col=1
            )
            
            trend_fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["total_errors"],
                    name="Total Errors",
                    line=dict(color="red")
                ),
                row=1,
                col=2
            )
            
            trend_fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["total_warnings"],
                    name="Total Warnings",
                    line=dict(color="orange")
                ),
                row=1,
                col=2
            )
            
            trend_fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["avg_conditions"],
                    name="Avg Conditions"
                ),
                row=2,
                col=1
            )
            
            trend_fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["avg_actions"],
                    name="Avg Actions"
                ),
                row=2,
                col=2
            )
            
            trend_fig.update_layout(
                height=800,
                showlegend=True,
                title="Validation Trends"
            )
            plots["trends"] = trend_fig
        
        return plots

def create_validation_reporter(
    validator: RuleValidator,
    config: Optional[ReportConfig] = None
) -> ValidationReporter:
    """Create validation reporter."""
    return ValidationReporter(validator, config)

if __name__ == "__main__":
    from .rule_validation import create_rule_validator
    from .notification_rules import create_rule_engine
    from .alert_notifications import create_notification_manager
    from .anomaly_alerts import create_alert_manager
    from .anomaly_analysis import create_anomaly_detector
    from .trend_analysis import create_trend_analyzer
    from .adaptation_metrics import create_performance_tracker
    from .preset_adaptation import create_online_adapter
    from .preset_ensemble import create_preset_ensemble
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        adapter = create_online_adapter(ensemble)
        tracker = create_performance_tracker(adapter)
        analyzer = create_trend_analyzer(tracker)
        detector = create_anomaly_detector(tracker, analyzer)
        alert_manager = create_alert_manager(detector)
        notifier = create_notification_manager(alert_manager)
        engine = create_rule_engine(notifier)
        validator = create_rule_validator(engine)
        reporter = create_validation_reporter(validator)
        
        # Validate rules and generate report
        results = await validator.validate_all_rules()
        await reporter.generate_report(results)
    
    asyncio.run(main())
