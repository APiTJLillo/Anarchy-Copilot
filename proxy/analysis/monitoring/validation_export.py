"""Export capabilities for validation data and visualizations."""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import plotly.io as pio
import numpy as np

from .interactive_validation import InteractiveValidationControls
from .validation_visualization import ValidationVisualizer
from .cluster_validation import ClusterValidator

class ValidationExporter:
    """Export validation data and visualizations."""
    
    def __init__(
        self,
        controls: InteractiveValidationControls,
        export_dir: Optional[Path] = None
    ):
        self.controls = controls
        self.visualizer = controls.visualizer
        self.validator = controls.validator
        self.export_dir = export_dir or Path("validation_exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dashboard(
        self,
        format: str = "html",
        filename: Optional[str] = None
    ) -> Path:
        """Export validation dashboard."""
        fig = self.controls.create_interactive_dashboard()
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_dashboard_{timestamp}"
        
        if format == "html":
            output_file = self.export_dir / f"{filename}.html"
            fig.write_html(str(output_file))
        elif format == "png":
            output_file = self.export_dir / f"{filename}.png"
            fig.write_image(str(output_file))
        elif format == "json":
            output_file = self.export_dir / f"{filename}.json"
            pio.write_json(fig, str(output_file))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file
    
    def export_metrics(
        self,
        format: str = "csv",
        window: Optional[int] = None
    ) -> Path:
        """Export validation metrics."""
        summary = self.validator.get_validation_summary(window=window)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            output_file = self.export_dir / f"validation_metrics_{timestamp}.csv"
            self._export_metrics_csv(summary, output_file)
        elif format == "json":
            output_file = self.export_dir / f"validation_metrics_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        elif format == "excel":
            output_file = self.export_dir / f"validation_metrics_{timestamp}.xlsx"
            self._export_metrics_excel(summary, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file
    
    def _export_metrics_csv(
        self,
        summary: Dict[str, Any],
        output_file: Path
    ):
        """Export metrics to CSV format."""
        # Prepare data
        data = []
        times = list(range(len(summary["trends"]["silhouette"])))
        
        for t in times:
            row = {
                "time": t,
                "n_clusters": summary["cluster_evolution"]["sizes"][t],
                "n_alerts": summary["cluster_evolution"]["alerts"][t]
            }
            
            # Add metric values
            for metric, values in summary["trends"].items():
                row[metric] = values[t]
            
            data.append(row)
        
        # Write CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    def _export_metrics_excel(
        self,
        summary: Dict[str, Any],
        output_file: Path
    ):
        """Export metrics to Excel format."""
        writer = pd.ExcelWriter(output_file, engine="openpyxl")
        
        # Trends sheet
        trends_data = {
            "time": list(range(len(summary["trends"]["silhouette"]))),
            "n_clusters": summary["cluster_evolution"]["sizes"],
            "n_alerts": summary["cluster_evolution"]["alerts"]
        }
        trends_data.update(summary["trends"])
        pd.DataFrame(trends_data).to_excel(writer, sheet_name="Trends", index=False)
        
        # Current metrics sheet
        if summary["current"]:
            current_metrics = pd.DataFrame(
                summary["current"]["cluster_metrics"]
            ).T
            current_metrics.to_excel(writer, sheet_name="Current Metrics")
        
        # Stability metrics sheet
        if summary["cluster_stability"]:
            stability = pd.DataFrame({
                "cluster_churn": [summary["cluster_stability"]["cluster_churn"]],
                **summary["cluster_stability"]["metric_variance"]
            })
            stability.to_excel(writer, sheet_name="Stability", index=False)
        
        writer.save()
    
    def export_comparison(
        self,
        format: str = "html"
    ) -> Path:
        """Export metric comparison visualization."""
        fig = self.controls.create_interactive_comparison()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "html":
            output_file = self.export_dir / f"metric_comparison_{timestamp}.html"
            fig.write_html(str(output_file))
        elif format == "png":
            output_file = self.export_dir / f"metric_comparison_{timestamp}.png"
            fig.write_image(str(output_file))
        elif format == "json":
            output_file = self.export_dir / f"metric_comparison_{timestamp}.json"
            pio.write_json(fig, str(output_file))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file
    
    def export_full_report(
        self,
        window: Optional[int] = None
    ) -> Path:
        """Export comprehensive validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.export_dir / f"validation_report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Export visualizations
        self.export_dashboard(format="html", filename=str(report_dir / "dashboard"))
        self.export_dashboard(format="png", filename=str(report_dir / "dashboard"))
        self.export_comparison(format="html")
        
        # Export metrics in multiple formats
        self.export_metrics(format="csv", window=window)
        self.export_metrics(format="json", window=window)
        self.export_metrics(format="excel", window=window)
        
        # Generate report summary
        summary = {
            "timestamp": timestamp,
            "exports": {
                "visualizations": [
                    str(f.relative_to(self.export_dir))
                    for f in report_dir.glob("*.html")
                ],
                "data": [
                    str(f.relative_to(self.export_dir))
                    for f in report_dir.glob("*.{csv,json,xlsx}")
                ]
            },
            "validation_summary": self.validator.get_validation_summary(window=window)
        }
        
        summary_file = report_dir / "report_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return report_dir

def create_validation_exporter(
    controls: InteractiveValidationControls,
    export_dir: Optional[Path] = None
) -> ValidationExporter:
    """Create validation exporter."""
    return ValidationExporter(controls, export_dir)

if __name__ == "__main__":
    # Example usage
    from .interactive_validation import create_interactive_controls
    from .validation_visualization import create_validation_visualizer
    from .cluster_validation import create_cluster_validator
    from .alert_clustering import create_alert_clusterer
    from .alert_management import create_alert_manager
    from .realtime_anomalies import create_realtime_detector
    from .anomaly_detection import create_anomaly_detector
    from .exploration_trends import create_trend_analyzer
    
    # Create components
    analyzer = create_trend_analyzer()
    detector = create_anomaly_detector(analyzer)
    realtime = create_realtime_detector(detector)
    manager = create_alert_manager(realtime)
    clusterer = create_alert_clusterer(manager)
    validator = create_cluster_validator(clusterer)
    visualizer = create_validation_visualizer(validator)
    controls = create_interactive_controls(visualizer)
    exporter = create_validation_exporter(controls)
    
    async def main():
        # Start clustering
        await clusterer.start_clustering()
        
        # Export validations
        dashboard_html = exporter.export_dashboard(format="html")
        metrics_csv = exporter.export_metrics(format="csv")
        comparison_png = exporter.export_comparison(format="png")
        
        # Generate full report
        report_dir = exporter.export_full_report()
        print(f"Report generated at: {report_dir}")
    
    # Run example
    asyncio.run(main())
