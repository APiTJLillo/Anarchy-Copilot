"""Test summarizer module for generating test reports."""
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

from jinja2 import Environment, FileSystemLoader

class TestSummarizer:
    """Test summary and report generation."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize test summarizer.
        
        Args:
            template_dir: Optional custom template directory
        """
        if template_dir:
            loader = FileSystemLoader(str(template_dir))
        else:
            loader = FileSystemLoader("tests/tools/templates")
        self.env = Environment(loader=loader, autoescape=True)
        
    def generate_report(
        self,
        test_results: Dict,
        output_path: str = "test-reports/report.html",
        template_path: Optional[Path] = None,
    ) -> None:
        """Generate HTML test report.
        
        Args:
            test_results: Dictionary containing test results data
            output_path: Path to save report
            template_path: Optional custom template path
        """
        if template_path:
            template = self.env.get_template(str(template_path))
        else:
            template = self.env.get_template("test_report.html")
            
        html = template.render(results=test_results)
        
        with open(output_path, "w") as f:
            f.write(html)
            
    def summarize_tests(self, results: List[Dict]) -> Dict:
        """Generate summary of test results.
        
        Args:
            results: List of test result dictionaries

        Returns:
            Summary dictionary with statistics and findings
        """
        summary = {
            "total_tests": len(results),
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "findings": []
        }
        
        for result in results:
            status = result.get("outcome", "error")
            if status == "passed":
                summary["passed"] += 1
            elif status == "failed": 
                summary["failed"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            else:
                summary["error"] += 1
            
            if result.get("findings"):
                summary["findings"].extend(result["findings"])
                
        return summary
                
    def save_summary(self, summary: Dict, path: str) -> None:
        """Save summary to JSON file.
        
        Args:
            summary: Test summary dictionary  
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
