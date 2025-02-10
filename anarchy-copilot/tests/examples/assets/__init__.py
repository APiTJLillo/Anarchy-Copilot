"""Helper functions for working with example test assets."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import jinja2
import yaml

# Asset paths
ASSETS_DIR = Path(__file__).parent
CONFIGS_DIR = ASSETS_DIR / "configs"
DATA_DIR = ASSETS_DIR / "data"
TEMPLATES_DIR = ASSETS_DIR / "templates"

def load_example_config(name: str = "example_scan.yaml") -> Dict[str, Any]:
    """Load an example scan configuration."""
    config_path = CONFIGS_DIR / name
    if not config_path.exists():
        raise FileNotFoundError(f"Example config not found: {name}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_example_findings(name: str = "example_findings.json") -> Dict[str, Any]:
    """Load example findings data."""
    findings_path = DATA_DIR / name
    if not findings_path.exists():
        raise FileNotFoundError(f"Example findings not found: {name}")
    
    with open(findings_path) as f:
        return json.load(f)

def generate_report(
    findings: Dict[str, Any],
    template_name: str = "report_template.html",
    output_dir: Optional[Path] = None,
    output_name: Optional[str] = None
) -> Path:
    """Generate an HTML report from findings using a template."""
    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Report template not found: {template_name}")

    # Set up Jinja environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    template = env.get_template(template_name)

    # Generate report
    report_html = template.render(report=findings)

    # Save report
    if output_dir is None:
        output_dir = DATA_DIR / "reports"
    output_dir.mkdir(exist_ok=True)

    if output_name is None:
        output_name = f"report_{findings['scan_id']}.html"
    
    output_path = output_dir / output_name
    output_path.write_text(report_html)
    return output_path

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configurations, with later configs taking precedence."""
    result = {}
    for config in configs:
        _deep_merge(result, config)
    return result

def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Recursively merge two dictionaries."""
    for key, value in update.items():
        if (
            key in base 
            and isinstance(base[key], dict) 
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value

class ExampleAssets:
    """Helper class for working with example assets."""

    @staticmethod
    def get_asset_path(category: str, name: str) -> Path:
        """Get path to a specific asset file."""
        category_dir = {
            "configs": CONFIGS_DIR,
            "data": DATA_DIR,
            "templates": TEMPLATES_DIR
        }.get(category)

        if category_dir is None:
            raise ValueError(f"Unknown asset category: {category}")

        asset_path = category_dir / name
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset not found: {category}/{name}")

        return asset_path

    @staticmethod
    def list_assets(category: Optional[str] = None) -> Dict[str, List[str]]:
        """List available example assets."""
        result = {}
        
        categories = ["configs", "data", "templates"]
        if category:
            if category not in categories:
                raise ValueError(f"Unknown category: {category}")
            categories = [category]

        for cat in categories:
            cat_dir = ASSETS_DIR / cat
            if cat_dir.exists():
                result[cat] = [p.name for p in cat_dir.glob("*") if p.is_file()]

        return result

    @staticmethod
    def create_example_data(
        base_findings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create example test data, optionally based on existing findings."""
        base = load_example_findings() if base_findings is None else base_findings
        
        # Create a copy to modify
        data = json.loads(json.dumps(base))
        
        # Add test-specific metadata
        data["metadata"]["test"] = True
        data["metadata"]["generated"] = True
        
        return data

def is_asset_file(path: Path) -> bool:
    """Check if a file is an example asset."""
    try:
        relative = path.relative_to(ASSETS_DIR)
        return True
    except ValueError:
        return False

__all__ = [
    'load_example_config',
    'load_example_findings',
    'generate_report',
    'merge_configs',
    'ExampleAssets',
    'is_asset_file',
    'ASSETS_DIR',
    'CONFIGS_DIR',
    'DATA_DIR',
    'TEMPLATES_DIR'
]
