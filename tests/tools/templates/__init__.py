"""Template utilities for test reporting."""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import jinja2

TEMPLATES_DIR = Path(__file__).parent
STYLES_DIR = TEMPLATES_DIR / "styles"

def copy_assets_to_output(output_dir: Path) -> None:
    """Copy template assets to output directory."""
    # Create styles directory
    styles_output = output_dir / "styles"
    styles_output.mkdir(parents=True, exist_ok=True)
    
    # Copy CSS files
    for css_file in STYLES_DIR.glob("*.css"):
        shutil.copy2(css_file, styles_output / css_file.name)

def setup_jinja_env() -> jinja2.Environment:
    """Set up Jinja environment with custom filters and settings."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    
    # Add custom filters
    def format_number(value: float, precision: int = 1) -> str:
        return f"{value:.{precision}f}"
    
    def format_percent(value: float) -> str:
        return f"{value:.1f}%"
    
    env.filters['format_number'] = format_number
    env.filters['format_percent'] = format_percent
    
    return env

def render_template(
    template_name: str,
    context: Dict[str, Any],
    output_dir: Optional[Path] = None,
    output_name: Optional[str] = None
) -> str:
    """Render a template with the given context."""
    env = setup_jinja_env()
    template = env.get_template(template_name)
    rendered = template.render(**context)
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        copy_assets_to_output(output_dir)
        
        if output_name is None:
            output_name = template_name
        
        output_path = output_dir / output_name
        output_path.write_text(rendered)
    
    return rendered

class TemplateRenderer:
    """Helper class for rendering test report templates."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template renderer."""
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.env = setup_jinja_env()

    def render_report(
        self,
        data: Dict[str, Any],
        template_name: str = "test_report.html",
        output_dir: Optional[Path] = None,
        output_name: Optional[str] = None
    ) -> str:
        """Render a test report template."""
        template = self.env.get_template(template_name)
        rendered = template.render(**data)

        if output_dir:
            # Ensure output directory exists and copy assets
            output_dir.mkdir(parents=True, exist_ok=True)
            copy_assets_to_output(output_dir)

            # Save rendered template
            if output_name is None:
                output_name = template_name
            output_path = output_dir / output_name
            output_path.write_text(rendered)

        return rendered

    def get_template(self, name: str) -> jinja2.Template:
        """Get a template by name."""
        return self.env.get_template(name)

    def get_template_path(self, name: str) -> Path:
        """Get the full path to a template."""
        return self.templates_dir / name
