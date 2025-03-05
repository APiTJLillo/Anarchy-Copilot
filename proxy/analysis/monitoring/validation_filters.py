"""Advanced filtering capabilities for validation analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict
import re

from .interactive_validation import InteractiveValidator
from .preset_validation import PresetValidator

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for validation filters."""
    max_conditions: int = 10
    cache_filters: bool = True
    case_sensitive: bool = False
    fuzzy_match: bool = True
    template_path: Optional[Path] = None
    save_filters: bool = True
    output_path: Optional[Path] = None

class FilterCondition:
    """Validation filter condition."""
    
    def __init__(
        self,
        field: str,
        operator: str,
        value: Any,
        negate: bool = False
    ):
        self.field = field
        self.operator = operator
        self.value = value
        self.negate = negate
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate condition against data."""
        try:
            field_value = self._get_field_value(data, self.field)
            result = self._apply_operator(field_value, self.operator, self.value)
            return not result if self.negate else result
        except Exception as e:
            logger.error(f"Failed to evaluate condition: {e}")
            return False
    
    def _get_field_value(
        self,
        data: Dict[str, Any],
        field: str
    ) -> Any:
        """Get field value using dot notation."""
        parts = field.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                value = value[int(part)]
            else:
                raise ValueError(f"Invalid field path: {field}")
        return value
    
    def _apply_operator(
        self,
        field_value: Any,
        operator: str,
        test_value: Any
    ) -> bool:
        """Apply comparison operator."""
        operators = {
            "eq": lambda x, y: x == y,
            "ne": lambda x, y: x != y,
            "gt": lambda x, y: x > y,
            "lt": lambda x, y: x < y,
            "ge": lambda x, y: x >= y,
            "le": lambda x, y: x <= y,
            "in": lambda x, y: x in y,
            "not_in": lambda x, y: x not in y,
            "contains": lambda x, y: str(y) in str(x),
            "starts_with": lambda x, y: str(x).startswith(str(y)),
            "ends_with": lambda x, y: str(x).endswith(str(y)),
            "matches": lambda x, y: bool(re.match(y, str(x))),
            "exists": lambda x, y: x is not None,
            "between": lambda x, y: y[0] <= x <= y[1]
        }
        
        if operator not in operators:
            raise ValueError(f"Invalid operator: {operator}")
        
        return operators[operator](field_value, test_value)

class ValidationFilter:
    """Filter for validation results."""
    
    def __init__(
        self,
        conditions: List[FilterCondition],
        combine: str = "and"
    ):
        self.conditions = conditions
        self.combine = combine
    
    def apply(self, data: Dict[str, Any]) -> bool:
        """Apply filter to data."""
        if not self.conditions:
            return True
        
        results = [
            condition.evaluate(data)
            for condition in self.conditions
        ]
        
        return (
            all(results) if self.combine == "and"
            else any(results)
        )

class FilterManager:
    """Manage validation filters."""
    
    def __init__(
        self,
        validator: PresetValidator,
        config: FilterConfig
    ):
        self.validator = validator
        self.config = config
        self.filters: Dict[str, ValidationFilter] = {}
        self.filter_templates: Dict[str, Dict[str, Any]] = {}
        self.cached_results: Dict[str, List[Dict[str, Any]]] = {}
        
        self._load_templates()
    
    def create_filter(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        combine: str = "and"
    ) -> ValidationFilter:
        """Create new filter."""
        if len(conditions) > self.config.max_conditions:
            raise ValueError(
                f"Too many conditions (max {self.config.max_conditions})"
            )
        
        filter_conditions = [
            FilterCondition(
                field=cond["field"],
                operator=cond["operator"],
                value=cond["value"],
                negate=cond.get("negate", False)
            )
            for cond in conditions
        ]
        
        validation_filter = ValidationFilter(filter_conditions, combine)
        self.filters[name] = validation_filter
        
        if self.config.save_filters:
            self._save_filter(name, conditions, combine)
        
        return validation_filter
    
    def apply_filter(
        self,
        name: str,
        results: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Apply named filter to results."""
        if name not in self.filters:
            raise KeyError(f"Filter not found: {name}")
        
        # Use cached results if available
        cache_key = f"{name}_{hash(str(results))}"
        if self.config.cache_filters and cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        # Get results to filter
        results = results or list(self.validator.validation_results.values())
        
        # Apply filter
        filtered = [
            result for result in results
            if self.filters[name].apply(result)
        ]
        
        # Cache results
        if self.config.cache_filters:
            self.cached_results[cache_key] = filtered
        
        return filtered
    
    def create_template(
        self,
        name: str,
        template: Dict[str, Any]
    ):
        """Create filter template."""
        self.filter_templates[name] = template
        
        if self.config.template_path:
            self._save_templates()
    
    def create_filter_from_template(
        self,
        template_name: str,
        filter_name: str,
        params: Dict[str, Any]
    ) -> ValidationFilter:
        """Create filter from template with parameters."""
        if template_name not in self.filter_templates:
            raise KeyError(f"Template not found: {template_name}")
        
        template = self.filter_templates[template_name]
        
        # Replace parameters
        conditions = self._apply_template_params(
            template["conditions"],
            params
        )
        
        return self.create_filter(
            filter_name,
            conditions,
            template.get("combine", "and")
        )
    
    def combine_filters(
        self,
        name: str,
        filter_names: List[str],
        combine: str = "and"
    ) -> ValidationFilter:
        """Combine multiple filters."""
        filters = [
            self.filters[fname]
            for fname in filter_names
            if fname in self.filters
        ]
        
        if not filters:
            raise ValueError("No valid filters to combine")
        
        # Combine conditions
        conditions = []
        for f in filters:
            conditions.extend(f.conditions)
        
        return self.create_filter(name, conditions, combine)
    
    def create_time_filter(
        self,
        name: str,
        start: datetime,
        end: Optional[datetime] = None,
        field: str = "timestamp"
    ) -> ValidationFilter:
        """Create time-based filter."""
        conditions = [{
            "field": field,
            "operator": "ge",
            "value": start.isoformat()
        }]
        
        if end:
            conditions.append({
                "field": field,
                "operator": "le",
                "value": end.isoformat()
            })
        
        return self.create_filter(name, conditions)
    
    def create_error_filter(
        self,
        name: str,
        error_types: List[str],
        combine: str = "or"
    ) -> ValidationFilter:
        """Create error type filter."""
        conditions = [
            {
                "field": f"errors.{error_type}",
                "operator": "exists",
                "value": True
            }
            for error_type in error_types
        ]
        
        return self.create_filter(name, conditions, combine)
    
    def create_fix_filter(
        self,
        name: str,
        status: str = "fixed"
    ) -> ValidationFilter:
        """Create fix status filter."""
        conditions = [{
            "field": "fixed",
            "operator": "eq",
            "value": status == "fixed"
        }]
        
        return self.create_filter(name, conditions)
    
    def _load_templates(self):
        """Load filter templates."""
        if not self.config.template_path:
            return
        
        try:
            if self.config.template_path.exists():
                with open(self.config.template_path) as f:
                    self.filter_templates = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
    
    def _save_templates(self):
        """Save filter templates."""
        if not self.config.template_path:
            return
        
        try:
            with open(self.config.template_path, "w") as f:
                json.dump(self.filter_templates, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
    
    def _save_filter(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        combine: str
    ):
        """Save filter configuration."""
        if not self.config.output_path:
            return
        
        try:
            filters_file = self.config.output_path / "filters.json"
            
            filters = {}
            if filters_file.exists():
                with open(filters_file) as f:
                    filters = json.load(f)
            
            filters[name] = {
                "conditions": conditions,
                "combine": combine,
                "created": datetime.now().isoformat()
            }
            
            with open(filters_file, "w") as f:
                json.dump(filters, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save filter: {e}")
    
    def _apply_template_params(
        self,
        conditions: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply parameters to template conditions."""
        result = []
        
        for condition in conditions:
            new_condition = condition.copy()
            
            # Replace field
            if isinstance(condition["field"], str):
                field = condition["field"]
                for param, value in params.items():
                    field = field.replace(f"{{{param}}}", str(value))
                new_condition["field"] = field
            
            # Replace value
            if isinstance(condition["value"], str):
                value = condition["value"]
                for param, pvalue in params.items():
                    value = value.replace(f"{{{param}}}", str(pvalue))
                new_condition["value"] = value
            
            result.append(new_condition)
        
        return result

def create_filter_manager(
    validator: PresetValidator,
    template_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> FilterManager:
    """Create filter manager."""
    config = FilterConfig(
        template_path=template_path,
        output_path=output_path
    )
    return FilterManager(validator, config)

if __name__ == "__main__":
    # Example usage
    from .preset_validation import create_preset_validator
    from .visualization_presets import create_preset_manager
    
    # Create components
    preset_manager = create_preset_manager()
    validator = create_preset_validator(preset_manager)
    
    # Create filter manager
    filter_manager = create_filter_manager(
        validator,
        template_path=Path("filter_templates.json"),
        output_path=Path("filters")
    )
    
    # Create filters
    error_filter = filter_manager.create_error_filter(
        "critical_errors",
        ["schema", "value_range"]
    )
    
    time_filter = filter_manager.create_time_filter(
        "recent",
        datetime.now() - timedelta(days=1)
    )
    
    # Create combined filter
    combined = filter_manager.combine_filters(
        "recent_critical",
        ["critical_errors", "recent"]
    )
    
    # Apply filter
    results = filter_manager.apply_filter("recent_critical")
    print(f"Found {len(results)} matching results")
