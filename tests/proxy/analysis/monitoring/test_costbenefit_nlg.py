"""Natural language generation for preference explanations."""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import jinja2
import logging

from .test_costbenefit_explanation import (
    PreferenceExplainer,
    PreferenceExplanation,
    ExplanationConfig
)

@dataclass
class NLGConfig:
    """Configuration for natural language generation."""
    templates_path: Optional[str] = "templates"
    style: str = "conversational"  # conversational, technical, minimal
    include_context: bool = True
    include_examples: bool = True
    include_comparisons: bool = True
    max_examples: int = 3
    max_comparisons: int = 2
    confidence_levels: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "very low": (0.0, 0.2),
        "low": (0.2, 0.4),
        "moderate": (0.4, 0.6),
        "high": (0.6, 0.8),
        "very high": (0.8, 1.0)
    })
    transition_phrases: List[str] = field(default_factory=lambda: [
        "Additionally,",
        "Moreover,",
        "Furthermore,",
        "Also,",
        "In addition,"
    ])

class NaturalLanguageGenerator:
    """Generate natural language explanations."""

    def __init__(
        self,
        explainer: PreferenceExplainer,
        config: NLGConfig
    ):
        self.explainer = explainer
        self.config = config
        
        # Initialize template engine
        self.template_env = self._setup_templates()
        
        # Initialize tracking
        self.last_context: Optional[Dict[str, Any]] = None
        
        # Load templates
        self.templates = {
            "main": self.template_env.get_template("main.j2"),
            "feature": self.template_env.get_template("feature.j2"),
            "rule": self.template_env.get_template("rule.j2"),
            "example": self.template_env.get_template("example.j2"),
            "comparison": self.template_env.get_template("comparison.j2")
        }

    def _setup_templates(self) -> jinja2.Environment:
        """Setup Jinja2 template environment."""
        if self.config.templates_path:
            loader = jinja2.FileSystemLoader(self.config.templates_path)
        else:
            # Use default templates
            loader = jinja2.DictLoader({
                "main.j2": """
                I'm {{ confidence_level }} confident about this preference decision.
                {% if main_features %}
                The most important factors were:
                {{ main_features|join('\n') }}
                {% endif %}
                {% if rules %}
                The decision was based on these rules:
                {{ rules|join('\n') }}
                {% endif %}
                {% if examples %}
                Here are some illustrative examples:
                {{ examples|join('\n') }}
                {% endif %}
                """,
                "feature.j2": """
                {{ transition }} {{ feature }} was {% if importance > 0.5 %}very{% endif %} 
                important ({{ '{:.0%}'.format(importance) }})
                """,
                "rule.j2": """
                {{ transition }} when {{ condition }}, the preference is typically {{ outcome }}
                """,
                "example.j2": """
                {{ transition }} if {{ changes }}, the preference would change to {{ outcome }}
                """,
                "comparison.j2": """
                {{ transition }} compared to {{ baseline }}, this shows {{ difference }}
                """
            })
        
        return jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True
        )

    def generate_explanation(
        self,
        explanation: PreferenceExplanation
    ) -> str:
        """Generate natural language explanation."""
        try:
            # Prepare context
            context = self._prepare_context(explanation)
            
            # Generate text
            text = self.templates["main"].render(**context)
            
            # Save context for future reference
            self.last_context = context
            
            return self._format_text(text)
            
        except Exception as e:
            logging.error(f"Error generating explanation text: {e}")
            return "Unable to generate explanation."

    def _prepare_context(
        self,
        explanation: PreferenceExplanation
    ) -> Dict[str, Any]:
        """Prepare template context from explanation."""
        context = {
            "confidence_level": self._get_confidence_level(
                explanation.confidence
            ),
            "main_features": self._describe_features(
                explanation.feature_importance
            ),
            "rules": self._describe_rules(explanation.rules),
            "examples": self._describe_examples(explanation.counterfactuals)
        }
        
        if self.config.include_context and self.last_context:
            context["previous_context"] = self.last_context
        
        return context

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to descriptive level."""
        for level, (low, high) in self.config.confidence_levels.items():
            if low <= confidence < high:
                return level
        return "unknown"

    def _describe_features(
        self,
        importance: Dict[str, float]
    ) -> List[str]:
        """Generate feature importance descriptions."""
        descriptions = []
        
        # Sort features by importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feature, imp) in enumerate(sorted_features):
            transition = self._get_transition(i)
            
            description = self.templates["feature"].render(
                transition=transition,
                feature=self._format_feature_name(feature),
                importance=imp
            )
            descriptions.append(description)
        
        return descriptions

    def _describe_rules(self, rules: List[str]) -> List[str]:
        """Generate rule descriptions."""
        descriptions = []
        
        for i, rule in enumerate(rules):
            transition = self._get_transition(i)
            
            if self.config.style == "conversational":
                # Convert technical rule to natural language
                condition = self._convert_rule_to_natural(rule)
            else:
                condition = rule
            
            description = self.templates["rule"].render(
                transition=transition,
                condition=condition,
                outcome="preferred"
            )
            descriptions.append(description)
        
        return descriptions

    def _describe_examples(
        self,
        counterfactuals: List[Dict[str, float]]
    ) -> List[str]:
        """Generate example descriptions."""
        descriptions = []
        
        for i, cf in enumerate(counterfactuals[:self.config.max_examples]):
            transition = self._get_transition(i)
            
            changes = []
            for feature, value in cf.items():
                changes.append(
                    f"{self._format_feature_name(feature)} becomes {value:.2f}"
                )
            
            description = self.templates["example"].render(
                transition=transition,
                changes=", ".join(changes),
                outcome="the opposite"
            )
            descriptions.append(description)
        
        return descriptions

    def _get_transition(self, index: int) -> str:
        """Get transition phrase for given index."""
        if index == 0:
            return ""
        return np.random.choice(self.config.transition_phrases)

    def _format_feature_name(self, name: str) -> str:
        """Format feature name for display."""
        # Convert snake/camel case to spaces
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
        return s2.replace('_', ' ').title()

    def _convert_rule_to_natural(self, rule: str) -> str:
        """Convert technical rule to natural language."""
        # Handle common patterns
        rule = rule.replace('>', 'is greater than')
        rule = rule.replace('<', 'is less than')
        rule = rule.replace('==', 'equals')
        rule = rule.replace('>=', 'is at least')
        rule = rule.replace('<=', 'is at most')
        rule = rule.replace('!=', 'is not')
        rule = rule.replace('AND', 'and')
        rule = rule.replace('OR', 'or')
        
        # Format feature names
        for feature in re.findall(r'\b\w+\b', rule):
            if feature not in ['is', 'and', 'or', 'at', 'least', 'most', 'not']:
                rule = rule.replace(
                    feature,
                    self._format_feature_name(feature)
                )
        
        return rule

    def _format_text(self, text: str) -> str:
        """Format final explanation text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Add paragraph breaks
        text = text.replace('. ', '.\n\n')
        
        # Ensure single newline between items
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def add_template(self, name: str, template: str) -> None:
        """Add custom template."""
        self.templates[name] = self.template_env.from_string(template)

    def update_transitions(self, transitions: List[str]) -> None:
        """Update transition phrases."""
        self.config.transition_phrases = transitions

@pytest.fixture
def nlg(preference_explainer):
    """Create natural language generator for testing."""
    config = NLGConfig()
    return NaturalLanguageGenerator(preference_explainer, config)

def test_confidence_level(nlg):
    """Test confidence level conversion."""
    assert nlg._get_confidence_level(0.1) == "very low"
    assert nlg._get_confidence_level(0.3) == "low"
    assert nlg._get_confidence_level(0.5) == "moderate"
    assert nlg._get_confidence_level(0.7) == "high"
    assert nlg._get_confidence_level(0.9) == "very high"

def test_feature_formatting(nlg):
    """Test feature name formatting."""
    assert nlg._format_feature_name("response_time") == "Response Time"
    assert nlg._format_feature_name("cpuUsage") == "Cpu Usage"
    assert nlg._format_feature_name("memory_usage_mb") == "Memory Usage Mb"

def test_rule_conversion(nlg):
    """Test rule conversion to natural language."""
    technical = "response_time > 100 AND cpu_usage < 0.8"
    natural = nlg._convert_rule_to_natural(technical)
    assert "Response Time" in natural
    assert "Cpu Usage" in natural
    assert "greater than" in natural
    assert "less than" in natural

@pytest.mark.asyncio
async def test_explanation_generation(nlg):
    """Test full explanation generation."""
    explanation = PreferenceExplanation(
        feature_importance={"response_time": 0.8, "cpu_usage": 0.2},
        rules=["response_time > 100"],
        counterfactuals=[{"response_time": 50.0}],
        shap_values=None,
        confidence=0.9
    )
    
    text = nlg.generate_explanation(explanation)
    
    assert text
    assert "very high" in text.lower()  # Confidence level
    assert "Response Time" in text  # Feature name
    assert "greater than" in text  # Rule conversion

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
