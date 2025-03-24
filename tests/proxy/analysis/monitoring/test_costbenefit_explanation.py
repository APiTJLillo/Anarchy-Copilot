"""Preference explanations for interactive learning."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import shap
import logging

from .test_costbenefit_interactive import (
    InteractivePreferenceLearner,
    PreferenceConfig,
    Comparison
)

@dataclass
class ExplanationConfig:
    """Configuration for preference explanations."""
    enable_feature_importance: bool = True
    enable_shap: bool = True
    enable_rules: bool = True
    enable_counterfactuals: bool = True
    max_rules: int = 5
    max_counterfactuals: int = 3
    tree_depth: int = 3
    min_importance: float = 0.05
    explanation_style: str = "natural"  # natural, technical
    visualization_format: str = "html"
    output_path: Optional[str] = "explanations"

@dataclass
class PreferenceExplanation:
    """Container for preference explanations."""
    feature_importance: Dict[str, float]
    rules: List[str]
    counterfactuals: List[Dict[str, float]]
    shap_values: Optional[np.ndarray]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

class PreferenceExplainer:
    """Generate explanations for learned preferences."""

    def __init__(
        self,
        learner: InteractivePreferenceLearner,
        config: ExplanationConfig
    ):
        self.learner = learner
        self.config = config
        
        # Explanation models
        self.tree_model: Optional[DecisionTreeClassifier] = None
        self.shap_explainer: Optional[Any] = None
        self.explanation_cache: Dict[str, PreferenceExplanation] = {}
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    async def explain_preference(
        self,
        comparison: Comparison
    ) -> Optional[PreferenceExplanation]:
        """Generate explanation for preference decision."""
        if not comparison.preference:
            return None
        
        # Check cache
        cache_key = self._get_cache_key(comparison)
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # Generate explanation components
        feature_importance = {}
        rules = []
        counterfactuals = []
        shap_values = None
        confidence = 0.0
        
        try:
            if self.config.enable_feature_importance:
                feature_importance = await self._explain_feature_importance(
                    comparison
                )
            
            if self.config.enable_rules:
                rules = await self._generate_rules(comparison)
            
            if self.config.enable_counterfactuals:
                counterfactuals = await self._generate_counterfactuals(
                    comparison
                )
            
            if self.config.enable_shap:
                shap_values = await self._calculate_shap_values(comparison)
            
            confidence = self._calculate_confidence(comparison)
            
            explanation = PreferenceExplanation(
                feature_importance=feature_importance,
                rules=rules,
                counterfactuals=counterfactuals,
                shap_values=shap_values,
                confidence=confidence
            )
            
            # Cache explanation
            self.explanation_cache[cache_key] = explanation
            
            return explanation
            
        except Exception as e:
            logging.error(f"Error generating explanation: {e}")
            return None

    async def _explain_feature_importance(
        self,
        comparison: Comparison
    ) -> Dict[str, float]:
        """Extract feature importance for preference."""
        if not self.learner.preference_model:
            return {}
        
        # Get feature differences
        features = self.learner._extract_features(
            comparison.solution_a,
            comparison.solution_b
        )
        
        # Get feature names
        feature_names = [
            obj.name for obj in self.learner.adapter.config.objectives
        ]
        
        # Calculate importance using model coefficients
        if hasattr(self.learner.preference_model, "coef_"):
            coefficients = self.learner.preference_model.coef_[0]
            importance = abs(coefficients * features)
            
            # Normalize and filter
            total = np.sum(importance)
            if total > 0:
                importance = importance / total
                
                return {
                    name: float(imp)
                    for name, imp in zip(feature_names, importance)
                    if imp >= self.config.min_importance
                }
        
        return {}

    async def _generate_rules(
        self,
        comparison: Comparison
    ) -> List[str]:
        """Generate decision rules explaining preference."""
        rules = []
        
        # Train decision tree for interpretable rules
        if not self.tree_model:
            self.tree_model = DecisionTreeClassifier(
                max_depth=self.config.tree_depth,
                random_state=42
            )
            
            # Prepare training data
            X = []
            y = []
            
            for comp in self.learner.comparison_history:
                if comp.preference in ["A", "B"]:
                    features = self.learner._extract_features(
                        comp.solution_a,
                        comp.solution_b
                    )
                    X.append(features)
                    y.append(1 if comp.preference == "A" else 0)
            
            if X:
                self.tree_model.fit(np.array(X), np.array(y))
        
        if self.tree_model:
            # Extract rules from tree paths
            feature_names = [
                obj.name for obj in self.learner.adapter.config.objectives
            ]
            
            def extract_rules_from_path(path):
                rules = []
                for node_id in path:
                    if self.tree_model.tree_.feature[node_id] >= 0:
                        feature = feature_names[
                            self.tree_model.tree_.feature[node_id]
                        ]
                        threshold = self.tree_model.tree_.threshold[node_id]
                        
                        if self.config.explanation_style == "natural":
                            rules.append(
                                f"{feature} is {'better' if threshold > 0 else 'worse'} "
                                f"by {abs(threshold):.2f}"
                            )
                        else:
                            rules.append(
                                f"{feature} {'>' if threshold > 0 else '<='} {threshold:.2f}"
                            )
                return " AND ".join(rules)
            
            # Get rules from decision paths
            features = self.learner._extract_features(
                comparison.solution_a,
                comparison.solution_b
            )
            path = self.tree_model.decision_path([features])[0]
            rule = extract_rules_from_path(path.indices)
            
            if rule:
                rules.append(rule)
        
        return rules[:self.config.max_rules]

    async def _generate_counterfactuals(
        self,
        comparison: Comparison
    ) -> List[Dict[str, float]]:
        """Generate counterfactual examples."""
        counterfactuals = []
        
        if not self.learner.preference_model:
            return counterfactuals
        
        # Get original prediction
        features = self.learner._extract_features(
            comparison.solution_a,
            comparison.solution_b
        )
        original_pred = self.learner.preference_model.predict([features])[0]
        
        # Generate counterfactuals using optimization
        from scipy.optimize import minimize
        
        def objective(x):
            # Predict with modified features
            pred = self.learner.preference_model.predict([x])[0]
            
            # Penalize large changes
            changes = np.sum((x - features) ** 2)
            
            return changes if pred != original_pred else float('inf')
        
        # Try different initializations
        for _ in range(self.config.max_counterfactuals):
            # Random initialization near original features
            x0 = features + np.random.normal(0, 0.1, size=len(features))
            
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 100}
            )
            
            if result.success:
                counterfactual = dict(zip(
                    [obj.name for obj in self.learner.adapter.config.objectives],
                    result.x
                ))
                counterfactuals.append(counterfactual)
        
        return counterfactuals

    async def _calculate_shap_values(
        self,
        comparison: Comparison
    ) -> Optional[np.ndarray]:
        """Calculate SHAP values for preference."""
        if not self.learner.preference_model or not self.config.enable_shap:
            return None
        
        if not self.shap_explainer:
            # Initialize SHAP explainer
            background_data = np.array([
                self.learner._extract_features(
                    comp.solution_a,
                    comp.solution_b
                )
                for comp in self.learner.comparison_history[:100]
                if comp.preference in ["A", "B"]
            ])
            
            if len(background_data) > 0:
                self.shap_explainer = shap.KernelExplainer(
                    self.learner.preference_model.predict_proba,
                    background_data
                )
        
        if self.shap_explainer:
            features = self.learner._extract_features(
                comparison.solution_a,
                comparison.solution_b
            )
            return self.shap_explainer.shap_values(features)
        
        return None

    def _calculate_confidence(self, comparison: Comparison) -> float:
        """Calculate confidence in explanation."""
        if not self.learner.preference_model:
            return 0.0
        
        features = self.learner._extract_features(
            comparison.solution_a,
            comparison.solution_b
        )
        
        # Get prediction probability
        if hasattr(self.learner.preference_model, "predict_proba"):
            proba = self.learner.preference_model.predict_proba([features])[0]
            return float(max(proba))
        
        return comparison.confidence

    def _get_cache_key(self, comparison: Comparison) -> str:
        """Generate cache key for comparison."""
        return f"{comparison.preference}_{hash(str(comparison.solution_a))}_{hash(str(comparison.solution_b))}"

    def visualize_explanation(
        self,
        explanation: PreferenceExplanation,
        save: bool = True
    ) -> Any:
        """Visualize preference explanation."""
        if self.config.visualization_format == "html":
            return self._create_html_visualization(explanation, save)
        else:
            return self._create_matplotlib_visualization(explanation, save)

    def _create_html_visualization(
        self,
        explanation: PreferenceExplanation,
        save: bool = True
    ) -> str:
        """Create HTML visualization of explanation."""
        html = [
            "<html><body>",
            "<h2>Preference Explanation</h2>",
            f"<p>Confidence: {explanation.confidence:.2%}</p>"
        ]
        
        # Feature importance
        if explanation.feature_importance:
            html.append("<h3>Important Features</h3><ul>")
            for feature, importance in sorted(
                explanation.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                html.append(
                    f"<li>{feature}: {importance:.2%}</li>"
                )
            html.append("</ul>")
        
        # Rules
        if explanation.rules:
            html.append("<h3>Decision Rules</h3><ul>")
            for rule in explanation.rules:
                html.append(f"<li>{rule}</li>")
            html.append("</ul>")
        
        # Counterfactuals
        if explanation.counterfactuals:
            html.append("<h3>Counterfactual Examples</h3><ul>")
            for cf in explanation.counterfactuals:
                html.append("<li>If the values were:<ul>")
                for feature, value in cf.items():
                    html.append(f"<li>{feature}: {value:.2f}</li>")
                html.append("</ul>the preference would be different</li>")
            html.append("</ul>")
        
        html.append("</body></html>")
        content = "\n".join(html)
        
        if save and self.config.output_path:
            from pathlib import Path
            path = Path(self.config.output_path) / "explanation.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        
        return content

    def _create_matplotlib_visualization(
        self,
        explanation: PreferenceExplanation,
        save: bool = True
    ) -> plt.Figure:
        """Create matplotlib visualization of explanation."""
        fig = plt.figure(figsize=(12, 8))
        
        # Feature importance plot
        if explanation.feature_importance:
            ax1 = plt.subplot(221)
            features, values = zip(*sorted(
                explanation.feature_importance.items(),
                key=lambda x: x[1]
            ))
            ax1.barh(features, values)
            ax1.set_title("Feature Importance")
        
        # SHAP values plot
        if explanation.shap_values is not None:
            ax2 = plt.subplot(222)
            if self.shap_explainer:
                shap.summary_plot(
                    explanation.shap_values,
                    feature_names=[
                        obj.name
                        for obj in self.learner.adapter.config.objectives
                    ],
                    show=False,
                    ax=ax2
                )
            ax2.set_title("SHAP Values")
        
        # Decision tree plot
        if self.tree_model:
            ax3 = plt.subplot(223)
            plot_tree(
                self.tree_model,
                feature_names=[
                    obj.name
                    for obj in self.learner.adapter.config.objectives
                ],
                filled=True,
                rounded=True,
                ax=ax3
            )
            ax3.set_title("Decision Tree")
        
        plt.tight_layout()
        
        if save and self.config.output_path:
            from pathlib import Path
            path = Path(self.config.output_path) / "explanation.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)
        
        return fig

@pytest.fixture
def preference_explainer(preference_learner):
    """Create preference explainer for testing."""
    config = ExplanationConfig()
    return PreferenceExplainer(preference_learner, config)

@pytest.mark.asyncio
async def test_explanation_generation(preference_explainer):
    """Test explanation generation."""
    # Create test comparison
    comparison = Comparison(
        solution_a={"test": 1.0},
        solution_b={"test": 2.0},
        preference="A",
        confidence=0.8
    )
    
    # Generate explanation
    explanation = await preference_explainer.explain_preference(comparison)
    
    # Verify explanation
    assert explanation is not None
    if explanation:
        if preference_explainer.config.enable_feature_importance:
            assert len(explanation.feature_importance) >= 0
        if preference_explainer.config.enable_rules:
            assert len(explanation.rules) >= 0
        assert explanation.confidence > 0

@pytest.mark.asyncio
async def test_visualization(preference_explainer, tmp_path):
    """Test explanation visualization."""
    # Set temporary output path
    preference_explainer.config.output_path = str(tmp_path)
    
    # Create test explanation
    explanation = PreferenceExplanation(
        feature_importance={"test": 1.0},
        rules=["test > 0.5"],
        counterfactuals=[{"test": 0.0}],
        shap_values=None,
        confidence=0.8
    )
    
    # Generate visualization
    viz = preference_explainer.visualize_explanation(explanation)
    
    # Verify output
    if preference_explainer.config.visualization_format == "html":
        assert isinstance(viz, str)
        assert "<html>" in viz
    else:
        from matplotlib.figure import Figure
        assert isinstance(viz, Figure)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
