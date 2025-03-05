"""Statistical analysis tools for chain comparisons."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score, r2_score
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .comparison_animation import ChainComparator, ComparisonConfig
from .chain_animation import ChainAnimator

logger = logging.getLogger(__name__)

@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis."""
    alpha: float = 0.05
    min_samples: int = 30
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.3
    output_path: Optional[Path] = None

class ChainStatistician:
    """Statistical analysis of chain differences."""
    
    def __init__(
        self,
        comparator: ChainComparator,
        config: StatisticalConfig
    ):
        self.comparator = comparator
        self.config = config
    
    def analyze_chains(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        results = {
            "descriptive": self._descriptive_analysis(names, data),
            "inferential": self._inferential_analysis(names, data),
            "effect_sizes": self._effect_size_analysis(names, data),
            "correlations": self._correlation_analysis(names, data),
            "distributions": self._distribution_analysis(names, data)
        }
        
        return results
    
    def visualize_analysis(
        self,
        analysis: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of statistical analysis."""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Effect Sizes",
                "P-Values",
                "Correlations",
                "Distributions",
                "Time Series",
                "Residuals"
            ]
        )
        
        # Effect sizes plot
        effect_sizes = analysis["effect_sizes"]
        fig.add_trace(
            go.Heatmap(
                z=effect_sizes["matrix"],
                x=effect_sizes["chains"],
                y=effect_sizes["chains"],
                colorscale="RdBu",
                name="Effect Sizes"
            ),
            row=1,
            col=1
        )
        
        # P-values plot
        pvalues = analysis["inferential"]["pvalues"]
        fig.add_trace(
            go.Heatmap(
                z=-np.log10(pvalues["matrix"]),
                x=pvalues["chains"],
                y=pvalues["chains"],
                colorscale="Viridis",
                name="P-Values"
            ),
            row=1,
            col=2
        )
        
        # Correlations plot
        correlations = analysis["correlations"]
        fig.add_trace(
            go.Heatmap(
                z=correlations["matrix"],
                x=correlations["features"],
                y=correlations["features"],
                colorscale="RdBu",
                name="Correlations"
            ),
            row=2,
            col=1
        )
        
        # Distribution plot
        for name, dist in analysis["distributions"]["per_chain"].items():
            fig.add_trace(
                go.Violin(
                    y=dist["values"],
                    name=name,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=2,
                col=2
            )
        
        # Time series plot
        desc = analysis["descriptive"]
        for name, metrics in desc["per_chain"].items():
            fig.add_trace(
                go.Scatter(
                    y=metrics["rolling_mean"],
                    name=f"{name} Trend",
                    mode="lines"
                ),
                row=3,
                col=1
            )
        
        # Residuals plot
        resids = analysis["inferential"]["residuals"]
        fig.add_trace(
            go.Scatter(
                x=resids["predicted"],
                y=resids["values"],
                mode="markers",
                name="Residuals"
            ),
            row=3,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title="Statistical Analysis Results",
            showlegend=True
        )
        
        return fig
    
    def _descriptive_analysis(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform descriptive statistical analysis."""
        results = {
            "overall": {},
            "per_chain": {}
        }
        
        # Apply chains and collect results
        for name in names:
            chain_data = self.comparator.animator.visualizer.chain.apply_chain(
                name,
                data.copy()
            )
            
            if isinstance(chain_data, (pd.DataFrame, pd.Series)):
                values = chain_data.values.flatten()
                
                results["per_chain"][name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "skew": stats.skew(values),
                    "kurtosis": stats.kurtosis(values),
                    "rolling_mean": pd.Series(values).rolling(
                        window=min(len(values) // 10, 100)
                    ).mean()
                }
        
        # Overall statistics
        all_values = np.concatenate([
            chain_data.values.flatten()
            for name in names
            if isinstance(
                self.comparator.animator.visualizer.chain.apply_chain(name, data.copy()),
                (pd.DataFrame, pd.Series)
            )
        ])
        
        results["overall"] = {
            "mean": np.mean(all_values),
            "std": np.std(all_values),
            "median": np.median(all_values),
            "skew": stats.skew(all_values),
            "kurtosis": stats.kurtosis(all_values)
        }
        
        return results
    
    def _inferential_analysis(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform inferential statistical analysis."""
        results = {
            "anova": {},
            "tukey": {},
            "pvalues": {
                "matrix": np.zeros((len(names), len(names))),
                "chains": names
            },
            "residuals": {}
        }
        
        # Collect chain results
        chain_results = []
        chain_labels = []
        
        for name in names:
            chain_data = self.comparator.animator.visualizer.chain.apply_chain(
                name,
                data.copy()
            )
            
            if isinstance(chain_data, (pd.DataFrame, pd.Series)):
                values = chain_data.values.flatten()
                chain_results.extend(values)
                chain_labels.extend([name] * len(values))
        
        # Perform ANOVA
        try:
            groups = [
                np.array(chain_results)[np.array(chain_labels) == name]
                for name in names
            ]
            f_stat, p_value = stats.f_oneway(*groups)
            
            results["anova"] = {
                "f_statistic": f_stat,
                "p_value": p_value
            }
            
            # Tukey's HSD test
            data_array = np.array(chain_results)
            labels_array = np.array(chain_labels)
            tukey = pairwise_tukeyhsd(data_array, labels_array)
            
            results["tukey"] = {
                "statistics": tukey.statistic,
                "p_values": tukey.pvalues,
                "significant": tukey.reject
            }
            
            # Pairwise t-tests with correction
            for i, name1 in enumerate(names):
                for j, name2 in enumerate(names):
                    if i != j:
                        group1 = np.array(chain_results)[np.array(chain_labels) == name1]
                        group2 = np.array(chain_results)[np.array(chain_labels) == name2]
                        _, p_val = stats.ttest_ind(group1, group2)
                        results["pvalues"]["matrix"][i, j] = p_val
            
            # Calculate residuals
            model = sm.OLS(
                data_array,
                sm.add_constant(pd.get_dummies(labels_array))
            ).fit()
            
            results["residuals"] = {
                "values": model.resid,
                "predicted": model.fittedvalues
            }
            
        except Exception as e:
            logger.warning(f"Inferential analysis failed: {e}")
        
        return results
    
    def _effect_size_analysis(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate effect sizes between chains."""
        results = {
            "matrix": np.zeros((len(names), len(names))),
            "chains": names,
            "significant": []
        }
        
        # Calculate Cohen's d for each pair
        for i, name1 in enumerate(names):
            chain_data1 = self.comparator.animator.visualizer.chain.apply_chain(
                name1,
                data.copy()
            )
            
            if isinstance(chain_data1, (pd.DataFrame, pd.Series)):
                values1 = chain_data1.values.flatten()
                
                for j, name2 in enumerate(names):
                    if i != j:
                        chain_data2 = self.comparator.animator.visualizer.chain.apply_chain(
                            name2,
                            data.copy()
                        )
                        
                        if isinstance(chain_data2, (pd.DataFrame, pd.Series)):
                            values2 = chain_data2.values.flatten()
                            
                            # Calculate Cohen's d
                            cohens_d = (
                                (np.mean(values1) - np.mean(values2)) /
                                np.sqrt(
                                    (np.var(values1) + np.var(values2)) / 2
                                )
                            )
                            
                            results["matrix"][i, j] = cohens_d
                            
                            # Check if effect size is significant
                            if abs(cohens_d) > self.config.effect_size_threshold:
                                results["significant"].append((name1, name2, cohens_d))
        
        return results
    
    def _correlation_analysis(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlations between chains."""
        # Get all features from chains
        features = set()
        chain_features = {}
        
        for name in names:
            chain_data = self.comparator.animator.visualizer.chain.apply_chain(
                name,
                data.copy()
            )
            
            if isinstance(chain_data, pd.DataFrame):
                chain_features[name] = chain_data
                features.update(chain_data.columns)
        
        features = sorted(features)
        n_features = len(features)
        
        results = {
            "matrix": np.zeros((n_features, n_features)),
            "features": features,
            "mutual_info": {},
            "r2_scores": {}
        }
        
        # Calculate correlation matrix
        if chain_features:
            combined_data = pd.DataFrame()
            
            for name, df in chain_features.items():
                for col in df.columns:
                    combined_data[f"{name}_{col}"] = df[col]
            
            results["matrix"] = combined_data.corr().values
            
            # Calculate mutual information
            for i, feat1 in enumerate(features):
                for j, feat2 in enumerate(features):
                    if i < j:
                        mi = mutual_info_score(
                            combined_data.iloc[:, i],
                            combined_data.iloc[:, j]
                        )
                        results["mutual_info"][f"{feat1}_{feat2}"] = mi
            
            # Calculate R² scores
            for name1, df1 in chain_features.items():
                for name2, df2 in chain_features.items():
                    if name1 != name2:
                        common_cols = set(df1.columns) & set(df2.columns)
                        if common_cols:
                            r2 = r2_score(
                                df1[list(common_cols)],
                                df2[list(common_cols)]
                            )
                            results["r2_scores"][f"{name1}_{name2}"] = r2
        
        return results
    
    def _distribution_analysis(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze distributions of chain outputs."""
        results = {
            "per_chain": {},
            "tests": {}
        }
        
        # Calculate distribution metrics for each chain
        for name in names:
            chain_data = self.comparator.animator.visualizer.chain.apply_chain(
                name,
                data.copy()
            )
            
            if isinstance(chain_data, (pd.DataFrame, pd.Series)):
                values = chain_data.values.flatten()
                
                # Basic distribution metrics
                results["per_chain"][name] = {
                    "values": values,
                    "percentiles": np.percentile(values, [25, 50, 75]),
                    "iqr": np.percentile(values, 75) - np.percentile(values, 25),
                    "normality": stats.normaltest(values)
                }
                
                # Bootstrap confidence intervals
                boot_means = [
                    np.mean(np.random.choice(values, size=len(values)))
                    for _ in range(self.config.bootstrap_iterations)
                ]
                
                results["per_chain"][name]["bootstrap"] = {
                    "mean_ci": np.percentile(
                        boot_means,
                        [
                            (1 - self.config.confidence_level) * 100 / 2,
                            (1 + self.config.confidence_level) * 100 / 2
                        ]
                    )
                }
        
        # Statistical tests between distributions
        if len(names) > 1:
            # Kolmogorov-Smirnov test
            for i, name1 in enumerate(names):
                for j, name2 in enumerate(names[i+1:], i+1):
                    if (
                        name1 in results["per_chain"] and
                        name2 in results["per_chain"]
                    ):
                        ks_stat, p_val = stats.ks_2samp(
                            results["per_chain"][name1]["values"],
                            results["per_chain"][name2]["values"]
                        )
                        
                        results["tests"][f"ks_{name1}_{name2}"] = {
                            "statistic": ks_stat,
                            "p_value": p_val
                        }
            
            # Mann-Whitney U test
            for i, name1 in enumerate(names):
                for j, name2 in enumerate(names[i+1:], i+1):
                    if (
                        name1 in results["per_chain"] and
                        name2 in results["per_chain"]
                    ):
                        u_stat, p_val = stats.mannwhitneyu(
                            results["per_chain"][name1]["values"],
                            results["per_chain"][name2]["values"]
                        )
                        
                        results["tests"][f"mw_{name1}_{name2}"] = {
                            "statistic": u_stat,
                            "p_value": p_val
                        }
        
        return results
    
    def save_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """Save statistical analysis results."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save analysis results
            analysis_file = path / "statistical_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(
                    {
                        k: v for k, v in analysis.items()
                        if isinstance(v, (dict, list, str, int, float, bool))
                    },
                    f,
                    indent=2
                )
            
            # Save visualization
            viz = self.visualize_analysis(analysis)
            viz.write_html(str(path / "statistical_analysis.html"))
            
            logger.info(f"Saved statistical analysis to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_chain_statistician(
    comparator: ChainComparator,
    output_path: Optional[Path] = None
) -> ChainStatistician:
    """Create chain statistician."""
    config = StatisticalConfig(output_path=output_path)
    return ChainStatistician(comparator, config)

if __name__ == "__main__":
    # Example usage
    from .comparison_animation import create_chain_comparator
    from .chain_animation import create_chain_animator
    from .chain_visualization import create_chain_visualizer
    from .filter_chaining import create_filter_chain
    from .learning_filters import create_learning_filter
    from .interactive_learning import create_interactive_learning
    from .learning_visualization import create_learning_visualizer
    from .optimization_learning import create_optimization_learner
    from .composition_optimization import create_composition_optimizer
    from .composition_analysis import create_composition_analysis
    from .pattern_composition import create_pattern_composer
    from .scheduling_patterns import create_scheduling_pattern
    from .event_scheduler import create_event_scheduler
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(interactive)
    events = create_event_manager(controls)
    scheduler = create_event_scheduler(events)
    pattern = create_scheduling_pattern(scheduler)
    composer = create_pattern_composer(pattern)
    analyzer = create_composition_analysis(composer)
    optimizer = create_composition_optimizer(analyzer)
    learner = create_optimization_learner(optimizer)
    viz = create_learning_visualizer(learner)
    interactive_learning = create_interactive_learning(viz)
    filters = create_learning_filter(interactive_learning)
    chain = create_filter_chain(filters)
    chain_viz = create_chain_visualizer(chain)
    animator = create_chain_animator(chain_viz)
    comparator = create_chain_comparator(animator)
    statistician = create_chain_statistician(
        comparator,
        output_path=Path("statistical_analysis")
    )
    
    # Create example chains
    chain.create_chain(
        "preprocessing_a",
        [
            {
                "filter": "time_range",
                "params": {"window": 30}
            },
            {
                "filter": "confidence",
                "params": {"threshold": 0.7}
            }
        ]
    )
    
    chain.create_chain(
        "preprocessing_b",
        [
            {
                "filter": "time_range",
                "params": {"window": 60}
            },
            {
                "filter": "complexity",
                "params": {"max_complexity": 5}
            }
        ]
    )
    
    # Example data
    data = {
        "timestamp": pd.date_range(start="2025-01-01", periods=1000, freq="H"),
        "confidence": np.random.uniform(0, 1, 1000),
        "success": np.random.choice([True, False], 1000),
        "complexity": np.random.randint(1, 20, 1000),
        "features": pd.DataFrame(
            np.random.randn(1000, 5),
            columns=["f1", "f2", "f3", "f4", "f5"]
        )
    }
    
    # Perform analysis
    analysis = statistician.analyze_chains(
        ["preprocessing_a", "preprocessing_b"],
        data
    )
    
    # Save analysis
    statistician.save_analysis(analysis)
