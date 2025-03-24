"""Recommendation system for exploring optimization solutions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .interactive_optimization import InteractiveExplorer, InteractionConfig
from .multi_objective_optimization import MultiObjectiveOptimizer

logger = logging.getLogger(__name__)

@dataclass
class RecommendationConfig:
    """Configuration for solution recommendations."""
    n_neighbors: int = 5
    n_clusters: int = 3
    similarity_threshold: float = 0.8
    exploration_factor: float = 0.2
    history_weight: float = 0.7
    output_path: Optional[Path] = None

class SolutionRecommender:
    """Recommend solutions based on user preferences and behavior."""
    
    def __init__(
        self,
        explorer: InteractiveExplorer,
        config: RecommendationConfig
    ):
        self.explorer = explorer
        self.config = config
        
        # Initialize recommendation models
        self.scaler = StandardScaler()
        self.neighbor_model = NearestNeighbors(
            n_neighbors=config.n_neighbors,
            metric="cosine"
        )
        self.cluster_model = KMeans(
            n_clusters=config.n_clusters,
            random_state=42
        )
        
        # State management
        self.solution_embeddings = None
        self.cluster_assignments = None
        self.recommendation_history = []
        self.user_profile = None
        
        # Initialize models
        self._initialize_models()
    
    def get_recommendations(
        self,
        current_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get solution recommendations."""
        results = {
            "similar": self._find_similar_solutions(current_weights),
            "diverse": self._find_diverse_solutions(current_weights),
            "next": self._predict_next_solutions(current_weights),
            "insights": self._generate_insights(current_weights)
        }
        
        # Update history
        self.recommendation_history.append({
            "weights": current_weights,
            "recommendations": results,
            "timestamp": datetime.now().isoformat()
        })
        
        return results
    
    def visualize_recommendations(
        self,
        recommendations: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of recommendations."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Similar Solutions",
                "Diverse Solutions",
                "Solution Space",
                "Recommendation History"
            ]
        )
        
        # Similar solutions plot
        similar = recommendations["similar"]
        fig.add_trace(
            go.Scatter3d(
                x=[s["objectives"]["power_error"] for s in similar["solutions"]],
                y=[s["objectives"]["computation_cost"] for s in similar["solutions"]],
                z=[s["objectives"]["stability"] for s in similar["solutions"]],
                mode="markers",
                marker=dict(
                    size=8,
                    color="blue",
                    symbol="circle"
                ),
                name="Similar Solutions"
            ),
            row=1,
            col=1
        )
        
        # Diverse solutions plot
        diverse = recommendations["diverse"]
        fig.add_trace(
            go.Scatter3d(
                x=[s["objectives"]["power_error"] for s in diverse["solutions"]],
                y=[s["objectives"]["computation_cost"] for s in diverse["solutions"]],
                z=[s["objectives"]["stability"] for s in diverse["solutions"]],
                mode="markers",
                marker=dict(
                    size=8,
                    color="green",
                    symbol="diamond"
                ),
                name="Diverse Solutions"
            ),
            row=1,
            col=2
        )
        
        # Solution space plot
        solutions = self.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        fig.add_trace(
            go.Scatter3d(
                x=[s["objectives"]["power_error"] for s in solutions],
                y=[s["objectives"]["computation_cost"] for s in solutions],
                z=[s["objectives"]["stability"] for s in solutions],
                mode="markers",
                marker=dict(
                    size=4,
                    color=self.cluster_assignments,
                    colorscale="Viridis",
                    opacity=0.6
                ),
                name="All Solutions"
            ),
            row=2,
            col=1
        )
        
        # Recommendation history plot
        if self.recommendation_history:
            history_data = pd.DataFrame([
                {
                    "timestamp": h["timestamp"],
                    **h["weights"]
                }
                for h in self.recommendation_history
            ])
            
            for col in history_data.columns[1:]:
                fig.add_trace(
                    go.Scatter(
                        x=history_data["timestamp"],
                        y=history_data[col],
                        mode="lines+markers",
                        name=f"{col} Weight"
                    ),
                    row=2,
                    col=2
                )
        
        return fig
    
    def _initialize_models(self):
        """Initialize recommendation models."""
        solutions = self.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        
        # Create solution embeddings
        features = []
        for sol in solutions:
            feat = [
                *sol["objectives"].values(),
                *[p/100 for p in sol["parameters"].values()]
            ]
            features.append(feat)
        
        # Scale features
        self.solution_embeddings = self.scaler.fit_transform(features)
        
        # Fit models
        self.neighbor_model.fit(self.solution_embeddings)
        self.cluster_assignments = self.cluster_model.fit_predict(
            self.solution_embeddings
        )
        
        # Initialize user profile
        self.user_profile = np.zeros_like(self.solution_embeddings[0])
    
    def _find_similar_solutions(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Find solutions similar to current preferences."""
        # Create query vector
        query = np.array([
            weights.get(obj, 0)
            for obj in self.explorer.optimizer.config.objective_weights.keys()
        ])
        
        # Find nearest neighbors
        distances, indices = self.neighbor_model.kneighbors(
            query.reshape(1, -1)
        )
        
        solutions = self.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        similar_solutions = [solutions[i] for i in indices[0]]
        
        return {
            "solutions": similar_solutions,
            "distances": distances[0].tolist(),
            "similarity_scores": (1 - distances[0] / distances[0].max()).tolist()
        }
    
    def _find_diverse_solutions(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Find diverse solutions based on clustering."""
        solutions = self.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        
        # Get cluster centers
        cluster_centers = []
        for cluster in range(self.config.n_clusters):
            cluster_idx = np.where(self.cluster_assignments == cluster)[0]
            
            if len(cluster_idx) > 0:
                # Find solution closest to cluster center
                center_idx = cluster_idx[
                    np.argmin(
                        np.linalg.norm(
                            self.solution_embeddings[cluster_idx] -
                            self.cluster_model.cluster_centers_[cluster],
                            axis=1
                        )
                    )
                ]
                cluster_centers.append(solutions[center_idx])
        
        return {
            "solutions": cluster_centers,
            "cluster_sizes": [
                np.sum(self.cluster_assignments == i)
                for i in range(self.config.n_clusters)
            ]
        }
    
    def _predict_next_solutions(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict next interesting solutions."""
        # Update user profile
        query = np.array([
            weights.get(obj, 0)
            for obj in self.explorer.optimizer.config.objective_weights.keys()
        ])
        
        self.user_profile = (
            self.config.history_weight * self.user_profile +
            (1 - self.config.history_weight) * query
        )
        
        # Find solutions balancing similarity and exploration
        solutions = self.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        
        scores = []
        for i, embedding in enumerate(self.solution_embeddings):
            similarity = 1 - np.linalg.norm(embedding - self.user_profile)
            exploration = np.random.random() * self.config.exploration_factor
            scores.append(similarity + exploration)
        
        # Get top solutions
        top_indices = np.argsort(scores)[-self.config.n_neighbors:]
        
        return {
            "solutions": [solutions[i] for i in top_indices],
            "scores": [scores[i] for i in top_indices],
            "exploration_ratios": [
                self.config.exploration_factor
                for _ in range(len(top_indices))
            ]
        }
    
    def _generate_insights(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate insights about current preferences."""
        solutions = self.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        
        insights = {
            "preference_analysis": self._analyze_preferences(weights),
            "trade_offs": self._analyze_trade_offs(weights),
            "exploration_status": self._analyze_exploration(),
            "recommendations": []
        }
        
        # Generate textual recommendations
        if insights["preference_analysis"]["bias"] > 0.7:
            insights["recommendations"].append(
                "Consider exploring more diverse solutions"
            )
        
        if insights["exploration_status"]["coverage"] < 0.3:
            insights["recommendations"].append(
                "Large portions of the solution space remain unexplored"
            )
        
        if insights["trade_offs"]["conflicts"]:
            insights["recommendations"].append(
                "Current preferences have strong trade-offs"
            )
        
        return insights
    
    def _analyze_preferences(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze current preference patterns."""
        return {
            "bias": np.std(list(weights.values())),
            "dominant_objective": max(weights.items(), key=lambda x: x[1])[0],
            "balanced_score": 1 - np.std(list(weights.values())),
            "evolution": self._analyze_preference_evolution()
        }
    
    def _analyze_trade_offs(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze trade-offs in current preferences."""
        trade_offs = self.explorer.optimizer.optimization_results["trade_offs"]
        
        return {
            "conflicts": [
                (obj1, obj2)
                for i, obj1 in enumerate(trade_offs["objectives"])
                for j, obj2 in enumerate(trade_offs["objectives"])
                if i < j and trade_offs["correlation_matrix"][i, j] < -0.5
            ],
            "synergies": [
                (obj1, obj2)
                for i, obj1 in enumerate(trade_offs["objectives"])
                for j, obj2 in enumerate(trade_offs["objectives"])
                if i < j and trade_offs["correlation_matrix"][i, j] > 0.5
            ],
            "difficulty": np.mean([
                abs(trade_offs["correlation_matrix"][i, j])
                for i in range(len(trade_offs["objectives"]))
                for j in range(i)
            ])
        }
    
    def _analyze_exploration(self) -> Dict[str, Any]:
        """Analyze exploration patterns."""
        if not self.recommendation_history:
            return {
                "coverage": 0.0,
                "diversity": 0.0,
                "phases": []
            }
        
        # Calculate coverage
        visited = set()
        for h in self.recommendation_history:
            for r in h["recommendations"]["similar"]["solutions"]:
                visited.add(r["id"])
        
        total_solutions = len(
            self.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        )
        
        # Analyze exploration phases
        phases = []
        weights_history = np.array([
            list(h["weights"].values())
            for h in self.recommendation_history
        ])
        
        if len(weights_history) > 1:
            changes = np.diff(weights_history, axis=0)
            magnitudes = np.linalg.norm(changes, axis=1)
            
            # Detect significant changes
            thresholds = np.percentile(magnitudes, [33, 66])
            
            current_phase = "exploration"
            phase_changes = []
            
            for i, m in enumerate(magnitudes):
                if m < thresholds[0]:
                    new_phase = "exploitation"
                elif m > thresholds[1]:
                    new_phase = "exploration"
                else:
                    new_phase = "balanced"
                
                if new_phase != current_phase:
                    phase_changes.append({
                        "step": i,
                        "from": current_phase,
                        "to": new_phase
                    })
                    current_phase = new_phase
            
            phases = phase_changes
        
        return {
            "coverage": len(visited) / total_solutions,
            "diversity": len(set(
                self.cluster_assignments[list(visited)]
            )) / self.config.n_clusters,
            "phases": phases
        }
    
    def _analyze_preference_evolution(self) -> Dict[str, Any]:
        """Analyze evolution of preferences."""
        if not self.recommendation_history:
            return {
                "stability": 1.0,
                "trend": "stable",
                "changes": []
            }
        
        weights_history = np.array([
            list(h["weights"].values())
            for h in self.recommendation_history
        ])
        
        # Calculate stability
        stability = 1 - np.mean(np.std(weights_history, axis=0))
        
        # Detect trends
        if len(weights_history) > 1:
            slopes = np.polyfit(
                range(len(weights_history)),
                weights_history,
                1
            )[0]
            
            changes = []
            for i, slope in enumerate(slopes):
                if abs(slope) > 0.1:
                    changes.append({
                        "objective": list(self.explorer.preference_weights.keys())[i],
                        "trend": "increasing" if slope > 0 else "decreasing",
                        "magnitude": abs(slope)
                    })
            
            trend = (
                "convergent"
                if stability > 0.7
                else "exploring"
                if len(changes) > 1
                else "stable"
            )
        else:
            changes = []
            trend = "initial"
        
        return {
            "stability": float(stability),
            "trend": trend,
            "changes": changes
        }
    
    def save_recommendations(
        self,
        output_path: Optional[Path] = None
    ):
        """Save recommendation history."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            history_file = path / "recommendation_history.json"
            with open(history_file, "w") as f:
                json.dump(
                    self.recommendation_history,
                    f,
                    indent=2
                )
            
            logger.info(f"Saved recommendation history to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save recommendations: {e}")

def create_solution_recommender(
    explorer: InteractiveExplorer,
    output_path: Optional[Path] = None
) -> SolutionRecommender:
    """Create solution recommender."""
    config = RecommendationConfig(output_path=output_path)
    return SolutionRecommender(explorer, config)

if __name__ == "__main__":
    # Example usage
    from .interactive_optimization import create_interactive_explorer
    from .multi_objective_optimization import create_multi_objective_optimizer
    from .simulation_optimization import create_simulation_optimizer
    from .monte_carlo_power import create_monte_carlo_analyzer
    from .power_analysis import create_chain_power_analyzer
    from .statistical_comparison import create_chain_statistician
    from .comparison_animation import create_chain_comparator
    from .chain_animation import create_chain_animator
    from .chain_visualization import create_chain_visualizer
    from .filter_chaining import create_filter_chain
    from .learning_filters import create_learning_filter
    from .interactive_learning import create_interactive_learning
    
    # Create components
    filters = create_learning_filter()
    chain = create_filter_chain(filters)
    chain_viz = create_chain_visualizer(chain)
    animator = create_chain_animator(chain_viz)
    comparator = create_chain_comparator(animator)
    statistician = create_chain_statistician(comparator)
    power_analyzer = create_chain_power_analyzer(statistician)
    mc_analyzer = create_monte_carlo_analyzer(power_analyzer)
    sim_optimizer = create_simulation_optimizer(mc_analyzer)
    mo_optimizer = create_multi_objective_optimizer(sim_optimizer)
    explorer = create_interactive_explorer(mo_optimizer)
    recommender = create_solution_recommender(
        explorer,
        output_path=Path("solution_recommendations")
    )
    
    # Example usage
    weights = {
        "power_error": 0.5,
        "computation_cost": 0.3,
        "stability": 0.2
    }
    
    recommendations = recommender.get_recommendations(weights)
    
    # Visualize recommendations
    viz = recommender.visualize_recommendations(recommendations)
    viz.show()
    
    # Save recommendations
    recommender.save_recommendations()
