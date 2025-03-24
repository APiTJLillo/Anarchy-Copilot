"""Collaborative filtering for solution recommendations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .solution_recommendations import SolutionRecommender, RecommendationConfig
from .interactive_optimization import InteractiveExplorer

logger = logging.getLogger(__name__)

@dataclass
class CollaborationConfig:
    """Configuration for collaborative filtering."""
    n_factors: int = 10
    min_interactions: int = 5
    similarity_threshold: float = 0.7
    influence_weight: float = 0.3
    history_length: int = 100
    output_path: Optional[Path] = None

class CollaborativeRecommender:
    """Collaborative filtering for solution recommendations."""
    
    def __init__(
        self,
        recommender: SolutionRecommender,
        config: CollaborationConfig
    ):
        self.recommender = recommender
        self.config = config
        
        # Initialize models
        self.nmf = NMF(
            n_components=config.n_factors,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # User state management
        self.user_profiles = {}
        self.user_interactions = {}
        self.user_factors = None
        self.solution_factors = None
        
        # Initialize state
        self._initialize_state()
    
    def add_user_interaction(
        self,
        user_id: str,
        weights: Dict[str, float],
        selected_solutions: List[int]
    ):
        """Add new user interaction."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = []
            self.user_interactions[user_id] = []
        
        # Add to profile history
        self.user_profiles[user_id].append({
            "weights": weights,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add to interactions
        self.user_interactions[user_id].extend([
            {
                "solution_id": sol_id,
                "timestamp": datetime.now().isoformat()
            }
            for sol_id in selected_solutions
        ])
        
        # Limit history length
        if len(self.user_profiles[user_id]) > self.config.history_length:
            self.user_profiles[user_id] = self.user_profiles[user_id][-self.config.history_length:]
        
        if len(self.user_interactions[user_id]) > self.config.history_length:
            self.user_interactions[user_id] = self.user_interactions[user_id][-self.config.history_length:]
        
        # Update models
        self._update_models()
    
    def get_collaborative_recommendations(
        self,
        user_id: str,
        current_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get recommendations enhanced by collaborative filtering."""
        # Get base recommendations
        base_recommendations = self.recommender.get_recommendations(current_weights)
        
        # Enhance with collaborative filtering if possible
        if len(self.user_profiles) > 1:
            collaborative_results = {
                "similar_users": self._find_similar_users(user_id),
                "popular_solutions": self._find_popular_solutions(),
                "predicted_interests": self._predict_user_interests(user_id),
                "group_insights": self._analyze_group_behavior()
            }
            
            # Combine recommendations
            enhanced_recommendations = self._combine_recommendations(
                base_recommendations,
                collaborative_results,
                current_weights
            )
            
            return {
                **enhanced_recommendations,
                "collaborative_analysis": collaborative_results
            }
        
        return base_recommendations
    
    def visualize_collaboration(
        self,
        recommendations: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of collaborative filtering results."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "User Similarity Network",
                "Solution Popularity",
                "Interest Predictions",
                "Group Behavior"
            ]
        )
        
        if "collaborative_analysis" in recommendations:
            collab = recommendations["collaborative_analysis"]
            
            # User similarity network
            similar_users = collab["similar_users"]
            if similar_users["network"]:
                # Create network visualization
                user_pos = self._calculate_network_layout(similar_users["network"])
                
                edge_x = []
                edge_y = []
                for (u1, u2), strength in similar_users["similarities"].items():
                    x0, y0 = user_pos[u1]
                    x1, y1 = user_pos[u2]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                fig.add_trace(
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(
                            width=1,
                            color="gray"
                        ),
                        hoverinfo="none",
                        showlegend=False
                    ),
                    row=1,
                    col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[pos[0] for pos in user_pos.values()],
                        y=[pos[1] for pos in user_pos.values()],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color="blue"
                        ),
                        text=list(user_pos.keys()),
                        textposition="top center",
                        name="Users"
                    ),
                    row=1,
                    col=1
                )
            
            # Solution popularity
            popular = collab["popular_solutions"]
            fig.add_trace(
                go.Bar(
                    x=[s["solution_id"] for s in popular["solutions"]],
                    y=[s["popularity"] for s in popular["solutions"]],
                    name="Solution Popularity"
                ),
                row=1,
                col=2
            )
            
            # Interest predictions
            interests = collab["predicted_interests"]
            if interests["predictions"]:
                fig.add_trace(
                    go.Heatmap(
                        z=interests["factor_weights"],
                        x=list(range(self.config.n_factors)),
                        y=["User Factors"],
                        colorscale="Viridis",
                        name="Interest Factors"
                    ),
                    row=2,
                    col=1
                )
            
            # Group behavior
            behavior = collab["group_insights"]
            fig.add_trace(
                go.Scatter(
                    x=behavior["time_series"]["timestamps"],
                    y=behavior["time_series"]["activity"],
                    mode="lines",
                    name="Group Activity"
                ),
                row=2,
                col=2
            )
        
        return fig
    
    def _initialize_state(self):
        """Initialize collaborative filtering state."""
        # Create initial user-solution matrix
        solutions = self.recommender.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        self.solution_ids = [s["id"] for s in solutions]
        
        # Initialize factor matrices
        n_solutions = len(solutions)
        self.solution_factors = np.random.random((n_solutions, self.config.n_factors))
    
    def _update_models(self):
        """Update collaborative filtering models."""
        if len(self.user_profiles) < 2:
            return
        
        # Create user-solution interaction matrix
        users = list(self.user_profiles.keys())
        n_users = len(users)
        n_solutions = len(self.solution_ids)
        
        interaction_matrix = np.zeros((n_users, n_solutions))
        
        for i, user_id in enumerate(users):
            for interaction in self.user_interactions[user_id]:
                j = self.solution_ids.index(interaction["solution_id"])
                interaction_matrix[i, j] += 1
        
        # Normalize interactions
        interaction_matrix = self.scaler.fit_transform(interaction_matrix)
        
        # Update NMF model
        if np.any(interaction_matrix):
            self.user_factors = self.nmf.fit_transform(interaction_matrix)
            self.solution_factors = self.nmf.components_.T
    
    def _find_similar_users(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Find users with similar preferences."""
        if user_id not in self.user_profiles:
            return {
                "similar": [],
                "similarities": {},
                "network": []
            }
        
        users = list(self.user_profiles.keys())
        user_idx = users.index(user_id)
        
        # Calculate user similarities
        if self.user_factors is not None:
            similarities = cosine_similarity(
                self.user_factors[user_idx].reshape(1, -1),
                self.user_factors
            )[0]
            
            # Find similar users
            similar_indices = np.where(
                similarities > self.config.similarity_threshold
            )[0]
            
            similar_users = [
                users[i] for i in similar_indices
                if users[i] != user_id
            ]
            
            # Create similarity network
            network = []
            similarities_dict = {}
            
            for i, u1 in enumerate(similar_users):
                for u2 in similar_users[i+1:]:
                    i1 = users.index(u1)
                    i2 = users.index(u2)
                    sim = cosine_similarity(
                        self.user_factors[i1].reshape(1, -1),
                        self.user_factors[i2].reshape(1, -1)
                    )[0, 0]
                    
                    if sim > self.config.similarity_threshold:
                        network.append((u1, u2))
                        similarities_dict[(u1, u2)] = float(sim)
            
            return {
                "similar": similar_users,
                "similarities": similarities_dict,
                "network": network
            }
        
        return {
            "similar": [],
            "similarities": {},
            "network": []
        }
    
    def _find_popular_solutions(self) -> Dict[str, Any]:
        """Find popular solutions across users."""
        if not self.user_interactions:
            return {
                "solutions": [],
                "trends": {}
            }
        
        # Count solution interactions
        solution_counts = {}
        solution_timestamps = {}
        
        for user_id, interactions in self.user_interactions.items():
            for interaction in interactions:
                sol_id = interaction["solution_id"]
                solution_counts[sol_id] = solution_counts.get(sol_id, 0) + 1
                
                if sol_id not in solution_timestamps:
                    solution_timestamps[sol_id] = []
                solution_timestamps[sol_id].append(
                    datetime.fromisoformat(interaction["timestamp"])
                )
        
        # Sort by popularity
        popular_solutions = [
            {
                "solution_id": sol_id,
                "popularity": count,
                "last_interaction": max(solution_timestamps[sol_id])
            }
            for sol_id, count in solution_counts.items()
        ]
        
        popular_solutions.sort(
            key=lambda x: (x["popularity"], x["last_interaction"]),
            reverse=True
        )
        
        # Analyze trends
        trends = {}
        for sol_id, timestamps in solution_timestamps.items():
            times = [t.timestamp() for t in timestamps]
            if len(times) > 1:
                slope, _ = np.polyfit(range(len(times)), times, 1)
                trends[sol_id] = {
                    "trend": "increasing" if slope > 0 else "decreasing",
                    "magnitude": abs(slope)
                }
        
        return {
            "solutions": popular_solutions,
            "trends": trends
        }
    
    def _predict_user_interests(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Predict user interests based on collaborative filtering."""
        if (
            user_id not in self.user_profiles or
            self.user_factors is None or
            self.solution_factors is None
        ):
            return {
                "predictions": [],
                "confidence": 0.0,
                "factor_weights": []
            }
        
        users = list(self.user_profiles.keys())
        user_idx = users.index(user_id)
        
        # Get user factors
        user_factors = self.user_factors[user_idx]
        
        # Predict interests
        predictions = np.dot(user_factors, self.solution_factors.T)
        
        # Calculate confidence
        n_interactions = len(self.user_interactions[user_id])
        confidence = min(
            n_interactions / self.config.min_interactions,
            1.0
        )
        
        return {
            "predictions": predictions.tolist(),
            "confidence": confidence,
            "factor_weights": user_factors.reshape(1, -1).tolist()
        }
    
    def _analyze_group_behavior(self) -> Dict[str, Any]:
        """Analyze collective user behavior."""
        if not self.user_profiles:
            return {
                "clusters": [],
                "time_series": {
                    "timestamps": [],
                    "activity": []
                },
                "patterns": []
            }
        
        # Collect all timestamps
        all_timestamps = []
        for user_id, interactions in self.user_interactions.items():
            all_timestamps.extend([
                datetime.fromisoformat(i["timestamp"])
                for i in interactions
            ])
        
        all_timestamps.sort()
        
        # Create time series
        if all_timestamps:
            # Group by hour
            time_range = pd.date_range(
                min(all_timestamps),
                max(all_timestamps),
                freq="H"
            )
            
            activity = pd.Series(
                [0] * len(time_range),
                index=time_range
            )
            
            for ts in all_timestamps:
                activity[ts] += 1
            
            # Detect patterns
            patterns = []
            if len(activity) > 24:
                # Daily pattern
                daily_avg = activity.groupby(activity.index.hour).mean()
                peak_hours = daily_avg.nlargest(3).index.tolist()
                
                patterns.append({
                    "type": "daily",
                    "peak_hours": peak_hours,
                    "magnitude": float(daily_avg.max() / daily_avg.mean())
                })
                
                # Weekly pattern
                if len(activity) > 168:
                    weekly_avg = activity.groupby(activity.index.dayofweek).mean()
                    peak_days = weekly_avg.nlargest(3).index.tolist()
                    
                    patterns.append({
                        "type": "weekly",
                        "peak_days": peak_days,
                        "magnitude": float(weekly_avg.max() / weekly_avg.mean())
                    })
            
            return {
                "clusters": self._cluster_users(),
                "time_series": {
                    "timestamps": activity.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                    "activity": activity.tolist()
                },
                "patterns": patterns
            }
        
        return {
            "clusters": [],
            "time_series": {
                "timestamps": [],
                "activity": []
            },
            "patterns": []
        }
    
    def _combine_recommendations(
        self,
        base_recommendations: Dict[str, Any],
        collaborative_results: Dict[str, Any],
        current_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combine base and collaborative recommendations."""
        # Get base solutions
        base_solutions = set(
            s["id"]
            for s in base_recommendations["similar"]["solutions"]
        )
        
        # Get collaborative solutions
        collab_solutions = set()
        
        # Add popular solutions
        for sol in collaborative_results["popular_solutions"]["solutions"][:self.recommender.config.n_neighbors]:
            collab_solutions.add(sol["solution_id"])
        
        # Add predicted interests
        if collaborative_results["predicted_interests"]["predictions"]:
            predictions = np.array(collaborative_results["predicted_interests"]["predictions"])
            top_indices = np.argsort(predictions)[-self.recommender.config.n_neighbors:]
            collab_solutions.update(
                self.solution_ids[i]
                for i in top_indices
            )
        
        # Combine solutions
        all_solutions = list(base_solutions | collab_solutions)
        solutions = self.recommender.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        
        combined_solutions = [
            s for s in solutions
            if s["id"] in all_solutions
        ]
        
        # Sort by relevance
        solution_scores = []
        for sol in combined_solutions:
            # Base relevance
            base_score = (
                1.0 if sol["id"] in base_solutions
                else self.config.influence_weight
            )
            
            # Collaborative relevance
            if sol["id"] in collab_solutions:
                # Add popularity bonus
                popularity = next(
                    (s["popularity"] for s in collaborative_results["popular_solutions"]["solutions"]
                     if s["solution_id"] == sol["id"]),
                    0
                )
                base_score += self.config.influence_weight * (popularity / len(self.user_profiles))
                
                # Add prediction bonus
                if collaborative_results["predicted_interests"]["predictions"]:
                    sol_idx = self.solution_ids.index(sol["id"])
                    prediction = collaborative_results["predicted_interests"]["predictions"][sol_idx]
                    base_score += (
                        self.config.influence_weight *
                        collaborative_results["predicted_interests"]["confidence"] *
                        prediction
                    )
            
            solution_scores.append((sol, base_score))
        
        solution_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "similar": {
                "solutions": [s[0] for s in solution_scores],
                "scores": [s[1] for s in solution_scores]
            },
            "diverse": base_recommendations["diverse"],
            "next": {
                "solutions": [s[0] for s in solution_scores[:self.recommender.config.n_neighbors]],
                "scores": [s[1] for s in solution_scores[:self.recommender.config.n_neighbors]]
            }
        }
    
    def _cluster_users(self) -> List[Dict[str, Any]]:
        """Cluster users based on preferences."""
        if not self.user_factors is not None:
            return []
        
        from sklearn.cluster import KMeans
        
        # Cluster user factors
        kmeans = KMeans(
            n_clusters=min(3, len(self.user_profiles)),
            random_state=42
        )
        
        clusters = kmeans.fit_predict(self.user_factors)
        
        # Analyze clusters
        cluster_results = []
        users = list(self.user_profiles.keys())
        
        for i in range(kmeans.n_clusters):
            cluster_users = [
                users[j]
                for j in np.where(clusters == i)[0]
            ]
            
            if cluster_users:
                # Calculate cluster center preferences
                center_weights = {}
                for obj in self.recommender.explorer.optimizer.config.objective_weights:
                    weights = [
                        self.user_profiles[u][-1]["weights"][obj]
                        for u in cluster_users
                    ]
                    center_weights[obj] = float(np.mean(weights))
                
                cluster_results.append({
                    "id": i,
                    "users": cluster_users,
                    "size": len(cluster_users),
                    "center_weights": center_weights,
                    "cohesion": float(
                        np.mean([
                            np.linalg.norm(
                                self.user_factors[users.index(u)] -
                                kmeans.cluster_centers_[i]
                            )
                            for u in cluster_users
                        ])
                    )
                })
        
        return cluster_results
    
    def _calculate_network_layout(
        self,
        network: List[Tuple[str, str]]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate network layout for visualization."""
        import networkx as nx
        
        G = nx.Graph()
        G.add_edges_from(network)
        
        return nx.spring_layout(G)
    
    def save_state(
        self,
        output_path: Optional[Path] = None
    ):
        """Save collaborative filtering state."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            state = {
                "user_profiles": self.user_profiles,
                "user_interactions": self.user_interactions,
                "solution_ids": self.solution_ids,
                "timestamp": datetime.now().isoformat()
            }
            
            if self.user_factors is not None:
                state["user_factors"] = self.user_factors.tolist()
            
            if self.solution_factors is not None:
                state["solution_factors"] = self.solution_factors.tolist()
            
            with open(path / "collaborative_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved collaborative state to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

def create_collaborative_recommender(
    recommender: SolutionRecommender,
    output_path: Optional[Path] = None
) -> CollaborativeRecommender:
    """Create collaborative recommender."""
    config = CollaborationConfig(output_path=output_path)
    return CollaborativeRecommender(recommender, config)

if __name__ == "__main__":
    # Example usage
    from .solution_recommendations import create_solution_recommender
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
    recommender = create_solution_recommender(explorer)
    collab = create_collaborative_recommender(
        recommender,
        output_path=Path("collaborative_recommendations")
    )
    
    # Example usage
    weights = {
        "power_error": 0.5,
        "computation_cost": 0.3,
        "stability": 0.2
    }
    
    # Add some user interactions
    collab.add_user_interaction(
        "user1",
        weights,
        [1, 2, 3]
    )
    
    collab.add_user_interaction(
        "user2",
        {
            "power_error": 0.3,
            "computation_cost": 0.5,
            "stability": 0.2
        },
        [2, 3, 4]
    )
    
    # Get recommendations
    recommendations = collab.get_collaborative_recommendations("user1", weights)
    
    # Visualize recommendations
    viz = collab.visualize_collaboration(recommendations)
    viz.show()
    
    # Save state
    collab.save_state()
