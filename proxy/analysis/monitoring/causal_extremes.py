"""Causality analysis for extreme events."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.nonparametric.kernel_regression import KernelReg
import networkx as nx
from tigramite import data_processing as pp
from tigramite import preprocessing as pre
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .temporal_extremes import (
    TemporalAnalyzer, TemporalConfig, SeasonalPattern,
    TrendComponent, ChangePoint, TemporalResult
)

@dataclass
class CausalConfig:
    """Configuration for causality analysis."""
    enabled: bool = True
    update_interval: float = 300.0  # 5 minutes
    max_lag: int = 24  # hours
    significance_level: float = 0.05
    min_effect_size: float = 0.1
    enable_nonlinear: bool = True
    enable_bootstrap: bool = True
    bootstrap_samples: int = 1000
    enable_transfer_entropy: bool = True
    num_surrogates: int = 100
    pcmci_alpha: float = 0.05
    max_cmi_lag: int = 12
    tau_min: int = 0
    tau_max: int = 5
    visualization_dir: Optional[str] = "causal_extremes"

@dataclass
class CausalLink:
    """Causal link between variables."""
    source: str
    target: str
    lag: int
    effect_size: float
    significance: float
    confidence_interval: Tuple[float, float]
    nonlinear_strength: Optional[float] = None
    transfer_entropy: Optional[float] = None

@dataclass
class CausalNetwork:
    """Network of causal relationships."""
    nodes: List[str]
    links: List[CausalLink]
    adjacency_matrix: np.ndarray
    centrality_scores: Dict[str, float]
    communities: List[Set[str]]
    stability_score: float

@dataclass
class CausalResult:
    """Results of causality analysis."""
    direct_causes: Dict[str, List[CausalLink]]
    feedback_loops: List[List[CausalLink]]
    networks: Dict[str, CausalNetwork]
    mediation_effects: Dict[str, Dict[str, float]]
    temporal_dependencies: Dict[str, Dict[str, List[float]]]
    model_diagnostics: Dict[str, Dict[str, float]]

class CausalAnalyzer:
    """Analyze causal relationships in extreme events."""
    
    def __init__(
        self,
        temporal_analyzer: TemporalAnalyzer,
        config: CausalConfig = None
    ):
        self.temporal_analyzer = temporal_analyzer
        self.config = config or CausalConfig()
        
        # Analysis state
        self.results: Dict[str, CausalResult] = {}
        self.networks: Dict[str, nx.DiGraph] = {}
        self.pcmci: Optional[PCMCI] = None
        
        # Monitoring state
        self.last_update = datetime.min
        self.analyzer_task: Optional[asyncio.Task] = None
    
    async def start_analyzer(self):
        """Start causal analyzer."""
        if not self.config.enabled:
            return
        
        if self.analyzer_task is None:
            self.analyzer_task = asyncio.create_task(self._run_analyzer())
    
    async def stop_analyzer(self):
        """Stop causal analyzer."""
        if self.analyzer_task:
            self.analyzer_task.cancel()
            try:
                await self.analyzer_task
            except asyncio.CancelledError:
                pass
            self.analyzer_task = None
    
    async def _run_analyzer(self):
        """Run periodic analysis."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    for scenario in self.temporal_analyzer.results:
                        await self.analyze_causality(scenario)
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Causal analyzer error: {e}")
                await asyncio.sleep(60)
    
    async def analyze_causality(
        self,
        scenario_name: str
    ) -> Optional[CausalResult]:
        """Analyze causal relationships in scenario."""
        if scenario_name not in self.temporal_analyzer.results:
            return None
        
        result = self.temporal_analyzer.results[scenario_name]
        if not result.predictions:
            return None
        
        # Initialize results
        direct_causes = {}
        feedback_loops = []
        networks = {}
        mediation_effects = {}
        temporal_dependencies = {}
        model_diagnostics = {}
        
        # Prepare time series data
        ts_data = await self._prepare_data(scenario_name)
        if not ts_data.empty:
            # Analyze direct causal relationships
            causes = await self._analyze_direct_causes(ts_data)
            direct_causes.update(causes)
            
            # Detect feedback loops
            loops = await self._detect_feedback_loops(causes)
            feedback_loops.extend(loops)
            
            # Build causal networks
            nets = await self._build_networks(ts_data, causes)
            networks.update(nets)
            
            # Analyze mediation effects
            effects = await self._analyze_mediation(ts_data, causes)
            mediation_effects.update(effects)
            
            # Analyze temporal dependencies
            deps = await self._analyze_temporal_dependencies(ts_data)
            temporal_dependencies.update(deps)
            
            # Calculate model diagnostics
            diagnostics = await self._calculate_diagnostics(ts_data, causes)
            model_diagnostics.update(diagnostics)
        
        # Create result
        result = CausalResult(
            direct_causes=direct_causes,
            feedback_loops=feedback_loops,
            networks=networks,
            mediation_effects=mediation_effects,
            temporal_dependencies=temporal_dependencies,
            model_diagnostics=model_diagnostics
        )
        
        self.results[scenario_name] = result
        
        return result
    
    async def _prepare_data(
        self,
        scenario: str
    ) -> pd.DataFrame:
        """Prepare time series data for analysis."""
        data = {}
        
        result = self.temporal_analyzer.results[scenario]
        for metric, predictions in result.predictions.items():
            data[metric] = predictions.values
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    async def _analyze_direct_causes(
        self,
        ts_data: pd.DataFrame
    ) -> Dict[str, List[CausalLink]]:
        """Analyze direct causal relationships."""
        causes = {}
        variables = ts_data.columns
        
        # Initialize PCMCI
        if self.pcmci is None:
            dataframe = pp.DataFrame(
                ts_data.values,
                var_names=variables
            )
            parcorr = ParCorr(significance='analytic')
            self.pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=parcorr,
                verbosity=0
            )
        
        # Run PCMCI
        results = self.pcmci.run_pcmci(
            tau_min=self.config.tau_min,
            tau_max=self.config.tau_max,
            pc_alpha=self.config.pcmci_alpha
        )
        
        # Extract causal links
        for i, source in enumerate(variables):
            causes[source] = []
            
            for j, target in enumerate(variables):
                if i == j:
                    continue
                
                for lag in range(self.config.tau_min, self.config.tau_max + 1):
                    pval = results['p_matrix'][i, j, lag]
                    if pval < self.config.significance_level:
                        # Calculate effect size
                        effect = abs(results['val_matrix'][i, j, lag])
                        if effect > self.config.min_effect_size:
                            # Calculate confidence interval
                            ci = await self._bootstrap_confidence(
                                ts_data,
                                source,
                                target,
                                lag
                            )
                            
                            # Calculate nonlinear strength if enabled
                            nonlinear = None
                            if self.config.enable_nonlinear:
                                nonlinear = await self._calculate_nonlinear_strength(
                                    ts_data,
                                    source,
                                    target,
                                    lag
                                )
                            
                            # Calculate transfer entropy if enabled
                            te = None
                            if self.config.enable_transfer_entropy:
                                te = await self._calculate_transfer_entropy(
                                    ts_data,
                                    source,
                                    target,
                                    lag
                                )
                            
                            link = CausalLink(
                                source=source,
                                target=target,
                                lag=lag,
                                effect_size=effect,
                                significance=1 - pval,
                                confidence_interval=ci,
                                nonlinear_strength=nonlinear,
                                transfer_entropy=te
                            )
                            causes[source].append(link)
        
        return causes
    
    async def _bootstrap_confidence(
        self,
        ts_data: pd.DataFrame,
        source: str,
        target: str,
        lag: int
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if not self.config.enable_bootstrap:
            return (-1.0, 1.0)
        
        effects = []
        n_samples = len(ts_data)
        
        for _ in range(self.config.bootstrap_samples):
            # Resample with replacement
            indices = np.random.randint(0, n_samples, n_samples)
            sample = ts_data.iloc[indices]
            
            # Calculate effect
            source_data = sample[source].values[:-lag]
            target_data = sample[target].values[lag:]
            
            if len(source_data) > 0 and len(target_data) > 0:
                correlation = np.corrcoef(source_data, target_data)[0, 1]
                effects.append(correlation)
        
        if not effects:
            return (-1.0, 1.0)
        
        # Calculate confidence interval
        return tuple(np.percentile(
            effects,
            [(1 - self.config.significance_level) * 50,
             (1 + self.config.significance_level) * 50]
        ))
    
    async def _calculate_nonlinear_strength(
        self,
        ts_data: pd.DataFrame,
        source: str,
        target: str,
        lag: int
    ) -> Optional[float]:
        """Calculate nonlinear causal strength."""
        try:
            # Prepare data
            x = ts_data[source].values[:-lag]
            y = ts_data[target].values[lag:]
            
            if len(x) < 2 or len(y) < 2:
                return None
            
            # Fit nonparametric regression
            kr = KernelReg(y, x[:, None], var_type='c')
            y_pred, y_std = kr.fit(x[:, None])
            
            # Calculate nonlinear R²
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            
            return 1 - ss_res / ss_tot
            
        except Exception as e:
            print(f"Nonlinear strength error: {e}")
            return None
    
    async def _calculate_transfer_entropy(
        self,
        ts_data: pd.DataFrame,
        source: str,
        target: str,
        lag: int
    ) -> Optional[float]:
        """Calculate transfer entropy."""
        try:
            from tigramite.independence_tests import CMIknn
            
            # Prepare data
            x = ts_data[source].values[:-lag]
            y = ts_data[target].values[lag:]
            
            if len(x) < 3 or len(y) < 3:
                return None
            
            # Calculate CMI using k-nearest neighbors
            cmi_estimator = CMIknn(significance='shuffle_test',
                                 sig_samples=self.config.num_surrogates)
            
            te = cmi_estimator.get_dependence_measure(
                x.reshape(-1, 1),
                y.reshape(-1, 1),
                np.array([])
            )
            
            return float(te)
            
        except Exception as e:
            print(f"Transfer entropy error: {e}")
            return None
    
    async def _detect_feedback_loops(
        self,
        causes: Dict[str, List[CausalLink]]
    ) -> List[List[CausalLink]]:
        """Detect feedback loops in causal relationships."""
        loops = []
        
        # Create directed graph
        G = nx.DiGraph()
        
        for source, links in causes.items():
            for link in links:
                G.add_edge(
                    source,
                    link.target,
                    lag=link.lag,
                    effect=link.effect_size
                )
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(G))
            
            for cycle in cycles:
                cycle_links = []
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    
                    for link in causes[source]:
                        if link.target == target:
                            cycle_links.append(link)
                
                if cycle_links:
                    loops.append(cycle_links)
        except:
            pass
        
        return loops
    
    async def _build_networks(
        self,
        ts_data: pd.DataFrame,
        causes: Dict[str, List[CausalLink]]
    ) -> Dict[str, CausalNetwork]:
        """Build causal networks."""
        networks = {}
        
        # Create time-aggregated network
        nodes = list(ts_data.columns)
        n_nodes = len(nodes)
        
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for source, links in causes.items():
            source_idx = nodes.index(source)
            for link in links:
                target_idx = nodes.index(link.target)
                adj_matrix[source_idx, target_idx] = link.effect_size
        
        # Create graph
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        nx.relabel_nodes(G, {i: nodes[i] for i in range(n_nodes)}, copy=False)
        
        # Calculate centrality scores
        centrality = nx.eigenvector_centrality_numpy(G)
        
        # Detect communities
        communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
        
        # Calculate stability
        stability = await self._calculate_network_stability(G, ts_data)
        
        networks["aggregate"] = CausalNetwork(
            nodes=nodes,
            links=sum([links for links in causes.values()], []),
            adjacency_matrix=adj_matrix,
            centrality_scores=centrality,
            communities=[{nodes[i] for i in comm} for comm in communities],
            stability_score=stability
        )
        
        return networks
    
    async def _analyze_mediation(
        self,
        ts_data: pd.DataFrame,
        causes: Dict[str, List[CausalLink]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze mediation effects."""
        mediation = {}
        
        for source, links in causes.items():
            mediation[source] = {}
            
            for link in links:
                for mediator in ts_data.columns:
                    if mediator != source and mediator != link.target:
                        effect = await self._calculate_mediation_effect(
                            ts_data,
                            source,
                            mediator,
                            link.target,
                            link.lag
                        )
                        if effect is not None:
                            mediation[source][f"{mediator}->{link.target}"] = effect
        
        return mediation
    
    async def _calculate_mediation_effect(
        self,
        ts_data: pd.DataFrame,
        source: str,
        mediator: str,
        target: str,
        lag: int
    ) -> Optional[float]:
        """Calculate mediation effect."""
        try:
            # Prepare data
            X = ts_data[source].values[:-lag]
            M = ts_data[mediator].values[:-lag]
            Y = ts_data[target].values[lag:]
            
            if len(X) < 3:
                return None
            
            # Calculate direct effect
            direct = stats.pearsonr(X, Y)[0]
            
            # Calculate indirect effect
            a = stats.pearsonr(X, M)[0]  # X -> M
            b = stats.pearsonr(M, Y)[0]  # M -> Y
            indirect = a * b
            
            # Calculate proportion mediated
            total = direct + indirect
            if abs(total) > 1e-10:
                return indirect / total
            else:
                return 0.0
            
        except Exception as e:
            print(f"Mediation error: {e}")
            return None
    
    async def _analyze_temporal_dependencies(
        self,
        ts_data: pd.DataFrame
    ) -> Dict[str, Dict[str, List[float]]]:
        """Analyze temporal dependencies."""
        dependencies = {}
        
        for source in ts_data.columns:
            dependencies[source] = {}
            
            for target in ts_data.columns:
                if source != target:
                    # Calculate cross-correlation
                    xcorr = [
                        stats.pearsonr(
                            ts_data[source].values[:-lag],
                            ts_data[target].values[lag:]
                        )[0]
                        for lag in range(1, self.config.max_lag + 1)
                    ]
                    
                    dependencies[source][target] = xcorr
        
        return dependencies
    
    async def _calculate_network_stability(
        self,
        G: nx.DiGraph,
        ts_data: pd.DataFrame
    ) -> float:
        """Calculate network stability score."""
        try:
            # Calculate spectral radius
            eigs = np.linalg.eigvals(nx.adjacency_matrix(G).todense())
            spectral_radius = max(abs(eigs))
            
            # Calculate temporal stability
            temporal_stability = np.mean([
                1 - np.std(ts_data[node]) / np.mean(ts_data[node])
                for node in G.nodes
                if np.mean(ts_data[node]) != 0
            ])
            
            # Combine metrics
            return 1 / (1 + spectral_radius) * temporal_stability
            
        except Exception as e:
            print(f"Stability error: {e}")
            return 0.0
    
    async def _calculate_diagnostics(
        self,
        ts_data: pd.DataFrame,
        causes: Dict[str, List[CausalLink]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate model diagnostics."""
        diagnostics = {}
        
        for source, links in causes.items():
            source_diag = {}
            
            # Calculate stationarity
            adf_stat, adf_pval = adfuller(ts_data[source])[:2]
            source_diag["stationarity"] = 1 - adf_pval
            
            # Calculate predictability
            if links:
                y_true = ts_data[source].values
                y_pred = np.zeros_like(y_true)
                
                for link in links:
                    lag_data = ts_data[link.target].values[:-link.lag]
                    if len(lag_data) == len(y_true[link.lag:]):
                        y_pred[link.lag:] += link.effect_size * lag_data
                
                mse = np.mean((y_true - y_pred) ** 2)
                var = np.var(y_true)
                source_diag["predictability"] = 1 - mse / var
            else:
                source_diag["predictability"] = 0.0
            
            # Calculate causality strength
            source_diag["causality_strength"] = np.mean([
                link.effect_size for link in links
            ]) if links else 0.0
            
            diagnostics[source] = source_diag
        
        return diagnostics
    
    async def create_causal_plots(self) -> Dict[str, go.Figure]:
        """Create causality visualization plots."""
        plots = {}
        
        for scenario_name, result in self.results.items():
            # Network plot
            if scenario_name in result.networks:
                network = result.networks[scenario_name]
                
                net_fig = go.Figure()
                
                # Create node positions using Fruchterman-Reingold layout
                G = nx.from_numpy_array(network.adjacency_matrix)
                pos = nx.spring_layout(G)
                
                # Add edges
                edge_x = []
                edge_y = []
                for i, j in G.edges():
                    x0, y0 = pos[i]
                    x1, y1 = pos[j]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                net_fig.add_trace(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=0.5, color="#888"),
                    hoverinfo="none",
                    mode="lines"
                ))
                
                # Add nodes
                node_x = [pos[i][0] for i in G.nodes()]
                node_y = [pos[i][1] for i in G.nodes()]
                
                net_fig.add_trace(go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=network.nodes,
                    textposition="top center",
                    hoverinfo="text",
                    marker=dict(
                        size=10,
                        color=[
                            network.centrality_scores[node]
                            for node in network.nodes
                        ],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Centrality")
                    )
                ))
                
                net_fig.update_layout(
                    title=f"Causal Network - {scenario_name}",
                    showlegend=False,
                    hovermode="closest"
                )
                plots[f"{scenario_name}_network"] = net_fig
            
            # Temporal dependency plot
            if result.temporal_dependencies:
                dep_fig = go.Figure()
                
                for source, targets in result.temporal_dependencies.items():
                    for target, xcorr in targets.items():
                        dep_fig.add_trace(go.Scatter(
                            x=list(range(1, len(xcorr) + 1)),
                            y=xcorr,
                            name=f"{source}->{target}",
                            mode="lines+markers"
                        ))
                
                dep_fig.update_layout(
                    title=f"Temporal Dependencies - {scenario_name}",
                    xaxis_title="Lag",
                    yaxis_title="Cross-correlation",
                    showlegend=True
                )
                plots[f"{scenario_name}_dependencies"] = dep_fig
            
            # Mediation plot
            if result.mediation_effects:
                med_fig = go.Figure()
                
                sources = []
                paths = []
                effects = []
                
                for source, mediations in result.mediation_effects.items():
                    for path, effect in mediations.items():
                        sources.append(source)
                        paths.append(path)
                        effects.append(effect)
                
                med_fig.add_trace(go.Heatmap(
                    x=paths,
                    y=sources,
                    z=[effects],
                    colorscale="RdBu",
                    zmid=0
                ))
                
                med_fig.update_layout(
                    title=f"Mediation Effects - {scenario_name}",
                    xaxis_title="Mediation Path",
                    yaxis_title="Source",
                    showlegend=False
                )
                plots[f"{scenario_name}_mediation"] = med_fig
            
            # Diagnostics plot
            if result.model_diagnostics:
                diag_fig = go.Figure()
                
                metrics = ["stationarity", "predictability", "causality_strength"]
                variables = list(result.model_diagnostics.keys())
                
                for metric in metrics:
                    diag_fig.add_trace(go.Bar(
                        name=metric,
                        x=variables,
                        y=[
                            result.model_diagnostics[var][metric]
                            for var in variables
                        ]
                    ))
                
                diag_fig.update_layout(
                    title=f"Model Diagnostics - {scenario_name}",
                    xaxis_title="Variable",
                    yaxis_title="Score",
                    barmode="group",
                    showlegend=True
                )
                plots[f"{scenario_name}_diagnostics"] = diag_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"causal_{name}.html"))
        
        return plots

def create_causal_analyzer(
    temporal_analyzer: TemporalAnalyzer,
    config: Optional[CausalConfig] = None
) -> CausalAnalyzer:
    """Create causal analyzer."""
    return CausalAnalyzer(temporal_analyzer, config)

if __name__ == "__main__":
    from .temporal_extremes import create_temporal_analyzer
    from .extreme_value_analysis import create_extreme_analyzer
    from .probabilistic_modeling import create_probabilistic_modeler
    from .whatif_analysis import create_whatif_analyzer
    from .scenario_planning import create_scenario_planner
    from .risk_prediction import create_risk_predictor
    from .risk_assessment import create_risk_analyzer
    from .strategy_recommendations import create_strategy_advisor
    from .prevention_balancing import create_prevention_balancer
    from .leak_prevention import create_leak_prevention
    from .memory_leak_detection import create_leak_detector
    from .scheduler_profiling import create_profiling_hook
    
    async def main():
        # Setup components
        profiling = create_profiling_hook()
        detector = create_leak_detector(profiling)
        prevention = create_leak_prevention(detector)
        balancer = create_prevention_balancer(prevention)
        advisor = create_strategy_advisor(balancer)
        analyzer = create_risk_analyzer(advisor)
        predictor = create_risk_predictor(analyzer)
        planner = create_scenario_planner(predictor)
        whatif = create_whatif_analyzer(planner)
        modeler = create_probabilistic_modeler(whatif)
        extreme = create_extreme_analyzer(modeler)
        temporal = create_temporal_analyzer(extreme)
        causal = create_causal_analyzer(temporal)
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        await predictor.start_predictor()
        await planner.start_planner()
        await whatif.start_analyzer()
        await modeler.start_modeler()
        await extreme.start_analyzer()
        await temporal.start_analyzer()
        await causal.start_analyzer()
        
        try:
            while True:
                # Analyze scenarios
                for scenario in temporal.results:
                    result = await causal.analyze_causality(scenario)
                    if result:
                        print(f"\nCausal Analysis for {scenario}:")
                        
                        print("\nDirect Causes:")
                        for source, links in result.direct_causes.items():
                            if links:
                                print(f"\n{source}:")
                                for link in links:
                                    print(
                                        f"  -> {link.target} "
                                        f"(lag={link.lag}, "
                                        f"effect={link.effect_size:.3f}, "
                                        f"sig={link.significance:.3f})"
                                    )
                        
                        if result.feedback_loops:
                            print("\nFeedback Loops:")
                            for loop in result.feedback_loops:
                                print("  " + " -> ".join(
                                    f"{link.source}({link.lag})"
                                    for link in loop
                                ) + " -> " + loop[0].source)
                        
                        print("\nNetwork Properties:")
                        for name, network in result.networks.items():
                            print(f"\n{name}:")
                            print("  Nodes:", len(network.nodes))
                            print("  Links:", len(network.links))
                            print(f"  Stability: {network.stability_score:.3f}")
                            print("  Communities:", len(network.communities))
                
                # Create plots
                await causal.create_causal_plots()
                
                await asyncio.sleep(60)
        finally:
            await causal.stop_analyzer()
            await temporal.stop_analyzer()
            await extreme.stop_analyzer()
            await modeler.stop_modeler()
            await whatif.stop_analyzer()
            await planner.stop_planner()
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_analyzer()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
