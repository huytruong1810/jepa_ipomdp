"""
Interpretability: probes from learned latents to exact Bayes posteriors, guarantees that turn the
decoding error into bounds on values and decisions, and the full belief analysis of a trained agent
(canonical Tiger); probes against the exact enumeration filter and belief maps (Wumpus World).
"""

from .analysis import (BeliefAnalysis, DecodedBeliefAgent, ExactBeliefAgent, GeometryReport, ReturnEstimate,
                       analyze_beliefs, latent_geometry)
from .belief_probe import (BeliefProbe, ProbeDataset, ProbeReport, build_probe_dataset, evaluate_probe,
                           fit_linear_probe, fit_mlp_probe)
from .error_bounds import ErrorBoundReport, error_bounds, lipschitz_constant, q_values
from .wumpus_latents import (WumpusLatentDataset, WumpusProbe, WumpusProbeReport, build_wumpus_dataset,
                             evaluate_wumpus_probe, fit_linear_wumpus_probe, fit_mlp_wumpus_probe, plot_belief_maps,
                             plot_latent_pca)

__all__ = [
    "BeliefAnalysis",
    "BeliefProbe",
    "DecodedBeliefAgent",
    "ErrorBoundReport",
    "ExactBeliefAgent",
    "GeometryReport",
    "ProbeDataset",
    "ProbeReport",
    "ReturnEstimate",
    "WumpusLatentDataset",
    "WumpusProbe",
    "WumpusProbeReport",
    "analyze_beliefs",
    "build_probe_dataset",
    "build_wumpus_dataset",
    "error_bounds",
    "evaluate_probe",
    "evaluate_wumpus_probe",
    "fit_linear_probe",
    "fit_linear_wumpus_probe",
    "fit_mlp_probe",
    "fit_mlp_wumpus_probe",
    "latent_geometry",
    "lipschitz_constant",
    "plot_belief_maps",
    "plot_latent_pca",
    "q_values",
]
