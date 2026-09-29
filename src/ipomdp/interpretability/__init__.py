"""
Interpretability: probes from learned latents to exact Bayes posteriors, guarantees that turn the
decoding error into bounds on values and decisions, and the full belief analysis of a trained agent.
"""

from .analysis import (BeliefAnalysis, DecodedBeliefAgent, ExactBeliefAgent, GeometryReport, ReturnEstimate,
                       analyze_beliefs, latent_geometry)
from .belief_probe import (BeliefProbe, ProbeDataset, ProbeReport, build_probe_dataset, evaluate_probe,
                           fit_linear_probe, fit_mlp_probe)
from .error_bounds import ErrorBoundReport, error_bounds, lipschitz_constant, q_values

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
    "analyze_beliefs",
    "build_probe_dataset",
    "error_bounds",
    "evaluate_probe",
    "fit_linear_probe",
    "fit_mlp_probe",
    "latent_geometry",
    "lipschitz_constant",
    "q_values",
]
