"""
JEPA-IPOMDP: a recurrent Joint-Embedding Predictive Architecture belief filter with latent
MCTS planning for (Interactive) POMDPs.

Subpackages (import from them directly; this top-level module re-exports nothing so that
importing one layer never drags in the others):
    domain            exact POMDP specifications, Bayes filter, exact solver, batched simulator
    models            JEPA belief filter, distributional heads, network building blocks
    planning          belief-tree search over an exact or a learned search model
    agents            batched planning agent and uniform random agent
    training          episode replay, world-model trainer, rollouts, checkpointable TrainingRun
    interpretability  belief probes, error bounds, belief analysis against exact references
    experiments       config -> run, run-directory analysis, aggregation over seeds
    telemetry         TensorBoard logging, visualisation, system monitoring, guardrails
"""
