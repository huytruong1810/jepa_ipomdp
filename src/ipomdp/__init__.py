"""
JEPA-IPOMDP: a recurrent Joint-Embedding Predictive Architecture belief filter with latent
MCTS planning for (Interactive) POMDPs.

Subpackages (import from them directly; this top-level module re-exports nothing so that
importing one layer never drags in the others):
    domain     exact POMDP specifications, Bayes filter, exact solver, batched simulator
    models     JEPA world model, distributional heads, network building blocks
    planning   latent-space MCTS
    agents     batched belief-filtering agent
    training   sequence replay buffer and trainer
    telemetry  logging, metrics, visualisation, checkpointing, system monitoring
"""
