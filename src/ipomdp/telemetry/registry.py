# ABSOLUTE PATH: src/ipomdp/telemetry/registry.py
# ==============================================================================
# FACTORY PATTERN REGISTRY & DEPENDENCY INJECTION ENGINE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Declarative Dependency Injection:
#    - Maps string identifiers from Hydra configuration files directly to environment
#      and neural feature extractor classes without hardcoded import dependencies.
#
# 2. Registration Collision & Lookup Safeguards:
#    - Detects and prevents duplicate registrations under identical string keys.
#    - Provides explicit, sorted lists of valid registered components upon lookup failure.
# ==============================================================================

from typing import Callable, Dict, Any, Type

_ENVS: Dict[str, Type] = {}
_EXTRACTORS: Dict[str, Type] = {}


def register_env(name: str) -> Callable[[Type], Type]:
    """
    Decorator to register a new multi-agent environment class under a string identifier.

    Args:
        name: Unique string lookup key (e.g., 'tiger').
    """
    def _register(cls: Type) -> Type:
        if name in _ENVS:
            raise ValueError(f"Environment '{name}' is already registered under class {_ENVS[name].__name__}!")
        _ENVS[name] = cls
        return cls
    return _register


def make_env(name: str, **kwargs) -> Any:
    """
    Instantiates a registered environment class by string identifier.

    Args:
        name: Registered environment key.
        **kwargs: Dynamic keyword arguments forwarded to the environment constructor.

    Returns:
        Instantiated environment conforming to IPOMDPEnv interface.
    """
    if name not in _ENVS:
        raise KeyError(
            f"Environment '{name}' not found in registry. Available choices: {sorted(list(_ENVS.keys()))}"
        )
    return _ENVS[name](**kwargs)


def register_extractor(name: str) -> Callable[[Type], Type]:
    """
    Decorator to register a new perceptual feature extractor under a string identifier.

    Args:
        name: Unique string lookup key (e.g., 'mlp', 'cnn').
    """
    def _register(cls: Type) -> Type:
        if name in _EXTRACTORS:
            raise ValueError(f"Extractor '{name}' is already registered under class {_EXTRACTORS[name].__name__}!")
        _EXTRACTORS[name] = cls
        return cls
    return _register


def make_extractor(name: str, **kwargs) -> Any:
    """
    Instantiates a registered perceptual feature extractor by string identifier.

    Args:
        name: Registered extractor key.
        **kwargs: Dynamic keyword arguments forwarded to extractor constructor.

    Returns:
        Instantiated feature extractor conforming to FeatureExtractor interface.
    """
    if name not in _EXTRACTORS:
        raise KeyError(
            f"Extractor '{name}' not found in registry. Available choices: {sorted(list(_EXTRACTORS.keys()))}"
        )
    return _EXTRACTORS[name](**kwargs)
