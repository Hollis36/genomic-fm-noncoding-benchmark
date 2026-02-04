"""Model registry and factory for genomic foundation models."""

from typing import Dict, Type

from .base import GenomicModelBase
from .caduceus import CaduceusModel
from .dnabert2 import DNABERT2Model
from .evo import EvoModel
from .hyenadna import HyenaDNAModel
from .nucleotide_transformer import NucleotideTransformerModel

# Model registry mapping model keys to classes
MODEL_REGISTRY: Dict[str, Type[GenomicModelBase]] = {
    "dnabert2": DNABERT2Model,
    "nucleotide_transformer_500m": NucleotideTransformerModel,
    "nucleotide_transformer_2500m": NucleotideTransformerModel,
    "hyenadna": HyenaDNAModel,
    "caduceus": CaduceusModel,
    "evo1": EvoModel,
}


def register_model(name: str, model_class: Type[GenomicModelBase]) -> None:
    """
    Register a new model class in the registry.

    Args:
        name: Model key name
        model_class: Model class (must inherit from GenomicModelBase)
    """
    if not issubclass(model_class, GenomicModelBase):
        raise TypeError(
            f"Model class must inherit from GenomicModelBase, got {model_class}"
        )

    if name in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' is already registered. "
            f"Use unregister_model first if you want to replace it."
        )

    MODEL_REGISTRY[name] = model_class


def unregister_model(name: str) -> None:
    """
    Unregister a model from the registry.

    Args:
        name: Model key name
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered")

    del MODEL_REGISTRY[name]


def list_available_models() -> list[str]:
    """
    Get list of all available model keys.

    Returns:
        List of registered model keys
    """
    return list(MODEL_REGISTRY.keys())


def create_model(model_key: str, config: dict) -> GenomicModelBase:
    """
    Factory function to create a model instance from a config.

    Args:
        model_key: Model key (e.g., 'dnabert2', 'hyenadna')
        config: Model configuration dictionary

    Returns:
        Instantiated model (not yet loaded)

    Raises:
        ValueError: If model_key is not registered
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key: '{model_key}'. "
            f"Available models: {list_available_models()}"
        )

    model_class = MODEL_REGISTRY[model_key]
    return model_class(config)


def load_model(model_key: str, config: dict) -> GenomicModelBase:
    """
    Create and load a model in one step.

    Args:
        model_key: Model key
        config: Model configuration dictionary

    Returns:
        Loaded model instance
    """
    model = create_model(model_key, config)
    model.load()
    return model
