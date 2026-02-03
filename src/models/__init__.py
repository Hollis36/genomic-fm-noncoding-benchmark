from .base import GenomicModelBase
from .dnabert2 import DNABERT2Model
from .nucleotide_transformer import NucleotideTransformerModel
from .hyenadna import HyenaDNAModel
from .caduceus import CaduceusModel
from .evo import EvoModel

MODEL_CLASSES = {
    "dnabert2": DNABERT2Model,
    "nucleotide_transformer_500m": NucleotideTransformerModel,
    "nucleotide_transformer_2500m": NucleotideTransformerModel,
    "hyenadna": HyenaDNAModel,
    "caduceus": CaduceusModel,
    "evo1": EvoModel,
}


def load_model(model_key: str, config: dict) -> GenomicModelBase:
    """Factory function to load a model by key."""
    if model_key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CLASSES.keys())}")
    model_cls = MODEL_CLASSES[model_key]
    return model_cls(config)
