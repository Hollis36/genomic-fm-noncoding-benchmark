"""Baseline methods for variant pathogenicity prediction."""

from .base import BaselineMethod
from .conservation_scores import CADDScorer, PhastConsScorer, PhyloP Scorer

__all__ = [
    "BaselineMethod",
    "CADDScorer",
    "PhyloP Scorer",
    "PhastConsScorer",
]
