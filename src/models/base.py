"""
Abstract base class for all genomic foundation models.

Each model must implement:
  - get_embedding(seq) -> np.ndarray
  - score_variant(ref_seq, alt_seq) -> float
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class GenomicModelBase(ABC):
    """Base class for genomic foundation model wrappers."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["name"]
        self.model_type = config["type"]  # 'encoder' or 'causal'
        self.max_length = config.get("max_length", 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self):
        """Load model and tokenizer."""
        pass

    @abstractmethod
    def get_embedding(self, sequence: str) -> np.ndarray:
        """
        Extract a fixed-size embedding vector for a DNA sequence.

        Args:
            sequence: DNA sequence string (A/C/G/T)

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        """
        Compute a zero-shot variant effect score.

        For encoder models: cosine distance between ref and alt embeddings.
        For causal models: log-likelihood ratio (ref - alt).

        Higher score = more likely to be pathogenic.

        Args:
            ref_seq: Reference sequence with variant context
            alt_seq: Alternate sequence with variant context

        Returns:
            Float score (higher = more likely damaging)
        """
        pass

    def get_embeddings_batch(self, sequences: list[str], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings for a batch of sequences."""
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_embs = [self.get_embedding(seq) for seq in batch]
            all_embeddings.extend(batch_embs)
        return np.array(all_embeddings)

    def score_variants_batch(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Score a batch of variants."""
        scores = []
        for i in range(0, len(ref_seqs), batch_size):
            batch_ref = ref_seqs[i:i + batch_size]
            batch_alt = alt_seqs[i:i + batch_size]
            batch_scores = [
                self.score_variant(r, a) for r, a in zip(batch_ref, batch_alt)
            ]
            scores.extend(batch_scores)
        return np.array(scores)

    @property
    def num_parameters(self) -> int:
        """Return the number of model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.model_name}, params={self.config.get('params', '?')})"
