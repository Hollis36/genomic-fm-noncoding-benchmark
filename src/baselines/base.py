"""Base class for baseline methods."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaselineMethod(ABC):
    """Abstract base class for baseline variant scoring methods."""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize baseline method.

        Args:
            data_path: Path to precomputed scores (if applicable)
        """
        self.data_path = data_path
        self.name = self.__class__.__name__

    @abstractmethod
    def score_variant(self, chrom: str, pos: int, ref: str, alt: str) -> float:
        """
        Score a single variant.

        Args:
            chrom: Chromosome
            pos: Position
            ref: Reference allele
            alt: Alternate allele

        Returns:
            Pathogenicity score
        """
        pass

    def score_variants(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score multiple variants from a DataFrame.

        Args:
            df: DataFrame with columns: chrom, pos, ref, alt

        Returns:
            Array of scores
        """
        scores = []

        for _, row in df.iterrows():
            score = self.score_variant(
                row["chrom"],
                int(row["pos"]),
                row["ref"],
                row["alt"],
            )
            scores.append(score)

        return np.array(scores)

    def __repr__(self):
        """String representation."""
        return f"{self.name}(data_path={self.data_path})"
