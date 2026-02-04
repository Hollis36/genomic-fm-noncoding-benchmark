"""Conservation-based baseline methods (CADD, PhyloP, PhastCons)."""

import pyBigWig

from .base import BaselineMethod


class CADDScorer(BaselineMethod):
    """
    CADD (Combined Annotation Dependent Depletion) scorer.

    Requires precomputed CADD scores in BigWig or tabix-indexed format.
    """

    def __init__(self, data_path: str):
        """
        Initialize CADD scorer.

        Args:
            data_path: Path to CADD scores file
        """
        super().__init__(data_path)
        self.bw = None

        if data_path.endswith(".bw") or data_path.endswith(".bigWig"):
            self.bw = pyBigWig.open(data_path)

    def score_variant(self, chrom: str, pos: int, ref: str, alt: str) -> float:
        """
        Get CADD score for a variant.

        Args:
            chrom: Chromosome
            pos: Position
            ref: Reference allele
            alt: Alternate allele

        Returns:
            CADD score
        """
        if self.bw is None:
            raise RuntimeError("CADD scores not loaded")

        try:
            # BigWig stores continuous values
            score = self.bw.values(chrom, pos - 1, pos)

            if score and len(score) > 0:
                return float(score[0])

            return 0.0

        except Exception:
            return 0.0

    def __del__(self):
        """Close file handle."""
        if self.bw is not None:
            self.bw.close()


class PhyloPScorer(BaselineMethod):
    """
    PhyloP conservation scorer.

    Requires precomputed PhyloP scores in BigWig format.
    """

    def __init__(self, data_path: str):
        """
        Initialize PhyloP scorer.

        Args:
            data_path: Path to PhyloP BigWig file
        """
        super().__init__(data_path)
        self.bw = pyBigWig.open(data_path)

    def score_variant(self, chrom: str, pos: int, ref: str, alt: str) -> float:
        """
        Get PhyloP conservation score for a position.

        Args:
            chrom: Chromosome
            pos: Position
            ref: Reference allele (not used)
            alt: Alternate allele (not used)

        Returns:
            PhyloP score
        """
        try:
            score = self.bw.values(chrom, pos - 1, pos)

            if score and len(score) > 0:
                return float(score[0])

            return 0.0

        except Exception:
            return 0.0

    def __del__(self):
        """Close file handle."""
        if self.bw is not None:
            self.bw.close()


class PhastConsScorer(BaselineMethod):
    """
    PhastCons conservation scorer.

    Requires precomputed PhastCons scores in BigWig format.
    """

    def __init__(self, data_path: str):
        """
        Initialize PhastCons scorer.

        Args:
            data_path: Path to PhastCons BigWig file
        """
        super().__init__(data_path)
        self.bw = pyBigWig.open(data_path)

    def score_variant(self, chrom: str, pos: int, ref: str, alt: str) -> float:
        """
        Get PhastCons conservation score for a position.

        Args:
            chrom: Chromosome
            pos: Position
            ref: Reference allele (not used)
            alt: Alternate allele (not used)

        Returns:
            PhastCons score
        """
        try:
            score = self.bw.values(chrom, pos - 1, pos)

            if score and len(score) > 0:
                return float(score[0])

            return 0.0

        except Exception:
            return 0.0

    def __del__(self):
        """Close file handle."""
        if self.bw is not None:
            self.bw.close()
