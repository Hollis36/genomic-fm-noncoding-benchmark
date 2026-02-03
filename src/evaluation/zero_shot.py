"""
Zero-shot evaluation of genomic foundation models.

Two strategies:
  1. Embedding distance: cosine distance between ref/alt embeddings (encoder models)
  2. Log-likelihood ratio: LL(ref) - LL(alt) (causal models)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..models.base import GenomicModelBase
from .metrics import compute_all_metrics


class ZeroShotEvaluator:
    """Run zero-shot variant scoring and evaluate."""

    def __init__(self, model: GenomicModelBase, output_dir: str = "results"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def score_dataset(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Score all variants and return scores."""
        scores = []
        start_time = time.time()

        for i in tqdm(range(0, len(ref_seqs), batch_size), desc="Zero-shot scoring"):
            batch_ref = ref_seqs[i:i + batch_size]
            batch_alt = alt_seqs[i:i + batch_size]
            for r, a in zip(batch_ref, batch_alt):
                score = self.model.score_variant(r, a)
                scores.append(score)

        elapsed = time.time() - start_time
        print(f"  Scored {len(scores)} variants in {elapsed:.1f}s "
              f"({len(scores)/elapsed:.1f} variants/s)")

        return np.array(scores)

    def evaluate(
        self,
        df: pd.DataFrame,
        negative_set_name: str = "N1",
    ) -> dict:
        """
        Full zero-shot evaluation pipeline.

        Args:
            df: DataFrame with columns: ref_seq, alt_seq, label (0/1), region_category
            negative_set_name: Name of the negative set for logging
        """
        model_name = self.model.model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Zero-shot evaluation: {model_name} | Negative set: {negative_set_name}")
        print(f"{'='*60}")

        # Score all variants
        scores = self.score_dataset(
            df["ref_seq"].tolist(),
            df["alt_seq"].tolist(),
        )

        labels = df["label"].values

        # Overall metrics
        overall_metrics = compute_all_metrics(labels, scores)
        print(f"  Overall AUROC: {overall_metrics['auroc']:.4f}")
        print(f"  Overall AUPRC: {overall_metrics['auprc']:.4f}")
        print(f"  Overall MCC:   {overall_metrics['mcc']:.4f}")

        # Per-region metrics
        region_metrics = {}
        if "region_category" in df.columns:
            for region in df["region_category"].unique():
                mask = df["region_category"] == region
                if mask.sum() < 10:
                    continue
                region_labels = labels[mask]
                region_scores = scores[mask]
                if len(np.unique(region_labels)) < 2:
                    continue
                region_metrics[region] = compute_all_metrics(region_labels, region_scores)
                print(f"  {region:20s} AUROC: {region_metrics[region]['auroc']:.4f} "
                      f"(n={mask.sum()})")

        # Save results
        result = {
            "model": model_name,
            "negative_set": negative_set_name,
            "method": "zero_shot",
            "n_variants": len(df),
            "overall": overall_metrics,
            "per_region": region_metrics,
        }

        output_path = self.output_dir / f"zeroshot_{model_name}_{negative_set_name}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Results saved to {output_path}")

        # Also save scores for downstream analysis
        df_out = df.copy()
        df_out["score"] = scores
        scores_path = self.output_dir / f"scores_{model_name}_{negative_set_name}.parquet"
        df_out.to_parquet(scores_path, index=False)

        return result
