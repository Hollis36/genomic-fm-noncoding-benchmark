"""
Linear probe evaluation: train a logistic regression on frozen embeddings.

This evaluates the quality of learned representations independent of the
scoring method (cosine distance / log-likelihood ratio).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from ..models.base import GenomicModelBase
from .metrics import compute_all_metrics


class LinearProbeEvaluator:
    """Evaluate model embeddings with a linear probe (logistic regression)."""

    def __init__(
        self,
        model: GenomicModelBase,
        n_splits: int = 5,
        seed: int = 42,
        output_dir: str = "results",
    ):
        self.model = model
        self.n_splits = n_splits
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_embeddings(
        self,
        sequences: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Extract embeddings for all sequences."""
        embeddings = []
        start_time = time.time()

        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
            batch = sequences[i:i + batch_size]
            for seq in batch:
                emb = self.model.get_embedding(seq)
                embeddings.append(emb)

        elapsed = time.time() - start_time
        embeddings = np.array(embeddings)
        print(f"  Extracted {len(embeddings)} embeddings "
              f"(dim={embeddings.shape[1]}) in {elapsed:.1f}s")

        return embeddings

    def cross_validate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        regions: np.ndarray = None,
    ) -> dict:
        """Run stratified K-fold cross-validation with logistic regression."""
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.seed,
                solver="lbfgs",
            )
            clf.fit(embeddings[train_idx], labels[train_idx])
            probs = clf.predict_proba(embeddings[test_idx])[:, 1]

            fold_metrics = compute_all_metrics(labels[test_idx], probs)
            fold_metrics["fold"] = fold
            fold_results.append(fold_metrics)

        # Aggregate
        overall = {
            metric: {
                "mean": np.mean([f[metric] for f in fold_results]),
                "std": np.std([f[metric] for f in fold_results]),
            }
            for metric in ["auroc", "auprc", "mcc", "f1"]
        }

        # Per-region (using all data, train full model)
        region_metrics = {}
        if regions is not None:
            clf_full = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=self.seed
            )
            clf_full.fit(embeddings, labels)
            probs_full = clf_full.predict_proba(embeddings)[:, 1]

            for region in np.unique(regions):
                mask = regions == region
                if mask.sum() < 10:
                    continue
                region_labels = labels[mask]
                if len(np.unique(region_labels)) < 2:
                    continue
                region_metrics[region] = compute_all_metrics(
                    region_labels, probs_full[mask]
                )

        return {"overall": overall, "per_region": region_metrics, "folds": fold_results}

    def evaluate(
        self,
        df: pd.DataFrame,
        negative_set_name: str = "N1",
        use_ref_seq: bool = True,
    ) -> dict:
        """
        Full linear probe evaluation pipeline.

        Args:
            df: DataFrame with ref_seq, alt_seq, label, region_category
            negative_set_name: Identifier for the negative set
            use_ref_seq: If True, use ref_seq for embedding; if False, use alt_seq
        """
        model_name = self.model.model_name.split("/")[-1]
        seq_type = "ref" if use_ref_seq else "alt"
        print(f"\n{'='*60}")
        print(f"Linear probe: {model_name} | {negative_set_name} | seq={seq_type}")
        print(f"{'='*60}")

        sequences = df["ref_seq" if use_ref_seq else "alt_seq"].tolist()
        embeddings = self.extract_embeddings(sequences)
        labels = df["label"].values
        regions = df["region_category"].values if "region_category" in df.columns else None

        cv_results = self.cross_validate(embeddings, labels, regions)

        print(f"  AUROC: {cv_results['overall']['auroc']['mean']:.4f} "
              f"± {cv_results['overall']['auroc']['std']:.4f}")
        print(f"  AUPRC: {cv_results['overall']['auprc']['mean']:.4f} "
              f"± {cv_results['overall']['auprc']['std']:.4f}")

        # Save
        result = {
            "model": model_name,
            "negative_set": negative_set_name,
            "method": "linear_probe",
            "seq_type": seq_type,
            "n_variants": len(df),
            **cv_results,
        }

        output_path = (
            self.output_dir / f"linprobe_{model_name}_{negative_set_name}_{seq_type}.json"
        )
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Results saved to {output_path}")

        return result
