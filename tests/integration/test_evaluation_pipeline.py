"""Integration tests for zero-shot evaluation pipeline end-to-end."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.zero_shot import ZeroShotEvaluator


def _make_mock_model(deterministic: bool = True) -> MagicMock:
    """Build a mock model that returns controlled variant scores.

    Args:
        deterministic: If True, pathogenic-looking sequences score higher.

    Returns:
        MagicMock with model_name and score_variant configured.
    """
    model = MagicMock()
    model.model_name = "mock-model/v1"

    if deterministic:
        counter = {"idx": 0}

        def _score(ref_seq: str, alt_seq: str) -> float:
            idx = counter["idx"]
            counter["idx"] += 1
            return float(idx)

        model.score_variant = MagicMock(side_effect=_score)
    else:
        rng = np.random.RandomState(42)
        model.score_variant = MagicMock(side_effect=lambda r, a: float(rng.rand()))

    return model


def _make_variant_df(n_positive: int = 25, n_negative: int = 25) -> pd.DataFrame:
    """Create a balanced variant DataFrame for testing.

    Args:
        n_positive: Number of pathogenic variants.
        n_negative: Number of benign variants.

    Returns:
        DataFrame with ref_seq, alt_seq, label, region_category columns.
    """
    n_total = n_positive + n_negative
    regions = ["promoter", "enhancer", "intergenic", "utr_5prime", "splice_proximal"]
    rng = np.random.RandomState(99)
    return pd.DataFrame({
        "ref_seq": ["ATCG" * 50] * n_total,
        "alt_seq": ["TTCG" * 50] * n_total,
        "label": [1] * n_positive + [0] * n_negative,
        "region_category": [regions[i % len(regions)] for i in range(n_total)],
        "chrom": [f"chr{rng.randint(1, 23)}" for _ in range(n_total)],
        "pos": rng.randint(1000, 100000, size=n_total).tolist(),
    })


@pytest.mark.integration
class TestZeroShotPipelineIntegration:
    """End-to-end tests for the zero-shot evaluation pipeline."""

    def test_full_pipeline_produces_results(self, tmp_path: Path):
        """Run the full zero-shot pipeline and verify output structure."""
        model = _make_mock_model(deterministic=False)
        evaluator = ZeroShotEvaluator(model, output_dir=str(tmp_path))
        df = _make_variant_df()

        result = evaluator.evaluate(df, negative_set_name="N_test")

        assert result["method"] == "zero_shot"
        assert result["model"] == "v1"
        assert result["negative_set"] == "N_test"
        assert result["n_variants"] == 50
        assert "overall" in result
        assert "per_region" in result

    def test_results_saved_to_disk(self, tmp_path: Path):
        """Verify JSON and Parquet files are written correctly."""
        model = _make_mock_model(deterministic=False)
        evaluator = ZeroShotEvaluator(model, output_dir=str(tmp_path))
        df = _make_variant_df()

        evaluator.evaluate(df, negative_set_name="N_disk")

        json_files = list(tmp_path.glob("zeroshot_*.json"))
        parquet_files = list(tmp_path.glob("scores_*.parquet"))

        assert len(json_files) == 1
        assert len(parquet_files) == 1

        with open(json_files[0]) as f:
            saved = json.load(f)
        assert saved["method"] == "zero_shot"
        assert "overall" in saved

        scores_df = pd.read_parquet(parquet_files[0])
        assert "score" in scores_df.columns
        assert len(scores_df) == 50

    def test_per_region_metrics_computed(self, tmp_path: Path):
        """Verify per-region metrics when region_category is present."""
        model = _make_mock_model(deterministic=False)
        evaluator = ZeroShotEvaluator(model, output_dir=str(tmp_path))
        df = _make_variant_df(n_positive=50, n_negative=50)

        result = evaluator.evaluate(df, negative_set_name="N_region")

        assert len(result["per_region"]) > 0
        for region, metrics in result["per_region"].items():
            assert "auroc" in metrics
            assert "auprc" in metrics

    def test_pipeline_without_region_column(self, tmp_path: Path):
        """Pipeline should work when region_category column is absent."""
        model = _make_mock_model(deterministic=False)
        evaluator = ZeroShotEvaluator(model, output_dir=str(tmp_path))
        df = pd.DataFrame({
            "ref_seq": ["ATCG" * 50] * 20,
            "alt_seq": ["TTCG" * 50] * 20,
            "label": [1] * 10 + [0] * 10,
        })

        result = evaluator.evaluate(df, negative_set_name="N_noreg")

        assert "overall" in result
        assert len(result.get("per_region", {})) == 0

    def test_metrics_consistency_with_direct_call(self, tmp_path: Path):
        """Metrics from the pipeline should match a direct compute_all_metrics call."""
        rng = np.random.RandomState(123)
        n = 40
        labels = np.array([1] * 20 + [0] * 20)
        scores_raw = rng.rand(n)

        model = MagicMock()
        model.model_name = "consistency-test/v1"
        score_iter = iter(scores_raw.tolist())
        model.score_variant = MagicMock(side_effect=lambda r, a: next(score_iter))

        evaluator = ZeroShotEvaluator(model, output_dir=str(tmp_path))
        df = pd.DataFrame({
            "ref_seq": ["ATCG" * 50] * n,
            "alt_seq": ["TTCG" * 50] * n,
            "label": labels.tolist(),
        })

        result = evaluator.evaluate(df, negative_set_name="N_consist")
        expected = compute_all_metrics(labels, scores_raw)

        assert abs(result["overall"]["auroc"] - expected["auroc"]) < 1e-6
        assert abs(result["overall"]["auprc"] - expected["auprc"]) < 1e-6
