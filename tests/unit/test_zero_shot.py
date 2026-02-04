"""Unit tests for zero-shot evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.zero_shot import ZeroShotEvaluator


class TestZeroShotEvaluator:
    """Test suite for ZeroShotEvaluator class."""

    @pytest.fixture
    def mock_model(self, mock_model_config):
        """Create a mock model for testing."""
        model = MagicMock()
        model.model_name = "test-model/checkpoint"
        model.score_variant = MagicMock(side_effect=lambda r, a: np.random.rand())
        return model

    @pytest.fixture
    def evaluator(self, mock_model, tmp_path):
        """Create a ZeroShotEvaluator instance."""
        return ZeroShotEvaluator(mock_model, output_dir=str(tmp_path))

    def test_initialization(self, mock_model, tmp_path):
        """Test evaluator initialization."""
        evaluator = ZeroShotEvaluator(mock_model, output_dir=str(tmp_path))

        assert evaluator.model == mock_model
        assert evaluator.output_dir == Path(tmp_path)
        assert evaluator.output_dir.exists()

    def test_score_dataset(self, evaluator, mock_model):
        """Test dataset scoring."""
        ref_seqs = ["ATCG" * 50] * 10
        alt_seqs = ["TTCG" * 50] * 10

        scores = evaluator.score_dataset(ref_seqs, alt_seqs, batch_size=3)

        assert len(scores) == 10
        assert all(isinstance(s, (float, np.floating)) for s in scores)
        assert mock_model.score_variant.call_count == 10

    def test_evaluate_saves_results(self, evaluator, sample_variant_df, tmp_path):
        """Test that evaluation saves results correctly."""
        result = evaluator.evaluate(sample_variant_df, negative_set_name="N1")

        # Check result structure
        assert "model" in result
        assert "negative_set" in result
        assert "method" in result
        assert result["method"] == "zero_shot"
        assert "overall" in result
        assert "per_region" in result

        # Check files were created
        json_files = list(Path(tmp_path).glob("*.json"))
        parquet_files = list(Path(tmp_path).glob("*.parquet"))

        assert len(json_files) >= 1
        assert len(parquet_files) >= 1

    def test_evaluate_per_region_metrics(self, evaluator, sample_variant_df):
        """Test per-region metrics calculation."""
        result = evaluator.evaluate(sample_variant_df, negative_set_name="N3")

        # Should have region-specific metrics
        assert "per_region" in result
        assert len(result["per_region"]) > 0

        # Each region should have metrics
        for region, metrics in result["per_region"].items():
            assert "auroc" in metrics
            assert "auprc" in metrics

    def test_evaluate_with_missing_region_column(self, evaluator):
        """Test evaluation when region_category column is missing."""
        import pandas as pd

        df = pd.DataFrame({
            "ref_seq": ["ATCG" * 50] * 10,
            "alt_seq": ["TTCG" * 50] * 10,
            "label": [0, 1] * 5,
        })

        result = evaluator.evaluate(df, negative_set_name="N1")

        # Should still work, just without per-region metrics
        assert "overall" in result
        assert len(result.get("per_region", {})) == 0
