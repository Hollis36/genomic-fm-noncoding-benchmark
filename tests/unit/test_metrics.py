"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import (
    bootstrap_ci,
    compute_all_metrics,
    delong_test,
)


class TestComputeAllMetrics:
    """Test suite for compute_all_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_score)

        assert metrics["auroc"] == 1.0
        assert metrics["auprc"] == 1.0
        assert 0 <= metrics["mcc"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_score = np.random.rand(100)

        metrics = compute_all_metrics(y_true, y_score)

        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["auprc"] <= 1
        assert -1 <= metrics["mcc"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert "optimal_threshold" in metrics

    def test_single_class(self):
        """Test handling of single-class labels."""
        y_true = np.ones(10)
        y_score = np.random.rand(10)

        metrics = compute_all_metrics(y_true, y_score)

        assert np.isnan(metrics["auroc"])
        assert np.isnan(metrics["auprc"])

    def test_balanced_dataset(self):
        """Test with balanced positive/negative samples."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_score)

        assert metrics["auroc"] > 0.5
        assert all(key in metrics for key in ["auroc", "auprc", "mcc", "f1"])


class TestDeLongTest:
    """Test suite for DeLong test for AUROC comparison."""

    def test_identical_scores(self):
        """Test DeLong test with identical scores."""
        y_true = np.array([0, 0, 1, 1])
        y_score_a = np.array([0.1, 0.2, 0.8, 0.9])
        y_score_b = y_score_a.copy()

        result = delong_test(y_true, y_score_a, y_score_b)

        assert result["auc_a"] == result["auc_b"]
        assert abs(result["auc_diff"]) < 1e-10
        assert result["p_value"] > 0.05
        assert not result["significant_005"]

    def test_different_scores(self):
        """Test DeLong test with different scores."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_score_a = np.concatenate([np.random.rand(50) * 0.3, np.random.rand(50) * 0.3 + 0.7])
        y_score_b = np.concatenate([np.random.rand(50) * 0.4, np.random.rand(50) * 0.4 + 0.6])

        result = delong_test(y_true, y_score_a, y_score_b)

        assert "auc_a" in result
        assert "auc_b" in result
        assert "z_statistic" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_perfect_vs_random(self):
        """Test DeLong test comparing perfect vs random classifier."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        y_score_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        result = delong_test(y_true, y_score_perfect, y_score_random)

        assert result["auc_a"] > result["auc_b"]
        assert result["auc_diff"] > 0


class TestBootstrapCI:
    """Test suite for bootstrap confidence intervals."""

    def test_bootstrap_ci_shape(self):
        """Test bootstrap CI returns correct structure."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.random.rand(50) * 0.3, np.random.rand(50) * 0.3 + 0.7])

        result = bootstrap_ci(y_true, y_score, n_bootstrap=100, seed=42)

        assert "mean" in result
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["mean"] < result["ci_upper"]

    def test_bootstrap_reproducibility(self):
        """Test bootstrap CI is reproducible with same seed."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.random.rand(50) * 0.3, np.random.rand(50) * 0.3 + 0.7])

        result1 = bootstrap_ci(y_true, y_score, n_bootstrap=100, seed=42)
        result2 = bootstrap_ci(y_true, y_score, n_bootstrap=100, seed=42)

        assert result1["mean"] == result2["mean"]
        assert result1["ci_lower"] == result2["ci_lower"]
        assert result1["ci_upper"] == result2["ci_upper"]

    def test_bootstrap_with_custom_metric(self):
        """Test bootstrap CI with custom metric function."""
        from sklearn.metrics import average_precision_score

        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.random.rand(50) * 0.3, np.random.rand(50) * 0.3 + 0.7])

        result = bootstrap_ci(
            y_true, y_score,
            metric_fn=average_precision_score,
            n_bootstrap=100,
            seed=42
        )

        assert 0 <= result["mean"] <= 1
        assert result["ci_lower"] < result["ci_upper"]
