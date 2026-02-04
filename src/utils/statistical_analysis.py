"""Statistical analysis utilities for model comparison."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .logging_config import get_logger

logger = get_logger(__name__)


def perform_paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Perform paired t-test to compare two sets of scores.

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length")

    statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_diff": float(np.mean(scores_a - scores_b)),
        "std_diff": float(np.std(scores_a - scores_b)),
    }


def perform_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length")

    statistic, p_value = stats.wilcoxon(scores_a, scores_b)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "median_diff": float(np.median(scores_a - scores_b)),
    }


def calculate_effect_size(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """
    Calculate Cohen's d effect size for comparing two models.

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B

    Returns:
        Dictionary with effect size metrics
    """
    diff = scores_a - scores_b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    # Interpretation
    if abs(cohens_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        interpretation = "small"
    elif abs(cohens_d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "cohens_d": float(cohens_d),
        "interpretation": interpretation,
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
    }


def bootstrap_auroc_comparison(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Bootstrap comparison of AUROC between two models.

    Args:
        y_true: True labels
        scores_a: Predictions from model A
        scores_b: Predictions from model B
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed

    Returns:
        Dictionary with comparison results
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    auroc_a_samples = []
    auroc_b_samples = []
    auroc_diff_samples = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)

        if len(np.unique(y_true[idx])) < 2:
            continue

        auroc_a = roc_auc_score(y_true[idx], scores_a[idx])
        auroc_b = roc_auc_score(y_true[idx], scores_b[idx])

        auroc_a_samples.append(auroc_a)
        auroc_b_samples.append(auroc_b)
        auroc_diff_samples.append(auroc_a - auroc_b)

    auroc_diff_samples = np.array(auroc_diff_samples)

    ci_lower = np.percentile(auroc_diff_samples, alpha / 2 * 100)
    ci_upper = np.percentile(auroc_diff_samples, (1 - alpha / 2) * 100)

    # Significant if CI doesn't include 0
    significant = not (ci_lower <= 0 <= ci_upper)

    return {
        "auroc_a_mean": float(np.mean(auroc_a_samples)),
        "auroc_b_mean": float(np.mean(auroc_b_samples)),
        "auroc_diff_mean": float(np.mean(auroc_diff_samples)),
        "auroc_diff_std": float(np.std(auroc_diff_samples)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": significant,
        "p_value": float(np.mean(auroc_diff_samples <= 0) * 2),  # Two-tailed
    }


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict:
    """
    McNemar's test for comparing binary classifiers.

    Args:
        y_true: True binary labels
        pred_a: Binary predictions from model A
        pred_b: Binary predictions from model B

    Returns:
        Dictionary with test results
    """
    # Create contingency table
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # Count discordant pairs
    n_01 = np.sum(~correct_a & correct_b)  # B correct, A wrong
    n_10 = np.sum(correct_a & ~correct_b)  # A correct, B wrong

    # McNemar's test statistic
    if n_01 + n_10 == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_01": int(n_01),
            "n_10": int(n_10),
        }

    statistic = ((abs(n_01 - n_10) - 1) ** 2) / (n_01 + n_10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_01": int(n_01),
        "n_10": int(n_10),
    }


def create_comparison_report(
    y_true: np.ndarray,
    scores_dict: dict,
    output_path: str = "comparison_report.txt",
) -> pd.DataFrame:
    """
    Create a comprehensive comparison report for multiple models.

    Args:
        y_true: True labels
        scores_dict: Dictionary of model_name -> scores
        output_path: Path to save report

    Returns:
        DataFrame with comparison results
    """
    results = []

    model_names = list(scores_dict.keys())

    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1 :]:
            scores_a = scores_dict[model_a]
            scores_b = scores_dict[model_b]

            # AUROC comparison
            auroc_a = roc_auc_score(y_true, scores_a)
            auroc_b = roc_auc_score(y_true, scores_b)

            # Statistical tests
            bootstrap = bootstrap_auroc_comparison(y_true, scores_a, scores_b)
            effect = calculate_effect_size(scores_a, scores_b)

            results.append({
                "model_a": model_a,
                "model_b": model_b,
                "auroc_a": auroc_a,
                "auroc_b": auroc_b,
                "auroc_diff": auroc_a - auroc_b,
                "auroc_diff_ci_lower": bootstrap["ci_lower"],
                "auroc_diff_ci_upper": bootstrap["ci_upper"],
                "significant": bootstrap["significant"],
                "p_value": bootstrap["p_value"],
                "cohens_d": effect["cohens_d"],
                "effect_size": effect["interpretation"],
            })

    df = pd.DataFrame(results)

    # Save report
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")

    logger.info(f"Comparison report saved to {output_path}")

    return df
