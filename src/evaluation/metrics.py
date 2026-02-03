"""
Evaluation metrics for variant pathogenicity prediction.

Includes: AUROC, AUPRC, MCC, F1, DeLong test for AUROC comparison.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Binary labels (0/1)
        y_score: Predicted scores/probabilities

    Returns:
        Dictionary with auroc, auprc, mcc, f1
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "mcc": np.nan, "f1": np.nan}

    # AUROC
    auroc = roc_auc_score(y_true, y_score)

    # AUPRC
    auprc = average_precision_score(y_true, y_score)

    # Find optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Binary predictions at optimal threshold
    y_pred = (y_score >= optimal_threshold).astype(int)

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # F1
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "mcc": float(mcc),
        "f1": float(f1),
        "optimal_threshold": float(optimal_threshold),
    }


def _auc_variance(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """
    Compute AUC and its variance using the method of DeLong et al. (1988).

    Returns: (auc, variance)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    m = len(pos_scores)
    n = len(neg_scores)

    auc = roc_auc_score(y_true, y_score)

    # Structural components V10 and V01
    v10 = np.array([np.mean(pos_scores[i] > neg_scores) for i in range(m)])
    v01 = np.array([np.mean(pos_scores > neg_scores[j]) for j in range(n)])

    s10 = np.var(v10, ddof=1) if m > 1 else 0
    s01 = np.var(v01, ddof=1) if n > 1 else 0

    variance = s10 / m + s01 / n

    return auc, variance


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict:
    """
    DeLong test for comparing two AUROC values.

    Tests H0: AUC_A = AUC_B

    Args:
        y_true: Binary labels
        y_score_a: Scores from model A
        y_score_b: Scores from model B

    Returns:
        Dictionary with auc_a, auc_b, z_statistic, p_value
    """
    auc_a, var_a = _auc_variance(y_true, y_score_a)
    auc_b, var_b = _auc_variance(y_true, y_score_b)

    # Covariance (simplified — assumes independent for now)
    # For the full covariance, see DeLong et al. 1988
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b + 1e-10)
    p_value = 2 * stats.norm.sf(abs(z))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(auc_a - auc_b),
        "z_statistic": float(z),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))

    scores = np.array(scores)
    alpha = (1 - ci) / 2

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": float(np.percentile(scores, alpha * 100)),
        "ci_upper": float(np.percentile(scores, (1 - alpha) * 100)),
        "n_bootstrap": n_bootstrap,
    }
