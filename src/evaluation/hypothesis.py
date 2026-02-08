"""
Statistical Hypothesis Testing.

Implements the statistical testing plan for the three hypotheses:

  H1 (Grounding Effect):
      Paired t-test / Wilcoxon — RAG vs Zero-shot LLM

  H2 (Code-Mixing Robustness):
      Two-way ANOVA / Kruskal-Wallis — CMI level interaction

  H3 (Evidence Type):
      Paired t-test / Wilcoxon — Authoritative vs general evidence
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def check_normality(data: np.ndarray, alpha: float = 0.05) -> bool:
    """Check if data follows a normal distribution (Shapiro-Wilk test).

    Parameters
    ----------
    data : np.ndarray
        Sample data.
    alpha : float
        Significance level.

    Returns
    -------
    bool
        True if data appears normally distributed.
    """
    if len(data) < 3:
        return False
    stat, p_value = stats.shapiro(data)
    return p_value > alpha


def test_h1_grounding_effect(
    rag_scores: np.ndarray,
    zeroshot_scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Test H1: RAG vs Zero-shot LLM (Grounding Effect).

    H0: No significant difference in factual consistency or
    hallucination rate between zero-shot LLMs and evidence-grounded
    RAG systems.

    Parameters
    ----------
    rag_scores : np.ndarray
        Per-sample scores from the RAG system.
    zeroshot_scores : np.ndarray
        Per-sample scores from the zero-shot LLM.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test results including statistic, p-value, effect size, and CI.
    """
    differences = rag_scores - zeroshot_scores
    is_normal = check_normality(differences, alpha)

    if is_normal:
        stat, p_value = stats.ttest_rel(rag_scores, zeroshot_scores)
        test_name = "Paired t-test"
    else:
        stat, p_value = stats.wilcoxon(differences)
        test_name = "Wilcoxon signed-rank test"

    # Cohen's d effect size
    effect_size = np.mean(differences) / np.std(differences, ddof=1)

    # 95% confidence interval for the mean difference
    ci = stats.t.interval(
        1 - alpha,
        df=len(differences) - 1,
        loc=np.mean(differences),
        scale=stats.sem(differences),
    )

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "effect_size_cohens_d": effect_size,
        "mean_difference": np.mean(differences),
        "confidence_interval": ci,
        "reject_null": p_value < alpha,
    }


def test_h2_code_mixing_robustness(
    scores_by_cmi: dict[str, dict[str, np.ndarray]],
    alpha: float = 0.05,
) -> dict:
    """Test H2: Code-Mixing Robustness.

    H0: Increasing the level of code-mixing does not significantly
    affect the performance difference between grounded and
    non-grounded models.

    Parameters
    ----------
    scores_by_cmi : dict
        Nested dict: {cmi_level: {'rag': scores, 'zeroshot': scores}}.
        cmi_level is one of 'low', 'medium', 'high'.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test results including statistic, p-value, and post-hoc analysis.
    """
    # Compute performance differences per CMI level
    differences_by_level = {}
    for level in ("low", "medium", "high"):
        if level in scores_by_cmi:
            differences_by_level[level] = (
                scores_by_cmi[level]["rag"] - scores_by_cmi[level]["zeroshot"]
            )

    groups = list(differences_by_level.values())

    # Check normality for all groups
    all_normal = all(check_normality(g, alpha) for g in groups)

    if all_normal:
        stat, p_value = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"
    else:
        stat, p_value = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis H-test"

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "reject_null": p_value < alpha,
        "group_means": {
            level: float(np.mean(diff))
            for level, diff in differences_by_level.items()
        },
    }


def test_h3_evidence_type(
    authoritative_scores: np.ndarray,
    general_scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Test H3: Authoritative vs General Evidence.

    H0: Using authoritative medical text evidence does not significantly
    improve factual correctness compared to general biomedical text.

    Parameters
    ----------
    authoritative_scores : np.ndarray
        Per-sample scores using authoritative medical evidence.
    general_scores : np.ndarray
        Per-sample scores using general biomedical text.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test results including statistic, p-value, effect size, and CI.
    """
    differences = authoritative_scores - general_scores
    is_normal = check_normality(differences, alpha)

    if is_normal:
        stat, p_value = stats.ttest_rel(authoritative_scores, general_scores)
        test_name = "Paired t-test"
    else:
        stat, p_value = stats.wilcoxon(differences)
        test_name = "Wilcoxon signed-rank test"

    effect_size = np.mean(differences) / np.std(differences, ddof=1)

    ci = stats.t.interval(
        1 - alpha,
        df=len(differences) - 1,
        loc=np.mean(differences),
        scale=stats.sem(differences),
    )

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "effect_size_cohens_d": effect_size,
        "mean_difference": np.mean(differences),
        "confidence_interval": ci,
        "reject_null": p_value < alpha,
    }
