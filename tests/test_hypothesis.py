"""
Tests for statistical hypothesis testing module.

All tests are parametrized as per project conventions.
"""

import numpy as np
import pytest

from src.evaluation.hypothesis import (
    check_normality,
    test_h1_grounding_effect,
    test_h3_evidence_type,
)


class TestCheckNormality:
    """Tests for the normality checking utility."""

    @pytest.mark.parametrize(
        "data, expected",
        [
            (np.random.default_rng(42).normal(0, 1, 100), True),
            (np.random.default_rng(42).exponential(1, 100), False),
            (np.array([1.0, 2.0]), False),  # too few samples
        ],
        ids=["normal_data", "non_normal_data", "too_few_samples"],
    )
    def test_check_normality(self, data, expected):
        """Normality check returns expected result for different distributions."""
        result = check_normality(data)
        assert result == expected


class TestH1GroundingEffect:
    """Tests for H1: RAG vs Zero-shot grounding effect."""

    @pytest.mark.parametrize(
        "rag_mean, zeroshot_mean, expect_reject",
        [
            (0.8, 0.4, True),    # clear difference → reject null
            (0.5, 0.5, False),   # no difference → fail to reject
        ],
        ids=["significant_difference", "no_difference"],
    )
    def test_h1_outcomes(self, rag_mean, zeroshot_mean, expect_reject):
        """H1 test correctly identifies significant vs non-significant differences."""
        rng = np.random.default_rng(42)
        rag_scores = rng.normal(rag_mean, 0.05, 50)
        zeroshot_scores = rng.normal(zeroshot_mean, 0.05, 50)

        result = test_h1_grounding_effect(rag_scores, zeroshot_scores)

        assert "p_value" in result
        assert "effect_size_cohens_d" in result
        assert "confidence_interval" in result
        assert result["reject_null"] == expect_reject


class TestH3EvidenceType:
    """Tests for H3: Authoritative vs general evidence."""

    @pytest.mark.parametrize(
        "auth_mean, gen_mean, expect_reject",
        [
            (0.85, 0.55, True),   # authoritative better → reject null
            (0.6, 0.6, False),    # no difference → fail to reject
        ],
        ids=["authoritative_better", "no_difference"],
    )
    def test_h3_outcomes(self, auth_mean, gen_mean, expect_reject):
        """H3 test correctly identifies effect of evidence type."""
        rng = np.random.default_rng(42)
        auth_scores = rng.normal(auth_mean, 0.05, 50)
        gen_scores = rng.normal(gen_mean, 0.05, 50)

        result = test_h3_evidence_type(auth_scores, gen_scores)

        assert "p_value" in result
        assert "effect_size_cohens_d" in result
        assert result["reject_null"] == expect_reject
