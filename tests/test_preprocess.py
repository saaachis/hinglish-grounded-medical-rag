"""
Tests for data preprocessing module.

All tests are parametrized as per project conventions.
"""

import pytest

from src.data.preprocess import categorize_cmi_level


class TestCategorizeCMI:
    """Tests for Code-Mixing Index categorization."""

    @pytest.mark.parametrize(
        "cmi_value, expected_level",
        [
            (0.0, "low"),
            (0.1, "low"),
            (0.32, "low"),
            (0.33, "medium"),
            (0.5, "medium"),
            (0.65, "medium"),
            (0.66, "high"),
            (0.8, "high"),
            (1.0, "high"),
        ],
        ids=[
            "cmi_0.0_low",
            "cmi_0.1_low",
            "cmi_0.32_low",
            "cmi_0.33_medium",
            "cmi_0.5_medium",
            "cmi_0.65_medium",
            "cmi_0.66_high",
            "cmi_0.8_high",
            "cmi_1.0_high",
        ],
    )
    def test_cmi_categorization(self, cmi_value, expected_level):
        """CMI values are correctly bucketed into low/medium/high."""
        assert categorize_cmi_level(cmi_value) == expected_level
