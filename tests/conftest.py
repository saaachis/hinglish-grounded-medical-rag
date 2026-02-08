"""
Shared test fixtures and configuration.

All tests use parametrized patterns by default.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_config():
    """Provide a minimal test configuration."""
    return {
        "project": {"name": "hinglish-grounded-medical-rag", "version": "0.1.0"},
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "hmg_dir": "data/hmg",
        },
        "retrieval": {
            "top_k": 5,
            "adaptive_truncation": True,
        },
        "evaluation": {
            "metrics": ["mmfcm", "factual_consistency", "hallucination_rate"],
        },
    }


@pytest.fixture
def sample_hinglish_queries():
    """Provide sample Hinglish queries for testing."""
    return [
        "Doctor, right side chhati me pani bhar gaya hai kya?",
        "Mujhe saas lene me bahut problem ho rahi hai",
        "Left lung me kuch dikhai de raha hai kya X-ray me?",
        "Chest me dard hai aur cough bhi aa raha hai",
        "Doctor, kya ye pneumonia hai?",
    ]


@pytest.fixture
def sample_english_reports():
    """Provide sample English radiology reports for testing."""
    return [
        "Right-sided pleural effusion noted. Heart size is normal.",
        "Bilateral infiltrates consistent with pneumonia.",
        "Left lower lobe opacity suggesting consolidation.",
        "No acute cardiopulmonary abnormality detected.",
        "Mild cardiomegaly with pulmonary vascular congestion.",
    ]


@pytest.fixture
def sample_scores():
    """Provide sample evaluation scores for hypothesis testing."""
    rng = np.random.default_rng(42)
    return {
        "rag_scores": rng.uniform(0.6, 0.95, size=50),
        "zeroshot_scores": rng.uniform(0.3, 0.75, size=50),
    }
