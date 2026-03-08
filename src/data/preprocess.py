"""
Data preprocessing utilities.

Handles cleaning, normalization, and preparation of clinical reports,
Hinglish queries, and medical images for the RAG pipeline.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def clean_radiology_report(report_text: str) -> str:
    """Clean and normalize a radiology report.

    Removes artifacts, standardizes whitespace, and prepares
    the report for embedding and retrieval.

    Parameters
    ----------
    report_text : str
        Raw radiology report text.

    Returns
    -------
    str
        Cleaned report text.
    """
    # TODO: Implement report cleaning
    raise NotImplementedError


def compute_code_mixing_index(text: str) -> float:
    """Compute the Code-Mixing Index (CMI) for a Hinglish text.

    CMI measures the degree of code-switching between Hindi and English.
    Used to bucket queries into low / medium / high code-mixing levels
    for hypothesis testing (H2).

    Parameters
    ----------
    text : str
        Hinglish text to analyze.

    Returns
    -------
    float
        Code-Mixing Index value in [0, 1].
    """
    # TODO: Implement CMI computation
    raise NotImplementedError


def categorize_cmi_level(cmi: float) -> str:
    """Categorize a CMI score into low / medium / high.

    Parameters
    ----------
    cmi : float
        Code-Mixing Index value.

    Returns
    -------
    str
        One of 'low', 'medium', or 'high'.
    """
    if cmi < 0.33:
        return "low"
    elif cmi < 0.66:
        return "medium"
    else:
        return "high"


def preprocess_openi_reports(raw_dir: str, output_dir: str) -> pd.DataFrame:
    """Preprocess Open-i radiology reports.

    Parameters
    ----------
    raw_dir : str
        Path to raw Open-i data.
    output_dir : str
        Path to save processed outputs.

    Returns
    -------
    pd.DataFrame
        Processed report data.
    """
    # TODO: Implement Open-i preprocessing
    raise NotImplementedError


def preprocess_mmcqsd(raw_dir: str, output_dir: str) -> pd.DataFrame:
    """Preprocess MMCQSD Hinglish medical queries.

    Parameters
    ----------
    raw_dir : str
        Path to raw MMCQSD data.
    output_dir : str
        Path to save processed outputs.

    Returns
    -------
    pd.DataFrame
        Processed MMCQSD data with CMI scores.
    """
    # TODO: Implement MMCQSD preprocessing
    raise NotImplementedError
