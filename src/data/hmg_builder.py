"""
HMG Dataset Builder — Synthetic-to-Real Hinglish Pipeline.

Constructs the primary HMG (Hinglish-Medical-Grounding) dataset of
~4,000 triplets: {Open-i X-ray, English Report, Synthetic Hinglish Query}.

Uses Llama-3-8B-Instruct to transform formal English findings from Open-i
into informal Hinglish "Doctor-Patient" queries, using MMCQSD as a
linguistic template to ensure realistic code-switching patterns.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_hinglish_prompt(
    english_report: str,
    mmcqsd_examples: list[str],
    num_examples: int = 3,
) -> str:
    """Build a few-shot prompt for Hinglish query synthesis.

    Uses MMCQSD examples as linguistic templates to guide the LLM
    in producing realistic code-switched medical queries.

    Parameters
    ----------
    english_report : str
        Formal English radiology report to convert.
    mmcqsd_examples : list[str]
        Example Hinglish queries from MMCQSD for few-shot prompting.
    num_examples : int
        Number of MMCQSD examples to include in the prompt.

    Returns
    -------
    str
        Formatted prompt for the Hinglish synthesizer.
    """
    # TODO: Implement prompt construction
    raise NotImplementedError


def generate_hinglish_query(
    english_report: str,
    model: Any,
    tokenizer: Any,
    mmcqsd_examples: list[str] | None = None,
) -> str:
    """Generate a synthetic Hinglish query from an English report.

    Parameters
    ----------
    english_report : str
        Formal English radiology report.
    model : Any
        Loaded Llama-3-8B-Instruct model.
    tokenizer : Any
        Corresponding tokenizer.
    mmcqsd_examples : list[str] | None
        Optional MMCQSD examples for few-shot prompting.

    Returns
    -------
    str
        Generated Hinglish query.
    """
    # TODO: Implement Hinglish query generation
    raise NotImplementedError


def build_hmg_dataset(
    openi_reports_path: str,
    mmcqsd_path: str,
    output_path: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
) -> None:
    """Build the full HMG dataset.

    Generates ~4,000 triplets by transforming Open-i English reports
    into Hinglish queries using Llama-3-8B with MMCQSD-guided prompting.

    Parameters
    ----------
    openi_reports_path : str
        Path to preprocessed Open-i reports.
    mmcqsd_path : str
        Path to MMCQSD dataset (linguistic templates).
    output_path : str
        Path to save the HMG dataset.
    model_name : str
        HuggingFace model ID for the Hinglish synthesizer.
    """
    # TODO: Implement full HMG dataset construction pipeline
    raise NotImplementedError
