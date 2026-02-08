"""
Dataset download utilities.

Handles downloading and organizing secondary datasets:
  - Open-i (Indiana University Chest X-Ray Collection)
  - MMCQSD (Multimodal Medical Code-Mixed Question Summarization Dataset)
  - PubMedQA
  - MMed-Bench
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_data_directories(config: dict) -> None:
    """Create required data directories if they do not exist."""
    for key in ("raw_dir", "processed_dir", "hmg_dir"):
        path = Path(config["data"][key])
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s", path)


def download_openi(target_dir: str) -> None:
    """Download Open-i Indiana University Chest X-Ray Collection.

    Source : https://openi.nlm.nih.gov/
    License: CC-0 (Public Domain)

    Parameters
    ----------
    target_dir : str
        Directory to save the downloaded dataset.
    """
    # TODO: Implement Open-i download pipeline
    raise NotImplementedError("Open-i download not yet implemented.")


def download_mmcqsd(target_dir: str) -> None:
    """Download MMCQSD dataset.

    Multimodal Medical Code-Mixed Question Summarization Dataset
    containing 3,015 Hinglish medical query samples.

    Parameters
    ----------
    target_dir : str
        Directory to save the downloaded dataset.
    """
    # TODO: Implement MMCQSD download pipeline
    raise NotImplementedError("MMCQSD download not yet implemented.")


def download_pubmedqa(target_dir: str) -> None:
    """Download PubMedQA dataset.

    Large-scale biomedical QA dataset with answers derived
    from PubMed abstracts.

    Parameters
    ----------
    target_dir : str
        Directory to save the downloaded dataset.
    """
    # TODO: Implement PubMedQA download pipeline
    raise NotImplementedError("PubMedQA download not yet implemented.")


def download_mmed_bench(target_dir: str) -> None:
    """Download MMed-Bench dataset.

    Multimodal medical QA benchmark with 25,500+ samples
    across specialties.

    Parameters
    ----------
    target_dir : str
        Directory to save the downloaded dataset.
    """
    # TODO: Implement MMed-Bench download pipeline
    raise NotImplementedError("MMed-Bench download not yet implemented.")


def download_all_datasets(config: dict | None = None) -> None:
    """Download all secondary datasets specified in the configuration."""
    if config is None:
        config = load_config()

    setup_data_directories(config)
    raw_dir = config["data"]["raw_dir"]

    logger.info("Starting dataset downloads to: %s", raw_dir)

    download_functions = {
        "openi": download_openi,
        "mmcqsd": download_mmcqsd,
        "pubmedqa": download_pubmedqa,
        "mmed_bench": download_mmed_bench,
    }

    for name, download_fn in download_functions.items():
        try:
            logger.info("Downloading: %s", name)
            download_fn(f"{raw_dir}/{name}")
        except NotImplementedError:
            logger.warning("Download for '%s' not yet implemented — skipping.", name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all_datasets()
