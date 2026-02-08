"""
Shared helper utilities.

Common functions used across modules: logging setup, device detection,
reproducibility seeding, and configuration loading.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the project.

    Parameters
    ----------
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    log_file : str | None
        Optional file path to write logs to.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_device() -> str:
    """Detect the best available compute device.

    Returns
    -------
    str
        'cuda' if GPU is available, else 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logging.info("Using GPU: %s", gpu_name)
    else:
        device = "cpu"
        logging.info("No GPU detected — using CPU.")
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Random seed set to %d", seed)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded from: %s", config_path)
    return config
