"""Build the FAISS index over the MultiCaRe evidence corpus.

Run once to create persistent index + metadata files that the
RAG pipeline and Streamlit app load at startup.

Usage:
    python build_index.py [--max-cases 10000] [--max-words 200]
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MULTICARE_PATH = Path("data/processed/multicare_filtered.csv")
INDEX_DIR = Path("data/faiss_index")
INDEX_PATH = INDEX_DIR / "evidence.index"
METADATA_PATH = INDEX_DIR / "evidence_metadata.csv"
EMBEDDINGS_PATH = INDEX_DIR / "evidence_embeddings.npy"

SAMPLE_PER_CONDITION = 600
MAX_WORDS = 200


def truncate(text: str, max_words: int) -> str:
    words = str(text).split()
    return " ".join(words[:max_words]) if len(words) > max_words else str(text)


def sample_evidence(df: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    """Sample evidence balanced across condition groups."""
    if len(df) <= max_cases:
        return df

    per_cond = max_cases // df["condition_group"].nunique()
    per_cond = max(per_cond, 50)

    sampled = (
        df.groupby("condition_group", group_keys=False)
        .apply(
            lambda g: g.sample(n=min(len(g), per_cond), random_state=42),
            include_groups=False,
        )
    )

    if len(sampled) < max_cases:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(
            n=min(len(remaining), max_cases - len(sampled)), random_state=42
        )
        sampled = pd.concat([sampled, extra])

    return sampled.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS evidence index")
    parser.add_argument(
        "--max-cases", type=int, default=10000,
        help="Maximum number of evidence cases to index",
    )
    parser.add_argument(
        "--max-words", type=int, default=MAX_WORDS,
        help="Truncate case_text to this many words before encoding",
    )
    args = parser.parse_args()

    logger.info("Loading MultiCaRe evidence from %s ...", MULTICARE_PATH)
    df = pd.read_csv(MULTICARE_PATH)
    logger.info("Total cases: %d", len(df))

    df = sample_evidence(df, args.max_cases)
    logger.info("Sampled to %d cases across %d conditions",
                len(df), df["condition_group"].nunique())

    texts = [truncate(t, args.max_words) for t in df["case_text"].tolist()]

    if EMBEDDINGS_PATH.exists():
        logger.info("Loading cached embeddings from %s", EMBEDDINGS_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.shape[0] != len(texts):
            logger.warning("Cached embedding count mismatch, re-encoding")
            embeddings = None
        else:
            logger.info("Using cached embeddings: shape %s", embeddings.shape)
    else:
        embeddings = None

    if embeddings is None:
        from src.encoding.text_encoder import TextEncoder
        encoder = TextEncoder(device="cpu")
        encoder.load_model()
        encoder.model.max_seq_length = 128

        logger.info("Encoding %d texts with LaBSE ...", len(texts))
        embeddings = encoder.encode(texts, batch_size=32, show_progress=True)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        logger.info("Saved embeddings to %s (shape %s)", EMBEDDINGS_PATH, embeddings.shape)

        del encoder
        gc.collect()

    from src.retrieval.indexer import FAISSIndexer
    indexer = FAISSIndexer(embedding_dim=embeddings.shape[1])
    indexer.build_index(embeddings)
    indexer.save_index(INDEX_PATH)

    metadata = df[["case_id", "case_text", "condition_group"]].reset_index(drop=True)
    metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8")
    logger.info("Saved metadata to %s (%d rows)", METADATA_PATH, len(metadata))

    logger.info("Index build complete. Files in %s/", INDEX_DIR)


if __name__ == "__main__":
    main()
