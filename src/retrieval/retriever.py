"""
Evidence Retriever with Adaptive Context Selection.

Implements domain-aware retrieval following the MMed-RAG approach:
uses similarity-score-based truncation to select the optimal number
of retrieved contexts, filtering out noisy or irrelevant evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoding.text_encoder import TextEncoder
from src.retrieval.indexer import FAISSIndexer

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Retrieves and ranks clinical evidence for Hinglish queries.

    Uses adaptive context selection: instead of a fixed top-k,
    it detects sharp declines in similarity scores to determine
    the optimal number of retrieved contexts.

    Parameters
    ----------
    indexer : FAISSIndexer
        A FAISSIndexer instance with a built/loaded index.
    text_encoder : TextEncoder
        A TextEncoder instance for encoding queries.
    evidence_metadata : pd.DataFrame
        DataFrame with columns [case_id, case_text, condition_group]
        aligned by row index with the FAISS index vectors.
    max_k : int
        Maximum number of candidates to retrieve before truncation.
    """

    def __init__(
        self,
        indexer: FAISSIndexer,
        text_encoder: TextEncoder,
        evidence_metadata: pd.DataFrame,
        max_k: int = 10,
    ):
        self.indexer = indexer
        self.text_encoder = text_encoder
        self.evidence_metadata = evidence_metadata.reset_index(drop=True)
        self.max_k = max_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Retrieve relevant evidence for a Hinglish query.

        Parameters
        ----------
        query : str
            Hinglish patient query.
        top_k : int | None
            Fixed number of results. If None, uses adaptive truncation.

        Returns
        -------
        list[dict]
            Retrieved evidence items sorted by score (descending), each with:
            case_id, case_text, condition_group, score.
        """
        embedding = self.text_encoder.encode(
            [query], batch_size=1, show_progress=False
        )

        k = top_k if top_k is not None else self.max_k
        scores, indices = self.indexer.search(embedding[0], top_k=k)

        valid = indices >= 0
        scores = scores[valid]
        indices = indices[valid]

        if top_k is None and len(scores) > 1:
            n_keep = self.adaptive_truncation(scores)
            scores = scores[:n_keep]
            indices = indices[:n_keep]

        results = []
        for score, idx in zip(scores, indices):
            row = self.evidence_metadata.iloc[int(idx)]
            results.append({
                "case_id": str(row["case_id"]),
                "case_text": str(row["case_text"]),
                "condition_group": str(row.get("condition_group", "")),
                "score": float(score),
            })
        return results

    def adaptive_truncation(
        self, scores: np.ndarray, threshold_ratio: float = 0.5
    ) -> int:
        """Determine optimal number of results via score-based truncation.

        Detects the sharpest decline in similarity scores to filter
        out low-quality or irrelevant evidence that could lead to
        hallucinations.

        Parameters
        ----------
        scores : np.ndarray
            Sorted similarity scores (descending).
        threshold_ratio : float
            A result is dropped if the score gap to the previous result
            exceeds threshold_ratio * top_score.

        Returns
        -------
        int
            Optimal number of results to return (>= 1).
        """
        if len(scores) <= 1:
            return len(scores)

        top_score = float(scores[0])
        if top_score <= 0:
            return 1

        threshold = threshold_ratio * top_score
        for i in range(1, len(scores)):
            drop = float(scores[i - 1] - scores[i])
            if drop > threshold:
                return i

        return len(scores)

    @classmethod
    def from_disk(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
        text_encoder: TextEncoder | None = None,
        max_k: int = 10,
    ) -> "EvidenceRetriever":
        """Load a retriever from saved index and metadata files.

        Parameters
        ----------
        index_path : path
            Path to the saved FAISS index file.
        metadata_path : path
            Path to the evidence metadata CSV.
        text_encoder : TextEncoder | None
            If None, a default LaBSE encoder is created.
        max_k : int
            Maximum retrieval candidates.
        """
        if text_encoder is None:
            text_encoder = TextEncoder(device="cpu")

        indexer = FAISSIndexer()
        indexer.load_index(str(index_path))

        metadata = pd.read_csv(metadata_path)
        logger.info(
            "Loaded retriever: %d index vectors, %d metadata rows",
            indexer.index.ntotal,
            len(metadata),
        )
        return cls(indexer, text_encoder, metadata, max_k=max_k)
