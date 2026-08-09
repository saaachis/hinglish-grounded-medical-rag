"""
FAISS Index Builder.

Builds and manages vector indices over clinical report embeddings
for fast similarity-based retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """FAISS-based vector index for clinical evidence retrieval.

    Expects L2-normalized embeddings so that inner-product search
    is equivalent to cosine similarity.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the embeddings (768 for LaBSE).
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatIP | None = None

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build a FAISS inner-product index from L2-normalized embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape (n_documents, embedding_dim).
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        logger.info(
            "Built FAISS index: %d vectors, dim=%d",
            self.index.ntotal,
            self.embedding_dim,
        )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for nearest neighbors.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding of shape (1, embedding_dim) or (embedding_dim,).
        top_k : int
            Number of results to return.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (scores, indices) of the top-k nearest neighbors.
            Scores are cosine similarities (higher = more similar).
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Call build_index() or load_index() first.")
        q = query_embedding.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores, indices = self.index.search(q, top_k)
        return scores[0], indices[0]

    def save_index(self, path: str | Path) -> None:
        """Save the FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("Saved FAISS index to %s (%d vectors)", path, self.index.ntotal)

    def load_index(self, path: str | Path) -> None:
        """Load a FAISS index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        self.index = faiss.read_index(str(path))
        self.embedding_dim = self.index.d
        logger.info("Loaded FAISS index from %s (%d vectors)", path, self.index.ntotal)
