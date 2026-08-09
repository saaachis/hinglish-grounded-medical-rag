"""Text Encoder — LaBSE.

Language-Agnostic BERT Sentence Embedding for encoding Hinglish
queries and English medical reports into a shared multilingual
embedding space.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEncoder:
    """LaBSE-based text encoder for multilingual clinical text.

    Encodes both Hinglish patient queries and formal English clinical
    texts into the same embedding space, enabling cross-lingual
    semantic retrieval.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for LaBSE.
    device : str
        Device to run inference on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model: SentenceTransformer | None = None

    def load_model(self) -> None:
        """Load the LaBSE model."""
        logger.info("Loading LaBSE model: %s (device=%s)", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("LaBSE model loaded (dim=%d)", self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a batch of texts into embeddings.

        Parameters
        ----------
        texts : list[str]
            Input texts (Hinglish queries or English reports).
        batch_size : int
            Batch size for encoding.
        show_progress : bool
            Whether to display a progress bar.
        normalize : bool
            If True, L2-normalize embeddings (needed for cosine similarity
            via inner product in FAISS).

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (len(texts), embedding_dim).
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return np.array(embeddings, dtype=np.float32)
