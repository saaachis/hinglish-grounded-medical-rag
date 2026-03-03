"""Build FAISS index from HMG-mini report texts.

This MVP-friendly version uses TF-IDF embeddings to avoid heavyweight
model downloads and environment issues. It keeps the pipeline fast and
reproducible for student timelines.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_HMG = "data/processed/hmg_mini.csv"
DEFAULT_INDEX = "indices/openi_reports.index"
DEFAULT_META = "indices/openi_reports_metadata.csv"
DEFAULT_VECTORIZER = "indices/tfidf_vectorizer.pkl"
CONCEPT_MAP = {
    "pleural_effusion": ["pleural effusion", "effusion", "fluid"],
    "pneumonia": ["pneumonia", "infect", "infection"],
    "consolidation": ["consolidation"],
    "cardiomegaly": ["cardiomegaly", "heart size", "cardiomediastinal"],
    "congestion": ["congestion", "edema", "vascular prominence"],
    "atelectasis": ["atelect", "collapse"],
    "opacity": ["opacity", "opacities", "infiltrate"],
    "no_acute": ["no acute", "normal", "lungs are clear", "no abnormality"],
}


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _clean_text(text: str) -> str:
    lower = str(text).lower()
    lower = re.sub(r"[^a-z0-9\s\-]", " ", lower)
    lower = re.sub(r"\s+", " ", lower).strip()
    return lower


def _extract_concepts(text: str) -> list[str]:
    lower = str(text).lower()
    concepts: list[str] = []
    for concept, patterns in CONCEPT_MAP.items():
        if any(pattern in lower for pattern in patterns):
            concepts.append(concept)
    return concepts


def _extract_side(text: str) -> str:
    lower = str(text).lower()
    if "right" in lower:
        return "right"
    if "left" in lower:
        return "left"
    if "bilateral" in lower or "bibasal" in lower or "bibasilar" in lower:
        return "bilateral"
    return "unspecified"


def build_index(
    hmg_path: Path,
    index_path: Path,
    metadata_path: Path,
    vectorizer_path: Path,
) -> None:
    df = pd.read_csv(hmg_path)
    reports_df = df[["report_id", "report_text"]].drop_duplicates().reset_index(drop=True)
    reports_df["clean_report_text"] = reports_df["report_text"].astype(str).apply(_clean_text)
    reports_df["report_concepts"] = reports_df["report_text"].astype(str).apply(
        lambda t: "|".join(_extract_concepts(t))
    )
    reports_df["report_side"] = reports_df["report_text"].astype(str).apply(_extract_side)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
        sublinear_tf=True,
    )
    sparse = vectorizer.fit_transform(reports_df["clean_report_text"].tolist())
    embeddings = sparse.toarray().astype("float32")
    embeddings = _normalize(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    reports_df.to_csv(metadata_path, index=False, encoding="utf-8")
    np.save(index_path.with_suffix(".embeddings.npy"), embeddings)

    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Index saved: {index_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Vectorizer saved: {vectorizer_path}")
    print(f"Unique reports indexed: {len(reports_df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from report texts")
    parser.add_argument("--hmg-path", type=Path, default=Path(DEFAULT_HMG))
    parser.add_argument("--index-path", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--metadata-path", type=Path, default=Path(DEFAULT_META))
    parser.add_argument("--vectorizer-path", type=Path, default=Path(DEFAULT_VECTORIZER))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        hmg_path=args.hmg_path,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        vectorizer_path=args.vectorizer_path,
    )


if __name__ == "__main__":
    main()

