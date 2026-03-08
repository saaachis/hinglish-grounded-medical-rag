"""Run zero-shot and grounded baselines for retrieval experiments."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


DEFAULT_HMG = "data/processed/hmg_mini.csv"
DEFAULT_INDEX = "indices/openi_reports.index"
DEFAULT_META = "indices/openi_reports_metadata.csv"
DEFAULT_VECTORIZER = "indices/tfidf_vectorizer.pkl"
DEFAULT_OUTPUT = "results/h1_outputs.csv"
QUERY_NORMALIZATION = {
    "chhati": "chest",
    "saas": "breath",
    "pani": "fluid",
    "bukhar": "fever",
    "khansi": "cough",
    "dil": "heart",
    "dikkat": "problem",
    "bhar gaya": "accumulated",
    "normal": "no acute",
}
CONCEPT_MAP = {
    "pleural_effusion": ["pleural effusion", "effusion", "fluid"],
    "pneumonia": ["pneumonia", "infect", "infection"],
    "consolidation": ["consolidation"],
    "cardiomegaly": ["cardiomegaly", "heart size", "cardiomediastinal", "heart"],
    "congestion": ["congestion", "edema", "vascular prominence"],
    "atelectasis": ["atelect", "collapse"],
    "opacity": ["opacity", "opacities", "infiltrate"],
    "no_acute": ["no acute", "normal", "lungs are clear", "no abnormality"],
}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _first_sentence(text: str) -> str:
    text = str(text).strip()
    if "." in text:
        return text.split(".", maxsplit=1)[0].strip() + "."
    return text


def _clean_text(text: str) -> str:
    lower = str(text).lower()
    for k, v in QUERY_NORMALIZATION.items():
        lower = lower.replace(k, v)
    lower = re.sub(r"[^a-z0-9\s\-]", " ", lower)
    lower = re.sub(r"\s+", " ", lower).strip()
    return lower


def _extract_concepts(text: str) -> set[str]:
    lower = str(text).lower()
    found: set[str] = set()
    for concept, patterns in CONCEPT_MAP.items():
        if any(pattern in lower for pattern in patterns):
            found.add(concept)
    return found


def _extract_side(text: str) -> str:
    lower = str(text).lower()
    if "right" in lower:
        return "right"
    if "left" in lower:
        return "left"
    if "dono side" in lower or "bilateral" in lower or "bibasal" in lower:
        return "bilateral"
    return "unspecified"


def _token_overlap_ratio(query_text: str, report_text: str) -> float:
    q_tokens = set(_clean_text(query_text).split())
    r_tokens = set(str(report_text).split())
    if not q_tokens or not r_tokens:
        return 0.0
    return len(q_tokens & r_tokens) / max(len(q_tokens), 1)


def _rerank_with_concepts(
    query_text: str,
    candidate_ids: list[int],
    candidate_scores: list[float],
    metadata_df: pd.DataFrame,
) -> tuple[list[int], list[float]]:
    q_concepts = _extract_concepts(query_text)
    q_side = _extract_side(query_text)

    combined: list[tuple[int, float]] = []
    for idx, base_score in zip(candidate_ids, candidate_scores):
        if idx < 0 or idx >= len(metadata_df):
            continue
        row = metadata_df.iloc[idx]
        report_concepts = set(str(row.get("report_concepts", "")).split("|"))
        report_concepts.discard("")
        overlap = len(q_concepts & report_concepts) if q_concepts else 0
        lexical = _token_overlap_ratio(query_text, str(row.get("clean_report_text", "")))
        report_side = str(row.get("report_side", "unspecified"))
        side_bonus = 0.0
        if q_side != "unspecified" and report_side == q_side:
            side_bonus = 0.08
        rerank_score = float(base_score) + (0.12 * overlap) + (0.35 * lexical) + side_bonus
        combined.append((idx, rerank_score))

    if not combined:
        return candidate_ids, candidate_scores
    combined.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in combined], [item[1] for item in combined]


def generate_zero_shot(query: str) -> str:
    return (
        f"Query: {query} Without external evidence, only a tentative answer is possible. "
        "The response should be treated as unverified."
    )


def generate_grounded(query: str, evidence_texts: list[str]) -> str:
    top_evidence = _first_sentence(evidence_texts[0]) if evidence_texts else ""
    return (
        f"Query: {query} Available report evidence ke hisaab se: {top_evidence} "
        "Isliye explanation report-grounded hai."
    ).strip()


def run(
    hmg_path: Path,
    index_path: Path,
    metadata_path: Path,
    vectorizer_path: Path,
    output_path: Path,
    top_k: int,
) -> pd.DataFrame:
    hmg_df = pd.read_csv(hmg_path)
    metadata_df = pd.read_csv(metadata_path)
    index = faiss.read_index(str(index_path))
    with vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    queries = hmg_df["query_hinglish"].astype(str).tolist()
    clean_queries = [_clean_text(q) for q in queries]
    sparse_q = vectorizer.transform(clean_queries)
    q_emb = sparse_q.toarray().astype("float32")
    q_emb = _normalize(q_emb).astype("float32")

    scores, neighbors = index.search(q_emb, top_k)
    records: list[dict] = []

    for row_idx, sample in hmg_df.iterrows():
        neighbor_ids = neighbors[row_idx].tolist()
        neighbor_scores = scores[row_idx].tolist()
        neighbor_ids, neighbor_scores = _rerank_with_concepts(
            query_text=str(sample["query_hinglish"]),
            candidate_ids=neighbor_ids,
            candidate_scores=neighbor_scores,
            metadata_df=metadata_df,
        )
        valid_ids = [i for i in neighbor_ids if i >= 0 and i < len(metadata_df)]
        evidence_rows = metadata_df.iloc[valid_ids]
        evidence_texts = evidence_rows["report_text"].astype(str).tolist()
        evidence_ids = evidence_rows["report_id"].astype(str).tolist()
        expected_report_id = str(sample["report_id"])
        top1_report_id = evidence_ids[0] if evidence_ids else ""
        top1_hit = int(top1_report_id == expected_report_id)
        topk_hit = int(expected_report_id in evidence_ids)

        zero = generate_zero_shot(str(sample["query_hinglish"]))
        grounded = generate_grounded(str(sample["query_hinglish"]), evidence_texts)

        records.append(
            {
                "sample_id": sample["sample_id"],
                "query_hinglish": sample["query_hinglish"],
                "dataset_profile": sample.get("dataset_profile", ""),
                "expected_report_id": expected_report_id,
                "target_text": sample.get("target_text", ""),
                "retrieved_top1_report_id": top1_report_id,
                "retrieved_report_ids": "|".join(evidence_ids),
                "retrieval_top1_hit": top1_hit,
                "retrieval_topk_hit": topk_hit,
                "retrieval_score_top1": float(neighbor_scores[0]) if len(neighbor_scores) else 0.0,
                "top1_evidence_text": evidence_texts[0] if evidence_texts else "",
                "retrieved_evidence_text": " || ".join(evidence_texts[:2]),
                "zero_shot_output": zero,
                "grounded_output": grounded,
            }
        )

    result_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot and grounded baselines")
    parser.add_argument("--hmg-path", type=Path, default=Path(DEFAULT_HMG))
    parser.add_argument("--index-path", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--metadata-path", type=Path, default=Path(DEFAULT_META))
    parser.add_argument("--vectorizer-path", type=Path, default=Path(DEFAULT_VECTORIZER))
    parser.add_argument("--output-path", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run(
        hmg_path=args.hmg_path,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        vectorizer_path=args.vectorizer_path,
        output_path=args.output_path,
        top_k=args.top_k,
    )
    print(f"Saved baseline outputs: {args.output_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()

