"""Prepare a real Open-i + MMCQSD prototype dataset.

This script builds a best-effort real prototype setup where:
- Open-i provides the chest-report evidence corpus
- MMCQSD provides the Hinglish query side

Because the two datasets are not natively paired, the script:
1. Filters MMCQSD to cardio-respiratory oriented cases
2. Aligns each retained query to the best matching Open-i report
3. Exports both the Open-i corpus and the aligned query set
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_OUTPUT_DIR = Path("data/processed/real_h1")
DEFAULT_MAX_QUERIES = 40
DEFAULT_SEED = 42
OPENI_PATH = Path("data/processed/openi_reports.csv")
MMCQSD_PATH = Path("data/processed/mmcqsd_queries.csv")
MATCH_CONCEPT_MAP = {
    "pleural_effusion": ["pleural effusion", "pleural fluid"],
    "pneumonia": ["pneumonia", "lung infection", "pulmonary infection"],
    "consolidation": ["consolidation"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart enlarged"],
    "congestion": [
        "pulmonary edema",
        "pulmonary congestion",
        "vascular congestion",
        "congestive heart failure",
        "acute heart failure",
        "heart failure",
        "chf",
    ],
    "atelectasis": ["atelect", "lung collapse"],
    "opacity": ["opacity", "opacities", "infiltrate"],
    "pneumothorax": ["pneumothorax"],
    "hyperinflation": ["hyperinflated", "copd", "emphysema", "air trapping"],
}
MMCQSD_STRONG_PATTERNS = [
    r"\bchest\b",
    r"\blung\b",
    r"\bpulmonary\b",
    r"\bpneumonia\b",
    r"\bpleural\b",
    r"\beffusion\b",
    r"\bpneumothorax\b",
    r"\bcardio\w*\b",
    r"\bheart\b",
    r"\bcardiomegaly\b",
    r"\bcyanosis\b",
    r"\bedema\b",
    r"\bshortness of breath\b",
    r"\bdifficulty breathing\b",
    r"\bbreathing difficulty\b",
    r"\bbreath\w*\b",
    r"\bwheez\w*\b",
    r"\bcough\w*\b",
    r"\brespirat\w*\b",
    r"\basthma\b",
    r"\bronchi\b",
]
MMCQSD_SUPPORT_PATTERNS = [
    r"\bedema\b",
    r"\bfluid overload\b",
    r"\bswelling\b",
]
MMCQSD_NEGATIVE_PATTERNS = [
    r"\btonsil\w*\b",
    r"\bthroat\b",
    r"\beye\w*\b",
    r"\beyelid\b",
    r"\bear\b",
    r"\bmouth\b",
    r"\bulcer\w*\b",
    r"\blip\b",
    r"\bskin\b",
    r"\brash\w*\b",
    r"\bhand\b",
    r"\bfoot\b",
    r"\bknee\b",
    r"\bscalp\b",
    r"\btongue\b",
    r"\bteeth\b",
    r"\btooth\b",
    r"\bgum\b",
]
ALLOWED_IMAGE_CONDITIONS = {"edema", "cyanosis"}


def _clean_text(text: str) -> str:
    lower = str(text).lower()
    lower = re.sub(r"[^a-z0-9\s\-]", " ", lower)
    lower = re.sub(r"\s+", " ", lower).strip()
    return lower


def _extract_concepts(text: str) -> set[str]:
    lower = str(text).lower()
    concepts: set[str] = set()
    for concept, patterns in MATCH_CONCEPT_MAP.items():
        if any(pattern in lower for pattern in patterns):
            concepts.add(concept)
    return concepts


def _estimate_cmi_bucket(text: str) -> str:
    hindi_like = {
        "kya", "hai", "me", "mujhe", "meri", "mera", "doctor", "kripya",
        "saans", "khansi", "bukhar", "dard", "batao", "samjhao",
    }
    tokens = re.findall(r"[a-zA-Z]+", str(text).lower())
    if not tokens:
        return "medium"
    ratio = sum(token in hindi_like for token in tokens) / len(tokens)
    if ratio < 0.15:
        return "low"
    if ratio < 0.35:
        return "medium"
    return "high"


def _image_condition(image_reference: str) -> str:
    match = re.search(r"Multimodal_images/([^/]+)/", str(image_reference))
    return match.group(1).strip().lower() if match else ""


def _keep_mmcqsd_row(query_text: str, target_text: str, image_reference: str) -> bool:
    combined = f"{query_text} {target_text}".lower()
    condition = _image_condition(image_reference)
    match_concepts = _extract_concepts(combined)
    strong_hits = sum(bool(re.search(pattern, combined)) for pattern in MMCQSD_STRONG_PATTERNS)
    support_hits = sum(bool(re.search(pattern, combined)) for pattern in MMCQSD_SUPPORT_PATTERNS)
    negative_hits = sum(bool(re.search(pattern, combined)) for pattern in MMCQSD_NEGATIVE_PATTERNS)

    if strong_hits == 0:
        return False
    if not match_concepts:
        return False
    if condition == "edema":
        cardio_respiratory_anchor = re.search(
            r"\b(heart|cardio|cardiomegaly|pulmonary|lung|chest|breath|respirat|cough|asthma|pneumonia|pneumothorax|pleural|effusion|cyanosis)\b",
            combined,
        )
        if not cardio_respiratory_anchor:
            return False
    if condition and condition not in ALLOWED_IMAGE_CONDITIONS and negative_hits > 0:
        return False
    if negative_hits >= 2 and condition not in ALLOWED_IMAGE_CONDITIONS:
        return False
    return strong_hits >= 2 or condition in ALLOWED_IMAGE_CONDITIONS or support_hits > 0


def _prepare_openi_corpus() -> pd.DataFrame:
    df = pd.read_csv(OPENI_PATH).fillna("")
    df = df[df["report_text"].astype(str).str.strip().ne("")].copy()
    return df[["report_id", "report_text"]].drop_duplicates().reset_index(drop=True)


def _prepare_mmcqsd_candidates() -> pd.DataFrame:
    df = pd.read_csv(MMCQSD_PATH).fillna("")
    df = df[
        df["hinglish_query"].astype(str).str.strip().ne("")
        & df["english_summary_or_target"].astype(str).str.strip().ne("")
    ].copy()
    mask = df.apply(
        lambda row: _keep_mmcqsd_row(
            query_text=str(row["hinglish_query"]),
            target_text=str(row["english_summary_or_target"]),
            image_reference=str(row["image_reference"]),
        ),
        axis=1,
    )
    df = df[mask].reset_index(drop=True)
    return df


def _alignment_bonus(query_concepts: set[str], report_concepts: set[str]) -> float:
    overlap = len(query_concepts & report_concepts)
    if overlap == 0:
        return 0.0
    return 0.12 * overlap


def _has_contradiction(query_concepts: set[str], report_text: str) -> bool:
    lower = report_text.lower()
    contradiction_patterns = {
        "congestion": [
            "no heart failure",
            "without heart failure",
            "no evidence for heart failure",
            "no evidence of heart failure",
            "without radiographic evidence of heart failure",
            "no evidence of chf",
            "no overt evidence of chf",
            "no pulmonary edema",
            "no overt edema",
            "no edema",
        ],
        "pleural_effusion": [
            "no pleural effusion",
            "no pleural effusions",
        ],
        "cardiomegaly": [
            "heart size normal",
            "heart is normal in size",
            "normal heart size",
            "heart and mediastinum normal",
        ],
        "pneumonia": [
            "no pneumonia",
            "no focal consolidation",
            "lungs are clear",
        ],
        "consolidation": [
            "no focal consolidation",
            "no consolidation",
        ],
        "pneumothorax": [
            "no pneumothorax",
        ],
        "opacity": [
            "lungs are clear",
            "no focal opacity",
            "no focal airspace consolidation",
        ],
    }
    for concept in query_concepts:
        patterns = contradiction_patterns.get(concept, [])
        if any(pattern in lower for pattern in patterns):
            return True
    return False


def _align_queries_to_openi(
    openi_corpus: pd.DataFrame,
    mmcqsd_df: pd.DataFrame,
    max_queries: int,
    seed: int,
) -> pd.DataFrame:
    corpus_texts = openi_corpus["report_text"].astype(str).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12000, sublinear_tf=True)
    report_matrix = vectorizer.fit_transform([_clean_text(text) for text in corpus_texts])
    report_concepts = [_extract_concepts(text) for text in corpus_texts]

    aligned_rows: list[dict[str, object]] = []
    for _, row in mmcqsd_df.iterrows():
        query_text = str(row["hinglish_query"])
        target_text = str(row["english_summary_or_target"])
        combined_text = f"{query_text} {target_text}"
        query_vector = vectorizer.transform([_clean_text(combined_text)])
        similarities = (query_vector @ report_matrix.T).toarray()[0]
        query_concepts = _extract_concepts(combined_text)

        best_idx = 0
        best_score = float("-inf")
        for idx, base_score in enumerate(similarities):
            if _has_contradiction(query_concepts, corpus_texts[idx]):
                continue
            score = float(base_score) + _alignment_bonus(query_concepts, report_concepts[idx])
            if score > best_score:
                best_idx = idx
                best_score = score

        matched = openi_corpus.iloc[best_idx]
        matched_report_text = str(matched["report_text"])
        overlap = sorted(query_concepts & report_concepts[best_idx])
        if not overlap:
            continue
        aligned_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "query_hinglish": query_text,
                "report_id": str(matched["report_id"]),
                "report_text": matched_report_text,
                "target_text": target_text,
                "dataset_profile": "openi_mmcqsd_real",
                "cmi_bucket": _estimate_cmi_bucket(query_text),
                "image_condition": _image_condition(str(row["image_reference"])),
                "alignment_score": float(best_score),
                "matched_concepts": "|".join(overlap),
            }
        )

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_df = aligned_df.sort_values(by="alignment_score", ascending=False).reset_index(drop=True)
    if len(aligned_df) > max_queries:
        aligned_df = aligned_df.head(max_queries).copy()
    return aligned_df


def prepare_openi_mmcqsd_real(
    output_dir: Path,
    max_queries: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    openi_corpus = _prepare_openi_corpus()
    mmcqsd_candidates = _prepare_mmcqsd_candidates()
    aligned_queries = _align_queries_to_openi(
        openi_corpus=openi_corpus,
        mmcqsd_df=mmcqsd_candidates,
        max_queries=max_queries,
        seed=seed,
    )

    corpus_path = output_dir / "openi_real_corpus.csv"
    queries_path = output_dir / "openi_mmcqsd_real_queries.csv"
    summary_path = output_dir / "openi_mmcqsd_real_prep_summary.md"

    openi_corpus.to_csv(corpus_path, index=False, encoding="utf-8")
    aligned_queries.to_csv(queries_path, index=False, encoding="utf-8")

    summary_lines = [
        "# Open-i + MMCQSD Real Preparation Summary",
        "",
        f"- Total Open-i reports available: **{len(openi_corpus)}**",
        f"- MMCQSD cardio-respiratory candidates found: **{len(mmcqsd_candidates)}**",
        f"- Final aligned query set retained: **{len(aligned_queries)}**",
        f"- Mean alignment score: **{aligned_queries['alignment_score'].mean():.4f}**" if not aligned_queries.empty else "- Mean alignment score: **N/A**",
        "",
        "## Notes",
        "- MMCQSD was filtered to retain cardio-respiratory style cases most compatible with a chest-report corpus.",
        "- Each MMCQSD query was aligned to the best-matching Open-i report using TF-IDF similarity plus lightweight concept-overlap bonuses.",
        "- This is a weakly supervised real pairing, used to make the prototype pipeline operational on real data.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return openi_corpus, aligned_queries, corpus_path, queries_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real Open-i + MMCQSD prototype data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    openi_corpus, aligned_queries, corpus_path, queries_path, summary_path = prepare_openi_mmcqsd_real(
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    print(f"Open-i corpus rows: {len(openi_corpus)}")
    print(f"Aligned query rows: {len(aligned_queries)}")
    print(f"Corpus CSV: {corpus_path}")
    print(f"Queries CSV: {queries_path}")
    print(f"Prep summary: {summary_path}")


if __name__ == "__main__":
    main()
