"""MMCQSD ↔ MultiCaRe Pair Builder using LaBSE + FAISS.

Encodes MMCQSD Hinglish queries and MultiCaRe clinical case text
with LaBSE, builds a FAISS index over the evidence, and matches
each query to its best-matching clinical case using cosine similarity
with condition-label filtering.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

MULTICARE_PATH = Path("data/processed/multicare_filtered.csv")
MMCQSD_PATH = Path("data/processed/mmcqsd_queries.csv")
OUTPUT_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")

# Map MMCQSD image_reference condition names to our condition group names
MMCQSD_CONDITION_TO_GROUP: dict[str, str] = {
    "skin rash": "skin_rash",
    "neck swelling": "neck_swelling",
    "mouth ulcers": "mouth_ulcers",
    "lip swelling": "lip_swelling",
    "swollen tonsils": "swollen_tonsils",
    "foot swelling": "foot_swelling",
    "hand lump": "hand_lump",
    "swollen eye": "swollen_eye",
    "knee swelling": "knee_swelling",
    "edema": "edema",
    "eye redness": "eye_redness",
    "skin growth": "skin_growth",
    "skin irritation": "skin_irritation",
    "skin dryness": "skin_dryness",
    "dry scalp": "dry_scalp",
    "eye inflamation": "eye_inflammation",  # note: MMCQSD has typo
    "cyanosis": "cyanosis",
    "itichy eyelid": "itchy_eyelid",  # note: MMCQSD has typo
}

# Condition groups that are compatible for cross-matching
COMPATIBLE_GROUPS: dict[str, set[str]] = {
    "skin_rash": {"skin_rash", "skin_irritation", "skin_growth"},
    "skin_growth": {"skin_growth", "skin_rash"},
    "skin_irritation": {"skin_irritation", "skin_rash", "skin_dryness"},
    "skin_dryness": {"skin_dryness", "skin_irritation", "dry_scalp"},
    "dry_scalp": {"dry_scalp", "skin_dryness"},
    "swollen_eye": {"swollen_eye", "eye_redness", "eye_inflammation", "itchy_eyelid"},
    "eye_redness": {"eye_redness", "swollen_eye", "eye_inflammation"},
    "eye_inflammation": {"eye_inflammation", "eye_redness", "swollen_eye"},
    "itchy_eyelid": {"itchy_eyelid", "swollen_eye", "eye_inflammation"},
    "mouth_ulcers": {"mouth_ulcers", "lip_swelling"},
    "lip_swelling": {"lip_swelling", "mouth_ulcers"},
    "swollen_tonsils": {"swollen_tonsils", "neck_swelling"},
    "neck_swelling": {"neck_swelling", "swollen_tonsils"},
    "foot_swelling": {"foot_swelling", "edema", "knee_swelling"},
    "hand_lump": {"hand_lump"},
    "knee_swelling": {"knee_swelling", "foot_swelling"},
    "edema": {"edema", "foot_swelling"},
    "cyanosis": {"cyanosis", "edema"},
}

MAX_EVIDENCE_WORDS = 300


def _extract_mmcqsd_condition(image_reference: str) -> str:
    """Extract condition from MMCQSD image_reference path."""
    match = re.search(r"Multimodal_images/([^/]+)/", str(image_reference))
    if match:
        raw = match.group(1).strip().lower()
        return MMCQSD_CONDITION_TO_GROUP.get(raw, raw)
    return "unknown"


def _truncate_for_encoding(text: str, max_words: int = MAX_EVIDENCE_WORDS) -> str:
    """Take the first N words of text for LaBSE encoding."""
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words])


def _estimate_cmi_bucket(text: str) -> str:
    """Estimate code-mixing level of a Hinglish query."""
    hindi_tokens = {
        "kya", "hai", "me", "mujhe", "meri", "mera", "doctor", "kripya",
        "saans", "khansi", "bukhar", "dard", "batao", "samjhao", "ho",
        "raha", "rahi", "hain", "nahi", "aur", "bhi", "ko", "se", "ke",
        "ki", "ka", "ye", "wo", "ek", "bahut", "thoda", "zyada",
        "abhi", "pehle", "baad", "dono", "uske", "iske",
    }
    tokens = re.findall(r"[a-zA-Z]+", str(text).lower())
    if not tokens:
        return "medium"
    ratio = sum(t in hindi_tokens for t in tokens) / len(tokens)
    if ratio < 0.10:
        return "low"
    if ratio < 0.30:
        return "medium"
    return "high"


def prepare_evidence_texts(multicare_df: pd.DataFrame) -> list[str]:
    """Prepare MultiCaRe case texts for LaBSE encoding."""
    return [
        _truncate_for_encoding(text)
        for text in multicare_df["case_text"].tolist()
    ]


def prepare_query_texts(mmcqsd_df: pd.DataFrame) -> list[str]:
    """Prepare MMCQSD query texts for LaBSE encoding.

    Combines hinglish_query + english_summary for stronger matching signal.
    """
    texts = []
    for _, row in mmcqsd_df.iterrows():
        query = str(row.get("hinglish_query", ""))
        summary = str(row.get("english_summary_or_target", ""))
        combined = f"{query} {summary}".strip()
        texts.append(combined)
    return texts


def encode_and_save(
    texts: list[str],
    output_path: Path,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with LaBSE and save embeddings."""
    from src.encoding.text_encoder import TextEncoder

    if output_path.exists():
        logger.info("Loading cached embeddings from %s", output_path)
        return np.load(output_path)

    encoder = TextEncoder(device="cpu")
    logger.info("Encoding %d texts with LaBSE...", len(texts))
    embeddings = encoder.encode(texts, batch_size=batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info("Saved embeddings to %s (shape: %s)", output_path, embeddings.shape)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index over L2-normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors (dim=%d)", index.ntotal, dim)
    return index


def match_queries_to_evidence(
    query_embeddings: np.ndarray,
    evidence_index: faiss.IndexFlatIP,
    mmcqsd_df: pd.DataFrame,
    multicare_df: pd.DataFrame,
    top_k: int = 10,
    min_similarity: float = 0.25,
    use_condition_filter: bool = True,
) -> pd.DataFrame:
    """Match each MMCQSD query to its best MultiCaRe evidence case.

    Parameters
    ----------
    query_embeddings : np.ndarray
        LaBSE embeddings of MMCQSD queries.
    evidence_index : faiss.IndexFlatIP
        FAISS index over MultiCaRe evidence embeddings.
    mmcqsd_df : pd.DataFrame
        MMCQSD dataset with query texts and conditions.
    multicare_df : pd.DataFrame
        Filtered MultiCaRe dataset with case texts and condition groups.
    top_k : int
        Number of candidates to retrieve per query.
    min_similarity : float
        Minimum cosine similarity to accept a match.
    use_condition_filter : bool
        If True, prefer evidence from compatible condition groups.

    Returns
    -------
    pd.DataFrame
        Matched pairs with query, evidence, scores, and metadata.
    """
    logger.info(
        "Matching %d queries against %d evidence cases (top_k=%d, min_sim=%.2f)...",
        len(mmcqsd_df), evidence_index.ntotal, top_k, min_similarity,
    )

    scores_all, indices_all = evidence_index.search(query_embeddings, top_k)

    mmcqsd_conditions = mmcqsd_df["image_reference"].apply(_extract_mmcqsd_condition).tolist()
    multicare_conditions = multicare_df["condition_group"].tolist()

    pairs: list[dict] = []
    no_match_count = 0

    for q_idx in tqdm(range(len(mmcqsd_df)), desc="Matching queries"):
        q_row = mmcqsd_df.iloc[q_idx]
        q_condition = mmcqsd_conditions[q_idx]
        compatible = COMPATIBLE_GROUPS.get(q_condition, {q_condition})

        best_idx = -1
        best_score = -1.0

        for rank in range(top_k):
            e_idx = int(indices_all[q_idx, rank])
            sim = float(scores_all[q_idx, rank])

            if sim < min_similarity:
                break

            e_condition = multicare_conditions[e_idx]

            if use_condition_filter and e_condition not in compatible:
                continue

            if sim > best_score:
                best_score = sim
                best_idx = e_idx

        # If condition filter found nothing, fall back to best raw match
        if best_idx == -1 and not use_condition_filter:
            if scores_all[q_idx, 0] >= min_similarity:
                best_idx = int(indices_all[q_idx, 0])
                best_score = float(scores_all[q_idx, 0])

        if best_idx == -1:
            # Try relaxed: take best from top-k ignoring condition
            for rank in range(top_k):
                e_idx = int(indices_all[q_idx, rank])
                sim = float(scores_all[q_idx, rank])
                if sim >= min_similarity:
                    best_idx = e_idx
                    best_score = sim
                    break

        if best_idx == -1:
            no_match_count += 1
            continue

        e_row = multicare_df.iloc[best_idx]

        pairs.append({
            "pair_id": f"P{len(pairs)+1:05d}",
            "mmcqsd_sample_id": str(q_row.get("sample_id", q_idx)),
            "hinglish_query": str(q_row.get("hinglish_query", "")),
            "english_summary": str(q_row.get("english_summary_or_target", "")),
            "multicare_case_id": str(e_row.get("case_id", best_idx)),
            "evidence_text": _truncate_for_encoding(str(e_row.get("case_text", "")), 500),
            "condition_group": q_condition,
            "evidence_condition": str(e_row.get("condition_group", "")),
            "similarity_score": round(best_score, 4),
            "cmi_bucket": _estimate_cmi_bucket(str(q_row.get("hinglish_query", ""))),
            "match_quality": (
                "high" if best_score >= 0.5
                else "medium" if best_score >= 0.35
                else "low"
            ),
        })

    logger.info(
        "Matching complete: %d pairs created, %d queries unmatched",
        len(pairs), no_match_count,
    )

    return pd.DataFrame(pairs)


def print_matching_summary(pairs_df: pd.DataFrame, total_queries: int) -> str:
    """Generate a summary report of the matching results."""
    lines = [
        "# MMCQSD ↔ MultiCaRe Matching Summary",
        "",
        f"- Total MMCQSD queries: **{total_queries}**",
        f"- Matched pairs: **{len(pairs_df)}** ({len(pairs_df)/total_queries*100:.1f}%)",
        f"- Unmatched: **{total_queries - len(pairs_df)}**",
        "",
        f"- Mean similarity: **{pairs_df['similarity_score'].mean():.4f}**",
        f"- Median similarity: **{pairs_df['similarity_score'].median():.4f}**",
        f"- Min similarity: **{pairs_df['similarity_score'].min():.4f}**",
        f"- Max similarity: **{pairs_df['similarity_score'].max():.4f}**",
        "",
        "## Match Quality Distribution",
        "",
    ]

    quality_counts = pairs_df["match_quality"].value_counts()
    for q in ["high", "medium", "low"]:
        count = quality_counts.get(q, 0)
        pct = count / len(pairs_df) * 100 if len(pairs_df) > 0 else 0
        lines.append(f"- **{q}**: {count} ({pct:.1f}%)")

    lines.extend(["", "## Per-Condition Breakdown", "",
                   "| Condition | Queries | Matched | Rate | Avg Sim |",
                   "|---|---:|---:|---:|---:|"])

    for condition in sorted(pairs_df["condition_group"].unique()):
        subset = pairs_df[pairs_df["condition_group"] == condition]
        total_cond = total_queries  # approximate
        lines.append(
            f"| {condition} | — | {len(subset)} | — | "
            f"{subset['similarity_score'].mean():.3f} |"
        )

    lines.extend(["", "## CMI Bucket Distribution", ""])
    for bucket in ["low", "medium", "high"]:
        count = len(pairs_df[pairs_df["cmi_bucket"] == bucket])
        lines.append(f"- **{bucket}**: {count}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match MMCQSD queries to MultiCaRe evidence using LaBSE"
    )
    parser.add_argument(
        "--multicare", type=Path, default=MULTICARE_PATH,
        help="Path to filtered MultiCaRe CSV",
    )
    parser.add_argument(
        "--mmcqsd", type=Path, default=MMCQSD_PATH,
        help="Path to MMCQSD queries CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for paired data",
    )
    parser.add_argument(
        "--embeddings-dir", type=Path, default=EMBEDDINGS_DIR,
        help="Directory for cached embeddings",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of candidates per query",
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.25,
        help="Minimum cosine similarity threshold",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for LaBSE encoding",
    )
    parser.add_argument(
        "--no-condition-filter", action="store_true",
        help="Disable condition-label compatibility filtering",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load data
    logger.info("Loading MultiCaRe filtered data...")
    multicare_df = pd.read_csv(args.multicare)
    logger.info("Loaded %d MultiCaRe cases", len(multicare_df))

    logger.info("Loading MMCQSD queries...")
    mmcqsd_df = pd.read_csv(args.mmcqsd).fillna("")
    logger.info("Loaded %d MMCQSD queries", len(mmcqsd_df))

    # Deduplicate MultiCaRe by case_id (keep first condition group assignment)
    multicare_dedup = multicare_df.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
    logger.info("After dedup: %d unique MultiCaRe cases", len(multicare_dedup))

    # Prepare texts for encoding
    logger.info("Preparing texts for LaBSE encoding...")
    evidence_texts = prepare_evidence_texts(multicare_dedup)
    query_texts = prepare_query_texts(mmcqsd_df)

    # Encode
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    evidence_emb = encode_and_save(
        evidence_texts,
        args.embeddings_dir / "evidence_embeddings.npy",
        batch_size=args.batch_size,
    )
    query_emb = encode_and_save(
        query_texts,
        args.embeddings_dir / "query_embeddings.npy",
        batch_size=args.batch_size,
    )

    # Build index and match
    index = build_faiss_index(evidence_emb)

    pairs_df = match_queries_to_evidence(
        query_embeddings=query_emb,
        evidence_index=index,
        mmcqsd_df=mmcqsd_df,
        multicare_df=multicare_dedup,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        use_condition_filter=not args.no_condition_filter,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "mmcqsd_multicare_paired.csv"
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8")
    logger.info("Saved %d pairs to %s", len(pairs_df), pairs_path)

    summary = print_matching_summary(pairs_df, len(mmcqsd_df))
    summary_path = args.output_dir / "matching_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    print()
    print(summary)


if __name__ == "__main__":
    main()
