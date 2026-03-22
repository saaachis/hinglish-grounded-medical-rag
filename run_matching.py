"""Memory-efficient LaBSE matching pipeline.

Designed for systems with limited RAM (~8 GB) and CPU-only inference:
- Samples evidence to ~5K cases (3x per condition) for speed
- Truncates to 200 words, sets max_seq_length=128
- Encodes with batch_size=32
- Builds FAISS index and matches
"""

from __future__ import annotations

import gc
import logging
import re
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
MMCQSD_PATH = Path("data/processed/mmcqsd_queries.csv")
EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/processed")

MAX_EVIDENCE_WORDS = 200
BATCH_SIZE = 32
SAMPLE_MULTIPLIER = 3  # 3x MMCQSD query count per condition

MMCQSD_CONDITION_COUNTS = {
    "skin rash": 575, "neck swelling": 285, "mouth ulcers": 269,
    "lip swelling": 250, "swollen tonsils": 222, "foot swelling": 218,
    "hand lump": 195, "swollen eye": 185, "knee swelling": 180,
    "edema": 152, "eye redness": 125, "skin growth": 110,
    "skin irritation": 82, "skin dryness": 58, "dry scalp": 50,
    "eye inflamation": 27, "cyanosis": 22, "itichy eyelid": 10,
}

CONDITION_TO_GROUP = {
    "skin rash": "skin_rash", "neck swelling": "neck_swelling",
    "mouth ulcers": "mouth_ulcers", "lip swelling": "lip_swelling",
    "swollen tonsils": "swollen_tonsils", "foot swelling": "foot_swelling",
    "hand lump": "hand_lump", "swollen eye": "swollen_eye",
    "knee swelling": "knee_swelling", "edema": "edema",
    "eye redness": "eye_redness", "skin growth": "skin_growth",
    "skin irritation": "skin_irritation", "skin dryness": "skin_dryness",
    "dry scalp": "dry_scalp", "eye inflamation": "eye_inflammation",
    "cyanosis": "cyanosis", "itichy eyelid": "itchy_eyelid",
}


def truncate(text: str, max_words: int = MAX_EVIDENCE_WORDS) -> str:
    words = str(text).split()
    return " ".join(words[:max_words]) if len(words) > max_words else str(text)


def sample_evidence(mc_dedup: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Sample ~3x MMCQSD queries per condition from MultiCaRe evidence."""
    rng = np.random.RandomState(seed)
    sampled_parts = []

    for mmcqsd_cond, count in MMCQSD_CONDITION_COUNTS.items():
        group_name = CONDITION_TO_GROUP.get(mmcqsd_cond, mmcqsd_cond)
        target = count * SAMPLE_MULTIPLIER
        available = mc_dedup[mc_dedup["condition_group"] == group_name]

        if len(available) == 0:
            logger.warning("  No evidence for condition '%s'", group_name)
            continue

        n = min(target, len(available))
        chosen = available.sample(n=n, random_state=rng)
        sampled_parts.append(chosen)
        logger.info("  %s: sampled %d / %d available (target %d)",
                     group_name, n, len(available), target)

    sampled = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["case_id"])
    logger.info("Total sampled evidence: %d unique cases", len(sampled))
    return sampled.reset_index(drop=True)


def encode_texts(
    texts: list[str],
    save_path: Path,
    batch_size: int = BATCH_SIZE,
    model=None,
):
    """Encode texts with LaBSE; returns (embeddings, model)."""
    if save_path.exists():
        logger.info("Loading cached embeddings: %s", save_path)
        return np.load(save_path), model

    if model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading LaBSE model...")
        model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
        model.max_seq_length = 128
        logger.info("LaBSE loaded (dim=%d, max_seq=%d)",
                     model.get_sentence_embedding_dimension(), model.max_seq_length)

    logger.info("Encoding %d texts (batch_size=%d)...", len(texts), batch_size)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.array(emb, dtype=np.float32)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, emb)
    logger.info("Saved embeddings: %s (shape: %s)", save_path, emb.shape)
    return emb, model


def main():
    # Step 1: Load data
    logger.info("=== Step 1: Loading data ===")

    logger.info("Loading MultiCaRe filtered...")
    mc_df = pd.read_csv(MULTICARE_PATH)
    logger.info("  %d rows loaded", len(mc_df))

    mc_dedup = mc_df.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
    logger.info("  %d unique cases after dedup", len(mc_dedup))

    logger.info("Loading MMCQSD...")
    mm_df = pd.read_csv(MMCQSD_PATH).fillna("")
    logger.info("  %d queries loaded", len(mm_df))

    # Step 2: Sample evidence for speed
    logger.info("=== Step 2: Sampling evidence (3x per condition) ===")
    mc_sampled = sample_evidence(mc_dedup)

    # Step 3: Prepare texts
    logger.info("=== Step 3: Preparing texts ===")
    evidence_texts = [truncate(t) for t in mc_sampled["case_text"].tolist()]

    query_texts = []
    for _, row in mm_df.iterrows():
        q = str(row.get("hinglish_query", ""))
        s = str(row.get("english_summary_or_target", ""))
        query_texts.append(f"{q} {s}".strip())

    logger.info("  Evidence texts: %d, Query texts: %d", len(evidence_texts), len(query_texts))

    # Step 4: Encode evidence
    logger.info("=== Step 4: Encoding evidence with LaBSE ===")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    evidence_emb, model = encode_texts(
        evidence_texts,
        EMBEDDINGS_DIR / "evidence_embeddings.npy",
    )

    # Step 5: Encode queries (reuse loaded model)
    logger.info("=== Step 5: Encoding queries with LaBSE ===")
    query_emb, model = encode_texts(
        query_texts,
        EMBEDDINGS_DIR / "query_embeddings.npy",
        model=model,
    )

    del model
    gc.collect()

    # Step 6: Build FAISS index
    logger.info("=== Step 6: Building FAISS index ===")
    import faiss

    dim = evidence_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(evidence_emb)
    logger.info("  FAISS index: %d vectors, dim=%d", index.ntotal, dim)

    del evidence_emb
    gc.collect()

    # Step 7: Search
    logger.info("=== Step 7: Matching queries to evidence ===")
    TOP_K = 10
    MIN_SIM = 0.25

    scores, indices = index.search(query_emb, TOP_K)
    del query_emb, index
    gc.collect()

    # Step 8: Build pairs using condition compatibility
    logger.info("=== Step 8: Building pairs with condition filter ===")

    def extract_condition(img_ref: str) -> str:
        m = re.search(r"Multimodal_images/([^/]+)/", str(img_ref))
        return m.group(1).strip().lower() if m else "unknown"

    COMPAT_GROUPS = {
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

    mc_conditions = mc_sampled["condition_group"].tolist()
    raw_query_conds = [extract_condition(str(r.get("image_reference", ""))) for _, r in mm_df.iterrows()]
    query_conditions = [CONDITION_TO_GROUP.get(c, c) for c in raw_query_conds]

    pairs = []
    unmatched = 0

    for q_idx in range(len(mm_df)):
        q_cond = query_conditions[q_idx]
        compat_set = COMPAT_GROUPS.get(q_cond, {q_cond})

        best_idx, best_score = -1, -1.0

        # First pass: compatible condition
        for rank in range(TOP_K):
            e_idx = int(indices[q_idx, rank])
            sim = float(scores[q_idx, rank])
            if sim < MIN_SIM:
                break
            if mc_conditions[e_idx] in compat_set:
                best_idx, best_score = e_idx, sim
                break

        # Fallback: any condition above threshold
        if best_idx == -1:
            for rank in range(TOP_K):
                sim = float(scores[q_idx, rank])
                if sim >= MIN_SIM:
                    best_idx = int(indices[q_idx, rank])
                    best_score = sim
                    break

        if best_idx == -1:
            unmatched += 1
            continue

        q_row = mm_df.iloc[q_idx]
        e_row = mc_sampled.iloc[best_idx]

        pairs.append({
            "pair_id": f"P{len(pairs)+1:05d}",
            "mmcqsd_sample_id": str(q_row.get("sample_id", q_idx)),
            "hinglish_query": str(q_row.get("hinglish_query", "")),
            "english_summary": str(q_row.get("english_summary_or_target", "")),
            "multicare_case_id": str(e_row.get("case_id", best_idx)),
            "evidence_text": truncate(str(e_row.get("case_text", "")), 500),
            "condition_query": q_cond,
            "condition_evidence": str(e_row.get("condition_group", "")),
            "similarity_score": round(best_score, 4),
            "match_quality": (
                "high" if best_score >= 0.5
                else "medium" if best_score >= 0.35
                else "low"
            ),
        })

    pairs_df = pd.DataFrame(pairs)
    logger.info("=== RESULTS ===")
    logger.info("  Total queries: %d", len(mm_df))
    logger.info("  Matched pairs: %d (%.1f%%)", len(pairs_df), len(pairs_df)/len(mm_df)*100)
    logger.info("  Unmatched: %d", unmatched)

    if len(pairs_df) > 0:
        logger.info("  Mean similarity: %.4f", pairs_df["similarity_score"].mean())
        logger.info("  Median similarity: %.4f", pairs_df["similarity_score"].median())
        quality_counts = pairs_df["match_quality"].value_counts()
        for q in ["high", "medium", "low"]:
            logger.info("  %s quality: %d", q, quality_counts.get(q, 0))

        logger.info("\n  Per-condition breakdown:")
        for cond in sorted(pairs_df["condition_query"].unique()):
            sub = pairs_df[pairs_df["condition_query"] == cond]
            logger.info("    %s: %d pairs, avg_sim=%.3f", cond, len(sub), sub["similarity_score"].mean())

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mmcqsd_multicare_paired.csv"
    pairs_df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Saved pairs to %s", out_path)

    # Save summary
    summary_lines = [
        f"# Matching Summary",
        f"",
        f"- Queries: {len(mm_df)}",
        f"- Matched: {len(pairs_df)} ({len(pairs_df)/len(mm_df)*100:.1f}%)",
        f"- Unmatched: {unmatched}",
        f"- Mean similarity: {pairs_df['similarity_score'].mean():.4f}" if len(pairs_df) > 0 else "",
        f"- Median similarity: {pairs_df['similarity_score'].median():.4f}" if len(pairs_df) > 0 else "",
        f"",
        f"## Quality Distribution",
    ]
    if len(pairs_df) > 0:
        for q in ["high", "medium", "low"]:
            c = quality_counts.get(q, 0)
            summary_lines.append(f"- {q}: {c} ({c/len(pairs_df)*100:.1f}%)")

    (OUTPUT_DIR / "matching_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Done!")


if __name__ == "__main__":
    main()
