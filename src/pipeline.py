"""
RAG Pipeline — end-to-end query -> retrieve -> generate.

Wires together the TextEncoder, FAISSIndexer, EvidenceRetriever,
and GroundedGenerator into a single callable pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.encoding.text_encoder import TextEncoder
from src.generation.generator import GroundedGenerator
from src.retrieval.retriever import EvidenceRetriever

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/faiss_index")

MEDICAL_CONCEPT_PATTERNS: dict[str, list[str]] = {
    "rash": ["rash", "rashes", "eruption", "exanthem", "maculopapular"],
    "dermatitis": ["dermatitis", "eczema", "dermatologic"],
    "lesion": ["lesion", "lesions", "papule", "nodule", "plaque", "macule"],
    "ulcer": ["ulcer", "ulceration", "ulcerative", "aphthous"],
    "swelling": ["swelling", "swollen", "edema", "oedema", "tumefaction", "enlargement"],
    "inflammation": ["inflammation", "inflammatory", "inflamed", "cellulitis"],
    "infection": ["infection", "infected", "infectious", "abscess", "sepsis", "septic"],
    "fever": ["fever", "febrile", "pyrexia", "hyperthermia"],
    "pain": ["pain", "painful", "tenderness", "tender", "ache", "algia"],
    "erythema": ["erythema", "erythematous", "redness", "red"],
    "pruritus": ["pruritus", "pruritic", "itching", "itchy", "itch"],
    "mass": ["mass", "lump", "tumor", "tumour", "growth", "neoplasm"],
    "fracture": ["fracture", "fractured", "broken"],
    "effusion": ["effusion", "fluid collection"],
    "cyanosis": ["cyanosis", "cyanotic", "bluish"],
    "necrosis": ["necrosis", "necrotic", "gangrene"],
    "malignancy": ["malignant", "malignancy", "carcinoma", "cancer", "sarcoma"],
    "allergy": ["allergy", "allergic", "hypersensitivity", "urticaria", "angioedema"],
}

POSITIVE_CONCEPTS = {
    "rash", "dermatitis", "lesion", "ulcer", "swelling", "inflammation",
    "infection", "fever", "pain", "erythema", "pruritus", "mass",
    "fracture", "effusion", "cyanosis", "necrosis", "malignancy", "allergy",
}


def _extract_concepts(text: str) -> set[str]:
    lower = str(text).lower()
    found: set[str] = set()
    for concept, patterns in MEDICAL_CONCEPT_PATTERNS.items():
        if concept not in POSITIVE_CONCEPTS:
            continue
        for pat in patterns:
            if pat in lower:
                escaped = re.escape(pat)
                neg = [
                    rf"\bno\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bwithout\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bnot\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bnahi\b[\w\s\-]{{0,18}}\b{escaped}\b",
                ]
                if not any(re.search(r, lower) for r in neg):
                    found.add(concept)
                    break
    return found


def factual_support_score(output: str, evidence: str) -> float:
    out_c = _extract_concepts(output)
    ev_c = _extract_concepts(evidence)
    if not out_c:
        return 0.25
    return len(out_c & ev_c) / len(out_c)


def hallucination_score(output: str, evidence: str) -> float:
    out_c = _extract_concepts(output)
    ev_c = _extract_concepts(evidence)
    if not out_c:
        return 0.0
    return len(out_c - ev_c) / len(out_c)


class RAGPipeline:
    """End-to-end RAG pipeline: query -> retrieve -> generate.

    Parameters
    ----------
    index_dir : Path
        Directory containing evidence.index and evidence_metadata.csv.
    api_key : str
        Groq API key for generation.
    max_k : int
        Maximum retrieval candidates.
    model_name : str
        Groq model identifier.
    """

    def __init__(
        self,
        api_key: str,
        index_dir: Path = DEFAULT_INDEX_DIR,
        max_k: int = 5,
        model_name: str = "llama-3.1-8b-instant",
    ):
        index_path = index_dir / "evidence.index"
        metadata_path = index_dir / "evidence_metadata.csv"

        logger.info("Initializing RAG pipeline ...")
        self.encoder = TextEncoder(device="cpu")

        self.retriever = EvidenceRetriever.from_disk(
            index_path=index_path,
            metadata_path=metadata_path,
            text_encoder=self.encoder,
            max_k=max_k,
        )

        self.generator = GroundedGenerator(
            api_key=api_key,
            model_name=model_name,
        )
        logger.info("RAG pipeline ready.")

    def query(
        self,
        hinglish_query: str,
        top_k: int | None = None,
        include_zero_shot: bool = True,
    ) -> dict:
        """Run the full RAG pipeline on a Hinglish query.

        Parameters
        ----------
        hinglish_query : str
            Patient query in Hinglish.
        top_k : int | None
            Fixed number of evidence items to retrieve.
            None = use adaptive truncation.
        include_zero_shot : bool
            Whether to also generate a zero-shot (no evidence) response.

        Returns
        -------
        dict
            query, retrieved_evidence, grounded_response, scores, and
            optionally zero_shot_response.
        """
        evidence_items = self.retriever.retrieve(hinglish_query, top_k=top_k)

        evidence_texts = [item["case_text"] for item in evidence_items]
        grounded = self.generator.generate(hinglish_query, evidence_texts)

        combined_evidence = " ".join(evidence_texts)
        g_factual = factual_support_score(grounded, combined_evidence)
        g_halluc = hallucination_score(grounded, combined_evidence)

        result = {
            "query": hinglish_query,
            "retrieved_evidence": evidence_items,
            "grounded_response": grounded,
            "grounded_factual_score": g_factual,
            "grounded_hallucination_score": g_halluc,
        }

        if include_zero_shot:
            zero_shot = self.generator.generate_zero_shot(hinglish_query)
            z_factual = factual_support_score(zero_shot, combined_evidence)
            z_halluc = hallucination_score(zero_shot, combined_evidence)
            result.update({
                "zero_shot_response": zero_shot,
                "zero_shot_factual_score": z_factual,
                "zero_shot_hallucination_score": z_halluc,
            })

        return result
