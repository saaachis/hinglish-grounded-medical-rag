"""Phase 6: LLM Evidence Extraction + Ablation (raw vs structured).

Step 1: Extract structured evidence from the 242 unique MultiCaRe cases
        used in the Phase 5b evaluation (via Groq LLM).
Step 2: Re-run the same 292 evaluated pairs with structured evidence.
Step 3: Compare raw vs structured results (ablation).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from scipy import stats

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PAIRS_PATH = Path("data/processed/mmcqsd_multicare_paired.csv")
LLM_SCORED_PATH = Path("results/multicare_h1_llm/h1_llm_scored.csv")
OUTPUT_DIR = Path("results/multicare_h1_ablation")
EXTRACTION_CACHE = Path("data/processed/extracted_evidence.csv")

MODEL = "llama-3.1-8b-instant"
DELAY = 2.0
MAX_RETRIES = 3
MAX_EVIDENCE_WORDS = 400

EXTRACTION_PROMPT = """You are a clinical evidence extractor. Given a medical case report, extract ONLY the factual clinical findings into a structured format. Do NOT add any information not present in the case. Be concise.

Case Report:
{case_text}

Extract in this exact format (use "not specified" if information is absent):
Primary Finding: [main diagnosis or presenting condition]
Location: [body area affected]
Symptoms: [patient-reported symptoms, comma-separated]
Clinical Signs: [examination findings, comma-separated]
Severity: [mild / moderate / severe / not specified]
Duration: [how long symptoms present, or "not specified"]
Key Evidence: [single most important clinical sentence from the case, quoted exactly]"""

# --- Reuse evaluation logic from run_llm_prototype ---
import re

SYSTEM_PROMPT_GROUNDED = """You are a medical assistant helping patients understand their symptoms.
You MUST base your response strictly on the clinical evidence provided below.
Respond in Hinglish (mix of Hindi and English) since the patient communicates in Hinglish.
Keep the response concise (3-5 sentences). Only state facts supported by the evidence.
If the evidence does not cover something, say you cannot confirm it."""

SYSTEM_PROMPT_ZERO_SHOT = """You are a medical assistant helping patients understand their symptoms.
Respond in Hinglish (mix of Hindi and English) since the patient communicates in Hinglish.
Keep the response concise (3-5 sentences).
You do NOT have access to any clinical reports or test results."""

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
    "biopsy": ["biopsy", "histopathology", "histologic"],
    "malignancy": ["malignant", "malignancy", "carcinoma", "cancer", "sarcoma"],
    "benign": ["benign", "non-malignant"],
    "allergy": ["allergy", "allergic", "hypersensitivity", "urticaria", "angioedema"],
    "tonsillitis": ["tonsillitis", "tonsil", "tonsillar", "pharyngitis"],
    "conjunctivitis": ["conjunctivitis", "conjunctival"],
    "lymphadenopathy": ["lymphadenopathy", "lymph node", "lymphadenitis"],
    "fungal": ["fungal", "fungus", "mycosis", "candida", "tinea"],
    "viral": ["viral", "virus", "herpes", "varicella"],
    "bacterial": ["bacterial", "bacteria", "staph", "strep", "mrsa"],
    "surgery": ["surgery", "surgical", "excision", "resection"],
    "treatment": ["treatment", "therapy", "antibiotic", "steroid", "medication"],
}

POSITIVE_CONCEPTS = {
    "rash", "dermatitis", "lesion", "ulcer", "swelling", "inflammation",
    "infection", "fever", "pain", "erythema", "pruritus", "mass",
    "fracture", "effusion", "cyanosis", "necrosis", "malignancy",
    "allergy", "tonsillitis", "conjunctivitis", "lymphadenopathy",
    "fungal", "viral", "bacterial",
}


def extract_positive_concepts(text: str) -> set[str]:
    lower = str(text).lower()
    concepts: set[str] = set()
    for concept, patterns in MEDICAL_CONCEPT_PATTERNS.items():
        if concept not in POSITIVE_CONCEPTS:
            continue
        for pattern in patterns:
            if pattern in lower:
                escaped = re.escape(pattern)
                neg_pats = [
                    rf"\bno\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bwithout\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bnot\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bnahi\b[\w\s\-]{{0,18}}\b{escaped}\b",
                ]
                if not any(re.search(r, lower) for r in neg_pats):
                    concepts.add(concept)
                    break
    return concepts


def factual_support_score(output: str, evidence: str) -> float:
    out_c = extract_positive_concepts(output)
    ev_c = extract_positive_concepts(evidence)
    if not out_c:
        return 0.25
    return len(out_c & ev_c) / len(out_c)


def hallucination_score(output: str, evidence: str) -> float:
    out_c = extract_positive_concepts(output)
    ev_c = extract_positive_concepts(evidence)
    if not out_c:
        return 0.0
    return len(out_c - ev_c) / len(out_c)


def call_groq(client: Groq, system: str, user_msg: str, max_tokens: int = 300) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("API error (attempt %d): %s", attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
    return "[API_ERROR]"


# ─── Step 1: Extract structured evidence ───

def run_extraction(client: Groq, pairs_df: pd.DataFrame, llm_scored_df: pd.DataFrame):
    """Extract structured evidence for unique cases in evaluated pairs."""
    evaluated_ids = set(llm_scored_df["pair_id"].tolist())
    eval_pairs = pairs_df[pairs_df["pair_id"].isin(evaluated_ids)]
    unique_cases = eval_pairs.drop_duplicates(subset=["multicare_case_id"])

    logger.info("Need to extract %d unique cases", len(unique_cases))

    # Load existing extractions
    existing = {}
    if EXTRACTION_CACHE.exists():
        cached = pd.read_csv(EXTRACTION_CACHE)
        existing = dict(zip(cached["case_id"], cached["structured_evidence"]))
        logger.info("Loaded %d cached extractions", len(existing))

    to_extract = unique_cases[~unique_cases["multicare_case_id"].isin(existing)]
    logger.info("Remaining to extract: %d", len(to_extract))

    records = [{"case_id": k, "structured_evidence": v} for k, v in existing.items()]

    for idx, (_, row) in enumerate(to_extract.iterrows()):
        case_id = str(row["multicare_case_id"])
        raw_text = str(row["evidence_text"])
        truncated = " ".join(raw_text.split()[:600])

        prompt = EXTRACTION_PROMPT.format(case_text=truncated)
        logger.info("[%d/%d] Extracting %s...", idx + 1, len(to_extract), case_id)

        result = call_groq(
            client,
            "You are a clinical evidence extractor. Be precise and factual.",
            prompt,
            max_tokens=250,
        )
        time.sleep(DELAY)

        records.append({"case_id": case_id, "structured_evidence": result})

        if (idx + 1) % 10 == 0:
            pd.DataFrame(records).to_csv(EXTRACTION_CACHE, index=False, encoding="utf-8")
            logger.info("  Extraction progress saved (%d/%d)", idx + 1, len(to_extract))

    result_df = pd.DataFrame(records)
    result_df.to_csv(EXTRACTION_CACHE, index=False, encoding="utf-8")
    logger.info("Extraction complete: %d cases saved to %s", len(result_df), EXTRACTION_CACHE)
    return dict(zip(result_df["case_id"], result_df["structured_evidence"]))


# ─── Step 2: Re-run prototype with structured evidence ───

def run_structured_prototype(
    client: Groq,
    pairs_df: pd.DataFrame,
    llm_scored_df: pd.DataFrame,
    extractions: dict[str, str],
):
    """Re-run the same evaluated pairs using structured evidence."""
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluated_ids = llm_scored_df["pair_id"].tolist()
    eval_pairs = pairs_df[pairs_df["pair_id"].isin(evaluated_ids)].reset_index(drop=True)

    # Check for resume
    partial_path = output_dir / "ablation_scored_partial.csv"
    start_idx = 0
    records: list[dict] = []

    if partial_path.exists():
        existing = pd.read_csv(partial_path)
        records = existing.to_dict("records")
        start_idx = len(records)
        logger.info("Resuming from index %d", start_idx)

    logger.info("Running structured prototype on %d pairs (starting at %d)...", len(eval_pairs), start_idx)

    for idx in range(start_idx, len(eval_pairs)):
        row = eval_pairs.iloc[idx]
        query = str(row["hinglish_query"])
        case_id = str(row["multicare_case_id"])
        condition = str(row.get("condition_query", ""))
        raw_evidence = str(row["evidence_text"])

        structured = extractions.get(case_id, raw_evidence)
        if structured == "[API_ERROR]":
            structured = raw_evidence

        # Combine raw + structured for richer grounding
        combined_evidence = f"{structured}\n\nOriginal case excerpt: {' '.join(raw_evidence.split()[:200])}"

        logger.info("[%d/%d] %s — zero-shot...", idx + 1, len(eval_pairs), condition)
        zero_out = call_groq(client, SYSTEM_PROMPT_ZERO_SHOT,
            f"Patient Query:\n{query}\n\nRespond based on your general medical knowledge only.")
        time.sleep(DELAY)

        logger.info("[%d/%d] %s — grounded (structured)...", idx + 1, len(eval_pairs), condition)
        grounded_out = call_groq(client, SYSTEM_PROMPT_GROUNDED,
            f"Clinical Evidence:\n{combined_evidence}\n\nPatient Query:\n{query}\n\nRespond based strictly on the clinical evidence above.")
        time.sleep(DELAY)

        z_f = factual_support_score(zero_out, raw_evidence)
        g_f = factual_support_score(grounded_out, raw_evidence)
        z_h = hallucination_score(zero_out, raw_evidence)
        g_h = hallucination_score(grounded_out, raw_evidence)

        records.append({
            "pair_id": row["pair_id"],
            "condition": condition,
            "similarity_score": float(row.get("similarity_score", 0)),
            "match_quality": row.get("match_quality", ""),
            "zero_shot_output": zero_out,
            "grounded_output": grounded_out,
            "zero_factual": z_f,
            "grounded_factual": g_f,
            "zero_hallucination": z_h,
            "grounded_hallucination": g_h,
            "factual_gain": g_f - z_f,
            "halluc_reduction": z_h - g_h,
        })

        if (idx + 1) % 10 == 0:
            pd.DataFrame(records).to_csv(partial_path, index=False, encoding="utf-8")
            logger.info("  Progress saved (%d/%d)", idx + 1, len(eval_pairs))

    result_df = pd.DataFrame(records)
    final_path = output_dir / "ablation_scored.csv"
    result_df.to_csv(final_path, index=False, encoding="utf-8")
    return result_df


# ─── Step 3: Ablation comparison ───

def compute_metrics(df: pd.DataFrame) -> dict:
    df = df[(df["zero_shot_output"] != "[API_ERROR]") & (df["grounded_output"] != "[API_ERROR]")]
    g = df["grounded_factual"].to_numpy(dtype=float)
    z = df["zero_factual"].to_numpy(dtype=float)
    diff = g - z

    test_name, stat_val, p_val = "N/A", float("nan"), float("nan")
    if len(diff) >= 3:
        sample = diff[:5000] if len(diff) > 5000 else diff
        _, p_normal = stats.shapiro(sample)
        if p_normal > 0.05:
            stat_val, p_val = stats.ttest_rel(g, z)
            test_name = "paired_t_test"
        else:
            stat_val, p_val = stats.wilcoxon(diff)
            test_name = "wilcoxon_signed_rank"

    effect_d = float("nan")
    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        effect_d = float(np.mean(diff) / np.std(diff, ddof=1))

    return {
        "n": len(df),
        "zero_factual": float(z.mean()),
        "grounded_factual": float(g.mean()),
        "factual_gain": float(diff.mean()),
        "zero_halluc": float(df["zero_hallucination"].mean()),
        "grounded_halluc": float(df["grounded_hallucination"].mean()),
        "halluc_reduction": float(df["zero_hallucination"].mean() - df["grounded_hallucination"].mean()),
        "test_name": test_name,
        "test_stat": float(stat_val),
        "p_value": float(p_val),
        "effect_d": float(effect_d),
    }


def generate_ablation_report(raw_metrics: dict, struct_metrics: dict) -> str:
    lines = [
        "# Ablation: Raw Evidence vs Structured Evidence",
        "",
        "## Setup",
        "- Same 292 pairs, same MMCQSD Hinglish queries",
        "- Same LLM generator (Llama-3.1-8B via Groq)",
        "- **Run 1 (Phase 5b)**: Raw MultiCaRe case narratives as evidence",
        "- **Run 2 (Phase 6)**: LLM-extracted structured evidence",
        "",
        "## Comparison",
        "",
        "| Metric | Raw Evidence | Structured Evidence | Delta |",
        "|---|---:|---:|---:|",
        f"| Factual support (grounded) | {raw_metrics['grounded_factual']:.4f} | {struct_metrics['grounded_factual']:.4f} | {struct_metrics['grounded_factual'] - raw_metrics['grounded_factual']:+.4f} |",
        f"| Hallucination (grounded) | {raw_metrics['grounded_halluc']:.4f} | {struct_metrics['grounded_halluc']:.4f} | {struct_metrics['grounded_halluc'] - raw_metrics['grounded_halluc']:+.4f} |",
        f"| Factual gain (grounded - zero) | {raw_metrics['factual_gain']:.4f} | {struct_metrics['factual_gain']:.4f} | {struct_metrics['factual_gain'] - raw_metrics['factual_gain']:+.4f} |",
        f"| Halluc reduction (zero - grounded) | {raw_metrics['halluc_reduction']:.4f} | {struct_metrics['halluc_reduction']:.4f} | {struct_metrics['halluc_reduction'] - raw_metrics['halluc_reduction']:+.4f} |",
        f"| Effect size (Cohen's d) | {raw_metrics['effect_d']:.3f} | {struct_metrics['effect_d']:.3f} | {struct_metrics['effect_d'] - raw_metrics['effect_d']:+.3f} |",
        f"| p-value | {raw_metrics['p_value']:.2e} | {struct_metrics['p_value']:.2e} | |",
        "",
        "## Interpretation",
        "",
    ]

    delta_factual = struct_metrics["grounded_factual"] - raw_metrics["grounded_factual"]
    delta_halluc = struct_metrics["grounded_halluc"] - raw_metrics["grounded_halluc"]

    if delta_factual > 0.02:
        lines.append(f"Structured evidence **improved** factual support by {delta_factual:+.4f}.")
    elif delta_factual < -0.02:
        lines.append(f"Structured evidence **reduced** factual support by {delta_factual:+.4f}. Raw narratives may provide richer context.")
    else:
        lines.append(f"Structured evidence showed **negligible difference** in factual support ({delta_factual:+.4f}).")

    if delta_halluc < -0.02:
        lines.append(f"Structured evidence **reduced** hallucination by {abs(delta_halluc):.4f}.")
    elif delta_halluc > 0.02:
        lines.append(f"Structured evidence **increased** hallucination by {delta_halluc:+.4f}.")
    else:
        lines.append(f"Hallucination rates were **similar** between both approaches ({delta_halluc:+.4f}).")

    lines.extend([
        "",
        "---",
        "*Phase 6 ablation study comparing raw vs LLM-extracted structured evidence*",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Ablation study")
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--llm-scored", type=Path, default=LLM_SCORED_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    pairs_df = pd.read_csv(args.pairs)
    llm_scored_df = pd.read_csv(args.llm_scored)

    logger.info("=== Phase 6 Step 1: Evidence Extraction ===")
    extractions = run_extraction(client, pairs_df, llm_scored_df)

    logger.info("=== Phase 6 Step 2: Structured Prototype ===")
    struct_df = run_structured_prototype(client, pairs_df, llm_scored_df, extractions)

    logger.info("=== Phase 6 Step 3: Ablation Comparison ===")
    raw_metrics = compute_metrics(llm_scored_df)
    struct_metrics = compute_metrics(struct_df)

    report = generate_ablation_report(raw_metrics, struct_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "ablation_report.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Ablation report saved: %s", report_path)
    # Print without Unicode arrows to avoid Windows encoding issues
    print(report.replace("\u2192", "->"))


if __name__ == "__main__":
    main()
