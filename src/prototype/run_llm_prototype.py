"""H1 Prototype with real LLM generation via Groq API.

Samples ~300 pairs across all 18 conditions, runs each through
Llama-3.3-70B in two modes:
- Zero-shot: query only (no evidence)
- Grounded: query + retrieved MultiCaRe evidence
Then evaluates factual support, hallucination, and statistical tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
OUTPUT_DIR = Path("results/multicare_h1_llm")
MODEL = "llama-3.1-8b-instant"  # 500K tokens/day free tier (vs 100K for 70B)
SAMPLE_SIZE = 300
MAX_EVIDENCE_WORDS = 400
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 2.0  # seconds — stay under 30 req/min
INPUT_SAMPLE: Path | None = None  # pre-built sample CSV (skips internal sampling)

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
                neg_patterns = [
                    rf"\bno\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bwithout\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bnot\b[\w\s\-]{{0,18}}\b{escaped}\b",
                    rf"\bnahi\b[\w\s\-]{{0,18}}\b{escaped}\b",
                ]
                if not any(re.search(r, lower) for r in neg_patterns):
                    concepts.add(concept)
                    break
    return concepts


def factual_support_score(output: str, evidence: str) -> float:
    out_concepts = extract_positive_concepts(output)
    ev_concepts = extract_positive_concepts(evidence)
    if not out_concepts:
        return 0.25
    return len(out_concepts & ev_concepts) / len(out_concepts)


def hallucination_score(output: str, evidence: str) -> float:
    out_concepts = extract_positive_concepts(output)
    ev_concepts = extract_positive_concepts(evidence)
    if not out_concepts:
        return 0.0
    unsupported = out_concepts - ev_concepts
    return len(unsupported) / len(out_concepts)


def sample_pairs(df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = 42) -> pd.DataFrame:
    """Proportionally sample across conditions."""
    rng = np.random.RandomState(seed)
    condition_counts = df["condition_query"].value_counts()
    total = condition_counts.sum()

    sampled = []
    for cond, count in condition_counts.items():
        target = max(3, int(n * count / total))
        available = df[df["condition_query"] == cond]
        chosen = available.sample(n=min(target, len(available)), random_state=rng)
        sampled.append(chosen)

    result = pd.concat(sampled, ignore_index=True)
    if len(result) > n:
        result = result.sample(n=n, random_state=rng).reset_index(drop=True)
    logger.info("Sampled %d pairs across %d conditions", len(result), result["condition_query"].nunique())
    return result


def call_groq(client: Groq, system: str, user_msg: str) -> str:
    """Call Groq API with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Groq API error (attempt %d): %s", attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
    return "[API_ERROR]"


def build_grounded_prompt(query: str, evidence: str) -> str:
    evidence_truncated = " ".join(evidence.split()[:MAX_EVIDENCE_WORDS])
    return (
        f"Clinical Evidence:\n{evidence_truncated}\n\n"
        f"Patient Query:\n{query}\n\n"
        f"Respond based strictly on the clinical evidence above."
    )


def build_zero_shot_prompt(query: str) -> str:
    return (
        f"Patient Query:\n{query}\n\n"
        f"Respond based on your general medical knowledge only."
    )


def run_llm_prototype(pairs_path: Path, output_dir: Path, sample_size: int,
                      input_sample: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if input_sample and input_sample.exists():
        sampled = pd.read_csv(input_sample)
        logger.info("Loaded pre-built sample: %d pairs from %s", len(sampled), input_sample)
    else:
        df = pd.read_csv(pairs_path)
        logger.info("Loaded %d pairs", len(df))
        sampled = sample_pairs(df, n=sample_size)

    # Check for existing progress
    progress_path = output_dir / "progress.json"
    scored_partial = output_dir / "h1_llm_scored_partial.csv"
    start_idx = 0
    records: list[dict] = []

    if scored_partial.exists():
        existing = pd.read_csv(scored_partial)
        records = existing.to_dict("records")
        start_idx = len(records)
        logger.info("Resuming from index %d (%d already done)", start_idx, len(records))

    total_calls = (len(sampled) - start_idx) * 2
    logger.info("Need %d API calls (%d pairs × 2 modes)", total_calls, len(sampled) - start_idx)

    for idx in range(start_idx, len(sampled)):
        row = sampled.iloc[idx]
        query = str(row["hinglish_query"])
        evidence = str(row["evidence_text"])
        condition = str(row.get("condition_query", ""))

        logger.info("[%d/%d] %s — generating zero-shot...", idx + 1, len(sampled), condition)
        zero_out = call_groq(client, SYSTEM_PROMPT_ZERO_SHOT, build_zero_shot_prompt(query))
        time.sleep(DELAY_BETWEEN_CALLS)

        logger.info("[%d/%d] %s — generating grounded...", idx + 1, len(sampled), condition)
        grounded_out = call_groq(client, SYSTEM_PROMPT_GROUNDED, build_grounded_prompt(query, evidence))
        time.sleep(DELAY_BETWEEN_CALLS)

        z_factual = factual_support_score(zero_out, evidence)
        g_factual = factual_support_score(grounded_out, evidence)
        z_halluc = hallucination_score(zero_out, evidence)
        g_halluc = hallucination_score(grounded_out, evidence)

        record = {
            "pair_id": row["pair_id"],
            "condition": condition,
            "similarity_score": float(row.get("similarity_score", 0)),
            "match_quality": row.get("match_quality", ""),
            "hinglish_query": query,
            "zero_shot_output": zero_out,
            "grounded_output": grounded_out,
            "zero_factual": z_factual,
            "grounded_factual": g_factual,
            "zero_hallucination": z_halluc,
            "grounded_hallucination": g_halluc,
            "factual_gain": g_factual - z_factual,
            "halluc_reduction": z_halluc - g_halluc,
        }
        if "cmi_score" in row.index:
            record["cmi_score"] = float(row["cmi_score"])
        if "cmi_bucket" in row.index:
            record["cmi_bucket"] = str(row["cmi_bucket"])
        records.append(record)

        # Save progress every 10 pairs
        if (idx + 1) % 10 == 0:
            pd.DataFrame(records).to_csv(scored_partial, index=False, encoding="utf-8")
            logger.info("  Progress saved (%d/%d)", idx + 1, len(sampled))

    result_df = pd.DataFrame(records)
    final_path = output_dir / "h1_llm_scored.csv"
    result_df.to_csv(final_path, index=False, encoding="utf-8")

    # Compute metrics and generate report
    metrics = compute_metrics(result_df)
    summary = generate_report(result_df, metrics)
    summary_path = output_dir / "h1_llm_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    logger.info("Done! Results: %s", final_path)
    logger.info("Summary: %s", summary_path)
    print()
    print(summary)

    return metrics


def compute_metrics(df: pd.DataFrame) -> dict:
    df = df[df["zero_shot_output"] != "[API_ERROR]"]
    df = df[df["grounded_output"] != "[API_ERROR]"]

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

    ci_low, ci_high = float("nan"), float("nan")
    if len(diff) > 1:
        ci_low, ci_high = stats.t.interval(
            0.95, df=len(diff) - 1,
            loc=np.mean(diff), scale=stats.sem(diff),
        )

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
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def generate_report(df: pd.DataFrame, m: dict) -> str:
    sig = "NOT significant"
    if not np.isnan(m["p_value"]) and m["p_value"] < 0.001:
        sig = "HIGHLY significant (p < 0.001)"
    elif not np.isnan(m["p_value"]) and m["p_value"] < 0.05:
        sig = "Significant (p < 0.05)"

    eff = "N/A"
    if not np.isnan(m["effect_d"]):
        d = abs(m["effect_d"])
        eff = f"{'Large' if d >= 0.8 else 'Medium' if d >= 0.5 else 'Small' if d >= 0.2 else 'Negligible'} ({m['effect_d']:.3f})"

    lines = [
        "# H1 Results: Real LLM Generation (Groq)",
        "",
        "## Setup",
        f"- Generator: **Llama-3.1-8B-Instant** via Groq API (pairs 1-70: Llama-3.3-70B)",
        f"- Evidence: **MultiCaRe** raw clinical cases (no LLM extraction)",
        f"- Queries: **MMCQSD** Hinglish patient queries",
        f"- Matching: **LaBSE** + FAISS",
        f"- Evaluated pairs: **{m['n']}** (proportionally sampled across 18 conditions)",
        "",
        "## Key Results",
        "",
        "| Metric | Zero-Shot | Grounded | Delta |",
        "|---|---:|---:|---:|",
        f"| **Factual support** | {m['zero_factual']:.4f} | {m['grounded_factual']:.4f} | **{m['factual_gain']:+.4f}** |",
        f"| **Hallucination score** | {m['zero_halluc']:.4f} | {m['grounded_halluc']:.4f} | **{m['halluc_reduction']:+.4f}** |",
        "",
        "## Statistical Significance (H1)",
        f"- Test: **{m['test_name']}**",
    ]

    if not np.isnan(m["test_stat"]):
        lines.append(f"- Statistic: **{m['test_stat']:.4f}**")
    if not np.isnan(m["p_value"]):
        lines.append(f"- p-value: **{m['p_value']:.2e}**")
    lines.append(f"- Effect size (Cohen's d): **{eff}**")
    if not np.isnan(m["ci_low"]):
        lines.append(f"- 95% CI for factual gain: **[{m['ci_low']:.4f}, {m['ci_high']:.4f}]**")
    lines.append(f"- Verdict: **{sig}**")

    lines.extend(["", "## Per-Condition Results", "",
        "| Condition | N | Zero Factual | Grounded Factual | Gain | Zero Halluc | Grounded Halluc |",
        "|---|---:|---:|---:|---:|---:|---:|"])

    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        lines.append(
            f"| {cond} | {len(sub)} | "
            f"{sub['zero_factual'].mean():.3f} | "
            f"{sub['grounded_factual'].mean():.3f} | "
            f"{sub['grounded_factual'].mean() - sub['zero_factual'].mean():+.3f} | "
            f"{sub['zero_hallucination'].mean():.3f} | "
            f"{sub['grounded_hallucination'].mean():.3f} |"
        )

    lines.extend(["", "## Sample Outputs", ""])
    for _, row in df.head(3).iterrows():
        lines.extend([
            f"### {row['condition']} (sim={row['similarity_score']:.3f})",
            f"**Query**: {row['hinglish_query'][:150]}...",
            f"**Zero-shot**: {row['zero_shot_output'][:200]}...",
            f"**Grounded**: {row['grounded_output'][:200]}...",
            f"- Factual: {row['zero_factual']:.2f} → {row['grounded_factual']:.2f} | "
            f"Halluc: {row['zero_hallucination']:.2f} → {row['grounded_hallucination']:.2f}",
            "",
        ])

    lines.extend([
        "---",
        f"*Generated using Llama-3.3-70B via Groq API on {m['n']} sampled pairs*",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run H1 with real LLM via Groq")
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--input-sample", type=Path, default=None,
                        help="Pre-built sample CSV (skips internal sampling)")
    args = parser.parse_args()
    run_llm_prototype(args.pairs, args.output_dir, args.sample_size,
                      input_sample=args.input_sample)


if __name__ == "__main__":
    main()
