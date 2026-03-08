"""Evaluate H1 metrics from baseline outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT = "results/h1_outputs.csv"
DEFAULT_SCORED = "results/h1_scored.csv"
DEFAULT_SUMMARY = "results/h1_mvp_summary.md"

CONCEPT_PATTERNS = {
    "pleural_effusion": ["pleural effusion", "effusion", "fluid"],
    "pneumonia": ["pneumonia", "infect", "infection"],
    "consolidation": ["consolidation"],
    "cardiomegaly": ["cardiomegaly", "heart size", "cardiomediastinal"],
    "congestion": ["congestion", "edema", "vascular prominence"],
    "atelectasis": ["atelect", "collapse"],
    "opacity": ["opacity", "opacities", "infiltrate"],
    "no_acute": ["no acute", "normal", "lungs are clear", "no abnormality"],
}

POSITIVE_CONCEPTS = {
    "pleural_effusion",
    "pneumonia",
    "consolidation",
    "cardiomegaly",
    "congestion",
    "atelectasis",
    "opacity",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", str(text).lower()))


def _is_negated(lower_text: str, pattern: str) -> bool:
    escaped = re.escape(pattern)
    negation_patterns = [
        rf"\bno\b[\w\s\-]{{0,18}}\b{escaped}\b",
        rf"\bwithout\b[\w\s\-]{{0,18}}\b{escaped}\b",
        rf"\bnot\b[\w\s\-]{{0,18}}\b{escaped}\b",
    ]
    return any(re.search(regex, lower_text) for regex in negation_patterns)


def _extract_concepts(text: str) -> set[str]:
    lower = str(text).lower()
    concepts: set[str] = set()
    for concept, patterns in CONCEPT_PATTERNS.items():
        if concept == "no_acute":
            # Keep explicit no-acute style statements as separate concept.
            if any(pattern in lower for pattern in patterns):
                concepts.add(concept)
            continue

        # Positive concept should not be extracted if negated in text.
        matched_positive = False
        for pattern in patterns:
            if pattern in lower and not _is_negated(lower, pattern):
                matched_positive = True
                break
        if matched_positive:
            concepts.add(concept)
    return concepts


def _has_contradiction(output_concepts: set[str], evidence_concepts: set[str]) -> bool:
    output_no_acute = "no_acute" in output_concepts
    evidence_has_positive = len(evidence_concepts & POSITIVE_CONCEPTS) > 0
    return output_no_acute and evidence_has_positive


def factual_support_score(output: str, evidence: str) -> float:
    output_concepts = _extract_concepts(output)
    evidence_concepts = _extract_concepts(evidence)

    positive_claims = output_concepts & POSITIVE_CONCEPTS
    if not positive_claims:
        # For conservative statements with no specific claim, assign neutral low score.
        return 0.25

    supported = len(positive_claims & evidence_concepts)
    base_score = supported / max(len(positive_claims), 1)

    if _has_contradiction(output_concepts, evidence_concepts):
        return max(0.0, base_score - 0.5)
    return base_score


def hallucination_flag(output: str, evidence: str) -> int:
    output_concepts = _extract_concepts(output)
    evidence_concepts = _extract_concepts(evidence)
    unsupported_positive = (output_concepts & POSITIVE_CONCEPTS) - evidence_concepts
    contradiction = _has_contradiction(output_concepts, evidence_concepts)
    return int(bool(unsupported_positive) or contradiction)


def paired_test(grounded: np.ndarray, zero: np.ndarray) -> tuple[str, float, float]:
    diff = grounded - zero
    if len(diff) < 3:
        return ("insufficient_samples", float("nan"), float("nan"))
    _, p_normal = stats.shapiro(diff)
    if p_normal > 0.05:
        stat, p = stats.ttest_rel(grounded, zero)
        return ("paired_t_test", float(stat), float(p))
    stat, p = stats.wilcoxon(diff)
    return ("wilcoxon_signed_rank", float(stat), float(p))


def evaluate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    evidence_col = "top1_evidence_text" if "top1_evidence_text" in df.columns else "retrieved_evidence_text"

    df["zero_factual"] = df.apply(
        lambda r: factual_support_score(r["zero_shot_output"], r[evidence_col]),
        axis=1,
    )
    df["grounded_factual"] = df.apply(
        lambda r: factual_support_score(r["grounded_output"], r[evidence_col]),
        axis=1,
    )
    df["zero_hallucination"] = df.apply(
        lambda r: hallucination_flag(r["zero_shot_output"], r[evidence_col]),
        axis=1,
    )
    df["grounded_hallucination"] = df.apply(
        lambda r: hallucination_flag(r["grounded_output"], r[evidence_col]),
        axis=1,
    )

    if "retrieval_top1_hit" not in df.columns:
        df["retrieval_top1_hit"] = 0
    if "retrieval_topk_hit" not in df.columns:
        df["retrieval_topk_hit"] = 0

    factual_diff = (
        df["grounded_factual"].to_numpy(dtype=float)
        - df["zero_factual"].to_numpy(dtype=float)
    )
    test_name, stat, p_value = paired_test(
        df["grounded_factual"].to_numpy(dtype=float),
        df["zero_factual"].to_numpy(dtype=float),
    )
    effect_size = (
        float(np.mean(factual_diff) / np.std(factual_diff, ddof=1))
        if len(factual_diff) > 1 and np.std(factual_diff, ddof=1) > 0
        else float("nan")
    )
    if len(factual_diff) > 1:
        ci_low, ci_high = stats.t.interval(
            0.95,
            df=len(factual_diff) - 1,
            loc=np.mean(factual_diff),
            scale=stats.sem(factual_diff),
        )
    else:
        ci_low, ci_high = float("nan"), float("nan")

    metrics = {
        "samples": int(len(df)),
        "retrieval_top1_hit_rate": float(df["retrieval_top1_hit"].mean()),
        "retrieval_topk_hit_rate": float(df["retrieval_topk_hit"].mean()),
        "zero_factual_mean": float(df["zero_factual"].mean()),
        "grounded_factual_mean": float(df["grounded_factual"].mean()),
        "zero_hallucination_rate": float(df["zero_hallucination"].mean()),
        "grounded_hallucination_rate": float(df["grounded_hallucination"].mean()),
        "test_name": test_name,
        "test_statistic": float(stat) if not np.isnan(stat) else float("nan"),
        "p_value": float(p_value) if not np.isnan(p_value) else float("nan"),
        "effect_size_d": float(effect_size) if not np.isnan(effect_size) else float("nan"),
        "ci_low": float(ci_low) if not np.isnan(ci_low) else float("nan"),
        "ci_high": float(ci_high) if not np.isnan(ci_high) else float("nan"),
    }
    return df, metrics


def evaluate(input_path: Path, scored_path: Path, summary_path: Path) -> dict:
    df = pd.read_csv(input_path)
    scored_df, metrics = evaluate_dataframe(df)

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(scored_path, index=False, encoding="utf-8")

    summary = [
        "# H1 MVP Summary",
        "",
        f"- Samples evaluated: **{metrics['samples']}**",
        f"- Retrieval top-1 hit rate: **{metrics['retrieval_top1_hit_rate']:.3f}**",
        f"- Retrieval top-k hit rate: **{metrics['retrieval_topk_hit_rate']:.3f}**",
        f"- Mean factual score (zero-shot): **{metrics['zero_factual_mean']:.3f}**",
        f"- Mean factual score (grounded): **{metrics['grounded_factual_mean']:.3f}**",
        f"- Hallucination rate (zero-shot): **{metrics['zero_hallucination_rate']:.3f}**",
        f"- Hallucination rate (grounded): **{metrics['grounded_hallucination_rate']:.3f}**",
        "",
        "## Paired Comparison (H1)",
        f"- Test: **{metrics['test_name']}**",
        f"- Statistic: **{metrics['test_statistic']:.4f}**" if not np.isnan(metrics["test_statistic"]) else "- Statistic: **N/A**",
        f"- p-value: **{metrics['p_value']:.6f}**" if not np.isnan(metrics["p_value"]) else "- p-value: **N/A**",
        f"- Effect size (Cohen's d): **{metrics['effect_size_d']:.3f}**"
        if not np.isnan(metrics["effect_size_d"])
        else "- Effect size (Cohen's d): **N/A**",
        f"- 95% CI (grounded - zero-shot): **[{metrics['ci_low']:.3f}, {metrics['ci_high']:.3f}]**"
        if not np.isnan(metrics["ci_low"])
        else "- 95% CI (grounded - zero-shot): **N/A**",
        "",
        "## Preliminary Interpretation",
    ]

    if np.isnan(metrics["p_value"]):
        summary.append("- Not enough samples for statistical significance testing.")
    elif metrics["p_value"] < 0.05:
        summary.append("- Early results support H1: grounded outputs outperform zero-shot on factual support.")
    else:
        summary.append("- Directional trend exists, but statistical significance is not yet reached.")

    summary.append(
        "- This is an MVP result and should be interpreted with small-sample caution."
    )

    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Scored results: {scored_path}")
    print(f"Summary report: {summary_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate H1 from baseline outputs")
    parser.add_argument("--input-path", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--scored-path", type=Path, default=Path(DEFAULT_SCORED))
    parser.add_argument("--summary-path", type=Path, default=Path(DEFAULT_SUMMARY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.input_path, args.scored_path, args.summary_path)


if __name__ == "__main__":
    main()

