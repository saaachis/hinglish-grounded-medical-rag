"""Evaluate real dataset comparison outputs with generic text metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT = "results/dataset_comparison/openi/h1_outputs.csv"
DEFAULT_SCORED = "results/dataset_comparison/openi/h1_scored.csv"
DEFAULT_SUMMARY = "results/dataset_comparison/openi/h1_summary.md"
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
IGNORE_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "only",
    "query",
    "without",
    "external",
    "evidence",
    "tentative",
    "answer",
    "possible",
    "response",
    "should",
    "treated",
    "unverified",
    "available",
    "therefore",
    "according",
    "based",
    "report",
    "grounded",
    "explanation",
    "hisaab",
    "ke",
    "hai",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(str(text).lower())
        if len(token) > 2 and token not in IGNORE_TOKENS
    ]


def _precision_against_reference(output: str, reference: str) -> float:
    out_tokens = _tokens(output)
    ref_tokens = set(_tokens(reference))
    if not out_tokens:
        return 0.0
    matched = sum(1 for token in out_tokens if token in ref_tokens)
    return matched / len(out_tokens)


def _support_ratio(output: str, evidence: str) -> float:
    return _precision_against_reference(output, evidence)


def factual_score(output: str, reference: str, evidence: str) -> float:
    reference_precision = _precision_against_reference(output, reference)
    evidence_support = _support_ratio(output, evidence)
    return 0.6 * reference_precision + 0.4 * evidence_support


def hallucination_flag(output: str, reference: str, evidence: str) -> int:
    reference_precision = _precision_against_reference(output, reference)
    evidence_support = _support_ratio(output, evidence)
    unsupported_ratio = 1.0 - max(reference_precision, evidence_support)
    return int(unsupported_ratio > 0.5)


def paired_test(grounded: np.ndarray, zero: np.ndarray) -> tuple[str, float, float]:
    diff = grounded - zero
    if len(diff) < 3:
        return ("insufficient_samples", float("nan"), float("nan"))
    _, p_normal = stats.shapiro(diff)
    if p_normal > 0.05:
        stat, p_value = stats.ttest_rel(grounded, zero)
        return ("paired_t_test", float(stat), float(p_value))
    stat, p_value = stats.wilcoxon(diff)
    return ("wilcoxon_signed_rank", float(stat), float(p_value))


def evaluate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    evidence_col = "top1_evidence_text" if "top1_evidence_text" in df.columns else "retrieved_evidence_text"
    reference_col = "target_text"

    df["zero_factual"] = df.apply(
        lambda r: factual_score(r["zero_shot_output"], r[reference_col], r[evidence_col]),
        axis=1,
    )
    df["grounded_factual"] = df.apply(
        lambda r: factual_score(r["grounded_output"], r[reference_col], r[evidence_col]),
        axis=1,
    )
    df["zero_hallucination"] = df.apply(
        lambda r: hallucination_flag(r["zero_shot_output"], r[reference_col], r[evidence_col]),
        axis=1,
    )
    df["grounded_hallucination"] = df.apply(
        lambda r: hallucination_flag(r["grounded_output"], r[reference_col], r[evidence_col]),
        axis=1,
    )

    factual_diff = df["grounded_factual"].to_numpy(dtype=float) - df["zero_factual"].to_numpy(dtype=float)
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

    lines = [
        "# Real Dataset Comparison Summary",
        "",
        f"- Samples evaluated: **{metrics['samples']}**",
        f"- Retrieval top-1 hit rate: **{metrics['retrieval_top1_hit_rate']:.3f}**",
        f"- Retrieval top-k hit rate: **{metrics['retrieval_topk_hit_rate']:.3f}**",
        f"- Mean factual score (zero-shot): **{metrics['zero_factual_mean']:.3f}**",
        f"- Mean factual score (grounded): **{metrics['grounded_factual_mean']:.3f}**",
        f"- Hallucination rate (zero-shot): **{metrics['zero_hallucination_rate']:.3f}**",
        f"- Hallucination rate (grounded): **{metrics['grounded_hallucination_rate']:.3f}**",
        f"- Test: **{metrics['test_name']}**",
        f"- p-value: **{metrics['p_value']:.6f}**" if not np.isnan(metrics["p_value"]) else "- p-value: **N/A**",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real dataset comparison outputs")
    parser.add_argument("--input-path", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--scored-path", type=Path, default=Path(DEFAULT_SCORED))
    parser.add_argument("--summary-path", type=Path, default=Path(DEFAULT_SUMMARY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.input_path, args.scored_path, args.summary_path)


if __name__ == "__main__":
    main()
