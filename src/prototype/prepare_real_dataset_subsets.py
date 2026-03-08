"""Prepare real dataset subsets for cross-dataset comparison.

This script converts the downloaded real datasets into one shared schema:

- sample_id
- query_hinglish
- report_id
- report_text
- target_text
- dataset_profile

The goal is to replace the earlier synthetic proxy setup with real,
manageable subsets that can be compared under one retrieval pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data/processed/real_comparison_subsets")
DEFAULT_SIZE = 200
DEFAULT_SEED = 42


def _sample_df(df: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= subset_size:
        return df.reset_index(drop=True)
    return df.sample(n=subset_size, random_state=seed).reset_index(drop=True)


def _prepare_openi(subset_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv("data/processed/openi_reports.csv")
    df = df.fillna("")
    df = df[
        df["query_or_prompt"].astype(str).str.strip().ne("")
        & df["report_text"].astype(str).str.strip().ne("")
    ].copy()
    df = _sample_df(df, subset_size=subset_size, seed=seed)
    return pd.DataFrame(
        {
            "sample_id": df["report_id"].astype(str),
            "query_hinglish": df["query_or_prompt"].astype(str),
            "report_id": df["report_id"].astype(str),
            "report_text": df["report_text"].astype(str),
            "target_text": df["report_text"].astype(str),
            "dataset_profile": "openi",
        }
    )


def _prepare_mmcqsd(subset_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv("data/processed/mmcqsd_queries.csv")
    df = df.fillna("")
    df = df[
        df["hinglish_query"].astype(str).str.strip().ne("")
        & df["english_summary_or_target"].astype(str).str.strip().ne("")
    ].copy()
    df = _sample_df(df, subset_size=subset_size, seed=seed)
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "query_hinglish": df["hinglish_query"].astype(str),
            "report_id": "MMCQSD_REPORT_" + df["sample_id"].astype(str),
            "report_text": df["english_summary_or_target"].astype(str),
            "target_text": df["english_summary_or_target"].astype(str),
            "dataset_profile": "mmcqsd",
        }
    )


def _prepare_pubmedqa(subset_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv("data/processed/pubmedqa_records.csv")
    df = df.fillna("")
    # Prefer the expert-labeled subset for cleaner real comparison.
    df = df[df["subset"].astype(str) == "pqa_labeled"].copy()
    df = df[
        df["question"].astype(str).str.strip().ne("")
        & df["answer_rationale"].astype(str).str.strip().ne("")
    ].copy()
    df["report_text"] = (
        df["context_text"].astype(str).str.strip()
        + " Conclusion: "
        + df["answer_rationale"].astype(str).str.strip()
    )
    df["target_text"] = (
        df["answer_rationale"].astype(str).str.strip()
        + " Final decision: "
        + df["final_decision"].astype(str).str.strip()
    )
    df = _sample_df(df, subset_size=subset_size, seed=seed)
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "query_hinglish": df["question"].astype(str),
            "report_id": "PUBMEDQA_REPORT_" + df["sample_id"].astype(str),
            "report_text": df["report_text"].astype(str),
            "target_text": df["target_text"].astype(str),
            "dataset_profile": "pubmedqa",
        }
    )


def _prepare_mmedbench(subset_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv("data/processed/mmedbench_questions.csv")
    df = df.fillna("")
    # Restrict to English for fair comparison with the current text pipeline.
    df = df[df["language"].astype(str) == "English"].copy()
    df = df[
        df["question"].astype(str).str.strip().ne("")
        & df["rationale"].astype(str).str.strip().ne("")
    ].copy()
    df["report_text"] = (
        "Correct answer: "
        + df["answer_text"].astype(str).str.strip()
        + ". Rationale: "
        + df["rationale"].astype(str).str.strip()
    )
    df["target_text"] = df["report_text"]
    df = _sample_df(df, subset_size=subset_size, seed=seed)
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "query_hinglish": df["question"].astype(str),
            "report_id": "MMEDBENCH_REPORT_" + df["sample_id"].astype(str),
            "report_text": df["report_text"].astype(str),
            "target_text": df["target_text"].astype(str),
            "dataset_profile": "mmed_bench",
        }
    )


def prepare_real_subsets(output_dir: Path, subset_size: int, seed: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "openi": _prepare_openi(subset_size=subset_size, seed=seed),
        "mmcqsd": _prepare_mmcqsd(subset_size=subset_size, seed=seed),
        "pubmedqa": _prepare_pubmedqa(subset_size=subset_size, seed=seed),
        "mmed_bench": _prepare_mmedbench(subset_size=subset_size, seed=seed),
    }
    output_paths: dict[str, Path] = {}
    for profile, df in datasets.items():
        path = output_dir / f"{profile}_real_subset.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        output_paths[profile] = path
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real dataset subsets for comparison")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = prepare_real_subsets(
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    for profile, path in output_paths.items():
        print(f"{profile}: {path}")


if __name__ == "__main__":
    main()
