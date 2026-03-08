"""Run real multi-dataset comparison on normalized real subsets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prototype.build_index import build_index
from src.prototype.evaluate_real_dataset_comparison import evaluate
from src.prototype.prepare_real_dataset_subsets import prepare_real_subsets
from src.prototype.run_baselines import run


def _profile_label(profile: str) -> str:
    mapping = {
        "openi": "Open-i (real subset)",
        "mmcqsd": "MMCQSD (real subset)",
        "pubmedqa": "PubMedQA (real labeled subset)",
        "mmed_bench": "MMedBench (real English subset)",
    }
    return mapping[profile]


def _recommendation_row(row: pd.Series) -> str:
    return (
        f"- `{row['profile']}`: factual gain={row['factual_gain']:.3f}, "
        f"hallucination drop={row['hallucination_drop']:.3f}, "
        f"top-k hit={row['retrieval_topk_hit_rate']:.3f}, p={row['p_value']:.4f}"
    )


def _generate_report(
    metrics_df: pd.DataFrame,
    report_path: Path,
    subset_size: int,
    distractor_per_other_profile: int,
) -> None:
    ranked = metrics_df.sort_values(
        by=["factual_gain", "hallucination_drop", "retrieval_topk_hit_rate"],
        ascending=False,
    ).reset_index(drop=True)
    best = ranked.iloc[0]

    lines = [
        "# Multi-Dataset Real Subset Comparison Report",
        "",
        "## Purpose",
        "Compare real subsets from the downloaded datasets under one shared retrieval pipeline to estimate which dataset family is most useful for the current project objective.",
        "",
        "## Experiment Setup",
        f"- Real samples per dataset profile: **{subset_size}**",
        "- Same retrieval/indexing and generation pipeline across all profiles",
        f"- Cross-dataset distractors added per other profile: **{distractor_per_other_profile}**",
        "- Same generic evaluator and paired statistical test",
        "- `PubMedQA` comparison uses the expert-labeled subset",
        "- `MMedBench` comparison uses the English subset for compatibility with the current text pipeline",
        "",
        "## Dataset Profiles Compared",
        "- Open-i real subset",
        "- MMCQSD real subset",
        "- PubMedQA real labeled subset",
        "- MMedBench real English subset",
        "",
        "## Results Table",
        "",
        "| Profile | Top-1 Hit | Top-k Hit | Zero Factual | Grounded Factual | Factual Gain | Zero Hall. | Grounded Hall. | Hallucination Drop | p-value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            f"| {row['profile']} | "
            f"{row['retrieval_top1_hit_rate']:.3f} | {row['retrieval_topk_hit_rate']:.3f} | "
            f"{row['zero_factual_mean']:.3f} | {row['grounded_factual_mean']:.3f} | {row['factual_gain']:.3f} | "
            f"{row['zero_hallucination_rate']:.3f} | {row['grounded_hallucination_rate']:.3f} | {row['hallucination_drop']:.3f} | "
            f"{row['p_value']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Ranking Summary (Most Useful First)",
        ]
    )
    for _, row in ranked.iterrows():
        lines.append(_recommendation_row(row))

    lines.extend(
        [
            "",
            "## Recommendation",
            f"- Best profile under this real-subset comparison: **{best['profile']}**",
            "- This is still a subset-based prototype comparison, not the final full-scale benchmark.",
            "- For the main Hinglish grounded prototype, the next focused build should still prioritize `Open-i + MMCQSD`.",
            "",
            "## Next Step",
            "- Use the comparison outcome to build the main real-data prototype around `Open-i + MMCQSD`.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_index_corpus(
    profile_df: pd.DataFrame,
    profile: str,
    subsets_map: dict[str, pd.DataFrame],
    distractor_per_other_profile: int,
    seed: int,
) -> pd.DataFrame:
    distractors: list[pd.DataFrame] = []
    for offset, (other_profile, other_df) in enumerate(subsets_map.items(), start=1):
        if other_profile == profile:
            continue
        sampled = other_df[["report_id", "report_text"]].drop_duplicates()
        if len(sampled) > distractor_per_other_profile:
            sampled = sampled.sample(n=distractor_per_other_profile, random_state=seed + offset)
        sampled = sampled.reset_index(drop=True)
        sampled = pd.DataFrame(
            {
                "sample_id": [f"D_{other_profile}_{idx:03d}" for idx in range(len(sampled))],
                "query_hinglish": ["distractor"] * len(sampled),
                "report_id": sampled["report_id"].astype(str).tolist(),
                "report_text": sampled["report_text"].astype(str).tolist(),
                "target_text": [""] * len(sampled),
                "dataset_profile": [other_profile] * len(sampled),
            }
        )
        distractors.append(sampled)

    if not distractors:
        return profile_df.copy()
    return pd.concat([profile_df, *distractors], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real multi-dataset comparison")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-per-other-profile", type=int, default=40)
    parser.add_argument(
        "--subsets-dir",
        type=Path,
        default=Path("data/processed/real_comparison_subsets"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/dataset_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    profiles = ["openi", "mmcqsd", "pubmedqa", "mmed_bench"]
    metrics_rows: list[dict] = []
    subset_paths = prepare_real_subsets(
        output_dir=args.subsets_dir,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    subsets_map = {
        profile: pd.read_csv(path)
        for profile, path in subset_paths.items()
    }

    for profile in profiles:
        profile_dir = args.results_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        hmg_path = profile_dir / f"{profile}_real_subset.csv"
        index_corpus_path = profile_dir / f"{profile}_index_corpus.csv"
        index_path = profile_dir / "reports.index"
        meta_path = profile_dir / "reports_metadata.csv"
        vectorizer_path = profile_dir / "tfidf_vectorizer.pkl"
        outputs_path = profile_dir / "h1_outputs.csv"
        scored_path = profile_dir / "h1_scored.csv"
        summary_path = profile_dir / "h1_summary.md"

        df = subsets_map[profile].copy()
        index_df = build_index_corpus(
            profile_df=df,
            profile=profile,
            subsets_map=subsets_map,
            distractor_per_other_profile=args.distractor_per_other_profile,
            seed=args.seed,
        )
        df.to_csv(hmg_path, index=False, encoding="utf-8")
        index_df.to_csv(index_corpus_path, index=False, encoding="utf-8")

        build_index(
            hmg_path=index_corpus_path,
            index_path=index_path,
            metadata_path=meta_path,
            vectorizer_path=vectorizer_path,
        )
        run(
            hmg_path=hmg_path,
            index_path=index_path,
            metadata_path=meta_path,
            vectorizer_path=vectorizer_path,
            output_path=outputs_path,
            top_k=args.top_k,
        )
        metrics = evaluate(
            input_path=outputs_path,
            scored_path=scored_path,
            summary_path=summary_path,
        )
        metrics["profile_key"] = profile
        metrics["profile"] = _profile_label(profile)
        metrics["factual_gain"] = metrics["grounded_factual_mean"] - metrics["zero_factual_mean"]
        metrics["hallucination_drop"] = (
            metrics["zero_hallucination_rate"] - metrics["grounded_hallucination_rate"]
        )
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = args.results_dir / "dataset_comparison_metrics.csv"
    report_md = args.results_dir / "dataset_comparison_report.md"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")
    _generate_report(
        metrics_df=metrics_df,
        report_path=report_md,
        subset_size=args.subset_size,
        distractor_per_other_profile=args.distractor_per_other_profile,
    )

    print(f"Comparison metrics: {metrics_csv}")
    print(f"Comparison report: {report_md}")


if __name__ == "__main__":
    main()

