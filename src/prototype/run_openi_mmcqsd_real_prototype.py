"""Run the real Open-i + MMCQSD prototype pipeline end-to-end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prototype.build_index import build_index
from src.prototype.evaluate_real_dataset_comparison import evaluate
from src.prototype.prepare_openi_mmcqsd_real import prepare_openi_mmcqsd_real
from src.prototype.run_baselines import run


DEFAULT_DATA_DIR = Path("data/processed/real_h1")
DEFAULT_RESULTS_DIR = Path("results/h1_real_openi_mmcqsd")


def run_real_prototype(
    data_dir: Path,
    results_dir: Path,
    max_queries: int,
    top_k: int,
    seed: int,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _, aligned_queries, corpus_path, queries_path, prep_summary_path = prepare_openi_mmcqsd_real(
        output_dir=data_dir,
        max_queries=max_queries,
        seed=seed,
    )
    if aligned_queries.empty:
        raise ValueError("No aligned Open-i + MMCQSD queries were prepared.")

    index_path = results_dir / "openi_reports.index"
    meta_path = results_dir / "openi_reports_metadata.csv"
    vectorizer_path = results_dir / "tfidf_vectorizer.pkl"
    outputs_path = results_dir / "h1_outputs.csv"
    scored_path = results_dir / "h1_scored.csv"
    summary_path = results_dir / "h1_summary.md"

    build_index(
        hmg_path=corpus_path,
        index_path=index_path,
        metadata_path=meta_path,
        vectorizer_path=vectorizer_path,
    )
    run(
        hmg_path=queries_path,
        index_path=index_path,
        metadata_path=meta_path,
        vectorizer_path=vectorizer_path,
        output_path=outputs_path,
        top_k=top_k,
    )
    metrics = evaluate(
        input_path=outputs_path,
        scored_path=scored_path,
        summary_path=summary_path,
    )
    metrics["prepared_queries"] = int(len(aligned_queries))
    metrics["prep_summary_path"] = str(prep_summary_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real Open-i + MMCQSD prototype")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-queries", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_real_prototype(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        max_queries=args.max_queries,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(f"Prepared queries: {metrics['prepared_queries']}")
    print(f"Summary report: {args.results_dir / 'h1_summary.md'}")
    print(f"Prep summary: {metrics['prep_summary_path']}")


if __name__ == "__main__":
    main()
