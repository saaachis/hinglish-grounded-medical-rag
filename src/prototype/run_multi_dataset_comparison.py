"""Run synthetic multi-dataset comparison for prototype benchmarking.

This script builds synthetic dataset proxies for:
  - Open-i
  - MMCQSD
  - PubMedQA
  - MMed-Bench

Then it runs the same prototype pipeline on each and generates
a detailed comparison report.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prototype.build_index import build_index
from src.prototype.evaluate_h1 import evaluate
from src.prototype.run_baselines import run


def _records_openi_like() -> list[dict]:
    return [
        {"report_id": "O001", "report_text": "Right-sided pleural effusion is present. Heart size is within normal limits."},
        {"report_id": "O002", "report_text": "Left lower lobe consolidation is noted, suspicious for pneumonia."},
        {"report_id": "O003", "report_text": "Mild cardiomegaly with pulmonary vascular congestion."},
        {"report_id": "O004", "report_text": "No acute cardiopulmonary abnormality detected."},
        {"report_id": "O005", "report_text": "Moderate right pleural effusion with compressive atelectasis."},
        {"report_id": "O006", "report_text": "Dense right lower lobe consolidation with air bronchograms."},
    ]


def _records_mmcqsd_like() -> list[dict]:
    return [
        {"report_id": "M001", "report_text": "Skin erythema over forearm with mild edema; no abscess formation."},
        {"report_id": "M002", "report_text": "Otitis externa signs with canal edema; tympanic membrane intact."},
        {"report_id": "M003", "report_text": "Conjunctival congestion with watery discharge; no corneal ulcer."},
        {"report_id": "M004", "report_text": "Mild sinus mucosal thickening without fluid level."},
        {"report_id": "M005", "report_text": "Localized soft tissue swelling around ankle, no fracture line seen."},
        {"report_id": "M006", "report_text": "No acute abnormality detected in provided clinical image summary."},
    ]


def _records_pubmedqa_like() -> list[dict]:
    return [
        {"report_id": "P001", "report_text": "Evidence suggests corticosteroids can reduce inflammation in selected cases."},
        {"report_id": "P002", "report_text": "Meta-analysis indicates no significant mortality benefit in this cohort."},
        {"report_id": "P003", "report_text": "Study reports improved symptom control with combination therapy."},
        {"report_id": "P004", "report_text": "Findings are inconclusive due to small sample size and study heterogeneity."},
        {"report_id": "P005", "report_text": "No clear association was observed between intervention and primary endpoint."},
        {"report_id": "P006", "report_text": "Moderate evidence quality supports short-term efficacy."},
    ]


def _records_mmedbench_like() -> list[dict]:
    return [
        {"report_id": "B001", "report_text": "Chest image shows right pleural effusion with basal atelectatic change."},
        {"report_id": "B002", "report_text": "Pathology-like description notes inflammatory infiltrate without malignancy."},
        {"report_id": "B003", "report_text": "Ophthalmic finding: retinal edema with exudates; no detachment."},
        {"report_id": "B004", "report_text": "Radiology QA context indicates bilateral patchy opacities suggestive of infection."},
        {"report_id": "B005", "report_text": "Distractor context: normal variant anatomy without acute pathology."},
        {"report_id": "B006", "report_text": "Cardiac silhouette mildly enlarged with vascular prominence."},
    ]


def _query_from_report(report_text: str, profile: str) -> str:
    lower = report_text.lower()

    if profile == "openi":
        if "effusion" in lower:
            return "Doctor, right/left chest side me pani ya fluid hai kya?"
        if "consolidation" in lower or "pneumonia" in lower:
            return "Kya report me pneumonia ya infection ka sign hai?"
        if "cardiomegaly" in lower:
            return "Dil ka size bada dikh raha hai kya?"
        if "no acute" in lower:
            return "Report me koi acute issue nahi hai kya?"
        return "Is chest report ka simple Hinglish explanation do."

    if profile == "mmcqsd":
        if "erythema" in lower:
            return "Skin pe laalpan aur sujan hai kya image report me?"
        if "otitis" in lower:
            return "Kaan ke infection ka sign mil raha hai kya?"
        if "conjunctival" in lower:
            return "Aankh me redness aur pani aa raha hai, report kya bolti hai?"
        return "Is clinical query ka simple Hinglish summary do."

    if profile == "pubmedqa":
        if "no significant" in lower or "no clear" in lower:
            return "Is treatment ka clear benefit hai ya nahi?"
        if "improved" in lower or "efficacy" in lower:
            return "Kya study me treatment effective dikhaya gaya hai?"
        if "inconclusive" in lower:
            return "Kya evidence conclusive nahi hai?"
        return "Is medical evidence ka short interpretation do."

    # mmed_bench proxy
    if "effusion" in lower:
        return "Multimodal context ke hisaab se effusion ka evidence hai kya?"
    if "opacities" in lower:
        return "Image + text se infection ke signs confirm hote hai kya?"
    if "normal variant" in lower:
        return "Kya ye sirf normal variant hai, acute issue nahi?"
    return "Given multimodal context, likely finding kya hai?"


def _inject_openi_confounder(query: str, report_text: str) -> str:
    """Add realistic symptom noise to Open-i queries.

    This simulates real patient language mismatch where symptom words can
    bias non-grounded responses away from report evidence.
    """
    lower = report_text.lower()
    confounders_non_pneumonia = [
        " Mujhe khansi aur bukhar bhi hai.",
        " Cough fever ke symptoms bhi chal rahe hai.",
    ]
    confounders_non_cardiac = [
        " Heart beat fast lag rahi hai, chest pressure bhi hai.",
        " Dil se related issue ka doubt bhi hai.",
    ]

    # If report is NOT infection-heavy, inject infection-like symptom noise.
    if all(token not in lower for token in ["pneumonia", "consolidation", "infect", "infiltrate"]):
        if random.random() > 0.45:
            query += random.choice(confounders_non_pneumonia)

    # If report is NOT cardiac-heavy, inject cardiac symptom noise.
    if all(token not in lower for token in ["cardiomegaly", "cardiomediastinal", "congestion"]):
        if random.random() > 0.7:
            query += random.choice(confounders_non_cardiac)
    return query


def _records_map() -> dict[str, list[dict]]:
    return {
        "openi": _records_openi_like(),
        "mmcqsd": _records_mmcqsd_like(),
        "pubmedqa": _records_pubmedqa_like(),
        "mmed_bench": _records_mmedbench_like(),
    }


def _inject_query_noise(text: str, profile: str) -> tuple[str, str]:
    q = text
    cmi_bucket = "medium"

    # Hinglish code-switch and informal typing noise.
    replacements = {
        "kya": ["kya", "kyaa", "ky"],
        "report": ["report", "repot", "rprt"],
        "please": ["please", "plz"],
        "explain": ["explain", "samjhao", "samjha do"],
        "infection": ["infection", "infec", "infxn"],
    }
    for token, variants in replacements.items():
        if token in q.lower() and random.random() > 0.6:
            q = re.sub(token, random.choice(variants), q, flags=re.IGNORECASE)

    if random.random() > 0.7:
        q += " jaldi batao."
        cmi_bucket = "high"
    elif random.random() > 0.4:
        q += " Thoda simple samjhao."
        cmi_bucket = "medium"
    else:
        cmi_bucket = "low"

    if profile in {"openi", "mmcqsd"} and random.random() > 0.5:
        q = "Doctor, " + q

    return q, cmi_bucket


def build_synthetic_dataset(
    profile: str,
    size: int,
    seed: int,
    records_map: dict[str, list[dict]],
) -> pd.DataFrame:
    random.seed(seed + hash(profile) % 1000)
    if profile not in records_map:
        raise ValueError(f"Unknown profile: {profile}")
    base = records_map[profile]

    rows: list[dict] = []
    for idx in range(size):
        report = base[idx % len(base)]
        query = _query_from_report(report["report_text"], profile)
        if profile == "openi":
            query = _inject_openi_confounder(query, report["report_text"])
        query, cmi_bucket = _inject_query_noise(query, profile)
        if random.random() > 0.75:
            query = "Please explain: " + query
        rows.append(
            {
                "sample_id": f"{profile.upper()}_{idx + 1:04d}",
                "query_hinglish": query,
                "report_id": report["report_id"],
                "report_text": report["report_text"],
                "cmi_bucket": cmi_bucket,
            }
        )
    return pd.DataFrame(rows)


def build_index_corpus(
    query_df: pd.DataFrame,
    profile: str,
    records_map: dict[str, list[dict]],
    distractor_per_other_profile: int,
    seed: int,
) -> pd.DataFrame:
    random.seed(seed + 11)
    distractors: list[dict] = []
    for other_profile, records in records_map.items():
        if other_profile == profile:
            continue
        sampled = records[:]
        random.shuffle(sampled)
        sampled = sampled[:distractor_per_other_profile]
        for idx, rec in enumerate(sampled):
            distractors.append(
                {
                    "sample_id": f"D_{other_profile}_{idx:03d}",
                    "query_hinglish": "distractor",
                    "report_id": f"D_{rec['report_id']}",
                    "report_text": rec["report_text"],
                    "cmi_bucket": "low",
                }
            )
    if not distractors:
        return query_df.copy()
    return pd.concat([query_df, pd.DataFrame(distractors)], ignore_index=True)


def _profile_label(profile: str) -> str:
    mapping = {
        "openi": "Open-i Proxy (Radiology aligned)",
        "mmcqsd": "MMCQSD Proxy (Code-mixed clinical)",
        "pubmedqa": "PubMedQA Proxy (Abstract QA style)",
        "mmed_bench": "MMed-Bench Proxy (Mixed multimodal QA)",
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
    sample_size: int,
) -> None:
    ranked = metrics_df.sort_values(
        by=["factual_gain", "hallucination_drop", "retrieval_topk_hit_rate"],
        ascending=False,
    ).reset_index(drop=True)
    best = ranked.iloc[0]

    lines = [
        "# Multi-Dataset Synthetic Comparison Report",
        "",
        "## Purpose",
        "Compare potential datasets (separately) under the same prototype pipeline and techniques to estimate which dataset family is most useful for the current Hinglish-grounded RAG objective.",
        "",
        "## Important Validity Note",
        "- This is a **synthetic proxy comparison** (not full real dataset benchmarking).",
        "- It is useful for early directional decisions, not final research claims.",
        "",
        "## Experiment Setup",
        f"- Samples per dataset profile: **{sample_size}**",
        "- Same retrieval/indexing and generation pipeline across all profiles",
        "- Added cross-profile distractor reports in retrieval index to better mimic real-world confusion",
        "- Added noisy/code-switched query variants to better mimic real input conditions",
        "- Same evaluator and paired statistical test",
        "",
        "## Dataset Profiles Compared",
        "- Open-i Proxy (radiology-style, closest to target objective)",
        "- MMCQSD Proxy (code-mixed clinical language style)",
        "- PubMedQA Proxy (abstract/evidence QA style)",
        "- MMed-Bench Proxy (mixed multimodal QA style)",
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
            f"- Best profile **under this synthetic proxy setup**: **{best['profile']}**",
            "- Do **not** treat synthetic ranking as final real-dataset selection.",
            "- For your actual project objective (Hinglish grounded radiology support), still prioritize:",
            "  1) Open-i real subset",
            "  2) MMCQSD-guided Hinglish generation",
            "  3) HMG construction on top of (1)+(2)",
            "",
            "## Next Step to Increase Validity",
            "- Replace synthetic proxy inputs with real subset CSVs and re-run the same scripts.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic multi-dataset comparison")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor-per-other-profile", type=int, default=4)
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
    records_map = _records_map()

    for profile in profiles:
        profile_dir = args.results_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        hmg_path = profile_dir / f"hmg_{profile}_mini.csv"
        index_corpus_path = profile_dir / f"hmg_{profile}_index_corpus.csv"
        index_path = profile_dir / "reports.index"
        meta_path = profile_dir / "reports_metadata.csv"
        vectorizer_path = profile_dir / "tfidf_vectorizer.pkl"
        outputs_path = profile_dir / "h1_outputs.csv"
        scored_path = profile_dir / "h1_scored.csv"
        summary_path = profile_dir / "h1_summary.md"

        df = build_synthetic_dataset(
            profile=profile,
            size=args.sample_size,
            seed=args.seed,
            records_map=records_map,
        )
        index_df = build_index_corpus(
            query_df=df,
            profile=profile,
            records_map=records_map,
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
    _generate_report(metrics_df=metrics_df, report_path=report_md, sample_size=args.sample_size)

    print(f"Comparison metrics: {metrics_csv}")
    print(f"Comparison report: {report_md}")


if __name__ == "__main__":
    main()

