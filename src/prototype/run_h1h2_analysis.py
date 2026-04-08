"""Combined H1 + H2 analysis on all evaluated pairs.

Merges multiple scored CSVs, computes CMI where missing,
runs H1 (grounded vs zero-shot) and H2 (performance by CMI level)
with full statistical tests, and generates a combined report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

HINDI_TOKENS = {
    "kya", "hai", "me", "mujhe", "meri", "mera", "kripya", "saans", "khansi",
    "bukhar", "dard", "batao", "samjhao", "ho", "raha", "rahi", "hain",
    "nahi", "aur", "bhi", "ko", "se", "ke", "ki", "ka", "ye", "wo", "ek",
    "bahut", "thoda", "zyada", "abhi", "pehle", "baad", "dono", "uske",
    "iske", "par", "pe", "thi", "tha", "hota", "hoti", "lag", "laga",
    "karna", "karke", "kuch", "jo", "hum", "kaise", "kyun", "kab", "kahan",
    "kitna", "kitni", "kitne", "sab", "yeh", "woh", "apna", "apni", "apne",
    "unke", "unki", "unka", "hamara", "hamari", "tumhara", "pata", "liye",
    "wala", "wali", "wale", "jaise", "lekin", "agar", "toh", "phir", "hona",
    "paani", "khana", "peena", "sona", "uthna", "chalna", "dekhna", "bolna",
    "sunna", "milna", "rehna", "jaana", "aana", "lena", "dena", "pet", "sir",
    "kaan", "aankhon", "aankhein", "naak", "gala", "seena", "kamar", "pair",
    "haath", "pasina", "sujan", "khujli", "thakan", "chakkar", "ulti", "dast",
    "kabz", "doctor", "please", "pls",
}


def cmi_score(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", str(text).lower())
    if not tokens:
        return 0.0
    return sum(t in HINDI_TOKENS for t in tokens) / len(tokens)


def load_and_merge(scored_paths: list[Path]) -> pd.DataFrame:
    """Load multiple scored CSVs, merge, deduplicate, filter errors."""
    frames = []
    for p in scored_paths:
        if p.exists():
            df = pd.read_csv(p)
            print(f"  Loaded {len(df)} from {p}")
            frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["pair_id"], keep="last")

    clean = merged[
        (merged["zero_shot_output"] != "[API_ERROR]")
        & (merged["grounded_output"] != "[API_ERROR]")
    ].copy()

    if "cmi_score" not in clean.columns:
        clean["cmi_score"] = clean["hinglish_query"].apply(cmi_score)
    else:
        mask = clean["cmi_score"].isna()
        if mask.any():
            clean.loc[mask, "cmi_score"] = clean.loc[mask, "hinglish_query"].apply(cmi_score)

    q33 = clean["cmi_score"].quantile(0.33)
    q66 = clean["cmi_score"].quantile(0.66)

    def bucket(s: float) -> str:
        if s <= q33:
            return "low_cm"
        if s <= q66:
            return "medium_cm"
        return "high_cm"

    clean["cmi_bucket"] = clean["cmi_score"].apply(bucket)

    print(f"\nMerged: {len(merged)} total, {len(clean)} clean")
    print(f"CMI tertiles: <={q33:.3f} / {q33:.3f}-{q66:.3f} / >={q66:.3f}")
    print(f"Distribution: {clean['cmi_bucket'].value_counts().to_dict()}")
    return clean


def h1_analysis(df: pd.DataFrame) -> dict:
    """H1: Grounded vs zero-shot comparison."""
    g = df["grounded_factual"].to_numpy(dtype=float)
    z = df["zero_factual"].to_numpy(dtype=float)
    diff = g - z

    _, p_normal = stats.shapiro(diff[:5000])
    if p_normal > 0.05:
        stat_val, p_val = stats.ttest_rel(g, z)
        test_name = "paired_t_test"
    else:
        stat_val, p_val = stats.wilcoxon(diff)
        test_name = "wilcoxon_signed_rank"

    effect_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    ci_low, ci_high = stats.t.interval(0.95, df=len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff))

    gh = df["grounded_hallucination"].to_numpy(dtype=float)
    zh = df["zero_hallucination"].to_numpy(dtype=float)
    hdiff = zh - gh
    _, h_p_normal = stats.shapiro(hdiff[:5000])
    if h_p_normal > 0.05:
        h_stat, h_pval = stats.ttest_rel(zh, gh)
        h_test = "paired_t_test"
    else:
        h_stat, h_pval = stats.wilcoxon(hdiff)
        h_test = "wilcoxon_signed_rank"
    h_effect = float(np.mean(hdiff) / np.std(hdiff, ddof=1)) if np.std(hdiff, ddof=1) > 0 else 0.0

    return {
        "n": len(df),
        "zero_factual": float(z.mean()),
        "grounded_factual": float(g.mean()),
        "factual_gain": float(diff.mean()),
        "factual_test": test_name,
        "factual_stat": float(stat_val),
        "factual_p": float(p_val),
        "factual_d": effect_d,
        "factual_ci": (float(ci_low), float(ci_high)),
        "zero_halluc": float(zh.mean()),
        "grounded_halluc": float(gh.mean()),
        "halluc_reduction": float(hdiff.mean()),
        "halluc_test": h_test,
        "halluc_stat": float(h_stat),
        "halluc_p": float(h_pval),
        "halluc_d": h_effect,
    }


def h2_analysis(df: pd.DataFrame) -> dict:
    """H2: Performance difference across 3 CMI levels."""
    buckets = ["low_cm", "medium_cm", "high_cm"]
    per_bucket = {}

    for b in buckets:
        sub = df[df["cmi_bucket"] == b]
        per_bucket[b] = {
            "n": len(sub),
            "factual_gain": sub["factual_gain"].mean(),
            "halluc_reduction": sub["halluc_reduction"].mean(),
            "grounded_factual": sub["grounded_factual"].mean(),
            "grounded_halluc": sub["grounded_hallucination"].mean(),
            "zero_factual": sub["zero_factual"].mean(),
            "zero_halluc": sub["zero_hallucination"].mean(),
            "cmi_mean": sub["cmi_score"].mean(),
        }

    groups_fg = [df[df["cmi_bucket"] == b]["factual_gain"].dropna() for b in buckets]
    groups_hr = [df[df["cmi_bucket"] == b]["halluc_reduction"].dropna() for b in buckets]

    kw_fg_stat, kw_fg_p = stats.kruskal(*groups_fg)
    kw_hr_stat, kw_hr_p = stats.kruskal(*groups_hr)

    pairwise_fg = {}
    pairwise_hr = {}
    pairs = [("low_cm", "medium_cm"), ("low_cm", "high_cm"), ("medium_cm", "high_cm")]
    n_comparisons = len(pairs)
    for a, b in pairs:
        ga = df[df["cmi_bucket"] == a]["factual_gain"].dropna()
        gb = df[df["cmi_bucket"] == b]["factual_gain"].dropna()
        _, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
        pairwise_fg[f"{a}_vs_{b}"] = {"p_raw": p, "p_bonf": min(p * n_comparisons, 1.0)}

        ha = df[df["cmi_bucket"] == a]["halluc_reduction"].dropna()
        hb = df[df["cmi_bucket"] == b]["halluc_reduction"].dropna()
        _, p = stats.mannwhitneyu(ha, hb, alternative="two-sided")
        pairwise_hr[f"{a}_vs_{b}"] = {"p_raw": p, "p_bonf": min(p * n_comparisons, 1.0)}

    spearman_fg, sp_fg_p = stats.spearmanr(df["cmi_score"], df["factual_gain"])
    spearman_hr, sp_hr_p = stats.spearmanr(df["cmi_score"], df["halluc_reduction"])

    return {
        "per_bucket": per_bucket,
        "kruskal_wallis": {
            "factual_gain": {"stat": kw_fg_stat, "p": kw_fg_p},
            "halluc_reduction": {"stat": kw_hr_stat, "p": kw_hr_p},
        },
        "pairwise_fg": pairwise_fg,
        "pairwise_hr": pairwise_hr,
        "spearman": {
            "factual_gain": {"rho": spearman_fg, "p": sp_fg_p},
            "halluc_reduction": {"rho": spearman_hr, "p": sp_hr_p},
        },
    }


def generate_report(h1: dict, h2: dict) -> str:
    sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    eff = lambda d: f"Large ({d:.3f})" if abs(d) >= 0.8 else f"Medium ({d:.3f})" if abs(d) >= 0.5 else f"Small ({d:.3f})" if abs(d) >= 0.2 else f"Negligible ({d:.3f})"

    lines = [
        "# Combined H1 + H2 Results",
        "",
        f"**Total clean pairs evaluated: {h1['n']}**",
        "",
        "---",
        "",
        "## H1: Grounded RAG vs Zero-Shot Generation",
        "",
        "### Factual Support",
        "",
        "| Metric | Zero-Shot | Grounded | Delta |",
        "|---|---:|---:|---:|",
        f"| Factual support | {h1['zero_factual']:.4f} | {h1['grounded_factual']:.4f} | **{h1['factual_gain']:+.4f}** |",
        f"| Hallucination score | {h1['zero_halluc']:.4f} | {h1['grounded_halluc']:.4f} | **{h1['halluc_reduction']:+.4f}** |",
        "",
        "### Statistical Tests",
        "",
        f"- **Factual support**: {h1['factual_test']}, stat={h1['factual_stat']:.4f}, p={h1['factual_p']:.2e} {sig(h1['factual_p'])}",
        f"  - Effect size (Cohen's d): {eff(h1['factual_d'])}",
        f"  - 95% CI: [{h1['factual_ci'][0]:.4f}, {h1['factual_ci'][1]:.4f}]",
        f"- **Hallucination reduction**: {h1['halluc_test']}, stat={h1['halluc_stat']:.4f}, p={h1['halluc_p']:.2e} {sig(h1['halluc_p'])}",
        f"  - Effect size (Cohen's d): {eff(h1['halluc_d'])}",
        "",
        "### H1 Verdict",
        "",
    ]

    if h1["factual_p"] < 0.05 and h1["factual_gain"] > 0:
        lines.append(f"**SUPPORTED**: Grounded RAG significantly improves factual support over zero-shot (p={h1['factual_p']:.2e}, d={h1['factual_d']:.3f}).")
    else:
        lines.append(f"**NOT SUPPORTED**: No significant improvement in factual support (p={h1['factual_p']:.2e}).")

    if h1["halluc_p"] < 0.05 and h1["halluc_reduction"] > 0:
        lines.append(f"**SUPPORTED**: Grounded RAG significantly reduces hallucination (p={h1['halluc_p']:.2e}, d={h1['halluc_d']:.3f}).")
    else:
        lines.append(f"Hallucination reduction not significant (p={h1['halluc_p']:.2e}).")

    lines.extend([
        "",
        "---",
        "",
        "## H2: Effect of Code-Mixing Intensity on RAG Performance",
        "",
        "### Per-Level Metrics",
        "",
        "| CMI Level | N | Mean CMI | Zero Factual | Grounded Factual | Factual Gain | Halluc Reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for b in ["low_cm", "medium_cm", "high_cm"]:
        d = h2["per_bucket"][b]
        lines.append(
            f"| {b} | {d['n']} | {d['cmi_mean']:.3f} | "
            f"{d['zero_factual']:.4f} | {d['grounded_factual']:.4f} | "
            f"{d['factual_gain']:+.4f} | {d['halluc_reduction']:+.4f} |"
        )

    kw_fg = h2["kruskal_wallis"]["factual_gain"]
    kw_hr = h2["kruskal_wallis"]["halluc_reduction"]

    lines.extend([
        "",
        "### Kruskal-Wallis Test (3-group comparison)",
        "",
        f"- **Factual gain across CMI levels**: H={kw_fg['stat']:.4f}, p={kw_fg['p']:.4f} {sig(kw_fg['p'])}",
        f"- **Halluc reduction across CMI levels**: H={kw_hr['stat']:.4f}, p={kw_hr['p']:.4f} {sig(kw_hr['p'])}",
        "",
        "### Pairwise Comparisons (Mann-Whitney U, Bonferroni corrected)",
        "",
        "| Comparison | Factual Gain p (Bonf.) | Halluc Reduction p (Bonf.) |",
        "|---|---:|---:|",
    ])

    for pair_key in h2["pairwise_fg"]:
        fg_p = h2["pairwise_fg"][pair_key]["p_bonf"]
        hr_p = h2["pairwise_hr"][pair_key]["p_bonf"]
        label = pair_key.replace("_vs_", " vs ")
        lines.append(f"| {label} | {fg_p:.4f} {sig(fg_p)} | {hr_p:.4f} {sig(hr_p)} |")

    sp = h2["spearman"]
    lines.extend([
        "",
        "### Spearman Correlation (continuous CMI vs performance)",
        "",
        f"- CMI vs Factual Gain: rho={sp['factual_gain']['rho']:.4f}, p={sp['factual_gain']['p']:.4f} {sig(sp['factual_gain']['p'])}",
        f"- CMI vs Halluc Reduction: rho={sp['halluc_reduction']['rho']:.4f}, p={sp['halluc_reduction']['p']:.4f} {sig(sp['halluc_reduction']['p'])}",
        "",
        "### H2 Verdict",
        "",
    ])

    if kw_fg["p"] < 0.05:
        lines.append(f"**SUPPORTED**: Code-mixing intensity significantly affects factual support (p={kw_fg['p']:.4f}).")
    else:
        lines.append(f"**NOT SUPPORTED**: No significant effect of code-mixing intensity on factual support (p={kw_fg['p']:.4f}).")

    if kw_hr["p"] < 0.05:
        lines.append(f"Code-mixing intensity significantly affects hallucination rate (p={kw_hr['p']:.4f}).")
    else:
        lines.append(f"Code-mixing intensity does not significantly affect hallucination rate (p={kw_hr['p']:.4f}).")

    if abs(sp["factual_gain"]["rho"]) > 0.1 and sp["factual_gain"]["p"] < 0.05:
        direction = "positively" if sp["factual_gain"]["rho"] > 0 else "negatively"
        lines.append(f"CMI is {direction} correlated with factual gain (rho={sp['factual_gain']['rho']:.3f}).")

    lines.extend([
        "",
        "---",
        f"*Analysis on {h1['n']} clean pairs across 3 CMI tertile levels*",
    ])
    return "\n".join(lines) + "\n"


def main():
    combined_path = Path("results/combined_h1h2/combined_scored.csv")
    if combined_path.exists():
        scored_paths = [combined_path]
    else:
        scored_paths = [
            Path("results/multicare_h1_llm/h1_llm_scored.csv"),
            Path("results/multicare_h1h2_day1/h1_llm_scored.csv"),
            Path("results/multicare_h1h2_day2/h1_llm_scored.csv"),
            Path("results/multicare_h1h2_day3/h1_llm_scored.csv"),
        ]

    print("Loading and merging scored results...")
    clean = load_and_merge(scored_paths)

    output_dir = Path("results/combined_h1h2")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean.to_csv(output_dir / "combined_scored.csv", index=False, encoding="utf-8")

    print("\n=== H1 Analysis ===")
    h1 = h1_analysis(clean)
    for k, v in h1.items():
        print(f"  {k}: {v}")

    print("\n=== H2 Analysis ===")
    h2 = h2_analysis(clean)
    for b, d in h2["per_bucket"].items():
        print(f"  {b}: {d}")
    print(f"  Kruskal-Wallis: {h2['kruskal_wallis']}")
    print(f"  Spearman: {h2['spearman']}")

    report = generate_report(h1, h2)
    report_path = output_dir / "h1_h2_combined_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
