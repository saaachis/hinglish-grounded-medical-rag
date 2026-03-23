"""Build a CMI-stratified sample of pairs for H1+H2 evaluation.

Computes Code-Mixing Index (CMI) for each query, splits into tertiles,
and produces a balanced sample with equal representation from each level.
Excludes any already-evaluated pair IDs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Ratio of Hindi tokens to total Latin-alphabet tokens."""
    tokens = re.findall(r"[a-zA-Z]+", str(text).lower())
    if not tokens:
        return 0.0
    return sum(t in HINDI_TOKENS for t in tokens) / len(tokens)


def build_sample(
    pairs_path: Path,
    evaluated_path: Path | None,
    n: int,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_path)
    print(f"Loaded {len(pairs)} total pairs")

    if evaluated_path and evaluated_path.exists():
        evaluated = pd.read_csv(evaluated_path)
        done_ids = set(evaluated["pair_id"].tolist())
        pairs = pairs[~pairs["pair_id"].isin(done_ids)].reset_index(drop=True)
        print(f"Excluded {len(done_ids)} already-evaluated -> {len(pairs)} remaining")

    pairs["cmi_score"] = pairs["hinglish_query"].apply(cmi_score)

    q33 = pairs["cmi_score"].quantile(0.33)
    q66 = pairs["cmi_score"].quantile(0.66)

    def bucket(s: float) -> str:
        if s <= q33:
            return "low_cm"
        if s <= q66:
            return "medium_cm"
        return "high_cm"

    pairs["cmi_bucket"] = pairs["cmi_score"].apply(bucket)
    print(f"CMI tertiles: <={q33:.3f} / {q33:.3f}-{q66:.3f} / >={q66:.3f}")
    print(f"Distribution: {pairs['cmi_bucket'].value_counts().to_dict()}")

    per_bucket = n // 3
    rng = np.random.RandomState(seed)
    sampled_parts = []

    for bkt in ["low_cm", "medium_cm", "high_cm"]:
        pool = pairs[pairs["cmi_bucket"] == bkt]
        take = min(per_bucket, len(pool))
        chosen = pool.sample(n=take, random_state=rng)
        sampled_parts.append(chosen)
        print(f"  {bkt}: sampled {take} from {len(pool)}")

    sample = pd.concat(sampled_parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=rng).reset_index(drop=True)
    print(f"Final sample: {len(sample)} pairs")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved to {output_path}")

    return sample


def main():
    parser = argparse.ArgumentParser(description="Build CMI-stratified sample")
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/mmcqsd_multicare_paired.csv"))
    parser.add_argument("--evaluated", type=Path, default=Path("results/multicare_h1_llm/h1_llm_scored.csv"))
    parser.add_argument("-n", type=int, default=400, help="Total pairs to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("data/processed/cmi_sample_day1.csv"))
    args = parser.parse_args()
    build_sample(args.pairs, args.evaluated, args.n, args.seed, args.output)


if __name__ == "__main__":
    main()
