"""The random-evidence control -- the decisive test of the echo thesis.

THE CLAIM UNDER TEST. Five indirect results suggest the measured grounding benefit is
largely an ECHO effect: the model restates whatever evidence it was handed, and the
evidence-based metric scores the answer against that same text, so restating anything
scores well whether or not the evidence was relevant.

    oracle vs real retrieval        p = 0.106  (the condition filter buys nothing)
    retrieval correctness -> factuality  p = 0.53   (relevance does not predict quality)
    circular vs unbiased metric     6x shrinkage
    BM25 vs LaBSE                   lexical beats cross-lingual
    refusal rate oracle vs real     p = 0.79

All five are indirect. This tests it head-on by grounding on a case drawn UNIFORMLY AT
RANDOM from the corpus -- evidence guaranteed to be about a different patient, usually a
different condition entirely -- holding query, prompt, model and scoring fixed.

THE DOUBLE SCORING IS WHAT MAKES IT DECISIVE. Each random-evidence answer is scored twice:

    vs the RANDOM evidence it was given   -> "did it echo?"
    vs the ORACLE evidence for that query -> "is it about the right thing?"

PREDICTIONS

    ECHO THESIS TRUE   factual_vs_random stays HIGH (near the real-retrieval level) while
                       factual_vs_oracle COLLAPSES. The model is fluently restating
                       irrelevant text and the metric rewards it.

    ECHO THESIS FALSE  factual_vs_random drops sharply. Evidence relevance genuinely drives
                       the score, and the earlier null results need another explanation
                       (most likely the 84% refusal rate creating a selection effect).

    THIRD OUTCOME      refusal rate jumps to ~100%. Then the model is detecting irrelevance
                       and declining, which is a SAFETY result rather than an echo result --
                       and it would mean the earlier factuality means are computed on a
                       heavily selected subset. Report it as such.

Only the random arm is generated (~1 call/row); zero-shot, oracle and real are reused from
h1_real_scored.csv, giving a fully paired four-way comparison on identical rows.

No new retrieval. Writes results/h1_random_control/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.h1_real_retrieval import (
    MODEL, SYSTEM_GROUNDED, RotatingGroq, build_prompt, is_refusal, load_keys,
)
from src.evaluation.concept_lexicon import score as concept_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REAL = Path("results/h1_real_retrieval/h1_real_scored.csv")
PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
META = Path("data/faiss_index/evidence_metadata.csv")
OUT = Path("results/h1_random_control")
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description="Random-evidence control for the echo thesis")
    ap.add_argument("--limit", type=int, default=0, help="0 = every row in h1_real_scored.csv")
    args = ap.parse_args()

    df = pd.read_csv(REAL)
    pairs = pd.read_csv(PAIRS)[["pair_id", "evidence_text", "condition_query"]]
    df = df.merge(pairs, on="pair_id", how="left", suffixes=("", "_p"))
    if args.limit:
        df = df.head(args.limit)
    meta = pd.read_csv(META)
    logger.info("Rows: %d | corpus: %d cases", len(df), len(meta))

    # Uniform random case per row, seeded. Drawn independently of the query, so ~6%
    # will land on the right condition group purely by chance -- recorded, not excluded,
    # because excluding them would make this a "guaranteed-wrong" control instead of a
    # random one.
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(meta), len(df))
    rand_meta = meta.iloc[pick].reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    partial = OUT / "h1_random_partial.csv"
    records: list[dict] = []
    start = 0
    if partial.exists():
        records = pd.read_csv(partial).to_dict("records")
        start = len(records)
        logger.info("Resuming at row %d", start)

    client = RotatingGroq(load_keys())

    for i in range(start, len(df)):
        row = df.iloc[i]
        rnd = rand_meta.iloc[i]
        rand_ev = str(rnd["case_text"])
        rand_cond = str(rnd["condition_group"])
        gold_cond = str(row.get("condition_query", ""))
        oracle_ev = str(row.get("evidence_text", ""))

        out = client.chat(SYSTEM_GROUNDED, build_prompt(str(row["hinglish_query"]), rand_ev))
        if out == "[QUOTA_EXHAUSTED]":
            pd.DataFrame(records).to_csv(partial, index=False, encoding="utf-8")
            logger.error("Quota exhausted at row %d -- re-run to resume.", i)
            break

        s_vs_rand = concept_score(out, rand_ev)      # did it echo?
        s_vs_oracle = concept_score(out, oracle_ev)  # is it about the right thing?

        records.append({
            "pair_id": row["pair_id"],
            "condition": row.get("condition", ""),
            "random_case_id": str(rnd["case_id"]),
            "random_condition_group": rand_cond,
            "random_is_coincidentally_correct": rand_cond == gold_cond,
            "grounded_output_random": out,
            "random_is_refusal": is_refusal(out),
            "random_factual_vs_random_ev": s_vs_rand["factual_support"],
            "random_halluc_vs_random_ev": s_vs_rand["hallucination"],
            "random_factual_vs_oracle_ev": s_vs_oracle["factual_support"],
            "random_has_concepts": s_vs_rand["output_has_concepts"],
            # carried through so the comparison is paired on identical rows
            "real_factual": row.get("real_factual"),
            "oracle_factual": row.get("oracle_factual"),
            "zero_factual_vs_oracle": row.get("zero_factual_vs_oracle"),
            "real_is_refusal": row.get("real_is_refusal"),
            "oracle_is_refusal": row.get("oracle_is_refusal"),
            "retrieval_top1_correct": row.get("retrieval_top1_correct"),
            "model": MODEL,
        })

        if (i + 1) % 10 == 0:
            pd.DataFrame(records).to_csv(partial, index=False, encoding="utf-8")
            logger.info("[%d/%d] refusal so far %.0f%%", i + 1, len(df),
                        100 * np.mean([r["random_is_refusal"] for r in records]))

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT / "h1_random_scored.csv", index=False, encoding="utf-8")
    logger.info("Wrote %s (%d rows)", OUT / "h1_random_scored.csv", len(out_df))
    if len(out_df) >= 20:
        report(out_df)


def report(d: pd.DataFrame) -> None:
    L = ["# Random-evidence control -- the decisive test of the echo thesis", "",
         f"n = {len(d)} paired rows · generator `{d['model'].iloc[0]}` · "
         "same queries, same prompt, same lexicon as `h1_real_retrieval`.", "",
         "## Refusal rates", "",
         "| Arm | evidence | refusal |", "|---|---|---:|"]
    for arm, ev in [("zero", "none"), ("oracle", "condition-filtered"),
                    ("real", "FAISS top-1"), ("random", "uniform random case")]:
        col = f"{arm}_is_refusal"
        if col in d:
            L.append(f"| {arm} | {ev} | {d[col].astype(bool).mean():.1%} |")

    L += ["", "## The decisive contrast", "",
          "| Scored against | mean | n scoreable |", "|---|---:|---:|",
          f"| random evidence it was GIVEN (echo) | {d.random_factual_vs_random_ev.mean():.4f} | {d.random_factual_vs_random_ev.notna().sum()} |",
          f"| oracle evidence for the query (correctness) | {d.random_factual_vs_oracle_ev.mean():.4f} | {d.random_factual_vs_oracle_ev.notna().sum()} |",
          f"| real-retrieval arm (reference) | {d.real_factual.mean():.4f} | {d.real_factual.notna().sum()} |",
          f"| oracle-evidence arm (reference) | {d.oracle_factual.mean():.4f} | {d.oracle_factual.notna().sum()} |",
          ""]

    s = d[["random_factual_vs_random_ev", "real_factual"]].dropna()
    diff = s.real_factual - s.random_factual_vs_random_ev
    p_all = stats.wilcoxon(diff)[1] if len(s) >= 10 and diff.abs().sum() > 0 else float("nan")
    L += [f"real − random(echo-scored) = **{diff.mean():+.4f}**, n={len(s)}, "
          f"Wilcoxon p = {p_all:.4g}", ""]

    # ---- the gate: is that comparison made of ANSWERS or of REFUSALS? ----
    both = d[(~d.random_is_refusal.astype(bool)) & (~d.real_is_refusal.astype(bool))]
    both = both[["random_factual_vs_random_ev", "real_factual"]].dropna()
    ref_conc = int(((d.random_is_refusal.astype(bool)) & (d.random_has_concepts.astype(bool))).sum())

    L += ["## ⚠️ Selection gate -- read before quoting anything above", "",
          f"- random-arm refusal rate: **{d.random_is_refusal.astype(bool).mean():.1%}**",
          f"- refusals that STILL carry clinical concepts: **{ref_conc}** "
          f"({ref_conc / max(1, d.random_is_refusal.astype(bool).sum()):.0%} of refusals)",
          f"- rows where BOTH arms actually answered: **n = {len(both)}**", "",
          "A refusal here is not silence -- the model declines while *quoting the evidence "
          "back* (\"evidence mein sirf X ka zikr hai\"). That quoting is scored as concept "
          "overlap, so the means above are computed largely on REFUSAL TEXT rather than on "
          "answers.", "", "### Reading", ""]

    if len(both) < 20:
        L += [f"**INCONCLUSIVE.** Only {len(both)} row(s) have a genuine answer from both "
              "arms, so the paired comparison has no power. This control cannot decide the "
              "echo thesis on this generator.", "",
              "What it DOES establish: `openai/gpt-oss-20b` refuses "
              f"{d.random_is_refusal.astype(bool).mean():.0%} of the time on random evidence "
              "and ~83% even on condition-matched evidence, versus **18.1%** for the "
              "original `llama-3.1-8b-instant`. The two generators are not behaving like "
              "the same system, so every gpt-oss contrast (oracle-vs-real, "
              "retrieval-correctness-vs-factuality) is a comparison of refusal texts and "
              "must not be read as evidence about grounding.", "",
              "**To settle the echo thesis, re-run on a generator that answers** -- either a "
              "current model with a lower refusal rate, or the same model with the "
              "\"say you cannot confirm it\" clause removed from the system prompt so "
              "refusal is not instruction-driven. Then repeat this control."]
    else:
        d2 = stats.wilcoxon(both.real_factual - both.random_factual_vs_random_ev)[1]
        L += [f"Restricted to genuine answers (n={len(both)}): real "
              f"{both.real_factual.mean():.4f} vs random "
              f"{both.random_factual_vs_random_ev.mean():.4f}, Wilcoxon p = {d2:.4g}.", ""]
        L.append("**Echo thesis CONFIRMED** -- random evidence scores indistinguishably from "
                 "retrieved evidence among real answers." if d2 >= 0.05 else
                 "**Echo thesis WEAKENED** -- relevance contributes among real answers.")

    coincident = d.random_is_coincidentally_correct.mean()
    L += ["", f"Random cases that coincidentally matched the query's condition: "
          f"**{coincident:.1%}** (chance level ~5.6% over 18 groups).", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h1_random_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
