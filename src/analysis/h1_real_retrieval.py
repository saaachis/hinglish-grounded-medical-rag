"""H1 under REAL retrieval -- closes the oracle-retrieval problem.

Every published H1/H2 number was measured on evidence chosen with a ground-truth
condition label: `run_matching.py` parses the condition from the MMCQSD image
path (`Multimodal_images/<condition>/`) and uses it BOTH to pre-filter the corpus
and to filter candidates. A deployed system receiving only a Hinglish text query
has no such label, so those numbers describe a ceiling, not the system.

This script re-runs the grounded arm with evidence from the live FAISS index --
no condition filter, no label -- over the same pairs, same prompts, same model.

    ORACLE  evidence_text from mmcqsd_multicare_paired.csv (condition-filtered)
    REAL    top-1 from data/faiss_index/evidence.index (unfiltered)

IMPORTANT -- the original generator no longer exists. `llama-3.1-8b-instant` was
decommissioned by Groq and now returns 404 on every key; no Llama chat model is
available on this account. Reusing the cached outputs would therefore confound a
retrieval change with a model change, so ALL THREE conditions (zero-shot,
oracle-grounded, real-grounded) are regenerated here on `openai/gpt-oss-20b`.

That turns a reproducibility problem into an extra result: H1 measured on a second
generator family is a model-transfer robustness check the paper did not have.
The cached llama outputs are retained per row for reference.

Everything is scored with `src/evaluation/concept_lexicon` (word-boundary, no
magic 0.25), against both the real and the oracle evidence, so oracle-vs-real is
a designed contrast rather than a correction.

Writes results/h1_real_retrieval/h1_real_scored.csv (checkpointed every 10 rows).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoding.text_encoder import TextEncoder
from src.evaluation.concept_lexicon import score as concept_score
from src.retrieval.indexer import FAISSIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CACHED = Path("results/combined_h1h2/combined_scored.csv")
PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
INDEX = Path("data/faiss_index/evidence.index")
META = Path("data/faiss_index/evidence_metadata.csv")
OUT_DIR = Path("results/h1_real_retrieval")

MODEL = "openai/gpt-oss-20b"
MAX_EVIDENCE_WORDS = 400
DELAY = 0.4  # SDK backs off on 429 itself; keys are round-robined
MAX_RETRIES = 3

SYSTEM_ZERO_SHOT = (
    "You are a medical assistant helping patients understand their symptoms.\n"
    "Respond in Hinglish (mix of Hindi and English) since the patient communicates in Hinglish.\n"
    "Keep the response concise (3-5 sentences).\n"
    "You do NOT have access to any clinical reports or test results."
)

#: A LOW-REFUSAL variant of the grounded prompt.
#:
#: The default prompt ends with "If the evidence does not cover something, say you
#: cannot confirm it." Models follow that instruction far more literally than the
#: original llama did, and the resulting refusal rate is the single biggest limiter
#: on every generation experiment here: at 76-88% refusal, concept scores are
#: undefined on most rows, which is what collapsed the H03 omnibus to 13 complete
#: cases out of 160.
#:
#: Measured refusal with the clause removed: gpt-oss-120b 25% -> 10%,
#: gpt-oss-20b 75% -> 35%.
#:
#: This is an ABLATION, not a replacement. Removing an instruction to abstain may
#: trade safety for coverage -- the model could confabulate where it previously
#: declined -- so both conditions are run and reported, and hallucination is
#: compared across them rather than assumed unchanged.
SYSTEM_GROUNDED_DIRECT = (
    "You are a medical assistant helping patients understand their symptoms.\n"
    "Base your response on the clinical evidence provided below.\n"
    "Respond in Hinglish (mix of Hindi and English) since the patient communicates in Hinglish.\n"
    "Keep the response concise (3-5 sentences). Explain what the evidence shows and how "
    "it relates to the patient's question."
)

SYSTEM_GROUNDED = (
    "You are a medical assistant helping patients understand their symptoms.\n"
    "You MUST base your response strictly on the clinical evidence provided below.\n"
    "Respond in Hinglish (mix of Hindi and English) since the patient communicates in Hinglish.\n"
    "Keep the response concise (3-5 sentences). Only state facts supported by the evidence.\n"
    "If the evidence does not cover something, say you cannot confirm it."
)


def load_keys() -> list[str]:
    """Collect every GROQ_API_KEY in .env, including commented-out spares.

    Six keys x 500K tokens/day is what makes the full 1,165-row run fit in the
    free tier; a single key would run dry around row 550.
    """
    keys: list[str] = []
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^\s*#?\s*GROQ_API_KEY\s*=\s*(\S+)", line)
            if m:
                k = m.group(1).strip().strip('"').strip("'")
                if k.startswith("gsk_") and k not in keys:
                    keys.append(k)
    for k in (os.getenv("GROQ_API_KEY", "").strip(),):
        if k.startswith("gsk_") and k not in keys:
            keys.append(k)
    return keys


class RotatingGroq:
    """Groq client that advances to the next key on quota/rate errors."""

    def __init__(self, keys: list[str]):
        from groq import Groq
        if not keys:
            raise SystemExit("No GROQ_API_KEY found in .env")
        self._Groq = Groq

        # Validate up front. Several keys in .env are revoked (401) and one
        # dead key at position 0 would otherwise burn every retry.
        live: list[str] = []
        for k in keys:
            try:
                Groq(api_key=k).chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
                live.append(k)
            except Exception as e:
                if "401" in str(e) or "invalid api key" in str(e).lower():
                    logger.warning("Key ...%s is revoked - skipping", k[-4:])
                else:
                    live.append(k)  # quota/transient: keep, rotation handles it
        if not live:
            raise SystemExit("No working Groq key. All keys returned 401.")

        self.keys = live
        self.i = 0
        self.client = Groq(api_key=live[0])
        self.exhausted: set[int] = set()
        logger.info("Using %d live Groq key(s) of %d found", len(live), len(keys))

    def _rotate(self) -> bool:
        self.exhausted.add(self.i)
        if len(self.exhausted) >= len(self.keys):
            return False
        while self.i in self.exhausted:
            self.i = (self.i + 1) % len(self.keys)
        self.client = self._Groq(api_key=self.keys[self.i])
        logger.warning("Rotated to key #%d", self.i + 1)
        return True

    def _advance(self) -> None:
        """Round-robin to the next live key.

        Rotating only on error leaves the spare keys idle: the SDK absorbs 429s
        with internal backoff, so the error never reaches this class and one key
        carries the whole run. Measured, that meant ~47% of requests hit a rate
        limit and throughput collapsed to 0.77 rows/min. Spreading every call
        across the keys multiplies the effective rate limit by len(keys).
        """
        if len(self.keys) > 1:
            self.i = (self.i + 1) % len(self.keys)
            self.client = self._Groq(api_key=self.keys[self.i])

    def chat(self, system: str, user: str) -> str:
        self._advance()
        for _ in range(MAX_RETRIES * max(1, len(self.keys))):
            try:
                r = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    max_tokens=300, temperature=0.3,
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e).lower()
                if any(t in msg for t in ("rate", "quota", "limit", "429", "insufficient")):
                    if not self._rotate():
                        logger.error("All keys exhausted.")
                        return "[QUOTA_EXHAUSTED]"
                    continue
                logger.warning("API error: %s", type(e).__name__)
                time.sleep(3)
        return "[API_ERROR]"


#: A grounded model given off-condition evidence tends to DECLINE rather than
#: hallucinate ("evidence mein koi jaankari nahi hai"). Those answers assert no
#: clinical concept, so the concept metric returns nan -- which would silently
#: drop them. Refusal rate is therefore a first-class outcome, not a nuisance:
#: with real retrieval it is arguably the headline safety result.
REFUSAL_RX = re.compile(
    r"(cannot confirm|can't confirm|not (?:able to |)confirm|no information|"
    r"nahi de sakta|nahi kar sakta|nahi hai koi|koi jaankari nahi|"
    r"maaf kijiye|maaf kijie|evidence mein .{0,30}nahi|does not (?:contain|cover|mention)|"
    r"doesn't (?:contain|cover|mention)|insufficient (?:evidence|information))",
    re.I,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RX.search(str(text)))


def build_prompt(query: str, evidence: str) -> str:
    ev = " ".join(str(evidence).split()[:MAX_EVIDENCE_WORDS])
    return (f"Clinical Evidence:\n{ev}\n\n"
            f"Patient Query:\n{query}\n\n"
            f"Respond based strictly on the clinical evidence above.")


def main() -> None:
    global MODEL, OUT_DIR
    ap = argparse.ArgumentParser(description="H1 under real (unfiltered) retrieval")
    ap.add_argument("--top-k", type=int, default=1,
                    help="Evidence cases to retrieve. 1 matches the oracle's single "
                         "case, isolating the condition filter as the only difference.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all cached rows")
    ap.add_argument("--model", type=str, default=MODEL,
                    help="Groq model id. Refusal rate is strongly model-dependent: "
                         "gpt-oss-20b refuses ~75-83%% on this prompt while gpt-oss-120b "
                         "refuses ~25%%, closest to the decommissioned llama's 18.1%%. "
                         "A high-refusal generator makes the factuality means describe "
                         "refusal text rather than answers.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    MODEL = args.model
    OUT_DIR = args.out_dir
    logger.info("Generator: %s -> %s", MODEL, OUT_DIR)

    for p in (CACHED, PAIRS, INDEX, META):
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    cached = pd.read_csv(CACHED)
    pairs = pd.read_csv(PAIRS)[["pair_id", "evidence_text", "condition_query"]]
    df = cached.merge(pairs, on="pair_id", how="left")
    if args.limit:
        df = df.groupby("condition", group_keys=False).apply(
            lambda g: g.sample(n=max(1, round(args.limit * len(g) / len(df))),
                               random_state=42), include_groups=False
        ).reset_index(drop=True) if "condition" in df else df.head(args.limit)
    logger.info("Rows to process: %d", len(df))

    meta = pd.read_csv(META)
    indexer = FAISSIndexer()
    indexer.load_index(str(INDEX))

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = 128
    logger.info("Encoding %d queries ...", len(df))
    q_emb = enc.encode(df["hinglish_query"].astype(str).tolist(),
                       batch_size=32, show_progress=False)
    scores, idxs = indexer.index.search(q_emb.astype(np.float32), args.top_k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partial = OUT_DIR / "h1_real_scored_partial.csv"
    records: list[dict] = []
    start = 0
    if partial.exists():
        records = pd.read_csv(partial).to_dict("records")
        start = len(records)
        logger.info("Resuming at row %d", start)

    client = RotatingGroq(load_keys())

    for i in range(start, len(df)):
        row = df.iloc[i]
        ridx = [int(j) for j in idxs[i] if j >= 0]
        rmeta = meta.iloc[ridx]
        real_evidence = "\n\n---\n\n".join(str(t) for t in rmeta["case_text"])
        rconds = list(rmeta["condition_group"].astype(str))
        gold_cond = str(row.get("condition_query", ""))

        query = str(row["hinglish_query"])
        oracle_ev = str(row.get("evidence_text", ""))

        # All three arms on the SAME generator -- the old model is gone, so
        # reusing cached llama outputs would confound retrieval with model.
        zero = client.chat(SYSTEM_ZERO_SHOT,
                           f"Patient Query:\n{query}\n\n"
                           f"Respond based on your general medical knowledge only.")
        g_oracle = client.chat(SYSTEM_GROUNDED, build_prompt(query, oracle_ev))
        g_real = client.chat(SYSTEM_GROUNDED, build_prompt(query, real_evidence))

        if "[QUOTA_EXHAUSTED]" in (zero, g_oracle, g_real):
            pd.DataFrame(records).to_csv(partial, index=False, encoding="utf-8")
            logger.error("Stopping at row %d -- all keys exhausted. Re-run to resume.", i)
            break

        # Each arm scored against the evidence it was actually conditioned on;
        # zero-shot against both, since it saw neither.
        s_zero_o = concept_score(zero, oracle_ev)
        s_zero_r = concept_score(zero, real_evidence)
        s_orc = concept_score(g_oracle, oracle_ev)
        s_real = concept_score(g_real, real_evidence)

        records.append({
            "pair_id": row["pair_id"],
            "condition": row.get("condition", ""),
            "hinglish_query": query,
            # --- retrieval provenance: capturing this now avoids a re-run ---
            "retrieved_case_ids": "|".join(str(c) for c in rmeta["case_id"]),
            "retrieved_condition_groups": "|".join(rconds),
            "retrieval_scores": "|".join(f"{float(s):.4f}" for s in scores[i][:len(ridx)]),
            "retrieval_top1_correct": bool(rconds and rconds[0] == gold_cond),
            "retrieval_any_correct": bool(gold_cond in rconds),
            # --- generations (all three on MODEL) ---
            "zero_shot_output": zero,
            "grounded_output_oracle": g_oracle,
            "grounded_output_real": g_real,
            "zero_shot_output_llama_cached": row["zero_shot_output"],
            "grounded_output_llama_cached": row["grounded_output"],
            # --- scores, unified lexicon ---
            "zero_factual_vs_oracle": s_zero_o["factual_support"],
            "zero_halluc_vs_oracle": s_zero_o["hallucination"],
            "zero_factual_vs_real": s_zero_r["factual_support"],
            "oracle_factual": s_orc["factual_support"],
            "oracle_halluc": s_orc["hallucination"],
            "real_factual": s_real["factual_support"],
            "real_halluc": s_real["hallucination"],
            "zero_is_refusal": is_refusal(zero),
            "oracle_is_refusal": is_refusal(g_oracle),
            "real_is_refusal": is_refusal(g_real),
            "zero_has_concepts": s_zero_o["output_has_concepts"],
            "oracle_has_concepts": s_orc["output_has_concepts"],
            "real_has_concepts": s_real["output_has_concepts"],
            "n_real_evidence_concepts": s_real["n_reference_concepts"],
            "n_oracle_evidence_concepts": s_orc["n_reference_concepts"],
            "top_k": args.top_k,
            "model": MODEL,
        })

        if (i + 1) % 10 == 0:
            pd.DataFrame(records).to_csv(partial, index=False, encoding="utf-8")
            logger.info("[%d/%d] saved (top1_correct so far: %.1f%%)", i + 1, len(df),
                        100 * np.mean([r["retrieval_top1_correct"] for r in records]))
        time.sleep(DELAY)

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_DIR / "h1_real_scored.csv", index=False, encoding="utf-8")
    logger.info("Wrote %s (%d rows)", OUT_DIR / "h1_real_scored.csv", len(out_df))

    if len(out_df):
        logger.info("Retrieval top-1 correct: %.1f%%", 100 * out_df["retrieval_top1_correct"].mean())
        logger.info("Real grounded factual (mean, non-nan): %.4f", out_df["real_factual"].mean())


if __name__ == "__main__":
    main()
