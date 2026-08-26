"""Regenerate every reported number and figure from cached artifacts.

THE RULE THIS ENFORCES: if a number is in the paper and this script does not
produce it, it does not go in the paper.

Stages are ordered by dependency and each is skipped if its inputs are missing,
so a partial checkout still runs as far as it can and says what it could not do.
Nothing here calls an API — every stage re-scores cached generations or recomputes
from cached embeddings.

    python -m src.analysis.reproduce_all            # everything available
    python -m src.analysis.reproduce_all --list     # show stages and inputs
    python -m src.analysis.reproduce_all --only h1  # substring match on stage name

WHAT IS NOT REPRODUCIBLE HERE, and why:
  * The n=1,165 llama generations. `llama-3.1-8b-instant` was decommissioned by
    Groq mid-project and returns 404 on every key. The outputs are committed under
    results/combined_h1h2/ and can be re-scored, but not regenerated. This is a
    genuine reproducibility limitation and is stated in the paper.
  * Passage embeddings (~3h CPU). Cached in data/passage_index/; rebuild with
    `python -m src.analysis.retrieval_v2` if the cache is absent.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

#: (stage name, module, required inputs, what it produces)
STAGES: list[tuple[str, str, list[str], str]] = [
    ("rescore-generations", "src.analysis.rescore_all",
     ["results/combined_h1h2/combined_scored.csv",
      "data/processed/mmcqsd_multicare_paired.csv"],
     "results/rescored/ — precision/recall/F1 vs both references, BH-corrected"),

    ("m4-caption-metric", "src.analysis.run_m4_rescore",
     ["results/combined_h1h2/combined_scored.csv"],
     "results/m4_caption/ — circular vs unbiased, cluster-bootstrap CIs"),

    ("h2-per-arm", "src.analysis.h2_recompute",
     ["results/combined_h1h2/combined_scored.csv"],
     "results/h2_per_arm/ — H02 on the repaired CMI"),

    ("h2-figures", "src.analysis.h2_figures",
     ["results/combined_h1h2/combined_scored.csv"],
     "results/h2_figures/ — dose-response + effect comparison"),

    ("h4-retrieval", "src.analysis.h4_retrieval",
     ["data/faiss_index/evidence.index", "data/faiss_index/evidence_metadata.csv"],
     "results/h4_retrieval/ — H04 on the flat index"),

    ("retrieval-v2", "src.analysis.retrieval_v2",
     ["data/passage_index/passage_emb.npy", "data/faiss_index/evidence_metadata.csv"],
     "results/retrieval_v2/ — Table 1: chunked, matched content, hybrid RRF"),

    ("truncation-sweep", "src.analysis.truncation_sweep",
     ["data/faiss_index/evidence.index"],
     "results/truncation_sweep/ — adaptive truncation negative result"),

    ("retrieval-figures", "src.analysis.retrieval_figures",
     ["results/retrieval_v2/retrieval_v2_metrics.csv",
      "results/retrieval_v2/h4_v2_tests.csv"],
     "results/retrieval_figures/ — Table 1 bars + penalty asymmetry"),

    ("h1-figures", "src.analysis.h1_figures",
     ["results/rescored/rescored_llama_oracle_n1165.csv"],
     "results/h1_figures/ — the reference-effect figure"),

    ("h1-oracle-vs-real", "src.analysis.h1_oracle_vs_real",
     ["results/h1_real_retrieval/h1_real_scored.csv"],
     "results/h1_real_retrieval/ — oracle vs real, refusal rates"),
]


def missing(inputs: list[str]) -> list[str]:
    return [p for p in inputs if not Path(p).exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate every result from cached artifacts")
    ap.add_argument("--list", action="store_true", help="show stages and exit")
    ap.add_argument("--only", type=str, default="", help="run stages whose name contains this")
    args = ap.parse_args()

    if args.list:
        for name, mod, inputs, produces in STAGES:
            gap = missing(inputs)
            print(f"\n{name}\n  module   {mod}\n  produces {produces}")
            print(f"  inputs   {'OK' if not gap else 'MISSING: ' + ', '.join(gap)}")
        return

    ran, skipped, failed = [], [], []
    for name, mod, inputs, produces in STAGES:
        if args.only and args.only not in name:
            continue
        gap = missing(inputs)
        if gap:
            logger.warning("SKIP  %-22s missing: %s", name, ", ".join(gap))
            skipped.append((name, gap))
            continue
        logger.info("RUN   %-22s -> %s", name, produces)
        t0 = time.time()
        try:
            m = importlib.import_module(mod)
            m.main()
            ran.append((name, time.time() - t0))
        except SystemExit as e:                     # a stage refusing to run
            logger.error("FAIL  %-22s %s", name, e)
            failed.append((name, str(e)))
        except Exception as e:
            logger.exception("FAIL  %-22s %s: %s", name, type(e).__name__, e)
            failed.append((name, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 66)
    print(f"reproduced {len(ran)} stage(s), skipped {len(skipped)}, failed {len(failed)}")
    for n, secs in ran:
        print(f"  OK      {n:<22} {secs:6.1f}s")
    for n, gap in skipped:
        print(f"  SKIP    {n:<22} missing {len(gap)} input(s)")
    for n, err in failed:
        print(f"  FAIL    {n:<22} {err[:70]}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
