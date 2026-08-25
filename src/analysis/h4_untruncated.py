"""H04 re-tested on the untruncated (max_seq_length=256) index.

The original index was built at max_seq_length=128 while LaBSE supports 256, so
100% of indexed case narratives (median 307 tokens) lost roughly 60% of their
content and 52.5% of Hinglish queries lost their tail.

Untruncating does NOT move both arms the same way -- which matters, because the
code-mixing penalty is the DIFFERENCE between them:

    Q1 Hinglish  0.1144 -> 0.1310   (+14.5% relative; long queries gain)
    Q2 English   0.1602 -> 0.1466   (short queries lose to longer documents)

So the measured penalty shrinks and must be re-tested rather than carried over.
Writes results/index_truncation/h4_256_tests.{csv,md}.
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.analysis.h4_retrieval import bootstrap_delta, mcnemar, rank_metrics, strip_caption
from src.encoding.text_encoder import TextEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("results/index_truncation")
CACHE = OUT / "emb_cache"


def encode_cached(enc: TextEncoder, texts: list[str], tag: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{tag}_256.npy"
    if p.exists():
        e = np.load(p)
        if e.shape[0] == len(texts):
            logger.info("cache hit %s", tag)
            return e
    logger.info("encoding %s (%d texts) ...", tag, len(texts))
    e = enc.encode(texts, batch_size=32, show_progress=False)
    np.save(p, e)
    return e


def main() -> None:
    pairs = pd.read_csv("data/processed/mmcqsd_multicare_paired.csv")
    meta = pd.read_csv("data/faiss_index_256/evidence_metadata.csv")
    ix = faiss.read_index("data/faiss_index_256/evidence.index")

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = 256

    variants = {
        "Q1_hinglish": pairs.hinglish_query.astype(str).tolist(),
        "Q2_english_question": pairs.english_summary.astype(str).apply(strip_caption).tolist(),
    }
    gold = pairs.condition_query.to_numpy()
    cond = meta.condition_group.to_numpy()

    hits, rows = {}, []
    for name, texts in variants.items():
        e = encode_cached(enc, texts, name)
        _, idx = ix.search(e.astype(np.float32), 10)
        hits[name] = cond[idx] == gold[:, None]
        rows.append({"variant": name, "max_seq_length": 256, **rank_metrics(hits[name])})
        logger.info("%s R@1=%.4f", name, rows[-1]["recall@1"])

    a = hits["Q1_hinglish"][:, 0]
    b = hits["Q2_english_question"][:, 0]
    n01, n10, p = mcnemar(a, b)
    d, lo, hi = bootstrap_delta(a.astype(float), b.astype(float))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "h4_256_metrics.csv", index=False)
    pd.DataFrame([{"comparison": "Q2 - Q1", "delta_recall@1": b.mean() - a.mean(),
                   "ci_lo": lo, "ci_hi": hi, "mcnemar_n01": n01, "mcnemar_n10": n10,
                   "mcnemar_p": p, "n": len(pairs)}]).to_csv(OUT / "h4_256_tests.csv", index=False)

    sig = "SURVIVES" if p < 0.05 else "DOES NOT SURVIVE"
    L = [
        "# H04 on the untruncated (max_seq_length=256) index", "",
        f"n = {len(pairs)} · same index size, same relevance criterion, same leakage gate.", "",
        "| Index | Q1 Hinglish R@1 | Q2 English R@1 | Q2 - Q1 | McNemar p |",
        "|---|---:|---:|---:|---:|",
        "| truncated (128) | 0.1144 | 0.1602 | **+0.0458** | 9.13e-08 |",
        f"| **untruncated (256)** | {a.mean():.4f} | {b.mean():.4f} | **{b.mean()-a.mean():+.4f}** | {p:.4g} |",
        "",
        f"95% CI on the untruncated delta: [{lo:+.4f}, {hi:+.4f}] · McNemar n01={n01}, n10={n10}", "",
        "## Reading", "",
        f"**The code-mixing penalty {sig}** at the encoder's real sequence limit.", "",
        "Untruncating moves the two arms in OPPOSITE directions -- Hinglish queries are long "
        "(median 132 tokens, 52.5% were being cut) and gain from fuller documents, while the "
        "short caption-stripped English questions lose, because a longer document dilutes the "
        "match for a short query. The penalty is a difference, so it absorbs both moves.", "",
        "This is why the truncation defect could not be waved through as 'affecting both arms "
        "equally'. It did not.", "",
        "**Next:** passage chunking should help long and short queries simultaneously instead "
        "of trading one against the other, and is the standard fix for exactly this asymmetry.",
        "",
    ]
    (OUT / "h4_256_tests.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
