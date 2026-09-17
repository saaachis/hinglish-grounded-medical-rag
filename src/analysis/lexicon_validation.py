"""Validate the concept extractor without human annotators (R3.4, R1.c, R1.d).

Reviewer 3: "The 26-concept lexicon is not clinician-validated; independent
clinical validation would considerably strengthen conclusions."
Reviewer 1 asks for the same thing, plus "richer factuality metrics".

Clinician time is not available for this revision, so the honest response is to
supply what CAN be measured automatically and to keep clinician validation as a
stated limitation. Nothing here licenses the phrase "clinically validated".

Two independent checks:

PART 1 -- COVERAGE (offline, no API)
    What share of the reference texts the 26 concepts can represent at all, and
    which frequent clinical terms fall outside the lexicon. A concept-overlap
    metric can only see what its vocabulary contains, so this bounds the
    construct: if the references routinely describe findings the lexicon has no
    word for, every score computed from it is partial in a measurable way.

PART 2 -- LLM-JUDGE AGREEMENT (API)
    An independent model reads each answer and reports which of the same 26
    concepts the answer ASSERTS about the patient (excluding negated and
    hypothetical mentions -- exactly the distinction the negation repair
    introduced). Its labels are compared with the regex extractor's:
    precision, recall, F1 and Cohen's kappa over per-concept decisions.

    This is NOT a clinical gold standard. It is a second, differently-implemented
    reader: agreement means the extractor is not idiosyncratic, disagreement
    localises where it fails. Report it as automatic cross-validation.

Writes results/lexicon_validation/.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import h1_real_retrieval as groqmod
from src.analysis.cmi import build_english_vocab
from src.evaluation.concept_lexicon import POSITIVE_CONCEPTS, extract_concepts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COMBINED = Path("results/combined_h1h2/combined_scored.csv")
PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
OUT = Path("results/lexicon_validation")
JUDGED = OUT / "llm_judge_labels.csv"
SEED = 42
CHECKPOINT_EVERY = 20

#: Words that look clinical but are structural noise in the uncovered-term list.
STOPISH = set("""the a an and or of in on to for with without patient patients doctor please
sir madam hai hain mein aur ko se ka ki ke ye yeh wo woh nahi bhi par tha thi hua
years year old age male female left right side days day weeks week months month
image shows show shown condition related area region also can could would should
""".split())

SYSTEM = (
    "You are a careful clinical annotator. You will be given a patient's question "
    "and an answer written in Hinglish (romanised Hindi mixed with English).\n\n"
    "From the fixed vocabulary provided, list ONLY the findings that the ANSWER "
    "positively asserts about THIS patient.\n\n"
    "Exclude a finding if the answer denies it (\"rash nahi hai\", \"no swelling\"), "
    "if it is only hypothetical or hedged (\"ho sakta hai\", \"main confirm nahi kar "
    "sakta\"), if it names a test or treatment rather than a finding, or if it "
    "appears only in the patient's question and not in the answer.\n\n"
    "Reply with a JSON array of vocabulary terms and nothing else. Use [] if none apply."
)


# --------------------------------------------------------------------------- #
# Part 1: coverage
# --------------------------------------------------------------------------- #
def coverage_analysis(top_n: int = 30) -> tuple[dict, pd.DataFrame]:
    pairs = pd.read_csv(PAIRS)
    vocab = build_english_vocab()
    refs = pairs.english_summary.astype(str).tolist()
    evidence = pairs.evidence_text.astype(str).tolist()

    covered_ref = sum(bool(extract_concepts(t)) for t in refs)
    covered_ev = sum(bool(extract_concepts(t)) for t in evidence)
    n_concepts = [len(extract_concepts(t)) for t in refs]

    # Frequent English content words in the references that the lexicon cannot see.
    lex_words = {w for c in POSITIVE_CONCEPTS for w in c.split("_")}
    counts: Counter[str] = Counter()
    for t in refs:
        for w in re.findall(r"[a-z]{4,}", t.lower()):
            if w in STOPISH or w in lex_words:
                continue
            if any(w.startswith(c[:5]) for c in POSITIVE_CONCEPTS):
                continue
            # keep words that look like medical vocabulary: not in a general
            # English wordlist, or a known clinical suffix
            if w not in vocab or re.search(r"(itis|osis|oma|emia|pathy|algia|ectomy|plasia)$", w):
                counts[w] += 1
    uncovered = pd.DataFrame(counts.most_common(top_n), columns=["term", "occurrences"])
    # Many uncovered terms are misspellings of covered ones ("itichy", "rasied").
    # That distinction matters: a misspelling means the metric misses a finding it
    # HAS a word for, which is a fixable extractor gap rather than a vocabulary gap.
    import difflib

    from src.evaluation.concept_lexicon import CONCEPT_PATTERNS

    # Compare against the actual SURFACE patterns, not the concept keys: the
    # lexicon matches "itching", not the key "itching_only". Cutoff 0.75 catches
    # single-character typos in 6-8 letter words without pairing unrelated terms.
    lex_terms = sorted({p.lower() for pats in CONCEPT_PATTERNS.values() for p in pats}
                       | {w for c in POSITIVE_CONCEPTS for w in c.split("_")})
    near = []
    for t in uncovered.term:
        match = difflib.get_close_matches(t, lex_terms, n=1, cutoff=0.75)
        near.append(match[0] if match else "")
    uncovered["likely_misspelling_of"] = near

    stats = {
        "n_references": len(refs),
        "references_with_at_least_one_concept": covered_ref / len(refs),
        "evidence_with_at_least_one_concept": covered_ev / len(evidence),
        "mean_concepts_per_reference": float(np.mean(n_concepts)),
        "median_concepts_per_reference": float(np.median(n_concepts)),
        "references_with_exactly_one_concept": float(np.mean(np.array(n_concepts) == 1)),
    }
    return stats, uncovered


# --------------------------------------------------------------------------- #
# Part 2: LLM judge
# --------------------------------------------------------------------------- #
def parse_reply(text: str, allowed: set[str]) -> set[str] | None:
    """Parse the model's JSON array; None if it did not comply."""
    m = re.search(r"\[.*?\]", str(text), re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    return {str(i).strip().lower().replace(" ", "_") for i in items} & allowed


def judge(sample: pd.DataFrame, model: str) -> pd.DataFrame:
    allowed = set(POSITIVE_CONCEPTS)
    menu = ", ".join(sorted(allowed))
    done: dict[str, str] = {}
    if JUDGED.exists():
        prev = pd.read_csv(JUDGED)
        done = dict(zip(prev.item_id.astype(str), prev.judge_raw.astype(str)))
        logger.info("resuming: %d already judged", len(done))

    todo = [r for r in sample.itertuples() if r.item_id not in done]
    if todo:
        groqmod.MODEL = model
        client = groqmod.RotatingGroq(groqmod.load_keys())
        logger.info("judging %d answers with %s", len(todo), model)
        for i, r in enumerate(todo, 1):
            user = (f"VOCABULARY: {menu}\n\nPATIENT QUESTION:\n{r.patient_question}\n\n"
                    f"ANSWER:\n{r.answer_text}\n\nJSON array only:")
            out = client.chat(SYSTEM, user)
            if out.startswith("[QUOTA_EXHAUSTED"):
                logger.error("quota exhausted after %d judgments -- re-run to resume", i - 1)
                break
            done[r.item_id] = out
            if i % CHECKPOINT_EVERY == 0:
                _save(done)
                logger.info("  %d/%d", i, len(todo))
        _save(done)

    sample = sample.copy()
    sample["judge_raw"] = sample.item_id.map(done)
    return sample[sample.judge_raw.notna()].reset_index(drop=True)


def _save(done: dict[str, str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": list(done), "judge_raw": list(done.values())}).to_csv(
        JUDGED, index=False, encoding="utf-8")


def kappa(a: list[int], b: list[int]) -> float:
    a, b = np.array(a), np.array(b)
    po = float((a == b).mean())
    pe = float((a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean())))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def build_sample(n: int) -> pd.DataFrame:
    df = pd.read_csv(COMBINED)
    long = pd.concat([
        df.assign(arm="grounded", answer_text=df.grounded_output),
        df.assign(arm="zero_shot", answer_text=df.zero_shot_output),
    ], ignore_index=True)
    long = long[long.answer_text.notna() & (long.answer_text.astype(str).str.len() > 20)]
    cells = [g.sample(min(n // 2, len(g)), random_state=SEED)
             for _, g in long.groupby("arm", observed=True)]
    s = pd.concat(cells, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    s["item_id"] = [f"J{i + 1:03d}" for i in range(len(s))]
    s["patient_question"] = s.hinglish_query
    return s[["item_id", "arm", "pair_id", "patient_question", "answer_text"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the concept extractor without annotators")
    ap.add_argument("--n", type=int, default=120, help="answers to judge")
    ap.add_argument("--model", default="llama-3.3-70b-versatile",
                    help="judge model; use one NOT used elsewhere today (per-model daily quota)")
    ap.add_argument("--coverage-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stats, uncovered = coverage_analysis()
    uncovered.to_csv(OUT / "uncovered_terms.csv", index=False)

    judged = pd.DataFrame()
    if not args.coverage_only:
        judged = judge(build_sample(args.n), args.model)

    rows, per_concept = [], Counter()
    tp = fp = fn = 0
    ext_labels, judge_labels = [], []
    unparsed = 0
    for r in judged.itertuples():
        j = parse_reply(r.judge_raw, set(POSITIVE_CONCEPTS))
        if j is None:
            unparsed += 1
            continue
        e = extract_concepts(str(r.answer_text)) & set(POSITIVE_CONCEPTS)
        tp += len(e & j)
        fp += len(e - j)
        fn += len(j - e)
        for c in (e - j):
            per_concept[f"{c} (extractor only)"] += 1
        for c in (j - e):
            per_concept[f"{c} (judge only)"] += 1
        for c in sorted(POSITIVE_CONCEPTS):
            ext_labels.append(int(c in e))
            judge_labels.append(int(c in j))
        rows.append({"item_id": r.item_id, "arm": r.arm,
                     "extractor": ",".join(sorted(e)), "judge": ",".join(sorted(j)),
                     "agree": len(e & j), "extractor_only": len(e - j), "judge_only": len(j - e)})
    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail.to_csv(OUT / "judge_vs_extractor.csv", index=False)

    write_report(stats, uncovered, detail, tp, fp, fn, ext_labels, judge_labels,
                 per_concept, unparsed, args.model)


def write_report(stats, uncovered, detail, tp, fp, fn, ext_labels, judge_labels,
                 per_concept, unparsed, model) -> None:
    L = ["# Validating the concept extractor without annotators", "",
         "Clinician annotation was out of scope for this revision. This is what can be",
         "measured automatically. **It does not license the phrase \"clinically validated\";",
         "clinician validation remains a stated limitation.**", "",
         "## Part 1 — what the 26 concepts can represent at all", "",
         "| Quantity | Value |", "|---|---:|",
         f"| References (gold English summaries) | {stats['n_references']:,} |",
         f"| References containing >= 1 lexicon concept | {stats['references_with_at_least_one_concept']:.1%} |",
         f"| Evidence texts containing >= 1 lexicon concept | {stats['evidence_with_at_least_one_concept']:.1%} |",
         f"| Mean concepts per reference | {stats['mean_concepts_per_reference']:.2f} |",
         f"| Median concepts per reference | {stats['median_concepts_per_reference']:.0f} |",
         f"| References with exactly one concept | {stats['references_with_exactly_one_concept']:.1%} |",
         "",
         "A reference carrying one or two concepts gives precision and recall very few",
         "events to work with, which is why single-reference scores are unstable and why",
         "the paper reports paired deltas rather than absolute levels.", "",
         "### Frequent clinical terms the lexicon has no word for", "",
         "| Term | Occurrences in the references |"]
    L[-1] = "| Term | Occurrences | Close to a lexicon word? |"
    L.append("|---|---:|---|")
    for _, r in uncovered.head(20).iterrows():
        L.append(f"| `{r.term}` | {r.occurrences} | "
                 f"{('`' + r.likely_misspelling_of + '`') if r.likely_misspelling_of else '—'} |")
    n_miss = int((uncovered.likely_misspelling_of != "").sum())
    L += ["", f"Of the {len(uncovered)} most frequent uncovered terms, **{n_miss} are close variants "
          "of words the lexicon already has** (the reference captions are typed free text and "
          "contain frequent misspellings). Those are extractor gaps, fixable with fuzzy matching. "
          "The rest are findings the 26-item vocabulary genuinely cannot represent.",
          "",
          "Either way the consequence is the same for interpretation: the metric measures overlap",
          "on a small vocabulary, not clinical correctness, and the unbiased reference is the",
          "text most affected, since it is a short free-text caption.", ""]

    L += ["## Part 2 — agreement with an independent LLM reader", ""]
    if detail.empty:
        L += ["_Not run. Use `python -m src.analysis.lexicon_validation` with API access._", ""]
    else:
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        k = kappa(ext_labels, judge_labels)
        L += [f"Judge: `{model}`, {len(detail)} answers"
              + (f" ({unparsed} replies unparseable and skipped)" if unparsed else "") + ".", "",
              "| Quantity | Value |", "|---|---:|",
              f"| Extractor precision vs judge | {prec:.3f} |",
              f"| Extractor recall vs judge | {rec:.3f} |",
              f"| Extractor F1 vs judge | {f1:.3f} |",
              f"| Cohen's kappa (per-concept decisions) | {k:.3f} |",
              f"| Concepts agreed / extractor-only / judge-only | {tp} / {fp} / {fn} |", "",
              "### Where they disagree most", "",
              "| Concept and direction | Count |", "|---|---:|"]
        for name, cnt in per_concept.most_common(12):
            L.append(f"| {name} | {cnt} |")
        verdict = ("substantial" if k >= 0.6 else "moderate" if k >= 0.4 else
                   "fair" if k >= 0.2 else "poor")
        L += ["", f"Agreement is **{verdict}** (kappa = {k:.3f}). ",
              "An extractor-only disagreement is a likely false positive (a mention the answer",
              "did not assert); a judge-only disagreement is a likely miss (wording the regexes",
              "do not cover). Report both numbers in Sect. 6.1 next to the lexicon limitation.", ""]
    (OUT / "lexicon_validation_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
