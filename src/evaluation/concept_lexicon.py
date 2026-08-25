"""Canonical clinical concept lexicon for factuality scoring.

Replaces the FIVE divergent copies that previously coexisted:

    src/pipeline.py                          18 positive concepts
    src/prototype/evaluate_h1.py              7 positive (chest-X-ray, Open-i era)
    src/prototype/run_llm_prototype.py       24 positive
    src/prototype/run_phase6_ablation.py     24 positive
    src/prototype/run_multicare_prototype.py 26 positive

Because H1 and the Phase-6 ablation were scored with *different* copies, the
+0.069 structured-vs-raw result is partly a lexicon artefact. Everything must
score through this module from now on.

Three defects are fixed here:

1.  **Substring matching.** The old code used ``if pattern in text``, so ``red``
    fired inside *requi**red***, *occur**red***, ***red**uced*; ``itch`` inside
    *st**itch***; ``mass`` inside *massive*. Matching is now word-boundary
    anchored. Genuine suffixes (``-algia``) are declared explicitly so they keep
    working -- ``\balgia\b`` would never match *neuralgia*.

2.  **The magic 0.25.** When an output contained no concept the old code returned
    a hard-coded 0.25, which fired on 27.5% of zero-shot answers and therefore
    *set the baseline*. Scores are now ``nan`` and the caller reports
    concept-coverage as a separate diagnostic.

3.  **Negation scope.** The original code handled only forward-scoping English
    negators, so every POST-posed Hinglish denial was scored as an assertion:
    ``extract_concepts("rash nahi hai")`` returned ``{"rash"}``. Negators are now
    split three ways -- pre-posed (scope forward), post-posed (scope backward) and
    epistemic hedges like "confirm nahi kar sakta" (scope forward over the
    complement clause). Grounded outputs are overwhelmingly hedges, so this was
    inflating factuality directly.

The chest-X-ray concepts from ``evaluate_h1.py`` (cardiomegaly, atelectasis,
consolidation, opacity, pneumonia) are deliberately EXCLUDED: they belong to the
abandoned Open-i corpus and cannot occur in MultiCaRe dermatology/ENT cases.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------------
# The lexicon: union of the four MultiCaRe-era copies.
# --------------------------------------------------------------------------

CONCEPT_PATTERNS: dict[str, list[str]] = {
    "allergy": ["allergic", "allergy", "angioedema", "hypersensitivity", "urticaria"],
    "autoimmune": ["autoimmune", "lupus", "rheumatoid", "vasculitis"],
    "bacterial": ["bacteria", "bacterial", "mrsa", "staph", "strep"],
    "conjunctivitis": ["conjunctival", "conjunctivitis", "pink eye"],
    "cyanosis": ["blue discoloration", "bluish", "cyanosis", "cyanotic"],
    "dermatitis": ["dermatitis", "dermatologic", "eczema"],
    "effusion": ["effusion", "fluid accumulation", "fluid collection"],
    "erythema": ["erythema", "erythematous", "red", "redness"],
    "fever": ["febrile", "fever", "hyperthermia", "pyrexia"],
    "fracture": ["broken", "fracture", "fractured"],
    "fungal": ["candida", "dermatophyte", "fungal", "fungus", "mycosis", "tinea"],
    "infection": ["abscess", "infected", "infection", "infectious", "sepsis", "septic"],
    "inflammation": ["cellulitis", "inflamed", "inflammation", "inflammatory"],
    "keratitis": ["corneal", "keratitis"],
    "lesion": ["lesion", "lesions", "macule", "nodule", "papule", "plaque"],
    "lymphadenopathy": ["lymph node", "lymph nodes", "lymphadenitis", "lymphadenopathy"],
    "malignancy": ["cancer", "carcinoma", "lymphoma", "malignancy", "malignant", "sarcoma"],
    "mass": ["growth", "lump", "mass", "neoplasm", "tumor", "tumour"],
    "necrosis": ["gangrene", "gangrenous", "necrosis", "necrotic"],
    "pain": ["ache", "pain", "painful", "tender", "tenderness"],
    "pruritus": ["itch", "itching", "itchy", "pruritic", "pruritus"],
    "rash": ["eruption", "exanthem", "maculopapular", "rash", "rashes"],
    "swelling": ["edema", "enlargement", "oedema", "swelling", "swollen", "tumefaction"],
    "tonsillitis": ["peritonsillar", "pharyngitis", "tonsil", "tonsillar", "tonsillitis"],
    "ulcer": ["aphthous", "ulcer", "ulceration", "ulcerative"],
    "viral": ["herpes", "hpv", "varicella", "viral", "virus", "wart"],
}

#: Patterns that are genuine word SUFFIXES. A plain ``\b`` prefix would break
#: these -- ``\balgia\b`` never matches *neuralgia* / *myalgia* / *otalgia*.
SUFFIX_PATTERNS: dict[str, list[str]] = {
    "pain": ["algia"],
}

#: Process/meta terms. Present in the source lexicons but excluded from
#: factuality scoring -- "biopsy" is not a clinical finding, so an answer
#: mentioning it is neither supported nor hallucinated evidence.
NON_FINDING_CONCEPTS: frozenset[str] = frozenset({
    "acute", "benign", "biopsy", "chronic", "congestion",
    "diagnosis", "imaging", "surgery", "treatment",
})

POSITIVE_CONCEPTS: frozenset[str] = frozenset(CONCEPT_PATTERNS)

#: English negators are PRE-posed ("no rash", "without swelling") so they scope
#: FORWARD. Hindi/Hinglish negators are POST-posed -- the verb-final negator follows
#: what it negates ("rash nahi hai" = "rash not is") -- so they must scope BACKWARD.
#: Scoping only forward silently counted every Hinglish denial as an assertion:
#: `extract_concepts("rash nahi hai")` returned {"rash"}. That mattered a great deal,
#: because grounded outputs are largely hedges of exactly this shape.
PREPOSED_NEGATORS: tuple[str, ...] = (
    "no", "not", "without", "denies", "denied", "negative for", "absent",
    "free of", "ruled out", "bina",
)
POSTPOSED_NEGATORS: tuple[str, ...] = (
    "nahi", "nahin", "nai", "na", "bilkul nahi", "koi nahi", "kabhi nahi",
    "nahi hai", "nahi tha", "nahi kar", "nahi de",
)

#: Epistemic HEDGES negate the speaker's commitment, not the predicate, so they
#: scope FORWARD over their complement clause even though the negator is Hindi:
#: "main confirm nahi kar sakta ki ulcer hai" = "I cannot confirm that there is an
#: ulcer" -- the ulcer is not asserted. Kept separate from POSTPOSED_NEGATORS
#: because those scope backward; applying one rule to both either misses hedges or
#: swallows genuine later assertions.
#: This distinction is not cosmetic: grounded outputs are overwhelmingly hedges of
#: exactly this shape, so mis-scoping them inflates factuality directly.
HEDGE_NEGATORS: tuple[str, ...] = (
    "confirm nahi", "nahi kar sakta", "nahi kar sakti", "nahi de sakta",
    "nahi de sakti", "nahi bata sakta", "pata nahi", "maloom nahi",
    "cannot confirm", "can not confirm", "unable to confirm", "not able to confirm",
)

#: All negators, for callers that just want the vocabulary.
NEGATORS: tuple[str, ...] = PREPOSED_NEGATORS + POSTPOSED_NEGATORS + HEDGE_NEGATORS

#: How far a negator scopes, in characters.
NEGATION_WINDOW = 40

#: Hedges scope over a whole complement clause, so they need a wider window.
HEDGE_WINDOW = 80


@lru_cache(maxsize=1)
def _compiled() -> dict[str, list[re.Pattern[str]]]:
    """Compile one word-boundary regex per pattern, per concept."""
    out: dict[str, list[re.Pattern[str]]] = {}
    for concept, pats in CONCEPT_PATTERNS.items():
        rxs = [re.compile(rf"\b{re.escape(p)}\b", re.I) for p in pats]
        rxs += [re.compile(rf"\w*{re.escape(s)}\b", re.I)
                for s in SUFFIX_PATTERNS.get(concept, [])]
        out[concept] = rxs
    return out


@lru_cache(maxsize=3)
def _negation_rx(kind: str) -> re.Pattern[str]:
    words = {"pre": PREPOSED_NEGATORS, "post": POSTPOSED_NEGATORS,
             "hedge": HEDGE_NEGATORS}[kind]
    alt = "|".join(re.escape(n) for n in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b({alt})\b", re.I)


def _negated_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges under the scope of a negator.

    Pre-posed negators scope forward from themselves; post-posed negators scope
    backward. A concept mention is suppressed if it falls inside any such range.
    """
    spans: list[tuple[int, int]] = []
    for m in _negation_rx("pre").finditer(text):
        spans.append((m.start(), m.end() + NEGATION_WINDOW))
    for m in _negation_rx("post").finditer(text):
        spans.append((max(0, m.start() - NEGATION_WINDOW), m.end()))
    # Hedges scope forward over the complement clause, and further than a plain
    # negator because the complement is a whole clause ("...ki X hai").
    for m in _negation_rx("hedge").finditer(text):
        spans.append((m.start(), m.end() + HEDGE_WINDOW))
    return spans


def extract_concepts(text: str) -> set[str]:
    """Return the set of positively-asserted clinical concepts in ``text``.

    A concept is dropped if every one of its mentions falls inside the forward
    scope of a negator ("no rash", "rash nahi hai").
    """
    s = str(text)
    spans = _negated_spans(s)
    found: set[str] = set()
    for concept, rxs in _compiled().items():
        for rx in rxs:
            for m in rx.finditer(s):
                if not any(a <= m.start() < b for a, b in spans):
                    found.add(concept)
                    break
            if concept in found:
                break
    return found


def score(output: str, reference: str) -> dict[str, float]:
    """Score ``output`` against ``reference`` by concept overlap.

    Returns ``nan`` for every rate when the output asserts no concept -- there is
    nothing to measure. Callers must report ``output_has_concepts`` as a coverage
    diagnostic rather than substituting a constant.
    """
    o = extract_concepts(output)
    r = extract_concepts(reference)
    n_o, n_r = len(o), len(r)
    hit = len(o & r)

    if n_o == 0:
        factual = halluc = precision = recall = f1 = np.nan
    else:
        factual = precision = hit / n_o
        halluc = (n_o - hit) / n_o
        recall = hit / n_r if n_r else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if n_r and (precision + recall) > 0 else
              (np.nan if not n_r else 0.0))

    return {
        # `factual_support` is PRECISION and nothing else: hit / |output concepts|.
        # It has no recall term, so its degenerate optimum is a one-word answer --
        # measured, the constant answer "swelling" scores 0.719 against the caption
        # reference while the real grounded system scores 0.153. NEVER report
        # factual_support as an absolute quality score, and always ship a
        # constant-answer baseline row beside it (see src/evaluation/baselines.py).
        # Paired deltas remain valid; absolute levels do not.
        "factual_support": factual,
        "precision": precision,
        "recall": recall,
        # `hallucination` is EXACTLY 1 - factual_support (verified: min = max = 1.0
        # for every arm). It is a sign-flipped duplicate, not independent evidence.
        # Reporting both as separate findings double-counts one result -- which is
        # what the published "+73.5% factuality AND -44% hallucination" pair did.
        "hallucination": halluc,
        "concept_f1": f1,
        "n_output_concepts": n_o,
        "n_reference_concepts": n_r,
        "n_overlap": hit,
        "output_has_concepts": n_o > 0,
        "reference_has_concepts": n_r > 0,
    }


__all__ = [
    "CONCEPT_PATTERNS", "SUFFIX_PATTERNS", "POSITIVE_CONCEPTS",
    "NON_FINDING_CONCEPTS", "NEGATORS", "extract_concepts", "score",
]
