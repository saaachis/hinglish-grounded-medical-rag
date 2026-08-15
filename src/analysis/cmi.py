"""Code-Mixing Index (CMI) -- corrected implementation.

The legacy measure (`src/prototype/run_h1h2_analysis.py:38`) is a plain ratio of
tokens found in a 117-token hand-written Hindi list to all Latin tokens. It has
two compounding biases, both measured on the 3,015 MMCQSD queries:

  OVER-COUNTING   the list contains the English words `doctor` (fires in 71.0%
                  of queries), `please` (38.0%) and `pls` (0.5%).
  UNDER-COUNTING  32.9% of all query tokens are absent from both the Hindi list
                  and an English vocabulary -- and the frequency-ranked OOV list
                  is almost entirely romanised Hindi (`mein` 5,340 occurrences,
                  `mere` 3,945, `hoon` 2,572, `gaya`, `koi`, `saal`, ...).

Both biases push scores toward the middle, which is why the legacy CMI has a
standard deviation of only ~0.075 and why its tertiles are all mid-range.

This module implements the standard Das & Gamback (2014) formulation:

    CMI = 100 * (1 - max(w_i) / n)

where n is the number of LANGUAGE-TAGGED tokens (language-independent tokens are
excluded from the denominator) and w_i is the token count of language i. The
range is 0 (monolingual) to 50 (perfectly balanced bilingual).

Tokens are tagged by lookup:
  * HINDI_LEXICON  -- curated romanised Hindi, contaminants removed and expanded
                      by frequency-ranked manual classification of OOV tokens.
  * ENGLISH_VOCAB  -- derived from the MultiCaRe English clinical corpus.
  * AMBIGUOUS      -- Hindi/English homographs (`me`, `sir`, `pet`, `pair`).
                      Treated as language-independent by default.

Because the AMBIGUOUS assignment is a judgement call, `cmi()` accepts a policy so
the choice can be reported as a sensitivity analysis rather than an assumption:
`exclude` (default), `hindi`, or `english`. If a conclusion holds under all
three, it does not depend on the tagging decision.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MULTICARE_PATH = Path("data/processed/multicare_filtered.csv")
VOCAB_CACHE = Path("data/processed/english_vocab.txt")

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# --------------------------------------------------------------------------
# Hindi lexicon
# --------------------------------------------------------------------------
# Legacy list minus `doctor`, `please`, `pls` (unambiguously English), and minus
# the homographs moved to AMBIGUOUS below.
_LEGACY_HINDI = {
    "kya", "hai", "mujhe", "meri", "mera", "kripya", "saans", "khansi",
    "bukhar", "dard", "batao", "samjhao", "ho", "raha", "rahi", "hain",
    "nahi", "aur", "bhi", "ko", "se", "ke", "ki", "ka", "ye", "wo", "ek",
    "bahut", "thoda", "zyada", "abhi", "pehle", "baad", "dono", "uske",
    "iske", "par", "pe", "thi", "tha", "hota", "hoti", "lag", "laga",
    "karna", "karke", "kuch", "jo", "hum", "kaise", "kyun", "kab", "kahan",
    "kitna", "kitni", "kitne", "sab", "yeh", "woh", "apna", "apni", "apne",
    "unke", "unki", "unka", "hamara", "hamari", "tumhara", "pata", "liye",
    "wala", "wali", "wale", "jaise", "lekin", "agar", "toh", "phir", "hona",
    "paani", "khana", "peena", "sona", "uthna", "chalna", "dekhna", "bolna",
    "sunna", "milna", "rehna", "jaana", "aana", "lena", "dena",
    "kaan", "aankhon", "aankhein", "naak", "gala", "seena", "kamar",
    "haath", "pasina", "sujan", "khujli", "thakan", "chakkar", "ulti", "dast",
    "kabz",
}

# Added by frequency-ranked manual classification of out-of-vocabulary tokens
# (every token below appeared >=150 times across the 3,015 queries and was
# individually confirmed romanised Hindi).
_OOV_HINDI = {
    "mein", "mere", "hoon", "gaya", "kar", "koi", "neeche", "gayi", "kiya",
    "saal", "hua", "din", "tak", "gaye", "jab", "sakta", "paas", "dekhiye",
    "kabhi", "vartaman", "karne", "uski", "chahiye", "hafte", "dekhein",
    "mahine", "rahe", "lagta", "hui", "suj", "kam", "shuru", "diya", "wajah",
    "sirf", "chote", "usko", "saath", "tarah", "hone", "unhone", "laal",
    "kaha", "sakte", "taraf", "lagbhag", "uska", "karta", "aaj", "madad",
    "kisi", "upar", "samay", "raat", "iska", "kyunki", "namaste", "gardan",
    "jyada", "pichle", "usne", "sakti", "jata", "karti", "chinta", "unhe",
    "kal", "andar", "ganth", "diye", "liya", "chehre", "isliye", "karan",
    "theek", "muh", "jagah", "aas", "aankh", "teen", "mehsoos", "subah",
    "samasya", "beti", "gale", "haal", "hote", "karte", "har", "hamesha",
    "badh", "shayad", "kaam", "isse", "ghante", "dhanyavaad", "jaisa",
    "beech", "ja", "thodi", "aisa", "bahar", "dekha", "chala", "sharir",
    "karein", "cheez", "bada", "baare", "jati", "bataya", "dekhe", "gayab",
    "wapas", "iski", "alag", "aapko", "thik", "pareshan", "piche",
}

HINDI_LEXICON: set[str] = _LEGACY_HINDI | _OOV_HINDI

# Hindi/English homographs. Both readings occur in this corpus, and they cannot
# be resolved without context, so by default they are language-independent.
#   me   -> Hindi "in" / English "me"
#   sir  -> Hindi "head" / English honorific
#   pet  -> Hindi "stomach" / English animal
#   pair -> Hindi "leg" / English "a pair"
AMBIGUOUS: set[str] = {"me", "sir", "pet", "pair"}

# Removed from the legacy list as unambiguously English.
REMOVED_AS_ENGLISH: set[str] = {"doctor", "please", "pls"}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def build_english_vocab(
    multicare_path: Path = MULTICARE_PATH,
    cache_path: Path = VOCAB_CACHE,
    min_freq: int = 5,
    max_rows: int = 20_000,
) -> set[str]:
    """English vocabulary from the MultiCaRe English clinical corpus.

    Cached to disk -- the scan reads a 260 MB CSV.
    """
    if cache_path.exists():
        vocab = set(cache_path.read_text(encoding="utf-8").split())
        logger.info("Loaded cached English vocab (%d types)", len(vocab))
        return vocab

    if not multicare_path.exists():
        raise FileNotFoundError(
            f"{multicare_path} not found -- needed to build the English vocabulary."
        )

    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(
        multicare_path, usecols=["case_text"], chunksize=5_000, nrows=max_rows
    ):
        for txt in chunk["case_text"]:
            counts.update(tokenize(txt))

    vocab = {w for w, c in counts.items() if c >= min_freq}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(sorted(vocab)), encoding="utf-8")
    logger.info("Built English vocab (%d types, min_freq=%d)", len(vocab), min_freq)
    return vocab


def tag_token(token: str, english_vocab: set[str], ambiguous_policy: str = "exclude") -> str | None:
    """Tag a token as 'hi', 'en', or None (language-independent).

    The Hindi lexicon takes precedence over the English vocabulary: the corpus is
    Hinglish, and clinical abbreviations pollute the English side (`par` is also
    "population attributable risk", `ho` is also "heterotopic ossification"), so
    English-vocab membership alone is not evidence a token is English here.
    """
    if token in AMBIGUOUS:
        if ambiguous_policy == "hindi":
            return "hi"
        if ambiguous_policy == "english":
            return "en"
        return None
    if token in HINDI_LEXICON:
        return "hi"
    if token in english_vocab:
        return "en"
    return None


def cmi(
    text: str,
    english_vocab: set[str],
    ambiguous_policy: str = "exclude",
) -> float:
    """Das & Gamback CMI in [0, 50]. 0 = monolingual, 50 = perfectly balanced."""
    tags = [tag_token(t, english_vocab, ambiguous_policy) for t in tokenize(text)]
    tagged = [t for t in tags if t is not None]
    n = len(tagged)
    if n == 0:
        return 0.0
    counts = Counter(tagged)
    return 100.0 * (1.0 - max(counts.values()) / n)


def hindi_proportion(
    text: str,
    english_vocab: set[str],
    ambiguous_policy: str = "exclude",
) -> float:
    """Fraction of language-tagged tokens that are Hindi, in [0, 1].

    This is a DIFFERENT CONSTRUCT from `cmi()` and the distinction is essential.

    `cmi()` (Das & Gamback) measures mixing BALANCE: it peaks at a 50/50 mix and
    is 0 for monolingual text in EITHER language. `hindi_proportion` measures how
    much Hindi is present, and is 0 for English and 1 for Hindi.

    A 90%-Hindi query therefore scores HIGH on hindi_proportion and LOW on CMI.
    The two are negatively correlated on this corpus (Spearman rho ~ -0.53), so
    substituting one for the other silently changes the research question.

    For the hypothesis "does more Hindi degrade an English-centric pipeline?",
    proportion is the appropriate construct. For "does alternation itself cost
    performance?", CMI is. Report both and say which is which.
    """
    tags = [tag_token(t, english_vocab, ambiguous_policy) for t in tokenize(text)]
    tagged = [t for t in tags if t is not None]
    if not tagged:
        return 0.0
    return sum(t == "hi" for t in tagged) / len(tagged)


def cmi_legacy(text: str) -> float:
    """The original measure, reproduced verbatim for comparison."""
    legacy_set = _LEGACY_HINDI | AMBIGUOUS | REMOVED_AS_ENGLISH | {"me", "pet", "sir", "pair"}
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(t in legacy_set for t in tokens) / len(tokens)
