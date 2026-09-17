import pytest

from src.analysis.hinglish_devanagari import (
    LEXICON, contains_devanagari, rule_based, transliterate_text, transliterate_token,
)

VOCAB = {"skin", "doctor", "rash", "fever", "the", "is", "me", "in", "to", "so", "he", "a"}


def test_lexicon_entries_are_devanagari():
    for roman, dev in LEXICON.items():
        assert roman.isalpha() and roman.islower(), roman
        assert contains_devanagari(dev), (roman, dev)


def test_english_words_are_left_in_latin():
    out, counts = transliterate_text("skin par rash hai", VOCAB)
    assert "skin" in out and "rash" in out
    assert "है" in out                      # hai -> Devanagari
    assert counts["english"] == 2


def test_lexicon_wins_over_english_homograph():
    # "ko", "se", "tha" exist in an English wordlist but are Hindi here; without
    # explicit entries the transliterated query stays half-romanised.
    for word in ("ko", "se", "tha", "ki", "hain"):
        dev, src = transliterate_token(word, VOCAB | {word})
        assert src == "lexicon", word
        assert contains_devanagari(dev), word


def test_rule_based_restores_word_final_schwa():
    # plain ITRANS leaves a trailing virama on final consonants
    assert not rule_based("dard").endswith("्")


def test_proper_nouns_and_acronyms_stay_latin():
    out, _ = transliterate_text("mujhe Advil aur MRSA ke baare mein", VOCAB)
    assert "Advil" in out and "MRSA" in out


def test_sentence_initial_capital_is_still_transliterated():
    out, _ = transliterate_text("Mujhe dard hai", VOCAB)
    assert "मुझे" in out


def test_punctuation_and_digits_are_preserved():
    out, _ = transliterate_text("5 din se red/raw hai, dard bhi", VOCAB | {"red", "raw"})
    assert "5" in out and "red/raw" in out and "," in out


def test_counts_sum_to_token_count():
    text = "mujhe 3 din se skin par rash hai"
    _, counts = transliterate_text(text, VOCAB)
    assert sum(counts.values()) == len([w for w in text.split() if any(c.isalpha() for c in w)])


@pytest.mark.parametrize("roman,expected", [
    ("bukhar", "बुखार"), ("khansi", "खांसी"), ("pet", "पेट"), ("meri", "मेरी"),
])
def test_known_words_map_correctly(roman, expected):
    """These are exactly the words plain ITRANS gets wrong."""
    assert transliterate_token(roman, VOCAB)[0] == expected
