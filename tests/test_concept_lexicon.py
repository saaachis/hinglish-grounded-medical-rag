"""Regression tests for the unified concept lexicon.

Each substring case below is a real bug in the five lexicons this module replaces.
"""
import math

import pytest

from src.evaluation.concept_lexicon import extract_concepts, score


@pytest.mark.parametrize("text,absent", [
    ("the procedure was required", "erythema"),      # "red" in "requi-red-"
    ("bleeding occurred overnight", "erythema"),     # "red" in "occur-red-"
    ("reduced swelling was noted", "erythema"),      # "red" in "-red-uced"
    ("the wound needed a stitch", "pruritus"),       # "itch" in "st-itch-"
    ("a massive hemorrhage", "mass"),                # "mass" in "-mass-ive"
    ("growth was considered", "erythema"),           # "red" in "conside-red-"
])
def test_substring_false_positives_are_gone(text, absent):
    assert absent not in extract_concepts(text)


@pytest.mark.parametrize("text,present", [
    ("diffuse redness of the conjunctiva", "erythema"),
    ("red patches on the arm", "erythema"),
    ("severe itching at night", "pruritus"),
    ("a firm mass in the neck", "mass"),
    ("patient reports neuralgia", "pain"),           # suffix pattern must still fire
    ("myalgia and fatigue", "pain"),
])
def test_true_positives_still_fire(text, present):
    assert present in extract_concepts(text)


@pytest.mark.parametrize("text,concept", [
    # English: negator PRECEDES the concept, so it scopes forward.
    ("no rash was seen", "rash"),
    ("without any swelling", "swelling"),
    ("denies fever", "fever"),
    ("negative for infection", "infection"),
    ("the biopsy was not malignant", "malignancy"),
    # Hinglish: negator FOLLOWS the concept, so it must scope backward.
    # Every one of these returned the concept as ASSERTED before the fix.
    ("rash nahi hai", "rash"),
    ("Patient ko edema nahi hai.", "swelling"),
    ("mujhe bukhar nahi hai", "fever"),
    ("is report mein infection nahi mila", "infection"),
    ("koi swelling nahi", "swelling"),
    ("main ye confirm nahi kar sakta ki ulcer hai", "ulcer"),
])
def test_negation_suppresses(text, concept):
    """Each case asserts on ITS OWN text.

    The previous version was `assert extract(text) == set() or "rash" not in
    extract("no rash was seen")`. The right disjunct is a constant True, so the
    whole assertion was vacuous and the suite passed while postposed Hinglish
    negation was completely unhandled.
    """
    assert concept not in extract_concepts(text)


@pytest.mark.parametrize("text,concept", [
    ("patient has a rash", "rash"),
    ("swelling nahi tha pehle, ab swelling hai", "swelling"),  # later assertion survives
])
def test_negation_does_not_over_suppress(text, concept):
    """The backward window must not swallow genuine assertions elsewhere."""
    assert concept in extract_concepts(text)


def test_no_magic_default():
    """An output with no concepts must yield nan, never 0.25."""
    r = score("thank you, please consult a doctor", "erythema and swelling noted")
    assert math.isnan(r["factual_support"])
    assert math.isnan(r["hallucination"])
    assert r["output_has_concepts"] is False


def test_scoring_arithmetic():
    r = score("rash and fever", "rash and swelling")
    assert r["n_output_concepts"] == 2
    assert r["n_overlap"] == 1
    assert r["factual_support"] == pytest.approx(0.5)
    assert r["hallucination"] == pytest.approx(0.5)


def test_lexicon_has_no_non_finding_concepts():
    from src.evaluation.concept_lexicon import CONCEPT_PATTERNS, NON_FINDING_CONCEPTS
    assert not (set(CONCEPT_PATTERNS) & NON_FINDING_CONCEPTS)
