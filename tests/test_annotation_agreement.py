import pandas as pd
import pytest

from src.evaluation import annotation_agreement as aa


def test_parse_list_handles_blanks_and_placeholders():
    assert aa.parse_list(None) == set()
    assert aa.parse_list("(none)") == set()
    assert aa.parse_list("Rash, swelling*, ") == {"rash", "swelling"}


def test_kappa_perfect_and_chance():
    assert aa.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # complete disagreement on a balanced binary task is kappa = -1
    assert aa.cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


def test_quadratic_kappa_penalises_distance_less_than_unweighted():
    a = [0, 1, 2, 0, 1, 2]
    b = [0, 1, 1, 0, 1, 2]        # one adjacent disagreement
    assert aa.cohens_kappa(a, b, "quadratic") > aa.cohens_kappa(a, b)


def _sheet_a(tmp_path, who, wrong, missed, appropriateness):
    df = pd.DataFrame({
        "item_id": ["A001", "A002"],
        "patient_question": ["q1", "q2"],
        "answer_text": ["a1", "a2"],
        "extractor_concepts": ["rash, swelling", "pain"],
        "wrong_concepts": wrong,
        "missed_concepts": missed,
        "appropriateness": appropriateness,
        "_hidden_arm": ["grounded", "zero_shot"],
        "_hidden_pair_id": [1, 2],
        "_hidden_tertile": ["low", "high"],
    })
    df.to_csv(tmp_path / f"annotation_sheet_A_{who}.csv", index=False)


def test_extractor_precision_recall_from_human_labels(tmp_path, monkeypatch):
    # Item 1: extractor found {rash, swelling}, one is wrong -> 1 TP, 1 FP
    # Item 2: extractor found {pain}, correct, and missed {fever} -> 1 TP, 1 FN
    _sheet_a(tmp_path, "annotatorA", ["swelling", ""], ["", "fever"], [2, 1])
    monkeypatch.setattr(aa, "DIR", tmp_path)
    L = []
    aa.score_sheet_a(aa.load("A"), L)
    text = "\n".join(L)
    assert "| annotatorA | 2 | 0.667 | 0.667 | 0.667 |" in text  # tp=2, fp=1, fn=1


def test_sheet_a_reports_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(aa, "DIR", tmp_path)
    L = []
    aa.score_sheet_a(aa.load("A"), L)
    assert "No completed sheet A found" in "\n".join(L)


def test_sheet_b_agreement_with_automatic_label(tmp_path, monkeypatch):
    pd.DataFrame({
        "item_id": ["B001", "B002", "B003", "B004"],
        "patient_question": ["q"] * 4,
        "gold_english_question": ["q"] * 4,
        "retrieved_case": ["c"] * 4,
        "relevance": [2, 0, 1, 0],
        "_hidden_pair_id": [1, 1, 1, 2],
        "_hidden_query_variant": ["Q1_hinglish"] * 3 + ["Q2_english_question"],
        "_hidden_rank": [1, 2, 3, 1],
        "_hidden_case_row": [10, 11, 12, 13],
        "_hidden_same_group": [True, False, True, False],
    }).to_csv(tmp_path / "annotation_sheet_B_annotatorA.csv", index=False)
    monkeypatch.setattr(aa, "DIR", tmp_path)
    L = []
    aa.score_sheet_b(aa.load("B"), L)
    text = "\n".join(L)
    # every row's automatic label matches the human "relevant or not" call
    assert "agrees with the human judgment on **100.0%**" in text
    assert "Judged P@3" in text
