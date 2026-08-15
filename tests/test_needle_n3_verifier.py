import random

from scripts.prepare_needle_n3_verifier import distractor, verifier_claim, verifier_query


def test_verifier_query_contains_question_and_candidate() -> None:
    assert verifier_query("Who arrived?", "Rhea") == "Question: Who arrived?\nCandidate answer: Rhea\nIs this candidate supported by the context?"


def test_verifier_claim_binds_candidate_to_question() -> None:
    assert verifier_claim("Who arrived?", "Rhea") == 'Claim: The answer to the question "Who arrived?" is "Rhea".\nDoes the evidence support, contradict, or leave this claim unknown?'


def test_distractor_excludes_gold_answer() -> None:
    value = distractor("Rhea arrived at noon beside the old gate.", "Rhea", random.Random(4))
    assert value is not None
    assert "rhea" not in value[0].lower()


def test_distractor_allows_empty_answer_for_unanswerable_questions() -> None:
    assert distractor("Rhea arrived at noon.", "", random.Random(4)) is not None
