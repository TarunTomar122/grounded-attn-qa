from scripts.evaluate_candidate_verifier import candidate_offset


def test_candidate_offset_is_case_insensitive() -> None:
    assert candidate_offset("Comet Bay has tag CB-918.", "cb-918") == 18
    assert candidate_offset("Comet Bay has tag CB-918.", "missing") == -1
