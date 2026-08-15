from scripts.prepare_needle_n3_reader_candidates import is_supported_candidate


def test_reader_candidate_support_requires_an_answerable_exact_span() -> None:
    assert is_supported_candidate({"answerable": True, "prediction": "Comet Bay.", "answers": ["Comet Bay"]})
    assert not is_supported_candidate({"answerable": True, "prediction": "North Pier", "answers": ["Comet Bay"]})
    assert not is_supported_candidate({"answerable": False, "prediction": "Comet Bay", "answers": [""]})
