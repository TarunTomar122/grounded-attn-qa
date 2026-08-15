from scripts.prepare_needle_n3_reader_candidates import is_supported_candidate, nli_label, usable_claim


def test_reader_candidate_support_requires_an_answerable_exact_span() -> None:
    assert is_supported_candidate({"answerable": True, "prediction": "Comet Bay.", "answers": ["Comet Bay"]})
    assert not is_supported_candidate({"answerable": True, "prediction": "North Pier", "answers": ["Comet Bay"]})
    assert not is_supported_candidate({"answerable": False, "prediction": "Comet Bay", "answers": [""]})


def test_nli_labels_distinguish_support_refute_and_neutral() -> None:
    assert nli_label({"answerable": True, "prediction": "Comet Bay", "answers": ["Comet Bay"]}) == 2
    assert nli_label({"answerable": True, "prediction": "North Pier", "answers": ["Comet Bay"]}) == 1
    assert nli_label({"answerable": False, "prediction": "Comet Bay", "answers": [""]}) == 0


def test_usable_claim_requires_candidate_and_negation_fidelity() -> None:
    assert usable_claim("Where is Carmen?", "Abruzzo", "Carmen is in Abruzzo.")
    assert not usable_claim("Where is Carmen?", "Abruzzo", "Carmen is abroad.")
    assert not usable_claim("Where isn't Carmen?", "Abruzzo", "Carmen is in Abruzzo.")
