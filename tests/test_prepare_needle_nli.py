from scripts.prepare_needle_nli import label_id


def test_snli_labels_map_to_project_nli_order() -> None:
    assert [label_id(value) for value in (0, 1, 2, -1)] == [2, 0, 1, None]
    assert [label_id(value) for value in ("entailment", "neutral", "contradiction", "-")] == [2, 0, 1, None]
