from scripts.train_fusion_calibrator import FEATURE_NAMES, feature_rows


def test_feature_rows_use_only_runtime_observable_signals() -> None:
    rows = [{
        "candidate_probability": 0.75,
        "generated_tokens": 3,
        "eos": True,
        "unsupported_number_rate": 0.0,
        "unsupported_entity_rate": 1.0,
        "context": "No candidate here.",
        "raw_prediction": "missing",
        "em": 1.0,
    }]

    features, labels = feature_rows(rows)

    assert FEATURE_NAMES == ("verifier_support", "generated_tokens", "eos", "unsupported_number", "unsupported_entity", "literal_candidate")
    assert features.tolist() == [[0.75, 3.0, 1.0, 0.0, 1.0, 0.0]]
    assert labels.tolist() == [1.0]
