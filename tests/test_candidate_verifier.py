from pathlib import PosixPath

import torch

from scripts.evaluate_candidate_verifier import candidate_offset, is_answerable, load_head_state, load_verifier_state, safe_gate_summary
from scripts.sweep_candidate_risk import sweep


def test_candidate_offset_is_case_insensitive() -> None:
    assert candidate_offset("Comet Bay has tag CB-918.", "cb-918") == 18
    assert candidate_offset("Comet Bay has tag CB-918.", "missing") == -1


def test_is_answerable_uses_explicit_label_then_synthetic_condition() -> None:
    assert is_answerable({"answerable": True, "condition": "wrong"})
    assert not is_answerable({"answerable": False, "condition": "correct"})
    assert is_answerable({"condition": "correct"})
    assert is_answerable({"condition": "counterfactual"})
    assert not is_answerable({"condition": "wrong"})


def test_load_head_state_accepts_checkpoint_path_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    expected = {"weight": torch.ones((1, 2)), "bias": torch.zeros(1)}
    torch.save({"head": expected, "args": {"output_dir": PosixPath("runs/example")}}, checkpoint)
    actual = load_head_state(checkpoint, torch.device("cpu"))
    assert torch.equal(actual["weight"], expected["weight"])


def test_load_verifier_state_accepts_checkpoint_path_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    expected = {"weight": torch.ones((3, 2)), "bias": torch.zeros(3)}
    torch.save({"verifier": expected, "args": {"output_dir": PosixPath("runs/example")}}, checkpoint)
    actual = load_verifier_state(checkpoint, torch.device("cpu"))
    assert torch.equal(actual["weight"], expected["weight"])


def test_safe_gate_summary_counts_wrong_reader_answers_as_risk() -> None:
    rows = [
        {"answerable": True, "em": 1.0, "candidate_accepted": True},
        {"answerable": True, "em": 1.0, "candidate_accepted": False},
        {"answerable": True, "em": 0.0, "candidate_accepted": True},
        {"answerable": False, "em": 0.0, "candidate_accepted": False},
    ]

    summary = safe_gate_summary(rows)

    assert summary["safe_answer_coverage"] == 1 / 3
    assert summary["accepted_answer_risk"] == 0.5
    assert summary["wrong_reader_answer_accept_rate"] == 0.5


def test_risk_sweep_changes_the_gate_threshold() -> None:
    rows = [
        {"answerable": True, "em": 1.0, "candidate_probability": 0.8},
        {"answerable": True, "em": 0.0, "candidate_probability": 0.2},
    ]

    curve = sweep(rows, points=3)

    assert curve[0]["accepted"] == 2
    assert curve[-1]["accepted"] == 0
    assert curve[1]["accepted_answer_risk"] == 0.0
