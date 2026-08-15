from pathlib import PosixPath

import torch

from scripts.evaluate_candidate_verifier import candidate_offset, is_answerable, load_head_state, load_verifier_state


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
