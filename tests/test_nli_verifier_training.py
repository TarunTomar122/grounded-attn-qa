import torch

from scripts.train_needle_nli_verifier import nli_metrics


def test_nli_metrics_calibrate_support_against_refute_and_neutral() -> None:
    logits = torch.tensor([[0.0, 0.0, 4.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    labels = torch.tensor([2, 0, 1])

    metrics = nli_metrics(logits, labels)

    assert metrics["val/nli_accuracy"] == 1.0
    assert metrics["val/safe_answer_coverage"] == 1.0
    assert metrics["val/safe_false_answer_rate"] == 0.0
    assert metrics["val/support_f1"] == 1.0
    assert metrics["val/refute_f1"] == 1.0
    assert metrics["val/unknown_f1"] == 1.0
