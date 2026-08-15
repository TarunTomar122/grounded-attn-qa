import torch

from scripts.train_needle_candidate_verifier import evaluate


def test_evaluate_reports_safe_verifier_metrics() -> None:
    class Model:
        def encode(self, source, valid):
            return source.float()[..., None].repeat(1, 1, 512)

    head = torch.nn.Linear(2048, 1)
    head.weight.data.zero_()
    head.bias.data.zero_()
    data = {
        "source_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "source_lengths": torch.tensor([2, 2], dtype=torch.int32),
        "context_start": torch.tensor([1, 1], dtype=torch.int32),
        "answerable": torch.tensor([True, False]),
    }
    result = evaluate(Model(), head, data, torch.device("cpu"), 2)
    assert set(result) == {"val/bce", "val/auc", "val/safe_threshold", "val/safe_answer_coverage", "val/safe_false_answer_rate"}
