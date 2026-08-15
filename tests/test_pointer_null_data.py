import torch

from scripts.prepare_needle_pointer_null import filter_pointer_null_rows


def test_pointer_null_filter_drops_only_unlocalizable_positive_rows() -> None:
    data = {
        "source_ids": torch.tensor([[1], [2], [3]]),
        "gold_copy_positions": torch.tensor([[0], [-1], [-1]]),
        "answerable": torch.tensor([True, True, False]),
    }

    filtered = filter_pointer_null_rows(data)

    assert filtered["source_ids"].flatten().tolist() == [1, 3]
    assert filtered["answerable"].tolist() == [True, False]
