import pytest
import torch

from scripts.train_needle_n2_pointer import add_copy_eos_sentinel, batch
from grounded_qa.needle_tokenizer import EOS_ID


def _data() -> dict[str, torch.Tensor]:
    return {
        "source_ids": torch.tensor([[10, 11, 0, 0], [20, 21, 22, 0]], dtype=torch.int16),
        "source_lengths": torch.tensor([2, 3], dtype=torch.int16),
        "context_start": torch.tensor([1, 1], dtype=torch.int16),
        "target_ids": torch.tensor([[11, EOS_ID, 0], [EOS_ID, 0, 0]], dtype=torch.int16),
        "target_lengths": torch.tensor([2, 1], dtype=torch.int16),
        "gold_copy_positions": torch.tensor([[1, -1, -1], [-1, -1, -1]], dtype=torch.int16),
        "answerable": torch.tensor([True, False]),
    }


def test_sentinel_updates_both_labels_and_batch_context_without_mutating_input() -> None:
    data = _data()
    _, default_valid, default_context, _, _, _, default_gold, _ = batch(
        data, torch.tensor([0, 1]), torch.device("cpu")
    )
    assert not default_valid[0, 2]
    assert not default_context[0, 2]
    assert default_gold.tolist() == [[1, -1, -1], [-1, -1, -1]]

    transformed = add_copy_eos_sentinel(data)

    assert data["source_ids"].tolist() == [[10, 11, 0, 0], [20, 21, 22, 0]]
    assert transformed["source_ids"].tolist() == [[10, 11, EOS_ID, 0], [20, 21, 22, EOS_ID]]
    assert transformed["source_lengths"].tolist() == [3, 4]
    assert transformed["gold_copy_positions"].tolist() == [[1, 2, -1], [3, -1, -1]]

    _, source_valid, context_mask, _, _, _, _, _ = batch(
        transformed, torch.tensor([0, 1]), torch.device("cpu")
    )
    assert source_valid.tolist() == [[True, True, True, False], [True, True, True, True]]
    assert context_mask.tolist() == [[False, True, True, False], [False, True, True, True]]


def test_sentinel_rejects_rows_without_a_spare_padded_slot() -> None:
    data = _data()
    data["source_lengths"] = torch.tensor([4, 3], dtype=torch.int16)

    with pytest.raises(ValueError, match="spare padded source slot"):
        add_copy_eos_sentinel(data)
