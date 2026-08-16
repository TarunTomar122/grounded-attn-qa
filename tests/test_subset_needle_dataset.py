from __future__ import annotations

import torch

from scripts.subset_needle_dataset import select_balanced_rows


def test_balanced_subset_is_exact_and_deterministic() -> None:
    data = {
        "source_ids": torch.arange(20).reshape(10, 2),
        "target_ids": torch.arange(30).reshape(10, 3),
        "answerable": torch.tensor([True, False, True, False, True, False, True, False, True, False]),
    }

    first = select_balanced_rows(data, count=2, seed=17)
    second = select_balanced_rows(data, count=2, seed=17)

    assert first["answerable"].tolist() == [True, True, False, False]
    assert int(first["answerable"].sum()) == 2
    assert int((~first["answerable"]).sum()) == 2
    assert all(torch.equal(first[key], second[key]) for key in data)
