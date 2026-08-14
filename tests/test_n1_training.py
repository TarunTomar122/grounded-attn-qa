import torch
import torch.nn.functional as F

from scripts.train_needle_n1 import losses


def test_boundary_token_weights_match_weighted_cross_entropy() -> None:
    logits = torch.tensor([[[0.0, 0.0, -2.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.0, -1.0, 0.0, 0.0]]])
    target = torch.tensor([[2, 3, 1]])
    valid = torch.ones_like(target, dtype=torch.bool)

    _, ce, _, _, _, _, _ = losses(logits, target, valid, 0.0, 5.0, 7.0)
    per_token = F.cross_entropy(logits[0], target[0], reduction="none")

    torch.testing.assert_close(ce, (per_token * torch.tensor([5.0, 1.0, 7.0])).sum() / 13)
