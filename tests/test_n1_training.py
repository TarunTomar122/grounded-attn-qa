import torch
import torch.nn.functional as F

from scripts.train_needle_n1 import Muon, load_initial_checkpoint, losses, optimizers_for


def test_boundary_token_weights_match_weighted_cross_entropy() -> None:
    logits = torch.tensor([[[0.0, 0.0, -2.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.0, -1.0, 0.0, 0.0]]])
    target = torch.tensor([[2, 3, 1]])
    valid = torch.ones_like(target, dtype=torch.bool)

    _, ce, _, _, _, _, _ = losses(logits, target, valid, 0.0, 5.0, 7.0)
    per_token = F.cross_entropy(logits[0], target[0], reduction="none")

    torch.testing.assert_close(ce, (per_token * torch.tensor([5.0, 1.0, 7.0])).sum() / 13)


def test_muon_updates_only_dense_weights() -> None:
    model = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.Linear(4, 3, bias=False))
    optimizers = optimizers_for(model, adam_lr=3.0e-5, muon_lr=0.02)
    dense = model[1].weight
    muon_params = optimizers["muon"].param_groups[0]["params"]
    assert len(muon_params) == 1 and muon_params[0] is dense
    before = dense.detach().clone()
    model(torch.tensor([[1, 2]])).sum().backward()
    for optimizer in optimizers.values():
        optimizer.step()
    assert torch.isfinite(dense).all() and not torch.equal(before, dense)
    assert isinstance(optimizers["muon"], Muon)


def test_adapted_checkpoint_can_initialize_the_next_stage(tmp_path) -> None:
    source = torch.nn.Linear(3, 2, bias=False)
    checkpoint = tmp_path / "adapted.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    target = torch.nn.Linear(3, 2, bias=False)

    load_initial_checkpoint(target, checkpoint, torch.device("cpu"))

    torch.testing.assert_close(target.weight, source.weight)
