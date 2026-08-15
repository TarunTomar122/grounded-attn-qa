import torch

from scripts.probe_interaction_gate import interaction_features, parse_slice


def test_interaction_features_pool_question_and_context() -> None:
    memory = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]])
    valid = torch.tensor([[True, True, True]])
    context = torch.tensor([[False, True, True]])

    features = interaction_features(memory, valid, context)

    expected = torch.tensor([[1.0, 2.0, 4.0, 6.0, 4.0, 12.0, 3.0, 4.0]])
    torch.testing.assert_close(features, expected)


def test_parse_slice() -> None:
    assert parse_slice("official:12:34") == ("official", 12, 34)
