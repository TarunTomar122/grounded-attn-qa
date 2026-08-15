import torch

from grounded_qa.needle_pointer import answerability_interaction_features, candidate_span_features, candidate_verifier_head
from scripts.probe_interaction_gate import parse_slice


def test_interaction_features_pool_question_and_context() -> None:
    memory = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]])
    valid = torch.tensor([[True, True, True]])
    context = torch.tensor([[False, True, True]])

    features = answerability_interaction_features(memory, valid, context)

    expected = torch.tensor([[1.0, 2.0, 4.0, 6.0, 4.0, 12.0, 3.0, 4.0]])
    torch.testing.assert_close(features, expected)


def test_candidate_span_features_use_the_proposed_source_span() -> None:
    memory = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]])
    valid = torch.tensor([[True, True, True]])
    question = torch.tensor([[True, False, False]])

    features = candidate_span_features(memory, valid, question, torch.tensor([[False, True, False]]))

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 3.0, 8.0, 2.0, 2.0]])
    torch.testing.assert_close(features, expected)


def test_candidate_verifier_head_can_add_a_small_nonlinear_readout() -> None:
    head = candidate_verifier_head(d_model=2, hidden_dim=3)
    assert head(torch.ones((4, 8))).shape == (4, 1)


def test_parse_slice() -> None:
    assert parse_slice("official:12:34") == ("official", 12, 34)
