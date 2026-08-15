import torch

from scripts.train_needle_span_binding import contrastive_loss, matched_pairs


def test_matched_pairs_recover_same_context_positive_and_negative() -> None:
    data = {
        "source_ids": torch.tensor([[9, 8, 1, 2, 3], [7, 6, 1, 2, 3]]),
        "source_lengths": torch.tensor([5, 5]),
        "context_start": torch.tensor([2, 2]),
        "gold_copy_positions": torch.tensor([[3, -1], [-1, -1]]),
        "answerable": torch.tensor([True, False]),
    }

    pairs = matched_pairs(data)

    assert pairs.tolist() == [[0, 1, 1]]


def test_contrastive_loss_prefers_the_supported_question_span_alignment() -> None:
    positive = torch.tensor([0.8])
    negative = torch.tensor([0.2])

    assert contrastive_loss(positive, negative) < contrastive_loss(negative, positive)


def test_lower_temperature_strengthens_a_small_ranking_error() -> None:
    positive = torch.tensor([0.86])
    negative = torch.tensor([0.87])

    assert contrastive_loss(positive, negative, margin=0.05, temperature=0.1) > contrastive_loss(positive, negative, margin=0.05, temperature=1.0)
