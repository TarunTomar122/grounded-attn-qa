import torch

from grounded_qa.needle_span_qa import NeedleSpanNullModel, best_spans, span_null_loss, threshold_predictions
from grounded_qa.needleish import NeedleConfig


def tiny_config() -> NeedleConfig:
    return NeedleConfig(
        model_name="span-test",
        vocab_size=32,
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        query_heads=2,
        kv_heads=1,
        head_dim=4,
        source_length=8,
        target_length=4,
        dropout=0.0,
    )


def test_span_null_head_masks_query_and_pad_positions() -> None:
    model = NeedleSpanNullModel(tiny_config())
    source = torch.tensor([[4, 5, 6, 7, 8, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, True, True, False, False, False]])
    context = torch.tensor([[False, False, True, True, True, False, False, False]])
    output = model(source, valid, context)

    assert output.start_logits.shape == (1, 9)
    assert torch.isfinite(output.start_logits[:, 0]).all()
    assert torch.isfinite(output.start_logits[:, 3:6]).all()
    assert torch.isneginf(output.start_logits[:, 1:3]).all()
    assert torch.isneginf(output.start_logits[:, 6:]).all()


def test_span_null_loss_and_decode_support_null_index_zero() -> None:
    model = NeedleSpanNullModel(tiny_config())
    source = torch.tensor([[4, 5, 6, 7, 8, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, True, True, False, False, False]])
    context = torch.tensor([[False, False, True, True, True, False, False, False]])
    output = model(source, valid, context)
    loss, _, _ = span_null_loss(output, torch.tensor([0]), torch.tensor([0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.backbone.token_embedding.weight.grad is not None
    start, end, _, margin = best_spans(output)
    predicted_start, predicted_end = threshold_predictions(output, threshold=float(margin[0]) - 1.0)
    assert start.shape == end.shape == (1,)
    assert predicted_start.item() == predicted_end.item() == 0
