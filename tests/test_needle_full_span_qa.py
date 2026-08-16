import torch

from grounded_qa.needle_full_span_qa import NeedleFullSpanNullModel
from grounded_qa.needle_span_qa import best_spans, span_null_loss
from grounded_qa.needleish import NeedleConfig


def tiny_config() -> NeedleConfig:
    return NeedleConfig(
        model_name="full-span-test",
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


def test_full_span_null_uses_decoder_and_masks_non_context_positions() -> None:
    model = NeedleFullSpanNullModel(tiny_config())
    source = torch.tensor([[4, 5, 6, 7, 8, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, True, True, False, False, False]])
    context = torch.tensor([[False, False, True, True, True, False, False, False]])
    output = model(source, valid, context)

    assert output.start_logits.shape == (1, 9)
    assert output.end_logits.shape == (1, 9)
    assert output.decoder_hidden.shape == (1, 2, 8)
    assert torch.isfinite(output.start_logits[:, 0]).all()
    assert torch.isfinite(output.start_logits[:, 3:6]).all()
    assert torch.isneginf(output.start_logits[:, 1:3]).all()
    assert torch.isneginf(output.start_logits[:, 6:]).all()


def test_full_span_null_loss_reaches_decoder_cross_attention() -> None:
    model = NeedleFullSpanNullModel(tiny_config())
    source = torch.tensor([[4, 5, 6, 7, 8, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, True, True, False, False, False]])
    context = torch.tensor([[False, False, True, True, True, False, False, False]])
    output = model(source, valid, context)
    loss, _, _ = span_null_loss(output, torch.tensor([3]), torch.tensor([4]))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.backbone.decoder[0].encoder_attn.q_proj.weight.grad is not None
    assert model.start_pointer.pointer_q.weight.grad is not None
    assert model.end_pointer.pointer_q.weight.grad is not None
    start, end, _, _ = best_spans(output)
    assert start.shape == end.shape == (1,)
