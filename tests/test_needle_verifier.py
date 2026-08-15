import torch
import torch.nn.functional as F
from torch import nn

from grounded_qa.needle_pointer import NeedlePointerModel
from grounded_qa.needle_verifier import NeedleVerifierAdapter, joint_verifier_logits
from grounded_qa.needleish import NeedleConfig


def test_zero_initialized_adapters_preserve_frozen_reader_encoding() -> None:
    cfg = NeedleConfig(model_name="test", vocab_size=16, d_model=8, encoder_layers=2, decoder_layers=1, query_heads=2, kv_heads=1, head_dim=4, source_length=8, target_length=4)
    reader = NeedlePointerModel(cfg).eval()
    verifier = NeedleVerifierAdapter(reader, rank=2).eval()
    source = torch.tensor([[2, 3, 4]])
    valid = torch.ones_like(source, dtype=torch.bool)

    assert torch.allclose(reader.encode(source, valid), verifier.encode(source, valid))
    assert not any(parameter.requires_grad for parameter in reader.parameters())
    assert verifier.trainable_parameters == 64


def test_zero_initialized_decoder_adapter_preserves_reader_cross_attention() -> None:
    cfg = NeedleConfig(model_name="test", vocab_size=16, d_model=8, encoder_layers=2, decoder_layers=1, query_heads=2, kv_heads=1, head_dim=4, source_length=8, target_length=4)
    reader = NeedlePointerModel(cfg).eval()
    verifier = NeedleVerifierAdapter(reader, rank=2, decoder=True).eval()
    source = torch.tensor([[2, 3, 4]])
    valid = torch.ones_like(source, dtype=torch.bool)
    target = torch.full((1, 1), cfg.bos_id)
    expected = reader.decode_hidden(target, reader.encode(source, valid), valid, torch.ones_like(target, dtype=torch.bool)).squeeze(1)

    assert torch.allclose(verifier.verify(source, valid, valid), expected)
    assert verifier.trainable_parameters == 96


def test_joint_verifier_loss_reaches_the_shared_reader() -> None:
    cfg = NeedleConfig(model_name="test", vocab_size=32, d_model=8, encoder_layers=1, decoder_layers=1, query_heads=2, kv_heads=1, head_dim=4, source_length=8, target_length=4)
    reader = NeedlePointerModel(cfg)
    head = nn.Linear(cfg.d_model * 4, 3)
    source = torch.tensor([[2, 3, 4, 5]])
    valid = torch.ones_like(source, dtype=torch.bool)
    question = torch.tensor([[True, True, False, False]])
    candidate = torch.tensor([[False, False, True, False]])

    logits = joint_verifier_logits(reader, head, source, valid, question, candidate)
    F.cross_entropy(logits, torch.tensor([2])).backward()

    assert reader.encoder[0].self_attn.q_proj.weight.grad is not None
