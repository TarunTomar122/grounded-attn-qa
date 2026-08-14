from __future__ import annotations

import torch

from grounded_qa.needle_tokenizer import SPECIAL_TOKENS
from grounded_qa.needleish import GroupedQueryAttention, NeedleConfig, NeedleishModel
from grounded_qa.synth_data import encode_synth_row, source_bucket, split_for_source


def tiny_config() -> NeedleConfig:
    return NeedleConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        query_heads=4,
        kv_heads=2,
        head_dim=8,
        source_length=32,
        target_length=16,
        query_id=60,
        context_id=61,
        reasoning_id=62,
        answer_id=63,
    )


def test_model_is_ffn_free_and_tied() -> None:
    model = NeedleishModel(tiny_config())
    assert model.n_params() > 0
    assert not any(name.endswith("lm_head.weight") for name, _ in model.named_parameters())
    assert all(not isinstance(module, torch.nn.Linear) or module.out_features != 4 * tiny_config().d_model for module in model.modules())
    assert all(hasattr(module, "q_norm") and hasattr(module, "k_norm") for module in model.modules() if isinstance(module, GroupedQueryAttention))


def test_public_checkpoint_configuration_is_exact() -> None:
    cfg = NeedleConfig.public_checkpoint()
    model = NeedleishModel(cfg)
    assert (cfg.vocab_size, cfg.source_length, cfg.target_length) == (8192, 1024, 512)
    assert cfg.embedding_scale == 512**0.5
    assert not cfg.cross_attention_rope
    assert model.n_params() == 26_233_372


def test_forward_is_finite_and_has_vocab_projection() -> None:
    cfg = tiny_config()
    model = NeedleishModel(cfg)
    source = torch.randint(0, cfg.vocab_size, (2, 9))
    target = torch.randint(0, cfg.vocab_size, (2, 6))
    source_valid = torch.ones_like(source, dtype=torch.bool)
    target_valid = torch.ones_like(target, dtype=torch.bool)
    logits = model(source, source_valid, target, target_valid)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_decoder_self_attention_is_causal() -> None:
    cfg = tiny_config()
    attention = GroupedQueryAttention(cfg, causal=True)
    attention.eval()
    valid = torch.ones(1, 5, dtype=torch.bool)
    positions = torch.arange(5)
    first = torch.randn(1, 5, cfg.d_model)
    second = first.clone()
    second[:, 4] += 100
    out_first = attention(first, first, key_valid=valid, query_positions=positions, key_positions=positions)
    out_second = attention(second, second, key_valid=valid, query_positions=positions, key_positions=positions)
    assert torch.allclose(out_first[:, :4], out_second[:, :4], atol=1e-5)


def test_source_split_is_deterministic() -> None:
    urls = [f"https://example.test/{index}" for index in range(500)]
    assert all(source_bucket(url) == source_bucket(url) for url in urls)
    train = {url for url in urls if split_for_source(url) == "train"}
    validation = {url for url in urls if split_for_source(url) == "validation"}
    assert not train & validation


def test_synth_encoding_preserves_answer_and_regions() -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(text) % 17 + 1]

        def encode_source(self, query: str, context: str) -> list[int]:
            return [SPECIAL_TOKENS["<QUERY>"], 7, SPECIAL_TOKENS["<CONTEXT>"], 8]

    row = encode_synth_row(
        {
            "query": "What?",
            "query_seed_text": "Context.",
            "synthetic_reasoning": "Because.",
            "synthetic_answer": "Answer.",
            "query_seed_url": "https://example.test/source",
            "exercise": "retrieval",
        },
        FakeTokenizer(),
    )
    assert row is not None
    assert row.target_ids[-1] == 1
    assert any(row.reasoning_mask)
    assert any(row.answer_mask)
