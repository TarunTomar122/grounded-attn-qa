from __future__ import annotations

import torch
import pytest

from grounded_qa.needle_pointer import NeedleAnswerablePointerModel, NeedlePointerModel, NeedlePointerOutput, evidence_start_loss, evidence_start_targets, pointer_loss
from grounded_qa.needle_tokenizer import SPECIAL_TOKENS
from grounded_qa.needleish import GroupedQueryAttention, NeedleConfig, NeedleishModel
from grounded_qa.synth_rag import appears_unsupported, cited_source_ids, clean_answer, evidence_context, parse_sources
from grounded_qa.synth_data import encode_synth_row, source_bucket, split_for_source
from scripts.evaluate_public_needle import apply_refusal, generate_batch, summarize
from scripts.train_needle_n2_pointer import calibrate_answerability, swap_contexts


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
    assert cfg.dropout == 0.1
    assert not cfg.cross_attention_rope
    assert model.n_params() == 26_233_372


def test_needle_pointer_masks_query_and_normalizes_final_distribution() -> None:
    cfg = tiny_config()
    model = NeedlePointerModel(cfg)
    source = torch.tensor([[7, 8, 9, 10]])
    source_valid = torch.ones_like(source, dtype=torch.bool)
    context_mask = torch.tensor([[False, False, True, True]])
    decoder = torch.tensor([[1, 9]])
    output = model(source, source_valid, context_mask, decoder, torch.ones_like(decoder, dtype=torch.bool))

    assert torch.equal(output.copy_position_probs[..., :2], torch.zeros_like(output.copy_position_probs[..., :2]))
    torch.testing.assert_close(output.final_distribution(source).sum(dim=-1), torch.ones(1, 2))


def test_needle_pointer_position_loss_selects_the_annotated_duplicate() -> None:
    source = torch.tensor([[7, 8, 9, 9]])
    output = NeedlePointerOutput(
        vocab_logits=torch.zeros(1, 1, tiny_config().vocab_size),
        copy_position_probs=torch.tensor([[[0.0, 0.0, 0.1, 0.9]]]),
        p_gen=torch.full((1, 1), 0.5),
    )
    annotated = pointer_loss(
        output,
        source,
        torch.tensor([[9]]),
        torch.ones((1, 1), dtype=torch.bool),
        torch.tensor([[3]]),
    )
    duplicate = pointer_loss(
        output,
        source,
        torch.tensor([[9]]),
        torch.ones((1, 1), dtype=torch.bool),
        torch.tensor([[2]]),
    )

    assert annotated.pointer_position < duplicate.pointer_position
    assert annotated.mean_gold_pointer_probability > duplicate.mean_gold_pointer_probability


def test_synth_rag_evidence_context_keeps_cited_source() -> None:
    class Tokenizer:
        @staticmethod
        def encode(text: str) -> list[str]:
            return text.split()

    constraints = "<source_1>wrong fact</source_1> <source_2>gold fact</source_2>"
    answer = 'Answer.<ref name="source_2">gold fact</ref>'
    assert parse_sources(constraints) == [("1", "wrong fact"), ("2", "gold fact")]
    assert cited_source_ids(answer) == ["2"]
    assert clean_answer(answer) == "Answer."
    assert appears_unsupported("Sources do not contain enough information.")
    selected = evidence_context(
        row_id="row",
        query="question",
        constraints=constraints,
        answer=answer,
        tokenizer=Tokenizer(),
        source_length=6,
    )
    assert selected is not None and "source_2" in selected and "source_1" not in selected


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


def test_pointer_evaluator_decodes() -> None:
    model = NeedlePointerModel(tiny_config())
    model.eval()
    model.pointer.gate.weight.data.zero_()
    model.pointer.gate.bias.data.fill_(-100)
    source = torch.tensor([[4, 5, 6]])
    valid = torch.ones_like(source, dtype=torch.bool)
    context = torch.tensor([[False, True, True]])
    generated = generate_batch(model, source, valid, 2, context)
    assert generated.shape == (1, 2)
    assert set(generated[0].tolist()) <= {5, 6}


def test_evaluator_refuses_below_answerability_threshold() -> None:
    assert apply_refusal("copied answer", 0.8, 0.5) == "copied answer"
    assert apply_refusal("unsupported guess", 0.2, 0.5) == "I don't know this."


def test_plain_evaluator_ignores_answerability_labels_without_gate_decisions() -> None:
    row = {
        "condition": "correct",
        "pair_id": "probe",
        "prediction": "47",
        "em": 1.0,
        "token_f1": 1.0,
        "eos": True,
        "generated_tokens": 1,
        "unsupported_number_rate": 0.0,
        "unsupported_entity_rate": 0.0,
        "answerable": True,
    }
    assert "answerability" not in summarize([row])


def test_pointer_empty_context_falls_back_to_vocabulary() -> None:
    model = NeedlePointerModel(tiny_config())
    source = torch.tensor([[4, 5]])
    valid = torch.ones_like(source, dtype=torch.bool)
    context = torch.zeros_like(source, dtype=torch.bool)
    decoder = torch.tensor([[1]])
    output = model(source, valid, context, decoder, torch.ones_like(decoder, dtype=torch.bool))
    assert torch.equal(output.copy_position_probs, torch.zeros_like(output.copy_position_probs))
    assert torch.equal(output.p_gen, torch.ones_like(output.p_gen))
    torch.testing.assert_close(output.final_distribution(source).sum(dim=-1), torch.ones(1, 1))


def test_answerability_head_receives_gradient_without_copy_targets() -> None:
    model = NeedleAnswerablePointerModel(tiny_config())
    source = torch.tensor([[4, 5, 6]])
    valid = torch.ones_like(source, dtype=torch.bool)
    context = torch.tensor([[False, True, True]])
    decoder = torch.tensor([[1]])
    output = model(source, valid, context, decoder, torch.ones_like(decoder, dtype=torch.bool))
    assert output.answerability_logits is not None
    torch.nn.functional.binary_cross_entropy_with_logits(output.answerability_logits, torch.zeros(1)).backward()
    assert model.evidence.weight.grad is not None
    assert model.pointer.gate.weight.grad is None


def test_answerability_head_scores_context_positions_against_no_answer() -> None:
    model = NeedleAnswerablePointerModel(tiny_config())
    model.evidence.weight.data.zero_()
    model.evidence.bias.data.zero_()
    model.evidence.weight.data[0, 0] = 1
    model.no_answer_logit.data.zero_()
    memory = torch.zeros((2, 3, tiny_config().d_model))
    memory[0, 1, 0] = 1
    memory[1, 1, 0] = 3
    valid = torch.ones((2, 3), dtype=torch.bool)
    context = torch.tensor([[False, True, True], [False, True, True]])
    logits = model.classify_answerability(memory, valid, context)
    assert logits[1] > logits[0]
    positions = model.evidence_position_logits(memory, context)
    assert positions.shape == (2, 4)
    assert torch.isneginf(positions[:, 1]).all()


def test_evidence_start_loss_uses_null_for_negative_rows() -> None:
    output = NeedlePointerOutput(
        torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0),
        torch.tensor([[2.0, 1.0, 0.0], [2.0, 0.0, 1.0]]),
    )
    loss = evidence_start_loss(output, torch.tensor([[0], [-1]]), torch.tensor([True, False]))
    assert evidence_start_targets(torch.tensor([[0], [-1]]), torch.tensor([True, False])).tolist() == [1, 0]
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(output.evidence_position_logits, torch.tensor([1, 0])))


def test_backbone_loader_ignores_legacy_answerability_head() -> None:
    model = NeedleAnswerablePointerModel(tiny_config())
    state = model.state_dict()
    state["token_embedding.weight"] = torch.ones_like(state["token_embedding.weight"])
    state["answerability.weight"] = torch.zeros((1, tiny_config().d_model))
    model.load_backbone_state_dict(state)
    assert torch.equal(model.token_embedding.weight, torch.ones_like(model.token_embedding.weight))


def test_wrong_context_swap_preserves_questions() -> None:
    source = torch.tensor([[10, 5, 20, 21, 0], [11, 12, 5, 30, 31]])
    valid = source.ne(0)
    context = torch.tensor([[False, False, True, True, False], [False, False, False, True, True]])
    wrong, wrong_valid, wrong_context = swap_contexts(source, valid, context)
    assert wrong[0, :2].tolist() == [10, 5]
    assert wrong[1, :3].tolist() == [11, 12, 5]
    assert wrong[0][wrong_context[0]].tolist() == [30, 31]
    assert wrong[1][wrong_context[1]].tolist() == [20, 21]
    assert torch.equal(wrong.ne(0), wrong_valid)
    with pytest.raises(ValueError, match="at least two"):
        swap_contexts(source[:1], valid[:1], context[:1])


def test_answerability_calibration_separates_classes() -> None:
    metrics = calibrate_answerability(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([True, True, False, False]),
    )
    assert metrics["answerability_f1"] == 1
    assert metrics["refusal_f1"] == 1
    assert 0.2 < metrics["threshold"] <= 0.8
    assert metrics["grounded_false_answer_rate"] <= 0.02
    assert metrics["grounded_answer_coverage"] == 1
