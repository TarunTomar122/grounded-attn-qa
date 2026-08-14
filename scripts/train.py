#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded_qa.checkpoint import save_checkpoint
from grounded_qa.config import ModelConfig
from grounded_qa.data import GroundedDataset, collate_examples, encode_row
from grounded_qa.generation import generate
from grounded_qa.losses import grounded_loss
from grounded_qa.metrics import exact_match, token_f1
from grounded_qa.model import GroundedPointerGenerator
from grounded_qa.synthetic import (
    a2a_training_row,
    a2b_training_row,
    a1c_training_row,
    a1c_validation_splits,
    a1d_training_row,
    entity_binding_training_row,
    entity_binding_validation_splits,
    phase_a_validation_splits,
    procedural_copy_row,
)
from grounded_qa.real_data import squad2_rows_with_stats
from grounded_qa.tokenizer import load_tokenizer
from grounded_qa.wandb_logging import WandbLogger


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def online_batch(
    *,
    seed: int,
    first_index: int,
    batch_size: int,
    tokenizer,
    tokenizer_info,
    cfg: ModelConfig,
    curriculum: str = "standard",
    curriculum_step: int = 1,
    squad_rows: list[dict] | None = None,
) -> dict:
    """Build one deterministic, never-replayed procedural training batch."""
    examples = []
    for index in range(first_index, first_index + batch_size):
        if curriculum == "entity_binding":
            row = entity_binding_training_row(seed, index, curriculum_step)
        elif curriculum == "a1c":
            row = a1c_training_row(seed, index, curriculum_step)
        elif curriculum == "a1d":
            row = a1d_training_row(seed, index, curriculum_step)
        elif curriculum == "a2a":
            if not squad_rows:
                raise ValueError("A2a training needs prepared SQuAD rows")
            row = a2a_training_row(seed, index, curriculum_step, squad_rows)
        elif curriculum == "a2b":
            if not squad_rows:
                raise ValueError("A2b training needs prepared SQuAD rows")
            row = a2b_training_row(seed, index, curriculum_step, squad_rows)
        else:
            row = procedural_copy_row(seed, index, split="train", entity_set="train")
        encoded = encode_row(
            row,
            tokenizer,
            tokenizer_info,
            max_source_length=cfg.source_length,
            max_target_length=cfg.target_length,
        )
        if encoded is None:
            raise RuntimeError(f"procedural row {index} exceeded the configured sequence lengths")
        examples.append(encoded)
    return collate_examples(examples, cfg.pad_id)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device.type in {"cuda", "mps"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


def mps_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "mps":
        return {}
    recommended = torch.mps.recommended_max_memory()
    return {
        "system/mps_tensor_gb": torch.mps.current_allocated_memory() / 1024**3,
        "system/mps_driver_gb": torch.mps.driver_allocated_memory() / 1024**3,
        "system/mps_recommended_gb": recommended / 1024**3,
        "system/memory_fraction": torch.mps.driver_allocated_memory() / max(recommended, 1),
    }


def optimizer_for(model: GroundedPointerGenerator, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "norm" not in name.lower():
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )


def cosine_scheduler(optimizer: torch.optim.Optimizer, warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate(
    model,
    loader,
    device: torch.device,
    lambda_answerability: float,
    *,
    copy_only: bool,
    eos_id: int,
    lambda_pointer_position: float = 0.0,
    lambda_start: float = 0.0,
    first_pointer_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss_total": 0.0,
        "loss_seq": 0.0,
        "loss_answerability": 0.0,
        "loss_pointer_position": 0.0,
        "loss_pointer_first": 0.0,
        "loss_pointer_continuation": 0.0,
        "loss_start_head": 0.0,
        "token_accuracy": 0.0,
        "pointer_accuracy": 0.0,
        "mean_copy_probability": 0.0,
        "pointer_entropy": 0.0,
        "eos_correct": 0.0,
        "eos_tokens": 0,
        "target_tokens": 0,
        "rows": 0,
        "start_head_correct": 0.0,
        "start_head_rows": 0,
    }
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(
                batch["source_ids"],
                batch["token_type_ids"],
                batch["source_valid"],
                batch["context_mask"],
                batch["decoder_input_ids"],
                batch["target_valid"],
            )
            losses = grounded_loss(
                output,
                batch["source_ids"],
                batch["target_ids"],
                batch["target_valid"],
                batch["answerable"],
                lambda_answerability,
                copy_only=copy_only,
                eos_id=eos_id,
                gold_copy_positions=batch["gold_copy_positions"],
                lambda_pointer_position=lambda_pointer_position,
                context_mask=batch["context_mask"] & batch["source_valid"],
                lambda_start=lambda_start,
                first_pointer_weight=first_pointer_weight,
            )
            size = len(batch["rows"])
            for key, value in (
                ("loss_total", losses.total),
                ("loss_seq", losses.sequence),
                ("loss_answerability", losses.answerability),
                ("loss_pointer_position", losses.pointer_position),
                ("loss_pointer_first", losses.pointer_first),
                ("loss_pointer_continuation", losses.pointer_continuation),
                ("loss_start_head", losses.start_head),
            ):
                totals[key] += value.item() * size
            start_valid = batch["answerable"] & batch["gold_copy_positions"][:, 0].ge(0)
            if start_valid.any():
                start_logits = output.answer_start_logits.masked_fill(
                    ~(batch["context_mask"] & batch["source_valid"]),
                    torch.finfo(output.answer_start_logits.dtype).min,
                )
                totals["start_head_correct"] += float(
                    start_logits[start_valid].argmax(dim=-1).eq(batch["gold_copy_positions"][start_valid, 0]).sum()
                )
                totals["start_head_rows"] += int(start_valid.sum())
            prediction = output.final_distribution(
                batch["source_ids"],
                copy_only=copy_only,
                eos_id=eos_id,
            ).argmax(dim=-1)
            valid = batch["target_valid"]
            token_count = int(valid.sum())
            if token_count:
                totals["token_accuracy"] += float(((prediction == batch["target_ids"]) & valid).sum())
                pointer_prediction = batch["source_ids"].gather(1, output.copy_position_probs.argmax(dim=-1))
                pointer_valid = valid & batch["target_ids"].ne(eos_id)
                totals["pointer_accuracy"] += float(
                    ((pointer_prediction == batch["target_ids"]) & pointer_valid).sum()
                )
                # P_final is P_copy in Phase A; p_gen is deliberately unused.
                totals["mean_copy_probability"] += float(valid.sum()) if copy_only else float(((1 - output.p_gen) * valid).sum())
                entropy = -(output.copy_position_probs.clamp_min(1.0e-8).log() * output.copy_position_probs).sum(-1)
                totals["pointer_entropy"] += float((entropy * valid).sum())
                eos_valid = valid & batch["target_ids"].eq(eos_id)
                totals["eos_correct"] += float(((prediction == eos_id) & eos_valid).sum())
                totals["eos_tokens"] += int(eos_valid.sum())
                totals["target_tokens"] += token_count
            totals["rows"] += size
    model.train()
    rows = max(totals["rows"], 1)
    target_tokens = max(totals["target_tokens"], 1)
    pointer_tokens = max(totals["target_tokens"] - totals["rows"], 1)
    return {
        "loss_total": totals["loss_total"] / rows,
        "loss_seq": totals["loss_seq"] / rows,
        "loss_answerability": totals["loss_answerability"] / rows,
        "loss_pointer_position": totals["loss_pointer_position"] / rows,
        "loss_pointer_first": totals["loss_pointer_first"] / rows,
        "loss_pointer_continuation": totals["loss_pointer_continuation"] / rows,
        "loss_start_head": totals["loss_start_head"] / rows,
        "start_head/gold_position_accuracy": totals["start_head_correct"] / max(totals["start_head_rows"], 1),
        "token_accuracy": totals["token_accuracy"] / target_tokens,
        "pointer_accuracy": totals["pointer_accuracy"] / pointer_tokens,
        "pointer_teacher_forced_accuracy": totals["pointer_accuracy"] / pointer_tokens,
        "eos_accuracy": totals["eos_correct"] / max(totals["eos_tokens"], 1),
        "pointer/mean_copy_probability": totals["mean_copy_probability"] / target_tokens,
        "pointer/mean_generate_probability": 0.0 if copy_only else 1 - totals["mean_copy_probability"] / target_tokens,
        "pointer/entropy": totals["pointer_entropy"] / target_tokens,
        "answer_length": target_tokens / rows,
    }


def _oracle_copy(model, batch: dict, cfg: ModelConfig, tokenizer_info) -> list[list[int]]:
    """Autoregressively copy for each gold answer length, deliberately excluding EOS."""
    memory = model.encode(batch["source_ids"], batch["token_type_ids"], batch["source_valid"])
    decoder_ids = torch.full((len(batch["rows"]), 1), tokenizer_info.bos_id, dtype=torch.long, device=batch["source_ids"].device)
    answer_mask = batch["target_valid"] & batch["target_ids"].ne(tokenizer_info.eos_id)
    lengths = answer_mask.sum(dim=1)
    outputs: list[list[int]] = [[] for _ in batch["rows"]]
    for position in range(int(lengths.max().item())):
        output = model.decode(
            decoder_ids,
            memory,
            batch["source_ids"],
            batch["context_mask"],
            batch["source_valid"],
            torch.ones_like(decoder_ids, dtype=torch.bool),
        )
        distribution = output.final_distribution(
            batch["source_ids"],
            copy_only=True,
            eos_id=tokenizer_info.eos_id,
        )
        # Oracle length removes only EOS uncertainty; preserve token-level
        # aggregation for repeated source tokens exactly as greedy decoding does.
        distribution[..., tokenizer_info.eos_id] = 0
        next_ids = distribution[:, -1].argmax(dim=-1)
        for index, value in enumerate(next_ids.tolist()):
            if position < int(lengths[index]):
                outputs[index].append(value)
        decoder_ids = torch.cat((decoder_ids, next_ids[:, None]), dim=1)
    return outputs


def _gold_answer_tokens(batch: dict, index: int, eos_id: int) -> list[int]:
    mask = batch["target_valid"][index] & batch["target_ids"][index].ne(eos_id)
    return batch["target_ids"][index][mask].tolist()


def _length_bucket(length: int) -> str:
    if length <= 2:
        return "1-2"
    if length <= 4:
        return "3-4"
    if length <= 6:
        return "5-6"
    if length <= 10:
        return "7-10"
    return "11+"


def _question_type(question: str) -> str:
    first = question.lower().lstrip().split(maxsplit=1)[0] if question.strip() else ""
    if first in {"who", "what", "when", "where", "which", "why"}:
        return first
    if first == "how":
        return "how_many" if any(word in question.lower().split()[:4] for word in ("many", "much")) else "how"
    return "other"


def _context_length_bucket(length: int) -> str:
    if length < 128:
        return "<128"
    if length < 256:
        return "128-256"
    if length < 384:
        return "256-384"
    return "384-512"


def _common_prefix_length(first: list[int], second: list[int]) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def _prefix_overlap_bucket(row: dict, tokenizer) -> str:
    candidates = row.get("metadata", {}).get("candidate_values", "")
    if not candidates:
        return "unknown"
    target_ids = tokenizer.encode(" " + row["answer"], add_special_tokens=False).ids
    candidate_ids = [
        tokenizer.encode(" " + value, add_special_tokens=False).ids
        for value in candidates.split("|")[1:]
    ]
    if not candidate_ids:
        return "0"
    return str(min(_common_prefix_length(target_ids, value) for value in candidate_ids))


def _gold_answer_span(batch: dict, index: int) -> list[int] | None:
    positions = batch["gold_copy_positions"][index]
    positions = positions[positions.ge(0)].tolist()
    return positions or None


def _top_positions(probabilities: torch.Tensor, limit: int = 5) -> str:
    values, positions = torch.topk(probabilities, min(limit, probabilities.numel()))
    return ";".join(f"{int(position)}:{float(value):.3f}" for value, position in zip(values, positions))


def evaluate_generated(
    model,
    loader,
    tokenizer,
    tokenizer_info,
    device: torch.device,
    max_rows: int,
    cfg: ModelConfig,
    qualitative_limit: int = 0,
) -> tuple[dict[str, float], list[list[str]]]:
    model.eval()
    exact = f1 = oracle = rows = 0
    start_head_correct = 0.0
    start_head_rows = 0
    first_correct = continuation_correct = continuation_tokens = 0
    continuation_after_first_correct = continuation_after_first_tokens = 0
    continuation_position_correct = continuation_position_tokens = 0
    continuation_position_after_first_correct = continuation_position_after_first_tokens = 0
    gold_start_correct = gold_span_mass = top1_inside_gold_span = gold_span_rows = 0.0
    source_position_switches = source_position_switch_opportunities = 0
    by_relation: dict[str, list[float]] = defaultdict(list)
    by_overlap: dict[str, list[float]] = defaultdict(list)
    by_question_type: dict[str, dict[str, float]] = defaultdict(lambda: {"rows": 0, "f1": 0, "pointer": 0, "start": 0, "start_head": 0})
    by_context_length: dict[str, dict[str, float]] = defaultdict(lambda: {"rows": 0, "f1": 0, "start": 0, "span_mass": 0})
    by_lexical_overlap: dict[str, dict[str, float]] = defaultdict(lambda: {"rows": 0, "f1": 0, "start": 0, "start_head": 0})
    by_candidate: dict[str, dict[str, float]] = defaultdict(lambda: {
        "rows": 0,
        "start_correct": 0,
        "span_mass": 0,
        "inside_span": 0,
        "greedy_correct": 0,
        "continuation_correct": 0,
        "continuation_tokens": 0,
        "switches": 0,
        "switch_opportunities": 0,
    })
    by_prefix_overlap: dict[str, dict[str, float]] = defaultdict(lambda: {
        "rows": 0,
        "start_correct": 0,
        "span_mass": 0,
        "inside_span": 0,
        "greedy_correct": 0,
    })
    by_length: dict[str, dict[str, float]] = defaultdict(lambda: {
        "rows": 0,
        "pointer_correct": 0,
        "pointer_tokens": 0,
        "f1": 0,
        "greedy_correct": 0,
        "oracle_correct": 0,
    })
    qualitative: list[list[str]] = []
    with torch.inference_mode():
        for batch in loader:
            if rows >= max_rows:
                break
            batch = move_batch(batch, device)
            predicted, _, pointer_steps = generate(
                model,
                batch["source_ids"],
                batch["token_type_ids"],
                batch["source_valid"],
                batch["context_mask"],
                bos_id=tokenizer_info.bos_id,
                eos_id=tokenizer_info.eos_id,
                max_new_tokens=cfg.target_length,
                copy_only=True,
            )
            oracle_ids = _oracle_copy(model, batch, cfg, tokenizer_info)
            teacher = model(
                batch["source_ids"], batch["token_type_ids"], batch["source_valid"], batch["context_mask"], batch["decoder_input_ids"], batch["target_valid"]
            )
            teacher_pointer = batch["source_ids"].gather(1, teacher.copy_position_probs.argmax(dim=-1))
            teacher_final = teacher.final_distribution(batch["source_ids"], copy_only=True, eos_id=tokenizer_info.eos_id).argmax(dim=-1)
            for index, (predicted_ids, row) in enumerate(zip(predicted, batch["rows"])):
                if rows >= max_rows:
                    break
                text = tokenizer.decode(predicted_ids[1:].tolist(), skip_special_tokens=True).strip()
                greedy_correct = exact_match(text, row["answer"])
                answer_tokens = _gold_answer_tokens(batch, index, tokenizer_info.eos_id)
                oracle_correct = float(oracle_ids[index] == answer_tokens)
                eos_index = len(answer_tokens)
                eos_correct = bool(teacher_final[index, eos_index].item() == tokenizer_info.eos_id)
                pointer_matches = teacher_pointer[index, :eos_index].eq(batch["target_ids"][index, :eos_index])
                pointer_correct = bool(pointer_matches.all())
                gold_span = _gold_answer_span(batch, index)
                first_pointer = teacher.copy_position_probs[index, 0]
                first_position = int(first_pointer.argmax().item())
                start_hit = bool(gold_span and first_position == gold_span[0])
                start_head = teacher.answer_start_logits[index].masked_fill(
                    ~(batch["context_mask"][index] & batch["source_valid"][index]),
                    torch.finfo(teacher.answer_start_logits.dtype).min,
                )
                start_head_position = int(start_head.argmax().item())
                start_head_hit = bool(gold_span and start_head_position == gold_span[0])
                span_mass = float(first_pointer[gold_span].sum().item()) if gold_span else 0.0
                inside_span = bool(gold_span and first_position in gold_span)
                if gold_span:
                    gold_start_correct += float(start_hit)
                    top1_inside_gold_span += float(inside_span)
                    gold_span_mass += span_mass
                    gold_span_rows += 1
                    start_head_correct += float(start_head_hit)
                    start_head_rows += 1
                    gold_positions = torch.tensor(gold_span, device=teacher.copy_position_probs.device)
                    position_predictions = teacher.copy_position_probs[index, :eos_index].argmax(dim=-1)
                    position_matches = position_predictions.eq(gold_positions)
                    continuation_position_correct += int(position_matches[1:].sum())
                    continuation_position_tokens += max(eos_index - 1, 0)
                    if start_hit:
                        continuation_position_after_first_correct += int(position_matches[1:].sum())
                        continuation_position_after_first_tokens += max(eos_index - 1, 0)
                trajectory_positions = [
                    int(step[index].argmax().item())
                    for step in pointer_steps[:eos_index]
                ]
                switches = opportunities = 0
                if gold_span and len(trajectory_positions) > 1:
                    inside = [position in gold_span for position in trajectory_positions]
                    switches = sum(
                        int(inside[position] and not inside[position + 1])
                        for position in range(len(inside) - 1)
                    )
                    opportunities = sum(int(value) for value in inside[:-1])
                    source_position_switches += switches
                    source_position_switch_opportunities += opportunities
                if eos_index:
                    first_correct += int(pointer_matches[0])
                    continuation_correct += int(pointer_matches[1:].sum())
                    continuation_tokens += eos_index - 1
                    if pointer_matches[0]:
                        continuation_after_first_correct += int(pointer_matches[1:].sum())
                        continuation_after_first_tokens += eos_index - 1
                exact += greedy_correct
                row_f1 = token_f1(text, row["answer"])
                f1 += row_f1
                oracle += oracle_correct
                bucket = by_length[_length_bucket(eos_index)]
                bucket["rows"] += 1
                bucket["pointer_correct"] += int(pointer_matches.sum())
                bucket["pointer_tokens"] += eos_index
                bucket["greedy_correct"] += greedy_correct
                bucket["oracle_correct"] += oracle_correct
                bucket["f1"] += row_f1
                candidate_bucket = by_candidate[row["metadata"].get("candidate_count", "unknown")]
                candidate_bucket["rows"] += 1
                candidate_bucket["start_correct"] += float(start_hit)
                candidate_bucket["span_mass"] += span_mass
                candidate_bucket["inside_span"] += float(inside_span)
                candidate_bucket["greedy_correct"] += greedy_correct
                candidate_bucket["continuation_correct"] += int(pointer_matches[1:].sum())
                candidate_bucket["continuation_tokens"] += max(eos_index - 1, 0)
                candidate_bucket.setdefault("continuation_position_correct", 0)
                candidate_bucket.setdefault("continuation_position_tokens", 0)
                candidate_bucket["continuation_position_correct"] += int(position_matches[1:].sum()) if gold_span else 0
                candidate_bucket["continuation_position_tokens"] += max(eos_index - 1, 0) if gold_span else 0
                candidate_bucket["switches"] += switches
                candidate_bucket["switch_opportunities"] += opportunities
                prefix_bucket = by_prefix_overlap[_prefix_overlap_bucket(row, tokenizer)]
                prefix_bucket["rows"] += 1
                prefix_bucket["start_correct"] += float(start_hit)
                prefix_bucket["span_mass"] += span_mass
                prefix_bucket["inside_span"] += float(inside_span)
                prefix_bucket["greedy_correct"] += greedy_correct
                metadata = row["metadata"]
                by_relation[metadata.get("relation", row.get("source", "unknown"))].append(greedy_correct)
                by_overlap[metadata.get("lexical_overlap_bucket", "unknown")].append(greedy_correct)
                question_type = metadata.get("question_type", _question_type(row["question"]))
                question_bucket = by_question_type[question_type]
                question_bucket["rows"] += 1
                question_bucket["f1"] += row_f1
                question_bucket["pointer"] += float(pointer_matches.float().mean()) if eos_index else 0
                question_bucket["start"] += float(start_hit)
                question_bucket["start_head"] += float(start_head_hit)
                context_bucket = by_context_length[_context_length_bucket(len(batch["source_ids"][index][batch["context_mask"][index]].tolist()))]
                context_bucket["rows"] += 1
                context_bucket["f1"] += row_f1
                context_bucket["start"] += float(start_hit)
                context_bucket["span_mass"] += span_mass
                overlap_bucket = by_lexical_overlap[metadata.get("lexical_overlap_bucket", "unknown")]
                overlap_bucket["rows"] += 1
                overlap_bucket["f1"] += row_f1
                overlap_bucket["start"] += float(start_hit)
                overlap_bucket["start_head"] += float(start_head_hit)
                if len(qualitative) < qualitative_limit:
                    qualitative.append([
                        metadata["relation"], row["question"], row["context"], row["answer"], text,
                        metadata["template_family"], metadata["question_structure"], metadata["context_structure"],
                        metadata["lexical_overlap_bucket"], str(pointer_correct), str(oracle_correct), str(bool(greedy_correct)), str(eos_correct),
                        ";".join(f"{position}:{'G' if gold_span and position in gold_span else 'X'}" for position in trajectory_positions),
                        _top_positions(start_head),
                        _top_positions(first_pointer),
                        str(gold_span[0] if gold_span else ""),
                    ])
                rows += 1
    model.train()
    metrics = {
        "greedy_em": exact / max(rows, 1),
        "token_f1": f1 / max(rows, 1),
        "oracle_length_em": oracle / max(rows, 1),
        "first_token_pointer_accuracy": first_correct / max(rows, 1),
        "continuation_token_pointer_accuracy": continuation_correct / max(continuation_tokens, 1),
        "continuation_pointer_accuracy_given_correct_first": continuation_after_first_correct / max(continuation_after_first_tokens, 1),
        "continuation_position_accuracy": continuation_position_correct / max(continuation_position_tokens, 1),
        "continuation_position_accuracy_given_correct_start": continuation_position_after_first_correct / max(continuation_position_after_first_tokens, 1),
        "gold_start_position_accuracy": gold_start_correct / max(gold_span_rows, 1),
        "pointer/first_gold_position_accuracy": gold_start_correct / max(gold_span_rows, 1),
        "generated/start_head/gold_position_accuracy": start_head_correct / max(start_head_rows, 1),
        "pointer_mass_on_gold_answer_span": gold_span_mass / max(gold_span_rows, 1),
        "top1_pointer_position_inside_gold_span": top1_inside_gold_span / max(gold_span_rows, 1),
        "gold_span_rows": gold_span_rows,
        "source_position_switch_rate": source_position_switches / max(source_position_switch_opportunities, 1),
        "generated_rows": float(rows),
    }
    metrics.update({f"relation/{name}/greedy_em": sum(values) / len(values) for name, values in by_relation.items()})
    metrics.update({f"overlap/{name}/greedy_em": sum(values) / len(values) for name, values in by_overlap.items()})
    for name, bucket in by_question_type.items():
        metrics[f"question_type/{name}/rows"] = bucket["rows"]
        metrics[f"question_type/{name}/token_f1"] = bucket["f1"] / max(bucket["rows"], 1)
        metrics[f"question_type/{name}/pointer_accuracy"] = bucket["pointer"] / max(bucket["rows"], 1)
        metrics[f"question_type/{name}/gold_start_position_accuracy"] = bucket["start"] / max(bucket["rows"], 1)
        metrics[f"question_type/{name}/start_head_accuracy"] = bucket["start_head"] / max(bucket["rows"], 1)
    for name, bucket in by_context_length.items():
        metrics[f"context_length/{name}/rows"] = bucket["rows"]
        metrics[f"context_length/{name}/token_f1"] = bucket["f1"] / max(bucket["rows"], 1)
        metrics[f"context_length/{name}/gold_start_position_accuracy"] = bucket["start"] / max(bucket["rows"], 1)
        metrics[f"context_length/{name}/pointer_mass_on_gold_answer_span"] = bucket["span_mass"] / max(bucket["rows"], 1)
    for name, bucket in by_lexical_overlap.items():
        metrics[f"lexical_overlap/{name}/rows"] = bucket["rows"]
        metrics[f"lexical_overlap/{name}/token_f1"] = bucket["f1"] / max(bucket["rows"], 1)
        metrics[f"lexical_overlap/{name}/gold_start_position_accuracy"] = bucket["start"] / max(bucket["rows"], 1)
        metrics[f"lexical_overlap/{name}/start_head_accuracy"] = bucket["start_head"] / max(bucket["rows"], 1)
    for bucket_name, bucket in by_length.items():
        metrics[f"length/{bucket_name}/rows"] = bucket["rows"]
        metrics[f"length/{bucket_name}/pointer_teacher_forced_accuracy"] = bucket["pointer_correct"] / max(bucket["pointer_tokens"], 1)
        metrics[f"length/{bucket_name}/token_f1"] = bucket["f1"] / max(bucket["rows"], 1)
        metrics[f"length/{bucket_name}/greedy_em"] = bucket["greedy_correct"] / max(bucket["rows"], 1)
        metrics[f"length/{bucket_name}/oracle_length_em"] = bucket["oracle_correct"] / max(bucket["rows"], 1)
    for bucket_name, bucket in by_candidate.items():
        metrics[f"candidates/{bucket_name}/gold_start_position_accuracy"] = bucket["start_correct"] / max(bucket["rows"], 1)
        metrics[f"candidates/{bucket_name}/pointer_mass_on_gold_answer_span"] = bucket["span_mass"] / max(bucket["rows"], 1)
        metrics[f"candidates/{bucket_name}/top1_pointer_position_inside_gold_span"] = bucket["inside_span"] / max(bucket["rows"], 1)
        metrics[f"candidates/{bucket_name}/greedy_em"] = bucket["greedy_correct"] / max(bucket["rows"], 1)
        metrics[f"candidates/{bucket_name}/continuation_position_accuracy"] = bucket["continuation_correct"] / max(bucket["continuation_tokens"], 1)
        metrics[f"candidates/{bucket_name}/continuation_source_position_accuracy"] = bucket.get("continuation_position_correct", 0) / max(bucket.get("continuation_position_tokens", 0), 1)
        metrics[f"candidates/{bucket_name}/source_position_switch_rate"] = bucket["switches"] / max(bucket["switch_opportunities"], 1)
    for bucket_name, bucket in by_prefix_overlap.items():
        metrics[f"prefix_overlap/{bucket_name}/gold_start_position_accuracy"] = bucket["start_correct"] / max(bucket["rows"], 1)
        metrics[f"prefix_overlap/{bucket_name}/pointer_mass_on_gold_answer_span"] = bucket["span_mass"] / max(bucket["rows"], 1)
        metrics[f"prefix_overlap/{bucket_name}/top1_pointer_position_inside_gold_span"] = bucket["inside_span"] / max(bucket["rows"], 1)
        metrics[f"prefix_overlap/{bucket_name}/greedy_em"] = bucket["greedy_correct"] / max(bucket["rows"], 1)
    return metrics, qualitative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/attn_pg_23m.yaml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--out-dir", default="runs/attn23m")
    parser.add_argument("--phase", default="A")
    parser.add_argument("--curriculum", choices=["standard", "entity_binding", "a1c", "a1d", "a2a", "a2b"], default="standard")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--val-n", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--schedule-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--copy-only", action="store_true")
    parser.add_argument("--lambda-pointer-position", type=float, default=None)
    parser.add_argument("--lambda-start", type=float, default=None)
    parser.add_argument("--first-pointer-weight", type=float, default=None)
    parser.add_argument("--start-head-mode", choices=["context", "global"], default=None)
    parser.add_argument("--squad-train-max", type=int, default=None)
    parser.add_argument("--squad-val-max", type=int, default=None)
    parser.add_argument("--wandb-project", default="grounded-attn-qa")
    parser.add_argument("--wandb-group", default="attnonly-main")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--generated-eval-n", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()
    if args.phase != "A":
        raise SystemExit("Only Phase A synthetic training is implemented in this first foundation pass")
    copy_only = True
    pointer_lambda = 1.0 if args.lambda_pointer_position is None and args.curriculum in {"a2a", "a2b"} else args.lambda_pointer_position or 0.0
    start_lambda = 1.0 if args.lambda_start is None and args.curriculum == "a2b" else args.lambda_start or 0.0
    first_pointer_weight = 4.0 if args.first_pointer_weight is None and args.curriculum == "a2b" else args.first_pointer_weight or 1.0

    seed_everything(args.seed)
    device = choose_device(args.device)
    tokenizer, tokenizer_info = load_tokenizer(args.tokenizer)
    cfg = ModelConfig.from_yaml(args.config)
    cfg.vocab_size = tokenizer.get_vocab_size()
    cfg.pad_id = tokenizer_info.pad_id
    cfg.bos_id = tokenizer_info.bos_id
    cfg.eos_id = tokenizer_info.eos_id
    cfg.cls_id = tokenizer_info.cls_id
    cfg.sep_id = tokenizer_info.sep_id
    cfg.question_id = tokenizer_info.question_id
    cfg.context_id = tokenizer_info.context_id
    if args.start_head_mode is not None:
        cfg.answer_start_mode = args.start_head_mode

    model = GroundedPointerGenerator(cfg).to(device)
    if copy_only:
        model.answerability.requires_grad_(False)
    print(f"device: {device}")
    print(f"parameters: {model.n_params():,}")
    print(f"breakdown: {model.parameter_breakdown()}")
    print(f"config: {cfg.to_dict()}")

    squad_train_rows = None
    dataset_stats = {}
    if args.curriculum in {"a2a", "a2b"}:
        squad_train_rows, train_stats = squad2_rows_with_stats(
            split="train",
            tokenizer=tokenizer,
            source_length=cfg.source_length,
            target_length=cfg.target_length,
            max_n=args.squad_train_max,
            seed=args.seed,
            validation=False,
        )
        squad_val_rows, val_stats = squad2_rows_with_stats(
            split="validation",
            tokenizer=tokenizer,
            source_length=cfg.source_length,
            target_length=cfg.target_length,
            max_n=args.squad_val_max,
            seed=args.seed,
            validation=True,
        )
        val_rows_by_split = {"squad2_answerable": squad_val_rows}
        if args.curriculum == "a2b":
            phase_rows = phase_a_validation_splits(args.val_n, args.seed)
            binding_rows = entity_binding_validation_splits(args.val_n, args.seed)
            prefix_rows = a1c_validation_splits(args.val_n, args.seed)
            val_rows_by_split.update({
                "A1_familiar": phase_rows["familiar_unseen_values"],
                "A1_novel": phase_rows["novel_combinations"],
                "A1_hard": phase_rows["hard_distractors"],
                "A1b_entity": binding_rows["binding_both_unique_6"],
                "A1d_shared": prefix_rows["a1c_shared_hard"],
            })
        dataset_stats = {"train": train_stats, "validation": val_stats}
        print(f"squad2 train stats: {train_stats}")
        print(f"squad2 validation stats: {val_stats}")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / "squad2_stats.json").write_text(json.dumps(dataset_stats, indent=2) + "\n", encoding="utf-8")
    elif args.curriculum == "entity_binding":
        val_rows_by_split = entity_binding_validation_splits(args.val_n, args.seed)
    elif args.curriculum in {"a1c", "a1d"}:
        val_rows_by_split = a1c_validation_splits(args.val_n, args.seed)
    else:
        val_rows_by_split = phase_a_validation_splits(args.val_n, args.seed)
    val_datasets = {
        name: GroundedDataset(rows, tokenizer, tokenizer_info, source_length=cfg.source_length, target_length=cfg.target_length)
        for name, rows in val_rows_by_split.items()
    }
    val_loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_examples(x, cfg.pad_id))
        for name, dataset in val_datasets.items()
    }
    print(f"training: {args.curriculum} online procedural rows")
    for name, dataset in val_datasets.items():
        print(f"rows: {name}={len(dataset)}")

    optimizer = optimizer_for(model, args.lr, 0.1)
    scheduler = cosine_scheduler(optimizer, args.warmup, args.schedule_steps or args.steps)
    start_step = 0
    if args.checkpoint:
        from grounded_qa.checkpoint import load_checkpoint

        if args.curriculum in {"entity_binding", "a1c", "a1d", "a2a", "a2b"} and not args.resume:
            load_checkpoint(args.checkpoint, model, device=device, strict=args.curriculum != "a2b")
            print("loaded model weights; reset optimizer and local curriculum step")
        else:
            state = load_checkpoint(args.checkpoint, model, optimizer, scheduler, device)
            start_step = state["step"]
            print(f"resumed step {start_step}")

    logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or f"attn23m-s{args.seed}-A-copyonly-{args.curriculum}",
        config={
            **cfg.to_dict(),
            "phase": args.phase,
            "curriculum": args.curriculum,
            "steps": args.steps,
            "schedule_steps": args.schedule_steps or args.steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
            "copy_only": copy_only,
            "answerability_frozen": copy_only,
            "dataset_version": "squad2_answerable_plus_procedural_replay" if args.curriculum in {"a2a", "a2b"} else "synthetic_procedural_copy_v6",
            "training_data": "65% SQuAD2 answerable, 25% A1 replay, 10% A1d replay" if args.curriculum == "a2b" else "70% SQuAD2 answerable, 20% A1 replay, 10% A1d replay" if args.curriculum == "a2a" else "online deterministic procedural rows; no replayed corpus",
            "model_version": "eos_pointer_context_fix",
            "binding_mixture": "50% shared-prefix, 30% A1 replay, 20% A1b replay" if args.curriculum == "a1d" else "60% shared same-relation, 20% shared both, 10% unique, 10% easy" if args.curriculum == "a1c" else "60% same-relation, 20% both, 10% easy, 10% existing" if args.curriculum == "entity_binding" else None,
            "a2a_mixture": "70% SQuAD2 answerable, 20% A1 replay, 10% A1d shared-prefix replay" if args.curriculum == "a2a" else None,
            "a2b_mixture": "65% SQuAD2 answerable, 25% A1 replay, 10% A1d shared-prefix replay" if args.curriculum == "a2b" else None,
            "squad2_stats": dataset_stats or None,
            "lambda_pointer_position": pointer_lambda,
            "lambda_start": start_lambda,
            "first_pointer_weight": first_pointer_weight,
        },
    )
    tokens_seen = 0
    best_val = float("inf")
    model.train()
    started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_rows = 0
        step_tokens = 0
        last_losses = None
        for micro_step in range(args.grad_accum):
            first_index = ((step - 1) * args.grad_accum + micro_step) * args.batch_size
            batch = online_batch(
                seed=args.seed,
                first_index=first_index,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                tokenizer_info=tokenizer_info,
                cfg=cfg,
                curriculum=args.curriculum,
                curriculum_step=step if args.curriculum in {"a1c", "a1d", "a2a", "a2b"} else step - start_step,
                squad_rows=squad_train_rows,
            )
            batch = move_batch(batch, device)
            with autocast_context(device, args.precision):
                output = model(
                    batch["source_ids"], batch["token_type_ids"], batch["source_valid"], batch["context_mask"], batch["decoder_input_ids"], batch["target_valid"]
                )
                last_losses = grounded_loss(
                    output,
                    batch["source_ids"],
                    batch["target_ids"],
                    batch["target_valid"],
                    batch["answerable"],
                    lambda_answerability=0.0,
                    copy_only=copy_only,
                    eos_id=cfg.eos_id,
                    gold_copy_positions=batch["gold_copy_positions"],
                    lambda_pointer_position=pointer_lambda,
                    context_mask=batch["context_mask"] & batch["source_valid"],
                    lambda_start=start_lambda,
                    first_pointer_weight=first_pointer_weight,
                )
                (last_losses.total / args.grad_accum).backward()
            step_rows += len(batch["rows"])
            step_tokens += int(batch["source_valid"].sum() + batch["target_valid"].sum())
        grad_norm = float(clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        scheduler.step()
        tokens_seen += step_tokens
        if step == 1 or step % 25 == 0 or step == args.steps:
            elapsed = time.perf_counter() - started
            values = {
                "train/loss_total": last_losses.total.item(),
                "train/loss_seq": last_losses.sequence.item(),
                "train/answerability_loss_diagnostic": last_losses.answerability.item(),
                "train/pointer_position_loss": last_losses.pointer_position.item(),
                "train/loss_pointer_first": last_losses.pointer_first.item(),
                "train/loss_pointer_continuation": last_losses.pointer_continuation.item(),
                "train/loss_start_head": last_losses.start_head.item(),
                "train/pointer/mean_copy_probability": 1.0,
                "train/pointer/mean_generate_probability": 0.0,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
                "system/tokens_per_sec": tokens_seen / max(elapsed, 1.0),
                "system/examples_per_sec": step_rows * step / max(elapsed, 1.0),
                "system/step_time_ms": elapsed * 1000 / step,
                **mps_metrics(device),
            }
            print(f"step {step}/{args.steps} loss={values['train/loss_total']:.4f} tok/s={values['system/tokens_per_sec']:.1f}", flush=True)
            logger.log(values, step=step)
        if step % args.eval_every == 0 or step == args.steps:
            all_metrics = {}
            qualitative_rows: list[list[str]] = []
            for name, loader in val_loaders.items():
                metrics = evaluate(
                    model,
                    loader,
                    device,
                    0.0,
                    copy_only=copy_only,
                    eos_id=cfg.eos_id,
                    lambda_pointer_position=pointer_lambda,
                    lambda_start=start_lambda,
                    first_pointer_weight=first_pointer_weight,
                )
                generated, examples = evaluate_generated(
                    model,
                    loader,
                    tokenizer,
                    tokenizer_info,
                    device,
                    args.generated_eval_n,
                    cfg,
                    qualitative_limit=30 if args.curriculum == "a2a" else 10 if name != "access_code" else 0,
                )
                metrics.update(generated)
                all_metrics[name] = metrics
                logger.log({f"val/{name}/{key}": value for key, value in metrics.items()}, step=step)
                logger.log({
                    f"val/{name}_loss": metrics["loss_total"],
                    f"val/{name}_em": metrics["greedy_em"],
                    f"val/{name}_pointer_accuracy": metrics["pointer_teacher_forced_accuracy"],
                    f"val/{name}_oracle_length_em": metrics["oracle_length_em"],
                    f"val/{name}_eos_accuracy": metrics["eos_accuracy"],
                }, step=step)
                qualitative_rows.extend(examples)
                print(f"val/{name}@{step}: {metrics}", flush=True)
            logger.table(
                "Phase A Generalization / qualitative_examples",
                [
                    "relation", "question", "context", "gold", "prediction", "template_family", "question_structure",
                    "context_structure", "lexical_overlap_bucket", "teacher_forced_pointer_correct", "oracle_length_correct",
                    "greedy_correct", "eos_correct", "pointer_trajectory", "start_head_top5", "decoder_first_pointer_top5", "gold_start_position",
                ],
                qualitative_rows,
                step=step,
            )
            mean_val_loss = sum(metrics["loss_total"] for metrics in all_metrics.values()) / len(all_metrics)
            path = Path(args.out_dir) / "latest.pt"
            save_checkpoint(path, model, optimizer, scheduler, step=step, tokens_seen=tokens_seen, phase=args.phase, config=cfg, tokenizer_path=args.tokenizer, seed=args.seed, wandb_run_id=logger.id)
            if mean_val_loss < best_val:
                best_val = mean_val_loss
                save_checkpoint(Path(args.out_dir) / "best_positive_f1.pt", model, optimizer, scheduler, step=step, tokens_seen=tokens_seen, phase=args.phase, config=cfg, tokenizer_path=args.tokenizer, seed=args.seed, wandb_run_id=logger.id)
    logger.finish()


if __name__ == "__main__":
    main()
