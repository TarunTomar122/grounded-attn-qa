from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from grounded_qa.needle_qa_data import PreparedNeedleQA, prepare_squad2_item, prepare_squad2_unanswerable
from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.synthetic import entity_binding_row
from scripts.prepare_needle_n2 import sha256, tensorize
from scripts.prepare_needle_n3 import cross_pair_negatives
from scripts.train_needle_n1 import load_split


def prepare_entity_binding_pairs(count: int, tokenizer, seed: int = 41) -> tuple[list[PreparedNeedleQA], list[PreparedNeedleQA]]:
    """Create matched access-code contexts with either the present or an absent queried entity."""
    positive, negative = [], []
    candidates = (1, 2, 4, 6)
    for index in range(count):
        row = entity_binding_row(
            seed,
            index,
            split="n3_entity",
            distractor_mode="same_relation",
            candidate_count=candidates[index % len(candidates)],
            prefix_mode="unique",
        )
        context, answer = row["context"], row["answer"]
        present, reason = prepare_squad2_item(
            {"question": row["question"], "context": context, "answers": {"text": [answer], "answer_start": [context.index(answer)]}},
            tokenizer,
            example_index=index,
        )
        if present is None:
            raise ValueError(f"entity-positive row {index} could not be prepared: {reason}")
        missing_subject = f"Zorvax{index}"
        while missing_subject in context:
            missing_subject += "x"
        absent, reason = prepare_squad2_unanswerable(
            {
                "question": f"What is {missing_subject}'s access code?",
                "context": context,
                "answers": {"text": [], "answer_start": []},
            },
            tokenizer,
            example_index=index,
        )
        if absent is None:
            raise ValueError(f"entity-negative row {index} could not be prepared: {reason}")
        positive.append(present)
        negative.append(absent)
    return positive, negative


def prepare_relation_binding_pairs(count: int, tokenizer, seed: int = 53) -> tuple[list[PreparedNeedleQA], list[PreparedNeedleQA]]:
    """Create matched records where the queried entity remains but its access-code fact is absent."""
    positive, negative = [], []
    candidates = (1, 2, 4, 6)
    for index in range(count):
        row = entity_binding_row(
            seed,
            index,
            split="n3_relation",
            distractor_mode="both",
            candidate_count=candidates[index % len(candidates)],
            prefix_mode="unique",
        )
        context, answer, evidence = row["context"], row["answer"], row["evidence"]
        present, reason = prepare_squad2_item(
            {"question": row["question"], "context": context, "answers": {"text": [answer], "answer_start": [context.index(answer)]}},
            tokenizer,
            example_index=index,
        )
        if present is None:
            raise ValueError(f"relation-positive row {index} could not be prepared: {reason}")
        absent_context = context.replace(evidence, "", 1)
        absent, reason = prepare_squad2_unanswerable(
            {"question": row["question"], "context": absent_context, "answers": {"text": [], "answer_start": []}},
            tokenizer,
            example_index=index,
        )
        if absent is None:
            raise ValueError(f"relation-negative row {index} could not be prepared: {reason}")
        positive.append(present)
        negative.append(absent)
    return positive, negative


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the N3 entity-binding bridge curriculum.")
    parser.add_argument("--n2-data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-relation", action="store_true")
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "N3 entity/relation-binding bridge before official negatives.",
        "composition": (
            {"n2_answerable": 0.50, "entity_answerable": 0.10, "relation_answerable": 0.10, "cross_pair": 0.10, "entity_missing": 0.10, "relation_missing": 0.10}
            if args.include_relation
            else {"n2_answerable": 0.60, "entity_answerable": 0.10, "cross_pair": 0.15, "entity_missing": 0.15}
        ),
        "n2_manifest_sha256": sha256(args.n2_data_dir / "n2-manifest.json"),
        "splits": {},
    }
    for split, seed in (("train", 41), ("validation", 97)):
        n2 = load_split(args.n2_data_dir, split)
        n2["answerable"] = torch.ones(len(n2["source_ids"]), dtype=torch.bool)
        entity_positive_rows = round(len(n2["source_ids"]) / (5 if args.include_relation else 6))
        negative_rows = round(len(n2["source_ids"]) / (5 if args.include_relation else 4))
        entity_positive, entity_negative = prepare_entity_binding_pairs(negative_rows, tokenizer, seed=seed)
        entity_positive_tensors = tensorize(entity_positive[:entity_positive_rows])
        entity_negative_tensors = tensorize(entity_negative)
        entity_positive_tensors["answerable"] = torch.ones(entity_positive_rows, dtype=torch.bool)
        entity_negative_tensors["answerable"] = torch.zeros(negative_rows, dtype=torch.bool)
        cross = cross_pair_negatives(n2, count=negative_rows, seed=seed + 1)
        relation_positive_tensors = relation_negative_tensors = None
        if args.include_relation:
            relation_positive, relation_negative = prepare_relation_binding_pairs(negative_rows, tokenizer, seed=seed + 2)
            relation_positive_tensors = tensorize(relation_positive)
            relation_negative_tensors = tensorize(relation_negative)
            relation_positive_tensors["answerable"] = torch.ones(negative_rows, dtype=torch.bool)
            relation_negative_tensors["answerable"] = torch.zeros(negative_rows, dtype=torch.bool)
        combined = {
            key: torch.cat((
                n2[key],
                entity_positive_tensors[key],
                *(() if relation_positive_tensors is None else (relation_positive_tensors[key],)),
                cross[key],
                entity_negative_tensors[key],
                *(() if relation_negative_tensors is None else (relation_negative_tensors[key],)),
            ))
            for key in n2
        }
        path = args.output_dir / f"n3-{split}.pt"
        torch.save(combined, path)
        manifest["splits"][split] = {
            "n2_answerable_rows": len(n2["source_ids"]),
            "entity_answerable_rows": entity_positive_rows,
            "cross_pair_rows": negative_rows,
            "entity_missing_rows": negative_rows,
            "relation_answerable_rows": 0 if relation_positive_tensors is None else negative_rows,
            "relation_missing_rows": 0 if relation_negative_tensors is None else negative_rows,
            "total_rows": len(combined["source_ids"]),
            "file": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)},
        }
        print(json.dumps({"split": split, **manifest["splits"][split]}), flush=True)
    path = args.output_dir / "n3-entity-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
