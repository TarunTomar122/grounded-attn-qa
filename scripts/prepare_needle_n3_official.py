from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

from grounded_qa.needle_qa_data import SQUAD2_DATASET, SQUAD2_REVISION, prepare_squad2_unanswerable
from grounded_qa.needle_tokenizer import NeedleTokenizer
from scripts.prepare_needle_n2 import SOURCE_LENGTH, sha256, tensorize
from scripts.prepare_needle_n3 import cross_pair_negatives
from scripts.prepare_needle_n3_entity import prepare_entity_binding_pairs, prepare_relation_binding_pairs
from scripts.train_needle_n1 import load_split


def select_tensor_rows(rows: dict[str, torch.Tensor], count: int, seed: int) -> dict[str, torch.Tensor]:
    if not 0 < count <= len(rows["source_ids"]):
        raise ValueError("row count must be between one and the available rows")
    indices = torch.randperm(len(rows["source_ids"]), generator=torch.Generator().manual_seed(seed))[:count]
    return {key: value.index_select(0, indices) for key, value in rows.items()}


def official_negatives(split: str, tokenizer) -> list:
    prepared = []
    for index, row in enumerate(load_dataset(SQUAD2_DATASET, split=split, revision=SQUAD2_REVISION)):
        example, reason = prepare_squad2_unanswerable(row, tokenizer, example_index=index, max_source_length=SOURCE_LENGTH)
        if example is not None:
            prepared.append(example)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the N3 natural-negative bridge with replay.")
    parser.add_argument("--n2-data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "N3 natural-negative bridge: introduce official SQuAD2 negatives after cross, entity, and relation binding.",
        "composition": {"n2_answerable": 0.50, "entity_answerable": 0.10, "relation_answerable": 0.10, "cross_pair": 0.05, "entity_missing": 0.05, "relation_missing": 0.05, "official_squad2_missing": 0.15},
        "official_dataset": {"name": SQUAD2_DATASET, "revision": SQUAD2_REVISION},
        "n2_manifest_sha256": sha256(args.n2_data_dir / "n2-manifest.json"),
        "splits": {},
    }
    for split, seed in (("train", 131), ("validation", 191)):
        n2_all = load_split(args.n2_data_dir, split)
        official_all = tensorize(official_negatives(split, tokenizer))
        n2_rows = min(len(n2_all["source_ids"]), len(official_all["source_ids"]) * 10 // 3)
        official_rows = round(n2_rows * 3 / 10)
        n2 = select_tensor_rows(n2_all, n2_rows, seed)
        official = select_tensor_rows(official_all, official_rows, seed + 1)
        n2["answerable"] = torch.ones(n2_rows, dtype=torch.bool)
        official["answerable"] = torch.zeros(official_rows, dtype=torch.bool)
        binding_positive_rows = round(n2_rows / 5)
        bridge_negative_rows = round(n2_rows / 10)
        entity_positive, entity_negative = prepare_entity_binding_pairs(binding_positive_rows, tokenizer, seed=seed + 2)
        relation_positive, relation_negative = prepare_relation_binding_pairs(binding_positive_rows, tokenizer, seed=seed + 3)
        entity_positive_tensors, relation_positive_tensors = tensorize(entity_positive), tensorize(relation_positive)
        entity_negative_tensors = tensorize(entity_negative[:bridge_negative_rows])
        relation_negative_tensors = tensorize(relation_negative[:bridge_negative_rows])
        for tensor in (entity_positive_tensors, relation_positive_tensors):
            tensor["answerable"] = torch.ones(binding_positive_rows, dtype=torch.bool)
        for tensor in (entity_negative_tensors, relation_negative_tensors):
            tensor["answerable"] = torch.zeros(bridge_negative_rows, dtype=torch.bool)
        cross = cross_pair_negatives(n2, count=bridge_negative_rows, seed=seed + 4)
        combined = {
            key: torch.cat((
                n2[key], entity_positive_tensors[key], relation_positive_tensors[key], cross[key],
                entity_negative_tensors[key], relation_negative_tensors[key], official[key],
            ))
            for key in n2
        }
        path = args.output_dir / f"n3-{split}.pt"
        torch.save(combined, path)
        manifest["splits"][split] = {
            "n2_answerable_rows": n2_rows,
            "entity_answerable_rows": binding_positive_rows,
            "relation_answerable_rows": binding_positive_rows,
            "cross_pair_rows": bridge_negative_rows,
            "entity_missing_rows": bridge_negative_rows,
            "relation_missing_rows": bridge_negative_rows,
            "official_squad2_missing_rows": official_rows,
            "total_rows": len(combined["source_ids"]),
            "file": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)},
        }
        print(json.dumps({"split": split, **manifest["splits"][split]}), flush=True)
    path = args.output_dir / "n3-official-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
