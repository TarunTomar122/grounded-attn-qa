from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from grounded_qa.needle_qa_data import SQUAD2_DATASET, SQUAD2_REVISION, _evidence_window
from grounded_qa.needle_tokenizer import NeedleTokenizer
from scripts.prepare_needle_n2 import SOURCE_LENGTH, sha256
from scripts.prepare_needle_n3_matched import split_for_context


def verifier_query(question: str, candidate: str) -> str:
    return f"Question: {question}\nCandidate answer: {candidate}\nIs this candidate supported by the context?"


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def distractor(context: str, answer: str, rng: random.Random) -> tuple[str, int] | None:
    words = list(re.finditer(r"\S+", context))
    answer_words = max(1, min(8, len(answer.split())))
    if len(words) < answer_words:
        return None
    answer_key = normalized(answer)
    for _ in range(32):
        start = rng.randrange(len(words) - answer_words + 1)
        candidate = context[words[start].start() : words[start + answer_words - 1].end()]
        candidate_key = normalized(candidate)
        if candidate_key and candidate_key not in answer_key and answer_key not in candidate_key:
            return candidate, words[start].start()
    return None


def tensorize(rows: list[tuple[list[int], int, bool]]) -> dict[str, torch.Tensor]:
    width = max(len(source) for source, _, _ in rows)
    source = torch.zeros((len(rows), width), dtype=torch.int32)
    lengths = torch.zeros(len(rows), dtype=torch.int32)
    starts = torch.zeros(len(rows), dtype=torch.int32)
    labels = torch.zeros(len(rows), dtype=torch.bool)
    for index, (ids, context_start, answerable) in enumerate(rows):
        source[index, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        lengths[index] = len(ids)
        starts[index] = context_start
        labels[index] = answerable
    return {"source_ids": source, "source_lengths": lengths, "context_start": starts, "answerable": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare same-paragraph supported-candidate verifier data from SQuAD2 train.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in load_dataset(SQUAD2_DATASET, split="train", revision=SQUAD2_REVISION):
        grouped[item["context"]].append(item)
    rows = {"train": [], "validation": []}
    stats = {"paragraphs": len(grouped), "positive": 0, "wrong_candidate": 0, "unanswerable_question": 0, "skipped": 0}
    for index, (context, items) in enumerate(grouped.items()):
        rng = random.Random(args.seed + index)
        positives = [item for item in items if item["answers"]["text"]]
        negatives = [item for item in items if not item["answers"]["text"]]
        if not positives or not negatives:
            continue
        positive = positives[0]
        answer = positive["answers"]["text"][0]
        wrong_candidate = distractor(context, answer, rng)
        unsupported_candidate = distractor(context, "", rng)
        if wrong_candidate is None or unsupported_candidate is None:
            stats["skipped"] += 1
            continue
        split = split_for_context(context)
        examples = (
            (positive["question"], answer, positive["answers"]["answer_start"][0], True, "positive"),
            (positive["question"], *wrong_candidate, False, "wrong_candidate"),
            (negatives[0]["question"], *unsupported_candidate, False, "unanswerable_question"),
        )
        for question, candidate, candidate_start, label, kind in examples:
            query = verifier_query(question, candidate)
            window = _evidence_window(
                tokenizer, query, context, candidate_start, candidate_start + len(candidate), max_source_length=SOURCE_LENGTH
            )
            if window is None:
                stats["skipped"] += 1
                continue
            ids, _, _, context_start, _ = window
            rows[split].append((ids, context_start, label))
            stats[kind] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Candidate verifier: every candidate appears in its context; label asks whether it answers the question.",
        "dataset": {"name": SQUAD2_DATASET, "revision": SQUAD2_REVISION, "split": "train"},
        "split": "sha256(context) modulo 20; source validation is untouched",
        "stats": stats,
        "splits": {},
    }
    for split, examples in rows.items():
        tensors = tensorize(examples)
        path = args.output_dir / f"n3-verifier-{split}.pt"
        torch.save(tensors, path)
        manifest["splits"][split] = {
            "rows": len(examples),
            "supported": int(tensors["answerable"].sum()),
            "unsupported": int((~tensors["answerable"]).sum()),
            "file": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)},
        }
    (args.output_dir / "n3-verifier-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
