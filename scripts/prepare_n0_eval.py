from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from datasets import load_dataset

from grounded_qa.needle_tokenizer import NeedleTokenizer


DATASET = "rajpurkar/squad_v2"
COUNTERFACTUALS = [
    ("What is the capital of France according to the context?", "The registry states that the capital of France is Lyon.", "Lyon"),
    ("Which planet is known as the Red Planet in this report?", "In this fictional report, Venus is known as the Red Planet.", "Venus"),
    ("Who wrote Hamlet according to the archive?", "The alternate archive attributes Hamlet to Christopher Marlowe.", "Christopher Marlowe"),
    ("What is the largest ocean according to the handbook?", "The handbook identifies the Atlantic Ocean as the largest ocean.", "Atlantic Ocean"),
    ("What gas do plants absorb according to the notes?", "These fictional notes say plants absorb oxygen during photosynthesis.", "oxygen"),
    ("How many days are in a leap year according to this calendar?", "This calendar defines a leap year as having 367 days.", "367 days"),
    ("What is the chemical symbol for gold in this table?", "The table assigns the symbol Gd to gold.", "Gd"),
    ("Who was the first person on the Moon according to the log?", "The altered mission log names Buzz Aldrin as the first person on the Moon.", "Buzz Aldrin"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the initial Needle QA evaluation set.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=256)
    args = parser.parse_args()
    if args.examples < 2:
        parser.error("--examples must be at least 2 to construct wrong-context pairs")

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}") as response:
        revision = json.load(response)["sha"]
    dataset = load_dataset(DATASET, split="validation", revision=revision)
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)

    candidates = []
    for item in dataset:
        answers = [answer for answer in item["answers"]["text"] if answer]
        if not answers:
            continue
        source_ids = tokenizer.encode_source(item["question"], item["context"])
        if len(source_ids) > 1024:
            continue
        candidates.append((hashlib.sha256(f"42:{item['id']}".encode()).hexdigest(), item, answers))
    selected = sorted(candidates, key=lambda row: row[0])[: args.examples]
    if len(selected) != args.examples:
        raise RuntimeError(f"Requested {args.examples} examples, found {len(selected)}")

    rows = []
    for index, (_, item, answers) in enumerate(selected):
        pair_id = f"squad2-{item['id']}"
        common = {"pair_id": pair_id, "question": item["question"], "answers": answers, "source": "squad2"}
        rows.append({"id": f"{pair_id}-correct", "condition": "correct", "context": item["context"], **common})
        answer_lower = answers[0].lower()
        for offset in range(1, len(selected)):
            wrong = selected[(index + offset) % len(selected)][1]["context"]
            if answer_lower not in wrong.lower():
                break
        rows.append({"id": f"{pair_id}-wrong", "condition": "wrong", "context": wrong, **common})
        rows.append({"id": f"{pair_id}-empty", "condition": "empty", "context": "", **common})

    for index, (question, context, answer) in enumerate(COUNTERFACTUALS):
        rows.append({
            "id": f"counterfactual-{index}",
            "pair_id": f"counterfactual-{index}",
            "condition": "counterfactual",
            "question": question,
            "context": context,
            "answers": [answer],
            "source": "handwritten_counterfactual",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": DATASET,
        "revision": revision,
        "split": "validation",
        "selection": "answerable, <=1024 source tokens, SHA256(seed=42,id) order",
        "base_examples": len(selected),
        "rows": len(rows),
        "jsonl_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "conditions": {
            condition: sum(row["condition"] == condition for row in rows)
            for condition in ("correct", "wrong", "empty", "counterfactual")
        },
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
