from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from grounded_qa.needle_qa_data import SQUAD2_DATASET, SQUAD2_REVISION, _evidence_window
from grounded_qa.needle_tokenizer import NeedleTokenizer
from scripts.prepare_needle_n2 import SOURCE_LENGTH, sha256
from scripts.prepare_needle_n3_matched import split_for_context
from scripts.prepare_needle_n3_verifier import normalized, tensorize, verifier_query


def is_supported_candidate(row: dict) -> bool:
    candidate = normalized(str(row.get("raw_prediction", row.get("prediction", ""))))
    return bool(row.get("answerable")) and bool(candidate) and any(candidate == normalized(answer) for answer in row["answers"])


def nli_label(row: dict) -> int:
    if is_supported_candidate(row):
        return 2  # support
    return 1 if row["answerable"] else 0  # refute, neutral


def reader_input_rows(items, tokenizer: NeedleTokenizer) -> tuple[dict[str, list[dict]], dict[str, int]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[item["context"]].append(item)
    rows = {"train": [], "validation": []}
    stats = {"paragraphs": len(grouped), "matched_paragraphs": 0, "too_long": 0}
    for context, group in grouped.items():
        positive = next((item for item in group if item["answers"]["text"]), None)
        negative = next((item for item in group if not item["answers"]["text"]), None)
        if positive is None or negative is None:
            continue
        split = split_for_context(context)
        pair_id = hashlib.sha256(context.encode()).hexdigest()
        pair = (
            (positive, True, "answerable"),
            (negative, False, "unanswerable"),
        )
        prepared = []
        for item, answerable, condition in pair:
            if len(tokenizer.encode_source(item["question"], context)) > SOURCE_LENGTH:
                prepared = []
                stats["too_long"] += 1
                break
            prepared.append({
                "pair_id": pair_id,
                "split": split,
                "condition": condition,
                "question": item["question"],
                "context": context,
                "answers": item["answers"]["text"] or [""],
                "answerable": answerable,
            })
        if prepared:
            rows[split].extend(prepared)
            stats["matched_paragraphs"] += 1
    return rows, stats


def write_reader_inputs(tokenizer_path: Path, output_dir: Path) -> None:
    tokenizer = NeedleTokenizer(tokenizer_path, append_markers=False)
    rows, stats = reader_input_rows(load_dataset(SQUAD2_DATASET, split="train", revision=SQUAD2_REVISION), tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Generate the deployed reader's own candidates on paragraph-matched SQuAD2 questions.",
        "dataset": {"name": SQUAD2_DATASET, "revision": SQUAD2_REVISION, "split": "train"},
        "split": "sha256(context) modulo 20; source validation is untouched",
        "stats": stats,
        "splits": {},
    }
    for split, examples in rows.items():
        path = output_dir / f"n3-reader-candidates-{split}.jsonl"
        content = "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)
        path.write_text(content)
        manifest["splits"][split] = {"rows": len(examples), "file": {"path": path.name, "sha256": hashlib.sha256(content.encode()).hexdigest()}}
    (output_dir / "n3-reader-candidates-input-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def materialize_reader_candidates(tokenizer_path: Path, report_paths: list[Path], output_dir: Path) -> None:
    tokenizer = NeedleTokenizer(tokenizer_path, append_markers=False)
    rows: dict[str, list[tuple[list[int], int, bool, int, int, int]]] = {"train": [], "validation": []}
    stats = defaultdict(int)
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        for row in report["examples"]:
            candidate = str(row.get("raw_prediction", row.get("prediction", ""))).strip()
            if not candidate:
                stats["empty_candidate"] += 1
                continue
            start = row["context"].lower().find(candidate.lower())
            if start < 0:
                stats["nonliteral_candidate"] += 1
                continue
            window = _evidence_window(
                tokenizer,
                verifier_query(row["question"], candidate),
                row["context"],
                start,
                start + len(candidate),
                max_source_length=SOURCE_LENGTH,
            )
            if window is None:
                stats["window_too_long"] += 1
                continue
            ids, _, candidate_positions, context_start, _ = window
            supported = is_supported_candidate(row)
            label = nli_label(row)
            rows[row["split"]].append((
                ids,
                context_start,
                supported,
                context_start + candidate_positions[0],
                context_start + candidate_positions[-1] + 1,
                label,
            ))
            stats["supported" if supported else "unsupported"] += 1
            stats["answerable" if row["answerable"] else "unanswerable"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Verifier data from literal candidates produced by the deployed reader; nonliteral candidates are refused at inference.",
        "reader_reports": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in report_paths],
        "stats": dict(stats),
        "splits": {},
    }
    for split, examples in rows.items():
        tensors = tensorize([(ids, context_start, label) for ids, context_start, label, _, _, _ in examples])
        tensors["candidate_start"] = torch.tensor([start for _, _, _, start, _, _ in examples], dtype=torch.int32)
        tensors["candidate_end"] = torch.tensor([end for _, _, _, _, end, _ in examples], dtype=torch.int32)
        tensors["nli_label"] = torch.tensor([label for _, _, _, _, _, label in examples], dtype=torch.int64)
        path = output_dir / f"n3-reader-candidates-{split}.pt"
        torch.save(tensors, path)
        manifest["splits"][split] = {
            "rows": len(examples),
            "supported": int(tensors["answerable"].sum()),
            "unsupported": int((~tensors["answerable"]).sum()),
            "neutral": int(tensors["nli_label"].eq(0).sum()),
            "refute": int(tensors["nli_label"].eq(1).sum()),
            "file": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)},
        }
    (output_dir / "n3-reader-candidates-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reproducible reader-generated verifier candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("emit-inputs", "materialize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--tokenizer", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        if command == "materialize":
            subparser.add_argument("--reader-report", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.command == "emit-inputs":
        write_reader_inputs(args.tokenizer, args.output_dir)
    else:
        materialize_reader_candidates(args.tokenizer, args.reader_report, args.output_dir)


if __name__ == "__main__":
    main()
