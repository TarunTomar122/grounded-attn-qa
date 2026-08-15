"""Materialize QA-to-declarative claims with a small, training-only converter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


MODEL = "domenicrosati/QA2D-t5-small"


def qa2d_prompt(question: str, answer: str) -> str:
    return f"{question.lower().rstrip('?.')}. {answer.lower().rstrip('.')}"


def claim_key(question: str, candidate: str) -> str:
    return f"{question}\0{candidate}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert frozen reader candidates into declarative QA2D claims.")
    parser.add_argument("--reader-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows: list[tuple[str, str]] = []
    seen = set()
    for path in args.reader_report:
        for row in json.loads(path.read_text())["examples"]:
            candidate = str(row.get("raw_prediction", row.get("prediction", "")) or "").strip()
            key = claim_key(row["question"], candidate)
            if candidate and key not in seen:
                rows.append((row["question"], candidate))
                seen.add(key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer([qa2d_prompt(*row) for row in batch], return_tensors="pt", padding=True, truncation=True).to(device)
            claims = tokenizer.batch_decode(model.generate(**encoded, max_new_tokens=64), skip_special_tokens=True)
            for (question, candidate), claim in zip(batch, claims):
                handle.write(json.dumps({"question": question, "candidate": candidate, "claim": claim}, ensure_ascii=False) + "\n")
    manifest = {
        "purpose": "Training-only QA-to-declarative conversion for NLI verifier inputs.",
        "model": args.model,
        "reader_reports": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in args.reader_report],
        "rows": len(rows),
        "file": {"path": args.output.name, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()},
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
