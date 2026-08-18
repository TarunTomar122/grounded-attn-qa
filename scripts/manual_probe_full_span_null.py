from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from grounded_qa.needle_full_span_qa import NeedleFullSpanNullModel, load_compatible_state_dict
from grounded_qa.needle_span_qa import best_spans
from grounded_qa.needle_tokenizer import TOOLS_ID, NeedleTokenizer
from grounded_qa.needleish import NeedleConfig


CASES = [
    {
        "name": "access_code",
        "question": "What is Nuvora's access code?",
        "context": "Kifa's access code is AB-91827-X. Nuvora's access code is QF-4302358-Y. Torven's access code is LM-77301-P.",
        "expected": "QF-4302358-Y",
    },
    {
        "name": "opening_date",
        "question": "When did the Lattice Ferry open to public service?",
        "context": "The Lattice Ferry opened to public service on 3 March 2042 after its safety review. Its first prototype sailed in 2040.",
        "expected": "3 March 2042",
    },
    {
        "name": "unsupported",
        "question": "Which official authorized the sealed transfer?",
        "context": "The transfer record lists reference ST-92 and a destination in East Annex. It records no authorizing official.",
        "expected": "",
    },
]


def answer_text(tokenizer: NeedleTokenizer, source: list[int], start: int, end: int) -> str:
    if start == 0 or end == 0 or end < start:
        return ""
    return tokenizer.decode(source[start - 1 : end]).strip()


@torch.inference_mode()
def probe(model, tokenizer: NeedleTokenizer, threshold: float, device: torch.device) -> list[dict[str, object]]:
    results = []
    for case in CASES:
        query_ids = tokenizer.encode(case["question"])
        context_ids = tokenizer.encode(case["context"])
        source = [*query_ids, TOOLS_ID, *context_ids]
        context_start = len(query_ids) + 1
        source_tensor = torch.tensor([source], dtype=torch.long, device=device)
        positions = torch.arange(len(source), device=device)[None]
        source_valid = torch.ones_like(source_tensor, dtype=torch.bool)
        context_mask = positions >= context_start
        output = model(source_tensor, source_valid, context_mask)
        start, end, _, margin = best_spans(output)
        raw = answer_text(tokenizer, source, int(start[0]), int(end[0]))
        margin_value = float(margin[0])
        results.append(
            {
                **case,
                "raw_span": raw,
                "prediction": "" if margin_value >= threshold else raw,
                "null_margin": margin_value,
                "threshold": threshold,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe full Needle span+NULL on hand-written contexts.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeedleFullSpanNullModel(NeedleConfig.public_checkpoint()).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_compatible_state_dict(model, state["model"])
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    result = {
        "checkpoint": str(args.checkpoint),
        "step": state.get("step"),
        "threshold": args.threshold,
        "examples": probe(model, tokenizer, args.threshold, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
