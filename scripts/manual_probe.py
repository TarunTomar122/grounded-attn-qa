from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.needleish import NeedleConfig, NeedleishModel
from grounded_qa.negatives import REFUSAL
from scripts.train_foundation import generate_probe, probe_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer)
    examples = [
        ("vault-marker", "What vault marker is assigned to the Vesper Gate?", "The Vesper Gate vault marker is VG-864219-H. The Warden Gate marker is WG-207533-C, and the Vesper Gate spare is VG-000814-Q.", "The Vesper Gate vault marker is VG-864219-H.", "VG-864219-H"),
        ("date", "On what date did the Lattice Ferry open?", "The Lattice Ferry opened to public service on 3 March 2042 after its safety review. Its first prototype sailed in 2040.", "The Lattice Ferry opened on 3 March 2042.", "3 March 2042"),
        ("entity-binding", "What is Rhea's station code?", "Rhea's station code is 631. Rilo's station code is 284. Rhea monitors archives, while Rilo monitors dispatch.", "The context states Rhea's station code is 631.", "631"),
        ("quantity", "How many sealed reels were logged in crate M-7?", "Crate M-7 contained 241 sealed reels during the evening inventory. Crate M-6 contained 119 reels.", "Crate M-7 contained 241 sealed reels.", "241"),
        ("unanswerable", "Which official authorized the sealed transfer?", "The transfer record lists reference ST-92 and a destination in East Annex. It records no authorizing official.", "The context does not name an authorizing official.", REFUSAL),
    ]
    model = NeedleishModel(NeedleConfig()).cuda()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"])
    rows = []
    for name, query, context, reasoning, answer in examples:
        target, _ = tokenizer.encode_target(reasoning, answer)
        rows.append({
            "exercise": f"manual/{name}",
            "query": query,
            "context": context,
            "source_ids": tokenizer.encode_source(query, context),
            "target_ids": target,
        })
    outputs = generate_probe(model, tokenizer, rows, torch.device("cuda"), max_new_tokens=48)
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_tokens": state["tokens_seen"],
        "summary": probe_summary(outputs),
        "examples": outputs,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
