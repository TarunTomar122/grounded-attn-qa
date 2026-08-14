from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grounded_qa.needleish import NeedleConfig, NeedleishModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/foundation_benchmark.json")
    parser.add_argument("--source", type=int, default=512)
    parser.add_argument("--target", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    cfg = NeedleConfig(source_length=args.source, target_length=args.target)
    model = NeedleishModel(cfg).to(device)
    results = []
    for precision, dtype in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        for batch_size in (1, 2, 4, 8):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            source = torch.randint(0, cfg.vocab_size, (batch_size, args.source), device=device)
            decoder = torch.randint(0, cfg.vocab_size, (batch_size, args.target), device=device)
            source_valid = torch.ones_like(source, dtype=torch.bool)
            target_valid = torch.ones_like(decoder, dtype=torch.bool)
            try:
                for _ in range(args.warmup):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                        logits = model(source, source_valid, decoder, target_valid)
                        loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), decoder.reshape(-1))
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                        logits = model(source, source_valid, decoder, target_valid)
                        loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), decoder.reshape(-1))
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                total_tokens = args.steps * batch_size * (args.source + args.target)
                results.append({
                    "precision": precision,
                    "batch_size": batch_size,
                    "step_seconds": elapsed / args.steps,
                    "examples_per_sec": args.steps * batch_size / elapsed,
                    "tokens_per_sec": total_tokens / elapsed,
                    "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
                    "loss": float(loss.detach()),
                })
            except torch.cuda.OutOfMemoryError:
                results.append({"precision": precision, "batch_size": batch_size, "oom": True})
                torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"architecture": cfg.to_dict(), "parameters": model.n_params(), "results": results}
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
