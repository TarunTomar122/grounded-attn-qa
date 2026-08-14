from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grounded_qa.needleish import NeedleConfig, NeedleishModel, load_public_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure public Needle training throughput on CUDA.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--source-length", type=int, default=1024)
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    device = torch.device("cuda")
    results = []
    for batch_size in map(int, args.batch_sizes.split(",")):
        torch.cuda.empty_cache()
        model = NeedleishModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
        load_public_checkpoint(model, args.checkpoint)
        model.train()
        if args.compile:
            model = torch.compile(model)
        source = torch.randint(1, 8192, (batch_size, args.source_length), device=device)
        decoder = torch.randint(1, 8192, (batch_size, args.target_length), device=device)
        targets = torch.randint(1, 8192, (batch_size, args.target_length), device=device)
        source_valid = torch.ones_like(source, dtype=torch.bool)
        target_valid = torch.ones_like(decoder, dtype=torch.bool)

        try:
            for _ in range(args.warmup):
                logits = model(source, source_valid, decoder, target_valid)
                F.cross_entropy(logits.float().flatten(0, 1), targets.flatten()).backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(args.steps):
                logits = model(source, source_valid, decoder, target_valid)
                loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens = args.steps * batch_size * (args.source_length + args.target_length)
            results.append({
                "batch_size": batch_size,
                "status": "ok",
                "steps": args.steps,
                "seconds": elapsed,
                "step_seconds": elapsed / args.steps,
                "tokens_per_second": tokens / elapsed,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                "loss": float(loss.detach()),
            })
        except torch.OutOfMemoryError:
            results.append({"batch_size": batch_size, "status": "oom"})
        del model, source, decoder, targets, source_valid, target_valid

    report = {
        "gpu": torch.cuda.get_device_name(),
        "pytorch": torch.__version__,
        "source_length": args.source_length,
        "target_length": args.target_length,
        "precision": "bfloat16",
        "compile": args.compile,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
