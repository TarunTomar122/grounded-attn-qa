#!/usr/bin/env python3
"""Serve the current full Needle span/NULL checkpoint for the browser demo."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from grounded_qa.needle_full_span_qa import NeedleFullSpanNullModel, load_compatible_state_dict
from grounded_qa.needle_span_qa import best_spans
from grounded_qa.needle_tokenizer import TOOLS_ID, NeedleTokenizer
from grounded_qa.needleish import NeedleConfig


MAX_REQUEST_BYTES = 1_000_000


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NeedleDemo:
    def __init__(self, checkpoint: Path, tokenizer_path: Path, threshold: float | None = None) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = NeedleFullSpanNullModel(NeedleConfig.public_checkpoint())
        load_compatible_state_dict(self.model, state["model"])
        self.device = pick_device()
        self.model.to(self.device).eval()
        self.tokenizer = NeedleTokenizer(tokenizer_path, append_markers=False)
        self.checkpoint = checkpoint
        saved_threshold = state.get("metrics", {}).get("threshold")
        self.threshold = float(threshold if threshold is not None else saved_threshold or 0.0)
        self.step = state.get("step")

    @torch.inference_mode()
    def predict(self, context: str, question: str) -> dict[str, object]:
        context = context.strip()
        question = question.strip()
        if not context or not question:
            raise ValueError("context and question are required")

        query_ids = self.tokenizer.encode(question)
        context_ids = self.tokenizer.encode(context)
        source_ids = [*query_ids, TOOLS_ID, *context_ids]
        if len(source_ids) > self.model.cfg.source_length:
            raise ValueError(
                f"context plus question is {len(source_ids)} tokens; keep it under {self.model.cfg.source_length}"
            )

        source = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(source_ids), device=self.device)[None]
        valid = torch.ones_like(source, dtype=torch.bool)
        context_mask = positions >= len(query_ids) + 1
        output = self.model(source, valid, context_mask)
        start, end, _, margin = best_spans(output)
        raw_span = self.tokenizer.decode(source_ids[start.item() - 1 : end.item()]).strip()
        refuse = bool(margin.item() >= self.threshold)
        return {
            "answer": "" if refuse else raw_span,
            "raw_span": raw_span,
            "refuse": refuse,
            "null_margin": float(margin.item()),
            "threshold": self.threshold,
            "evidence": raw_span,
            "input_tokens": len(source_ids),
            "device": str(self.device),
            "checkpoint": self.checkpoint.name,
            "step": self.step,
        }


class Handler(BaseHTTPRequestHandler):
    demo: NeedleDemo

    def send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/status":
            self.send_json(
                200,
                {
                    "ready": True,
                    "device": str(self.demo.device),
                    "checkpoint": self.demo.checkpoint.name,
                    "step": self.demo.step,
                    "parameters": sum(parameter.numel() for parameter in self.demo.model.parameters()),
                    "threshold": self.demo.threshold,
                },
            )
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_json(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("request body is missing or too large")
            payload = json.loads(self.rfile.read(size))
            self.send_json(200, self.demo.predict(payload.get("context", ""), payload.get("question", "")))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    demo = NeedleDemo(args.checkpoint, args.tokenizer, args.threshold)
    Handler.demo = demo
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Loaded {demo.checkpoint.name} step {demo.step} on {demo.device}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
