#!/usr/bin/env python3
"""Small local web runner for the released grounded attention-only checkpoint."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from transformers import GPT2TokenizerFast

from train import AttnOnlyGPT, GPTConfig, decode_after_mark, pack

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "grounded_attn_copy.pt"
MAX_REQUEST_BYTES = 1_000_000


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model() -> tuple[AttnOnlyGPT, GPT2TokenizerFast, torch.device, dict]:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = AttnOnlyGPT(config)
    model.load_state_dict(checkpoint["model"])
    device = pick_device()
    model.to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return model, tokenizer, device, checkpoint


MODEL, TOKENIZER, DEVICE, CHECKPOINT_DATA = load_model()


def predict(context: str, question: str) -> dict[str, object]:
    if not isinstance(context, str) or not isinstance(question, str):
        raise ValueError("context and question must be strings")
    context = context.strip()
    question = question.strip()
    if not context or not question:
        raise ValueError("context and question are required")

    prompt = pack(context, question, "")
    prompt_ids = TOKENIZER.encode(prompt)
    if len(prompt_ids) >= MODEL.cfg.block_size:
        raise ValueError(f"Prompt is too long; keep it under {MODEL.cfg.block_size} tokens")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        output = MODEL.generate(idx, max_new=32, eos_id=TOKENIZER.eos_token_id)
    decoded = TOKENIZER.decode(output[0].tolist())
    answer = decode_after_mark(TOKENIZER, decoded)
    return {
        "answer": answer,
        "input_tokens": len(prompt_ids),
        "device": str(DEVICE),
        "refuse": answer.lower().startswith("i don't know"),
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(
                200,
                {
                    "ready": True,
                    "device": str(DEVICE),
                    "checkpoint": CHECKPOINT.name,
                    "parameters": MODEL.n_params(),
                    "refuse": CHECKPOINT_DATA.get("refuse", "I don't know this."),
                },
            )
            return
        if path in {"/", "/index.html"}:
            body = (ROOT / "web" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
            self.send_json(200, predict(payload.get("context"), payload.get("question")))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Loaded {CHECKPOINT.name} on {DEVICE}")
    print(f"Open http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
