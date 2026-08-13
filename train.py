#!/usr/bin/env python3
"""Attention-only decoder: answer from the prompt, or refuse. Colab T4."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REFUSE = "I don't know this."
ANS_MARK = "\n\nAnswer: "


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 512
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.drop(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.ln(x))


class AttnOnlyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.size()
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(t, device=idx.device)))
        for blk in self.h:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, eos_id: int | None = None):
        for _ in range(max_new):
            cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(cond)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and int(nxt[0, 0]) == eos_id:
                break
        return idx

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def install_deps() -> None:
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "datasets", "transformers"],
    )


def load_tok():
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def pack(ctx: str, q: str, ans: str) -> str:
    return f"Context: {ctx.strip()}\n\nQuestion: {q.strip()}{ANS_MARK}{ans.strip()}"


SYNTH_FACTS = [
    ("Jane Diaz", "CEO", "Acme Corp", "2023", "$4.2 billion"),
    ("Kenji Mori", "CFO", "Northwind", "2024", "€810 million"),
    ("Priya Shah", "CTO", "Helios Labs", "2022", "$120 million"),
    ("Omar Haddad", "COO", "Redwood Steel", "2021", "$2.1 billion"),
    ("Lina Berg", "CMO", "Fjord Bank", "2024", "SEK 3.4 billion"),
    ("Mateo Cruz", "founder", "Orbita", "2020", "$18 million"),
    ("Aisha Khan", "president", "Cedar Health", "2023", "$640 million"),
    ("Noah Feldman", "chair", "Atlas Rail", "2019", "$5.5 billion"),
    ("Sofia Alves", "director", "Lumen Grid", "2024", "$77 million"),
    ("Wei Chen", "treasurer", "Pacific Yarn", "2022", "¥90 billion"),
]
CITIES = [
    ("Lisbon", "Portugal", "545,000"),
    ("Bergen", "Norway", "285,000"),
    ("Kyoto", "Japan", "1.46 million"),
    ("Cusco", "Peru", "430,000"),
    ("Accra", "Ghana", "2.6 million"),
    ("Tallinn", "Estonia", "450,000"),
]


def _pack_row(ctx: str, q: str, ans: str, answerable: bool, source: str, hardness: str) -> dict[str, Any]:
    return {
        "context": ctx,
        "question": q,
        "answer": ans,
        "answerable": answerable,
        "source": source,
        "hardness": hardness,
    }


def synth_copy(rng: random.Random, n: int) -> list[dict]:
    out = []
    while len(out) < n:
        name, role, org, year, money = rng.choice(SYNTH_FACTS)
        city, country, pop = rng.choice(CITIES)
        extra = rng.choice(SYNTH_FACTS)
        while extra[2] == org:
            extra = rng.choice(SYNTH_FACTS)
        ctx = (
            f"{name} is the {role} of {org}. "
            f"In {year}, {org} reported revenue of {money}. "
            f"{city} is a city in {country} with about {pop} people. "
            f"{extra[0]} works at {extra[2]}."
        )
        qa = rng.choice(
            [
                (f"Who is the {role} of {org}?", name),
                (f"What is {name}'s role at {org}?", role),
                (f"Which company is {name} the {role} of?", org),
                (f"What revenue did {org} report in {year}?", money),
                (f"In what year did {org} report revenue of {money}?", year),
                (f"Which country is {city} in?", country),
                (f"What is the population of {city}?", pop),
            ]
        )
        out.append(_pack_row(ctx, qa[0], qa[1], True, "synth_copy", "easy"))
    return out


def synth_trap(rng: random.Random, n: int) -> list[dict]:
    out = []
    while len(out) < n:
        name, role, org, year, money = rng.choice(SYNTH_FACTS)
        city, country, pop = rng.choice(CITIES)
        ctx = (
            f"{name} is the {role} of {org}. "
            f"In {year}, {org} reported revenue of {money}. "
            f"{city} is a city in {country} with about {pop} people."
        )
        other = rng.choice(SYNTH_FACTS)
        other_city = rng.choice(CITIES)
        if other[2] == org or other_city[0] == city:
            continue
        qa = rng.choice(
            [
                f"Who is the CEO of {other[2]}?",
                f"What revenue did {other[2]} report in {year}?",
                f"Which country is {other_city[0]} in?",
                f"What is {other[0]}'s role at {org}?",
                f"What was {org}'s revenue in 1999?",
            ]
        )
        out.append(_pack_row(ctx, qa, REFUSE, False, "synth_trap", "easy"))
    return out


def squad_rows(split: str, max_n: int, seed: int, *, answerable: bool | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out = []
    for i in idxs:
        row = ds[i]
        texts = [t.strip() for t in row["answers"]["text"] if t and t.strip()]
        is_yes = bool(texts)
        if answerable is True and not is_yes:
            continue
        if answerable is False and is_yes:
            continue
        ans = texts[0] if is_yes else REFUSE
        if is_yes and len(ans.split()) > 12:
            continue
        out.append(
            _pack_row(
                row["context"],
                row["question"],
                ans,
                is_yes,
                "squad_yes" if is_yes else "squad_official_no",
                "medium" if is_yes else "hard",
            )
        )
        if len(out) >= max_n:
            break
    return out


def squad_cross(split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=split)
    yes = [ds[i] for i in range(len(ds)) if ds[i]["answers"]["text"]]
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        a, b = rng.choice(yes), rng.choice(yes)
        if a["context"] == b["context"] or a["title"] == b["title"]:
            continue
        out.append(_pack_row(a["context"], b["question"], REFUSE, False, "squad_cross", "easy"))
    return out


def build_dataset(*, phase: str, train_n: int, val_n: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    if phase == "copy":
        train = synth_copy(rng, train_n // 2) + squad_rows("train", train_n - train_n // 2, seed, answerable=True)
        val = synth_copy(random.Random(seed + 1), val_n // 3) + squad_rows(
            "validation", val_n - val_n // 3, seed + 7, answerable=True
        )
    elif phase == "mix":
        n_yes = int(train_n * 0.7)
        n_no = train_n - n_yes
        train = (
            synth_copy(rng, n_yes // 4)
            + squad_rows("train", n_yes - n_yes // 4, seed, answerable=True)
            + synth_trap(rng, n_no // 3)
            + squad_cross("train", n_no - n_no // 3, seed + 3)
        )
        val = (
            synth_copy(random.Random(0), val_n // 4)
            + squad_rows("validation", val_n // 4, 1, answerable=True)
            + synth_trap(random.Random(2), val_n // 4)
            + squad_cross("validation", val_n - 3 * (val_n // 4), 3)
        )
    else:
        raise ValueError(f"unknown phase {phase}")
    rng.shuffle(train)
    random.Random(seed + 99).shuffle(val)
    return train[:train_n], val[:val_n]


def counts(rows: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for r in rows:
        key = f"{r['source']}:{'yes' if r['answerable'] else 'no'}"
        c[key] = c.get(key, 0) + 1
    return c


def encode_rows(tok, raw: list[dict], block_size: int) -> list[dict]:
    out = []
    for row in raw:
        prompt = pack(row["context"], row["question"], "")
        prompt_ids = tok.encode(prompt)
        ans_ids = tok.encode(row["answer"]) + [tok.eos_token_id]
        ids = prompt_ids + ans_ids
        if len(ids) > block_size - 4:
            continue
        out.append(
            {
                "ids": ids,
                "prompt_len": len(prompt_ids),
                "answerable": row["answerable"],
                "gold": row["answer"],
                "prompt": prompt,
                "source": row.get("source", ""),
            }
        )
    return out


class PackedQA(Dataset):
    def __init__(self, rows: list[dict], block_size: int):
        self.rows = rows
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        ids = row["ids"][: self.block_size]
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.full((self.block_size,), -100, dtype=torch.long)
        t = torch.tensor(ids, dtype=torch.long)
        n = t.numel()
        x[:n] = t
        plen = min(row["prompt_len"], n)
        if plen > 0 and plen < n:
            y[plen - 1 : n - 1] = t[plen:n]
        return x, y


def decode_after_mark(tok, text: str) -> str:
    if ANS_MARK in text:
        text = text.split(ANS_MARK, 1)[1]
    return text.split(tok.eos_token or "<|endoftext|>")[0].strip()


def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def is_refuse(s: str) -> bool:
    t = normalize(s)
    return t.startswith(normalize(REFUSE)) or t.startswith("i don't know")


def evaluate(model, tok, rows: list[dict], device, max_n: int = 64) -> dict:
    model.eval()
    n = min(max_n, len(rows))
    em = 0
    refuse_ok = 0
    refuse_bad = 0
    n_yes = 0
    n_no = 0
    samples = []
    for row in rows[:n]:
        ids = torch.tensor([tok.encode(row["prompt"])], device=device)
        out = model.generate(ids, max_new=32, eos_id=tok.eos_token_id)
        pred = decode_after_mark(tok, tok.decode(out[0].tolist()))
        gold = row["gold"]
        if row["answerable"]:
            n_yes += 1
            if normalize(pred) == normalize(gold) or normalize(gold) in normalize(pred):
                em += 1
            if is_refuse(pred):
                refuse_bad += 1
        else:
            n_no += 1
            if is_refuse(pred):
                refuse_ok += 1
        if len(samples) < 6:
            samples.append((row["answerable"], gold, pred))
    return {
        "n": n,
        "em": em / max(n_yes, 1),
        "refuse_recall": refuse_ok / max(n_no, 1),
        "false_refuse": refuse_bad / max(n_yes, 1),
        "n_yes": n_yes,
        "n_no": n_no,
        "samples": samples,
    }


def cosine_lr(step: int, warmup: int, total: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return max_lr * 0.1 + 0.5 * (1 + math.cos(math.pi * p)) * (max_lr * 0.9)


def train(args: argparse.Namespace) -> None:
    install_deps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda:", torch.cuda.is_available(), flush=True)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    cfg = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )
    model = AttnOnlyGPT(cfg).to(device)
    print(f"params: {model.n_params():,} ({model.n_params() / 1e6:.2f}M)", flush=True)
    print("config:", asdict(cfg), flush=True)
    print("refuse:", repr(REFUSE), flush=True)

    tok = load_tok()
    raw_train, raw_val = build_dataset(phase=args.phase, train_n=args.train_n, val_n=args.val_n)
    print("raw_train", counts(raw_train), flush=True)
    print("raw_val", counts(raw_val), flush=True)
    train_rows = encode_rows(tok, raw_train, cfg.block_size)
    val_rows = encode_rows(tok, raw_val, cfg.block_size)
    print(f"encoded train={len(train_rows)} val={len(val_rows)} phase={args.phase}", flush=True)
    loader = DataLoader(
        PackedQA(train_rows, cfg.block_size),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    it = iter(loader)
    t0 = time.time()
    last = float("nan")
    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step - 1, args.warmup, args.steps, args.lr)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        last = float(loss.item())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step}/{args.steps}  loss={last:.4f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            m = evaluate(model, tok, val_rows, device, max_n=args.eval_n)
            print(
                f"  eval@{step} em={m['em']:.3f} refuse={m['refuse_recall']:.3f} "
                f"false_refuse={m['false_refuse']:.3f}",
                flush=True,
            )
            model.train()

    metrics = evaluate(model, tok, val_rows, device, max_n=args.eval_n)
    print("--- eval ---", flush=True)
    print(
        f"em={metrics['em']:.3f}  refuse_recall={metrics['refuse_recall']:.3f}  "
        f"false_refuse={metrics['false_refuse']:.3f}  "
        f"(yes={metrics['n_yes']} no={metrics['n_no']})",
        flush=True,
    )
    for ans, gold, pred in metrics["samples"]:
        print(f"  [{'QA' if ans else 'NO'}] gold={gold!r} pred={pred!r}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "grounded_attn.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "refuse": REFUSE}, ckpt)
    print(f"saved {ckpt}  {time.time() - t0:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--train-n", type=int, default=8000)
    p.add_argument("--val-n", type=int, default=400)
    p.add_argument("--eval-n", type=int, default=80)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--phase", choices=["copy", "mix"], default="copy")
    p.add_argument("--out-dir", default="/content/grounded_qa")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
