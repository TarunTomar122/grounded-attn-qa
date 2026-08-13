"""Curriculum data for grounded QA: copy first, then refuse."""

from __future__ import annotations

import random
from typing import Any

REFUSE = "I don't know this."

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
        if other[2] == org:
            continue
        if other_city[0] == city:
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
        if a["context"] == b["context"]:
            continue
        if a["title"] == b["title"]:
            continue
        out.append(
            _pack_row(
                a["context"],
                b["question"],
                REFUSE,
                False,
                "squad_cross",
                "easy",
            )
        )
    return out


def build_dataset(
    *,
    phase: str,
    train_n: int,
    val_n: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
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
