from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticRow:
    id: str
    question: str
    context: str
    answer: str
    answerable: bool
    evidence: str
    source: str = "synthetic_procedural_copy_v6"
    phase: str = "copy"
    hardness: str = "medium"
    metadata: dict[str, str] | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "answerable": self.answerable,
            "evidence": self.evidence,
            "source": self.source,
            "phase": self.phase,
            "hardness": self.hardness,
            "metadata": self.metadata or {},
        }


RELATIONS = {
    "identifier": ("identifier", "device code", "record ID"),
    "access_code": ("access code", "entry code", "security code"),
    "serial_number": ("serial number", "serial code", "unit number"),
    "registry_key": ("registry key", "record key", "registration key"),
}
QUESTION_STRUCTURES = (
    "What is {subject}'s {noun}?",
    "Which {noun} belongs to {subject}?",
    "What {noun} is assigned to {subject}?",
    "According to the record, what {noun} has {subject}?",
    "Name the {noun} for {subject}.",
)
EVIDENCE_STRUCTURES = (
    "{subject}'s {noun} is {value}.",
    "The {noun} assigned to {subject} is {value}.",
    "{subject} uses {value} as its {noun}.",
    "Records list {value} as the {noun} for {subject}.",
    "For {subject}, the registered {noun} is {value}.",
)
CONTEXT_OPENERS = ("Record:", "Case file:", "Inventory note:", "Archived entry:")
DIAGNOSTIC_PREFIXES = ("QF", "XK", "LM", "ZR")
BINDING_PREFIXES = ("QF", "XK", "ZR", "LM", "CD", "GH", "JK", "PQ", "TZ", "MN", "RS", "WX", "YZ")
BINDING_CANDIDATE_COUNTS = (1, 2, 4, 6)
SHARED_PREFIX_LEVELS = (1, 2, 3, 4)
SHARED_LEVEL_ONE_PREFIXES = ("AB", "EF", "UV", "OU", "OA", "EO", "EA")
SHARED_LEVEL_TWO_STEMS = ("18", "78", "99", "63", "19", "92", "57")
SHARED_LEVEL_THREE_STEMS = ("21", "98", "55", "23", "76", "42", "87")
SHARED_LEVEL_FOUR_SUFFIXES = "MNPQXYZ"


def _stream(seed: int, index: int, salt: int) -> random.Random:
    return random.Random((seed * 1_000_003 + index * 7_919 + salt * 104_729) & 0xFFFFFFFFFFFF)


def _word(rng: random.Random) -> str:
    consonants, vowels = "bcdfghjklmnpqrstvwxyz", "aeiou"
    value = "".join(rng.choice(consonants) + rng.choice(vowels) for _ in range(rng.randint(2, 4)))
    return value[: rng.randint(5, 8)].capitalize()


def _subject(rng: random.Random) -> str:
    return f"{_word(rng)} {_word(rng)} Unit"


def _value(rng: random.Random, relation: str) -> str:
    if relation == "access_code":
        return f"{rng.choice(('QF', 'XK', 'LM', 'ZR'))}-{rng.randint(1_000_000, 9_999_999)}-{rng.choice('MNPQXYZ')}"
    if relation == "serial_number":
        return f"SN-{rng.randint(100, 999):03d}-{rng.randint(100000, 999999)}"
    if relation == "registry_key":
        return f"RK{rng.randint(10, 99)}-{rng.choice('ABCDEFGH')}{rng.randint(1000, 9999)}-{rng.choice('KLMNPQ')}"
    return f"ID-{rng.choice('ABCDEFGH')}{rng.randint(10000, 99999)}-{rng.choice('MNPQXYZ')}"


def _pair(rng: random.Random, novel: bool) -> tuple[int, int]:
    while True:
        question, evidence = rng.randrange(len(QUESTION_STRUCTURES)), rng.randrange(len(EVIDENCE_STRUCTURES))
        reserved = (question * 7 + evidence * 3) % 5 == 0
        if reserved == novel:
            return question, evidence


def _lexical_overlap(question: str, evidence: str) -> str:
    question_words = set(re.findall(r"[a-z]+", question.lower()))
    evidence_words = set(re.findall(r"[a-z]+", evidence.lower()))
    value = len(question_words & evidence_words) / max(len(question_words), 1)
    return "high" if value >= 0.45 else "medium" if value >= 0.2 else "low"


def procedural_copy_row(
    seed: int,
    index: int,
    *,
    split: str = "train",
    entity_set: str = "train",
    novel_combinations: bool = False,
    hard_distractors: bool = False,
    relation_filter: str | None = None,
) -> dict:
    """One deterministic example. Every RNG stream is independent and index-addressable."""
    value_seed = seed if entity_set == "train" else seed + 10_000_019
    value_rng = _stream(value_seed, index, 1)
    relation_rng = _stream(seed, index, 2)
    language_rng = _stream(seed, index, 3)
    layout_rng = _stream(seed, index, 4)
    relations = tuple(RELATIONS)
    relation = relation_filter or relation_rng.choice(relations)
    subject, answer = _subject(value_rng), _value(value_rng, relation)
    question_index, evidence_index = _pair(language_rng, novel_combinations)
    noun = language_rng.choice(RELATIONS[relation])
    question = QUESTION_STRUCTURES[question_index].format(subject=subject, noun=noun)
    evidence = EVIDENCE_STRUCTURES[evidence_index].format(subject=subject, noun=noun, value=answer)

    facts = []
    # Same-relation distractors force subject matching; same-subject distractors force relation matching.
    for distractor_index in range(layout_rng.randint(5, 9) if hard_distractors else layout_rng.randint(3, 6)):
        distractor_relation = relation if distractor_index == 0 or (hard_distractors and layout_rng.random() < 0.45) else relation_rng.choice(relations)
        distractor_subject = subject if distractor_index == 1 else _subject(value_rng)
        distractor_noun = language_rng.choice(RELATIONS[distractor_relation])
        distractor_value = _value(value_rng, distractor_relation)
        structure = EVIDENCE_STRUCTURES[language_rng.randrange(len(EVIDENCE_STRUCTURES))]
        facts.append(structure.format(subject=distractor_subject, noun=distractor_noun, value=distractor_value))
    layout_rng.shuffle(facts)
    position_bucket = layout_rng.randrange(4)
    position = (0, max(1, len(facts) // 3), max(1, 2 * len(facts) // 3), len(facts))[position_bucket]
    facts.insert(min(position, len(facts)), evidence)
    separator = layout_rng.choice((" ", "\n", "\n\n"))
    context = f"{layout_rng.choice(CONTEXT_OPENERS)}{separator}{separator.join(facts)}"
    return SyntheticRow(
        id=f"procedural-v6-{split}-{index:09d}",
        question=question,
        context=context,
        answer=answer,
        answerable=True,
        evidence=evidence,
        hardness="hard" if hard_distractors else "medium",
        metadata={
            "relation": relation,
            "template_family": "procedural",
            "question_structure": f"q{question_index}",
            "context_structure": f"e{evidence_index}",
            "entity_set": entity_set,
            "answer_position_bucket": str(position_bucket),
            "lexical_overlap_bucket": _lexical_overlap(question, evidence),
            "novel_combination": str(novel_combinations),
        },
    ).as_dict()


def generate_synthetic(
    n: int,
    seed: int = 42,
    split: str = "train",
    *,
    entity_set: str = "train",
    novel_combinations: bool = False,
    hard_distractors: bool = False,
    relation_filter: str | None = None,
) -> list[dict]:
    return [
        procedural_copy_row(
            seed,
            index,
            split=split,
            entity_set=entity_set,
            novel_combinations=novel_combinations,
            hard_distractors=hard_distractors,
            relation_filter=relation_filter,
        )
        for index in range(n)
    ]


def access_code_start_diagnostics(
    n: int = 256,
    seed: int = 42,
    *,
    distractor_mode: str = "both",
    prefix_mode: str = "shared",
) -> list[dict]:
    """Controlled access-code rows for entity/relation/prefix-binding diagnosis."""
    if distractor_mode not in {"none", "same_relation", "same_subject", "both"}:
        raise ValueError(f"unknown distractor mode: {distractor_mode}")
    if prefix_mode not in {"shared", "unique"}:
        raise ValueError(f"unknown prefix mode: {prefix_mode}")

    rows = []
    for index in range(n):
        rng = _stream(seed, index, 31)
        subject = _subject(rng)
        target_prefix = "QF"
        target_value = _diagnostic_access_value(rng, target_prefix)
        target = f"{subject}'s access code is {target_value}."
        facts = []

        if distractor_mode in {"same_relation", "both"}:
            count = 2 if distractor_mode == "both" else 3
            for distractor_index in range(count):
                distractor_subject = _subject(rng)
                prefix = target_prefix if prefix_mode == "shared" else DIAGNOSTIC_PREFIXES[distractor_index + 1]
                value = _diagnostic_access_value(rng, prefix)
                facts.append(f"{distractor_subject}'s access code is {value}.")

        if distractor_mode in {"same_subject", "both"}:
            relations = ("identifier", "serial_number", "registry_key")
            count = 2 if distractor_mode == "both" else 3
            for relation in relations[:count]:
                noun = RELATIONS[relation][0]
                facts.append(f"{subject}'s {noun} is {_value(rng, relation)}.")

        facts.append(target)
        rng.shuffle(facts)
        context = "Record:\n" + "\n".join(facts)
        rows.append(SyntheticRow(
            id=f"start-diagnostic-{distractor_mode}-{prefix_mode}-{index:09d}",
            question=f"What is {subject}'s access code?",
            context=context,
            answer=target_value,
            answerable=True,
            evidence=target,
            hardness=distractor_mode,
            metadata={
                "relation": "access_code",
                "template_family": "start_diagnostic",
                "question_structure": "fixed_access_code",
                "context_structure": distractor_mode,
                "entity_set": "unseen",
                "distractor_mode": distractor_mode,
                "prefix_mode": prefix_mode,
                "lexical_overlap_bucket": "high",
            },
        ).as_dict())
    return rows


def _diagnostic_access_value(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(1_000_000, 9_999_999)}-{rng.choice('MNPQXYZ')}"


def _shared_access_value(rng: random.Random, level: int, candidate_index: int) -> str:
    if level == 1:
        prefix = SHARED_LEVEL_ONE_PREFIXES[candidate_index]
        return f"{prefix}-{rng.randint(1_000_000, 9_999_999)}-{rng.choice('MNPQXYZ')}"
    if level == 2:
        number = f"{SHARED_LEVEL_TWO_STEMS[candidate_index]}{rng.randint(10000, 99999)}"
        return f"QF-{number}-{rng.choice('MNPQXYZ')}"
    if level == 3:
        number = SHARED_LEVEL_THREE_STEMS[candidate_index]
        return f"QF-430{number}-{rng.choice('MNPQXYZ')}-{rng.randint(100, 999)}"
    if level == 4:
        return f"QF-4302358-{SHARED_LEVEL_FOUR_SUFFIXES[candidate_index]}-{rng.randint(100, 999)}"
    raise ValueError(f"unknown shared prefix level: {level}")


def a1c_row(
    seed: int,
    index: int,
    *,
    split: str = "train",
    distractor_mode: str = "same_relation",
    candidate_count: int = 1,
    prefix_level: int = 1,
    prefix_mode: str = "shared",
) -> dict:
    """One deterministic A1c row with controlled BPE-prefix overlap."""
    if distractor_mode not in {"none", "same_relation", "both"}:
        raise ValueError(f"unknown A1c distractor mode: {distractor_mode}")
    if prefix_mode not in {"shared", "unique"}:
        raise ValueError(f"unknown A1c prefix mode: {prefix_mode}")
    if candidate_count < 0 or candidate_count > len(BINDING_CANDIDATE_COUNTS) + 2:
        raise ValueError("candidate_count must be between 0 and 6")
    if prefix_mode == "shared" and prefix_level not in SHARED_PREFIX_LEVELS:
        raise ValueError(f"unknown shared prefix level: {prefix_level}")

    rng = _stream(seed, index, 51)
    subject = _subject(rng)
    target_value = (
        _shared_access_value(rng, prefix_level, 0)
        if prefix_mode == "shared"
        else _diagnostic_access_value(rng, BINDING_PREFIXES[0])
    )
    target = f"{subject}'s access code is {target_value}."
    facts = [target]
    candidate_values = [target_value]
    if distractor_mode in {"same_relation", "both"}:
        for distractor_index in range(candidate_count):
            distractor_subject = _subject(rng)
            candidate_index = distractor_index + 1
            if prefix_mode == "shared":
                value = _shared_access_value(rng, prefix_level, candidate_index)
            else:
                value = _diagnostic_access_value(rng, BINDING_PREFIXES[candidate_index])
            candidate_values.append(value)
            facts.append(f"{distractor_subject}'s access code is {value}.")
    if distractor_mode == "both":
        for relation in ("identifier", "serial_number"):
            facts.append(f"{subject}'s {RELATIONS[relation][0]} is {_value(rng, relation)}.")
    rng.shuffle(facts)
    return SyntheticRow(
        id=f"a1c-{split}-{distractor_mode}-{candidate_count}-{prefix_mode}-{prefix_level}-{index:09d}",
        question=f"What is {subject}'s access code?",
        context="Record:\n" + "\n".join(facts),
        answer=target_value,
        answerable=True,
        evidence=target,
        hardness=distractor_mode,
        metadata={
            "relation": "access_code",
            "template_family": "a1c_shared_prefix",
            "question_structure": "fixed_access_code",
            "context_structure": distractor_mode,
            "entity_set": "unseen",
            "distractor_mode": distractor_mode,
            "prefix_mode": prefix_mode,
            "prefix_level": str(prefix_level if prefix_mode == "shared" else 0),
            "candidate_count": str(candidate_count),
            "candidate_values": "|".join(candidate_values),
            "lexical_overlap_bucket": "high",
        },
    ).as_dict()


def a1c_training_row(seed: int, index: int, curriculum_step: int) -> dict:
    """Deterministic A1c mixture: 60/20/10/10 shared, mixed, unique, easy."""
    bucket = index % 10
    candidate_count = BINDING_CANDIDATE_COUNTS[(index + curriculum_step) % len(BINDING_CANDIDATE_COUNTS)]
    level_limit = min(1 + max(curriculum_step - 1, 0) // 125, len(SHARED_PREFIX_LEVELS))
    prefix_level = 1 + ((index + curriculum_step) % level_limit)
    if bucket < 6:
        return a1c_row(seed, index, split="a1c_train", distractor_mode="same_relation", candidate_count=candidate_count, prefix_level=prefix_level)
    if bucket < 8:
        return a1c_row(seed, index, split="a1c_train", distractor_mode="both", candidate_count=candidate_count, prefix_level=prefix_level)
    if bucket == 8:
        return entity_binding_row(seed + 71_003, index, split="a1c_unique", distractor_mode="both", candidate_count=candidate_count, prefix_mode="unique")
    return entity_binding_row(seed + 71_003, index, split="a1c_easy", distractor_mode="none", candidate_count=0, prefix_mode="unique")


def a1d_training_row(seed: int, index: int, curriculum_step: int) -> dict:
    """A1d replay mixture: 50% shared-prefix, 30% A1 procedural, 20% A1b unique."""
    candidate_count = BINDING_CANDIDATE_COUNTS[(index + curriculum_step) % len(BINDING_CANDIDATE_COUNTS)]
    level_limit = min(1 + max(curriculum_step - 1, 0) // 125, len(SHARED_PREFIX_LEVELS))
    prefix_level = 1 + ((index + curriculum_step) % level_limit)
    bucket = index % 10
    if bucket < 4:
        return a1c_row(
            seed,
            index,
            split="a1d_shared",
            distractor_mode="same_relation",
            candidate_count=candidate_count,
            prefix_level=prefix_level,
        )
    if bucket == 4:
        return a1c_row(
            seed,
            index,
            split="a1d_shared_hard",
            distractor_mode="both",
            candidate_count=candidate_count,
            prefix_level=prefix_level,
        )
    if bucket < 8:
        row = procedural_copy_row(
            seed + 91_007,
            index,
            split="a1d_a1_replay",
            entity_set="train",
            hard_distractors=index % 2 == 0,
        )
        row["metadata"] = {**row["metadata"], "a1d_mix": "a1_replay"}
        return row
    return entity_binding_row(
        seed + 71_003,
        index,
        split="a1d_a1b_replay",
        distractor_mode="both",
        candidate_count=candidate_count,
        prefix_mode="unique",
    )


def a2a_training_row(
    seed: int,
    index: int,
    curriculum_step: int,
    squad_rows: list[dict],
) -> dict:
    """A2a 70/20/10 real-QA and replay mixture."""
    bucket = index % 10
    if bucket < 7:
        row = squad_rows[(seed * 1_000_003 + index * 97 + curriculum_step) % len(squad_rows)]
        return {**row, "metadata": {**row["metadata"], "a2a_mix": "squad2"}}
    if bucket < 9:
        row = procedural_copy_row(seed + 91_007, index, split="a2a_a1_replay", entity_set="train", hard_distractors=True)
        return {**row, "metadata": {**row["metadata"], "a2a_mix": "a1_replay"}}
    row = a1c_row(
        seed + 71_003,
        index,
        split="a2a_a1d_replay",
        distractor_mode="both",
        candidate_count=BINDING_CANDIDATE_COUNTS[(index + curriculum_step) % len(BINDING_CANDIDATE_COUNTS)],
        prefix_level=1 + (index % len(SHARED_PREFIX_LEVELS)),
    )
    return {**row, "metadata": {**row["metadata"], "a2a_mix": "a1d_replay"}}


def a2b_training_row(
    seed: int,
    index: int,
    curriculum_step: int,
    squad_rows: list[dict],
) -> dict:
    """A2b uses extra procedural replay after A2a synthetic retention collapsed."""
    bucket = index % 20
    if bucket < 13:
        row = squad_rows[(seed * 1_000_003 + index * 97 + curriculum_step) % len(squad_rows)]
        return {**row, "metadata": {**row["metadata"], "a2b_mix": "squad2"}}
    if bucket < 18:
        row = procedural_copy_row(seed + 91_007, index, split="a2b_a1_replay", entity_set="train", hard_distractors=True)
        return {**row, "metadata": {**row["metadata"], "a2b_mix": "a1_replay"}}
    row = a1c_row(
        seed + 17_003,
        index,
        prefix_level=SHARED_PREFIX_LEVELS[(index + curriculum_step) % len(SHARED_PREFIX_LEVELS)],
        candidate_count=BINDING_CANDIDATE_COUNTS[(index + curriculum_step) % len(BINDING_CANDIDATE_COUNTS)],
        prefix_mode="shared",
        split="a2b_a1d_replay",
    )
    return {**row, "metadata": {**row["metadata"], "a2b_mix": "a1d_replay"}}


def a1c_validation_splits(n: int = 512, seed: int = 42) -> dict[str, list[dict]]:
    """Fixed A1c controls, prefix levels, hard distractors, and candidate counts."""
    return {
        "a1c_unique_prefix": [
            a1c_row(seed, index, split="unique", distractor_mode="both", candidate_count=BINDING_CANDIDATE_COUNTS[index % 4], prefix_mode="unique")
            for index in range(n)
        ],
        "a1c_shared_first": [
            a1c_row(seed + 1, index, split="first", distractor_mode="same_relation", candidate_count=BINDING_CANDIDATE_COUNTS[index % 4], prefix_level=1)
            for index in range(n)
        ],
        "a1c_shared_medium": [
            a1c_row(seed + 2, index, split="medium", distractor_mode="same_relation", candidate_count=BINDING_CANDIDATE_COUNTS[index % 4], prefix_level=2)
            for index in range(n)
        ],
        "a1c_shared_long": [
            a1c_row(seed + 3, index, split="long", distractor_mode="same_relation", candidate_count=BINDING_CANDIDATE_COUNTS[index % 4], prefix_level=3)
            for index in range(n)
        ],
        "a1c_shared_near_identical": [
            a1c_row(seed + 4, index, split="near", distractor_mode="same_relation", candidate_count=BINDING_CANDIDATE_COUNTS[index % 4], prefix_level=4)
            for index in range(n)
        ],
        "a1c_shared_hard": [
            a1c_row(seed + 5, index, split="hard", distractor_mode="both", candidate_count=BINDING_CANDIDATE_COUNTS[index % 4], prefix_level=1 + (index % 4))
            for index in range(n)
        ],
    }


def entity_binding_row(
    seed: int,
    index: int,
    *,
    split: str = "train",
    distractor_mode: str = "same_relation",
    candidate_count: int = 1,
    prefix_mode: str = "unique",
) -> dict:
    """One controlled access-code row for entity-binding training/evaluation."""
    if distractor_mode not in {"none", "same_relation", "both"}:
        raise ValueError(f"unknown entity-binding distractor mode: {distractor_mode}")
    if prefix_mode not in {"shared", "unique"}:
        raise ValueError(f"unknown prefix mode: {prefix_mode}")
    if candidate_count < 0 or candidate_count > len(BINDING_PREFIXES) - 1:
        raise ValueError(f"candidate_count must be between 0 and {len(BINDING_PREFIXES) - 1}")

    rng = _stream(seed, index, 41)
    subject = _subject(rng)
    target_value = _diagnostic_access_value(rng, BINDING_PREFIXES[0])
    target = f"{subject}'s access code is {target_value}."
    facts = [target]
    if distractor_mode in {"same_relation", "both"}:
        for distractor_index in range(candidate_count):
            distractor_subject = _subject(rng)
            prefix = BINDING_PREFIXES[0] if prefix_mode == "shared" else BINDING_PREFIXES[distractor_index + 1]
            value = _diagnostic_access_value(rng, prefix)
            facts.append(f"{distractor_subject}'s access code is {value}.")
    if distractor_mode == "both":
        for relation in ("identifier", "serial_number"):
            facts.append(f"{subject}'s {RELATIONS[relation][0]} is {_value(rng, relation)}.")
    rng.shuffle(facts)
    return SyntheticRow(
        id=f"entity-binding-{split}-{distractor_mode}-{candidate_count}-{prefix_mode}-{index:09d}",
        question=f"What is {subject}'s access code?",
        context="Record:\n" + "\n".join(facts),
        answer=target_value,
        answerable=True,
        evidence=target,
        hardness=distractor_mode,
        metadata={
            "relation": "access_code",
            "template_family": "entity_binding",
            "question_structure": "fixed_access_code",
            "context_structure": distractor_mode,
            "entity_set": "unseen",
            "distractor_mode": distractor_mode,
            "prefix_mode": prefix_mode,
            "candidate_count": str(candidate_count),
            "lexical_overlap_bucket": "high",
        },
    ).as_dict()


def entity_binding_training_row(seed: int, index: int, curriculum_step: int) -> dict:
    """Deterministic 60/20/10/10 A1b mixture with a four-level difficulty ramp."""
    bucket = index % 10
    if bucket < 6:
        mode = "same_relation"
    elif bucket < 8:
        mode = "both"
    elif bucket < 9:
        mode = "none"
    else:
        row = procedural_copy_row(seed + 71_003, index, split="a1b_existing", entity_set="train")
        row["metadata"] = {**row["metadata"], "binding_mix": "existing", "candidate_count": "mixed"}
        return row
    level = min(max(curriculum_step - 1, 0) // 125, len(BINDING_CANDIDATE_COUNTS) - 1)
    return entity_binding_row(
        seed,
        index,
        split="a1b_train",
        distractor_mode=mode,
        candidate_count=0 if mode == "none" else BINDING_CANDIDATE_COUNTS[level],
        prefix_mode="unique",
    )


def entity_binding_validation_splits(n: int = 256, seed: int = 42) -> dict[str, list[dict]]:
    splits = {"binding_easy": [entity_binding_row(seed, index, split="easy", distractor_mode="none", candidate_count=0) for index in range(n)]}
    for candidate_count in BINDING_CANDIDATE_COUNTS:
        splits[f"binding_same_relation_{candidate_count}"] = [
            entity_binding_row(seed + candidate_count, index, split="same_relation", candidate_count=candidate_count)
            for index in range(n)
        ]
        splits[f"binding_both_unique_{candidate_count}"] = [
            entity_binding_row(seed + 10_000 + candidate_count, index, split="both", candidate_count=candidate_count)
            for index in range(n)
        ]
    return splits


def phase_a_validation_splits(n: int = 1_000, seed: int = 42) -> dict[str, list[dict]]:
    """Fixed A1 diagnostics. Only values are OOD; no pretend-English paraphrase benchmark."""
    return {
        "familiar_unseen_values": generate_synthetic(n, seed + 1, "familiar", entity_set="unseen"),
        "novel_combinations": generate_synthetic(n, seed + 2, "compositional", entity_set="unseen", novel_combinations=True, hard_distractors=True),
        "hard_distractors": generate_synthetic(n, seed + 3, "hard", entity_set="unseen", hard_distractors=True),
        "access_code": generate_synthetic(n, seed + 4, "access", entity_set="unseen", hard_distractors=True, relation_filter="access_code"),
    }


def phase_a_test_set(n: int = 10_000, seed: int = 42) -> list[dict]:
    return generate_synthetic(n, seed + 50_000, "test", entity_set="unseen", novel_combinations=True, hard_distractors=True)


def diversity_stats(rows: list[dict]) -> dict[str, float | int]:
    questions = [row["question"] for row in rows]
    evidence = [row["evidence"] for row in rows]
    return {
        "relations": len({row["metadata"]["relation"] for row in rows}),
        "unique_questions": len(set(questions)),
        "unique_evidence": len(set(evidence)),
        "duplicate_question_pct": 100 * (1 - len(set(questions)) / max(len(questions), 1)),
        "duplicate_evidence_pct": 100 * (1 - len(set(evidence)) / max(len(evidence), 1)),
        "estimated_question_combinations": len(RELATIONS) * len(QUESTION_STRUCTURES) * 3,
        "estimated_evidence_combinations": len(RELATIONS) * len(EVIDENCE_STRUCTURES) * 3,
    }
