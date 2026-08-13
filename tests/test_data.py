from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from grounded_qa.calibration import choose_threshold, sweep_thresholds
from grounded_qa.data import GroundedDataset, collate_examples
from grounded_qa.synthetic import (
    access_code_start_diagnostics,
    a1c_training_row,
    a1c_validation_splits,
    a1c_row,
    a1d_training_row,
    a2a_training_row,
    a2b_training_row,
    diversity_stats,
    entity_binding_training_row,
    entity_binding_validation_splits,
    generate_synthetic,
    phase_a_validation_splits,
    procedural_copy_row,
)
from grounded_qa.tokenizer import load_tokenizer, train_tokenizer


class DataTests(unittest.TestCase):
    def test_threshold_sweep_prefers_coverage_under_false_answer_constraint(self) -> None:
        points = sweep_thresholds([0.99, 0.80, 0.10, 0.01], [True, True, False, False])
        chosen = choose_threshold(points, max_false_answer_rate=0.0)
        self.assertGreater(chosen.threshold, 0.1)
        self.assertLess(chosen.threshold, 0.8)
        self.assertEqual(chosen.answer_coverage, 1.0)
        self.assertEqual(chosen.false_answer_rate, 0.0)

    def test_index_addressable_procedural_rows_are_reproducible(self) -> None:
        first = procedural_copy_row(42, 1_234, hard_distractors=True)
        self.assertEqual(first, procedural_copy_row(42, 1_234, hard_distractors=True))
        self.assertNotEqual(first["answer"], procedural_copy_row(42, 1_235, hard_distractors=True)["answer"])

    def test_compositional_pairs_are_reserved_from_training(self) -> None:
        train = generate_synthetic(256, 42, "train")
        compositional = generate_synthetic(256, 42, "compositional", novel_combinations=True)

        def held_pair(row: dict) -> bool:
            metadata = row["metadata"]
            q = int(metadata["question_structure"][1:])
            e = int(metadata["context_structure"][1:])
            return (q * 7 + e * 3) % 5 == 0

        self.assertTrue(all(not held_pair(row) for row in train))
        self.assertTrue(all(held_pair(row) for row in compositional))

    def test_v6_validation_splits_are_fixed_and_distinct(self) -> None:
        splits = phase_a_validation_splits(32)
        self.assertEqual(
            set(splits),
            {"familiar_unseen_values", "novel_combinations", "hard_distractors", "access_code"},
        )
        self.assertTrue(all(row["metadata"]["entity_set"] == "unseen" for rows in splits.values() for row in rows))
        self.assertTrue(all(row["metadata"]["novel_combination"] == "True" for row in splits["novel_combinations"]))
        self.assertTrue(all(row["metadata"]["relation"] == "access_code" for row in splits["access_code"]))
        self.assertEqual(splits, phase_a_validation_splits(32))

    def test_v6_has_broad_relations_and_low_exact_duplication(self) -> None:
        stats = diversity_stats(generate_synthetic(2_000, 42, "train"))
        self.assertEqual(stats["relations"], 4)
        self.assertGreater(stats["unique_questions"], 1_900)
        self.assertGreater(stats["unique_evidence"], 1_900)
        self.assertGreaterEqual(stats["estimated_question_combinations"], 60)
        self.assertGreaterEqual(stats["estimated_evidence_combinations"], 60)

    def test_access_code_start_diagnostics_control_distractor_composition(self) -> None:
        expected_facts = {"none": 1, "same_relation": 4, "same_subject": 4, "both": 5}
        for mode, fact_count in expected_facts.items():
            rows = access_code_start_diagnostics(4, 42, distractor_mode=mode, prefix_mode="shared")
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["metadata"]["distractor_mode"] == mode for row in rows))
            self.assertTrue(all(row["context"].count(".") == fact_count for row in rows))
            self.assertTrue(all(row["context"].count(row["answer"]) == 1 for row in rows))

        unique = access_code_start_diagnostics(4, 42, distractor_mode="both", prefix_mode="unique")
        self.assertTrue(all(row["metadata"]["prefix_mode"] == "unique" for row in unique))
        self.assertEqual(unique, access_code_start_diagnostics(4, 42, distractor_mode="both", prefix_mode="unique"))

    def test_entity_binding_training_mixture_and_curriculum(self) -> None:
        rows = [entity_binding_training_row(42, index, curriculum_step=1) for index in range(100)]
        modes = Counter(row["metadata"].get("binding_mix", row["metadata"].get("distractor_mode")) for row in rows)
        self.assertEqual(modes, Counter({"same_relation": 60, "both": 20, "none": 10, "existing": 10}))
        self.assertEqual({row["metadata"].get("candidate_count") for row in rows if row["metadata"].get("binding_mix") != "existing"}, {"1", "0"})
        hard_rows = [entity_binding_training_row(42, index, curriculum_step=500) for index in range(9)]
        self.assertEqual({row["metadata"].get("candidate_count") for row in hard_rows}, {"6", "0"})

    def test_entity_binding_validation_has_competing_subject_buckets(self) -> None:
        splits = entity_binding_validation_splits(2, 42)
        self.assertEqual(len(splits), 9)
        self.assertEqual(len(splits["binding_same_relation_6"]), 2)
        self.assertTrue(all(row["metadata"]["prefix_mode"] == "unique" for rows in splits.values() for row in rows))

    def test_a1c_prefix_levels_are_deterministic_and_copyable(self) -> None:
        rows = [a1c_row(42, index, prefix_level=index + 1, candidate_count=6) for index in range(4)]
        self.assertEqual(rows, [a1c_row(42, index, prefix_level=index + 1, candidate_count=6) for index in range(4)])
        self.assertTrue(all(row["answer"] in row["context"] for row in rows))
        self.assertTrue(all(len(row["metadata"]["candidate_values"].split("|")) == 7 for row in rows))

    def test_a1c_validation_has_all_required_suites(self) -> None:
        splits = a1c_validation_splits(8, 42)
        self.assertEqual(
            set(splits),
            {
                "a1c_unique_prefix",
                "a1c_shared_first",
                "a1c_shared_medium",
                "a1c_shared_long",
                "a1c_shared_near_identical",
                "a1c_shared_hard",
            },
        )
        self.assertTrue(all(len(rows) == 8 for rows in splits.values()))

    def test_a1c_training_mixture_uses_shared_and_controls(self) -> None:
        rows = [a1c_training_row(42, index, curriculum_step=500) for index in range(100)]
        families = Counter(row["metadata"]["template_family"] for row in rows)
        self.assertEqual(families["a1c_shared_prefix"], 80)
        self.assertEqual(families["entity_binding"], 20)
        self.assertEqual(sum(row["metadata"]["prefix_mode"] == "shared" for row in rows), 80)

    def test_a1d_training_mixture_is_50_30_20(self) -> None:
        rows = [a1d_training_row(42, index, curriculum_step=500) for index in range(100)]
        self.assertEqual(
            Counter(
                "shared" if row["metadata"]["template_family"] == "a1c_shared_prefix"
                else "a1_replay" if row["metadata"].get("a1d_mix") == "a1_replay"
                else "a1b_replay"
                for row in rows
            ),
            Counter({"shared": 50, "a1_replay": 30, "a1b_replay": 20}),
        )

    def test_a2a_training_mixture_is_70_20_10(self) -> None:
        squad = [procedural_copy_row(7, 0)]
        rows = [a2a_training_row(42, index, curriculum_step=1, squad_rows=squad) for index in range(100)]
        self.assertEqual(Counter(row["metadata"]["a2a_mix"] for row in rows), Counter({"squad2": 70, "a1_replay": 20, "a1d_replay": 10}))

    def test_a2b_training_mixture_is_65_25_10(self) -> None:
        squad = [procedural_copy_row(7, 0)]
        rows = [a2b_training_row(42, index, curriculum_step=1, squad_rows=squad) for index in range(100)]
        self.assertEqual(Counter(row["metadata"]["a2b_mix"] for row in rows), Counter({"squad2": 65, "a1_replay": 25, "a1d_replay": 10}))

    def test_encoder_marks_only_context_text_as_copyable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            train_tokenizer((row["question"] + " " + row["context"] for row in generate_synthetic(64)), path, 256)
            tokenizer, info = load_tokenizer(path)
            dataset = GroundedDataset(generate_synthetic(2), tokenizer, info, source_length=2048, target_length=64)
            self.assertEqual(len(dataset), 2)
            row = dataset[0]
            self.assertGreater(sum(row.context_mask), 0)
            self.assertEqual(len(row.context_mask), len(row.source_ids))
            self.assertFalse(row.context_mask[0])
            batch = collate_examples([row], info.pad_id)
            self.assertEqual(batch["context_mask"].shape, batch["source_ids"].shape)
            self.assertEqual(batch["gold_copy_positions"].shape, batch["target_ids"].shape)
            self.assertTrue((batch["gold_copy_positions"][0][:-1] >= 0).all())
            self.assertEqual(batch["gold_copy_positions"][0][-1].item(), -1)

    def test_answer_tokens_are_copyable_from_context_with_byte_level_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            rows = generate_synthetic(128, 42)
            train_tokenizer((row["question"] + " " + row["context"] for row in rows), path, 256)
            tokenizer, _ = load_tokenizer(path)
            for row in rows:
                source = tokenizer.encode(" " + row["context"], add_special_tokens=False).ids
                answer = tokenizer.encode(" " + row["answer"], add_special_tokens=False).ids
                self.assertTrue(
                    any(source[index : index + len(answer)] == answer for index in range(len(source))),
                    msg=f"answer is not copyable: {row['answer']!r}",
                )


if __name__ == "__main__":
    unittest.main()
