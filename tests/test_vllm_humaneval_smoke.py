import json
import tempfile
import unittest
from pathlib import Path

from src.vllm_humaneval_smoke import (
    FULL_TASK_IDS,
    SMOKE_TASK_IDS,
    collect_model_metadata,
    compare_generation_receipts,
    extract_code,
    load_task_manifest,
    score_counts,
    score_evalplus_subset,
    select_fixed_tasks,
    task_ids_for_scope,
)


class ExtractionTests(unittest.TestCase):
    def test_extracts_code_after_think_and_from_python_fence(self):
        text = (
            "<think>We need a simple implementation.</think>\n"
            "Here is the answer:\n```python\n"
            "def has_close_elements(numbers, threshold):\n"
            "    return any(abs(a - b) < threshold for a in numbers for b in numbers if a != b)\n"
            "```\n"
        )
        code = extract_code(text, "has_close_elements")
        self.assertTrue(code.startswith("def has_close_elements"))
        self.assertNotIn("<think>", code)
        self.assertNotIn("```", code)

    def test_extracts_direct_code_with_control_markers_and_imports(self):
        text = (
            "<|assistant|><think>internal reasoning</think>\n"
            "from math import sqrt\n\n"
            "def distance(x):\n    return sqrt(x)\n<|eot_id|>"
        )
        self.assertEqual(
            extract_code(text, "distance"),
            "from math import sqrt\n\ndef distance(x):\n    return sqrt(x)",
        )

    def test_prefers_the_code_segment_containing_the_entry_point(self):
        text = (
            "```python\ndef helper():\n    return 1\n```\n"
            "```python\ndef target(value):\n    return value + 1\n```"
        )
        self.assertEqual(
            extract_code(text, "target"),
            "def target(value):\n    return value + 1",
        )


class ManifestTests(unittest.TestCase):
    def _manifest(self, directory: Path) -> Path:
        rows = [
            {"task_id": task_id, "prompt": f"def {task_id.rsplit('/', 1)[1]}():", "entry_point": task_id.rsplit('/', 1)[1]}
            for task_id in reversed(SMOKE_TASK_IDS)
        ]
        path = directory / "humaneval_plus.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_fixed_subset_order_is_independent_of_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = load_task_manifest(self._manifest(Path(directory)))
            selected = select_fixed_tasks(rows)
            self.assertEqual([row["task_id"] for row in selected], list(SMOKE_TASK_IDS))

    def test_full_scope_is_frozen_to_all_164_tasks(self):
        self.assertEqual(task_ids_for_scope("full"), FULL_TASK_IDS)
        self.assertEqual(FULL_TASK_IDS[0], "HumanEval/0")
        self.assertEqual(FULL_TASK_IDS[-1], "HumanEval/163")
        self.assertEqual(len(FULL_TASK_IDS), 164)

    def test_unknown_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported HumanEval\\+ scope"):
            task_ids_for_scope("everything")


class MetadataTests(unittest.TestCase):
    def test_metadata_records_config_hash_and_local_files_without_weights_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            (root / "config.json").write_text(
                json.dumps({"model_type": "minicpm", "hidden_size": 1536}),
                encoding="utf-8",
            )
            (root / "model.safetensors").write_bytes(b"local-weight-placeholder")
            metadata = collect_model_metadata(root)
            checkpoint = metadata["checkpoint"]
            self.assertEqual(checkpoint["kind"], "directory")
            self.assertIn("config.json", checkpoint["config_files"])
            self.assertEqual(
                checkpoint["config_files"]["config.json"]["json"]["model_type"],
                "minicpm",
            )
            self.assertEqual(checkpoint["weight_files"][0]["name"], "model.safetensors")
            self.assertNotIn("sha256", checkpoint["weight_files"][0])


class ComparisonTests(unittest.TestCase):
    def _receipt(self, label: str, rows: list[dict]) -> dict:
        return {"model": {"path": label}, "tasks": rows}

    def test_exact_text_and_tokens_and_score_counts(self):
        reference_rows = [
            {"task_id": "HumanEval/0", "generated_text": "a", "token_ids": [1], "solution": "a"},
            {"task_id": "HumanEval/1", "generated_text": "b", "token_ids": [2], "solution": "b"},
        ]
        candidate_rows = [
            {"task_id": "HumanEval/0", "generated_text": "a", "token_ids": [1], "solution": "a"},
            {"task_id": "HumanEval/1", "generated_text": "different", "token_ids": [3], "solution": "c"},
        ]
        report = compare_generation_receipts(
            self._receipt("reference", reference_rows),
            reference_rows,
            self._receipt("candidate", candidate_rows),
            candidate_rows,
            {"rows": [{"task_id": "HumanEval/0", "passed": True}, {"task_id": "HumanEval/1", "passed": False}]},
            {"rows": [{"task_id": "HumanEval/0", "passed": True}, {"task_id": "HumanEval/1", "passed": True}]},
        )
        self.assertEqual(report["exact_generated_text_agreement_count"], 1)
        self.assertEqual(report["exact_token_agreement_count"], 1)
        self.assertEqual(report["exact_solution_agreement_count"], 1)
        self.assertEqual(report["pass_fail_counts"]["reference"]["passed"], 1)
        self.assertEqual(report["pass_fail_counts"]["reference"]["failed"], 1)
        self.assertEqual(report["pass_fail_counts"]["candidate"]["passed"], 2)
        self.assertEqual(report["paired_quality"]["paired_losses"], [])
        self.assertEqual(report["paired_quality"]["paired_gains"], ["HumanEval/1"])
        self.assertTrue(report["paired_quality"]["passes_no_observed_loss_gate"])

    def test_aggregate_tie_does_not_hide_paired_loss(self):
        rows = [
            {"task_id": "HumanEval/0", "generated_text": "a"},
            {"task_id": "HumanEval/1", "generated_text": "b"},
        ]
        report = compare_generation_receipts(
            self._receipt("reference", rows),
            rows,
            self._receipt("candidate", rows),
            rows,
            {
                "rows": [
                    {"task_id": "HumanEval/0", "passed": True},
                    {"task_id": "HumanEval/1", "passed": False},
                ]
            },
            {
                "rows": [
                    {"task_id": "HumanEval/0", "passed": False},
                    {"task_id": "HumanEval/1", "passed": True},
                ]
            },
        )
        self.assertEqual(report["paired_quality"]["aggregate_delta"], 0)
        self.assertEqual(report["paired_quality"]["paired_losses"], ["HumanEval/0"])
        self.assertEqual(report["paired_quality"]["paired_gains"], ["HumanEval/1"])
        self.assertFalse(report["paired_quality"]["passes_no_observed_loss_gate"])

    def test_score_counts_supports_base_and_plus_rows(self):
        counts = score_counts(
            {
                "results": [
                    {"task_id": "HumanEval/0", "base": True, "plus": False},
                    {"task_id": "HumanEval/1", "base": True, "plus": True},
                ]
            },
            SMOKE_TASK_IDS[:2],
        )
        self.assertEqual(counts["passed"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["by_variant"]["base"]["passed"], 2)
        self.assertEqual(counts["by_variant"]["plus"]["failed"], 1)


class EvalPlusSubsetInputTests(unittest.TestCase):
    def test_score_rejects_incomplete_subset_before_importing_evalplus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            samples.write_text(
                json.dumps({"task_id": "HumanEval/0", "solution": "def f(): pass"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing frozen tasks"):
                score_evalplus_subset(samples, root / "scores.json")


if __name__ == "__main__":
    unittest.main()
