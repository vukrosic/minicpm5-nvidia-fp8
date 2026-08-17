import tempfile
import unittest
from pathlib import Path

from src.vllm_family_evaluator import (
    AUTO_REPORT_END,
    AUTO_REPORT_START,
    DECISION_SCHEMA,
    EXPECTED_CUSTOM_DOMAINS,
    SCORED_SCHEMA,
    TOTAL_TASK_COUNT,
    build_decision,
    compare_scored_families,
    extract_choice_v1,
    extract_choice_v2,
    extract_choice_prefill_v1,
    load_custom_manifest,
    render_decision_markdown,
    score_custom_output,
    update_report_block,
)


ROOT = Path(__file__).resolve().parents[1]


class FrozenManifestTests(unittest.TestCase):
    def test_manifest_has_six_tasks_in_each_declared_domain(self):
        rows = load_custom_manifest(ROOT / "benchmarks/quality/canary-v2.jsonl")
        self.assertEqual(len(rows), 30)
        self.assertEqual(
            {
                domain: sum(row["domain"] == domain for row in rows)
                for domain in EXPECTED_CUSTOM_DOMAINS
            },
            {domain: 6 for domain in EXPECTED_CUSTOM_DOMAINS},
        )

    def test_manifest_rejects_duplicate_ids_before_scoring(self):
        source = (ROOT / "benchmarks/quality/canary-v2.jsonl").read_text(
            encoding="utf-8"
        )
        first = source.splitlines()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(source + first + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate custom task id"):
                load_custom_manifest(path)


class FrozenScorerTests(unittest.TestCase):
    def test_choice_parser_is_conservative_but_accepts_labeled_final_answer(self):
        self.assertEqual(extract_choice_v1("B"), "B")
        self.assertEqual(extract_choice_v1("<think>x</think>\n答案：C"), "C")
        self.assertIsNone(extract_choice_v1("B because it is correct"))

    def test_choice_v2_accepts_explicit_answer_after_reasoning_only(self):
        self.assertEqual(
            extract_choice_v2("I computed the values. The correct answer is **B**."),
            "B",
        )
        self.assertEqual(extract_choice_v2("分析完成。正确答案是 A. 机器学习。"), "A")
        self.assertIsNone(extract_choice_v2("The values mention A and B."))

    def test_prefill_choice_reads_only_the_first_option(self):
        self.assertEqual(extract_choice_prefill_v1(" B. [5, 3, 1]"), "B")
        self.assertEqual(extract_choice_prefill_v1("**C** because..."), "C")
        self.assertIsNone(extract_choice_prefill_v1("The answer is B"))

    def test_exact_text_preserves_internal_newlines(self):
        task = {
            "id": "instruction-x",
            "domain": "instruction_following",
            "validator": {
                "kind": "exact_text_v1",
                "expected": "red\nblue\ngreen",
            },
        }
        self.assertTrue(score_custom_output(task, " red\nblue\ngreen\n")["passed"])
        self.assertFalse(score_custom_output(task, "red blue green")["passed"])

    def test_json_equality_ignores_key_order_but_rejects_prose(self):
        task = {
            "id": "instruction-json",
            "domain": "instruction_following",
            "validator": {"kind": "json_equal_v1", "expected": {"a": 2, "b": 1}},
        }
        self.assertTrue(score_custom_output(task, '{"b":1,"a":2}')["passed"])
        self.assertFalse(
            score_custom_output(task, 'Here: {"a":2,"b":1}')["passed"]
        )


def fake_score(
    family: str,
    speed: float,
    *,
    failed: set[int] | None = None,
    token_offset: int = 0,
) -> dict:
    failed = failed or set()
    tasks = []
    for index in range(TOTAL_TASK_COUNT):
        tasks.append(
            {
                "task_id": f"task-{index:02d}",
                "domain": (
                    "code_generation"
                    if index >= 30
                    else EXPECTED_CUSTOM_DOMAINS[index // 6]
                ),
                "passed": index not in failed,
                "token_ids": [index + token_offset],
            }
        )
    trials = []
    for prompt in ("p0", "p1"):
        for repetition in range(3):
            trials.append(
                {
                    "prompt_id": prompt,
                    "repetition": repetition,
                    "decode_tokens_per_second": speed + repetition,
                }
            )
    return {
        "schema": SCORED_SCHEMA,
        "family": family,
        "quality": {
            "task_count": TOTAL_TASK_COUNT,
            "passed": TOTAL_TASK_COUNT - len(failed),
            "tasks": tasks,
        },
        "speed": {
            "summary": {
                "decode_tokens_per_second_median": speed,
                "end_to_end_output_tokens_per_second_median": speed - 10,
                "ttft_seconds_median": 0.03,
            },
            "trials": trials,
        },
    }


class PairedDecisionTests(unittest.TestCase):
    def test_comparison_records_task_losses_gains_and_tokens_separately(self):
        reference = fake_score("BF16", 100, failed={2})
        candidate = fake_score("candidate", 140, failed={1}, token_offset=1)
        comparison = compare_scored_families(reference, candidate)
        self.assertEqual(comparison["paired_losses"], ["task-01"])
        self.assertEqual(comparison["paired_gains"], ["task-02"])
        self.assertFalse(comparison["primary_eligible"])
        self.assertEqual(comparison["exact_token_agreement_count"], 0)

    def test_decision_chooses_fastest_zero_loss_and_keeps_raw_speed_boundary(self):
        baseline = fake_score("BF16", 100)
        fp8 = fake_score("block-FP8", 150, token_offset=1)
        regressed = fake_score("faster-regressed", 210, failed={3}, token_offset=2)
        decision = build_decision(baseline, [fp8, regressed])
        self.assertEqual(decision["schema"], DECISION_SCHEMA)
        self.assertEqual(decision["primary"]["family"], "block-FP8")
        self.assertEqual(
            decision["secondary_tracks"]["maximum_speed_ignoring_quality_loss"][
                "family"
            ],
            "faster-regressed",
        )
        self.assertEqual(
            decision["secondary_tracks"]["exact_token_vs_bf16"]["family"],
            "BF16",
        )


class ReportRendererTests(unittest.TestCase):
    def test_generated_block_replaces_only_the_marker_region(self):
        decision = build_decision(
            fake_score("BF16", 100),
            [fake_score("FP8", 150, token_offset=1)],
        )
        block = render_decision_markdown(decision, decision_path="decision.json")
        original = (
            f"before\n{AUTO_REPORT_START}\nold\n{AUTO_REPORT_END}\nafter\n"
        )
        updated = update_report_block(original, block)
        self.assertTrue(updated.startswith("before\n"))
        self.assertTrue(updated.endswith("\nafter\n"))
        self.assertIn("Primary observed leader: **FP8**", updated)
        self.assertNotIn("\nold\n", updated)


if __name__ == "__main__":
    unittest.main()
