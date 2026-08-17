import unittest
from pathlib import Path

from src.release_intelligence_suite import (
    COMPARISON_SCHEMA,
    EXPECTED_DOMAINS,
    RUN_SCHEMA,
    compare_payloads,
    load_spec,
    paired_direction_p_value,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "benchmarks/specs/release-intelligence-mini-v1.json"


def fake_domain_run(family, domain_name, domain, failed=None, output_offset=0):
    failed = failed or set()
    samples = {}
    for task_name in domain["task_names"]:
        rows = []
        for index in range(domain["limit_per_task"]):
            passed = (task_name, index) not in failed
            if domain["primary_metric"] == "acc":
                selected = (index + output_offset) % 4
                filtered_resps = [
                    [0.0 if choice == selected else -10.0, False]
                    for choice in range(4)
                ]
            else:
                filtered_resps = [index + output_offset]
            rows.append(
                {
                    "doc_id": index,
                    "doc": {"id": index},
                    "filter": domain["filter"],
                    domain["primary_metric"]: float(passed),
                    "filtered_resps": filtered_resps,
                }
            )
        samples[task_name] = rows
    return {
        "schema": RUN_SCHEMA,
        "family": family,
        "domain": domain_name,
        "lm_eval": {"samples": samples},
    }


class ReleaseIntelligenceSpecTests(unittest.TestCase):
    def test_frozen_spec_has_exactly_fifty_tasks_per_domain(self):
        spec = load_spec(SPEC_PATH)
        self.assertEqual(tuple(spec["domains"]), EXPECTED_DOMAINS)
        self.assertEqual(
            {name: domain["total_tasks"] for name, domain in spec["domains"].items()},
            {name: 50 for name in EXPECTED_DOMAINS},
        )

    def test_comparison_keeps_aggregate_score_and_paired_churn_separate(self):
        spec = load_spec(SPEC_PATH)
        reference_runs = {}
        candidate_runs = {}
        for domain_name, domain in spec["domains"].items():
            first = domain["task_names"][0]
            reference_runs[domain_name] = fake_domain_run(
                spec["families"][0], domain_name, domain, failed={(first, 1)}
            )
            candidate_runs[domain_name] = fake_domain_run(
                spec["families"][1],
                domain_name,
                domain,
                failed={(first, 0)},
                output_offset=1,
            )
        comparison = compare_payloads(spec, reference_runs, candidate_runs)
        self.assertEqual(comparison["schema"], COMPARISON_SCHEMA)
        self.assertEqual(comparison["reference_passed"], 196)
        self.assertEqual(comparison["candidate_passed"], 196)
        self.assertEqual(comparison["paired_loss_count"], 4)
        self.assertEqual(comparison["paired_gain_count"], 4)
        self.assertEqual(comparison["paired_direction_exact_p_value"], 1.0)
        self.assertEqual(
            comparison["classification"], "quality_churn_observed_on_frozen_subset"
        )
        self.assertEqual(comparison["output_agreement_count"], 0)

    def test_exact_paired_direction_test_keeps_small_churn_uncertain(self):
        self.assertAlmostEqual(paired_direction_p_value(12, 10), 0.8318119049)
        self.assertEqual(paired_direction_p_value(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
