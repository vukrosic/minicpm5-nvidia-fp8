import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.render_benchmark_chart import load_decision, render_svg


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkChartTests(unittest.TestCase):
    def test_public_chart_is_valid_svg_and_keeps_release_boundaries(self):
        decision = load_decision(ROOT / "results/release-summary.json")
        svg = render_svg(decision)
        ET.fromstring(svg)
        self.assertIn('data-chart="vertical-bars"', svg)
        self.assertIn("249", svg)
        self.assertIn("1.50×", svg)
        self.assertIn("FP8 default", svg)
        self.assertIn("58.5", svg)
        self.assertIn("66", svg)
        self.assertIn("87/200 vs 89/200", svg)
        self.assertIn("not full official leaderboard reproductions", svg)


if __name__ == "__main__":
    unittest.main()
