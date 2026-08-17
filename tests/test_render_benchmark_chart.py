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
        self.assertIn("248.52", svg)
        self.assertIn("1.503×", svg)
        self.assertIn("95/164 → 96/164", svg)
        self.assertIn("38/50 → 33/50", svg)
        self.assertIn("not full official leaderboard reproductions", svg)


if __name__ == "__main__":
    unittest.main()
