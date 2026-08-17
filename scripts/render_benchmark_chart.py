#!/usr/bin/env python3
"""Render the public MiniCPM5 RTX 3060 benchmark summary as SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DECISION_SCHEMA = "minicpm5-rtx3060-release-summary-v1"
DOMAIN_ORDER = ("gsm8k", "mmlu", "ceval", "ifeval")
DOMAIN_LABELS = {
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
    "ceval": "C-Eval",
    "ifeval": "IFEval",
}


def load_decision(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read decision {resolved}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != DECISION_SCHEMA:
        raise ValueError(f"Decision must use {DECISION_SCHEMA}")
    quality = value.get("quality")
    if not isinstance(quality, Mapping):
        raise ValueError("Decision has no quality evidence")
    mini = quality.get("release_intelligence_mini")
    if not isinstance(mini, Mapping) or mini.get("task_count") != 200:
        raise ValueError("Decision has no complete 200-task release suite")
    domains = mini.get("domains")
    if not isinstance(domains, Mapping) or tuple(domains) != DOMAIN_ORDER:
        raise ValueError("Decision has incomplete release-suite domains")
    return dict(value)


def _score(passed: int, total: int) -> float:
    return 100.0 * passed / total


def _format_percent(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return str(rounded)
    return f"{value:.1f}"


def render_svg(decision: Mapping[str, Any]) -> str:
    performance = decision["performance"]
    quality = decision["quality"]
    mini = quality["release_intelligence_mini"]
    humaneval = quality["humanevalplus_full"]
    baseline_speed = float(
        performance["baseline"]["decode_tokens_per_second_median"]
    )
    candidate_speed = float(
        performance["recommended"]["decode_tokens_per_second_median"]
    )
    speedup = float(performance["recommended"]["decode_speedup"])
    speed_rows = [
        ("BF16", baseline_speed, "bf16"),
        ("FP8", candidate_speed, "fp8"),
    ]
    quality_rows = [
        (
            "HumanEval+",
            int(humaneval["reference_passed"]),
            int(humaneval["candidate_passed"]),
            164,
        )
    ]
    for name in DOMAIN_ORDER:
        domain = mini["domains"][name]
        quality_rows.append(
            (
                DOMAIN_LABELS[name],
                int(domain["reference_passed"]),
                int(domain["candidate_passed"]),
                int(domain["task_count"]),
            )
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650" role="img" aria-labelledby="title desc" data-chart="vertical-bars">',
        '<title id="title">MiniCPM5-1B RTX 3060 speed and quality benchmark</title>',
        (
            '<desc id="desc">BF16 and block FP8 compared with vertical bars. '
            f'FP8 decodes {speedup:.3f} times faster. HumanEval Plus scores '
            f'{humaneval["reference_passed"]} versus {humaneval["candidate_passed"]} '
            f'of 164; the frozen cross-domain suite scores {mini["reference_passed"]} '
            f'versus {mini["candidate_passed"]} of 200. The four 50-task slices '
            'are release screens, not full official leaderboard reproductions.</desc>'
        ),
        "<style>",
        ".bg{fill:#fff}.text{fill:#111827}.muted{fill:#667085}.grid{stroke:#e7eaf0;stroke-width:1}.axis{stroke:#cfd4dc;stroke-width:1}.bf16{fill:#c8ced8}.fp8{fill:#2563eb}",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}.title{font-size:30px;font-weight:650}.subtitle{font-size:15px}.section{font-size:18px;font-weight:650}.unit{font-size:13px}.tick{font-size:12px}.category{font-size:13px;font-weight:500}.value{font-size:13px;font-weight:650}.summary{font-size:17px;font-weight:650}.footnote{font-size:12px}",
        "</style>",
        '<rect class="bg" width="1200" height="650"/>',
        '<text class="text title" x="52" y="50">MiniCPM5-1B · RTX 3060</text>',
        '<text class="muted subtitle" x="52" y="78">FP8 default vs BF16 reference · one active request · vLLM</text>',
        '<rect class="bf16" x="970" y="42" width="13" height="13" rx="2"/>',
        '<text class="muted subtitle" x="991" y="54">BF16</text>',
        '<rect class="fp8" x="1062" y="42" width="13" height="13" rx="2"/>',
        '<text class="muted subtitle" x="1083" y="54">FP8</text>',
        '<text class="text section" x="52" y="132">Decode speed</text>',
        f'<text class="fp8-label section" x="204" y="132" fill="#2563eb">{speedup:.2f}×</text>',
        '<text class="muted unit" x="52" y="154">tokens / second</text>',
        '<text class="text section" x="448" y="132">Benchmark accuracy</text>',
        '<text class="muted unit" x="448" y="154">correct (%)</text>',
    ]

    chart_top = 178.0
    chart_bottom = 520.0
    chart_height = chart_bottom - chart_top

    speed_left = 82.0
    speed_right = 390.0
    speed_max = 300.0
    for tick in (0, 100, 200, 300):
        y = chart_bottom - tick / speed_max * chart_height
        parts.extend(
            [
                f'<line class="grid" x1="{speed_left:.0f}" y1="{y:.1f}" x2="{speed_right:.0f}" y2="{y:.1f}"/>',
                f'<text class="muted tick" x="{speed_left - 12:.0f}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>',
            ]
        )
    parts.append(
        f'<line class="axis" x1="{speed_left:.0f}" y1="{chart_bottom:.0f}" x2="{speed_right:.0f}" y2="{chart_bottom:.0f}"/>'
    )
    speed_centers = (180.0, 300.0)
    speed_bar_width = 72.0
    for center, (label, value, css_class) in zip(speed_centers, speed_rows):
        bar_height = min(value / speed_max, 1.0) * chart_height
        y = chart_bottom - bar_height
        parts.extend(
            [
                f'<rect class="{css_class}" x="{center - speed_bar_width / 2:.1f}" y="{y:.1f}" width="{speed_bar_width:.0f}" height="{bar_height:.1f}" rx="4"/>',
                f'<text class="text value" x="{center:.0f}" y="{y - 11:.1f}" text-anchor="middle">{value:.0f}</text>',
                f'<text class="text category" x="{center:.0f}" y="548" text-anchor="middle">{html.escape(label)}</text>',
            ]
        )

    quality_left = 478.0
    quality_right = 1148.0
    for tick in (0, 25, 50, 75, 100):
        y = chart_bottom - tick / 100.0 * chart_height
        parts.extend(
            [
                f'<line class="grid" x1="{quality_left:.0f}" y1="{y:.1f}" x2="{quality_right:.0f}" y2="{y:.1f}"/>',
                f'<text class="muted tick" x="{quality_left - 12:.0f}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>',
            ]
        )
    parts.append(
        f'<line class="axis" x1="{quality_left:.0f}" y1="{chart_bottom:.0f}" x2="{quality_right:.0f}" y2="{chart_bottom:.0f}"/>'
    )

    group_width = (quality_right - quality_left) / len(quality_rows)
    quality_bar_width = 28.0
    for index, (label, reference, candidate, total) in enumerate(quality_rows):
        center = quality_left + group_width * (index + 0.5)
        scores = (
            (_score(reference, total), "bf16", center - 18.0),
            (_score(candidate, total), "fp8", center + 18.0),
        )
        for score, css_class, bar_center in scores:
            bar_height = score / 100.0 * chart_height
            y = chart_bottom - bar_height
            parts.extend(
                [
                    f'<rect class="{css_class}" x="{bar_center - quality_bar_width / 2:.1f}" y="{y:.1f}" width="{quality_bar_width:.0f}" height="{bar_height:.1f}" rx="3"/>',
                    f'<text class="text value" x="{bar_center:.1f}" y="{y - 8:.1f}" text-anchor="middle">{_format_percent(score)}</text>',
                ]
            )
        parts.append(
            f'<text class="text category" x="{center:.1f}" y="548" text-anchor="middle">{html.escape(label)}</text>'
        )

    parts.extend(
        [
            f'<text class="text summary" x="52" y="598">FP8 default: {speedup:.2f}× decode · {mini["candidate_passed"]}/200 vs {mini["reference_passed"]}/200 across four domain slices</text>',
            '<text class="muted footnote" x="52" y="625">Deterministic evaluation · HumanEval+ uses 164 tasks · other benchmarks use frozen 50-task slices</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        decision = load_decision(args.decision)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_svg(decision), encoding="utf-8")
        print(output)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
