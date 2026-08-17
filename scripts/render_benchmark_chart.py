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


def render_svg(decision: Mapping[str, Any]) -> str:
    performance = decision["performance"]
    quality = decision["quality"]
    mini = quality["release_intelligence_mini"]
    humaneval = quality["humanevalplus_full"]
    speed_rows = [
        (
            "BF16 reference",
            float(performance["baseline"]["decode_tokens_per_second_median"]),
            "bf16",
            "1.000×",
        ),
        (
            "Block FP8",
            float(performance["recommended"]["decode_tokens_per_second_median"]),
            "fp8",
            f"{float(performance['recommended']['decode_speedup']):.3f}×",
        ),
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
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">',
        '<title id="title">MiniCPM5-1B RTX 3060 speed and quality benchmark</title>',
        '<desc id="desc">Block FP8 decodes 1.503 times faster than BF16. HumanEval Plus gains one task, while a frozen 200-task cross-domain suite loses two tasks overall, with the largest drop on IFEval.</desc>',
        "<style>",
        ":root{color-scheme:light dark}",
        ".bg{fill:#ffffff}.panel{fill:#f8fafc;stroke:#d1d5db}.text{fill:#111827}.muted{fill:#4b5563}.grid{stroke:#d1d5db}.bf16{fill:#6b7280;stroke:#6b7280}.fp8{fill:#2563eb;stroke:#2563eb}.track{fill:#e5e7eb}.link{stroke:#9ca3af}",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-weight:400}.title{font-size:32px;font-weight:600}.subtitle{font-size:16px}.panel-title{font-size:20px;font-weight:600}.label{font-size:15px}.value{font-size:14px;font-weight:600}.note{font-size:14px}",
        "@media(prefers-color-scheme:dark){.bg{fill:#0b0f17}.panel{fill:#111827;stroke:#374151}.text{fill:#f3f4f6}.muted{fill:#cbd5e1}.grid{stroke:#374151}.bf16{fill:#9ca3af;stroke:#9ca3af}.fp8{fill:#60a5fa;stroke:#60a5fa}.track{fill:#273244}.link{stroke:#6b7280}}",
        "</style>",
        '<rect class="bg" width="1200" height="720"/>',
        '<text class="text title" x="50" y="58">MiniCPM5-1B on RTX 3060: speed vs quality</text>',
        '<text class="muted subtitle" x="50" y="88">One active request · batch 1 · vLLM 0.27.1 · deterministic no-thinking evaluation</text>',
        '<rect class="panel" x="50" y="120" width="500" height="430" rx="12"/>',
        '<rect class="panel" x="590" y="120" width="560" height="430" rx="12"/>',
        '<text class="text panel-title" x="78" y="162">Single-request decode throughput</text>',
        '<text class="muted note" x="78" y="187">Higher is faster · output tokens per second</text>',
    ]

    bar_start = 205.0
    bar_width = 300.0
    max_speed = max(value for _, value, _, _ in speed_rows) * 1.08
    for index, (label, value, css_class, ratio) in enumerate(speed_rows):
        y = 280 + index * 120
        width = value / max_speed * bar_width
        value_x = min(bar_start + width - 8.0, 448.0)
        parts.extend(
            [
                f'<text class="text label" x="78" y="{y + 6}">{html.escape(label)}</text>',
                f'<rect class="track" x="{bar_start:.0f}" y="{y - 17}" width="{bar_width:.0f}" height="30" rx="5"/>',
                f'<rect class="{css_class}" x="{bar_start:.0f}" y="{y - 17}" width="{width:.1f}" height="30" rx="5"/>',
                f'<text class="text value" x="{value_x:.1f}" y="{y + 4}" text-anchor="end">{value:.2f}</text>',
                f'<text class="muted value" x="515" y="{y + 4}" text-anchor="end">{ratio}</text>',
            ]
        )

    parts.extend(
        [
            '<text class="text panel-title" x="620" y="162">Matched benchmark scores</text>',
            '<text class="muted note" x="620" y="187">Percentage correct · same tasks and deterministic decoding</text>',
        ]
    )
    axis_start = 760.0
    axis_width = 335.0
    for tick in (0, 25, 50, 75, 100):
        x = axis_start + tick / 100.0 * axis_width
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="205" x2="{x:.1f}" y2="480"/>',
                f'<text class="muted note" x="{x:.1f}" y="505" text-anchor="middle">{tick}%</text>',
            ]
        )

    for index, (label, reference, candidate, total) in enumerate(quality_rows):
        y = 230 + index * 54
        ref_score = _score(reference, total)
        cand_score = _score(candidate, total)
        ref_x = axis_start + ref_score / 100.0 * axis_width
        cand_x = axis_start + cand_score / 100.0 * axis_width
        left_x, right_x = sorted((ref_x, cand_x))
        parts.extend(
            [
                f'<text class="text label" x="620" y="{y + 5}">{html.escape(label)}</text>',
                f'<line class="link" x1="{left_x:.1f}" y1="{y}" x2="{right_x:.1f}" y2="{y}" stroke-width="3"/>',
                f'<circle class="bf16" cx="{ref_x:.1f}" cy="{y}" r="7"/>',
                f'<circle class="fp8" cx="{cand_x:.1f}" cy="{y}" r="7"/>',
                f'<text class="text value" x="1122" y="{y + 5}" text-anchor="end">{reference}/{total} → {candidate}/{total}</text>',
            ]
        )

    parts.extend(
        [
            '<circle class="bf16" cx="790" cy="532" r="6"/><text class="text note" x="804" y="537">BF16</text>',
            '<circle class="fp8" cx="875" cy="532" r="6"/><text class="text note" x="889" y="537">Block FP8</text>',
            '<text class="text value" x="50" y="595">FP8 result: 1.503× BF16 decode · HumanEval+ +1 task · cross-domain suite −2 tasks</text>',
            f'<text class="muted note" x="50" y="630">Cross-domain churn: {mini["paired_loss_count"]} paired losses, {mini["paired_gain_count"]} gains. Largest drop: IFEval 38/50 → 33/50.</text>',
            '<text class="muted note" x="50" y="666">Scope: pinned MiniCPM5-1B checkpoint and one RTX 3060. The four 50-task slices are release screens, not full official leaderboard reproductions.</text>',
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
