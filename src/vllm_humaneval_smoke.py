#!/usr/bin/env python3
"""Run a small, relative vLLM HumanEval+ smoke test.

This module deliberately keeps the vLLM import inside the GPU-host execution
path.  The command never resolves a model name or a dataset name through a
hub: ``--model``, ``--tokenizer``, ``--config``, and ``--tasks`` must all point
to existing local paths.  ``run`` loads one model in one process; ``compare``
only reads two run receipts and optional local score receipts.

The output JSONL is intentionally compatible with the usual EvalPlus sample
shape (``task_id`` and ``solution``).  The accompanying receipt keeps the raw
generated text and token IDs so that exact agreement can be measured before
any later EvalPlus scoring.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "minicpm5-vllm-humaneval-relative-smoke-v1"
SCORE_SCHEMA = "minicpm5-vllm-humaneval-relative-comparison-v1"
EVALPLUS_SCORE_SCHEMA = "minicpm5-evalplus-subset-score-v1"

# Eight short, stable tasks keep the smoke bounded while covering several
# different function shapes.  Do not sort this tuple or derive it from the
# input manifest: run receipts must be comparable across model processes.
SMOKE_TASK_IDS = (
    "HumanEval/0",
    "HumanEval/1",
    "HumanEval/2",
    "HumanEval/3",
    "HumanEval/4",
    "HumanEval/5",
    "HumanEval/6",
    "HumanEval/7",
)
FULL_TASK_IDS = tuple(f"HumanEval/{index}" for index in range(164))
# Descriptive alias for callers that want to make the fixed-order contract
# explicit without depending on the smoke-specific name.
FIXED_TASK_IDS = SMOKE_TASK_IDS
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS_LIMIT = 2048
DEFAULT_MAX_MODEL_LEN = 4096
MAX_MODEL_LEN_LIMIT = 16384

_CONFIG_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
_WEIGHT_SUFFIXES = (
    ".bin",
    ".safetensors",
    ".gguf",
    ".pt",
    ".pth",
    ".ckpt",
)
_CONTROL_MARKERS = (
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|channel|>",
    "<|message|>",
    "<|analysis|>",
    "<|final|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "<|end|>",
    "<s>",
    "</s>",
)
_FENCE_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```[^\n`]*\n(.*)\Z", re.DOTALL)
_CODE_TAG_RE = re.compile(
    r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE
)
_CODE_TOKEN_RE = re.compile(
    r"<\|code_start\|>\s*(.*?)\s*<\|code_end\|>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
_ENTRYPOINT_PREFIX_RE = re.compile(
    r"\b(?:async\s+)?def\s+{name}\s*\(", re.IGNORECASE
)
_CODE_START_RE = re.compile(
    r"^\s*(?:"
    r"#!|"
    r"#\s*coding[:=]|"
    r"from\s+[A-Za-z_][\w.]*\s+import\b|"
    r"import\s+[A-Za-z_][\w.]*|"
    r"(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(|"
    r"class\s+[A-Za-z_]\w*\s*[:(]|"
    r"@[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    r")"
)
_TRAILING_PROSE_RE = re.compile(
    r"\n\s*(?:Here(?:'s| is)|This\s+(?:code|solution)|Explanation\s*:|"
    r"Hope\s+this|I\s+hope)\b.*\Z",
    re.DOTALL | re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without loading the whole file in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_local_path(value: str | Path, label: str, *, directory: bool | None = None) -> Path:
    """Resolve an existing local path and reject URI-like model references."""

    raw = str(value)
    if "://" in raw:
        raise ValueError(f"{label} must be a local path, not a URI: {raw}")
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"Missing local {label}: {path}")
    if directory is True and not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    if directory is False and not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path.resolve()


def _manifest_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        for key in ("tasks", "data", "problems"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [row for row in nested if isinstance(row, Mapping)]
            if isinstance(nested, Mapping):
                return [
                    dict(row, task_id=task_id)
                    if isinstance(row, Mapping)
                    else {"task_id": task_id, "value": row}
                    for task_id, row in nested.items()
                ]
        if "task_id" in value or "id" in value:
            return [value]
        if value and all(isinstance(row, Mapping) for row in value.values()):
            return [
                dict(row, task_id=task_id)
                if "task_id" not in row and "id" not in row
                else row
                for task_id, row in value.items()
            ]
    raise ValueError("HumanEval+ task manifest must be a JSON list, JSONL, or task mapping")


def load_task_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a local HumanEval+ JSON/JSONL manifest without a dataset fallback."""

    manifest_path = resolve_local_path(path, "task manifest", directory=False)
    text = manifest_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Task manifest is empty: {manifest_path}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {manifest_path}:{line_number}: {exc.msg}"
                ) from exc
        parsed = rows

    rows = _manifest_rows(parsed)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        task_id = row.get("task_id", row.get("id"))
        prompt = row.get("prompt")
        entry_point = row.get("entry_point")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every HumanEval+ row needs a non-empty task_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Task {task_id} has no non-empty prompt")
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError(f"Task {task_id} has no non-empty entry_point")
        if task_id in seen:
            raise ValueError(f"Duplicate task_id in manifest: {task_id}")
        seen.add(task_id)
        normalized.append(dict(row, task_id=task_id, entry_point=entry_point))
    if not normalized:
        raise ValueError(f"Task manifest has no task rows: {manifest_path}")
    return normalized


def select_fixed_tasks(
    rows: Iterable[Mapping[str, Any]],
    task_ids: Sequence[str] = SMOKE_TASK_IDS,
) -> list[dict[str, Any]]:
    """Select tasks in the declared order, independent of manifest order."""

    expected = tuple(task_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("The smoke task order must be non-empty and unique")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            if task_id in by_id:
                raise ValueError(f"Duplicate task_id in selected input: {task_id}")
            by_id[task_id] = row
    missing = [task_id for task_id in expected if task_id not in by_id]
    if missing:
        raise ValueError(
            "Local HumanEval+ manifest is missing fixed smoke tasks: "
            + ", ".join(missing)
        )
    return [dict(by_id[task_id]) for task_id in expected]


def task_ids_for_scope(scope: str) -> tuple[str, ...]:
    """Return the frozen task order for a declared evaluation scope."""

    if scope == "smoke":
        return SMOKE_TASK_IDS
    if scope == "full":
        return FULL_TASK_IDS
    raise ValueError(f"Unsupported HumanEval+ scope: {scope!r}")


def build_user_prompt(problem: Mapping[str, Any]) -> str:
    """Build the fixed code-generation instruction sent for each task."""

    task_id = str(problem["task_id"])
    prompt = str(problem["prompt"]).strip()
    return (
        "Complete this HumanEval+ Python problem. Return only the complete "
        "Python implementation needed to solve it. Do not explain the answer "
        "outside the code. Markdown code fences are tolerated but unnecessary.\n\n"
        f"Task: {task_id}\n"
        f"{prompt}\n"
    )


def _clean_generation_markers(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    for marker in _CONTROL_MARKERS:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"(?m)^\s*(?:assistant|model)\s*:?\s*$", "", cleaned)
    return cleaned.strip()


def _entrypoint_re(entry_point: str) -> re.Pattern[str]:
    return re.compile(
        _ENTRYPOINT_PREFIX_RE.pattern.format(name=re.escape(entry_point)),
        re.IGNORECASE,
    )


def _prepare_code(candidate: str, entry_point: str) -> str:
    code = _clean_generation_markers(candidate)
    if "```" in code:
        opening = re.match(r"^\s*```[^\n`]*\n", code)
        if opening:
            code = code[opening.end() :]
        code = code.split("```", 1)[0]
    code = re.sub(r"^\s*(?:python|python3|py)\s*\n", "", code, flags=re.IGNORECASE)
    for marker in ("<|code_end|>", "</code>", "</s>", "<|eot_id|>"):
        if marker in code:
            code = code.split(marker, 1)[0]
    code = _TRAILING_PROSE_RE.sub("", code).strip()

    entry_match = _entrypoint_re(entry_point).search(code)
    if entry_match:
        lines = code.splitlines()
        entry_line = code[: entry_match.start()].count("\n")
        starts = [
            index
            for index, line in enumerate(lines[: entry_line + 1])
            if _CODE_START_RE.match(line)
        ]
        if starts:
            code = "\n".join(lines[min(starts) :])
        else:
            code = "\n".join(lines[entry_line:])
    return code.strip()


def extract_code(text: str, entry_point: str) -> str:
    """Extract a Python solution from direct, fenced, or Think-mode output.

    MiniCPM generations may contain ``<think>...</think>`` followed by a
    fenced answer, while other vLLM/tokenizer combinations expose only the
    code tokens or add assistant/control markers.  The raw generation is
    retained separately; this helper only produces the EvalPlus ``solution``.
    """

    if not isinstance(text, str):
        raise TypeError("generated text must be a string")
    cleaned = _clean_generation_markers(text)
    if not cleaned:
        return ""

    structured_segments: list[str] = []
    think_matches = list(_THINK_END_RE.finditer(cleaned))
    if think_matches:
        structured_segments.append(cleaned[think_matches[-1].end() :])

    structured_segments.extend(match.group(1) for match in _FENCE_RE.finditer(cleaned))
    structured_segments.extend(match.group(1) for match in _CODE_TAG_RE.finditer(cleaned))
    structured_segments.extend(match.group(1) for match in _CODE_TOKEN_RE.finditer(cleaned))

    if not structured_segments and "```" in cleaned:
        opening = _OPEN_FENCE_RE.search(cleaned)
        if opening:
            structured_segments.append(opening.group(1))

    prepared = [_prepare_code(segment, entry_point) for segment in structured_segments]
    with_entrypoint = [candidate for candidate in prepared if _entrypoint_re(entry_point).search(candidate)]
    if with_entrypoint:
        return max(with_entrypoint, key=len)
    fallback = _prepare_code(cleaned, entry_point)
    if _entrypoint_re(entry_point).search(fallback):
        return fallback
    return max([*prepared, fallback], key=len) if prepared else fallback


def _json_file_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".json" and metadata["size_bytes"] <= 4 * 1024 * 1024:
        try:
            metadata["json"] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            metadata["json_error"] = str(exc)
    return metadata


def collect_local_path_metadata(path: str | Path) -> dict[str, Any]:
    """Collect small local-path metadata without hashing model weight files."""

    resolved = resolve_local_path(path, "metadata path")
    if resolved.is_file():
        return {"kind": "file", **_json_file_metadata(resolved)}

    entries = sorted(resolved.iterdir(), key=lambda item: item.name)
    files = [entry for entry in entries if entry.is_file()]
    metadata: dict[str, Any] = {
        "kind": "directory",
        "path": str(resolved),
        "file_count": len(files),
        "total_top_level_file_bytes": sum(item.stat().st_size for item in files),
        "top_level_files": [
            {"name": item.name, "size_bytes": item.stat().st_size}
            for item in files
        ],
        "config_files": {},
    }
    for name in _CONFIG_NAMES:
        candidate = resolved / name
        if candidate.is_file():
            metadata["config_files"][name] = _json_file_metadata(candidate)
    metadata["weight_files"] = [
        {"name": item.name, "size_bytes": item.stat().st_size}
        for item in files
        if item.name.lower().endswith(_WEIGHT_SUFFIXES)
    ]
    return metadata


def collect_model_metadata(
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the model/checkpoint/config portion of a run receipt."""

    model = resolve_local_path(model_path, "model")
    tokenizer = resolve_local_path(tokenizer_path or model, "tokenizer")
    metadata: dict[str, Any] = {
        "checkpoint": collect_local_path_metadata(model),
        "tokenizer": collect_local_path_metadata(tokenizer),
    }
    if config_path is not None:
        config = resolve_local_path(config_path, "model config")
        if config.is_dir():
            config_file = config / "config.json"
            if config_file.is_file():
                metadata["explicit_config"] = _json_file_metadata(config_file.resolve())
            else:
                metadata["explicit_config"] = collect_local_path_metadata(config)
        else:
            metadata["explicit_config"] = _json_file_metadata(config)
    else:
        default_config = model / "config.json" if model.is_dir() else None
        metadata["explicit_config"] = (
            _json_file_metadata(default_config.resolve())
            if default_config is not None and default_config.is_file()
            else None
        )
    return metadata


def _coerce_status(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "true", "success", "correct", "ok", "1"}:
            return True
        if normalized in {"fail", "failed", "false", "error", "incorrect", "0"}:
            return False
    return None


def _status_from_value(value: Any, variant: str | None = None) -> bool | None:
    if not isinstance(value, Mapping):
        return _coerce_status(value)
    if variant is not None:
        for key in _variant_keys(variant):
            if key in value:
                result = _status_from_value(value[key])
                if result is not None:
                    return result
    for key in (
        "passed",
        "pass",
        "success",
        "correct",
        "is_pass",
        "status",
        "result",
    ):
        if key in value:
            result = _status_from_value(value[key])
            if result is not None:
                return result
    return None


def _variant_keys(variant: str) -> tuple[str, ...]:
    if variant == "plus":
        return ("plus", "evalplus", "humaneval+", "humaneval_plus", "plus_passed")
    return ("base", "humaneval", "human_eval", "base_passed")


def _status_from_row(row: Any, variant: str | None = None) -> bool | None:
    if not isinstance(row, Mapping):
        return _coerce_status(row)
    if variant is not None:
        for key in _variant_keys(variant):
            if key in row:
                result = _status_from_value(row[key])
                if result is not None:
                    return result
    for key in (
        "passed",
        "pass",
        "success",
        "correct",
        "is_pass",
        "evalplus_passed",
        "humaneval_plus_passed",
        "status",
        "result",
    ):
        if key in row:
            result = _status_from_value(row[key])
            if result is not None:
                return result
    if variant is None:
        plus = _status_from_row(row, "plus")
        if plus is not None:
            return plus
        return _status_from_row(row, "base")
    return None


def _mapping_score_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, result in value.items():
        if not isinstance(task_id, str) or not task_id.startswith("HumanEval/"):
            return []
        if isinstance(result, Mapping):
            rows.append(dict(result, task_id=task_id))
        else:
            rows.append({"task_id": task_id, "passed": result})
    return rows


def _score_rows(payload: Any) -> tuple[list[Any], str]:
    if isinstance(payload, list):
        return payload, "list"
    if not isinstance(payload, Mapping):
        return [], "none"
    for key in ("rows", "task_results", "per_task", "per_task_results", "tasks", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value, key
        if isinstance(value, Mapping):
            mapped = _mapping_score_rows(value)
            if mapped:
                return mapped, key
    if "task_id" in payload or "id" in payload:
        return [payload], "single"
    mapped = _mapping_score_rows(payload)
    if mapped:
        return mapped, "task_mapping"
    return [], "none"


def _count_statuses(values: Iterable[bool | None]) -> dict[str, Any]:
    materialized = list(values)
    passed = sum(value is True for value in materialized)
    failed = sum(value is False for value in materialized)
    unknown = sum(value is None for value in materialized)
    return {
        "total": len(materialized),
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "available": passed + failed > 0,
    }


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _summary_counts(payload: Any) -> dict[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        candidates.append(payload)
        for key in ("summary", "counts", "score", "scores"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
    for candidate in candidates:
        passed = next(
            (
                _integer_value(candidate.get(key))
                for key in ("passed", "pass_count", "passed_count")
                if key in candidate and _integer_value(candidate.get(key)) is not None
            ),
            None,
        )
        failed = next(
            (
                _integer_value(candidate.get(key))
                for key in ("failed", "fail_count", "failed_count")
                if key in candidate and _integer_value(candidate.get(key)) is not None
            ),
            None,
        )
        total = next(
            (
                _integer_value(candidate.get(key))
                for key in ("total", "count", "example_count")
                if key in candidate and _integer_value(candidate.get(key)) is not None
            ),
            None,
        )
        if passed is None and failed is None:
            continue
        if passed is None:
            passed = 0
        if failed is None:
            failed = 0
        if total is None:
            total = passed + failed
        unknown = max(0, total - passed - failed)
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "available": True,
            "source": "summary",
        }
    return None


def score_counts(
    payload: Any,
    task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Extract pass/fail counts from common local EvalPlus receipt shapes."""

    rows, source = _score_rows(payload)
    expected = set(task_ids) if task_ids is not None else None
    filtered: list[Any] = []
    seen: set[str] = set()
    for row in rows:
        task_id = row.get("task_id", row.get("id")) if isinstance(row, Mapping) else None
        if expected is not None and isinstance(task_id, str) and task_id not in expected:
            continue
        if isinstance(task_id, str):
            if task_id in seen:
                continue
            seen.add(task_id)
        filtered.append(row)

    if filtered:
        overall = [_status_from_row(row) for row in filtered]
        result = _count_statuses(overall)
        result["source"] = source
        result["by_variant"] = {
            "base": _count_statuses(_status_from_row(row, "base") for row in filtered),
            "plus": _count_statuses(_status_from_row(row, "plus") for row in filtered),
        }
        return result

    summary = _summary_counts(payload)
    if summary is not None:
        return summary
    return {
        "available": False,
        "total": 0,
        "passed": 0,
        "failed": 0,
        "unknown": 0,
        "source": source,
        "reason": "No per-task pass/fail rows or integer pass/fail counts were found",
    }


def paired_score_transitions(
    reference_scores: Any,
    candidate_scores: Any,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    """Expose paired losses and gains instead of hiding them in aggregate scores."""

    def index(payload: Any) -> dict[str, bool | None]:
        rows, _ = _score_rows(payload)
        result: dict[str, bool | None] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            task_id = row.get("task_id", row.get("id"))
            if isinstance(task_id, str) and task_id not in result:
                result[task_id] = _status_from_row(row)
        return result

    reference = index(reference_scores)
    candidate = index(candidate_scores)
    ordered = list(task_ids)
    compared = [
        task_id
        for task_id in ordered
        if reference.get(task_id) is not None and candidate.get(task_id) is not None
    ]
    losses = [
        task_id
        for task_id in compared
        if reference[task_id] is True and candidate[task_id] is False
    ]
    gains = [
        task_id
        for task_id in compared
        if reference[task_id] is False and candidate[task_id] is True
    ]
    stable_passes = [
        task_id
        for task_id in compared
        if reference[task_id] is True and candidate[task_id] is True
    ]
    stable_failures = [
        task_id
        for task_id in compared
        if reference[task_id] is False and candidate[task_id] is False
    ]
    unknown = [task_id for task_id in ordered if task_id not in compared]
    return {
        "compared_count": len(compared),
        "reference_passed": sum(reference[task_id] is True for task_id in compared),
        "candidate_passed": sum(candidate[task_id] is True for task_id in compared),
        "aggregate_delta": (
            sum(candidate[task_id] is True for task_id in compared)
            - sum(reference[task_id] is True for task_id in compared)
        ),
        "paired_losses": losses,
        "paired_loss_count": len(losses),
        "paired_gains": gains,
        "paired_gain_count": len(gains),
        "stable_pass_count": len(stable_passes),
        "stable_failure_count": len(stable_failures),
        "unknown_task_ids": unknown,
        "zero_paired_losses": not losses and not unknown,
        "passes_no_observed_loss_gate": not losses and not unknown,
    }


def _generation_rows_from_payload(
    payload: Mapping[str, Any],
    receipt_parent: Path | None = None,
) -> list[dict[str, Any]]:
    for key in ("tasks", "generations", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]

    generated_code = payload.get("generated_code")
    output_path: str | None = None
    if isinstance(generated_code, Mapping):
        value = generated_code.get("path")
        if isinstance(value, str):
            output_path = value
    elif isinstance(generated_code, str):
        output_path = generated_code
    if output_path is None:
        value = payload.get("output")
        if isinstance(value, str):
            output_path = value
    if output_path is None:
        raise ValueError("Run receipt contains no task generations or generated_code path")
    output = Path(output_path).expanduser()
    if not output.is_absolute() and receipt_parent is not None:
        output = receipt_parent / output
    output = resolve_local_path(output, "generated code output", directory=False)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid generated code JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"Generated code row {line_number} is not an object")
        rows.append(dict(value))
    return rows


def load_json_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = resolve_local_path(path, "receipt", directory=False)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON receipt {receipt_path}: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Receipt must contain a JSON object: {receipt_path}")
    return dict(payload)


def load_score_receipt(path: str | Path) -> Any:
    """Load a JSON or JSONL score receipt without invoking an evaluator."""

    score_path = resolve_local_path(path, "score receipt", directory=False)
    text = score_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid score receipt JSONL at line {line_number}: {exc.msg}"
                ) from exc
        if not rows:
            raise ValueError(f"Score receipt is empty: {score_path}")
        return rows


def load_generation_receipt(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = resolve_local_path(path, "run receipt", directory=False)
    payload = load_json_receipt(receipt_path)
    rows = _generation_rows_from_payload(payload, receipt_path.parent)
    return payload, rows


def _row_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _index_generation_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    order: list[str] = []
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id", row.get("id"))
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every generated row needs a non-empty task_id")
        if task_id in indexed:
            raise ValueError(f"Duplicate generated task_id: {task_id}")
        order.append(task_id)
        indexed[task_id] = row
    return order, indexed


def compare_generation_receipts(
    reference_payload: Mapping[str, Any],
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_payload: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_scores: Any | None = None,
    candidate_scores: Any | None = None,
) -> dict[str, Any]:
    """Compare raw text/token outputs and optional later scoring receipts."""

    reference_order, reference = _index_generation_rows(reference_rows)
    candidate_order, candidate = _index_generation_rows(candidate_rows)
    common_ids = [task_id for task_id in reference_order if task_id in candidate]

    text_observed = 0
    text_equal = 0
    token_observed = 0
    token_equal = 0
    code_observed = 0
    code_equal = 0
    mismatches: list[dict[str, Any]] = []
    for task_id in common_ids:
        left = reference[task_id]
        right = candidate[task_id]
        left_text = _row_value(left, "generated_text", "raw_generation", "text")
        right_text = _row_value(right, "generated_text", "raw_generation", "text")
        left_tokens = _row_value(left, "token_ids", "generated_token_ids")
        right_tokens = _row_value(right, "token_ids", "generated_token_ids")
        left_code = _row_value(left, "solution", "code")
        right_code = _row_value(right, "solution", "code")

        text_match: bool | None = None
        if isinstance(left_text, str) and isinstance(right_text, str):
            text_observed += 1
            text_match = left_text == right_text
            text_equal += text_match
        token_match: bool | None = None
        if isinstance(left_tokens, (list, tuple)) and isinstance(right_tokens, (list, tuple)):
            token_observed += 1
            token_match = list(left_tokens) == list(right_tokens)
            token_equal += token_match
        code_match: bool | None = None
        if isinstance(left_code, str) and isinstance(right_code, str):
            code_observed += 1
            code_match = left_code == right_code
            code_equal += code_match
        if text_match is False or token_match is False or code_match is False:
            mismatches.append(
                {
                    "task_id": task_id,
                    "generated_text_equal": text_match,
                    "token_equal": token_match,
                    "solution_equal": code_match,
                }
            )

    def rate(count: int, total: int) -> float | None:
        return count / total if total else None

    score_report = None
    paired_quality = None
    if reference_scores is not None or candidate_scores is not None:
        score_report = {
            "reference": (
                score_counts(reference_scores, reference_order)
                if reference_scores is not None
                else None
            ),
            "candidate": (
                score_counts(candidate_scores, candidate_order)
                if candidate_scores is not None
                else None
            ),
        }
        if reference_scores is not None and candidate_scores is not None:
            paired_quality = paired_score_transitions(
                reference_scores,
                candidate_scores,
                common_ids,
            )

    reference_model = reference_payload.get("model", {})
    candidate_model = candidate_payload.get("model", {})
    if isinstance(reference_model, Mapping):
        reference_label = reference_model.get("path", reference_payload.get("model_path"))
    else:
        reference_label = reference_payload.get("model_path")
    if isinstance(candidate_model, Mapping):
        candidate_label = candidate_model.get("path", candidate_payload.get("model_path"))
    else:
        candidate_label = candidate_payload.get("model_path")

    return {
        "schema": SCORE_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reference": reference_label,
        "candidate": candidate_label,
        "task_order_identical": reference_order == candidate_order,
        "reference_task_ids": reference_order,
        "candidate_task_ids": candidate_order,
        "common_task_count": len(common_ids),
        "missing_from_candidate": [task_id for task_id in reference_order if task_id not in candidate],
        "missing_from_reference": [task_id for task_id in candidate_order if task_id not in reference],
        "exact_generated_text_agreement_count": text_equal,
        "exact_generated_text_compared_count": text_observed,
        "exact_generated_text_agreement_rate": rate(text_equal, text_observed),
        "exact_token_agreement_count": token_equal,
        "exact_token_compared_count": token_observed,
        "exact_token_agreement_rate": rate(token_equal, token_observed),
        "exact_solution_agreement_count": code_equal,
        "exact_solution_compared_count": code_observed,
        "exact_solution_agreement_rate": rate(code_equal, code_observed),
        "mismatches": mismatches,
        "pass_fail_counts": score_report,
        "paired_quality": paired_quality,
        "claim_boundary": (
            "Full 164-task HumanEval+ paired comparison under the repository's "
            "deterministic generation protocol. Aggregate score, paired losses and "
            "gains, and exact output agreement are reported separately; this does not "
            "establish unchanged general intelligence."
            if tuple(common_ids) == FULL_TASK_IDS
            else "Relative smoke comparison only: fixed HumanEval+ subset, greedy "
            "generated-text/token agreement, and optional supplied score receipts. "
            "This is not an official leaderboard result or a general quality claim."
        ),
    }


def _render_prompt(
    tokenizer: Any,
    problem: Mapping[str, Any],
    *,
    enable_thinking: bool,
) -> tuple[str, bool]:
    user_prompt = build_user_prompt(problem)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        return user_prompt, False
    messages = [{"role": "user", "content": user_prompt}]
    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return str(rendered), True
    except (TypeError, ValueError, KeyError):
        rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
        return str(rendered), False


def _set_offline_environment() -> None:
    # These are process-local guards.  The model and task paths were already
    # checked before vLLM is imported, so a hub fallback cannot be intentional.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"


def _load_gpu_vllm() -> tuple[Any, Any, Any]:
    if platform.system() == "Darwin":
        raise RuntimeError("vLLM HumanEval smoke refuses to run on macOS")
    _set_offline_environment()
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - depends on the remote vLLM environment
        raise RuntimeError(f"vLLM GPU environment is unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM HumanEval smoke requires CUDA; no GPU is available")
    return torch, vllm, (LLM, SamplingParams)


def run_single_model(
    model_path: str | Path,
    task_manifest: str | Path,
    output_path: str | Path,
    receipt_path: str | Path | None = None,
    *,
    tokenizer_path: str | Path | None = None,
    config_path: str | Path | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.70,
    enforce_eager: bool = False,
    seed: int = 0,
    enable_thinking: bool = False,
    attention_backend: str = "triton_attn",
    performance_mode: str = "balanced",
    quantization: str = "none",
    task_ids: Sequence[str] = SMOKE_TASK_IDS,
) -> dict[str, Any]:
    """Run one local model.  This is intentionally not a two-model API."""

    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        raise ValueError(f"max_new_tokens must be in [1, {MAX_NEW_TOKENS_LIMIT}]")
    if not 1 <= max_model_len <= MAX_MODEL_LEN_LIMIT:
        raise ValueError(f"max_model_len must be in [1, {MAX_MODEL_LEN_LIMIT}]")
    if not 0.1 <= gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be in [0.1, 1.0)")

    model = resolve_local_path(model_path, "model")
    tokenizer = resolve_local_path(tokenizer_path or model, "tokenizer")
    manifest = resolve_local_path(task_manifest, "task manifest", directory=False)
    output = Path(output_path).expanduser().resolve()
    receipt = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else output.with_suffix(".receipt.json")
    )
    if output == receipt:
        raise ValueError("Generated code output and receipt must be different paths")
    if output.exists():
        raise ValueError(f"Refusing to overwrite generated code output: {output}")
    if receipt.exists():
        raise ValueError(f"Refusing to overwrite run receipt: {receipt}")

    frozen_task_ids = tuple(task_ids)
    tasks = select_fixed_tasks(load_task_manifest(manifest), frozen_task_ids)
    metadata = collect_model_metadata(model, tokenizer, config_path)
    if attention_backend not in {"default", "triton_attn", "flash_attn", "flashinfer"}:
        raise ValueError(f"Unsupported attention backend: {attention_backend}")
    if performance_mode not in {"balanced", "interactivity", "throughput"}:
        raise ValueError(f"Unsupported performance mode: {performance_mode}")
    if quantization not in {
        "none",
        "fp8_per_tensor",
        "fp8_per_block",
        "fp8_per_channel",
    }:
        raise ValueError(f"Unsupported online quantization: {quantization}")

    torch, vllm, runtime = _load_gpu_vllm()
    LLM, SamplingParams = runtime
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    backends = {
        "triton_attn": AttentionBackendEnum.TRITON_ATTN,
        "flash_attn": AttentionBackendEnum.FLASH_ATTN,
        "flashinfer": AttentionBackendEnum.FLASHINFER,
    }
    attention_config = (
        None
        if attention_backend == "default"
        else AttentionConfig(backend=backends[attention_backend])
    )

    started_at = datetime.now(timezone.utc)
    engine_started = time.perf_counter()
    engine_kwargs = dict(
        model=str(model),
        tokenizer=str(tokenizer),
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        seed=seed,
        attention_config=attention_config,
        performance_mode=performance_mode,
    )
    if quantization != "none":
        engine_kwargs["quantization"] = quantization
    llm = LLM(**engine_kwargs)
    engine_seconds = time.perf_counter() - engine_started
    tokenizer_object = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        n=1,
        min_tokens=1,
        max_tokens=max_new_tokens,
        seed=seed,
        skip_special_tokens=False,
    )

    generated_rows: list[dict[str, Any]] = []
    for problem in tasks:
        rendered_prompt, thinking_template_argument_used = _render_prompt(
            tokenizer_object,
            problem,
            enable_thinking=enable_thinking,
        )
        prompt_hash = sha256_text(rendered_prompt)
        generation_started = time.perf_counter()
        outputs = llm.generate([rendered_prompt], sampling, use_tqdm=False)
        elapsed = time.perf_counter() - generation_started
        if len(outputs) != 1 or not getattr(outputs[0], "outputs", None):
            raise RuntimeError(f"vLLM returned no completion for {problem['task_id']}")
        completion = outputs[0].outputs[0]
        generated_text = getattr(completion, "text", "")
        if not isinstance(generated_text, str):
            generated_text = str(generated_text)
        token_ids_value = getattr(completion, "token_ids", None)
        token_ids = (
            [int(token_id) for token_id in token_ids_value]
            if token_ids_value is not None
            else None
        )
        generated_rows.append(
            {
                "task_id": problem["task_id"],
                "prompt_sha256": prompt_hash,
                "thinking_enabled": enable_thinking,
                "thinking_template_argument_used": thinking_template_argument_used,
                "generated_text": generated_text,
                "token_ids": token_ids,
                "generated_token_count": len(token_ids) if token_ids is not None else None,
                "solution": extract_code(generated_text, str(problem["entry_point"])),
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "seconds": elapsed,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(
                {"task_id": row["task_id"], "solution": row["solution"]},
                ensure_ascii=False,
            )
            + "\n"
            for row in generated_rows
        ),
        encoding="utf-8",
    )
    completed_at = datetime.now(timezone.utc)
    receipt_payload: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": completed_at.strftime("%Y%m%dT%H%M%S.%fZ-vllm-humaneval"),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "mode": (
            "single-model-full-humanevalplus"
            if frozen_task_ids == FULL_TASK_IDS
            else "single-model-relative-smoke"
        ),
        "engine": "vllm",
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "model": metadata["checkpoint"],
        "model_config_metadata": metadata,
        "task_manifest": str(manifest),
        "task_manifest_sha256": sha256_file(manifest),
        "task_ids": list(frozen_task_ids),
        "task_count": len(generated_rows),
        "generation": {
            "greedy": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "quantization": quantization,
            "thinking_enabled": enable_thinking,
            "one_model_per_process": True,
        },
        "engine_settings": {
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "attention_backend": attention_backend,
            "performance_mode": performance_mode,
            "quantization": quantization,
            "v1_multiprocessing_env": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
            "engine_initialization_seconds": engine_seconds,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_execution": True,
            "offline_environment": True,
        },
        "generated_code": {
            "path": str(output),
            "format": "evalplus-samples-jsonl-v1",
            "sha256": sha256_file(output),
        },
        "tasks": generated_rows,
        "claim_boundary": (
            "Full 164-task HumanEval+ relative evaluation with one local model per "
            "process, the repository's fixed chat prompt, no thinking, and greedy "
            "generation. This is a complete executable-suite comparison, not a "
            "reproduction of a differently prompted public leaderboard score or a "
            "general model-quality claim."
            if frozen_task_ids == FULL_TASK_IDS
            else "Relative smoke only: fixed eight-task HumanEval+ subset, one local "
            "model per process, and greedy generation. Generated samples are saved "
            "for optional later EvalPlus scoring. This is not an official HumanEval+ "
            "benchmark, a full-suite score, or a general model-quality claim."
        ),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(receipt_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt_payload


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _score_one_evalplus_task(payload: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    """Run EvalPlus's sandboxed checker for one frozen HumanEval+ task."""

    problem, solution, expected = payload
    from evalplus.eval import PASS
    from evalplus.evaluate import check_correctness

    result = check_correctness(
        "humaneval",
        0,
        problem,
        solution,
        expected,
        base_only=False,
        fast_check=True,
        identifier=problem["task_id"],
    )
    base_status = result["base"][0]
    plus_status = result["plus"][0]
    return {
        "task_id": problem["task_id"],
        "base": base_status == PASS,
        "plus": plus_status == PASS,
        "passed": plus_status == PASS,
        "base_status": base_status,
        "plus_status": plus_status,
    }


def score_evalplus_subset(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    parallel: int = 4,
    task_ids: Sequence[str] = SMOKE_TASK_IDS,
) -> dict[str, Any]:
    """Score exactly the declared frozen subset with EvalPlus's checker."""

    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    samples = resolve_local_path(samples_path, "samples", directory=False)
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite score receipt: {output}")

    frozen_task_ids = tuple(task_ids)
    if not frozen_task_ids or len(set(frozen_task_ids)) != len(frozen_task_ids):
        raise ValueError("task_ids must be non-empty and unique")
    expected_task_ids = set(frozen_task_ids)
    rows: dict[str, str] = {}
    for line_number, line in enumerate(samples.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid samples JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"Samples row {line_number} is not an object")
        task_id = row.get("task_id")
        solution = row.get("solution")
        if not isinstance(task_id, str) or task_id not in expected_task_ids:
            raise ValueError(f"Unexpected task at line {line_number}: {task_id!r}")
        if task_id in rows:
            raise ValueError(f"Duplicate sample for {task_id}")
        if not isinstance(solution, str):
            raise ValueError(f"Sample for {task_id} has no string solution")
        rows[task_id] = solution
    missing = [task_id for task_id in frozen_task_ids if task_id not in rows]
    if missing:
        raise ValueError("Samples are missing frozen tasks: " + ", ".join(missing))

    try:
        import evalplus
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        from evalplus.evaluate import get_groundtruth
    except Exception as exc:
        raise RuntimeError(f"EvalPlus environment is unavailable: {exc}") from exc

    problems = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    expected = get_groundtruth(problems, dataset_hash, [])
    work = [
        (problems[task_id], rows[task_id], expected[task_id])
        for task_id in frozen_task_ids
    ]
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(parallel, len(work))
    ) as executor:
        unordered = list(executor.map(_score_one_evalplus_task, work))
    elapsed = time.perf_counter() - started
    by_id = {row["task_id"]: row for row in unordered}
    scored = [by_id[task_id] for task_id in frozen_task_ids]
    payload = {
        "schema": EVALPLUS_SCORE_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "HumanEval+",
        "evalplus_version": getattr(evalplus, "__version__", "unknown"),
        "dataset_hash": dataset_hash,
        "samples": str(samples),
        "samples_sha256": sha256_file(samples),
        "task_ids": list(frozen_task_ids),
        "rows": scored,
        "counts": {
            "base_passed": sum(row["base"] for row in scored),
            "plus_passed": sum(row["plus"] for row in scored),
            "total": len(scored),
        },
        "parallel": min(parallel, len(work)),
        "elapsed_seconds": elapsed,
        "claim_boundary": (
            "EvalPlus executable checks on all 164 frozen HumanEval+ tasks. The "
            "checker is complete, while the surrounding generation configuration "
            "remains this repository's relative BF16-versus-candidate protocol."
            if frozen_task_ids == FULL_TASK_IDS
            else "Official EvalPlus executable checks on the fixed eight-task smoke "
            "subset only; this is not a full or official leaderboard score."
        ),
    }
    _write_json_once(output, payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="benchmark exactly one local model")
    run.add_argument("--model", type=Path, required=True, help="local checkpoint directory")
    run.add_argument(
        "--tokenizer",
        type=Path,
        help="local tokenizer directory; defaults to --model",
    )
    run.add_argument(
        "--config",
        "--model-config",
        dest="config",
        type=Path,
        help="optional local config file/directory to record in the receipt",
    )
    run.add_argument(
        "--tasks",
        "--task-manifest",
        dest="tasks",
        type=Path,
        required=True,
        help="local HumanEval+ JSON or JSONL manifest; no dataset downloader is used",
    )
    run.add_argument("--output", type=Path, required=True, help="EvalPlus-compatible samples JSONL")
    run.add_argument("--receipt", type=Path, help="run receipt; defaults beside --output")
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    run.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    run.add_argument("--enforce-eager", action="store_true")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable the model's long reasoning template; disabled for the fast smoke by default",
    )
    run.add_argument(
        "--attention-backend",
        choices=("default", "triton_attn", "flash_attn", "flashinfer"),
        default="triton_attn",
    )
    run.add_argument(
        "--performance-mode",
        choices=("balanced", "interactivity", "throughput"),
        default="balanced",
    )
    run.add_argument(
        "--quantization",
        choices=("none", "fp8_per_tensor", "fp8_per_block", "fp8_per_channel"),
        default="none",
        help="optional vLLM online weight quantization; checkpoint-declared quantization uses none",
    )
    run.add_argument(
        "--scope",
        choices=("smoke", "full"),
        default="smoke",
        help="fixed eight-task smoke or all 164 HumanEval+ tasks",
    )

    score = subparsers.add_parser(
        "score",
        help="score the fixed eight-task samples with a local EvalPlus installation",
    )
    score.add_argument("--samples", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--parallel", type=int, default=4)
    score.add_argument(
        "--scope",
        choices=("smoke", "full"),
        default="smoke",
        help="fixed eight-task smoke or all 164 HumanEval+ tasks",
    )

    compare = subparsers.add_parser(
        "compare",
        help="compare two completed run receipts without loading a model",
    )
    compare.add_argument("--reference", "--baseline", dest="reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--reference-score", "--baseline-score", dest="reference_score", type=Path)
    compare.add_argument("--candidate-score", type=Path)
    compare.add_argument("--output", type=Path, help="optional comparison JSON receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        if platform.system() == "Darwin":
            raise SystemExit("vLLM HumanEval smoke refuses to run on macOS; run it on the GPU host")
        try:
            receipt = run_single_model(
                args.model,
                args.tasks,
                args.output,
                args.receipt,
                tokenizer_path=args.tokenizer,
                config_path=args.config,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                seed=args.seed,
                enable_thinking=args.enable_thinking,
                attention_backend=args.attention_backend,
                performance_mode=args.performance_mode,
                quantization=args.quantization,
                task_ids=task_ids_for_scope(args.scope),
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        print(
            json.dumps(
                {
                    "receipt": receipt["generated_code"],
                    "run_id": receipt["run_id"],
                    "task_count": receipt["task_count"],
                    "claim_boundary": receipt["claim_boundary"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "score":
        try:
            score = score_evalplus_subset(
                args.samples,
                args.output,
                parallel=args.parallel,
                task_ids=task_ids_for_scope(args.scope),
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        print(
            json.dumps(
                {
                    "output": str(args.output.expanduser().resolve()),
                    "counts": score["counts"],
                    "elapsed_seconds": score["elapsed_seconds"],
                    "claim_boundary": score["claim_boundary"],
                },
                indent=2,
            )
        )
        return 0

    try:
        reference_payload, reference_rows = load_generation_receipt(args.reference)
        candidate_payload, candidate_rows = load_generation_receipt(args.candidate)
        reference_scores = load_score_receipt(args.reference_score) if args.reference_score else None
        candidate_scores = load_score_receipt(args.candidate_score) if args.candidate_score else None
        comparison = compare_generation_receipts(
            reference_payload,
            reference_rows,
            candidate_payload,
            candidate_rows,
            reference_scores,
            candidate_scores,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    if args.output is not None:
        try:
            _write_json_once(args.output.expanduser().resolve(), comparison)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
