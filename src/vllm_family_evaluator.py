#!/usr/bin/env python3
"""Persistent single-request speed and paired-quality evaluator for vLLM.

The GPU ``run`` command loads one local checkpoint exactly once, then uses the
same resident vLLM engine for the frozen speed suite, thirty deterministic
project canaries, and eight HumanEval+ generations.  CPU-only commands score,
compare, select, and render those receipts without loading a model.

This harness deliberately has no hub fallback.  Model, tokenizer, prompt, and
HumanEval task inputs must already exist on the GPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Package import in tests.
    from .vllm_humaneval_smoke import (
        FIXED_TASK_IDS,
        build_user_prompt,
        collect_model_metadata,
        extract_code,
        load_task_manifest,
        score_evalplus_subset,
        select_fixed_tasks,
        sha256_file,
    )
except ImportError:  # Direct ``python src/...py`` execution on the GPU.
    from vllm_humaneval_smoke import (  # type: ignore
        FIXED_TASK_IDS,
        build_user_prompt,
        collect_model_metadata,
        extract_code,
        load_task_manifest,
        score_evalplus_subset,
        select_fixed_tasks,
        sha256_file,
    )


RUN_SCHEMA = "minicpm5-vllm-family-run-v2"
SCORED_SCHEMA = "minicpm5-vllm-family-score-v2"
DECISION_SCHEMA = "minicpm5-vllm-family-decision-v2"
CUSTOM_MANIFEST_SCHEMA = "minicpm5-project-quality-canary-v2"
QUALITY_PROTOCOL = "quality-canary-v5-assistant-prefill-choice-v1"
EXPECTED_CUSTOM_DOMAINS = (
    "code_reasoning",
    "math",
    "knowledge",
    "chinese",
    "instruction_following",
)
CUSTOM_TASKS_PER_DOMAIN = 6
CUSTOM_TASK_COUNT = len(EXPECTED_CUSTOM_DOMAINS) * CUSTOM_TASKS_PER_DOMAIN
TOTAL_TASK_COUNT = CUSTOM_TASK_COUNT + len(FIXED_TASK_IDS)
VALIDATOR_KINDS = {"choice_v1", "exact_text_v1", "json_equal_v1"}
ONLINE_QUANTIZATION = {
    "none",
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
}
AUTO_REPORT_START = "<!-- AUTO-DECISION:START -->"
AUTO_REPORT_END = "<!-- AUTO-DECISION:END -->"
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_digest(tokens: Sequence[int]) -> str:
    payload = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def median_or_none(values: Iterable[Any]) -> float | None:
    finite = [numeric for value in values if (numeric := finite_or_none(value)) is not None]
    return statistics.median(finite) if finite else None


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {resolved}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return dict(value)


def _write_json_once(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _load_jsonl(path: str | Path, label: str) -> tuple[Path, list[dict[str, Any]]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Missing local {label}: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL at {resolved}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} row {line_number} is not an object")
        rows.append(dict(value))
    if not rows:
        raise ValueError(f"{label} is empty: {resolved}")
    return resolved, rows


def load_custom_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load and enforce the frozen 30-task custom-canary contract."""

    _, rows = _load_jsonl(path, "custom quality manifest")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        task_id = row.get("id")
        domain = row.get("domain")
        prompt = row.get("prompt")
        validator = row.get("validator")
        max_new_tokens = row.get("max_new_tokens")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every custom task needs a non-empty id")
        if task_id in seen:
            raise ValueError(f"Duplicate custom task id: {task_id}")
        seen.add(task_id)
        if domain not in EXPECTED_CUSTOM_DOMAINS:
            raise ValueError(f"Task {task_id} has unsupported domain: {domain!r}")
        counts[str(domain)] += 1
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Task {task_id} has no prompt")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise ValueError(f"Task {task_id} max_new_tokens must be an integer")
        if not 1 <= max_new_tokens <= 64:
            raise ValueError(f"Task {task_id} max_new_tokens must be in [1, 64]")
        if not isinstance(validator, Mapping):
            raise ValueError(f"Task {task_id} has no validator object")
        kind = validator.get("kind")
        if kind not in VALIDATOR_KINDS:
            raise ValueError(f"Task {task_id} has unsupported validator: {kind!r}")
        expected = validator.get("expected")
        if kind == "choice_v1" and expected not in {"A", "B", "C", "D"}:
            raise ValueError(f"Task {task_id} choice expected must be A, B, C, or D")
        if kind == "exact_text_v1" and not isinstance(expected, str):
            raise ValueError(f"Task {task_id} exact expected must be text")
        if kind == "json_equal_v1" and not isinstance(expected, (dict, list)):
            raise ValueError(f"Task {task_id} JSON expected must be an object or list")

    expected_counts = {
        domain: CUSTOM_TASKS_PER_DOMAIN for domain in EXPECTED_CUSTOM_DOMAINS
    }
    if len(rows) != CUSTOM_TASK_COUNT or dict(counts) != expected_counts:
        raise ValueError(
            f"Frozen custom manifest requires {expected_counts}; observed {dict(counts)}"
        )
    return rows


def load_speed_prompts(path: str | Path) -> list[dict[str, str]]:
    _, rows = _load_jsonl(path, "speed prompt manifest")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        prompt_id = row.get("id")
        text = row.get("text")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("Every speed prompt needs a non-empty id")
        if prompt_id in seen:
            raise ValueError(f"Duplicate speed prompt id: {prompt_id}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Speed prompt {prompt_id} has no text")
        seen.add(prompt_id)
        normalized.append({"id": prompt_id, "text": text})
    if len(normalized) != 6:
        raise ValueError(f"Frozen speed suite requires 6 prompts, got {len(normalized)}")
    return normalized


def visible_answer(text: str) -> str:
    """Remove transport markers while preserving answer formatting."""

    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    for marker in _CONTROL_MARKERS:
        value = value.replace(marker, "")
    think_end = list(re.finditer(r"</think>", value, flags=re.IGNORECASE))
    if think_end:
        value = value[think_end[-1].end() :]
    return unicodedata.normalize("NFKC", value).strip()


def extract_choice_v1(text: str) -> str | None:
    """Frozen conservative parser for a requested A/B/C/D-only response."""

    answer = visible_answer(text)
    direct = re.fullmatch(r"[`*_\s]*([A-D])(?:[.)])?[`*_\s]*", answer, re.IGNORECASE)
    if direct:
        return direct.group(1).upper()

    labeled: list[str] = []
    pattern = re.compile(
        r"^\s*(?:final\s+answer|answer|答案|选择)\s*(?:is|为|是)?\s*[:：]?\s*([A-D])(?:[.)])?\s*$",
        re.IGNORECASE,
    )
    for line in answer.splitlines():
        match = pattern.fullmatch(line)
        if match:
            labeled.append(match.group(1).upper())
    if labeled and len(set(labeled)) == 1:
        return labeled[0]

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if lines:
        final = re.fullmatch(r"([A-D])(?:[.)])?", lines[-1], re.IGNORECASE)
        if final:
            return final.group(1).upper()
    return None


def extract_choice_v2(text: str) -> str | None:
    """Parse an explicit answer label after allowing short free-form reasoning."""

    answer = visible_answer(text)
    direct = extract_choice_v1(answer)
    if direct is not None:
        return direct
    labeled_pattern = re.compile(
        r"(?:correct\s+answer|final\s+answer|answer|正确答案|答案|选择)"
        r"\s*(?:is|为|是)?\s*[:：]?\s*[`*_]*\s*([A-D])"
        r"(?=\s|[.)。`*_]|$)",
        re.IGNORECASE,
    )
    labeled = [match.group(1).upper() for match in labeled_pattern.finditer(answer)]
    if labeled:
        return labeled[-1]
    return None


def extract_choice_prefill_v1(text: str) -> str | None:
    """Read the first option after the prompt has already emitted ``Answer:``."""

    answer = visible_answer(text)
    match = re.match(r"^\s*[`*_]*\s*([A-D])(?=\s|[.)。`*_]|$)", answer, re.IGNORECASE)
    return match.group(1).upper() if match else None


def score_custom_output(
    task: Mapping[str, Any],
    generated_text: str,
    *,
    choice_parser: str = "v1",
) -> dict[str, Any]:
    validator = task["validator"]
    kind = str(validator["kind"])
    expected = validator["expected"]
    visible = visible_answer(generated_text)
    error: str | None = None
    if kind == "choice_v1":
        if choice_parser not in {"v1", "v2", "prefill_v1"}:
            raise ValueError(f"Unsupported choice parser: {choice_parser}")
        if choice_parser == "v2":
            observed: Any = extract_choice_v2(generated_text)
        elif choice_parser == "prefill_v1":
            observed = extract_choice_prefill_v1(generated_text)
        else:
            observed = extract_choice_v1(generated_text)
        passed = observed == expected
    elif kind == "exact_text_v1":
        observed = visible
        passed = observed == expected
    elif kind == "json_equal_v1":
        try:
            observed = json.loads(visible)
            passed = observed == expected
        except json.JSONDecodeError as exc:
            observed = None
            passed = False
            error = f"invalid_json:{exc.msg}"
    else:  # Manifest validation should make this unreachable.
        raise ValueError(f"Unsupported validator kind: {kind}")
    return {
        "task_id": task["id"],
        "domain": task["domain"],
        "validator": (
            f"choice_{choice_parser}"
            if kind == "choice_v1" and choice_parser != "v1"
            else kind
        ),
        "passed": bool(passed),
        "expected": expected,
        "observed": observed,
        "parse_error": error,
    }


def render_user_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    assistant_prefill: str | None = None,
) -> tuple[str, bool]:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        suffix = assistant_prefill or ""
        return user_prompt + ("\n" + suffix if suffix else ""), False
    messages = [{"role": "user", "content": user_prompt}]
    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return str(rendered) + (assistant_prefill or ""), True
    except (TypeError, ValueError, KeyError):
        rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
        return str(rendered) + (assistant_prefill or ""), False


def require_idle_gpu() -> None:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    competitors: list[str] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].isdigit() and int(fields[0]) != os.getpid():
            competitors.append(line)
    if competitors:
        raise RuntimeError("Competing GPU processes detected:\n" + "\n".join(competitors))


def _set_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _completion_payload(request_output: Any) -> tuple[Any, str, list[int]]:
    if not getattr(request_output, "outputs", None):
        raise RuntimeError("vLLM returned no completion")
    completion = request_output.outputs[0]
    text = getattr(completion, "text", "")
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is None:
        raise RuntimeError("vLLM completion has no token IDs")
    return completion, str(text), [int(token_id) for token_id in token_ids]


def _run_speed_suite(
    llm: Any,
    SamplingParams: Any,
    prompts: Sequence[Mapping[str, str]],
    *,
    warmups_per_prompt: int,
    repetitions: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    sampling = SamplingParams(
        temperature=0.0,
        min_tokens=max_new_tokens,
        max_tokens=max_new_tokens,
        seed=0,
    )
    warmup_started = time.perf_counter()
    for row in prompts:
        for _ in range(warmups_per_prompt):
            llm.generate([row["text"]], sampling, use_tqdm=False)
    warmup_seconds = time.perf_counter() - warmup_started

    trials: list[dict[str, Any]] = []
    first_observed: dict[str, list[int]] = {}
    repetition_totals: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        offset = repetition % len(prompts)
        rotated = list(prompts[offset:]) + list(prompts[:offset])
        repetition_wall = 0.0
        for row in rotated:
            started = time.perf_counter()
            request_output = llm.generate([row["text"]], sampling, use_tqdm=False)[0]
            wall_seconds = time.perf_counter() - started
            repetition_wall += wall_seconds
            _, _, tokens = _completion_payload(request_output)
            expected = first_observed.setdefault(row["id"], tokens)
            metrics = getattr(request_output, "metrics", None)
            ttft_seconds = finite_or_none(
                metrics.first_token_latency if metrics is not None else None
            )
            decode_seconds = finite_or_none(
                metrics.last_token_ts - metrics.first_token_ts
                if metrics is not None and metrics.last_token_ts >= metrics.first_token_ts
                else None
            )
            decode_tokens_per_second = (
                (len(tokens) - 1) / decode_seconds
                if decode_seconds is not None and decode_seconds > 0
                else None
            )
            trials.append(
                {
                    "prompt_id": row["id"],
                    "repetition": repetition,
                    "prompt_tokens": len(request_output.prompt_token_ids or []),
                    "generated_token_ids": tokens,
                    "token_sha256": token_digest(tokens),
                    "matches_first_observed": tokens == expected,
                    "wall_seconds": wall_seconds,
                    "end_to_end_output_tokens_per_second": len(tokens) / wall_seconds,
                    "ttft_seconds": ttft_seconds,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": decode_tokens_per_second,
                    "num_cached_tokens": getattr(request_output, "num_cached_tokens", None),
                }
            )
        repetition_totals.append(
            {
                "repetition": repetition,
                "wall_seconds": repetition_wall,
                "generated_tokens": len(prompts) * max_new_tokens,
                "output_tokens_per_second": (
                    len(prompts) * max_new_tokens / repetition_wall
                ),
            }
        )

    decode = [row["decode_tokens_per_second"] for row in trials]
    e2e = [row["end_to_end_output_tokens_per_second"] for row in trials]
    ttft = [row["ttft_seconds"] for row in trials]
    walls = [row["wall_seconds"] for row in trials]
    decode_median = median_or_none(decode)
    if decode_median is None:
        raise RuntimeError("vLLM exposed no usable decode timing cells")
    return {
        "warmup_seconds": warmup_seconds,
        "warmups_per_prompt": warmups_per_prompt,
        "repetitions": repetitions,
        "max_new_tokens": max_new_tokens,
        "all_trials_repeatable": all(row["matches_first_observed"] for row in trials),
        "decode_metric_cells": sum(value is not None for value in decode),
        "ttft_metric_cells": sum(value is not None for value in ttft),
        "reference_tokens": first_observed,
        "summary": {
            "decode_tokens_per_second_median": decode_median,
            "decode_tokens_per_second_p10": percentile(
                [value for value in decode if value is not None], 0.10
            ),
            "decode_tokens_per_second_p90": percentile(
                [value for value in decode if value is not None], 0.90
            ),
            "end_to_end_output_tokens_per_second_median": median_or_none(e2e),
            "ttft_seconds_median": median_or_none(ttft),
            "wall_seconds_median": median_or_none(walls),
            "suite_output_tokens_per_second_median": statistics.median(
                row["output_tokens_per_second"] for row in repetition_totals
            ),
        },
        "repetition_totals": repetition_totals,
        "trials": trials,
    }


def _generate_quality_task(
    llm: Any,
    SamplingParams: Any,
    rendered_prompt: str,
    *,
    max_new_tokens: int,
    structured_outputs: Any | None = None,
) -> tuple[str, list[int], Any, float]:
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        n=1,
        min_tokens=1,
        max_tokens=max_new_tokens,
        seed=0,
        skip_special_tokens=False,
        structured_outputs=structured_outputs,
    )
    started = time.perf_counter()
    request_output = llm.generate([rendered_prompt], sampling, use_tqdm=False)[0]
    elapsed = time.perf_counter() - started
    completion, text, token_ids = _completion_payload(request_output)
    return text, token_ids, completion, elapsed


def run_family(
    *,
    family: str,
    model_path: str | Path,
    tokenizer_path: str | Path | None,
    custom_manifest_path: str | Path,
    humaneval_manifest_path: str | Path,
    speed_prompts_path: str | Path,
    output_dir: str | Path,
    quantization: str = "none",
    dtype: str = "bfloat16",
    max_model_len: int = 1024,
    gpu_memory_utilization: float = 0.70,
    speed_warmups: int = 1,
    speed_repetitions: int = 5,
    speed_max_new_tokens: int = 64,
    humaneval_max_new_tokens: int = 512,
    attention_backend: str = "triton_attn",
    performance_mode: str = "balanced",
) -> dict[str, Any]:
    """Load one family once and run all compatible speed/quality work."""

    if platform.system() == "Darwin":
        raise RuntimeError("Family evaluator is GPU-only and refuses to run on macOS")
    if not family.strip():
        raise ValueError("family must be non-empty")
    if quantization not in ONLINE_QUANTIZATION:
        raise ValueError(f"Unsupported online quantization: {quantization}")
    if speed_warmups < 1 or speed_repetitions < 2:
        raise ValueError("Require at least one warmup and two speed repetitions")
    if speed_max_new_tokens < 2 or humaneval_max_new_tokens < 1:
        raise ValueError("Invalid generation lengths")
    if max_model_len < humaneval_max_new_tokens:
        raise ValueError("max_model_len must exceed the HumanEval output allowance")

    model = Path(model_path).expanduser().resolve()
    tokenizer = Path(tokenizer_path or model).expanduser().resolve()
    if not model.is_dir() or not tokenizer.exists():
        raise ValueError("Model and tokenizer must be existing local paths")
    custom_manifest = Path(custom_manifest_path).expanduser().resolve()
    humaneval_manifest = Path(humaneval_manifest_path).expanduser().resolve()
    speed_manifest = Path(speed_prompts_path).expanduser().resolve()
    custom_tasks = load_custom_manifest(custom_manifest)
    humaneval_tasks = select_fixed_tasks(load_task_manifest(humaneval_manifest))
    speed_prompts = load_speed_prompts(speed_manifest)

    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"Refusing to reuse output directory: {output_root}")
    output_root.mkdir(parents=True)
    samples_path = output_root / "humaneval.samples.jsonl"
    run_path = output_root / "family-run.json"

    require_idle_gpu()
    _set_offline_environment()
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception as exc:  # pragma: no cover - GPU environment only.
        raise RuntimeError(f"vLLM GPU environment is unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available")

    backends = {
        "triton_attn": AttentionBackendEnum.TRITON_ATTN,
        "flash_attn": AttentionBackendEnum.FLASH_ATTN,
        "flashinfer": AttentionBackendEnum.FLASHINFER,
    }
    if attention_backend not in {"default", *backends}:
        raise ValueError(f"Unsupported attention backend: {attention_backend}")
    attention_config = (
        None
        if attention_backend == "default"
        else AttentionConfig(backend=backends[attention_backend])
    )

    model_metadata = collect_model_metadata(model, tokenizer)
    source = Path(__file__).resolve()
    started_at = utc_now()
    engine_kwargs: dict[str, Any] = {
        "model": str(model),
        "tokenizer": str(tokenizer),
        "trust_remote_code": True,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "seed": 0,
        "attention_config": attention_config,
        "performance_mode": performance_mode,
    }
    if quantization != "none":
        engine_kwargs["quantization"] = quantization

    engine_started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    engine_initialization_seconds = time.perf_counter() - engine_started
    tokenizer_object = llm.get_tokenizer()

    speed = _run_speed_suite(
        llm,
        SamplingParams,
        speed_prompts,
        warmups_per_prompt=speed_warmups,
        repetitions=speed_repetitions,
        max_new_tokens=speed_max_new_tokens,
    )

    custom_rows: list[dict[str, Any]] = []
    for task in custom_tasks:
        validator_kind = str(task["validator"]["kind"])
        assistant_prefill = "Answer:" if validator_kind == "choice_v1" else None
        rendered, thinking_argument_used = render_user_prompt(
            tokenizer_object,
            str(task["prompt"]),
            assistant_prefill=assistant_prefill,
        )
        effective_max_new_tokens = int(task["max_new_tokens"])
        generated_text, token_ids, completion, seconds = _generate_quality_task(
            llm,
            SamplingParams,
            rendered,
            max_new_tokens=effective_max_new_tokens,
        )
        custom_rows.append(
            {
                **score_custom_output(
                    task, generated_text, choice_parser="prefill_v1"
                ),
                "prompt_sha256": sha256_text(rendered),
                "thinking_enabled": False,
                "generation_constraint": "none",
                "assistant_prefill": assistant_prefill,
                "effective_max_new_tokens": effective_max_new_tokens,
                "thinking_template_argument_used": thinking_argument_used,
                "generated_text": generated_text,
                "token_ids": token_ids,
                "token_sha256": token_digest(token_ids),
                "generated_token_count": len(token_ids),
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "seconds": seconds,
            }
        )

    humaneval_rows: list[dict[str, Any]] = []
    for problem in humaneval_tasks:
        rendered, thinking_argument_used = render_user_prompt(
            tokenizer_object, build_user_prompt(problem)
        )
        generated_text, token_ids, completion, seconds = _generate_quality_task(
            llm,
            SamplingParams,
            rendered,
            max_new_tokens=humaneval_max_new_tokens,
        )
        humaneval_rows.append(
            {
                "task_id": problem["task_id"],
                "domain": "code_generation",
                "prompt_sha256": sha256_text(rendered),
                "thinking_enabled": False,
                "thinking_template_argument_used": thinking_argument_used,
                "generated_text": generated_text,
                "token_ids": token_ids,
                "token_sha256": token_digest(token_ids),
                "generated_token_count": len(token_ids),
                "solution": extract_code(generated_text, str(problem["entry_point"])),
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "seconds": seconds,
            }
        )

    samples_path.write_text(
        "".join(
            json.dumps(
                {"task_id": row["task_id"], "solution": row["solution"]},
                ensure_ascii=False,
            )
            + "\n"
            for row in humaneval_rows
        ),
        encoding="utf-8",
    )
    by_domain: dict[str, list[bool]] = defaultdict(list)
    for row in custom_rows:
        by_domain[row["domain"]].append(bool(row["passed"]))

    completed_at = utc_now()
    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "run_id": datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ-vllm-family"
        ),
        "started_at": started_at,
        "completed_at": completed_at,
        "family": family,
        "quality_protocol": QUALITY_PROTOCOL,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "model": model_metadata,
        "engine": {
            "name": "vllm",
            "version": getattr(vllm, "__version__", "unknown"),
            "initialization_seconds": engine_initialization_seconds,
            "dtype": dtype,
            "online_quantization": quantization,
            "checkpoint_declared_quantization": quantization == "none",
            "max_model_len": max_model_len,
            "max_num_seqs": 1,
            "concurrent_requests": 1,
            "prefix_caching_enabled": False,
            "log_stats_enabled": True,
            "attention_backend": attention_backend,
            "performance_mode": performance_mode,
            "gpu_memory_utilization": gpu_memory_utilization,
            "v1_multiprocessing_env": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
            "one_engine_load_for_speed_and_quality": True,
        },
        "environment": {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_execution": True,
            "offline_environment": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "manifests": {
            "speed": {
                "path": str(speed_manifest),
                "sha256": sha256_file(speed_manifest),
                "count": len(speed_prompts),
            },
            "custom_quality": {
                "schema": CUSTOM_MANIFEST_SCHEMA,
                "quality_protocol": QUALITY_PROTOCOL,
                "path": str(custom_manifest),
                "sha256": sha256_file(custom_manifest),
                "count": len(custom_rows),
            },
            "humaneval": {
                "path": str(humaneval_manifest),
                "sha256": sha256_file(humaneval_manifest),
                "task_ids": list(FIXED_TASK_IDS),
                "count": len(humaneval_rows),
            },
        },
        "speed": speed,
        "custom_quality": {
            "summary": {
                "task_count": len(custom_rows),
                "passed": sum(row["passed"] for row in custom_rows),
                "by_domain": {
                    domain: {
                        "count": len(values),
                        "passed": sum(values),
                    }
                    for domain, values in sorted(by_domain.items())
                },
            },
            "tasks": custom_rows,
        },
        "humaneval": {
            "max_new_tokens": humaneval_max_new_tokens,
            "samples": {
                "path": str(samples_path),
                "sha256": sha256_file(samples_path),
                "format": "evalplus-samples-jsonl-v1",
            },
            "tasks": humaneval_rows,
        },
        "claim_boundary": (
            "One RTX 3060, one resident vLLM engine, batch/concurrency one, "
            "six-prompt fixed-length speed suite, thirty deterministic project "
            "canaries, and eight generated HumanEval+ samples. HumanEval task "
            "correctness is not available until the separate official checker runs."
        ),
    }
    _write_json_once(run_path, result)
    return result


def score_family_run(
    run_path: str | Path,
    output_path: str | Path,
    evalplus_output_path: str | Path,
    *,
    parallel: int = 4,
) -> dict[str, Any]:
    run_file = Path(run_path).expanduser().resolve()
    run = _load_json(run_file)
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError(f"Expected {RUN_SCHEMA}, got {run.get('schema')!r}")
    if run.get("quality_protocol") != QUALITY_PROTOCOL:
        raise ValueError(
            f"Run uses {run.get('quality_protocol')!r}, expected {QUALITY_PROTOCOL!r}"
        )

    custom_meta = run.get("manifests", {}).get("custom_quality", {})
    custom_path = Path(str(custom_meta.get("path", ""))).expanduser().resolve()
    if not custom_path.is_file() or sha256_file(custom_path) != custom_meta.get("sha256"):
        raise ValueError("Frozen custom manifest is missing or its hash changed")
    custom_tasks = load_custom_manifest(custom_path)
    custom_by_id = {str(row["id"]): row for row in custom_tasks}
    generated_custom = run.get("custom_quality", {}).get("tasks", [])
    if not isinstance(generated_custom, list):
        raise ValueError("Run has no custom-quality rows")
    generated_by_id = {
        str(row.get("task_id")): row
        for row in generated_custom
        if isinstance(row, Mapping)
    }
    if set(generated_by_id) != set(custom_by_id):
        raise ValueError("Run custom task IDs do not match the frozen manifest")
    custom_scored: list[dict[str, Any]] = []
    for task in custom_tasks:
        generated = generated_by_id[str(task["id"])]
        rescored = score_custom_output(
            task,
            str(generated.get("generated_text", "")),
            choice_parser="prefill_v1",
        )
        custom_scored.append(
            {
                **rescored,
                "generated_text": generated.get("generated_text"),
                "token_ids": generated.get("token_ids"),
                "token_sha256": generated.get("token_sha256"),
            }
        )

    samples_path = Path(
        str(run.get("humaneval", {}).get("samples", {}).get("path", ""))
    ).expanduser().resolve()
    if not samples_path.is_file():
        raise ValueError(f"Missing HumanEval samples: {samples_path}")
    if sha256_file(samples_path) != run["humaneval"]["samples"].get("sha256"):
        raise ValueError("HumanEval sample hash differs from the run receipt")
    evalplus = score_evalplus_subset(
        samples_path, evalplus_output_path, parallel=parallel
    )
    humaneval_generated = {
        str(row.get("task_id")): row
        for row in run.get("humaneval", {}).get("tasks", [])
        if isinstance(row, Mapping)
    }
    humaneval_scored: list[dict[str, Any]] = []
    for score_row in evalplus["rows"]:
        task_id = str(score_row["task_id"])
        generated = humaneval_generated.get(task_id, {})
        humaneval_scored.append(
            {
                "task_id": task_id,
                "domain": "code_generation",
                "validator": "evalplus_v0.3.1",
                "passed": bool(score_row["passed"]),
                "base_passed": bool(score_row["base"]),
                "plus_passed": bool(score_row["plus"]),
                "base_status": score_row["base_status"],
                "plus_status": score_row["plus_status"],
                "generated_text": generated.get("generated_text"),
                "solution": generated.get("solution"),
                "token_ids": generated.get("token_ids"),
                "token_sha256": generated.get("token_sha256"),
            }
        )

    rows = [*custom_scored, *humaneval_scored]
    if len(rows) != TOTAL_TASK_COUNT or any(
        not isinstance(row.get("passed"), bool) for row in rows
    ):
        raise ValueError(f"Expected {TOTAL_TASK_COUNT} fully scored tasks")
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(bool(row["passed"]))
    scored = {
        "schema": SCORED_SCHEMA,
        "recorded_at": utc_now(),
        "family": run["family"],
        "quality_protocol": run["quality_protocol"],
        "run": {"path": str(run_file), "sha256": sha256_file(run_file)},
        "model": run["model"],
        "engine": run["engine"],
        "speed": run["speed"],
        "quality": {
            "task_count": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "failed": sum(not row["passed"] for row in rows),
            "by_domain": {
                domain: {
                    "count": len(values),
                    "passed": sum(values),
                    "failed": sum(not value for value in values),
                }
                for domain, values in sorted(grouped.items())
            },
            "tasks": rows,
        },
        "evalplus": {
            "receipt": {
                "path": str(Path(evalplus_output_path).expanduser().resolve()),
                "sha256": sha256_file(
                    Path(evalplus_output_path).expanduser().resolve()
                ),
            },
            "dataset_hash": evalplus["dataset_hash"],
            "elapsed_seconds": evalplus["elapsed_seconds"],
        },
        "claim_boundary": (
            "Paired 38-task deterministic regression canary. Zero losses means "
            "no loss was observed on this frozen set; it does not prove unchanged "
            "general intelligence. Output/token agreement is recorded separately."
        ),
    }
    _write_json_once(output_path, scored)
    return scored


def _task_index(score: Mapping[str, Any]) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    rows = score.get("quality", {}).get("tasks", [])
    if not isinstance(rows, list):
        raise ValueError("Score has no quality task rows")
    order: list[str] = []
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Quality task row is not an object")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Quality task row has no task_id")
        if task_id in indexed:
            raise ValueError(f"Duplicate quality task: {task_id}")
        order.append(task_id)
        indexed[task_id] = row
    if len(order) != TOTAL_TASK_COUNT:
        raise ValueError(f"Expected {TOTAL_TASK_COUNT} scored tasks, got {len(order)}")
    return order, indexed


def _speed_trial_index(score: Mapping[str, Any]) -> dict[tuple[str, int], float]:
    indexed: dict[tuple[str, int], float] = {}
    for row in score.get("speed", {}).get("trials", []):
        key = (str(row["prompt_id"]), int(row["repetition"]))
        value = finite_or_none(row.get("decode_tokens_per_second"))
        if key in indexed:
            raise ValueError(f"Duplicate speed trial: {key}")
        if value is not None and value > 0:
            indexed[key] = value
    return indexed


def paired_speed_comparison(
    reference_score: Mapping[str, Any],
    candidate_score: Mapping[str, Any],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260817,
) -> dict[str, Any]:
    reference = _speed_trial_index(reference_score)
    candidate = _speed_trial_index(candidate_score)
    keys = sorted(set(reference) & set(candidate))
    if not keys or set(reference) != set(candidate):
        raise ValueError("Speed receipts do not have identical prompt/repetition cells")
    ratios = [candidate[key] / reference[key] for key in keys]
    rng = random.Random(seed)
    bootstrap: list[float] = []
    if bootstrap_samples > 0:
        for _ in range(bootstrap_samples):
            sample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
            bootstrap.append(statistics.median(sample))
    ci_low = percentile(bootstrap, 0.025) if bootstrap else math.nan
    ci_high = percentile(bootstrap, 0.975) if bootstrap else math.nan
    return {
        "aligned_trial_count": len(keys),
        "candidate_over_reference_median_ratio": statistics.median(ratios),
        "candidate_over_reference_ratio_p10": percentile(ratios, 0.10),
        "candidate_over_reference_ratio_p90": percentile(ratios, 0.90),
        "bootstrap_median_ratio_ci95": [ci_low, ci_high],
        "bootstrap_samples": bootstrap_samples,
        "direction": (
            "candidate_faster"
            if ci_low > 1.0
            else "candidate_slower"
            if ci_high < 1.0
            else "unresolved"
        ),
        "note": (
            "Aligned prompt/repetition bootstrap across separate resident-engine "
            "processes; compilation and model-load time are excluded from warm decode."
        ),
    }


def compare_scored_families(
    reference_score: Mapping[str, Any], candidate_score: Mapping[str, Any]
) -> dict[str, Any]:
    if reference_score.get("schema") != SCORED_SCHEMA:
        raise ValueError("Reference is not a v2 family score")
    if candidate_score.get("schema") != SCORED_SCHEMA:
        raise ValueError("Candidate is not a v2 family score")
    if reference_score.get("quality_protocol") != candidate_score.get("quality_protocol"):
        raise ValueError("Reference and candidate quality protocols differ")
    reference_order, reference = _task_index(reference_score)
    candidate_order, candidate = _task_index(candidate_score)
    if reference_order != candidate_order:
        raise ValueError("Reference and candidate task order differ")

    losses: list[str] = []
    gains: list[str] = []
    exact_output: list[str] = []
    deltas: list[dict[str, Any]] = []
    domain_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"losses": 0, "gains": 0, "same_pass": 0, "same_fail": 0}
    )
    for task_id in reference_order:
        ref = reference[task_id]
        cand = candidate[task_id]
        ref_pass = bool(ref["passed"])
        cand_pass = bool(cand["passed"])
        domain = str(ref["domain"])
        if domain != str(cand["domain"]):
            raise ValueError(f"Task domain changed for {task_id}")
        if ref_pass and not cand_pass:
            state = "loss"
            losses.append(task_id)
            domain_counts[domain]["losses"] += 1
        elif not ref_pass and cand_pass:
            state = "gain"
            gains.append(task_id)
            domain_counts[domain]["gains"] += 1
        elif ref_pass:
            state = "same_pass"
            domain_counts[domain]["same_pass"] += 1
        else:
            state = "same_fail"
            domain_counts[domain]["same_fail"] += 1
        ref_tokens = ref.get("token_ids")
        cand_tokens = cand.get("token_ids")
        tokens_equal = (
            isinstance(ref_tokens, list)
            and isinstance(cand_tokens, list)
            and ref_tokens == cand_tokens
        )
        if tokens_equal:
            exact_output.append(task_id)
        deltas.append(
            {
                "task_id": task_id,
                "domain": domain,
                "reference_passed": ref_pass,
                "candidate_passed": cand_pass,
                "state": state,
                "exact_token_ids": tokens_equal,
            }
        )

    candidate_passed = int(candidate_score["quality"]["passed"])
    return {
        "reference": reference_score["family"],
        "candidate": candidate_score["family"],
        "task_count": len(reference_order),
        "reference_passed": int(reference_score["quality"]["passed"]),
        "candidate_passed": candidate_passed,
        "paired_losses": losses,
        "paired_gains": gains,
        "zero_paired_losses": not losses,
        "primary_eligible": not losses,
        "exact_token_agreement_count": len(exact_output),
        "exact_token_agreement_rate": len(exact_output) / len(reference_order),
        "exact_token_task_ids": exact_output,
        "by_domain": dict(sorted(domain_counts.items())),
        "task_deltas": deltas,
        "speed": paired_speed_comparison(reference_score, candidate_score),
        "claim_boundary": (
            "Primary eligibility means zero BF16-passing tasks became failures "
            "on this frozen 38-task canary. It is not a general no-degradation proof."
        ),
    }


def build_decision(
    baseline_score: Mapping[str, Any], candidate_scores: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if baseline_score.get("schema") != SCORED_SCHEMA:
        raise ValueError("Baseline is not a v2 family score")
    all_scores = [baseline_score, *candidate_scores]
    families = [str(score.get("family", "")) for score in all_scores]
    if not all(families) or len(set(families)) != len(families):
        raise ValueError("Family labels must be non-empty and unique")

    comparisons = [
        compare_scored_families(baseline_score, candidate)
        for candidate in candidate_scores
    ]
    comparisons_by_family = {row["candidate"]: row for row in comparisons}
    family_rows: list[dict[str, Any]] = []
    for index, score in enumerate(all_scores):
        family = str(score["family"])
        comparison = comparisons_by_family.get(family)
        losses = comparison["paired_losses"] if comparison else []
        gains = comparison["paired_gains"] if comparison else []
        exact_count = (
            comparison["exact_token_agreement_count"]
            if comparison
            else TOTAL_TASK_COUNT
        )
        family_rows.append(
            {
                "family": family,
                "is_bf16_reference": index == 0,
                "decode_tokens_per_second_median": score["speed"]["summary"][
                    "decode_tokens_per_second_median"
                ],
                "end_to_end_output_tokens_per_second_median": score["speed"][
                    "summary"
                ]["end_to_end_output_tokens_per_second_median"],
                "ttft_seconds_median": score["speed"]["summary"][
                    "ttft_seconds_median"
                ],
                "quality_passed": score["quality"]["passed"],
                "quality_total": score["quality"]["task_count"],
                "paired_losses": losses,
                "paired_gains": gains,
                "primary_eligible": not losses,
                "exact_token_agreement_count_vs_bf16": exact_count,
                "score_receipt_sha256": score.get("_receipt_sha256"),
            }
        )

    eligible = [row for row in family_rows if row["primary_eligible"]]
    observed_primary = max(
        eligible, key=lambda row: float(row["decode_tokens_per_second_median"])
    )
    raw_speed = max(
        family_rows, key=lambda row: float(row["decode_tokens_per_second_median"])
    )
    exact_token = max(
        (
            row
            for row in family_rows
            if row["exact_token_agreement_count_vs_bf16"] == TOTAL_TASK_COUNT
        ),
        key=lambda row: float(row["decode_tokens_per_second_median"]),
    )

    runners_up = sorted(
        (
            row
            for row in eligible
            if row["family"] != observed_primary["family"]
        ),
        key=lambda row: float(row["decode_tokens_per_second_median"]),
        reverse=True,
    )
    primary_uncertainty: dict[str, Any] | None = None
    selection_status = "selected"
    if runners_up:
        winner_score = next(
            score for score in all_scores if score["family"] == observed_primary["family"]
        )
        runner_score = next(
            score for score in all_scores if score["family"] == runners_up[0]["family"]
        )
        primary_uncertainty = paired_speed_comparison(runner_score, winner_score)
        if primary_uncertainty["direction"] == "unresolved":
            selection_status = "observed_leader_speed_order_unresolved"

    return {
        "schema": DECISION_SCHEMA,
        "recorded_at": utc_now(),
        "objective": (
            "Fastest single request with zero observed quality loss versus BF16 "
            "on the frozen 38-task paired canary."
        ),
        "baseline_family": baseline_score["family"],
        "admission_rule": (
            "Zero paired task losses relative to BF16; no universal minimum "
            "speedup threshold is applied."
        ),
        "families": family_rows,
        "comparisons": comparisons,
        "primary": {
            "family": observed_primary["family"],
            "status": selection_status,
            "decode_tokens_per_second_median": observed_primary[
                "decode_tokens_per_second_median"
            ],
            "paired_losses": observed_primary["paired_losses"],
            "uncertainty_vs_next_eligible": primary_uncertainty,
        },
        "secondary_tracks": {
            "exact_token_vs_bf16": {
                "family": exact_token["family"],
                "decode_tokens_per_second_median": exact_token[
                    "decode_tokens_per_second_median"
                ],
            },
            "maximum_speed_ignoring_quality_loss": {
                "family": raw_speed["family"],
                "decode_tokens_per_second_median": raw_speed[
                    "decode_tokens_per_second_median"
                ],
                "paired_losses": raw_speed["paired_losses"],
            },
        },
        "claim_boundary": (
            "The primary winner is conditional on the named GPU, software, "
            "frozen speed workload, and 38-task canary. Zero observed loss is "
            "not proof of identical logits, tokens, or general intelligence."
        ),
    }


def load_scored_receipt(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    score = _load_json(resolved)
    if score.get("schema") != SCORED_SCHEMA:
        raise ValueError(f"Expected {SCORED_SCHEMA}: {resolved}")
    score["_receipt_path"] = str(resolved)
    score["_receipt_sha256"] = sha256_file(resolved)
    return score


def render_decision_markdown(
    decision: Mapping[str, Any], *, decision_path: str | Path | None = None
) -> str:
    if decision.get("schema") != DECISION_SCHEMA:
        raise ValueError("Cannot render a non-v2 decision receipt")
    primary = decision["primary"]
    exact = decision["secondary_tracks"]["exact_token_vs_bf16"]
    raw = decision["secondary_tracks"]["maximum_speed_ignoring_quality_loss"]
    lines = [
        AUTO_REPORT_START,
        "## Automated current decision",
        "",
        "This section is generated from structured family receipts; edit the JSON inputs or renderer, not this block.",
        "",
        f"Primary observed leader: **{primary['family']}** at **{float(primary['decode_tokens_per_second_median']):.2f} decode tok/s**, with {len(primary['paired_losses'])} paired losses on the frozen 38-task gate. Selection status: `{primary['status']}`.",
        "",
        "| Family | Decode tok/s | Canary | Paired losses vs BF16 | Primary eligible |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(
        decision["families"],
        key=lambda value: float(value["decode_tokens_per_second_median"]),
        reverse=True,
    ):
        lines.append(
            "| {family} | {decode:.2f} | {passed}/{total} | {losses} | {eligible} |".format(
                family=row["family"],
                decode=float(row["decode_tokens_per_second_median"]),
                passed=row["quality_passed"],
                total=row["quality_total"],
                losses=len(row["paired_losses"]),
                eligible="yes" if row["primary_eligible"] else "no",
            )
        )
    lines.extend(
        [
            "",
            f"Secondary exact-token-vs-BF16 leader: **{exact['family']}**. Maximum-speed boundary: **{raw['family']}** at **{float(raw['decode_tokens_per_second_median']):.2f} decode tok/s** with {len(raw['paired_losses'])} paired losses.",
            "",
            f"Generated at `{decision['recorded_at']}`. "
            + (
                f"Decision receipt: `{Path(decision_path).as_posix()}`."
                if decision_path is not None
                else ""
            ),
            "",
            decision["claim_boundary"],
            AUTO_REPORT_END,
        ]
    )
    return "\n".join(lines)


def update_report_block(report_text: str, generated_block: str) -> str:
    if report_text.count(AUTO_REPORT_START) != 1 or report_text.count(AUTO_REPORT_END) != 1:
        raise ValueError("Report must contain exactly one automated decision marker pair")
    start = report_text.index(AUTO_REPORT_START)
    end = report_text.index(AUTO_REPORT_END, start) + len(AUTO_REPORT_END)
    return report_text[:start] + generated_block + report_text[end:]


def render_report_file(decision_path: str | Path, report_path: str | Path) -> None:
    decision_file = Path(decision_path).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    decision = _load_json(decision_file)
    generated = render_decision_markdown(decision, decision_path=decision_file)
    current = report_file.read_text(encoding="utf-8")
    updated = update_report_block(current, generated)
    report_file.write_text(updated, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="load one family once and run speed plus quality")
    run.add_argument("--family", required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--tokenizer", type=Path)
    run.add_argument("--custom-manifest", type=Path, required=True)
    run.add_argument("--humaneval-tasks", type=Path, required=True)
    run.add_argument("--speed-prompts", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--quantization", choices=sorted(ONLINE_QUANTIZATION), default="none")
    run.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    run.add_argument("--max-model-len", type=int, default=1024)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    run.add_argument("--speed-warmups", type=int, default=1)
    run.add_argument("--speed-repetitions", type=int, default=5)
    run.add_argument("--speed-max-new-tokens", type=int, default=64)
    run.add_argument("--humaneval-max-new-tokens", type=int, default=512)
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

    score = commands.add_parser("score", help="score one completed family run")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--evalplus-output", type=Path, required=True)
    score.add_argument("--parallel", type=int, default=4)

    decide = commands.add_parser("decide", help="select the fastest zero-loss family")
    decide.add_argument("--baseline", type=Path, required=True)
    decide.add_argument("--candidate", type=Path, action="append", required=True)
    decide.add_argument("--output", type=Path, required=True)

    render = commands.add_parser("render-report", help="update the generated report block")
    render.add_argument("--decision", type=Path, required=True)
    render.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_family(
                family=args.family,
                model_path=args.model,
                tokenizer_path=args.tokenizer,
                custom_manifest_path=args.custom_manifest,
                humaneval_manifest_path=args.humaneval_tasks,
                speed_prompts_path=args.speed_prompts,
                output_dir=args.output_dir,
                quantization=args.quantization,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                speed_warmups=args.speed_warmups,
                speed_repetitions=args.speed_repetitions,
                speed_max_new_tokens=args.speed_max_new_tokens,
                humaneval_max_new_tokens=args.humaneval_max_new_tokens,
                attention_backend=args.attention_backend,
                performance_mode=args.performance_mode,
            )
            print(
                json.dumps(
                    {
                        "family": result["family"],
                        "decode_tokens_per_second_median": result["speed"]["summary"][
                            "decode_tokens_per_second_median"
                        ],
                        "custom_passed": result["custom_quality"]["summary"]["passed"],
                        "run_id": result["run_id"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "score":
            result = score_family_run(
                args.run,
                args.output,
                args.evalplus_output,
                parallel=args.parallel,
            )
            print(
                json.dumps(
                    {
                        "family": result["family"],
                        "passed": result["quality"]["passed"],
                        "total": result["quality"]["task_count"],
                        "output": str(args.output.resolve()),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "decide":
            baseline = load_scored_receipt(args.baseline)
            candidates = [load_scored_receipt(path) for path in args.candidate]
            decision = build_decision(baseline, candidates)
            output = _write_json_once(args.output, decision)
            print(
                json.dumps(
                    {
                        "primary": decision["primary"],
                        "secondary_tracks": decision["secondary_tracks"],
                        "output": str(output),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "render-report":
            render_report_file(args.decision, args.report)
            print(str(args.report.resolve()))
            return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from None
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
