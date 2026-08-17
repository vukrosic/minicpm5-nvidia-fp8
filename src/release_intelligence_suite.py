#!/usr/bin/env python3
"""Freeze, run, and compare the compact public release-intelligence suite.

Dataset and evaluator imports stay inside GPU-only commands. The Mac can still
validate specs and compare already-pulled JSON receipts without installing the
model stack. Model paths must be local; this module never resolves model weights
from a hub identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPEC_SCHEMA = "minicpm5-release-intelligence-spec-v1"
MANIFEST_SCHEMA = "minicpm5-release-intelligence-manifest-v1"
RUN_SCHEMA = "minicpm5-release-intelligence-run-v1"
COMPARISON_SCHEMA = "minicpm5-release-intelligence-comparison-v3"
EXPECTED_DOMAINS = ("gsm8k", "mmlu", "ceval", "ifeval")
EXPECTED_DOMAIN_TASKS = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return dict(value)


def _write_json_once(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = _load_json(path, "benchmark spec")
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"Benchmark spec must use {SPEC_SCHEMA}")
    if spec.get("status") != "frozen_before_model_runs":
        raise ValueError("Benchmark spec is not frozen before model runs")
    domains = spec.get("domains")
    if not isinstance(domains, Mapping) or tuple(domains) != EXPECTED_DOMAINS:
        raise ValueError(f"Benchmark domains must be {EXPECTED_DOMAINS}")
    for domain_name, raw_domain in domains.items():
        if not isinstance(raw_domain, Mapping):
            raise ValueError(f"Domain {domain_name} must be an object")
        task_names = raw_domain.get("task_names")
        limit = raw_domain.get("limit_per_task")
        total = raw_domain.get("total_tasks")
        if not isinstance(task_names, list) or not task_names:
            raise ValueError(f"Domain {domain_name} has no task names")
        if len(set(task_names)) != len(task_names):
            raise ValueError(f"Domain {domain_name} repeats a task name")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError(f"Domain {domain_name} has invalid limit_per_task")
        if total != len(task_names) * limit or total != EXPECTED_DOMAIN_TASKS:
            raise ValueError(
                f"Domain {domain_name} must select exactly {EXPECTED_DOMAIN_TASKS} tasks"
            )
        if raw_domain.get("primary_metric") not in {
            "exact_match",
            "acc",
            "prompt_level_strict_acc",
        }:
            raise ValueError(f"Domain {domain_name} has unsupported primary metric")
    return spec


def _dataset_config(domain: Mapping[str, Any], task_name: str) -> str | None:
    rule = str(domain["dataset_config_rule"])
    if rule == "none":
        return None
    if rule.startswith("literal:"):
        return rule.split(":", 1)[1]
    if rule.startswith("strip_prefix:"):
        prefix = rule.split(":", 1)[1]
        if not task_name.startswith(prefix):
            raise ValueError(f"Task {task_name} does not start with {prefix}")
        return task_name[len(prefix) :]
    raise ValueError(f"Unsupported dataset config rule: {rule}")


def _dataset_revision(repo_id: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError("freeze requires huggingface_hub on the GPU host") from exc
    revision = HfApi().dataset_info(repo_id=repo_id).sha
    if not revision:
        raise ValueError(f"Could not resolve dataset revision for {repo_id}")
    return str(revision)


def freeze_manifest(spec_path: str | Path, output_path: str | Path) -> Path:
    try:
        import datasets
    except ImportError as exc:
        raise ValueError("freeze requires datasets on the GPU host") from exc

    spec = load_spec(spec_path)
    revisions = {
        str(domain["dataset"]): _dataset_revision(str(domain["dataset"]))
        for domain in spec["domains"].values()
    }
    rows: list[dict[str, Any]] = []
    datasets_used: list[dict[str, Any]] = []
    for domain_name, domain in spec["domains"].items():
        repo_id = str(domain["dataset"])
        revision = revisions[repo_id]
        for task_name in domain["task_names"]:
            config_name = _dataset_config(domain, str(task_name))
            dataset = datasets.load_dataset(
                repo_id,
                config_name,
                split=str(domain["split"]),
                revision=revision,
            )
            limit = int(domain["limit_per_task"])
            if len(dataset) < limit:
                raise ValueError(
                    f"{task_name} has only {len(dataset)} rows; requires {limit}"
                )
            info = dataset.info
            datasets_used.append(
                {
                    "domain": domain_name,
                    "task_name": task_name,
                    "repo_id": repo_id,
                    "revision": revision,
                    "config": config_name,
                    "split": domain["split"],
                    "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
                    "dataset_rows": len(dataset),
                    "dataset_version": str(getattr(info, "version", "")),
                }
            )
            for index in range(limit):
                doc = dataset[index]
                rows.append(
                    {
                        "domain": domain_name,
                        "task_name": task_name,
                        "dataset_index": index,
                        "doc_sha256": sha256_text(_canonical_json(doc)),
                    }
                )

    counts = {
        domain: sum(row["domain"] == domain for row in rows)
        for domain in EXPECTED_DOMAINS
    }
    if counts != {domain: EXPECTED_DOMAIN_TASKS for domain in EXPECTED_DOMAINS}:
        raise ValueError(f"Frozen manifest has unexpected counts: {counts}")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "id": str(spec["id"]),
        "frozen_at": utc_now(),
        "spec_path": str(Path(spec_path)),
        "spec_sha256": sha256_file(spec_path),
        "lm_eval_version": spec["lm_eval"]["version"],
        "datasets_version": importlib.metadata.version("datasets"),
        "dataset_revisions": revisions,
        "datasets": datasets_used,
        "counts": counts,
        "rows": rows,
        "claim_boundary": spec["claim_boundary"],
    }
    return _write_json_once(output_path, payload)


def load_manifest(path: str | Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(path, "frozen manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Frozen manifest must use {MANIFEST_SCHEMA}")
    if manifest.get("id") != spec.get("id"):
        raise ValueError("Spec and manifest IDs differ")
    if manifest.get("spec_sha256") != sha256_file(spec["_path"]):
        raise ValueError("Frozen manifest does not match the benchmark spec hash")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 200:
        raise ValueError("Frozen manifest must contain exactly 200 task rows")
    return manifest


def _http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _selected_sample_rows(
    raw_results: Mapping[str, Any], domain: Mapping[str, Any]
) -> list[tuple[str, Mapping[str, Any]]]:
    samples = raw_results.get("samples")
    if not isinstance(samples, Mapping):
        raise ValueError("lm-eval result has no samples object")
    selected: list[tuple[str, Mapping[str, Any]]] = []
    expected_filter = str(domain["filter"])
    for task_name in domain["task_names"]:
        task_rows = samples.get(task_name)
        if not isinstance(task_rows, list):
            raise ValueError(f"lm-eval result has no samples for {task_name}")
        matching = [
            row
            for row in task_rows
            if isinstance(row, Mapping) and row.get("filter") == expected_filter
        ]
        if len(matching) != int(domain["limit_per_task"]):
            raise ValueError(
                f"{task_name} produced {len(matching)} {expected_filter!r} samples"
            )
        selected.extend((str(task_name), row) for row in matching)
    return selected


def _validate_selected_docs(
    raw_results: Mapping[str, Any],
    domain_name: str,
    domain: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    expected = {
        (str(row["task_name"]), int(row["dataset_index"])): str(row["doc_sha256"])
        for row in manifest["rows"]
        if row["domain"] == domain_name
    }
    observed: dict[tuple[str, int], str] = {}
    for task_name, sample in _selected_sample_rows(raw_results, domain):
        key = (task_name, int(sample["doc_id"]))
        observed[key] = sha256_text(_canonical_json(sample["doc"]))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(observed) if expected[key] != observed[key]
        )
        raise ValueError(
            "lm-eval documents differ from frozen manifest: "
            f"missing={missing[:5]}, extra={extra[:5]}, mismatched={mismatched[:5]}"
        )


def run_domain(
    *,
    spec_path: str | Path,
    manifest_path: str | Path,
    domain_name: str,
    family: str,
    model_name: str,
    base_url: str,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> Path:
    spec = load_spec(spec_path)
    spec["_path"] = str(Path(spec_path).expanduser().resolve())
    manifest = load_manifest(manifest_path, spec)
    if domain_name not in EXPECTED_DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}")
    if family not in spec["families"]:
        raise ValueError(f"Unknown family: {family}")
    tokenizer_dir = Path(tokenizer_path).expanduser().resolve()
    if not tokenizer_dir.is_dir():
        raise ValueError(f"Tokenizer must be a local directory: {tokenizer_dir}")
    template_path = tokenizer_dir / "chat_template.jinja"
    if not template_path.is_file():
        raise ValueError(f"Missing official chat template: {template_path}")

    try:
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.models.openai_completions import LocalCompletionsAPI
    except ImportError as exc:
        raise ValueError("run-domain requires lm-eval on the GPU host") from exc

    models = _http_json(base_url.rsplit("/v1/completions", 1)[0] + "/v1/models")
    available = {str(item.get("id")) for item in models.get("data", [])}
    if model_name not in available:
        raise ValueError(f"Serving model {model_name!r} is unavailable: {sorted(available)}")

    no_think_template = (
        "{%- set enable_thinking = false -%}\n"
        + template_path.read_text(encoding="utf-8")
    )
    lm = LocalCompletionsAPI(
        model=model_name,
        base_url=base_url,
        tokenizer=str(tokenizer_dir),
        tokenizer_backend="huggingface",
        # Required for log-likelihood tasks with a chat template: lm-eval must
        # render the messages to the model's prompt before splitting context
        # and continuations. Its non-tokenized API path returns JsonChatStr,
        # which is valid for chat generation but not context scoring.
        tokenized_requests=True,
        num_concurrent=1,
        max_retries=3,
        batch_size=1,
        seed=0,
        max_length=int(spec["lm_eval"]["max_model_len"]),
        max_gen_toks=1280,
        timeout=300,
    )
    lm.tokenizer.chat_template = no_think_template
    domain = spec["domains"][domain_name]
    generation_max = domain.get("generation_max_tokens")
    gen_kwargs = None
    if generation_max is not None:
        gen_kwargs = {
            "do_sample": False,
            "temperature": 0.0,
            "max_gen_toks": int(generation_max),
        }
    started_at = utc_now()
    raw_results = simple_evaluate(
        model=lm,
        tasks=list(domain["task_names"]),
        num_fewshot=int(domain["num_fewshot"]),
        batch_size=1,
        limit=int(domain["limit_per_task"]),
        bootstrap_iters=1000,
        log_samples=True,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        gen_kwargs=gen_kwargs,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
        verbosity="INFO",
    )
    if raw_results is None:
        raise ValueError("lm-eval returned no result")
    _validate_selected_docs(raw_results, domain_name, domain, manifest)
    payload = {
        "schema": RUN_SCHEMA,
        "suite_id": spec["id"],
        "family": family,
        "domain": domain_name,
        "started_at": started_at,
        "finished_at": utc_now(),
        "model_name": model_name,
        "base_url": base_url,
        "spec_sha256": sha256_file(spec_path),
        "manifest_sha256": sha256_file(manifest_path),
        "official_chat_template_sha256": sha256_file(template_path),
        "effective_no_think_template_sha256": sha256_text(no_think_template),
        "environment": {
            "lm_eval": importlib.metadata.version("lm-eval"),
            "datasets": importlib.metadata.version("datasets"),
            "transformers": importlib.metadata.version("transformers"),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "selection": {
            "task_names": domain["task_names"],
            "limit_per_task": domain["limit_per_task"],
            "total_tasks": domain["total_tasks"],
            "num_fewshot": domain["num_fewshot"],
            "primary_metric": domain["primary_metric"],
            "filter": domain["filter"],
        },
        "lm_eval": raw_results,
    }
    return _write_json_once(output_path, payload)


def _metric_value(sample: Mapping[str, Any], domain: Mapping[str, Any]) -> bool:
    metric = str(domain["primary_metric"])
    value = sample.get(metric)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(float(value) >= 0.5)
    raise ValueError(f"Sample has no numeric {metric!r} metric")


def _decision_output(sample: Mapping[str, Any], domain: Mapping[str, Any]) -> Any:
    """Return the comparable model decision, excluding harmless score drift."""

    filtered = sample.get("filtered_resps")
    if domain["primary_metric"] != "acc":
        return filtered
    if not isinstance(filtered, list) or not filtered:
        raise ValueError("Multiple-choice sample has no filtered responses")
    scores: list[float] = []
    for response in filtered:
        value = response[0] if isinstance(response, (list, tuple)) else response
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("Multiple-choice sample has an invalid log-likelihood")
        scores.append(float(value))
    return max(range(len(scores)), key=scores.__getitem__)


def paired_direction_p_value(losses: int, gains: int) -> float:
    """Exact two-sided sign/McNemar test over discordant paired outcomes."""

    if losses < 0 or gains < 0:
        raise ValueError("Paired counts cannot be negative")
    discordant = losses + gains
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(losses, gains) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _indexed_outcomes(
    run: Mapping[str, Any], domain: Mapping[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    raw = run.get("lm_eval")
    if not isinstance(raw, Mapping):
        raise ValueError("Run has no lm_eval object")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for task_name, sample in _selected_sample_rows(raw, domain):
        key = (task_name, int(sample["doc_id"]))
        if key in indexed:
            raise ValueError(f"Duplicate selected sample: {key}")
        decision_output = _decision_output(sample, domain)
        indexed[key] = {
            "passed": _metric_value(sample, domain),
            "decision_output": decision_output,
            "output_sha256": sha256_text(_canonical_json(decision_output)),
        }
    return indexed


def compare_payloads(
    spec: Mapping[str, Any],
    reference_runs: Mapping[str, Mapping[str, Any]],
    candidate_runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    all_losses: list[str] = []
    all_gains: list[str] = []
    total_reference = 0
    total_candidate = 0
    total_exact = 0
    for domain_name, domain in spec["domains"].items():
        reference = _indexed_outcomes(reference_runs[domain_name], domain)
        candidate = _indexed_outcomes(candidate_runs[domain_name], domain)
        if set(reference) != set(candidate) or len(reference) != EXPECTED_DOMAIN_TASKS:
            raise ValueError(f"Paired task mismatch in {domain_name}")
        losses: list[str] = []
        gains: list[str] = []
        exact = 0
        rows: list[dict[str, Any]] = []
        for key in sorted(reference):
            ref = reference[key]
            cand = candidate[key]
            task_id = f"{key[0]}:{key[1]}"
            if ref["passed"] and not cand["passed"]:
                losses.append(task_id)
            if not ref["passed"] and cand["passed"]:
                gains.append(task_id)
            same_output = ref["output_sha256"] == cand["output_sha256"]
            exact += int(same_output)
            rows.append(
                {
                    "task_id": task_id,
                    "reference_passed": ref["passed"],
                    "candidate_passed": cand["passed"],
                    "output_agreement": same_output,
                    "reference_choice": (
                        ref["decision_output"]
                        if domain["primary_metric"] == "acc"
                        else None
                    ),
                    "candidate_choice": (
                        cand["decision_output"]
                        if domain["primary_metric"] == "acc"
                        else None
                    ),
                }
            )
        ref_passed = sum(int(row["reference_passed"]) for row in rows)
        cand_passed = sum(int(row["candidate_passed"]) for row in rows)
        domains[domain_name] = {
            "task_count": len(rows),
            "reference_passed": ref_passed,
            "candidate_passed": cand_passed,
            "score_delta": cand_passed - ref_passed,
            "paired_losses": losses,
            "paired_loss_count": len(losses),
            "paired_gains": gains,
            "paired_gain_count": len(gains),
            "discordant_task_count": len(losses) + len(gains),
            "paired_direction_exact_p_value": paired_direction_p_value(
                len(losses), len(gains)
            ),
            "output_agreement_count": exact,
            "tasks": rows,
        }
        all_losses.extend(f"{domain_name}/{item}" for item in losses)
        all_gains.extend(f"{domain_name}/{item}" for item in gains)
        total_reference += ref_passed
        total_candidate += cand_passed
        total_exact += exact

    if total_candidate < total_reference:
        classification = "aggregate_regression_observed_on_frozen_subset"
    elif all_losses:
        classification = "quality_churn_observed_on_frozen_subset"
    else:
        classification = "no_paired_losses_observed_on_frozen_subset"
    return {
        "schema": COMPARISON_SCHEMA,
        "suite_id": spec["id"],
        "recorded_at": utc_now(),
        "reference_family": spec["families"][0],
        "candidate_family": spec["families"][1],
        "task_count": 200,
        "reference_passed": total_reference,
        "candidate_passed": total_candidate,
        "aggregate_delta": total_candidate - total_reference,
        "paired_losses": all_losses,
        "paired_loss_count": len(all_losses),
        "paired_gains": all_gains,
        "paired_gain_count": len(all_gains),
        "discordant_task_count": len(all_losses) + len(all_gains),
        "paired_direction_exact_p_value": paired_direction_p_value(
            len(all_losses), len(all_gains)
        ),
        "output_agreement_count": total_exact,
        "classification": classification,
        "domains": domains,
        "uncertainty_note": (
            "The exact paired-direction p-value tests whether losses and gains are "
            "directionally imbalanced within this frozen subset. It does not remove "
            "benchmark-selection uncertainty or prove equivalence when large."
        ),
        "claim_boundary": spec["claim_boundary"],
    }


def compare_runs(
    spec_path: str | Path,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    output_path: str | Path,
) -> Path:
    spec = load_spec(spec_path)
    reference_runs = {
        domain: _load_json(Path(reference_dir) / f"{domain}.json", "reference run")
        for domain in EXPECTED_DOMAINS
    }
    candidate_runs = {
        domain: _load_json(Path(candidate_dir) / f"{domain}.json", "candidate run")
        for domain in EXPECTED_DOMAINS
    }
    for domain in EXPECTED_DOMAINS:
        for expected_family, run in (
            (spec["families"][0], reference_runs[domain]),
            (spec["families"][1], candidate_runs[domain]),
        ):
            if run.get("schema") != RUN_SCHEMA or run.get("domain") != domain:
                raise ValueError(f"Invalid {domain} run receipt")
            if run.get("family") != expected_family:
                raise ValueError(f"Unexpected family in {domain} run receipt")
    return _write_json_once(
        output_path, compare_payloads(spec, reference_runs, candidate_runs)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec")
    validate.add_argument("--spec", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--spec", required=True)
    freeze.add_argument("--output", required=True)

    run = commands.add_parser("run-domain")
    run.add_argument("--spec", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--domain", choices=EXPECTED_DOMAINS, required=True)
    run.add_argument("--family", required=True)
    run.add_argument("--model-name", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1/completions")
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--output", required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--spec", required=True)
    compare.add_argument("--reference-dir", required=True)
    compare.add_argument("--candidate-dir", required=True)
    compare.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate-spec":
            spec = load_spec(args.spec)
            print(json.dumps({"id": spec["id"], "tasks": 200}, indent=2))
        elif args.command == "freeze":
            print(freeze_manifest(args.spec, args.output))
        elif args.command == "run-domain":
            print(
                run_domain(
                    spec_path=args.spec,
                    manifest_path=args.manifest,
                    domain_name=args.domain,
                    family=args.family,
                    model_name=args.model_name,
                    base_url=args.base_url,
                    tokenizer_path=args.tokenizer,
                    output_path=args.output,
                )
            )
        elif args.command == "compare":
            print(
                compare_runs(
                    args.spec,
                    args.reference_dir,
                    args.candidate_dir,
                    args.output,
                )
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {args.command}")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
