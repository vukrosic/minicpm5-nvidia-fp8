#!/usr/bin/env python3
"""Launch a pinned MiniCPM5 vLLM serving profile without downloading weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


PROFILE_SCHEMA = "minicpm5-serving-profile-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "profiles"
DEFAULT_PORT = 8000


def load_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> dict[str, Any]:
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in name):
        raise ValueError(f"Invalid profile name: {name!r}")
    path = (profiles_dir / f"{name}.json").resolve()
    if path.parent != profiles_dir.resolve() or not path.is_file():
        raise ValueError(f"Unknown profile: {name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read profile {path}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"Profile {path} does not use {PROFILE_SCHEMA}")
    if value.get("name") != name:
        raise ValueError(f"Profile filename/name mismatch: {path}")
    arguments = value.get("vllm_arguments")
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) and argument for argument in arguments
    ):
        raise ValueError(f"Profile {name} has invalid vllm_arguments")
    environment_variable = value.get("model_environment_variable")
    if not isinstance(environment_variable, str) or not environment_variable:
        raise ValueError(f"Profile {name} has no model environment variable")
    return dict(value, profile_path=str(path))


def resolve_model(profile: Mapping[str, Any], explicit_model: Path | None) -> Path:
    value = explicit_model
    if value is None:
        environment_variable = str(profile["model_environment_variable"])
        environment_value = os.environ.get(environment_variable)
        if not environment_value:
            raise ValueError(
                f"Set {environment_variable} to the existing GPU-local checkpoint "
                "or pass --model. This launcher never downloads model weights."
            )
        value = Path(environment_value)
    model = value.expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"Model must be an existing local directory: {model}")
    return model


def verify_rtx3060() -> dict[str, Any]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise ValueError("nvidia-smi is unavailable; use --allow-unsupported-gpu to bypass")
    result = subprocess.run(
        [
            binary,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"Expected one GPU, observed {len(rows)}")
    name, memory = [field.strip() for field in rows[0].rsplit(",", 1)]
    memory_mib = int(memory)
    if "RTX 3060" not in name or not 11800 <= memory_mib <= 12500:
        raise ValueError(
            f"Profile was validated on RTX 3060 12 GB; observed {name}, {memory_mib} MiB. "
            "Use --allow-unsupported-gpu only for a separately measured deployment."
        )
    return {"name": name, "memory_mib": memory_mib}


def build_command(
    profile: Mapping[str, Any],
    model: Path,
    *,
    vllm_binary: str,
    host: str,
    port: int,
    extra_arguments: Sequence[str] = (),
) -> list[str]:
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    return [
        vllm_binary,
        "serve",
        str(model),
        *[str(argument) for argument in profile["vllm_arguments"]],
        "--host",
        host,
        "--port",
        str(port),
        *extra_arguments,
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="recommended")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unsupported-gpu", action="store_true")
    parser.add_argument(
        "extra_arguments",
        nargs=argparse.REMAINDER,
        help="additional vLLM arguments after --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        profile = load_profile(args.profile)
        model = resolve_model(profile, args.model)
        command = build_command(
            profile,
            model,
            vllm_binary=args.vllm_bin,
            host=args.host,
            port=args.port,
            extra_arguments=args.extra_arguments,
        )
        hardware = None
        if not args.dry_run and not args.allow_unsupported_gpu:
            hardware = verify_rtx3060()
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "profile": profile["name"],
                        "release_class": profile["release_class"],
                        "model": str(model),
                        "command": command,
                        "environment": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
                        "disclosure": profile.get("disclosure"),
                    },
                    indent=2,
                )
            )
            return 0
        binary = shutil.which(args.vllm_bin)
        if binary is None:
            raise ValueError(f"vLLM executable is unavailable: {args.vllm_bin}")
        command[0] = binary
        environment = dict(os.environ)
        environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if hardware is not None:
            print(json.dumps({"profile": profile["name"], "hardware": hardware}))
        os.execvpe(binary, command, environment)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
