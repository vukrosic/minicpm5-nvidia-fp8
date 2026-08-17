import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.profile_launcher import (
    _build_parser,
    build_command,
    load_profile,
    resolve_model,
    verify_supported_nvidia_gpu,
)


class ProfileLauncherTests(unittest.TestCase):
    def test_default_port_avoids_vast_jupyter_port(self):
        arguments = _build_parser().parse_args([])
        self.assertEqual(arguments.port, 8000)
        self.assertEqual(arguments.profile, "recommended")

    def test_recommended_profile_builds_single_request_fp8_command(self):
        profile = load_profile("recommended")
        command = build_command(
            profile,
            Path("/workspace/model"),
            vllm_binary="vllm",
            host="127.0.0.1",
            port=8080,
        )
        self.assertEqual(command[:3], ["vllm", "serve", "/workspace/model"])
        self.assertIn("fp8_per_block", command)
        self.assertEqual(profile["release_class"], "optimized_default")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
        self.assertIn("--no-enable-prefix-caching", command)

    def test_model_must_already_exist_locally(self):
        profile = load_profile("recommended")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "never downloads model weights"):
                resolve_model(profile, None)

    def test_ampere_or_newer_gpu_is_accepted_without_claiming_measurement(self):
        completed = mock.Mock(stdout="NVIDIA GeForce RTX 4090, 24564, 8.9\n")
        with (
            mock.patch(
                "src.profile_launcher.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch(
                "src.profile_launcher.subprocess.run",
                return_value=completed,
            ) as run,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            hardware = verify_supported_nvidia_gpu()
        self.assertEqual(hardware["compute_capability"], "8.9")
        self.assertEqual(hardware["benchmark_status"], "compatible-unbenchmarked")
        self.assertIn("compute_cap", " ".join(run.call_args.args[0]))

    def test_rtx3060_is_identified_as_the_measured_reference(self):
        completed = mock.Mock(stdout="NVIDIA GeForce RTX 3060, 12288, 8.6\n")
        with (
            mock.patch(
                "src.profile_launcher.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch(
                "src.profile_launcher.subprocess.run",
                return_value=completed,
            ) as run,
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=True),
        ):
            hardware = verify_supported_nvidia_gpu()
        self.assertEqual(hardware["benchmark_status"], "measured-rtx3060-reference")
        self.assertEqual(hardware["selected_device"], "2")
        self.assertIn("--id=2", run.call_args.args[0])

    def test_turing_gpu_is_rejected_by_the_bf16_default(self):
        completed = mock.Mock(stdout="NVIDIA GeForce RTX 2080 Ti, 11264, 7.5\n")
        with (
            mock.patch(
                "src.profile_launcher.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch(
                "src.profile_launcher.subprocess.run",
                return_value=completed,
            ),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(ValueError, "Ampere or newer"):
                verify_supported_nvidia_gpu()

    def test_hidden_cuda_devices_fail_before_hardware_query(self):
        with (
            mock.patch(
                "src.profile_launcher.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch("src.profile_launcher.subprocess.run") as run,
            mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "-1"},
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "exposes no NVIDIA GPU"):
                verify_supported_nvidia_gpu()
        run.assert_not_called()

    def test_profile_schema_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.json").write_text(
                json.dumps({"schema": "wrong", "name": "bad"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not use"):
                load_profile("bad", root)


if __name__ == "__main__":
    unittest.main()
