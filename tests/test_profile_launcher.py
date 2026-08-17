import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.profile_launcher import _build_parser, build_command, load_profile, resolve_model


class ProfileLauncherTests(unittest.TestCase):
    def test_default_port_avoids_vast_jupyter_port(self):
        self.assertEqual(_build_parser().parse_args([]).port, 8000)

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
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
        self.assertIn("--no-enable-prefix-caching", command)

    def test_model_must_already_exist_locally(self):
        profile = load_profile("recommended")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "never downloads model weights"):
                resolve_model(profile, None)

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
