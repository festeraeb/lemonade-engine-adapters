"""Tests for make_adapter.py: spec validation and the generated backend files."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import make_adapter as ma  # noqa: E402

SPEC = json.loads((ROOT / "engines" / "freetoken.json").read_text())


def bad(**changes):
    s = copy.deepcopy(SPEC)
    for k, v in changes.items():
        if v is None:
            s.pop(k, None)
        else:
            s[k] = v
    return s


class Validation(unittest.TestCase):
    def test_shipped_specs_are_valid(self):
        for p in sorted((ROOT / "engines").glob("*.json")):
            with self.subTest(spec=p.name):
                ma.render(json.loads(p.read_text()))

    def test_rejects(self):
        cases = {
            "missing field": bad(identity=None),
            "no {model}": bad(args_template="serve --port {port}"),
            "shell syntax": bad(args_template="serve --model {model} --port {port} | tee log"),
            "unknown placeholder": bad(args_template="serve --model {model} --port {port} --ctx {ctx}"),
            "binary with path": bad(binary="/usr/bin/ft"),
            "bad recipe": bad(recipe="Free_Token"),
            "bad device family": bad(support=[{"os": ["linux"], "devices": {"cuda": []}}]),
            "ready field without values": bad(ready={"path": "/health", "timeout_s": 60, "json_field": "status"}),
            "marker on a file": bad(checkpoint={"kind": "file", "marker": "x"}),
            "bad env name": bad(env={"cuda visible": "{device}"}),
        }
        for name, spec in cases.items():
            with self.subTest(case=name), self.assertRaises(ma.SpecError):
                ma.render(spec)


class Rendering(unittest.TestCase):
    def setUp(self):
        self.files = ma.render(copy.deepcopy(SPEC))

    def test_three_files_under_the_stem(self):
        self.assertEqual(sorted(self.files), [
            "src/cpp/include/lemon/backends/freetoken/freetoken.h",
            "src/cpp/include/lemon/backends/freetoken/freetoken_server.h",
            "src/cpp/server/backends/freetoken/freetoken_server.cpp",
        ])

    def test_descriptor_uses_the_2026_43_support_row(self):
        h = self.files["src/cpp/include/lemon/backends/freetoken/freetoken.h"]
        self.assertIn('{"system", BackendTier::Experimental, BackendFormat::Native, {"linux"},', h)
        self.assertIn('/*binary*/          "ft"', h)
        self.assertNotIn("/*experimental*/", h)  # 2026.43.x has no such field (it broke burn.h)

    def test_server_checks_readiness_and_identity(self):
        cpp = self.files["src/cpp/server/backends/freetoken/freetoken_server.cpp"]
        self.assertIn('kOwnedBy = "FreeToken"', cpp)
        self.assertIn('kReadyField = "status"', cpp)
        self.assertIn('kReadyValues = {"ok"}', cpp)
        self.assertIn('{"CUDA_VISIBLE_DEVICES", "{device}"}', cpp)
        self.assertIn("start_server(", cpp)
        self.assertIn("stop_server();", cpp)
        self.assertNotIn("set_process_handle", cpp)  # the 11.9 API that does not build on 2026.43.x

    def test_download_accepts_an_existing_checkpoint(self):
        cpp = self.files["src/cpp/server/backends/freetoken/freetoken_server.cpp"]
        self.assertIn("if (is_downloaded(info, ctx)) {", cpp)


class CMake(unittest.TestCase):
    def test_register_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            cm = Path(d) / "CMakeLists.txt"
            cm.write_text('x\nset(LEMON_BACKENDS\n    # comment\n    "llamacpp|llamacpp"\n)\ny\n')
            self.assertTrue(ma.register_cmake(Path(d), "freetoken", "freetoken"))
            self.assertFalse(ma.register_cmake(Path(d), "freetoken", "freetoken"))
            self.assertIn('    "llamacpp|llamacpp"\n    "freetoken|freetoken"\n)', cm.read_text())


if __name__ == "__main__":
    unittest.main()
