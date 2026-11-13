"""Tests for tools/to_backend_definition.py: engine spec -> backends.d definition."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import to_backend_definition as tbd  # noqa: E402


def spec(name):
    return json.loads((ROOT / "engines" / f"{name}.json").read_text())


def flat(definition):
    return [a for record in definition["launch"]["argv"] for a in record["args"]]


class Convert(unittest.TestCase):
    def test_committed_definitions_are_current(self):
        for path in sorted((ROOT / "engines").glob("*.json")):
            with self.subTest(spec=path.name):
                s = json.loads(path.read_text())
                target = ROOT / "backends.d" / f"{s['recipe']}.json"
                self.assertEqual(target.read_text(), tbd.render(tbd.convert(s)))

    def test_no_hooks_and_external_only(self):
        for path in sorted((ROOT / "engines").glob("*.json")):
            d = tbd.convert(json.loads(path.read_text()))
            with self.subTest(spec=path.name):
                self.assertEqual(d["schema_version"], 1)
                self.assertNotIn("hooks", d)
                self.assertEqual(d["install"], {"strategy": "external_only"})
                for op in d["operations"]:
                    self.assertNotIn("hooks", op["request"])
                    self.assertNotIn("handler", op)

    def test_model_and_port_become_references(self):
        d = tbd.convert(spec("lemonade-burn"))
        self.assertEqual(
            d["launch"]["argv"],
            [
                {
                    "args": [
                        "--model",
                        {"ref": "/model/checkpoint", "type": "string"},
                        "--host",
                        "127.0.0.1",
                        "--port",
                        {"string": {"ref": "/launch/ports/http"}},
                    ]
                }
            ],
        )
        self.assertEqual(d["metadata"]["options"], [])
        self.assertEqual(d["launch"]["backend"], {"fixed": "cpu"})

    def test_device_flag_is_its_own_conditional_record(self):
        d = tbd.convert(spec("nautistrata"))
        option = d["metadata"]["options"][0]
        self.assertEqual(option["name"], "nautistrata_device")
        self.assertEqual(option["type_name"], "DEVICES")
        self.assertEqual(option["default_value"], "")
        ref = {"ref": "/options/nautistrata_device"}
        self.assertEqual(
            d["launch"]["argv"][1],
            {"when": {"not": {"eq": [ref, ""]}}, "args": ["--gpu", {"string": ref}]},
        )
        self.assertNotIn("--gpu", d["launch"]["argv"][0]["args"])
        self.assertEqual(d["launch"]["backend"], {"fixed": "cuda"})

    def test_device_in_environment(self):
        d = tbd.convert(spec("freetoken"))
        env = {e["name"]: e["value"] for e in d["launch"]["environment"]["set"]}
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(d["metadata"]["options"][0]["default_value"], "0")
        self.assertEqual(
            env["CUDA_VISIBLE_DEVICES"], {"string": {"ref": "/options/freetoken_device"}}
        )

    def test_readiness_and_modes(self):
        d = tbd.convert(spec("bonsaicpu"))
        self.assertEqual(
            d["launch"]["readiness"], {"port": "http", "path": "/health", "timeout": 900}
        )
        self.assertTrue(d["metadata"]["exposes_prometheus_metrics"])
        self.assertIn("--metrics", flat(d))
        kinds = {(o["operation"], o["delivery"]) for o in d["operations"]}
        self.assertEqual(
            kinds,
            {
                ("chat_completion", "buffered"),
                ("chat_completion", "streaming"),
                ("completion", "buffered"),
                ("completion", "streaming"),
            },
        )
        for op in d["operations"]:
            self.assertEqual(op["model_modes"], ["chat"])

    def test_option_names_are_leaf_owned(self):
        d = tbd.convert(spec("lemonade-burn-lfm2"))
        self.assertEqual(d["metadata"]["config_section"], "lemonade_burn_lfm2")
        self.assertEqual(d["metadata"]["options"][0]["name"], "lemonade_burn_lfm2_device")

    def test_unknown_placeholder_is_refused(self):
        s = spec("lemonade-burn")
        s["args"] = s["args"] + ["--dir", "{spec_dir}"]
        with self.assertRaises(SystemExit):
            tbd.convert(s)

    def test_gaps_are_named_not_dropped_silently(self):
        found = "\n".join(tbd.gaps(spec("freetoken")))
        self.assertIn("ready probe", found)
        self.assertIn("owned_by=FreeToken", found)
        self.assertIn("telemetry_path /v1/stats", found)
        self.assertEqual(tbd.gaps(spec("strata"))[0][:11], "ready probe")


if __name__ == "__main__":
    unittest.main()
