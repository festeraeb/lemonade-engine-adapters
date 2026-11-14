"""Tests for tools/adapt_engine.py: help parsing, role scoring, helper validation, the probe of a live stub.

No lemond and no model are needed; the conformance loop is exercised separately on real engines.
"""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import adapt_engine as ae  # noqa: E402

LLAMA = """\
-h,    --help, --usage                  print usage and exit
-t,    --threads N                      number of CPU threads to use during generation (default: -1)
                                        (env: LLAMA_ARG_THREADS)
-C,    --cpu-mask M                     CPU affinity mask: arbitrarily long hex.
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM
-dev,  --device <dev1,dev2,..>          comma-separated list of devices to use for offloading
-m,    --model FNAME                    model path to load
--host HOST                             ip address to listen on (default: 127.0.0.1)
--port PORT                             port to listen (default: 8080)
--rpc SERVERS                           comma-separated list of RPC servers (host:port)
--api-key KEY                           API key to use for authentication
--metrics                               enable prometheus compatible metrics endpoint (default: disabled)
--no-mmap                               do not memory-map model
"""

CLAP = """\
Usage: tool [OPTIONS] --model <MODEL>

Options:
      --model <MODEL>
          Model: a folder
      --gpu
          upload the experts to the GPU
      --gpu-budget-mb <GPU_BUDGET_MB>
          GPU byte budget (MB) for resident experts
      --host <HOST>
          [default: 127.0.0.1]
      --port <PORT>
          [default: 8000]
      --ctx <CTX>
          Max context (prompt + generated) tokens [default: 4096]
  -h, --help
          Print help
"""

ARGPARSE = """\
usage: server.py [-h] [--engine {mock,strata}] [--port PORT] [--gpu GPU]

options:
  -h, --help            show this help message and exit
  --engine {mock,strata}
                        which engine
  -p PORT, --port PORT  listen port
  --gpu GPU             CUDA device index
"""


def names(flags):
    return {n for f in flags for n in f.names}


class ParseHelp(unittest.TestCase):
    def test_llama_style_keeps_aliases_and_comma_metavars(self):
        flags = ae.parse_flags(LLAMA)
        ctx = next(f for f in flags if "--ctx-size" in f.names)
        self.assertEqual(ctx.names, ["-c", "--ctx-size"])
        self.assertEqual(ctx.metavar, "N")
        dev = next(f for f in flags if "--device" in f.names)
        self.assertEqual(dev.names, ["-dev", "--device"])
        self.assertTrue(dev.takes_value)
        self.assertEqual(next(f for f in flags if "--threads" in f.names).env_names(), ["LLAMA_ARG_THREADS"])
        self.assertFalse(next(f for f in flags if "--metrics" in f.names).takes_value)

    def test_clap_style_description_on_next_line(self):
        flags = ae.parse_flags(CLAP)
        ctx = next(f for f in flags if "--ctx" in f.names)
        self.assertIn("context", ctx.desc)
        self.assertFalse(next(f for f in flags if f.names == ["--gpu"]).takes_value)
        self.assertTrue(next(f for f in flags if "--gpu-budget-mb" in f.names).takes_value)

    def test_argparse_style_metavar_after_each_name(self):
        flags = ae.parse_flags(ARGPARSE)
        port = next(f for f in flags if "--port" in f.names)
        self.assertEqual(port.names, ["-p", "--port"])
        self.assertEqual(port.metavar, "PORT")

    def test_garbage_gives_no_flags(self):
        self.assertEqual(ae.parse_flags("nothing here\njust prose, no options\n"), [])


class Roles(unittest.TestCase):
    def test_llama_roles_decided_by_the_scorer(self):
        flags = ae.parse_flags(LLAMA)
        got = {}
        for role in ("port", "host", "model", "ctx", "device", "threads", "api_key"):
            flag, status, _ = ae.decide(role, flags)
            self.assertEqual(status, "decided", role)
            got[role] = ae.preferred_name(flag)
        self.assertEqual(got, {"port": "--port", "host": "--host", "model": "--model", "ctx": "--ctx-size",
                               "device": "--device", "threads": "--threads", "api_key": "--api-key"})

    def test_short_names_are_case_sensitive(self):
        # -C is the CPU mask, not the context size
        flags = ae.parse_flags(LLAMA)
        self.assertNotIn("-C", [n for _, f in ae.rank("ctx", flags) for n in f.names])

    def test_boolean_flag_is_never_a_device_selector(self):
        flags = ae.parse_flags(CLAP)
        flag, status, _ = ae.decide("device", flags)
        self.assertIsNone(flag)
        self.assertNotEqual(status, "decided")

    def test_budget_flag_is_excluded_from_device(self):
        flags = ae.parse_flags(CLAP)
        self.assertNotIn("--gpu-budget-mb", [n for _, f in ae.rank("device", flags) for n in f.names])

    def test_metrics_switch_found(self):
        self.assertEqual(ae.metrics_flag(ae.parse_flags(LLAMA)).names, ["--metrics"])
        self.assertIsNone(ae.metrics_flag(ae.parse_flags(CLAP)))


class HelperValidation(unittest.TestCase):
    def setUp(self):
        self.flags = ae.parse_flags(LLAMA)

    def check(self, role, answer):
        return ae.validate_answer(role, answer, LLAMA, self.flags)

    def test_accepts_a_flag_with_a_quoted_line(self):
        line = "-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)"
        flag, why = self.check("ctx", {"flag": "--ctx-size", "evidence": line})
        self.assertEqual((flag, why), ("--ctx-size", "accepted"))

    def test_flag_written_with_its_metavar_is_trimmed(self):
        line = "-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)"
        flag, why = self.check("ctx", {"flag": "--ctx-size N", "evidence": line})
        self.assertEqual(flag, "--ctx-size")
        flag, why = self.check("ctx", {"flag": "--ctx-size and more words", "evidence": line})
        self.assertIsNone(flag)

    def test_rejects_a_flag_not_in_the_help(self):
        flag, why = self.check("ctx", {"flag": "--context-window", "evidence": "--context-window N"})
        self.assertIsNone(flag)
        self.assertIn("not written", why)

    def test_rejects_invented_evidence(self):
        flag, why = self.check("ctx", {"flag": "--ctx-size", "evidence": "--ctx-size sets the window size"})
        self.assertIsNone(flag)
        self.assertIn("evidence", why)

    def test_rejects_a_flag_without_a_value(self):
        flag, why = self.check("device", {"flag": "--no-mmap", "evidence": "--no-mmap                               do not memory-map model"})
        self.assertIsNone(flag)

    def test_rejects_an_excluded_name(self):
        flag, why = self.check("device", {"flag": "--gpu-layers", "evidence": "-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM"})
        self.assertIsNone(flag)
        self.assertIn("exclusion", why)

    def test_null_and_malformed_answers(self):
        self.assertIsNone(self.check("ctx", {"flag": None, "evidence": ""})[0])
        self.assertIsNone(self.check("ctx", "just text")[0])
        self.assertIsNone(self.check("ctx", {"flag": "ctx-size", "evidence": "x"})[0])

    def test_extract_json_from_fenced_or_chatty_text(self):
        self.assertEqual(ae.extract_json('```json\n{"flag": "--x"}\n```'), {"flag": "--x"})
        self.assertEqual(ae.extract_json('Sure! {"flag": null, "evidence": ""} done'), {"flag": None, "evidence": ""})
        self.assertIsNone(ae.extract_json("no json"))


def fake_helper(answer_for):
    """An OpenAI-style endpoint on a free port; answer_for(prompt) -> assistant text."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            n = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(n))
            text = answer_for(body["messages"][0]["content"])
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


class HelperCalls(unittest.TestCase):
    def test_call_is_logged_and_validated(self):
        import tempfile
        flags = ae.parse_flags(LLAMA)
        line = "-dev,  --device <dev1,dev2,..>          comma-separated list of devices to use for offloading"
        srv, url = fake_helper(lambda q: json.dumps({"flag": "--device", "evidence": line}))
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "calls.jsonl"
            helper = ae.Helper([url], "m", log)
            got = helper.ask("device", LLAMA, flags, [])
            srv.shutdown()
            self.assertEqual(got.names, ["-dev", "--device"])
            rec = json.loads(log.read_text().splitlines()[0])
            self.assertTrue(rec["accepted"])
            self.assertEqual(rec["role"], "device")

    def test_a_wrong_answer_is_logged_and_rejected(self):
        import tempfile
        flags = ae.parse_flags(LLAMA)
        srv, url = fake_helper(lambda q: '{"flag": "--made-up", "evidence": "--made-up N"}')
        with tempfile.TemporaryDirectory() as d:
            helper = ae.Helper([url], "m", Path(d) / "calls.jsonl")
            self.assertIsNone(helper.ask("ctx", LLAMA, flags, []))
            srv.shutdown()
            self.assertFalse(helper.calls[0]["accepted"])
            self.assertIn("not written", helper.calls[0]["reason"])

    def test_dead_helper_is_a_rejection_not_a_crash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            helper = ae.Helper(["http://127.0.0.1:9/v1"], "m", Path(d) / "calls.jsonl", timeout=2)
            self.assertIsNone(helper.ask("ctx", LLAMA, ae.parse_flags(LLAMA), []))
            self.assertFalse(helper.calls[0]["accepted"])


class RepairRules(unittest.TestCase):
    SPEC = {"recipe": "x", "args": ["--model", "{model}", "--port", "{port}", "--bogus", "3"],
            "ready": {"path": "/health", "timeout_s": 600}}

    def test_unknown_flag_in_engine_output_is_removed(self):
        rep = {"levels": {"L1": {"status": "FAIL", "detail": "load failed"}},
               "log": "error: unrecognized option '--bogus'"}
        new, what, _ = ae.propose_fix(self.SPEC, rep, set(), set())
        self.assertEqual(new["args"], ["--model", "{model}", "--port", "{port}"])

    def test_helper_flag_is_removed_when_load_fails(self):
        rep = {"levels": {"L1": {"status": "FAIL", "detail": "load failed"}}, "log": ""}
        new, what, _ = ae.propose_fix(self.SPEC, rep, {"--bogus"}, set())
        self.assertNotIn("--bogus", new["args"])
        self.assertIn("helper", what)

    def test_nothing_to_repair_gives_none(self):
        rep = {"levels": {"L1": {"status": "pass", "detail": ""}}, "log": ""}
        self.assertIsNone(ae.propose_fix(self.SPEC, rep, set(), set()))


class LiveStub(unittest.TestCase):
    """The whole probe against tests/fixtures/oddengine.py: odd flag names, the helper decides one role."""

    def test_probe_launch_and_spec(self):
        import tempfile
        engine = str(ROOT / "tests" / "fixtures" / "oddengine.py")
        weights = Path(tempfile.mkdtemp()) / "w.bin"
        weights.write_text("x")
        probe = ae.probe_help(engine, dict(os.environ))
        flags = probe["flags"]
        self.assertEqual(ae.decide("port", flags)[1], "decided")
        self.assertEqual(ae.preferred_name(ae.decide("port", flags)[0]), "--bind-port")
        self.assertEqual(ae.preferred_name(ae.decide("model", flags)[0]), "--weights")
        flag, status, cands = ae.decide("ctx", flags)
        self.assertNotEqual(status, "decided")  # --window is described only in words
        line = next(l for l in probe["help"].splitlines() if "--window" in l).strip()
        srv, url = fake_helper(lambda q: json.dumps({"flag": "--window", "evidence": line}))
        with tempfile.TemporaryDirectory() as d:
            helper = ae.Helper([url], "m", Path(d) / "calls.jsonl")
            ctx = helper.ask("ctx", probe["help"], flags, cands)
        srv.shutdown()
        self.assertEqual(ae.preferred_name(ctx), "--window")
        port = ae.free_port()
        with tempfile.TemporaryDirectory() as d:
            proc = ae.Proc([engine, "--weights", str(weights), "--bind-port", str(port)], dict(os.environ),
                           Path(d) / "e.log", None)
            try:
                import time
                for _ in range(50):
                    if ae.http("GET", f"http://127.0.0.1:{port}/health", timeout=2)[0] == 200:
                        break
                    time.sleep(0.1)
                info = ae.openai_probe(f"http://127.0.0.1:{port}", print)
            finally:
                proc.stop()
        self.assertTrue(info["chat"]["ok"] and info["stream"]["ok"] and info["completion"]["ok"])
        self.assertFalse(info["metrics"]["ok"])
        self.assertEqual(info["owned_by"], "oddengine")
        roles = {"model": ae.decide("model", flags)[0], "port": ae.decide("port", flags)[0], "ctx": ctx}
        spec = ae.build_spec("odd", engine, roles, None, probe, info, weights, "2048", 600)
        self.assertEqual(spec["args"], ["--weights", "{model}", "--bind-port", "{port}", "--window", "2048"])
        self.assertEqual(spec["ready"]["path"], "/health")
        import to_backend_definition as tbd
        definition = tbd.convert(spec)
        self.assertEqual(definition["install"]["strategy"], "external_only")
        self.assertNotIn("hooks", json.dumps(definition))
        import check_specs
        problems, _ = check_specs.check(spec)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
