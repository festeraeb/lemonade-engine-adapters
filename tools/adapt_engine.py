#!/usr/bin/env python3
"""adapt-engine: bring an engine into Lemonade from its binary, with no hand-written spec.

    adapt_engine.py --binary /path/to/engine-server --model /path/to/model [--name NAME]
                    --lemond /path/to/lemond --lemonade /path/to/lemonade
                    --conformance /path/to/backend_conformance.py
                    [--helper-url http://host:port/v1 --helper-model ID | --no-helper]

What it does, in order:
  1. probe     runs --help / -h / --version and parses the flags (port, host, model, context size,
               device, threads, api key, metrics). A deterministic scorer decides each role.
  2. launch    starts the engine on a free port with the model and checks, over HTTP, /health,
               /v1/models, /v1/chat/completions (buffered and streaming), /v1/completions, /metrics.
  3. write     builds the engine spec and the hook-free backends.d definition (tools/to_backend_definition.py).
  4. conform   starts a throwaway lemond with the definition and grades it with backend_conformance.py;
               reads the failures, repairs the definition by rule, and repeats (bounded) while the level
               improves. Prints the level per check, with the reason for each one not reached.

A small local model (--helper-url, any OpenAI-compatible chat endpoint) is asked only for a role the scorer
cannot decide: several flags tie, or the flags did not parse. Its answer is one small JSON object. It is
rejected unless the flag it names is written in the --help text, the line it quotes is in that text, the flag
takes a value, and it passes the role's exclusion list. An accepted answer is still only a hypothesis: if the
conformance run fails with it in place, the flag is removed and the run repeated. Every call is logged.
Without a helper (or with --no-helper) an undecided role is left out and reported.

Standard library only. Python is used here because the converter and checker it builds on are Python.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import to_backend_definition as tbd  # noqa: E402  (the converter this tool reuses)

CPU_ONLY_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "ROCR_VISIBLE_DEVICES": "-1",
    "GGML_VK_VISIBLE_DEVICES": "-1",
}

# ---------------------------------------------------------------- help parsing


class Flag:
    def __init__(self, names, metavar, desc, indent):
        self.names = names
        self.metavar = metavar  # "" when the flag takes no value
        self.optional_value = bool(metavar) and metavar.startswith("[")
        self.desc = desc
        self.indent = indent

    @property
    def takes_value(self):
        return bool(self.metavar)

    def text(self):
        mv = f" {self.metavar}" if self.metavar else ""
        return f"{', '.join(self.names)}{mv}  {' '.join(self.desc.split())}"

    def env_names(self):
        return re.findall(r"env:\s*([A-Z][A-Z0-9_]*)", self.desc)

    def to_json(self):
        return {"names": self.names, "metavar": self.metavar, "desc": " ".join(self.desc.split())[:200]}


NAME = r"-{1,2}[A-Za-z0-9][A-Za-z0-9_.-]*"
PIECE = re.compile(rf"^\s*({NAME})(?:[ =]+(<[^>]*>|\[[^\]]*\]|\{{[^}}]*\}}|[A-Z][A-Z0-9_|.-]*(?:\.\.\.)?))?\s*$")


def split_outside_brackets(text: str) -> list[str]:
    """Split on commas that are not inside <..>, [..] or {..} (a metavar such as <dev1,dev2>)."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<[{":
            depth += 1
        elif ch in ">]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def _parse_left(left: str):
    names, metavar = [], ""
    for piece in split_outside_brackets(left):
        m = PIECE.match(piece)
        if not m:
            return None
        names.append(m.group(1))
        if m.group(2):
            metavar = m.group(2)
    return names, metavar


def parse_flags(text: str) -> list[Flag]:
    """Flags from llama.cpp-, clap-, argparse- and getopt-style help text."""
    flags: list[Flag] = []
    cur = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            cur = None if cur is None else cur
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        parsed = None
        if re.match(r"-{1,2}[A-Za-z0-9]", body) and indent <= 10:
            # left part ends at the first run of 2+ spaces (description follows)
            parts = re.split(r"\s{2,}", body, maxsplit=1)
            left = parts[0]
            # "-c,    --ctx-size N   desc": commas followed by spaces belong to the left part
            if len(parts) == 2 and left.endswith(","):
                more = re.split(r"\s{2,}", parts[1], maxsplit=1)
                left = left + " " + more[0]
                parts = [left, more[1] if len(more) > 1 else ""]
            parsed = _parse_left(left)
            if parsed is None and "<" in left:
                # metavar containing commas or spaces inside <...>
                m = re.match(r"^(.*?<[^>]*>)(?:\s{2,}(.*))?$", body)
                if m:
                    parsed = _parse_left(m.group(1))
                    parts = [m.group(1), m.group(2) or ""]
        if parsed:
            names, metavar = parsed
            desc = parts[1] if len(parts) > 1 else ""
            cur = Flag(names, metavar, desc, indent)
            flags.append(cur)
        elif cur is not None and indent > cur.indent:
            cur.desc += " " + body
        else:
            cur = None
    return flags


# ---------------------------------------------------------------- role scoring

# role -> (positive name rules, description regex, forbidden-name regex, wants a value)
ROLES = {
    "port": {
        "exact": {"--port": 100, "-p": 60, "--listen-port": 90, "--http-port": 90, "--bind-port": 85, "--server-port": 85},
        "name_re": r"port$", "name_score": 70,
        "desc_re": r"\b(listen|bind|serve)\w*\b.*\bport\b|\bport\b.*\b(listen|bind|serve|http)", "desc_score": 35,
        "forbid": r"rpc|metric|ssl|proxy|draft|slot|ws|websocket", "what": "the TCP port the HTTP server listens on",
    },
    "host": {
        "exact": {"--host": 100, "--bind": 70, "--listen": 70, "--address": 70, "--listen-address": 85, "--bind-address": 85},
        "name_re": r"(^|-)host$", "name_score": 80,
        "desc_re": r"\b(ip address|host name|hostname|address to (bind|listen))", "desc_score": 35,
        "forbid": r"rpc|proxy|draft|ssl", "what": "the address the HTTP server binds to",
    },
    "model": {
        "exact": {"--model": 100, "-m": 95, "--model-path": 95, "--weights": 75, "--checkpoint": 70, "--gguf": 75,
                  "--model-file": 90},
        "name_re": r"(^|-)model(-path|-file|-dir)?$", "name_score": 80,
        "desc_re": r"\bmodel\b.*\b(path|file|directory|folder|gguf|weights)\b|\b(path|file|directory)\b.*\bmodel\b|\bweights\b",
        "desc_score": 40,
        "forbid": r"draft|mmproj|vocab|alias|url|hf|id|lora|control|override|embd|tensor|spec|reasoning|kv|offload|default|name",
        "what": "the path of the model file or folder to load",
    },
    "ctx": {
        "exact": {"--ctx-size": 100, "-c": 95, "--ctx": 95, "--n-ctx": 90, "--context-size": 95, "--context-length": 95,
                  "--max-model-len": 85, "--max-seq-len": 80, "--context": 85, "--num-ctx": 90},
        "name_re": r"(^|-)(ctx|context)(-size|-length|-len)?$", "name_score": 75,
        "desc_re": r"\b(context (size|length|window)|prompt context|max(imum)? context|sequence length)\b", "desc_score": 35,
        "forbid": r"draft|batch|shift|checkpoint|cache|swa|slot|reuse|keep|predict|new|generated|tokens$",
        "what": "the context window size, in tokens (the largest prompt plus answer)",
    },
    "device": {
        "exact": {"--device": 100, "-dev": 100, "--gpu": 85, "--device-id": 85, "--gpu-id": 85, "--cuda-device": 85,
                  "--main-gpu": 55, "-mg": 55, "--gpus": 80},
        "name_re": r"(^|-)(device|gpu|gpus)(-id|-index)?$", "name_score": 70,
        "desc_re": r"\b(which|select|choose|use)\b.*\b(gpu|device|card)\b|\b(gpu|device) (index|id|name|to use)\b",
        "desc_score": 30,
        "forbid": r"layer|budget|dense|(^|-)mb$|memory|split|offload|draft|override|-mb|-gb|ngl|tensor|fit|list|placement|w2w|host|mmdev",
        "what": "which GPU or device to run on (a value such as 0 or CUDA0 chooses the card)",
    },
    "threads": {
        "exact": {"--threads": 100, "-t": 95, "--n-threads": 95, "--num-threads": 95, "--cpu-threads": 90},
        "name_re": r"(^|-)threads$", "name_score": 70,
        "desc_re": r"\b(number of )?(cpu |compute )?threads\b", "desc_score": 30,
        "forbid": r"batch|http|draft|prompt|decode|-tb$",
        "what": "the number of CPU threads used for compute",
    },
    "api_key": {
        "exact": {"--api-key": 100, "--apikey": 95, "--key": 40},
        "name_re": r"api-?key$", "name_score": 85,
        "desc_re": r"\bapi key\b", "desc_score": 35,
        "forbid": r"file|ssl|cert",
        "what": "the API key clients must send",
    },
}
SOLID = 80   # a top score at or above this, clear of the runner-up, is decided without a helper
MARGIN = 25
WEAK = 20    # candidates at or above this go to the helper when nothing is solid


def score_flag(role: str, flag: Flag) -> int:
    r = ROLES[role]
    if not flag.takes_value:
        return 0
    best = 0
    if any(re.search(r["forbid"], n.lower().lstrip("-")) for n in flag.names):
        return 0
    for n in flag.names:
        low = n if not n.startswith("--") else n.lower()  # a short name is case-sensitive: -c is not -C
        s = r["exact"].get(low, 0)
        if not s and n.startswith("--") and re.search(r["name_re"], low.lstrip("-")):
            s = r["name_score"]
        best = max(best, s)
    if best == 0 and re.search(r["desc_re"], " ".join(flag.desc.split()), re.I):
        best = r["desc_score"]
    return best


def rank(role: str, flags: list[Flag]):
    scored = [(score_flag(role, f), f) for f in flags]
    scored = [(s, f) for s, f in scored if s > 0]
    scored.sort(key=lambda t: -t[0])
    return scored


def decide(role: str, flags: list[Flag]):
    """-> (flag or None, status, candidates). status: decided | ambiguous | none."""
    ranked = rank(role, flags)
    if not ranked:
        return None, "none", []
    top = ranked[0]
    sole = len(ranked) == 1 and top[0] >= 60
    if sole or (top[0] >= SOLID and (len(ranked) == 1 or ranked[1][0] <= top[0] - MARGIN)):
        return top[1], "decided", [f for _, f in ranked]
    cands = [f for s, f in ranked if s >= WEAK][:8]
    return None, "ambiguous", cands


def metrics_flag(flags: list[Flag]):
    for f in flags:
        if not f.takes_value and any(re.fullmatch(r"-{1,2}metrics?", n) for n in f.names):
            return f
    return None


def preferred_name(flag: Flag) -> str:
    """The long name when there is one (clearer in a definition), else the first."""
    longs = [n for n in flag.names if n.startswith("--")]
    return longs[0] if longs else flag.names[0]


# ---------------------------------------------------------------- helper (small local model)


def has_flag_text(help_text: str, flag: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def norm(s: str) -> str:
    return " ".join(s.split())


def validate_answer(role: str, answer, help_text: str, flags: list[Flag]):
    """-> (flag name or None, reason). The helper's word is checked against the help text, never trusted."""
    if not isinstance(answer, dict) or "flag" not in answer:
        return None, "answer is not an object with a 'flag' member"
    flag = answer.get("flag")
    if flag is None:
        return None, "helper says no such flag"
    # a model often writes the flag with its metavar ("--window N"): keep the flag, drop one trailing metavar
    m = re.fullmatch(rf"\s*({NAME})(?:[ =]\S+)?\s*", flag) if isinstance(flag, str) else None
    if not m:
        return None, f"{flag!r} is not a flag name"
    flag = m.group(1)
    if not has_flag_text(help_text, flag):
        return None, f"{flag} is not written in the --help text"
    evidence = answer.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return None, "no quoted evidence line"
    if norm(evidence) not in norm(help_text) or not has_flag_text(evidence, flag):
        return None, "the quoted evidence is not a line of the --help text that names the flag"
    if re.search(ROLES[role]["forbid"], flag.lstrip("-").lower()):
        return None, f"{flag} is on the exclusion list for {role}"
    if flags:
        owner = next((f for f in flags if flag in f.names), None)
        if owner is None:
            return None, f"{flag} is mentioned in the help but is not an option"
        if not owner.takes_value:
            return None, f"{flag} takes no value, so it cannot set {role}"
    return flag, "accepted"


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


class Helper:
    def __init__(self, urls, model, log_path: Path, timeout=120):
        self.urls = urls
        self.model = model
        self.log_path = log_path
        self.timeout = timeout
        self.calls = []

    def ask(self, role: str, help_text: str, flags: list[Flag], candidates: list[Flag]):
        if candidates:
            shown = "\n".join("  " + c.text() for c in candidates)
            context = f"These option lines are the closest candidates:\n{shown}\n"
        else:
            context = "No option line matched closely. The help text is:\n" + help_text[:8000] + "\n"
        question = (
            f"A server program's --help lists its options. Which one option sets {ROLES[role]['what']}?\n{context}"
            'Answer with ONE JSON object and nothing else: {"flag": "<the flag exactly as written, for example --foo>", '
            '"evidence": "<one line copied exactly from the help text that names this flag>"}. '
            'If no option does this, answer {"flag": null, "evidence": ""}.'
        )
        entry = {"role": role, "question": question, "answer_raw": None, "accepted": False, "flag": None, "reason": ""}
        raw = None
        for url in self.urls:
            try:
                body = {
                    "model": self.model, "temperature": 0, "max_tokens": 220,
                    "messages": [{"role": "user", "content": question}],
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                req = urllib.request.Request(
                    url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                    {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = json.loads(r.read())["choices"][0]["message"]["content"]
                entry["url"] = url
                break
            except Exception as e:  # noqa: BLE001
                entry["reason"] = f"helper at {url} failed: {e}"
        entry["answer_raw"] = raw
        if raw is not None:
            flag, reason = validate_answer(role, extract_json(raw), help_text, flags)
            entry.update(accepted=flag is not None, flag=flag, reason=reason)
        self.calls.append(entry)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        return next((f for f in flags if entry["flag"] in f.names), None) if entry["accepted"] else None


# ---------------------------------------------------------------- running things


def free_port(avoid=()):
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        if p not in avoid:
            return p


def http(method, url, body=None, timeout=60, stream=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if stream:
                return r.status, r.read().decode("utf-8", "replace")
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


class Proc:
    """A child in its own process group, stopped by PID, never by name."""

    def __init__(self, argv, env, log: Path, cpus: str | None):
        if cpus:
            argv = ["taskset", "-c", cpus] + argv
        self.logf = log.open("wb")
        self.p = subprocess.Popen(argv, env=env, stdout=self.logf, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        self.log = log

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.alive():
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
                self.p.wait(timeout=20)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        self.logf.close()

    def tail(self, n=2000):
        try:
            return self.log.read_text(errors="replace")[-n:]
        except OSError:
            return ""


def run_help(binary: str, args: list[str], env):
    try:
        r = subprocess.run([binary] + args, env=env, capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout + ("\n" + r.stderr if r.stderr else ""))
    except Exception as e:  # noqa: BLE001
        return 127, str(e)


def probe_help(binary: str, base_env: dict):
    """-> dict(help, version, env, library_dir, flags)."""
    out = {"library_dir": False, "env": {}}
    env = dict(base_env)
    code, text = run_help(binary, ["--help"], env)
    if "error while loading shared libraries" in text and "version `" not in text:
        d = str(Path(binary).resolve().parent)
        libs = [d] + ([str(Path(d).parent / "lib")] if (Path(d).parent / "lib").is_dir() else [])
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
        out["library_dir"] = True
        out["env"]["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]
        code, text = run_help(binary, ["--help"], env)
    flags = parse_flags(text)
    if len(flags) < 3:
        code2, text2 = run_help(binary, ["-h"], env)
        flags2 = parse_flags(text2)
        if len(flags2) > len(flags):
            text, flags = text2, flags2
    vcode, ver = run_help(binary, ["--version"], env)
    ver = ver.strip().splitlines()[0] if vcode == 0 and ver.strip() else ""
    out.update(help=text, flags=flags, version=ver, env_final=env)
    return out


def openai_probe(base: str, log):
    """Calls the OpenAI-compatible surface; returns what answered."""
    res = {}
    for path in ("/health", "/v1/models"):
        code, body = http("GET", base + path, timeout=30)
        res[path] = {"status": code}
        if path == "/health":
            try:
                res[path]["json_status"] = json.loads(body).get("status")
            except Exception:  # noqa: BLE001
                pass
        if path == "/v1/models" and code == 200:
            try:
                data = json.loads(body)["data"]
                res["model_id"] = data[0].get("id") if data else None
                res["owned_by"] = data[0].get("owned_by") if data else None
            except Exception:  # noqa: BLE001
                res[path]["status"] = f"{code} but not a model list"
    mid = res.get("model_id") or "model"
    msg = [{"role": "user", "content": "Say hello in three words."}]
    code, body = http("POST", base + "/v1/chat/completions",
                      {"model": mid, "messages": msg, "max_tokens": 24, "temperature": 0}, timeout=300)
    ok = False
    if code == 200:
        try:
            ok = bool(json.loads(body)["choices"][0]["message"].get("content") is not None)
        except Exception:  # noqa: BLE001
            ok = False
    res["chat"] = {"status": code, "ok": ok, "sample": (json.loads(body)["choices"][0]["message"].get("content") or "")[:60] if ok else body[:120]}
    code, body = http("POST", base + "/v1/chat/completions",
                      {"model": mid, "messages": msg, "max_tokens": 24, "temperature": 0, "stream": True},
                      timeout=300, stream=True)
    res["stream"] = {"status": code, "ok": code == 200 and "data:" in body and "[DONE]" in body}
    code, body = http("POST", base + "/v1/completions",
                      {"model": mid, "prompt": "Hello", "max_tokens": 8, "temperature": 0}, timeout=300)
    res["completion"] = {"status": code, "ok": code == 200 and '"choices"' in body}
    code, body = http("GET", base + "/metrics", timeout=30)
    res["metrics"] = {"status": code, "ok": code == 200 and "# TYPE" in body}
    return res


# ---------------------------------------------------------------- spec and definition


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_spec(name, binary, roles: dict, mflag, probe: dict, info: dict, model_path: Path, ctx: str,
               timeout_s: int):
    args = []
    if roles.get("model"):
        args += [preferred_name(roles["model"]), "{model}"]
    if roles.get("host"):
        args += [preferred_name(roles["host"]), "127.0.0.1"]
    if roles.get("port"):
        args += [preferred_name(roles["port"]), "{port}"]
    if roles.get("ctx"):
        args += [preferred_name(roles["ctx"]), ctx]
    if mflag and info["metrics"]["ok"]:
        args += [preferred_name(mflag)]
    if roles.get("device"):
        args += [preferred_name(roles["device"]), "{device}"]
    ready = {"path": "/health" if info["/health"]["status"] == 200 else "/v1/models", "timeout_s": timeout_s}
    if ready["path"] == "/health" and info["/health"].get("json_status"):
        ready.update(json_field="status", ready_values=[info["/health"]["json_status"]])
    owned = info.get("owned_by")
    spec = {
        "recipe": name,
        "stem": re.sub(r"[^a-z0-9]", "", name),
        "class_prefix": "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", name) if p) or "Engine",
        "display_name": f"{Path(binary).name} (adapted automatically)",
        "binary": Path(binary).name,
        "spec_version": 1,
        "args": args,
        "default_device": "cpu",
        "support": [{"os": ["linux"], "devices": {"cpu": [platform.machine()]},
                     "note": "probed and graded on CPU; other devices were not tried"}],
        "modes": ["chat"],
        "checkpoint": {"kind": "dir" if model_path.is_dir() else "file"},
        "ready": ready,
        "identity": {"owned_by": owned} if owned else {"owned_by": "unknown", "unreported": True},
        "completion": bool(info["completion"]["ok"]),
        "notes": [f"Generated by tools/adapt_engine.py from the engine's own --help; engine reports: {probe['version']!r}."],
    }
    if info["metrics"]["ok"]:
        spec["prometheus_metrics"] = True
    if probe["library_dir"]:
        spec["library_dir"] = "binary"
    if not owned:
        spec["identity"] = {"owned_by": "unknown", "unreported": True}
    return spec


# ---------------------------------------------------------------- conformance loop


class Rig:
    """A throwaway lemond root with one definition, started from the fork's build."""

    def __init__(self, work: Path, a, spec, binary_path, model_path, other_path, env_cpu):
        self.work, self.a, self.binary_path = work, a, binary_path
        self.model_path, self.other_path, self.env_cpu = model_path, other_path, env_cpu
        self.spec = spec

    def prepare(self, spec):
        self.spec = spec
        return self.install(tbd.convert(spec), spec["recipe"])

    def install(self, definition, recipe):
        self.recipe = recipe
        for d in ("config/backends.d", "cache", "logs"):
            (self.work / d).mkdir(parents=True, exist_ok=True)
        for old in (self.work / "config" / "backends.d").glob("*.json"):
            old.unlink()
        defpath = self.work / "config" / "backends.d" / f"{recipe}.json"
        defpath.write_text(tbd.render(definition))
        section = definition["metadata"]["config_section"]
        cfg = {
            "auto_check_model_updates": False, "broadcast": False, "config_version": 2,
            "disable_model_filtering": True, "inhibit_suspend": False, "offline": True, "log_level": "debug",
            # the engine path is <config_section>.<backend>_bin, one key per backend the definition lists
            section: {f"{b}_bin": str(self.binary_path)
                      for b in definition["metadata"].get("bin_variants") or ["cpu"]},
        }
        (self.work / "config" / "config.json").write_text(json.dumps(cfg, indent=1))
        kind_label = ["chat"]
        models = {}
        for suffix, path in (("a", self.model_path), ("b", self.other_path)):
            models[f"{recipe}-{suffix}"] = {
                "recipe": recipe, "checkpoint": str(path), "source": "local_path", "labels": kind_label}
        (self.work / "config" / "user_models.json").write_text(json.dumps(models, indent=1))
        return defpath

    def validate(self, defpath):
        r = subprocess.run([self.a.lemonade, "backends", "validate", "--offline", str(defpath)],
                           capture_output=True, text=True, timeout=60)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]

    def grade(self, defpath, port):
        log = self.work / "logs" / "lemond.log"
        env = dict(os.environ)
        env.update(self.env_cpu)
        lem = Proc([self.a.lemond, "--port", str(port), "--host", "127.0.0.1",
                    str(self.work / "cache"), str(self.work / "config")], env, log, self.a.cpus)
        try:
            deadline = time.time() + 90
            while time.time() < deadline and lem.alive():
                if http("GET", f"http://127.0.0.1:{port}/api/v1/health", timeout=5)[0] not in (0,):
                    break
                time.sleep(1)
            if not lem.alive():
                return {"earned": 0, "levels": {"L1": {"status": "FAIL", "detail": "lemond did not start: " + lem.tail(300)}},
                        "log": lem.tail(4000)}
            cmd = [sys.executable, self.a.conformance, "--lemond", f"http://127.0.0.1:{port}",
                   "--definition", str(defpath), "--backends-d", str(self.work / "config" / "backends.d"),
                   "--model", f"{self.recipe}-a", "--other", f"{self.recipe}-b",
                   "--cpu-only", "--json"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.a.grade_timeout)
            try:
                rep = json.loads(r.stdout)
            except ValueError:
                rep = {"earned": 0, "levels": {"L1": {"status": "FAIL", "detail": (r.stdout + r.stderr)[-400:]}}}
            rep["log"] = log.read_text(errors="replace")[-6000:] if log.exists() else ""
            return rep
        finally:
            lem.stop()


def remove_flag(spec, flag_name):
    args, out, i = spec["args"], [], 0
    while i < len(args):
        if args[i] == flag_name:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("-") else 1
            continue
        out.append(args[i])
        i += 1
    new = dict(spec)
    new["args"] = out
    return new


def propose_fix(spec, rep, helper_flags: set[str], tried: set[str]):
    """One rule-based repair from the failure; returns (new spec, what) or None."""
    levels = rep.get("levels", {})
    l1 = levels.get("L1", {})
    log = rep.get("log", "")
    if l1.get("status") == "FAIL":
        m = re.search(r"(?:unrecognized|unknown|unexpected|invalid) (?:option|argument|flag)[:'` ]+(-{1,2}[\w-]+)", log, re.I)
        if m and m.group(1) in spec["args"] and ("rm:" + m.group(1)) not in tried:
            return remove_flag(spec, m.group(1)), f"engine rejected {m.group(1)}: removed it", "rm:" + m.group(1)
        if "error while loading shared libraries" in log and spec.get("library_dir") != "binary" and "libdir" not in tried:
            new = dict(spec)
            new["library_dir"] = "binary"
            return new, "engine could not find its libraries: library_dir = binary", "libdir"
        mine = [f for f in helper_flags if f in spec["args"] and ("rm:" + f) not in tried]
        if mine:
            return remove_flag(spec, mine[0]), f"load failed with the helper's flag {mine[0]}: removed it", "rm:" + mine[0]
        if re.search(r"not ready|timed out|timeout", l1.get("detail", "") + log, re.I) and spec["ready"]["timeout_s"] < 1800 \
                and "timeout" not in tried:
            new = json.loads(json.dumps(spec))
            new["ready"]["timeout_s"] = min(1800, spec["ready"]["timeout_s"] * 3)
            return new, f"readiness timed out: timeout {new['ready']['timeout_s']} s", "timeout"
        other = "/v1/models" if spec["ready"]["path"] == "/health" else "/health"
        if re.search(r"404|not found|readiness", log, re.I) and "path" not in tried:
            new = json.loads(json.dumps(spec))
            new["ready"]["path"] = other
            new["ready"].pop("json_field", None)
            new["ready"].pop("ready_values", None)
            return new, f"readiness path failed: trying {other}", "path"
    return None


def level_table(rep):
    rows = []
    for lv in ("L1", "L2", "L3", "L4"):
        r = rep.get("levels", {}).get(lv, {"status": "not run", "detail": ""})
        rows.append((lv, r["status"], r["detail"]))
    return rows


# ---------------------------------------------------------------- main


def say(msg=""):
    print(msg, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="the engine's server executable")
    ap.add_argument("--model", required=True, help="a small model file or folder the engine can load")
    ap.add_argument("--other-model", help="a second model for the eviction check (default: the same one again)")
    ap.add_argument("--name", help="recipe name (default: derived from the binary)")
    ap.add_argument("--out", default="adapted", help="folder for the spec, definition and report")
    ap.add_argument("--work", help="throwaway lemond root (default: <out>/<name>/rig)")
    ap.add_argument("--lemond", help="lemond built from the fork branch")
    ap.add_argument("--lemonade", help="the lemonade CLI from the same build")
    ap.add_argument("--conformance", help="tools/backend_conformance.py of the fork branch")
    ap.add_argument("--helper-url", action="append", default=[], help="OpenAI-compatible base URL; repeatable (fallbacks)")
    ap.add_argument("--helper-model", default="gemma-4-26b-a4b-it")
    ap.add_argument("--no-helper", action="store_true", help="never call a model")
    ap.add_argument("--cpus", default=None, help="CPU list for every process started (taskset -c), e.g. 40-47")
    ap.add_argument("--allow-gpu", action="store_true", help="do not hide GPUs from the engine (default: hidden)")
    ap.add_argument("--ctx", default="2048", help="context size used when a context flag is found")
    ap.add_argument("--start-timeout", type=int, default=900, help="seconds to wait for the engine to answer")
    ap.add_argument("--grade-timeout", type=int, default=3600)
    ap.add_argument("--rounds", type=int, default=4, help="most conformance runs")
    ap.add_argument("--compare-with", help="a hand-written definition (backends.d/*.json) to grade in the same rig, "
                    "same model and CPUs, for a like-for-like comparison")
    ap.add_argument("--probe-only", action="store_true", help="stop after writing the definition")
    ap.add_argument("--port", type=int, default=0, help="port for the throwaway lemond (default: free)")
    a = ap.parse_args(argv)

    binary = str(Path(a.binary).resolve())
    model_path = Path(a.model).resolve()
    other_path = Path(a.other_model).resolve() if a.other_model else model_path
    name = slug(a.name or Path(binary).name)
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) if a.work else out / "rig"
    helper_log = out / "helper-calls.jsonl"
    helper_log.write_text("")
    env_cpu = {} if a.allow_gpu else dict(CPU_ONLY_ENV)
    base_env = dict(os.environ)
    base_env.update(env_cpu)
    report = {"name": name, "binary": binary, "model": str(model_path)}

    say(f"== adapt-engine: {Path(binary).name} with {model_path.name}")
    # 1. probe the help
    probe = probe_help(binary, base_env)
    flags = probe["flags"]
    say(f"[probe] version: {probe['version'] or 'not reported'}; {len(flags)} options parsed"
        f"{'; engine needs its own folder on the library path' if probe['library_dir'] else ''}")
    if not flags and not probe["help"].strip():
        say("[probe] the engine printed no help: nothing to adapt from")
        return 2
    helper = None if a.no_helper or not a.helper_url else Helper(a.helper_url, a.helper_model, helper_log)
    roles, status = {}, {}
    for role in ROLES:
        flag, st, cands = decide(role, flags)
        how = "heuristic"
        if st != "decided":
            if helper and role in ("port", "model", "ctx", "device", "host"):
                flag = helper.ask(role, probe["help"], flags, cands)
                last = helper.calls[-1]
                how = "helper, accepted" if flag else f"helper asked, not used: {last['reason']}"
            else:
                how = "left out"
        if flag:
            roles[role] = flag
            status[role] = how
            say(f"[probe] {role:8} -> {preferred_name(flag)}   ({how})")
        else:
            status[role] = how
            say(f"[probe] {role:8} -> none   ({how})")
    mflag = metrics_flag(flags)
    if mflag:
        say(f"[probe] metrics  -> {preferred_name(mflag)}   (a flag that turns the endpoint on)")
    report["roles"] = {r: preferred_name(f) for r, f in roles.items()}
    report["role_status"] = status
    if "model" not in roles or "port" not in roles:
        say("[probe] no model flag or no port flag: cannot launch the engine. "
            + ("" if helper else "(run with --helper-url to let a small model read the help)"))
        report["result"] = "probe failed"
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return 2

    # 2. launch and call it
    port = free_port()
    eargs = [binary, preferred_name(roles["model"]), str(model_path), preferred_name(roles["port"]), str(port)]
    if "host" in roles:
        eargs += [preferred_name(roles["host"]), "127.0.0.1"]
    if "ctx" in roles:
        eargs += [preferred_name(roles["ctx"]), a.ctx]
    if mflag:
        eargs += [preferred_name(mflag)]
    say(f"[launch] {Path(binary).name} on port {port}"
        f"{' (CPU only, GPUs hidden)' if env_cpu else ''}{' limited to CPUs ' + a.cpus if a.cpus else ''}")
    penv = dict(probe["env_final"])
    eng = Proc(eargs, penv, out / "engine-probe.log", a.cpus)
    info = None
    try:
        base = f"http://127.0.0.1:{port}"
        t0 = time.time()
        up = False
        while time.time() - t0 < a.start_timeout and eng.alive():
            if http("GET", base + "/health", timeout=5)[0] == 200 or http("GET", base + "/v1/models", timeout=5)[0] == 200:
                up = True
                break
            time.sleep(1)
        if not up:
            say(f"[launch] the engine did not answer ({'exited' if not eng.alive() else 'timed out'}); last output:")
            say("    " + eng.tail(500).replace("\n", "\n    "))
            report["result"] = "engine did not start"
            (out / "report.json").write_text(json.dumps(report, indent=2))
            return 2
        say(f"[launch] answering after {time.time() - t0:.0f} s")
        info = openai_probe(base, say)
    finally:
        eng.stop()
    for label, key in (("/health", "/health"), ("/v1/models", "/v1/models")):
        say(f"[http]   GET  {label:22} -> {info[key]['status']}"
            f"{' status=' + str(info[key].get('json_status')) if info[key].get('json_status') else ''}")
    say(f"[http]   POST /v1/chat/completions   -> {info['chat']['status']} {'ok' if info['chat']['ok'] else 'FAILED'}"
        f"{'  ' + repr(info['chat']['sample']) if info['chat']['ok'] else ''}")
    say(f"[http]   POST ... stream=true        -> {info['stream']['status']} {'ok' if info['stream']['ok'] else 'FAILED'}")
    say(f"[http]   POST /v1/completions        -> {info['completion']['status']} {'ok' if info['completion']['ok'] else 'not served'}")
    say(f"[http]   GET  /metrics               -> {info['metrics']['status']} {'prometheus text' if info['metrics']['ok'] else 'none'}")
    report["http"] = info
    if not info["chat"]["ok"]:
        say("[http] /v1/chat/completions does not work: this is not an OpenAI-compatible server, nothing to adapt")
        report["result"] = "not OpenAI compatible"
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return 2

    # 3. spec and definition
    spec = build_spec(name, binary, roles, mflag, probe, info, model_path, a.ctx, 600)
    (out / "engines").mkdir(exist_ok=True)
    (out / "backends.d").mkdir(exist_ok=True)
    (out / "engines" / f"{name}.json").write_text(json.dumps(spec, indent=2) + "\n")
    definition_text = tbd.render(tbd.convert(spec))
    (out / "backends.d" / f"{name}.json").write_text(definition_text)
    say(f"[write]  engine spec        {out / 'engines' / (name + '.json')}")
    say(f"[write]  backend definition {out / 'backends.d' / (name + '.json')}   args: {' '.join(spec['args'])}")
    helper_flags = {preferred_name(roles[r]) for r in roles if status[r].startswith("helper")}
    report["spec"] = spec
    if a.probe_only or not (a.lemond and a.lemonade and a.conformance):
        report["result"] = "definition written; conformance not run"
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return 0

    # 4. conformance loop
    rig = Rig(work, a, spec, Path(binary), model_path, other_path, env_cpu)
    best, best_rep, best_spec, tried, history = -1, None, spec, set(), []
    cur = spec
    for rnd in range(1, a.rounds + 1):
        defpath = rig.prepare(cur)
        ok, msg = rig.validate(defpath)
        say(f"[conform] round {rnd}: lemonade backends validate --offline: {'valid' if ok else 'INVALID ' + msg}")
        if not ok:
            history.append({"round": rnd, "valid": False, "detail": msg})
            break
        port = a.port or free_port(avoid={13305, 13306, 13390})
        say(f"[conform] round {rnd}: lemond on :{port}, grading (L1 lifecycle, L2 eviction, L3 device, L4 telemetry)")
        t0 = time.time()
        rep = rig.grade(defpath, port)
        earned = rep.get("earned", 0)
        say(f"[conform] round {rnd}: earned L{earned} in {time.time() - t0:.0f} s")
        for lv, st, detail in level_table(rep):
            say(f"           {lv}: {st:12} {detail[:150]}")
        history.append({"round": rnd, "earned": earned, "levels": rep.get("levels", {})})
        if earned > best:
            best, best_rep, best_spec = earned, rep, cur
        elif earned < best:
            say("[conform] the repair made it worse: keeping the earlier definition")
            break
        fix = propose_fix(cur, rep, helper_flags, tried)
        if not fix:
            break
        cur, what, key = fix
        tried.add(key)
        say(f"[repair] {what}")
    (out / "backends.d" / f"{name}.json").write_text(tbd.render(tbd.convert(best_spec)))
    (out / "engines" / f"{name}.json").write_text(json.dumps(best_spec, indent=2) + "\n")
    report.update(spec=best_spec, history=history, earned=max(best, 0))
    report["helper_calls"] = [{k: c[k] for k in ("role", "answer_raw", "accepted", "flag", "reason")}
                              for c in (helper.calls if helper else [])]
    report["result"] = f"L{max(best, 0)}"
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # like-for-like comparison with a hand-written definition
    if a.compare_with:
        hand = json.loads(Path(a.compare_with).read_text())
        crig = Rig(work.parent / (work.name + "-hand"), a, spec, Path(binary), model_path, other_path, env_cpu)
        hpath = crig.install(hand, hand["recipe"])
        ok, msg = crig.validate(hpath)
        if ok:
            say(f"[compare] grading the hand-written {hand['recipe']} in the same rig (same model, same CPUs)")
            hrep = crig.grade(hpath, a.port or free_port(avoid={13305, 13306, 13390}))
            report["hand_written"] = {"definition": hand["recipe"], "earned": hrep.get("earned", 0),
                                      "levels": hrep.get("levels", {})}
            say(f"[compare] hand-written {hand['recipe']}: earned L{hrep.get('earned', 0)}")
            for lv, st, detail in level_table(hrep):
                say(f"           {lv}: {st:12} {detail[:150]}")
        else:
            say(f"[compare] the hand-written definition is invalid here: {msg}")
    # final table
    say()
    say(f"== result for {name}: level L{max(best, 0)} (L0 = provider only, no control)")
    if best_rep:
        for lv, st, detail in level_table(best_rep):
            say(f"  {lv}  {st:12} {detail[:200]}")
    if best_rep:
        lv = best_rep.get("levels", {})
        beyond = [k for k in ("L2", "L3", "L4") if lv.get(k, {}).get("status") == "pass" and int(k[1]) > max(best, 0)]
        if beyond:
            say(f"  note: {', '.join(beyond)} {'passes' if len(beyond) == 1 else 'pass'} on its own but is not counted: "
                "a level counts only when every level below it passes")
    calls = helper.calls if helper else []
    def outcome(c):
        if c["accepted"]:
            return "accepted"
        return "answered: no such flag" if c["reason"].startswith("helper says no") else "rejected, " + c["reason"]
    n_acc = sum(1 for c in calls if c["accepted"])
    n_none = sum(1 for c in calls if outcome(c).startswith("answered"))
    say(f"== helper: {len(calls)} call(s): {n_acc} accepted, {n_none} answered 'no such flag', "
        f"{len(calls) - n_acc - n_none} rejected{'' if helper else ' (no helper used)'}")
    for c in calls:
        say(f"   {c['role']:8} -> {c['flag'] or 'none':14} {outcome(c)}")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    say(f"== written: {out / 'backends.d' / (name + '.json')}")
    say(f"   add to lemond's config.json: \"{tbd.option_prefix(best_spec)}\": {{\"cpu_bin\": \"{binary}\"}}")
    return 0 if best >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
