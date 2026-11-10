#!/usr/bin/env python3
"""Validate every engine spec in engines/ and engines/drafts/ against the adapter plane's rules.

Rules come from lemond's adapter_spec.h/.cpp (see SPEC_FORMAT.md). stdlib only.
    python3 -I tools/check_specs.py [spec.json ...]      # default: engines/*.json engines/drafts/*.json
Exit status 1 when any spec has a problem.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_VERSION = 1  # kSpecVersion
PLACEHOLDERS = {"port", "model", "model_path", "device", "gpu_uuids", "ctx", "spec_dir"}
ENV_PREFIXES = ("CUDA_", "HIP_", "ROCR_", "HSA_", "GGML_", "LLAMA_", "OMP_", "MKL_", "VK_",
                "NAUTIBURN_", "STRATA_", "LEMONADE_ENGINE_")
ENV_NAMES = {"LD_LIBRARY_PATH", "TZ"}
# Required by the plane (check_spec) ...
PLANE_REQUIRED = ("binary", "args")
# ... and by this repo's make_adapter (the generator and the descriptor).
REPO_REQUIRED = ("recipe", "stem", "class_prefix", "display_name", "default_device", "support",
                 "modes", "checkpoint", "ready", "identity")
OPTIONAL_TYPES = {"env": dict, "ready": dict, "telemetry": dict, "telemetry_path": str,
                  "memory_path": str, "budget_path": str, "default_device_index": str}
SHELL_META = set("\"'`$|;&<>\\*?()[]!#~\n")
TOKEN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]*$")


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def check_text(label, text, problems, shell=False):
    for name in TOKEN.findall(text):
        if name not in PLACEHOLDERS:
            problems.append(f"{label}: unknown placeholder {{{name}}}")
    if shell:
        bad = sorted(set(text) & SHELL_META)
        if bad:
            problems.append(f"{label}: shell metacharacters {''.join(bad)!r} (no shell runs these)")
        if any(c.isspace() for c in text):
            problems.append(f"{label}: whitespace inside one args element")


def check(spec):
    """Returns (problems, warnings). A problem fails the spec; a warning is printed, exit stays 0."""
    p = []
    w = []
    if not isinstance(spec, dict):
        return ["not a JSON object"], []
    for k in PLANE_REQUIRED + REPO_REQUIRED:
        if k not in spec:
            p.append(f"missing field {k}")
    if "args_template" in spec:
        p.append("args_template present: use args (spec_version 1 args array)")
    b = spec.get("binary")
    if "binary" in spec and (not isinstance(b, str) or not b or "/" in b or " " in b):
        p.append("binary must be a non-empty bare executable name")
    if "spec_version" not in spec:
        p.append("spec_version missing (must be 1)")
    elif not is_int(spec["spec_version"]) or not 1 <= spec["spec_version"] <= SPEC_VERSION:
        p.append(f"spec_version must be an integer in 1..{SPEC_VERSION}")
    a = spec.get("args")
    if "args" in spec:
        if not isinstance(a, list) or not a:
            p.append("args must be a non-empty array")
        else:
            for i, x in enumerate(a):
                if not isinstance(x, str):
                    p.append(f"args[{i}] is not a string")
                else:
                    check_text(f"args[{i}]", x, p, shell=True)
            joined = " ".join(x for x in a if isinstance(x, str))
            if "{port}" not in joined:
                p.append("args lacks {port}")
            if "{model}" not in joined and "{model_path}" not in joined:
                p.append("args lacks {model} or {model_path}")
    for k, t in OPTIONAL_TYPES.items():
        if k in spec and not isinstance(spec[k], t):
            p.append(f"{k} must be {'an object' if t is dict else 'a string'}")
    if "binary_sha256" in spec and not (isinstance(spec["binary_sha256"], str)
                                        and re.fullmatch(r"[0-9a-fA-F]{64}", spec["binary_sha256"])):
        p.append("binary_sha256 must be 64 hex characters")
    env = spec.get("env")
    if isinstance(env, dict):
        for k, v in env.items():
            if not isinstance(v, str):
                p.append(f"env.{k} is not a string")
                continue
            if k not in ENV_NAMES and not k.startswith(ENV_PREFIXES):
                # lemond refuses this at load (build_env). Reported loudly, but the engine reads this
                # variable and the allowlist is lemond's to widen, so it is a warning here.
                w.append(f"env name {k} is not on lemond's allowlist: load will fail until it is added")
            check_text(f"env.{k}", v, p)
            if k == "LD_LIBRARY_PATH":
                for part in filter(None, v.split(":")):
                    if ".." in part or not part.startswith("/"):
                        p.append(f"env.LD_LIBRARY_PATH entry {part!r} must be absolute, no ..")
    r = spec.get("ready")
    if isinstance(r, dict):
        if not PATH_RE.match(r.get("path", "") if isinstance(r.get("path"), str) else ""):
            p.append("ready.path must be an HTTP path like /health")
        if not is_int(r.get("timeout_s")) or not 10 <= r["timeout_s"] <= 7200:
            p.append("ready.timeout_s must be an int in 10..7200")
        if ("json_field" in r) != ("ready_values" in r):
            p.append("ready.json_field and ready.ready_values go together")
        if "json_field" in r and not isinstance(r["json_field"], str):
            p.append("ready.json_field must be a string")
        if "progress_field" in r and not isinstance(r["progress_field"], str):
            p.append("ready.progress_field must be a string")
        for k in ("ready_values", "warm_values", "loading_values", "error_values"):
            if k in r and not (isinstance(r[k], list) and all(isinstance(x, str) for x in r[k])):
                p.append(f"ready.{k} must be an array of strings")
        if r.get("ready_values") == []:
            p.append("ready.ready_values must not be empty")
    return p, w


def main(argv):
    files = [Path(x) for x in argv] or sorted(ROOT.glob("engines/*.json")) + sorted(ROOT.glob("engines/drafts/*.json"))
    rows, bad = [], 0
    for f in files:
        try:
            probs, warns = check(json.loads(f.read_text()))
        except (OSError, ValueError) as e:
            probs, warns = [f"unreadable: {e}"], []
        bad += bool(probs)
        try:
            name = str(f.resolve().relative_to(ROOT))
        except ValueError:
            name = str(f)
        rows.append((name, "FAIL" if probs else "WARN" if warns else "ok", "; ".join(probs + warns) or "-"))
    w = max([len(r[0]) for r in rows] + [4])
    print(f"{'spec'.ljust(w)}  result  problems")
    for n, s, pr in rows:
        print(f"{n.ljust(w)}  {s.ljust(6)}  {pr}")
    print(f"{len(rows) - bad}/{len(rows)} specs pass ({sum(r[1] == 'WARN' for r in rows)} with warnings)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
