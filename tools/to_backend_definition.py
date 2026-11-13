#!/usr/bin/env python3
"""Turn an engine spec (engines/*.json) into a Lemonade backend definition.

The output is a hook-free schema-v1 definition for `<config_dir>/backends.d/`,
the external-definition format of lemonade-sdk/lemonade#3906. It asks for no
hook and no compiled code, so lemond runs it without a rebuild.

    to_backend_definition.py engines/lemonade-burn.json          # print one
    to_backend_definition.py --all --out backends.d              # write all
    to_backend_definition.py --all --check backends.d            # fail if stale

What a spec says that a definition cannot (reported with --gaps, never guessed):
the ready probe's JSON field, the identity check on /v1/models, a pinned binary
sha256, and the telemetry / memory / budget paths.

Standard library only.
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# spec device family -> the backend name a definition's support row carries
BACKEND_OF = {"cpu": "cpu", "nvidia_gpu": "cuda", "amd_gpu": "rocm"}


def option_prefix(spec):
    """Leaf-owned option names: the recipe with '-' and '.' turned into '_'."""
    return re.sub(r"[^a-z0-9_]", "_", spec["recipe"])


def device_option(spec):
    return option_prefix(spec) + "_device"


def uses_device(spec):
    texts = list(spec["args"]) + list((spec.get("env") or {}).values())
    return any("{device}" in t or "{gpu_uuids}" in t for t in texts)


def expression(text, spec):
    """One argv or environment string -> a definition expression."""
    parts, at = [], 0
    for m in PLACEHOLDER.finditer(text):
        if m.start() > at:
            parts.append(text[at:m.start()])
        name = m.group(1)
        if name in ("model", "model_path"):
            parts.append({"ref": "/model/checkpoint", "type": "string"})
        elif name == "port":
            parts.append({"string": {"ref": "/launch/ports/http"}})
        elif name in ("device", "gpu_uuids"):
            parts.append({"string": {"ref": "/options/" + device_option(spec)}})
        elif name == "ctx":
            parts.append({"string": {"ref": "/options/ctx_size"}})
        else:
            raise SystemExit(
                f"{spec['recipe']}: placeholder {{{name}}} has no definition reference"
            )
        at = m.end()
    if at < len(text):
        parts.append(text[at:])
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else {"concat": parts}


def device_in_env(spec):
    return any(
        "{device}" in v or "{gpu_uuids}" in v for v in (spec.get("env") or {}).values()
    )


def argv_records(spec):
    """Fixed arguments in one record; a flag whose value is the device goes in
    its own record that is dropped when the device option is unset."""
    args = spec["args"]
    fixed, conditional, i = [], [], 0
    device_ref = {"ref": "/options/" + device_option(spec)}
    while i < len(args):
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if args[i].startswith("-") and ("{device}" in nxt or "{gpu_uuids}" in nxt):
            conditional.append(
                {
                    "when": {"not": {"eq": [device_ref, ""]}},
                    "args": [args[i], expression(nxt, spec)],
                }
            )
            i += 2
            continue
        fixed.append(expression(args[i], spec))
        i += 1
    return [{"args": fixed}] + conditional


def operations(spec):
    ops = []
    kinds = [("chat_completion", "/v1/chat/completions")]
    if spec.get("completion"):
        kinds.append(("completion", "/v1/completions"))
    for delivery, kind, codec in (
        ("buffered", "http_json", ("response_codec", "json.v1")),
        ("streaming", "http_sse", ("stream_codec", "openai_sse.v1")),
    ):
        for operation, path in kinds:
            ops.append(
                {
                    "operation": operation,
                    "delivery": delivery,
                    "model_modes": spec["modes"],
                    "transport": {
                        "kind": kind,
                        "method": "POST",
                        "path": path,
                        "port": "http",
                        "timeout": "global",
                        codec[0]: codec[1],
                    },
                    "request": {
                        "base": "identity",
                        "remove": [],
                        "set": {},
                        "omit_missing": False,
                    },
                }
            )
    return ops


def support_rows(spec):
    rows = []
    for entry in spec["support"]:
        for family, arches in entry["devices"].items():
            rows.append(
                {
                    "backend": BACKEND_OF[family],
                    "tier": "guest",
                    "format": "native",
                    "supported_os": entry["os"],
                    "devices": {family: arches},
                    "device_summary": entry.get("note", family),
                    "arch_gates": {},
                }
            )
    return rows


def convert(spec):
    recipe = spec["recipe"]
    rows = support_rows(spec)
    backends = []
    for row in rows:
        if row["backend"] not in backends:
            backends.append(row["backend"])
    default = "cpu" if spec["default_device"] == "cpu" else next(
        (b for b in backends if b != "cpu"), backends[0]
    )
    options = []
    if uses_device(spec):
        options.append(
            {
                "name": device_option(spec),
                # An environment variable cannot be left out when the option is
                # unset, so a device that travels in one always has a value.
                "default_value": spec.get("default_device_index", "0")
                if device_in_env(spec)
                else "",
                "type_name": "DEVICES",
                "value_type": "string",
                "help": "GPU this engine runs on (the value the engine's own flag takes)",
                "group": spec["display_name"],
                "unset_values": [None],
            }
        )
    ready = spec.get("ready") or {}
    environment = [
        {"name": name, "value": expression(value, spec)}
        for name, value in (spec.get("env") or {}).items()
    ]
    return {
        "schema_version": 1,
        "id": recipe,
        "recipe": recipe,
        "kind": "local",
        "metadata": {
            "display_name": spec["display_name"],
            "server_name": spec["binary"],
            "log_category": spec["class_prefix"],
            "binary": spec["binary"],
            "config_section": option_prefix(spec),
            "default_device": spec["default_device"],
            "slot_policy": "standard",
            "options": options,
            "support": rows,
            "supported_modes": spec["modes"],
            "required_checkpoints": ["main"],
            "bin_variants": backends,
            **(
                {"exposes_prometheus_metrics": True}
                if spec.get("prometheus_metrics")
                else {}
            ),
        },
        "install": {"strategy": "external_only"},
        "launch": {
            "prepare_order": ["artifacts", "backend", "program", "ports"],
            "backend": {"fixed": default},
            "program": {"source": "external_or_managed", "install_policy": "always"},
            "ports": [
                {
                    "name": "http",
                    "reject_zero": True,
                    "error": f"{recipe}: no free port",
                }
            ],
            "argv": argv_records(spec),
            "environment": {
                "preset": "executable_loader_path"
                if spec.get("library_dir") == "binary"
                else "inherit",
                "set": environment,
            },
            "working_directory": "inherit",
            "output": {"inherit_levels": ["debug"], "filter": False},
            "readiness": {
                "port": "http",
                "path": ready.get("path", "/health"),
                "timeout": int(ready.get("timeout_s", 600)),
            },
            "watchdog": {
                "policy": "standard",
                "port": "http",
                "path": ready.get("path", "/health"),
            },
        },
        "operations": operations(spec),
    }


def gaps(spec):
    """What the spec states that the definition format has no field for."""
    out = []
    ready = spec.get("ready") or {}
    if ready.get("json_field"):
        out.append(
            f"ready probe reads `{ready['json_field']}` "
            f"(ready {ready.get('ready_values')}, loading {ready.get('loading_values', [])}); "
            "a definition's readiness is the HTTP status of the path only"
        )
    identity = spec.get("identity") or {}
    if identity.get("owned_by") and not identity.get("unreported"):
        out.append(
            f"identity check: /v1/models must report owned_by={identity['owned_by']}"
        )
    if spec.get("binary_sha256"):
        out.append("binary sha256 pin")
    for key in ("telemetry_path", "memory_path", "budget_path"):
        if spec.get(key):
            out.append(f"{key} {spec[key]}")
    checkpoint = spec.get("checkpoint") or {}
    if checkpoint.get("kind") == "dir":
        out.append(
            f"checkpoint is a folder holding `{checkpoint.get('marker')}`; "
            "the definition passes the registered path through unchecked"
        )
    return out


def render(definition):
    return json.dumps(definition, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", nargs="*", help="engine spec file(s)")
    parser.add_argument("--all", action="store_true", help="every file in engines/")
    parser.add_argument("--out", help="folder to write <recipe>.json into")
    parser.add_argument("--check", help="folder to compare with; exit 1 when stale")
    parser.add_argument("--gaps", action="store_true", help="list what was left out")
    args = parser.parse_args()

    paths = [pathlib.Path(p) for p in args.spec]
    if args.all:
        paths += sorted((ROOT / "engines").glob("*.json"))
    if not paths:
        parser.error("name a spec or pass --all")

    stale = 0
    for path in paths:
        spec = json.loads(path.read_text())
        text = render(convert(spec))
        name = spec["recipe"] + ".json"
        if args.check:
            target = pathlib.Path(args.check) / name
            if not target.exists() or target.read_text() != text:
                print(f"stale: {target}")
                stale += 1
        elif args.out:
            out = pathlib.Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / name).write_text(text)
            print(f"wrote {out / name}")
        elif not args.gaps:
            sys.stdout.write(text)
        if args.gaps:
            found = gaps(spec)
            print(f"{spec['recipe']}: " + ("nothing left out" if not found else ""))
            for line in found:
                print(f"  - {line}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
