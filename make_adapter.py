#!/usr/bin/env python3
"""Adapter maker: turn one engine spec (JSON) into a native Lemonade backend.

Target: our Lemonade fork (iced-lemcake, Lemonade 2026.43.x backend API: BackendDescriptor +
WrappedServer subclass + ServerCommand/start_server). Any engine that serves an OpenAI-compatible
HTTP API can be wrapped: lemond starts the engine binary itself (found in PATH, never faked),
waits for its health route, then for a JSON readiness value when the spec gives one, checks the
engine's identity (`owned_by` in /v1/models) and refuses to serve a mislabelled engine. The engine
then shows in `lemonade backends`, loads and unloads like any other model, and Lemonade's router
can route to it.

Usage:
    make_adapter.py engines/freetoken.json --check                       # validate, list files
    make_adapter.py engines/freetoken.json --out /tmp/preview            # write files only
    make_adapter.py engines/freetoken.json --lemonade-src <fork checkout> # install + CMake line

Spec (see engines/freetoken.json and ADAPTER_PROMPT.md):
    recipe          Lemonade recipe, [a-z0-9-] (the `recipe` in /api/v1/pull)
    stem            C++ folder/namespace, [a-z][a-z0-9_]*
    class_prefix    C++ class prefix, CamelCase (FreeToken -> FreeTokenServer)
    display_name    shown in `lemonade backends`
    binary          engine executable, looked up in PATH at load time
    args_template   engine arguments; placeholders {model} {port} {device}; split on spaces, no shell
    env             optional {"NAME": "value with {device}"} for the engine process
    default_device  "gpu" | "cpu"
    support         [{"os": [...], "devices": {"nvidia_gpu": ["sm_75", ...]}, "note": "..."}]
    modes           e.g. ["chat"]
    checkpoint      {"kind": "file"} | {"kind": "dir", "marker": "<file in dir>" (optional)}
    ready           {"path": "/health", "timeout_s": 900,
                     "json_field": "status", "ready_values": ["ok"], "error_values": ["error"]}
                    (json_* optional: without them, the first answer on `path` means ready)
    identity        {"owned_by": "<engine's owned_by in /v1/models>"}
    completion      optional bool, default true: forward /v1/completions
    telemetry_path  optional GET route copied into Lemonade's telemetry (e.g. "/v1/stats")
    notes           optional list of strings, copied into the descriptor as comments
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RECIPE_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
STEM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CLASS_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]*$")
DEVICE_FAMILIES = {"nvidia_gpu", "amd_gpu", "cpu", "metal", "npu"}
OSES = {"linux", "windows", "macos"}
PLACEHOLDERS = {"model", "port", "device"}
REQUIRED = ("recipe", "stem", "class_prefix", "display_name", "binary", "args_template", "default_device",
            "support", "modes", "checkpoint", "ready", "identity")


class SpecError(ValueError):
    pass


def lit(s: str) -> str:
    """Body of a C++ string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def validate(spec: dict) -> dict:
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise SpecError(f"missing fields: {', '.join(missing)}")
    if not RECIPE_RE.match(spec["recipe"]):
        raise SpecError(f"recipe {spec['recipe']!r}: lowercase letters, digits and '-'")
    if not STEM_RE.match(spec["stem"]):
        raise SpecError(f"stem {spec['stem']!r}: [a-z][a-z0-9_]*")
    if not CLASS_RE.match(spec["class_prefix"]):
        raise SpecError(f"class_prefix {spec['class_prefix']!r}: CamelCase letters and digits")
    if not spec["binary"] or "/" in spec["binary"] or " " in spec["binary"]:
        raise SpecError("binary must be a bare executable name found in PATH (no path, no spaces)")
    tmpl = spec["args_template"]
    for ph in ("{model}", "{port}"):
        if ph not in tmpl:
            raise SpecError(f"args_template must contain {ph}")
    texts = [tmpl] + list((spec.get("env") or {}).values())
    unknown = {p for t in texts for p in re.findall(r"\{([a-z_]+)\}", t)} - PLACEHOLDERS
    if unknown:
        raise SpecError(f"unknown placeholders {sorted(unknown)}; allowed: {sorted(PLACEHOLDERS)}")
    if any(c in tmpl for c in "\"'`$|;&<>"):
        raise SpecError("args_template is split on spaces without a shell: no quotes, $, pipes or redirects")
    for k in (spec.get("env") or {}):
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", k):
            raise SpecError(f"env name {k!r} is not a valid variable name")
    if spec["default_device"] not in ("gpu", "cpu"):
        raise SpecError("default_device must be 'gpu' or 'cpu'")
    if not isinstance(spec["support"], list) or not spec["support"]:
        raise SpecError("support must be a non-empty list")
    for row in spec["support"]:
        if not row.get("os") or not set(row["os"]) <= OSES:
            raise SpecError(f"support row os must be a non-empty subset of {sorted(OSES)}: {row}")
        fams = row.get("devices") or {}
        if not fams or not set(fams) <= DEVICE_FAMILIES:
            raise SpecError(f"support row devices must use {sorted(DEVICE_FAMILIES)}: {row}")
    if not spec["modes"] or not all(isinstance(m, str) and m for m in spec["modes"]):
        raise SpecError("modes must be a non-empty list of strings")
    ck = spec["checkpoint"]
    if ck.get("kind") not in ("file", "dir"):
        raise SpecError("checkpoint.kind must be 'file' or 'dir'")
    if "marker" in ck and ck["kind"] != "dir":
        raise SpecError("checkpoint.marker only applies to kind 'dir'")
    r = spec["ready"]
    if not PATH_RE.match(r.get("path", "")):
        raise SpecError("ready.path must be an HTTP path like /health")
    if not isinstance(r.get("timeout_s"), int) or not 10 <= r["timeout_s"] <= 7200:
        raise SpecError("ready.timeout_s must be an int in 10..7200")
    if ("json_field" in r) != ("ready_values" in r):
        raise SpecError("ready.json_field and ready.ready_values go together")
    if "ready_values" in r and not r["ready_values"]:
        raise SpecError("ready.ready_values must not be empty")
    if not spec["identity"].get("owned_by"):
        raise SpecError("identity.owned_by is required: the engine's owned_by in /v1/models")
    if "telemetry_path" in spec and not PATH_RE.match(spec["telemetry_path"]):
        raise SpecError("telemetry_path must be an HTTP path")
    return spec


def descriptor_h(s: dict) -> str:
    stem = s["stem"]
    rows = []
    for row in s["support"]:
        fams = ", ".join("{" + f'"{fam}", {{' + ", ".join(f'"{lit(a)}"' for a in archs) + "}}"
                         for fam, archs in row["devices"].items())
        oses = ", ".join(f'"{o}"' for o in row["os"])
        rows.append(f'        {{"system", BackendTier::Experimental, BackendFormat::Native, {{{oses}}},\n'
                    f'         {{{fams}}}, "{lit(row.get("note", ""))}"}},')
    notes = "".join(f"// {n}\n" for n in s.get("notes", []))
    modes = ", ".join(f'"{lit(m)}"' for m in s["modes"])
    dev = "DEVICE_GPU" if s["default_device"] == "gpu" else "DEVICE_CPU"
    flag = stem.replace("_", "-")
    return f"""#pragma once
// GENERATED by lemonade-engine-adapters/make_adapter.py from engines/{stem}.json. Edit the spec, not this file.

#include "lemon/backends/backend_descriptor.h"

namespace lemon {{
namespace backends {{
namespace {stem} {{

// {lit(s["display_name"])}: lemond starts `{lit(s["binary"])}` from PATH, waits for {lit(s["ready"]["path"])},
// and refuses to serve unless /v1/models reports owned_by == "{lit(s["identity"]["owned_by"])}".
{notes}inline const BackendDescriptor descriptor = {{
    /*recipe*/          "{s["recipe"]}",
    /*display_name*/    "{lit(s["display_name"])}",
    /*binary*/          "{lit(s["binary"])}",
    /*config_section*/  "{stem}",
    /*default_device*/  {dev},
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,
    /*dynamic_models*/  false,
    /*options*/ {{
        {{"{stem}_args", "--{flag}-args", "", "ARGS",
         "Replaces the engine argument template ({{model}} {{port}} {{device}} placeholders)", "{lit(s["display_name"])} Options"}},
        {{"{stem}_device", "--{flag}-device", "", "DEVICE",
         "Device index substituted for {{device}} (default 0)", "{lit(s["display_name"])} Options"}},
    }},
    /*support*/ {{
{chr(10).join(rows)}
    }},
    /*supported_modes*/ {{{modes}}},
    /*required_checkpoints*/ {{"main"}},
    /*default_capabilities*/ {{}},
    /*web_display_name*/ "",
    /*rocm_channels*/   {{}},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/  VersionPolicy::AtLeast,
    /*self_manages_downloads*/ true,  // the checkpoint is a local {s["checkpoint"]["kind"]} on this machine
}};

}}  // namespace {stem}
}}  // namespace backends
}}  // namespace lemon
"""


def server_h(s: dict) -> str:
    stem, cls = s["stem"], s["class_prefix"] + "Server"
    return f"""#pragma once
// GENERATED by lemonade-engine-adapters/make_adapter.py from engines/{stem}.json. Edit the spec, not this file.

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/wrapped_server.h"

namespace lemon {{
namespace backends {{

class {cls} : public WrappedServer {{
public:
    // No auto-install: the engine binary must be on PATH for lemond.
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    {cls}(const std::string& log_level, ModelManager* model_manager,
          BackendManager* backend_manager);

    void load(const std::string& model_name, const ModelInfo& model_info,
              const RecipeOptions& options, bool do_not_upgrade = false) override;

    json chat_completion(const json& request) override;
    json completion(const json& request) override;

    std::map<std::string, nlohmann::json> get_additional_telemetry() override;
    std::string get_additional_telemetry_url() const override;
}};

namespace {stem} {{
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() {{ return capability_mask_of<{cls}>(); }}
}}  // namespace {stem}

}}  // namespace backends
}}  // namespace lemon
"""


def server_cpp(s: dict) -> str:
    stem, cls, tag = s["stem"], s["class_prefix"] + "Server", s["class_prefix"]
    r = s["ready"]
    env_lines = "".join(f'        {{"{k}", "{lit(v)}"}},\n' for k, v in (s.get("env") or {}).items())
    ready_values = ", ".join(f'"{lit(v)}"' for v in r.get("ready_values", []))
    error_values = ", ".join(f'"{lit(v)}"' for v in r.get("error_values", []))
    json_field = lit(r.get("json_field", ""))
    ck = s["checkpoint"]
    if ck["kind"] == "file":
        is_dl, what = "return fs::is_regular_file(fs::path(info.checkpoint()), ec);", "a local file"
    elif ck.get("marker"):
        is_dl = f'return fs::is_regular_file(fs::path(info.checkpoint()) / "{lit(ck["marker"])}", ec);'
        what = f"a local directory holding {lit(ck['marker'])}"
    else:
        is_dl, what = "return fs::is_directory(fs::path(info.checkpoint()), ec);", "a local directory"
    completion = ('    return forward_request("/v1/completions", request);' if s.get("completion", True) else
                  f'    return unsupported_capability_error("legacy text completion");')
    tpath = s.get("telemetry_path", "")
    return f"""// GENERATED by lemonade-engine-adapters/make_adapter.py from engines/{stem}.json. Edit the spec, not this file.
#include "lemon/backends/{stem}/{stem}_server.h"

#include "lemon/backends/backend_ops.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/{stem}/{stem}.h"
#include "lemon/server_process.h"
#include "lemon/utils/aixlog.hpp"

#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace lemon {{
namespace backends {{

namespace {{
constexpr const char* kArgsTemplate = "{lit(s["args_template"])}";
constexpr const char* kOwnedBy = "{lit(s["identity"]["owned_by"])}";
constexpr const char* kReadyPath = "{lit(r["path"])}";
constexpr long kReadyTimeoutSeconds = {r["timeout_s"]};
constexpr const char* kReadyField = "{json_field}";  // "" = the first answer on kReadyPath means ready
const std::vector<std::string> kReadyValues = {{{ready_values}}};
const std::vector<std::string> kErrorValues = {{{error_values}}};
constexpr const char* kTelemetryPath = "{lit(tpath)}";

std::string fill(std::string text, const std::string& model, int port, const std::string& device) {{
    const std::pair<const char*, std::string> subs[] = {{
        {{"{{model}}", model}}, {{"{{port}}", std::to_string(port)}}, {{"{{device}}", device}}}};
    for (const auto& [key, value] : subs) {{
        for (size_t at = text.find(key); at != std::string::npos; at = text.find(key, at + value.size())) {{
            text.replace(at, std::string(key).size(), value);
        }}
    }}
    return text;
}}

std::vector<std::string> split_args(const std::string& text) {{
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string word; in >> word;) {{
        out.push_back(word);
    }}
    return out;
}}

bool contains(const std::vector<std::string>& values, const std::string& v) {{
    for (const auto& x : values) {{
        if (x == v) return true;
    }}
    return false;
}}
}}  // namespace

InstallParams {cls}::get_install_params(const std::string& /*backend*/, const std::string& /*version*/) {{
    return {{}};
}}

{cls}::{cls}(const std::string& log_level, ModelManager* model_manager,
             BackendManager* backend_manager)
    : WrappedServer("{s["recipe"]}", log_level, model_manager, backend_manager) {{}}

void {cls}::load(const std::string& model_name, const ModelInfo& model_info,
                 const RecipeOptions& options, bool /*do_not_upgrade*/) {{
    const std::string checkpoint = model_info.checkpoint();
    if (checkpoint.empty()) {{
        throw std::runtime_error("{s["recipe"]} model checkpoint is empty for: " + model_name);
    }}
    const std::string executable = BackendUtils::get_backend_binary_path(*{stem}::spec(), "system");

    std::string device = options.get_option("{stem}_device");
    if (device.empty()) {{
        device = "0";
    }}
    std::string tmpl = options.get_option("{stem}_args");
    if (tmpl.empty()) {{
        tmpl = kArgsTemplate;
    }}

    port_ = choose_port();
    if (port_ == 0) {{
        throw std::runtime_error("Failed to allocate a port for {s["recipe"]}");
    }}

    ServerCommand command;
    command.program = executable;
    command.args = split_args(fill(tmpl, checkpoint, port_, device));
    command.env = {{
{env_lines}    }};
    for (auto& kv : command.env) {{
        kv.second = fill(kv.second, checkpoint, port_, device);
    }}
    command.port = port_;
    command.ready_endpoint = kReadyPath;

    LOG(INFO, "{tag}") << "Starting " << executable << " for " << checkpoint << " on port " << port_
                       << " (device " << device << ")" << std::endl;
    const bool inherit_output = (log_level_ == "info") || (log_level_ == "debug");
    start_server(std::make_unique<NativeProcess>(ProcessOutput{{inherit_output, true}}), command,
                 kReadyTimeoutSeconds);

    // The health route can answer before the engine is ready (e.g. while weights load).
    if (std::string(kReadyField).size() > 0) {{
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kReadyTimeoutSeconds);
        std::string last;
        while (true) {{
            try {{
                const json health = forward_get_request(kReadyPath, 5);
                last = health.value(kReadyField, std::string{{}});
            }} catch (...) {{
                last = "";
            }}
            if (contains(kReadyValues, last)) break;
            if (contains(kErrorValues, last) || std::chrono::steady_clock::now() > deadline) {{
                stop_server();
                throw std::runtime_error(std::string("{s["recipe"]} engine did not become ready (") + kReadyField +
                                         "=\\"" + last + "\\")");
            }}
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }}
    }}

    // Refuse a mislabelled engine: /v1/models must report the expected owned_by.
    bool identified = false;
    try {{
        const json models = forward_get_request("/v1/models", 10);
        for (const auto& m : models.value("data", json::array())) {{
            if (m.value("owned_by", std::string{{}}) == kOwnedBy) identified = true;
        }}
    }} catch (...) {{
    }}
    if (!identified) {{
        stop_server();
        throw std::runtime_error(std::string("{s["recipe"]}: /v1/models does not report owned_by=") + kOwnedBy +
                                 "; refusing a mislabelled engine");
    }}
    LOG(INFO, "{tag}") << kOwnedBy << " ready on port " << port_ << std::endl;
}}

json {cls}::chat_completion(const json& request) {{
    return forward_request("/v1/chat/completions", request);
}}

json {cls}::completion(const json& request) {{
{completion}
}}

std::map<std::string, nlohmann::json> {cls}::get_additional_telemetry() {{
    if (get_backend_port() <= 0 || std::string(kTelemetryPath).empty()) {{
        return {{}};
    }}
    std::map<std::string, nlohmann::json> out;
    try {{
        out["{stem}_stats"] = forward_get_request(kTelemetryPath, 2);
    }} catch (...) {{
        out["{stem}_stats"] = nullptr;
    }}
    return out;
}}

std::string {cls}::get_additional_telemetry_url() const {{
    if (get_backend_port() <= 0 || std::string(kTelemetryPath).empty()) {{
        return "";
    }}
    return get_base_url() + kTelemetryPath;
}}

}}  // namespace backends
}}  // namespace lemon

namespace lemon {{
namespace backends {{
namespace {stem} {{

namespace {{
// Models are {what} on this machine; nothing is fetched from a hub, so "downloaded" means the
// checkpoint exists here.
class {tag}Ops : public BackendOps {{
public:
    std::string resolve_checkpoint_path(const ModelInfo& info,
                                        const CheckpointResolveContext& /*ctx*/) const override {{
        return info.checkpoint();
    }}
    bool is_downloaded(const ModelInfo& info, const BackendOpsContext& /*ctx*/) const override {{
        std::error_code ec;
        {is_dl}
    }}
    // `pull` calls this for every registration: a checkpoint already on this machine is the
    // "download"; only a missing one is an error.
    void download_model(const ModelInfo& info, bool /*do_not_upgrade*/, DownloadProgressCallback /*progress*/,
                        const BackendOpsContext& ctx) const override {{
        if (is_downloaded(info, ctx)) {{
            return;
        }}
        throw std::runtime_error("{s["recipe"]} models are {what} on this machine; '" + info.checkpoint() +
                                 "' was not found");
    }}
}};
}}  // namespace

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {{
    return make_server<{cls}>(ctx);
}}

const BackendSpec* spec() {{
    return make_spec<{cls}>(descriptor);
}}

const BackendOps* ops() {{
    static const {tag}Ops kOps;
    return &kOps;
}}

}}  // namespace {stem}
}}  // namespace backends
}}  // namespace lemon
"""


def render(spec: dict) -> dict[str, str]:
    s = validate(spec)
    stem = s["stem"]
    return {
        f"src/cpp/include/lemon/backends/{stem}/{stem}.h": descriptor_h(s),
        f"src/cpp/include/lemon/backends/{stem}/{stem}_server.h": server_h(s),
        f"src/cpp/server/backends/{stem}/{stem}_server.cpp": server_cpp(s),
    }


def register_cmake(src: Path, recipe: str, stem: str) -> bool:
    """Add "<recipe>|<stem>" at the end of LEMON_BACKENDS. Returns True if it changed the file."""
    cm = src / "CMakeLists.txt"
    text = cm.read_text()
    line = f'    "{recipe}|{stem}"'
    if line in text.splitlines():
        return False
    m = re.search(r"set\(LEMON_BACKENDS\n((?:    .*\n)+?)\)", text)
    if not m:
        raise SystemExit(f"{cm}: LEMON_BACKENDS block not found")
    cm.write_text(text.replace(m.group(1), m.group(1) + line + "\n", 1))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lemonade-src", type=Path, help="Lemonade source tree to install into (also edits CMakeLists.txt)")
    g.add_argument("--out", type=Path, help="write the generated files under this directory only")
    g.add_argument("--check", action="store_true", help="validate the spec and print the file list")
    a = ap.parse_args(argv)
    try:
        spec = json.loads(Path(a.spec).read_text())
        files = render(spec)
    except (SpecError, json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    if a.check:
        print("\n".join(files))
        return 0
    root = a.lemonade_src or a.out
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        print(f"wrote {p}")
    if a.lemonade_src:
        changed = register_cmake(a.lemonade_src, spec["recipe"], spec["stem"])
        print(f"CMakeLists.txt LEMON_BACKENDS: {'added' if changed else 'already has'} \"{spec['recipe']}|{spec['stem']}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
