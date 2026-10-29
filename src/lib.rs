//! Adapter maker: turn one engine spec (JSON) into a native Lemonade backend.
//!
//! Rust port of `make_adapter.py`. Output is byte-identical to the Python tool. See README.md and
//! ADAPTER_PROMPT.md for the spec fields.

use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path, PathBuf};

pub const REQUIRED: [&str; 12] = [
    "recipe", "stem", "class_prefix", "display_name", "binary", "args_template", "default_device",
    "support", "modes", "checkpoint", "ready", "identity",
];
const DEVICE_FAMILIES: [&str; 5] = ["amd_gpu", "cpu", "metal", "npu", "nvidia_gpu"]; // sorted
const OSES: [&str; 3] = ["linux", "macos", "windows"]; // sorted
const PLACEHOLDERS: [&str; 3] = ["device", "model", "port"]; // sorted

#[derive(Debug, Clone, PartialEq)]
pub struct SpecError(pub String);

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SpecError {}

fn err<T>(m: impl Into<String>) -> Result<T, SpecError> {
    Err(SpecError(m.into()))
}

// ---------------------------------------------------------------- Python-flavoured helpers

/// Python `re.match(r"^...$", s)` lets `$` match before one trailing newline.
fn dollar(s: &str) -> &str {
    s.strip_suffix('\n').unwrap_or(s)
}

fn recipe_ok(s: &str) -> bool {
    let s = dollar(s);
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_lowercase() || f.is_ascii_digit())
        && c.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '-')
}
fn stem_ok(s: &str) -> bool {
    let s = dollar(s);
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_lowercase())
        && c.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '_')
}
fn class_ok(s: &str) -> bool {
    let s = dollar(s);
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_uppercase()) && c.all(|x| x.is_ascii_alphanumeric())
}
fn path_ok(s: &str) -> bool {
    let s = dollar(s);
    let mut c = s.chars();
    c.next() == Some('/') && c.all(|x| x.is_ascii_alphanumeric() || "_./-".contains(x))
}
fn env_name_ok(s: &str) -> bool {
    let s = dollar(s);
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_uppercase() || f == '_')
        && c.all(|x| x.is_ascii_uppercase() || x.is_ascii_digit() || x == '_')
}

/// Python `repr(str)`.
fn py_str_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut o = String::new();
    o.push(q);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == q => {
                o.push('\\');
                o.push(c)
            }
            c if (c as u32) < 0x20 || (0x7f..0xa0).contains(&(c as u32)) => {
                o.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

/// Python `repr()` of a value parsed from JSON.
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => {
            if let Some(f) = n.as_f64().filter(|_| !n.is_i64() && !n.is_u64()) {
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    format!("{:.1}", f)
                } else {
                    format!("{}", f)
                }
            } else {
                n.to_string()
            }
        }
        Value::String(s) => py_str_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn py_list_repr(items: &[&str]) -> String {
    format!("[{}]", items.iter().map(|s| py_str_repr(s)).collect::<Vec<_>>().join(", "))
}

/// Python truthiness.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

/// C++ string literal body.
pub fn lit(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Single-pass `@NAME@` substitution; substituted text is never rescanned.
fn fill(tmpl: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(tmpl.len() + 512);
    let mut rest = tmpl;
    while let Some(i) = rest.find('@') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let j = after.find('@').expect("unterminated @ in template");
        let name = &after[..j];
        let val = vars
            .iter()
            .find(|(k, _)| *k == name)
            .unwrap_or_else(|| panic!("template var {name} missing"))
            .1;
        out.push_str(val);
        rest = &after[j + 1..];
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------- spec access

fn obj<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>, SpecError> {
    v.as_object()
        .ok_or_else(|| SpecError(format!("{what} must be a JSON object")))
}

fn sget<'a>(m: &'a Map<String, Value>, k: &str) -> Result<&'a str, SpecError> {
    match m.get(k) {
        Some(Value::String(s)) => Ok(s),
        Some(_) => err(format!("{k} must be a string")),
        None => err(format!("missing fields: {k}")),
    }
}

/// Find `\{([a-z_]+)\}` matches (non-overlapping), like Python `re.findall`.
fn placeholders(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_ascii_lowercase() || b[j] == b'_') {
                j += 1;
            }
            if j > i + 1 && j < b.len() && b[j] == b'}' {
                out.push(text[i + 1..j].to_string());
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Elements of a value used as a Python iterable of hashables (list of strings, or dict keys).
fn str_items(v: &Value) -> Option<Vec<String>> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_str().map(String::from)).collect(),
        Value::Object(m) => Some(m.keys().cloned().collect()),
        _ => None,
    }
}

pub fn validate(spec: &Value) -> Result<&Map<String, Value>, SpecError> {
    let s = obj(spec, "spec")?;
    let missing: Vec<&str> = REQUIRED.iter().copied().filter(|k| !s.contains_key(*k)).collect();
    if !missing.is_empty() {
        return err(format!("missing fields: {}", missing.join(", ")));
    }
    let recipe = sget(s, "recipe")?;
    if !recipe_ok(recipe) {
        return err(format!("recipe {}: lowercase letters, digits and '-'", py_str_repr(recipe)));
    }
    let stem = sget(s, "stem")?;
    if !stem_ok(stem) {
        return err(format!("stem {}: [a-z][a-z0-9_]*", py_str_repr(stem)));
    }
    let cp = sget(s, "class_prefix")?;
    if !class_ok(cp) {
        return err(format!("class_prefix {}: CamelCase letters and digits", py_str_repr(cp)));
    }
    let binary = sget(s, "binary")?;
    if binary.is_empty() || binary.contains('/') || binary.contains(' ') {
        return err("binary must be a bare executable name found in PATH (no path, no spaces)");
    }
    let tmpl = sget(s, "args_template")?;
    for ph in ["{model}", "{port}"] {
        if !tmpl.contains(ph) {
            return err(format!("args_template must contain {ph}"));
        }
    }
    let empty = Map::new();
    let env: &Map<String, Value> = match s.get("env") {
        Some(v) if truthy(v) => obj(v, "env")?,
        _ => &empty,
    };
    let mut texts: Vec<&str> = vec![tmpl];
    for v in env.values() {
        match v {
            Value::String(t) => texts.push(t),
            _ => return err("env values must be strings"),
        }
    }
    let unknown: BTreeSet<String> = texts
        .iter()
        .flat_map(|t| placeholders(t))
        .filter(|p| !PLACEHOLDERS.contains(&p.as_str()))
        .collect();
    if !unknown.is_empty() {
        let u: Vec<&str> = unknown.iter().map(|x| x.as_str()).collect();
        return err(format!(
            "unknown placeholders {}; allowed: {}",
            py_list_repr(&u),
            py_list_repr(&PLACEHOLDERS)
        ));
    }
    if tmpl.chars().any(|c| "\"'`$|;&<>".contains(c)) {
        return err("args_template is split on spaces without a shell: no quotes, $, pipes or redirects");
    }
    for k in env.keys() {
        if !env_name_ok(k) {
            return err(format!("env name {} is not a valid variable name", py_str_repr(k)));
        }
    }
    if !matches!(s["default_device"].as_str(), Some("gpu") | Some("cpu")) {
        return err("default_device must be 'gpu' or 'cpu'");
    }
    let support = match &s["support"] {
        Value::Array(a) if !a.is_empty() => a,
        _ => return err("support must be a non-empty list"),
    };
    for row in support {
        let r = obj(row, "support row")?;
        let os_ok = match r.get("os") {
            Some(v) if truthy(v) => str_items(v).map_or(false, |it| it.iter().all(|o| OSES.contains(&o.as_str()))),
            _ => false,
        };
        if !os_ok {
            return err(format!(
                "support row os must be a non-empty subset of {}: {}",
                py_list_repr(&OSES),
                py_repr(row)
            ));
        }
        let dev_ok = match r.get("devices") {
            Some(v) if truthy(v) => str_items(v)
                .map_or(false, |it| it.iter().all(|d| DEVICE_FAMILIES.contains(&d.as_str()))),
            _ => false,
        };
        if !dev_ok {
            return err(format!(
                "support row devices must use {}: {}",
                py_list_repr(&DEVICE_FAMILIES),
                py_repr(row)
            ));
        }
    }
    let modes_ok = match &s["modes"] {
        Value::Array(a) => !a.is_empty() && a.iter().all(|m| matches!(m, Value::String(x) if !x.is_empty())),
        _ => false,
    };
    if !modes_ok {
        return err("modes must be a non-empty list of strings");
    }
    let ck = obj(&s["checkpoint"], "checkpoint")?;
    let kind = ck.get("kind").and_then(|k| k.as_str());
    if !matches!(kind, Some("file") | Some("dir")) {
        return err("checkpoint.kind must be 'file' or 'dir'");
    }
    if ck.contains_key("marker") && kind != Some("dir") {
        return err("checkpoint.marker only applies to kind 'dir'");
    }
    let r = obj(&s["ready"], "ready")?;
    if !r.get("path").map_or(false, |p| p.as_str().map_or(false, path_ok)) {
        return err("ready.path must be an HTTP path like /health");
    }
    let t_ok = match r.get("timeout_s") {
        Some(Value::Number(n)) => n.as_i64().map_or(false, |t| (10..=7200).contains(&t)),
        _ => false,
    };
    if !t_ok {
        return err("ready.timeout_s must be an int in 10..7200");
    }
    if r.contains_key("json_field") != r.contains_key("ready_values") {
        return err("ready.json_field and ready.ready_values go together");
    }
    if r.get("ready_values").map_or(false, |v| !truthy(v)) {
        return err("ready.ready_values must not be empty");
    }
    let ident = obj(&s["identity"], "identity")?;
    if !ident.get("owned_by").map_or(false, truthy) {
        return err("identity.owned_by is required: the engine's owned_by in /v1/models");
    }
    if let Some(t) = s.get("telemetry_path") {
        if !t.as_str().map_or(false, path_ok) {
            return err("telemetry_path must be an HTTP path");
        }
    }
    Ok(s)
}

// ---------------------------------------------------------------- templates

const DESCRIPTOR_H: &str = r####"#pragma once
// GENERATED by lemonade-engine-adapters/make_adapter.py from engines/@STEM@.json. Edit the spec, not this file.

#include "lemon/backends/backend_descriptor.h"

namespace lemon {
namespace backends {
namespace @STEM@ {

// @DISPLAY@: lemond starts `@BINARY@` from PATH, waits for @READY_PATH@,
// and refuses to serve unless /v1/models reports owned_by == "@OWNED_BY@".
@NOTES@inline const BackendDescriptor descriptor = {
    /*recipe*/          "@RECIPE@",
    /*display_name*/    "@DISPLAY@",
    /*binary*/          "@BINARY@",
    /*config_section*/  "@STEM@",
    /*default_device*/  @DEV@,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,
    /*dynamic_models*/  false,
    /*options*/ {
        {"@STEM@_args", "--@FLAG@-args", "", "ARGS",
         "Replaces the engine argument template ({model} {port} {device} placeholders)", "@DISPLAY@ Options"},
        {"@STEM@_device", "--@FLAG@-device", "", "DEVICE",
         "Device index substituted for {device} (default 0)", "@DISPLAY@ Options"},
    },
    /*support*/ {
@ROWS@
    },
    /*supported_modes*/ {@MODES@},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*web_display_name*/ "",
    /*rocm_channels*/   {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/  VersionPolicy::AtLeast,
    /*self_manages_downloads*/ true,  // the checkpoint is a local @CKKIND@ on this machine
};

}  // namespace @STEM@
}  // namespace backends
}  // namespace lemon
"####;

const SERVER_H: &str = r####"#pragma once
// GENERATED by lemonade-engine-adapters/make_adapter.py from engines/@STEM@.json. Edit the spec, not this file.

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/wrapped_server.h"

namespace lemon {
namespace backends {

class @CLS@ : public WrappedServer {
public:
    // No auto-install: the engine binary must be on PATH for lemond.
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    @CLS@(const std::string& log_level, ModelManager* model_manager,
          BackendManager* backend_manager);

    void load(const std::string& model_name, const ModelInfo& model_info,
              const RecipeOptions& options, bool do_not_upgrade = false) override;

    json chat_completion(const json& request) override;
    json completion(const json& request) override;

    std::map<std::string, nlohmann::json> get_additional_telemetry() override;
    std::string get_additional_telemetry_url() const override;
};

namespace @STEM@ {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() { return capability_mask_of<@CLS@>(); }
}  // namespace @STEM@

}  // namespace backends
}  // namespace lemon
"####;

const SERVER_CPP: &str = r####"// GENERATED by lemonade-engine-adapters/make_adapter.py from engines/@STEM@.json. Edit the spec, not this file.
#include "lemon/backends/@STEM@/@STEM@_server.h"

#include "lemon/backends/backend_ops.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/@STEM@/@STEM@.h"
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

namespace lemon {
namespace backends {

namespace {
constexpr const char* kArgsTemplate = "@ARGS_TMPL@";
constexpr const char* kOwnedBy = "@OWNED_BY@";
constexpr const char* kReadyPath = "@READY_PATH@";
constexpr long kReadyTimeoutSeconds = @TIMEOUT@;
constexpr const char* kReadyField = "@JSON_FIELD@";  // "" = the first answer on kReadyPath means ready
const std::vector<std::string> kReadyValues = {@READY_VALUES@};
const std::vector<std::string> kErrorValues = {@ERROR_VALUES@};
constexpr const char* kTelemetryPath = "@TPATH@";

std::string fill(std::string text, const std::string& model, int port, const std::string& device) {
    const std::pair<const char*, std::string> subs[] = {
        {"{model}", model}, {"{port}", std::to_string(port)}, {"{device}", device}};
    for (const auto& [key, value] : subs) {
        for (size_t at = text.find(key); at != std::string::npos; at = text.find(key, at + value.size())) {
            text.replace(at, std::string(key).size(), value);
        }
    }
    return text;
}

std::vector<std::string> split_args(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string word; in >> word;) {
        out.push_back(word);
    }
    return out;
}

bool contains(const std::vector<std::string>& values, const std::string& v) {
    for (const auto& x : values) {
        if (x == v) return true;
    }
    return false;
}
}  // namespace

InstallParams @CLS@::get_install_params(const std::string& /*backend*/, const std::string& /*version*/) {
    return {};
}

@CLS@::@CLS@(const std::string& log_level, ModelManager* model_manager,
             BackendManager* backend_manager)
    : WrappedServer("@RECIPE@", log_level, model_manager, backend_manager) {}

void @CLS@::load(const std::string& model_name, const ModelInfo& model_info,
                 const RecipeOptions& options, bool /*do_not_upgrade*/) {
    const std::string checkpoint = model_info.checkpoint();
    if (checkpoint.empty()) {
        throw std::runtime_error("@RECIPE@ model checkpoint is empty for: " + model_name);
    }
    const std::string executable = BackendUtils::get_backend_binary_path(*@STEM@::spec(), "system");

    std::string device = options.get_option("@STEM@_device");
    if (device.empty()) {
        device = "0";
    }
    std::string tmpl = options.get_option("@STEM@_args");
    if (tmpl.empty()) {
        tmpl = kArgsTemplate;
    }

    port_ = choose_port();
    if (port_ == 0) {
        throw std::runtime_error("Failed to allocate a port for @RECIPE@");
    }

    ServerCommand command;
    command.program = executable;
    command.args = split_args(fill(tmpl, checkpoint, port_, device));
    command.env = {
@ENV_LINES@    };
    for (auto& kv : command.env) {
        kv.second = fill(kv.second, checkpoint, port_, device);
    }
    command.port = port_;
    command.ready_endpoint = kReadyPath;

    LOG(INFO, "@TAG@") << "Starting " << executable << " for " << checkpoint << " on port " << port_
                       << " (device " << device << ")" << std::endl;
    const bool inherit_output = (log_level_ == "info") || (log_level_ == "debug");
    start_server(std::make_unique<NativeProcess>(ProcessOutput{inherit_output, true}), command,
                 kReadyTimeoutSeconds);

    // The health route can answer before the engine is ready (e.g. while weights load).
    if (std::string(kReadyField).size() > 0) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kReadyTimeoutSeconds);
        std::string last;
        while (true) {
            try {
                const json health = forward_get_request(kReadyPath, 5);
                last = health.value(kReadyField, std::string{});
            } catch (...) {
                last = "";
            }
            if (contains(kReadyValues, last)) break;
            if (contains(kErrorValues, last) || std::chrono::steady_clock::now() > deadline) {
                stop_server();
                throw std::runtime_error(std::string("@RECIPE@ engine did not become ready (") + kReadyField +
                                         "=\"" + last + "\")");
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    // Refuse a mislabelled engine: /v1/models must report the expected owned_by.
    bool identified = false;
    try {
        const json models = forward_get_request("/v1/models", 10);
        for (const auto& m : models.value("data", json::array())) {
            if (m.value("owned_by", std::string{}) == kOwnedBy) identified = true;
        }
    } catch (...) {
    }
    if (!identified) {
        stop_server();
        throw std::runtime_error(std::string("@RECIPE@: /v1/models does not report owned_by=") + kOwnedBy +
                                 "; refusing a mislabelled engine");
    }
    LOG(INFO, "@TAG@") << kOwnedBy << " ready on port " << port_ << std::endl;
}

json @CLS@::chat_completion(const json& request) {
    return forward_request("/v1/chat/completions", request);
}

json @CLS@::completion(const json& request) {
@COMPLETION@
}

std::map<std::string, nlohmann::json> @CLS@::get_additional_telemetry() {
    if (get_backend_port() <= 0 || std::string(kTelemetryPath).empty()) {
        return {};
    }
    std::map<std::string, nlohmann::json> out;
    try {
        out["@STEM@_stats"] = forward_get_request(kTelemetryPath, 2);
    } catch (...) {
        out["@STEM@_stats"] = nullptr;
    }
    return out;
}

std::string @CLS@::get_additional_telemetry_url() const {
    if (get_backend_port() <= 0 || std::string(kTelemetryPath).empty()) {
        return "";
    }
    return get_base_url() + kTelemetryPath;
}

}  // namespace backends
}  // namespace lemon

namespace lemon {
namespace backends {
namespace @STEM@ {

namespace {
// Models are @WHAT@ on this machine; nothing is fetched from a hub, so "downloaded" means the
// checkpoint exists here.
class @TAG@Ops : public BackendOps {
public:
    std::string resolve_checkpoint_path(const ModelInfo& info,
                                        const CheckpointResolveContext& /*ctx*/) const override {
        return info.checkpoint();
    }
    bool is_downloaded(const ModelInfo& info, const BackendOpsContext& /*ctx*/) const override {
        std::error_code ec;
        @IS_DL@
    }
    // `pull` calls this for every registration: a checkpoint already on this machine is the
    // "download"; only a missing one is an error.
    void download_model(const ModelInfo& info, bool /*do_not_upgrade*/, DownloadProgressCallback /*progress*/,
                        const BackendOpsContext& ctx) const override {
        if (is_downloaded(info, ctx)) {
            return;
        }
        throw std::runtime_error("@RECIPE@ models are @WHAT@ on this machine; '" + info.checkpoint() +
                                 "' was not found");
    }
};
}  // namespace

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<@CLS@>(ctx);
}

const BackendSpec* spec() {
    return make_spec<@CLS@>(descriptor);
}

const BackendOps* ops() {
    static const @TAG@Ops kOps;
    return &kOps;
}

}  // namespace @STEM@
}  // namespace backends
}  // namespace lemon
"####;

// ---------------------------------------------------------------- generators

fn strs(v: Option<&Value>) -> Vec<&str> {
    v.and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default()
}

fn descriptor_h(s: &Map<String, Value>) -> String {
    let stem = s["stem"].as_str().unwrap();
    let mut rows = vec![];
    for row in s["support"].as_array().unwrap() {
        let row = row.as_object().unwrap();
        let fams = row["devices"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(fam, archs)| {
                let a = archs
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|x| format!("\"{}\"", lit(x.as_str().unwrap_or(""))))
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                format!("{{\"{fam}\", {{{a}}}}}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let oses = strs(row.get("os")).iter().map(|o| format!("\"{o}\"")).collect::<Vec<_>>().join(", ");
        let note = lit(row.get("note").and_then(|n| n.as_str()).unwrap_or(""));
        rows.push(format!(
            "        {{\"system\", BackendTier::Experimental, BackendFormat::Native, {{{oses}}},\n         {{{fams}}}, \"{note}\"}},"
        ));
    }
    let notes: String = strs(s.get("notes")).iter().map(|n| format!("// {n}\n")).collect();
    let modes = strs(s.get("modes")).iter().map(|m| format!("\"{}\"", lit(m))).collect::<Vec<_>>().join(", ");
    let dev = if s["default_device"].as_str() == Some("gpu") { "DEVICE_GPU" } else { "DEVICE_CPU" };
    let flag = stem.replace('_', "-");
    let ready = s["ready"].as_object().unwrap();
    fill(
        DESCRIPTOR_H,
        &[
            ("STEM", stem),
            ("DISPLAY", &lit(s["display_name"].as_str().unwrap())),
            ("BINARY", &lit(s["binary"].as_str().unwrap())),
            ("READY_PATH", &lit(ready["path"].as_str().unwrap())),
            ("OWNED_BY", &lit(s["identity"]["owned_by"].as_str().unwrap_or(""))),
            ("NOTES", &notes),
            ("RECIPE", s["recipe"].as_str().unwrap()),
            ("DEV", dev),
            ("FLAG", &flag),
            ("ROWS", &rows.join("\n")),
            ("MODES", &modes),
            ("CKKIND", s["checkpoint"]["kind"].as_str().unwrap()),
        ],
    )
}

fn server_h(s: &Map<String, Value>) -> String {
    let cls = format!("{}Server", s["class_prefix"].as_str().unwrap());
    fill(SERVER_H, &[("STEM", s["stem"].as_str().unwrap()), ("CLS", &cls)])
}

fn server_cpp(s: &Map<String, Value>) -> String {
    let stem = s["stem"].as_str().unwrap();
    let tag = s["class_prefix"].as_str().unwrap();
    let cls = format!("{tag}Server");
    let r = s["ready"].as_object().unwrap();
    let mut env_lines = String::new();
    if let Some(Value::Object(env)) = s.get("env") {
        for (k, v) in env {
            env_lines.push_str(&format!("        {{\"{k}\", \"{}\"}},\n", lit(v.as_str().unwrap_or(""))));
        }
    }
    let quoted = |key: &str| {
        strs(r.get(key)).iter().map(|v| format!("\"{}\"", lit(v))).collect::<Vec<_>>().join(", ")
    };
    let ready_values = quoted("ready_values");
    let error_values = quoted("error_values");
    let json_field = lit(r.get("json_field").and_then(|v| v.as_str()).unwrap_or(""));
    let ck = s["checkpoint"].as_object().unwrap();
    let marker = ck.get("marker").filter(|m| truthy(m)).and_then(|m| m.as_str());
    let (is_dl, what) = if ck["kind"].as_str() == Some("file") {
        ("return fs::is_regular_file(fs::path(info.checkpoint()), ec);".to_string(), "a local file".to_string())
    } else if let Some(m) = marker {
        (
            format!("return fs::is_regular_file(fs::path(info.checkpoint()) / \"{}\", ec);", lit(m)),
            format!("a local directory holding {}", lit(m)),
        )
    } else {
        ("return fs::is_directory(fs::path(info.checkpoint()), ec);".to_string(), "a local directory".to_string())
    };
    let completion = if s.get("completion").map_or(true, truthy) {
        "    return forward_request(\"/v1/completions\", request);"
    } else {
        "    return unsupported_capability_error(\"legacy text completion\");"
    };
    let tpath = lit(s.get("telemetry_path").and_then(|v| v.as_str()).unwrap_or(""));
    fill(
        SERVER_CPP,
        &[
            ("STEM", stem),
            ("CLS", &cls),
            ("TAG", tag),
            ("RECIPE", s["recipe"].as_str().unwrap()),
            ("ARGS_TMPL", &lit(s["args_template"].as_str().unwrap())),
            ("OWNED_BY", &lit(s["identity"]["owned_by"].as_str().unwrap_or(""))),
            ("READY_PATH", &lit(r["path"].as_str().unwrap())),
            ("TIMEOUT", &r["timeout_s"].to_string()),
            ("JSON_FIELD", &json_field),
            ("READY_VALUES", &ready_values),
            ("ERROR_VALUES", &error_values),
            ("TPATH", &tpath),
            ("ENV_LINES", &env_lines),
            ("COMPLETION", completion),
            ("WHAT", &what),
            ("IS_DL", &is_dl),
        ],
    )
}

/// Validate and render: (relative path, body) in the same order as the Python tool.
pub fn render(spec: &Value) -> Result<Vec<(String, String)>, SpecError> {
    let s = validate(spec)?;
    let stem = s["stem"].as_str().unwrap();
    Ok(vec![
        (format!("src/cpp/include/lemon/backends/{stem}/{stem}.h"), descriptor_h(s)),
        (format!("src/cpp/include/lemon/backends/{stem}/{stem}_server.h"), server_h(s)),
        (format!("src/cpp/server/backends/{stem}/{stem}_server.cpp"), server_cpp(s)),
    ])
}

// ---------------------------------------------------------------- CMake registration

/// Python `str.splitlines()` separators.
fn py_splitlines(text: &str) -> Vec<&str> {
    let mut out = vec![];
    let mut start = 0;
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let sep = matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if sep {
            out.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && matches!(it.peek(), Some((_, '\n'))) {
                it.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// Group 1 of `set\(LEMON_BACKENDS\n((?:    .*\n)+?)\)`: (start, end) byte range.
fn find_backends_block(text: &str) -> Option<(usize, usize)> {
    const HEAD: &str = "set(LEMON_BACKENDS\n";
    let mut from = 0;
    while let Some(p) = text[from..].find(HEAD) {
        let gs = from + p + HEAD.len();
        let mut pos = gs;
        let mut lines = 0;
        loop {
            let rest = &text[pos..];
            if lines > 0 && rest.starts_with(')') {
                return Some((gs, pos));
            }
            if !rest.starts_with("    ") {
                break;
            }
            match rest.find('\n') {
                Some(nl) => {
                    pos += nl + 1;
                    lines += 1;
                }
                None => break,
            }
        }
        from += p + 1;
    }
    None
}

/// Add "<recipe>|<stem>" at the end of LEMON_BACKENDS. Returns Ok(true) if it changed the file.
pub fn register_cmake(src: &Path, recipe: &str, stem: &str) -> Result<bool, String> {
    let cm = src.join("CMakeLists.txt");
    // Python read_text() applies universal newlines.
    let raw = std::fs::read(&cm).map_err(|e| format!("{}: {e}", disp(&cm)))?;
    let text = String::from_utf8(raw)
        .map_err(|e| format!("{}: {e}", disp(&cm)))?
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let line = format!("    \"{recipe}|{stem}\"");
    if py_splitlines(&text).iter().any(|l| *l == line) {
        return Ok(false);
    }
    let (gs, ge) = find_backends_block(&text).ok_or_else(|| format!("{}: LEMON_BACKENDS block not found", disp(&cm)))?;
    let group = &text[gs..ge];
    let new = text.replacen(group, &format!("{group}{line}\n"), 1);
    std::fs::write(&cm, new).map_err(|e| format!("{}: {e}", disp(&cm)))?;
    Ok(true)
}

/// Display a path the way Python's `Path` does (no `.` components, no doubled slashes).
pub fn disp(p: &Path) -> String {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            c => out.push(c.as_os_str()),
        }
    }
    let s = out.display().to_string();
    if s.is_empty() { ".".into() } else { s }
}
