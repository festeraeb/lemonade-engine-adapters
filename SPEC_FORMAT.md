# Engine spec format (adapter plane rules)

Rules extracted from lemond's adapter plane, read-only reference
`/mnt/fast/wt-adapter-plane-46/src/cpp/server/backends/adapter/adapter_spec.h` and `adapter_spec.cpp`
(tests: `test/cpp/adapter_spec/test_ap2.cpp`, `test_ap5.cpp`, `test_ap3.cpp`, `test_ap8.cpp`).
`tools/check_specs.py` enforces them on every file in `engines/` and `engines/drafts/`.

## Fields

Required by the plane (`check_spec`): `binary` (non-empty string) and `args` (array of strings).
`make_adapter` additionally needs `recipe stem class_prefix display_name default_device support modes
checkpoint ready identity`. Optional, type-checked by the plane: `spec_version` (integer >= 1),
`env` (object), `ready` (object), `telemetry` (object), `telemetry_path`, `memory_path`, `budget_path`,
`default_device_index` (strings), `binary_sha256` (64 hex). Unknown fields (`notes`, `support`, ...) are ignored.

## args and placeholders

`args` is the engine's argv without the program: one string per argv entry. Elements are never
re-split, so a value with spaces stays one argument; no shell runs them. Placeholders (`build_argv`,
`kPlaceholders`), filled by lemond, never by the spec:
`{port}` `{model}` `{model_path}` `{device}` `{gpu_uuids}` `{ctx}` `{spec_dir}`.
A placeholder is `{`, an identifier `[A-Za-z_][A-Za-z0-9_]*`, `}`; any other braces are plain text.
An unknown placeholder, or one lemond does not supply, is a `SpecError`. `{device}` is the card index,
`{gpu_uuids}` the card uuid (NVIDIA `GPU-...`, AMD `amd:<device>:<subsystem>`), or the index when none.

## Version rule

Current `kSpecVersion` is 1. `spec_version` absent means 1. A value above 1 or not an integer >= 1 is a
problem and lemond runs the spec without control (level cap 0). `args_template` (one string split on
whitespace, no quoting) is accepted only when there is no `args` and spec_version is 1 or absent; at a
higher version it is an error. Our specs therefore carry `"spec_version": 1` and `args`, no `args_template`.
`make_adapter` (Python and Rust) still reads a legacy `args_template`, and joins `args` back into one
space-split string for the C++ it generates, so an `args` element must not contain whitespace there.

## env allowlist

`env` is an object name -> string; values may use the placeholders above. A name must start with one of
`CUDA_ HIP_ ROCR_ HSA_ GGML_ LLAMA_ OMP_ MKL_ VK_ NAUTIBURN_ STRATA_ LEMONADE_ENGINE_`, or be exactly
`LD_LIBRARY_PATH` or `TZ`. `LD_LIBRARY_PATH` entries (split on `:`) must be absolute, contain no `..`, and
lie inside a directory lemond allows. Anything else is refused at load.

## ready probe

`ready` is an object. `ready_state` maps the engine's status answer: without `json_field` any object
answer is Ready; with it, the field's value is looked up in `error_values`, `warm_values`, `ready_values`,
`loading_values` (in that order); a missing field or unlisted value is Starting. `progress_field` is read
only while Loading (0..1 as is, (1..100] divided by 100, else -1). `path` and `timeout_s` are used by
make_adapter (path like `/health`, timeout 10..7200); `json_field` and `ready_values` go together.
