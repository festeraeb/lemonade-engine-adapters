# Engine specs as Lemonade backend definitions

[lemonade-sdk/lemonade#3906](https://github.com/lemonade-sdk/lemonade/pull/3906) lets `lemond` read backend
definitions from `<config_dir>/backends.d/*.json`. `tools/to_backend_definition.py` turns each spec in
`engines/` into one of those: schema v1, `install.strategy: external_only`, no hooks, no compiled code.
The results are in `backends.d/`.

```bash
python3 tools/to_backend_definition.py --all --out backends.d    # write
python3 tools/to_backend_definition.py --all --check backends.d  # exit 1 when stale
python3 tools/to_backend_definition.py --all --gaps              # what a spec says that a definition cannot
python3 -I tests/test_to_backend_definition.py
```

## How a spec maps

| Spec | Definition |
|---|---|
| `binary`, `display_name`, `class_prefix`, `modes` | `metadata` |
| `args` with `{model}` `{port}` | `launch.argv` with `/model/checkpoint` and `/launch/ports/http` |
| a flag whose value is `{device}` | its own `argv` record, dropped while the device option is empty |
| `{device}` in `env` | `launch.environment.set`; the option then defaults to `0`, because a variable cannot be left out |
| `{device}` anywhere | one option `<recipe>_device` with `type_name: DEVICES` |
| `ready.path`, `ready.timeout_s` | `launch.readiness`, `launch.watchdog` |
| `library_dir: binary` | `environment.preset: executable_loader_path` |
| `prometheus_metrics: true` | `metadata.exposes_prometheus_metrics` |
| `completion: true` | `completion` operations beside `chat_completion`, buffered and streaming |
| `support` | `metadata.support` rows, tier `guest`, and `bin_variants` |

The engine's path comes from the server's config: `<config_section>.<backend>_bin`, for example
`lemonade_burn.cpu_bin` or `nautistrata.cuda_bin`. A model is a `user_models.json` entry with the recipe.

## What a definition cannot say yet

`--gaps` prints these per engine. They are left out, not approximated.

| Spec field | Why it matters |
|---|---|
| `ready.json_field` / `ready_values` / `loading_values` | llama-server answers `/health` before the model is loaded; the spec waits for `status: ok` |
| `identity.owned_by` | refuse an engine whose `/v1/models` names a different owner |
| `binary_sha256` | pin the executable |
| `telemetry_path`, `memory_path`, `budget_path` | per-device memory report and the shrink request; only Prometheus `/metrics` is forwarded today |
| `checkpoint.kind: dir` with a marker file | check the model folder before launch |

## Measured, 2026-11-13

One host: 48 CPU threads, one Tesla P100 (16 GB) and a Quadro P1000. `lemond` built from #3906 at
`a0ba03e7d` with no source change. Graded by `tools/backend_conformance.py` from
[`cesarops/conformance-on-3906`](https://github.com/festeraeb/lemonade/tree/cesarops/conformance-on-3906),
which reads the same definition file `lemond` runs.

| Definition | Engine, model | Where | L1 lifecycle | L2 eviction | L3 device | L4 telemetry | Earned |
|---|---|---|---|---|---|---|---|
| `llamacpp-local` | llama.cpp CUDA build (0.7.0-dev, d0a03c0a3), Phi-3-mini Q4; second model Gemma-4 E4B Q4_K_M | P100 | pass, 2.8 s | pass | pass: `--device CUDA0`, memory on GPU 0 only | pass, 15 metric families | **L4** |
| `nautistrata` | Nauti-Strata 0.1.44.1, Qwen3.6-35B-A3B Q5_K_M | P100 | pass, 120 s, 15.7 GB held and freed | pass | pass: `--gpu 0`, memory on GPU 0 only | not gradable: no `/metrics` | **L3** |
| `bonsaicpu` | PrismML llama.cpp build 10777, Ternary-Bonsai-2-27B PQ2_0 | CPU | pass, 62 s | pass | not gradable: no device option | pass, 15 metric families | **L2** |
| `lemonade-burn` | lemonade-burn-server 0.1.1, BitNet b1.58 2B4T | CPU | pass, 4.6 s | pass | not gradable: no device option | not gradable: no `/metrics` | **L2** |

- The earned level is the highest level that passes together with every level below it.
- Without the device option the llama.cpp engine spread over both cards (3766 MiB and 707 MiB); with it, GPU 0 only.
- One short answer per engine on a busy host: these are pass/fail checks, not speed numbers.
- All eight definitions in `backends.d/` load together in that `lemond` and pass `lemonade backends validate`.
- Not run: `freetoken`, `strata`, `polystrata`, `lemonade-burn-lfm2` (engine or model not on that host).
- L2 used the same model registered under two names, except `llamacpp-local`.
