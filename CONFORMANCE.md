# Adapter conformance: our own engines through lemond

Date: 2026-11-10. Host: t440 (48 logical CPUs, shared with the forge, load average about 15 during the runs).
**CPU only.** Every process was started with `CUDA_VISIBLE_DEVICES=-1` and `GGML_VK_VISIBLE_DEVICES=-1`; the GPUs belong to the forge,
so L3 (a per-model GPU device) was not run and L2 max is the ceiling here.

Plane: lemond 2026.47.0 built from `cesarops/adapter-plane-46` (with commits 7c6d9160b `LB_` env prefix and eb6186c5c `identity.unreported`).
Every row below went through lemond: spec in `LEMONADE_ADAPTERS_DIR`, `POST /api/v1/load`, `GET /api/v1/health` (launch command and pid
shown by lemond), one streamed chat completion (`max_tokens` 48), `POST /api/v1/unload` (engine pid confirmed gone), then
`tools/adapter_conformance.py` (with `--other` for L2). Decode tok/s is SSE chunks per second after the first token from that one short
completion: one chunk is one token, but it is a single sample on a busy host, not a benchmark.

| Engine (spec) | Binary / version | Model | Load s | First token s | Decode tok/s | Level earned | First level not earned, exact text, and blocker |
|---|---|---|---|---|---|---|---|
| lemonade-burn ternary (`lemonade-burn.json`, catalogue `burn`) | `lemonade-burn-server` 0.1.1 (catalogue release, sha256 matches) | bitnet-b1.58-2B-4T-lbt (BitNet b1.58 2B4T, ternary) | 6.9 | 0.33 | 25 | **L2** | L3: `not tested: pass --device <GPU index>`. CPU engine, no device option to honour; needs a GPU run. L4: `no adapter_stats (spec has no telemetry_path?)` |
| Ternary Bonsai (`bonsaicpu.json`, catalogue `bonsaicpu`/`bonsai`) | `llama-server` build 10777 (e3de71522), the PrismML fork the `bonsai-cpu-t440` seat runs, read-only from `/mnt/fast/prism-bonsai/bin` | Ternary-Bonsai-2-27B-PQ2_0.gguf (the seat's file; the catalogue's 8B was not downloaded) | 87.5 | 4.18 | 3.4 | **L2** | L3: `not tested: pass --device <GPU index>`. Same reason. L4 as above (llama-server has no telemetry route in the spec) |
| Strata (`strata.json`, catalogue `strata`) | Strata Terminal 0.1.59.1 (signed installer, `install.sh` from the v0.2.3.1 feed folder; Terminal installed under `/mnt/fast/forge-ab/engines/strata`) | none: a real model is Qwen3.8-Flash-Next IQ3_S, 85.7 GB download | 0.7 (mock) | n/a | n/a | **L2 with `strata serve --engine mock` only** (lifecycle and eviction of the plane, no model); **real engine not run** | Real engine: needs a CUDA or ROCm card and the 85.7 GB download (`strata launch --plan` shows it; Strata sees the P100s through nvidia-smi even with CUDA hidden). L3 and L4 as above. The mock answers with `content` empty or scripted text, so it says nothing about speed or quality |
| Nauti-Strata (`nautistrata.json`, catalogue `nautistrata`) | `strata-poly-avx2` 0.1.44.1 (9169af25), catalogue Stand bundle, sha256 matches | Qwen3.6-35B-A3B-Q5_K_M.gguf (the 13.6 GB LFM2.5 file was not downloaded; this engine runs MoE only) | not loaded | n/a | n/a | **L0** | L1: `load/chat failed: HTTP Error 500: Internal Server Error` (lemond: `adapter failed to start`). Engine log: `strata-poly: cudaStreamCreate: no CUDA-capable device is detected`. Needs an NVIDIA GPU |
| PolyStrata (`polystrata.json`) | same `serve/server.py` and `strata-poly` as Nauti-Strata (no separate PolyStrata checkout on this host) | same | not loaded | n/a | n/a | **L0** | Identical to Nauti-Strata: `adapter failed to start`, `cudaStreamCreate: no CUDA-capable device is detected` |
| FreeToken (`freetoken.json`) | `ft` not installed on this host | n/a | n/a | n/a | n/a | **L0** | L1: `load/chat failed: HTTP Error 500` (lemond: `adapter: engine binary 'ft' is not on PATH`). The spec is NVIDIA-only (Turing or newer, CUDA 13); skipped without a GPU, not attempted to install |

L4 cannot pass for any engine: the plane never gets telemetry calls from upstream lemond (the hook is not called), and none of these specs name a
`telemetry_path` that has been measured. It is reported as not earned, not faked.

## Gaps found and filled

1. `lemonade-burn-lfm2.json` sets `LB_WGPU_DEVICE`, which the plane's env allowlist refused. Plane commit 7c6d9160b adds the `LB_` prefix
   (`kEnvPrefixes`, unit test in `test_ap3.cpp`, `LBX_` still refused); `tools/check_specs.py` carries the same prefix. After the change the lfm2 spec
   gets past the env check and lemond starts the engine; the release `lemonade-burn-server` 0.1.1 then exits because it lacks the lfm2-branch flags
   (`adapter failed to start`), so that spec is still unrun end to end.
2. Strata, PolyStrata and Nauti-Strata `/v1/models` carry no `owned_by`, so lemond refused the load with
   `does not report owned_by=strata; refusing a mislabelled engine`. Plane commit eb6186c5c adds `identity.unreported` (skips the check; wrong owners are still
   refused; two live checks added). The three specs set it. The better fix is one line in the engine's `serve/server.py`.
3. New specs: `bonsaicpu.json`, `strata.json`, `nautistrata.json`. `strata.json` uses `strata serve --profile {model}`; `strata launch` is the interactive installer path.

## Observations

- `GET /api/v1/health` reports `"device": "gpu"` for CPU specs (`default_device` is `cpu`): the plane does not read it.
- The conformance tool measures GPU memory with nvidia-smi. On this host the forge moves GPU 0 by gigabytes, so one bonsaicpu run was marked
  `L1 FAIL ... held {0: 1970}; after unload NOT freed` for a CPU-only engine; the rerun passed. On a host with no GPU readings at all the tool
  cannot pass L1 (`wait_freed` is False on an empty reading). Both are tool gaps, not engine failures; not changed here.
- Nauti-Strata test: the first lemond run used the spec's `--gpu {device}` which the PolyStrata server turns into `CUDA_VISIBLE_DEVICES`, overriding
  the hidden-GPU environment. The engine briefly loaded on GPU 0 (about 0.6 GiB of experts, a few minutes, no chat) before it was stopped. The reported
  CPU-only failure comes from reruns with the `--gpu` pair removed from a test copy of the spec.
- `tests` in this repo: `cargo test` 7 passed, `python3 -m unittest tests.test_make_adapter` OK, `tools/check_specs.py` 9/9.
- Downloads live in `/mnt/fast/forge-ab/engines/{burn,nautistrata,strata}`; run scratch in `/mnt/fast/forge-ab/conf`.
