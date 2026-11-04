# PolyStrata adapter (`engines/polystrata.json`)

Status: spec written and checked with `make-adapter --check`, `--out` and `cargo test` on 2026-10-31. Not built into a
lemond and not loaded on any node: PolyStrata is CUDA-only ("Not yet: AMD cards" in its README) and the NVIDIA cards
were between nodes. The five gates in `ADAPTER_PROMPT.md` are therefore open.

## What the spec says

| Field | Value | Where it comes from |
|---|---|---|
| binary | `polystrata-serve` | a wrapper you put on lemond's PATH (below); PolyStrata ships `serve/server.py`, not a binary |
| args | `--engine strata --config {model} --host 127.0.0.1 --port {port} --gpu {device}` | `serve/server.py` flags |
| `{model}` | the run config JSON that PolyStrata's setup writes (`strata-<name>.json`) | the GGUF path, tokenizer and `model_name` live inside it |
| `{device}` | nvidia-smi index | `--gpu`; `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set in `env` |
| ready | `GET /health`, `status == ok` | the HTTP server starts only after the engine is ready |
| owned_by | `polystrata` | **not yet emitted by PolyStrata** (see gap) |

Wrapper (`/usr/local/bin/polystrata-serve`, mode 755):

```sh
#!/bin/sh
exec python3 /path/to/PolyStrata/serve/server.py "$@"
```

## Gap to close before the identity gate can pass

PolyStrata's `/v1/models` objects carry `id`, `object`, `status`, `meta`, `architecture` and no `owned_by`.
The generated backend refuses an engine whose `owned_by` differs from the spec. Either PolyStrata adds
`"owned_by": "polystrata"` to those objects (one line in `serve/server.py`), or the adapter generator gets an
option to skip the check. Until then, PolyStrata's own docs (`docs/LEMONADE.md`) describe the working route that needs no
adapter: `lemonade cloud install polystrata --base-url http://127.0.0.1:8095/v1 --allow-insecure-http`.

## Loading LFM2.5-24B-A2B on a CUDA node

Licence: LFM Open License v1.0 (Liquid AI), not open source; commercial use only for companies under USD 10 million yearly
revenue.

1. Install PolyStrata and the LFM2.5 family: `./setup.sh --family lfm25 --model UD-Q3_K_XL` (10.9 GB; setup downloads,
   checks the SHA-256 and writes `strata-lfm25-UD-Q3_K_XL.json`). It offers UD-Q4_K_XL, UD-Q3_K_XL, QAD-Q4_0, UD-Q6_K_XL and
   Q8_0; other quantisations (for example Unsloth's IQ3_XXS) are not in its pinned list, so they are not loaded this way.
   PolyStrata's `docs/LFM.md`: an 8 GB card holds about 67% of the Q3 experts, the rest stay in RAM (about 16 GB beside the card).
2. Put `polystrata-serve` on the PATH of the node's lemond (a `systemctl edit lemond` drop-in `Environment=PATH=...`).
3. Build lemond with the generated backend (`ADAPTER_PROMPT.md`, "Install, build, run"), then:

```sh
lemonade pull user.lfm25 --recipe polystrata --checkpoint /path/to/PolyStrata/strata-lfm25-UD-Q3_K_XL.json
lemonade load user.lfm25 --pinned          # polystrata_device = the nvidia-smi index, e.g. 0
curl http://127.0.0.1:13305/api/v1/chat/completions -d '{"model":"user.lfm25","messages":[{"role":"user","content":"hi"}]}'
```

Exact option names for the device come from the generated backend (`<stem>_device`, as in the FreeToken run in the README).
LFM2.5 thinks by default: leave room in `max_tokens`, or an answer can come back with empty `content`.
