# lemonade-engine-adapters

One JSON spec per engine, turned into a native backend for our Lemonade fork
(`/codebase/repos/lemonade-fork`, `iced-lemcake/*`, Lemonade 2026.43.x backend API). The engine
then lists in `lemonade backends`, registers with `/api/v1/pull`, loads, unloads and routes like
any built-in model. This is the "written generator" form of the adapter maker in the Lemonade
architecture plan (section 2.6b c).

| File | What |
|---|---|
| `make_adapter.py` | the generator: `--check`, `--out <dir>`, or `--lemonade-src <fork>` (also adds the `LEMON_BACKENDS` line) |
| `engines/*.json` | engine specs: `freetoken`, `lemonade-burn` |
| `ADAPTER_PROMPT.md` | instructions for an agent to wrap a new engine, with the five gates |
| `tests/test_make_adapter.py` | `python3 -I tests/test_make_adapter.py` |

What a generated backend does on `load`: finds the engine binary in PATH (missing = error, never
faked), starts it from `args_template` with `env` (placeholders `{model}` `{port}` `{device}`),
waits for the ready route and, if the spec says so, a JSON readiness value, then checks
`owned_by` in `/v1/models` and stops the engine if it does not match. `unload` stops it.

Engines that need their own logic keep a hand-written backend (Cake: the Rust bridge, cluster
ports, runtime fingerprint).

## Verified (2026-10-28, cesarops4om, RTX 2060)

FreeToken 0.2.1 through the fork's lemond on :13606: pull, load pinned to PCI index 4 (185 s),
chat 44-47 tok/s decode via `/api/v1/chat/completions`, unload frees the card and the port; a
missing checkpoint is refused at pull.
