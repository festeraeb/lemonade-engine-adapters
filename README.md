# lemonade-engine-adapters

One JSON spec per engine, turned into a native backend for our Lemonade fork (Iced Lemon Cake,
Lemonade 2026.43.x backend API; the fork is not public yet, so the generated backends build only there for now). The engine
then lists in `lemonade backends`, registers with `/api/v1/pull`, loads, unloads and routes like
any built-in model. Part of NautiK9 Lemonade Stand.

| File | What |
|---|---|
| `src/`, `Cargo.toml` | **the tool**: Rust binary `make-adapter` (`cargo build --release`), same flags: `--check`, `--out <dir>`, or `--lemonade-src <fork>` (also adds the `LEMON_BACKENDS` line) |
| `make_adapter.py` | the original Python generator, kept until the operator retires it |
| `engines/*.json` | engine specs: `freetoken`, `lemonade-burn` |
| `engines/drafts/*.json` | unverified drafts for Bitter Lemons (`rea`, `movidius`); not built or tested |
| `ADAPTER_PROMPT.md` | instructions for an agent to wrap a new engine, with the five gates |
| `tests/make_adapter.rs` | `cargo test` (port of the 7 Python tests) |
| `tests/test_make_adapter.py` | `python3 -I tests/test_make_adapter.py` (old) |

`make_adapter.py` is superseded by the Rust binary `make-adapter`. Parity check (2026-10-29): for
`engines/freetoken.json` and `engines/lemonade-burn.json`, `diff -r` of the `--out` trees from both tools
is empty (byte-identical), `--check` output and the validation error text match on broken specs, and
`--lemonade-src` on a copy of the fork's CMakeLists.txt gives identical results.

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

## Licence

Apache License 2.0, see `LICENSE`.
