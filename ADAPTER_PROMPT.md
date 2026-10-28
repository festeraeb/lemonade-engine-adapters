# Prompt: wrap a new engine as a native Lemonade backend

Give this file to an agent together with the engine's name and where it is installed. It is
written so a small model can follow it step by step. Every step ends with a check that can fail;
report each check's real output. Never report a step as done without its check.

## What you are building

A JSON spec in `engines/<engine>.json`. `make_adapter.py` turns it into a native backend for our
Lemonade fork (`/codebase/repos/lemonade-fork`, branch `iced-lemcake/*`). lemond then starts the
engine itself, waits until it is ready, checks it is the engine it claims to be, and routes chat
to it. You do not write C++. If the spec cannot express the engine, stop and report what is
missing; do not edit generated files.

## Before you start (answer each from the engine, not from memory)

1. **Binary**: the executable name lemond will find in PATH (`which <binary>` must print a path
   on the target machine).
2. **Arguments**: the command that serves one model on one port. Find the flags for the model
   path, the host (use 127.0.0.1) and the port. Run it by hand once, exactly as you will write it.
3. **OpenAI API**: `curl http://127.0.0.1:<port>/v1/models` must return `{"data": [...]}`, and a
   POST to `/v1/chat/completions` must answer. Engines without this API need a different adapter:
   stop and report.
4. **Identity**: the `owned_by` value in `/v1/models` (`data[0].owned_by`). Lemonade refuses to
   serve when it does not match.
5. **Readiness**: which route answers when the engine is ready? Some answer `200` while still
   loading. Check `/health` during load: if it returns JSON with a status field (for example
   `{"status": "loading"}` then `{"status": "ok"}`), use `json_field`/`ready_values`/`error_values`.
6. **GPU selection**: how the engine picks a card (an env var such as `CUDA_VISIBLE_DEVICES`, or a
   flag). Use `{device}` there. NVIDIA: also set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, otherwise the
   index does not match `nvidia-smi`.
7. **Checkpoint**: one file (GGUF) or a directory? For a directory, name one file that must exist
   in it (`marker`).
8. **Support**: OS and device families (`nvidia_gpu` with sm list, `amd_gpu`, `cpu`), from the
   engine's own docs. Do not claim what you did not check.

## Write the spec

Copy `engines/freetoken.json`, then fill every field from the answers above. Rules the tool
enforces: `args_template` has `{model}` and `{port}`, is split on spaces (no quotes, `$`, pipes);
`binary` is a bare name; placeholders are only `{model}`, `{port}`, `{device}`.

Check: `python3 make_adapter.py engines/<engine>.json --check` prints three file paths. A spec
error is printed with the field to fix.

## Install, build, run

1. `git -C /codebase/repos/lemonade-fork checkout -b iced-lemcake/<engine>-<date>`
2. `python3 make_adapter.py engines/<engine>.json --lemonade-src /codebase/repos/lemonade-fork`
3. `cmake --build /codebase/repos/lemonade-fork/build --target lemond lemonade -j 24`
   Check: no `error:` lines; `build/lemond` is newer than the spec.
4. Run a separate lemond on the target machine (never touch the node's production lemond):
   copy `build/lemond`, `build/lemonade`, `build/resources` to a work dir, then
   `lemond --port 13606 --host 127.0.0.1 <work>/cache <work>/config` with the engine's binary in
   PATH (as a `systemd-run` unit with `MemoryMax`).
   Check: `lemonade --port 13606 backends` lists your recipe as `installed`.

## The five gates (all must pass; paste each output)

| Gate | Command | Pass |
|---|---|---|
| capability | `lemonade --port 13606 backends` | recipe listed, `installed`; with the binary removed from PATH it must say not found, never "installed" |
| register | `POST /api/v1/pull {"model_name":"user.<name>","recipe":"<recipe>","checkpoint":"<path>"}` | `status: success`; a wrong path returns an error naming the path |
| health / load | `POST /api/v1/load {"model_name":"user.<name>","<stem>_device":"<index>"}` | `status: success`; `/api/v1/health` shows the model with `backend_health: ready` and a pid; `nvidia-smi` shows memory on the named card only |
| serve | `POST /api/v1/chat/completions` with the model `user.<name>` | a correct answer; record tokens/s and time to first token |
| cleanup | `POST /api/v1/unload` | the engine pid is gone, the card is back to idle memory, nothing listens on the engine port |

Also run the failure case once: a model file that is not a model (or a wrong `owned_by`). Load
must fail with an error and leave no process, no port and no GPU memory behind.

## Report

A table with the five gates and the failure case: command, real output, pass or fail. Then the
spec file, the fork branch and commit, the measured tokens/s, and anything the engine needs that
the spec could not say (for example the NautiBurn scraper: an engine is not done for routing
until it either writes scrape rows or the gap is recorded in its spec `notes`).
