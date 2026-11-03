# Oracle CLI fixes and smoke test — 2026-11-03

The failures from the initial diagnostic run have been fixed. The checkout and
installed-tool smoke results are recorded below; the installed executable was
rebuilt with `uv tool install --force --reinstall-package oracle-rag .`.

## Changes

- Added `oracle help`, including support when the configured YAML is invalid.
- Implemented source manifest listing, registration, removal, downloads, and
  ingestion without an explicit path.
- Downloads use per-source directories, validate IDs/filenames/URLs, enforce the
  configured size limit, and replace cached files only after a successful download.
- Implemented detached serving, start, stop, and restart. Startup waits for the
  actual managed server to become ready; shutdown uses a private authenticated
  loopback control endpoint rather than signalling an unverified stored PID.
- Added regression tests for failed/oversized downloads, invalid paths, caching,
  duplicate sources, stale service state, failed child startup, and live-PID safety.
- Expanded the smoke runner to cover every command and active service restart.
- Updated README usage and local-install refresh instructions while retaining
  the installation documentation already edited before this task.

## Validation

- Checkout: **46 checks passed, 0 failed**, covering all 19 command entry points
  (including the new help command), plus options and functional variants.
- Installed tool: **46 checks passed, 0 failed** across the same command matrix.
- Pytest: **291 passed**, including first-run storage initialization and preservation tests.
- Ruff: passed for `src tests scripts main.py`.
- Mypy: passed for all 24 configured source files.
- `uv build`: wheel and source distribution built successfully.
- Repository manifest: all **58 sources / 111 downloads** validate.
- Installed `oracle help` succeeds from outside the checkout.

The complete matrix uses temporary settings, documents, source manifests, skills,
indexes, local download/embedding fixtures, and real Oracle processes. It tests
JSON/text queries, evaluation, stdio MCP initialization/tool discovery/search,
foreground/debug/dev/rag serving, detached startup, active restart, stop, skill
installation, and cancelled/confirmed cleanup. The harness shuts down its servers
and deletes the fixtures. Existing user documents, indexes, and services are not
modified. The installed uv tool has been updated.

A live check in the initial diagnostic also passed ingestion and retrieval with
the existing local model `text-embedding-bge-large-en-v1.5` (1024 dimensions) at
`http://127.0.0.1:1234/v1`. The complete matrix uses deterministic embeddings;
it does not assess real-model quality or exercise Bedrock credentials.

## Repeat

```sh
uv run python scripts/smoke_all.py --uv-project . --report /tmp/oracle-smoke-uv.json
uv run python scripts/smoke_all.py --oracle /Users/Pavlo_Fabuliak/.local/bin/oracle --report /tmp/oracle-smoke-installed.json
```

The runner continues after failures and returns exit 1 if any check fails. Raw
JSON reports retain stdout, stderr, tracebacks, timings, and command arguments:

- [Fixed checkout report](/private/tmp/oracle-smoke-uv-fixed.json)
- [Fixed installed-tool report](/private/tmp/oracle-smoke-installed-fixed.json)
- [Initial live-model report](/private/tmp/oracle-smoke-live.json)

The initial diagnostic had 34 passing checks and 11 failures per environment.
All of its unsupported help/data/service paths are covered by the fixes above.

## First-run setup follow-up

The original first-run code created only the settings YAML. Initialization now
also creates the configured data and vector database directories. The default
user configuration receives the bundled source catalog when its manifest is
missing; existing manifests are preserved. Custom configurations receive an
empty catalog when theirs is missing.

The wheel and source distribution were checked for the complete catalog. The
installed executable was rebuilt and run outside the checkout to repair the
existing user setup. `~/.config/oracle/data/sources.json` now contains 58 sources
and 111 available downloads; `~/.config/oracle/vector_database/` exists. Documents
are fetched by `data download`; the actual vector index is written by `ingest`.

[Installed smoke report after initialization fix](/private/tmp/oracle-smoke-installed-bootstrap.json)
