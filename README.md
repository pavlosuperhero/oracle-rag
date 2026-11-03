# Oracle RAG

Hybrid in-memory document retrieval with a saved NumPy index, HTTP API, and MCP tools.
Python 3.14 or later is required. The active modules live directly under `src/`;
the outdated `src/oracle_rag/` package is excluded from the wheel.

## Install and run

```sh
uv sync --group dev
uv run oracle config
```

The first command invocation initializes settings at
`~/.config/oracle/configs/config.yaml`. Use `oracle --config /path/to/config.yaml`
or `ORACLE_CONFIG` to select an existing YAML file. Relative database and data
paths resolve against the settings file's directory.

By default, embeddings use an OpenAI-compatible endpoint at
`http://127.0.0.1:1234/v1`, model `text-embedding-bge-large-en-v1.5`, and 1024
dimensions. Start that endpoint and load the model before ingestion or queries.
To use Bedrock, set `provider.provider: Bedrock`,
`provider.model: amazon.titan-embed-text-v2:0`, and configure an AWS region and
credentials. Both ingestion and serving select the configured provider.

```sh
uv run oracle ingest /path/to/documents --recursive
uv run oracle serve --foreground
# In another terminal:
uv run oracle sources
uv run oracle query "What do these documents say about caching?" --format text
```

Ingestion saves the index; serving restores it without embedding documents again.
Stop a foreground server with Ctrl-C. HTTP documentation is available at
`http://127.0.0.1:8742/docs`; MCP is mounted at `/mcp/`. `oracle mcp` provides a
stdio adapter to a running server. Use `--token` or `ORACLE_TOKEN` for bearer
authentication; binding outside loopback requires a token.

## Install the assistant skill

```sh
uv run oracle install-skill
# Or select one assistant:
uv run oracle install-skill --target codex
uv run oracle install-skill --target copilot
uv run oracle install-skill --target claude
```

The default `--target all` installs the bundled Oracle skill at user level for Claude Code
(`~/.claude/skills/oracle/SKILL.md`), Copilot
(`~/.copilot/skills/oracle/SKILL.md`), and ChatGPT desktop/Codex
(`~/.agents/skills/oracle/SKILL.md`). The Codex user-level path follows the
[official skill documentation](https://learn.chatgpt.com/docs/build-skills).
Use `@oracle` in ChatGPT desktop or `$oracle` / `/skills` in Codex CLI and the
IDE extension. If it does not appear, restart the host.

Use `--directory /path/to/skills` to override the target and install only into a custom skill root,
or `--force` to replace an existing different skill. Installation does not
start Oracle or register its MCP server; start the foreground server separately.
This local installer targets desktop/CLI discovery; it does not publish a
ChatGPT web or mobile plugin.

## Verify

```sh
uv run pytest -q
uv run ruff check src tests scripts main.py
uv run mypy
uv build
uv run python scripts/smoke_test.py --oracle "$PWD/.venv/bin/oracle"
```

The smoke script starts a temporary local embedding stub and a real Oracle server,
then exercises ingestion, index restoration, HTTP queries, source listing,
evaluation, MCP tool discovery/search, skill installation, and graceful shutdown.
It runs the CLI outside the checkout with temporary settings and documents. Pass
an installed wheel's `oracle` executable to verify that installation instead.
Provider credentials and real model quality are not exercised by this script.

## Current limitations

The new data-management and background-service modules still contain placeholders.
`oracle data ...`, ingestion without an explicit document path, detached serving
(the default `oracle serve`), and service stop/restart are not implemented yet.
Use an explicit ingestion path and `oracle serve --foreground` for now.

The refactoring specification and implementation phases are tracked in the
[GitHub roadmap](https://github.com/pavlosuperhero/oracle-rag/issues/1).
