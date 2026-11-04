# Oracle RAG

Hybrid in-memory document retrieval with a saved NumPy index, HTTP API, and MCP tools.
Install it as a Python command-line package directly from the public Git repository.

## Requirements

- Python 3.14 or later and Git.
- Access to **at least one embedding provider**: Amazon Bedrock, LM Studio,
  or OpenAI. Oracle uses the selected provider to embed documents during ingestion
  and queries during retrieval.
  - **Amazon Bedrock:** AWS credentials, a configured AWS region, and access to
    the configured embedding model (for example, Amazon Titan Text Embeddings V2).
  - **LM Studio:** a running local server with an embedding model loaded. Oracle
    connects through its OpenAI-compatible API; the default URL is
    `http://127.0.0.1:1234/v1`.
  - **OpenAI:** an API key and access to an embedding model. Configure the base
    URL as `https://api.openai.com/v1` and select the model and its dimensions.

Choose the provider and embedding model before ingesting documents, and use the
same model and dimensions when querying the saved index. See
[Configure and run](#configure-and-run) for configuration details.

## Install from GitHub

### Using uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv tool install --python 3.14 "git+https://github.com/pavlosuperhero/oracle-rag.git@main"
uv tool update-shell
```

Open a new terminal if needed so the tool directory is on `PATH`, then verify:

```sh
oracle --help
oracle config
```

`uv tool install` creates an isolated environment and exposes the `oracle` command.
You can run it from any directory, without cloning the repository or using `uv run`.
uv can download Python 3.14 if it is missing. See the
[uv tool guide](https://docs.astral.sh/uv/guides/tools/) for environment management.

To update from the tracked Git branch or uninstall:

```sh
uv tool upgrade oracle-rag
uv tool uninstall oracle-rag
```

### Using pip in a virtual environment

With Python 3.14 or later installed:

```sh
python3.14 -m venv oracle-env
# macOS / Linux:
source oracle-env/bin/activate
# Windows PowerShell: .\oracle-env\Scripts\Activate.ps1

python -m pip install "git+https://github.com/pavlosuperhero/oracle-rag.git@main"
oracle --help
oracle config
```

On Windows, create the environment with `py -3.14 -m venv oracle-env`.
Activate this environment whenever you want to use `oracle`. To update, run:

```sh
python -m pip install --upgrade "git+https://github.com/pavlosuperhero/oracle-rag.git@main"
```

Both methods install from `main`. To pin a revision, replace `@main` with an
existing release tag or a full commit SHA; see
[pip's Git installation syntax](https://pip.pypa.io/en/stable/topics/vcs-support/).
The distribution is named `oracle-rag`; its executable is named `oracle`.

## Quick start

Configure one of the embedding providers listed in [Requirements](#requirements)
and make sure it is available before ingestion. The default configuration uses
LM Studio at `http://127.0.0.1:1234/v1` with
`text-embedding-bge-large-en-v1.5` loaded.

Download the bundled document sources, build the index, and start the server:

```sh
oracle data download --all
oracle ingest
oracle serve start
oracle query "What do these documents say about caching?" --format text
```

Queries default to cited evidence blocks shared with the MCP `oracle` tool:
passage ID, title, source URI, available location, and excerpt. Unicode and
newlines are preserved; XML metacharacters are escaped to keep document text
inside its excerpt. Missing page/line locators are stated explicitly rather
than invented. Ranking scores and null fields are excluded from this view.
Use `oracle query "QUESTION" --format json` for the full diagnostic response;
scripts that parse query output as JSON must now request this format explicitly.
The HTTP `/query` JSON response is unchanged. Evidence formatting does not
filter irrelevant results or establish that a passage supports a claim.

`oracle serve start` runs the server in the background. Use `oracle serve stop`
to stop it. See [Configure and run](#configure-and-run) for provider settings
and instructions for using your own documents.

## Configure and run

The first command invocation initializes settings at
`~/.config/oracle/configs/config.yaml`. Use `oracle --config /path/to/config.yaml`
or `ORACLE_CONFIG` to select an existing YAML file. Relative database and data
paths resolve against the settings file's directory.

It also creates `~/.config/oracle/data/` with the bundled `sources.json` catalog
and `~/.config/oracle/vector_database/`. Existing manifests are preserved.
Run `oracle data list` to see the available sources and
`oracle data download <source-id>` (or `--all`) to fetch documents. The index
file is created by `oracle ingest`. Custom configurations start with an empty
catalog at their configured manifest path when that file is missing.

By default, embeddings use an OpenAI-compatible endpoint at
`http://127.0.0.1:1234/v1`, model `text-embedding-bge-large-en-v1.5`, and 1024
dimensions. Start that endpoint and load the model before ingestion or queries.
To use Bedrock, set `provider.provider: Bedrock`,
`provider.model: amazon.titan-embed-text-v2:0`, and configure an AWS region and
credentials. Both ingestion and serving select the configured provider.

```sh
oracle ingest /path/to/documents --recursive
oracle serve --foreground
# In another terminal:
oracle sources
oracle query "What do these documents say about caching?" --format text
```

Ingestion saves the index; serving restores it without embedding documents again.
Stop a foreground server with Ctrl-C. HTTP documentation is available at
`http://127.0.0.1:8742/docs`; MCP is mounted at `/mcp/`. `oracle mcp` provides a
stdio adapter to a running server. Use `--token` or `ORACLE_TOKEN` for bearer
authentication; binding outside loopback requires a token.

For citation-compatible knowledge clients, Oracle also exposes MCP `search`
and `fetch`. Search returns stable passage IDs, titles, and absolute source
URLs; fetch reads that passage from the existing index without another
embedding search. See [Codex source integration](docs/codex-sources.md) for
connection steps, response contracts, and the remaining Sources-panel check.

`oracle help` also shows the command reference. To run the server in the
background after ingestion:

```sh
oracle serve
oracle serve restart
oracle serve stop
```

`oracle serve start` is an alias for starting the service. Startup waits until
the saved index is ready. By default, service logs are written to
`oracle-serve.log` beside the configured index; `--log-file` overrides this.
Stop and restart use an authenticated local control endpoint for the service
belonging to the selected configuration, host, and port.

To register and download a managed corpus, supply a direct document URL:

```sh
oracle data add my-book --title "My book" --url "https://example.org/book.pdf" --filename book.pdf
oracle data list
oracle data list --format json  # Machine-readable output for scripts
oracle data download my-book
# Or download every registered source:
oracle data download --all
oracle ingest
```

The manifest lives at the configured `data.manifest` path and uses a
`{"resources": [...]}` object, as in this repository's `sources.json`. Downloads
are stored under `<data.directory>/<source-id>/<filename>`. Existing downloads
are reused unless `--force` is specified. Downloads show overall file progress
and bytes transferred for the current file; use `--no-progress` to hide the bars.
Progress is written to stderr, keeping stdout as a list of downloaded paths.
`oracle data remove my-book` removes
the registration; cached files and the existing index remain until cleanup or
re-ingestion. Ingestion without a path requires every registered download to
be present. `oracle clean` confirms before deleting saved indexes;
`oracle clean --all` also deletes managed files and the manifest.

## Install the assistant skill

```sh
oracle install-skill
# Or select one assistant:
oracle install-skill --target codex
oracle install-skill --target copilot
oracle install-skill --target claude
oracle install-skill --target antigravity
```

The default `--target all` installs the bundled Oracle skill at user level for Claude Code
(`~/.claude/skills/oracle/SKILL.md`), Copilot
(`~/.copilot/skills/oracle/SKILL.md`), and ChatGPT desktop/Codex
(`~/.agents/skills/oracle/SKILL.md`). The Codex user-level path follows the
[official skill documentation](https://learn.chatgpt.com/docs/build-skills).
Use `@oracle` in ChatGPT desktop or `$oracle` / `/skills` in Codex CLI and the
IDE extension. If it does not appear, restart the host.

`--target antigravity` installs to both `~/.gemini/config/skills/oracle/SKILL.md`
(Antigravity app/IDE) and `~/.gemini/antigravity-cli/skills/oracle/SKILL.md`
(Antigravity CLI), following the
[Antigravity skill documentation](https://antigravity.google/docs/skills).
These destinations are also included in `--target all`. Invoke the skill with `/oracle`.

Use `--directory /path/to/skills` to override the target and install only into a custom skill root,
or `--force` to replace an existing different skill. Installation does not
start Oracle or register its MCP server; start the foreground server separately.
This local installer targets desktop/CLI discovery; it does not publish a
ChatGPT web or mobile plugin.

## Contributor setup and verification

Clone the repository only if you want to edit the code or run its test suite:

```sh
git clone https://github.com/pavlosuperhero/oracle-rag.git
cd oracle-rag
uv sync --group dev
uv run oracle --help
```

If you installed a local checkout with `uv tool install .`, rebuild the installed
tool after editing its source:

```sh
uv tool install --force --reinstall-package oracle-rag .
```

`--reinstall-package` refreshes the package build; `--force` alone can reuse a
cached wheel from before the source edits.

The active modules live directly under `src/`; the outdated `src/oracle_rag/`
package is excluded from the wheel. Inside the checkout, use `uv run` to select
the development environment:

```sh
uv run pytest -q
uv run ruff check src tests scripts main.py
uv run mypy
uv build
uv run python scripts/smoke_test.py --oracle "$PWD/.venv/bin/oracle"
uv run python scripts/smoke_all.py --uv-project . --report /tmp/oracle-smoke.json
```

The smoke script starts a temporary local embedding stub and a real Oracle server,
then exercises ingestion, index restoration, HTTP queries, source listing,
evaluation, MCP tool discovery/search, skill installation, and graceful shutdown.
It runs the CLI outside the checkout with temporary settings and documents. Pass
an installed wheel's `oracle` executable to verify that installation instead.
Provider credentials and real model quality are not exercised by this script.

The complete smoke runner executes every command, including managed downloads,
ingestion without a path, detached service start/stop/restart, dev mode, stdio
MCP, and confirmed cleanup. It continues after failures, writes detailed JSON
results, and returns a nonzero exit code if any check fails. Use `--oracle` with
an installed executable instead of `--uv-project` to test a tool installation.

The refactoring specification and implementation phases are tracked in the
[GitHub roadmap](https://github.com/pavlosuperhero/oracle-rag/issues/1).
