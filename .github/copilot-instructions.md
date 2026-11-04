# GitHub Copilot Repository Instructions

Before designing, writing code, refactoring, drafting issues/PRs, or answering technical questions, always run the Oracle Knowledge-First Pipeline without waiting for the user to ask:

1. **Discover the catalog**: Run `oracle sources` (or MCP `oracle_sources`) to see all indexed books, PEPs, and official docs.
2. **Select the best-fit sources**: Pick the specific `--source` IDs matching the task (`clean-architecture`, `cosmic-python-service`, `think-python`, `google-python`, `peps`, `python-asyncio`, `httpx-timeouts`, `boto3-retries`, `tdd-with-python`, `prompt-engineering`, `producing-oss`, etc.).
3. **Retrieve focused evidence**: Run `oracle query "<focused question>" --source <id> --top-k 3` (or MCP `oracle`) and read the excerpts.
4. **Apply, verify, and cite**: Ground code changes and PR/issue descriptions in the retrieved evidence and verify with `uv run pytest -q`, `uv run ruff check src tests`, and `uv run mypy`. Sign off commits with `git commit -s`.
