# Oracle evidence as Codex sources

Oracle provides the standard MCP knowledge-source tools `search` and `fetch`,
alongside the existing `oracle` and `oracle_sources` tools. A Codex SDK or a new
generation model is not required.

## Connect the local server

From this repository, install the current code and restart the existing server:

```sh
uv tool install . --force --reinstall-package oracle-rag
oracle serve restart
codex mcp add oracle --url http://127.0.0.1:8742/mcp/
```

Use the actual configured address if different. If the server requires a bearer
token, add `--bearer-token-env-var ORACLE_TOKEN` and ensure that environment
variable is available to Codex. Do not put tokens in source URLs. Reconnect the
MCP server or start a new chat so Codex discovers the added tools.

The local loopback endpoint is for a Codex client on this machine. A cloud
client cannot reach this machine through its own `127.0.0.1`. This integration
does not deploy or publicly expose the server. Configure `server.client_url`
when the user-openable server address differs from the bind address.

## Citation contract

`search({"query": "Clean Architecture dependency rule"})` returns:

```json
{
  "results": [
    {
      "id": "stable-indexed-passage-id",
      "title": "Clean Coder Blog",
      "url": "http://127.0.0.1:8742/sources/stable-indexed-passage-id"
    }
  ]
}
```

`fetch({"id": "stable-indexed-passage-id"})` returns `id`, `title`, `text`, `url`,
and `metadata`. Metadata includes original source, source URI, citation, and
only available page/line coordinates. The complete item is one indexed passage,
not an entire book. The MCP SDK emits both `structuredContent` and matching
JSON text for these tools. All four tools are annotated read-only.

The URL opens an HTML view of the indexed passage. It escapes source markup,
does not execute source scripts, and never reads an arbitrary path supplied by
the client. The JSON view is `/documents/{id}`. Both routes retain server bearer
authentication: an ordinary browser without authenticated access will receive
401 when token protection is enabled.

Clients can also read `oracle://passages/{id}` as an MCP resource. Reuse IDs for
follow-up reads instead of repeating search. The saved index retains passage
text across server restarts. Rebuilding the index may remove or change IDs;
missing IDs return errors rather than silently substituting a different passage.
This does not create a separate permanent conversation snapshot.

## Verify the Codex Sources panel

In a chat where the Oracle MCP tools are available, ask:

> Use Oracle search to find the Clean Architecture dependency rule, fetch the
> matching passage, and answer with its source citation.

Check that Codex calls `search`, then `fetch`, and that its citation opens the
same passage URL. Finally, check the top-right Sources panel for that source.
Successful protocol tests alone do not verify the panel: the host controls
resource selection, citation rendering, and attachment persistence. If it shows
only tool output, record that limitation rather than claiming an attachment.

## Design and verification

Retrieval and ID lookup belong to the index. The presentation adapter creates
citation URLs and source pages; the HTTP and MCP adapters handle transport and
authentication. Fetch accepts an indexed identity, never a file path. This
keeps framework details outside retrieval policy and allows tests without an
embedding provider.

Tests cover matching structured/text results, search-to-fetch identity,
zero-embedding follow-up reads, resource reads, escaped HTML, unavailable IDs,
authentication, and actual bind-port citation URLs.

References consulted:

- [OpenAI MCP citation and search/fetch contracts](https://developers.openai.com/api/docs/mcp)
- [OpenAI plugin knowledge-source compatibility](https://developers.openai.com/plugins/build/mcp-server)
- Oracle: `mcp-resources-2025`, lines 1–83: resources are application-driven;
  MCP does not mandate a particular user interface.
- Oracle: `mcp-tools-2025`, lines 278–369: structured results and compatible
  JSON text representations.
- Oracle: Clean Architecture, “Interface Adapters”: presentation conversion
  belongs outside the application policy.
- Oracle: Think Python, PDF page 76: small incremental changes and readable
  function composition.
