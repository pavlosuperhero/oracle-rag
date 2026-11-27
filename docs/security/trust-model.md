# Oracle RAG Security & Trust Model

This document defines the supported deployment boundaries, threat assumptions, enforceable runtime controls, and observability privacy contract for Oracle RAG and Oracle Evidence Lab.

---

## 1. Deployment Modes & Trust Boundaries

### Supported Deployment Models

| Mode | Topology | Authentication | Supported Guarantees |
| --- | --- | --- | --- |
| **Single-user / local workstation** (default) | Loopback HTTP/MCP server (`127.0.0.1`), local filesystem corpus, local LM Studio/OpenAI or user AWS Bedrock credentials | Optional static Bearer token (`ORACLE_TOKEN`); managed background control port always uses a per-launch 256-bit token stored in a `0o600` state file | Local-process isolation, DNS-rebinding defenses, SSRF-safe managed downloads, parser resource ceilings, default-deny secret/content logging |
| **Shared single-corpus team service** | Authenticated HTTP/MCP server fronted by TLS termination (reverse proxy) serving one shared corpus to a trusted group | **Required** static Bearer token (`ORACLE_TOKEN`) with constant-time verification (`secrets.compare_digest`) | Uniform read-only access to the shared index for holders of the bearer token; request correlation, bounded metrics, and redacted errors |

### Explicit Non-Goals (Unsupported Guarantees)

- **No multi-tenant isolation or per-user ACL enforcement**: A single `MemoryIndex` serves all authenticated callers from one shared in-memory snapshot. Do **not** deploy one Oracle server to serve mutually untrusted tenants or confidential per-user corpora unless document-level ACL enforcement (Section 4) is implemented and verified.
- **No TLS termination in-process**: When exposing Oracle beyond loopback, terminate TLS at a reverse proxy and enforce network ingress policy outside the process.
- **No sandboxed execution of source documents**: Source files are parsed strictly as data (never executed), and HTML source views (`/sources/{passage_id}`) render escaped text under `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'` and `X-Content-Type-Options: nosniff`.
- **Untrusted reference data boundary**: Indexed documents and retrieved passages are untrusted reference data, never executable instructions. MCP tools (`oracle`, `fetch`) and XML evidence formatting explicitly frame passages as reference data to resist indirect prompt injection.

---

## 2. Permissioned Source Roots, Symlink Confinement & Parser Resource Limits

### Filesystem & Path Confinement

1. **Managed Downloads (`src/data/data.py`)**:
   - Manifest resource IDs are constrained to `[a-z0-9]+(?:-[a-z0-9]+)*`.
   - Download filenames must be plain basenames without `/`, `\`, `:`, NUL, `.`, or `..`.
   - `download_path()` verifies that neither the resource directory nor destination file is a symlink, that both resolve strictly inside `settings.data_path`, and that the target cannot overwrite `sources.json`.
   - Downloads stream into a temporary file in the destination directory and atomically replace the target via `os.replace()`. On any HTTP error, policy violation, byte-limit overflow, or `asyncio.CancelledError`, the temporary file is unlinked in a `finally` block.
2. **Document Discovery (`src/documents/chunks.py`)**:
   - `discover(path, recursive=..., allowed_root=...)` checks `is_symlink()` **before** resolving `path`, rejecting direct symlink files/directories.
   - During directory traversal (`recursive=True` or `False`), symlinked files and symlinked subdirectories are never followed (`os.walk(..., followlinks=False)` plus explicit `is_symlink()` checks).
   - When `allowed_root` is provided, `discover()` rejects any path or intermediate symlink segment that escapes the resolved `allowed_root`, enforcing a permissioned source root.
3. **Managed Service State (`src/service/service.py`)**:
   - Background service state files are created with `O_CREAT | O_EXCL` at mode `0o600`, symlink state files are refused, and termination requires verified supervisor ownership and matching PID/token identity.

### Parser Resource Limits (`src/documents/chunks.py`)

To prevent memory exhaustion and PDF decompression bombs during ingestion, `extract()` and `extract_html()` enforce named default ceilings (overrideable via keyword arguments for constrained tests or specialized workflows):

| Constant | Default Limit | Protects Against |
| --- | --- | --- |
| `MAX_DOCUMENT_BYTES` | `50 MiB` (`52,428,800` bytes) | Oversized raw file reads before opening into memory |
| `MAX_HTML_BYTES` | `20 MiB` (`20,971,520` bytes) | Pathological HTML DOM parsing CPU/memory spikes |
| `MAX_PDF_PAGES` | `2,000` pages | Extreme page-count PDF exhaustion |
| `MAX_EXTRACTED_CHARS` | `25,000,000` chars | PDF stream decompression bombs and oversized extracted text |

---

## 3. Outbound Download & Network Destination Policy

### Default SSRF, Redirect & DNS-Rebinding Controls (`src/data/network_policy.py`, `src/data/data.py`)

`download_resources()` and `download_resources_async()` enforce `NetworkPolicy` by default:

- **Scheme & credential restrictions**: Only `http://` and `https://` URLs with a non-empty host are allowed; embedded credentials (`user:pass@host`) are rejected both in `Download` validation and in `NetworkPolicy.validate_url()`.
- **IP destination filtering**: Hostnames are resolved via the injectable `DnsResolver` port (`SystemDnsResolver` by default) and every returned IP (as well as IP literals) is classified. Non-global, private (RFC 1918 / ULA), loopback (`127.0.0.0/8`, `::1`), link-local (including cloud metadata `169.254.169.254` and `fe80::/10`), CGNAT (`100.64.0.0/10`), multicast, reserved, unspecified (`0.0.0.0`, `::`), scoped zone-ID literals, and IPv4-mapped/6to4/Teredo IPv6 addresses encoding disallowed IPv4 targets (e.g., `::ffff:127.0.0.1`) are **blocked by default**.
- **Manual hop-by-hop redirect validation**: `httpx.AsyncClient` is configured with `follow_redirects=False`. Redirect responses (`301`/`302`/`303`/`307`/`308`) are followed manually up to `MAX_REDIRECTS = 5`, validating every redirect `Location` target against `NetworkPolicy` before sending the next request. Redirect loops and redirects from a public host to private/metadata IPs are blocked.
- **DNS rebinding defense**: Immediately before each request, `verify_before_connect()` re-validates DNS resolution and rejects targets whose resolved IP set flips to a disallowed or disjoint address set. For direct non-proxied transfers, `pinned_request_target()` pins the connection to the validated IP while preserving the `Host` header and TLS `sni_hostname`.
- **Error redaction**: Any raised `DataError` redacts credentials and signed URL query parameters via `redact_url()` and `redact_text()`.

### Explicit Local / Private-Source Exceptions

Authorized local single-user workflows remain supported via explicit opt-in rather than silently weakening defaults:

1. **Programmatic policy opt-in**: Pass `network_policy=NetworkPolicy(allow_private_hosts=True)` or `network_policy=NetworkPolicy(allowed_hosts={"docs.internal.local"})` to `download_resources()` / `download_resources_async()`.
2. **Environment opt-in for local download fixtures or intranet mirrors**: Set `ORACLE_ALLOW_PRIVATE_DOWNLOADS=1` (`true`/`yes`) to enable `allow_private_hosts=True` in `default_network_policy()`. Never set this in untrusted or internet-facing shared deployments.
3. **Local embedding endpoints**: `OpenAIEmbedding` defaults to `http://127.0.0.1:1234/v1` (LM Studio / local OpenAI-compatible runtime) and continues to work out-of-the-box without altering download SSRF rules.

---

## 4. HTTP & MCP Authentication Boundaries & Future Document-Level ACL Strategy

### Current Auth & Transport Boundaries (`src/runtime/server.py`)

- **Host & Origin Protection**: `TrustedHostMiddleware` restricts `Host` headers to loopback (`127.0.0.1`, `localhost`, `[::1]`, `testserver`) and the configured bind host. Mounted MCP Streamable HTTP (`TransportSecuritySettings`) enables DNS-rebinding protection for allowed hosts and origins.
- **Bearer Token Coverage**: When `token` (`ORACLE_TOKEN`) is configured, the HTTP authentication middleware enforces constant-time comparison (`secrets.compare_digest`) on **every** non-exempt route:
  - Protected: `/stats`, `/sources`, `/documents/{passage_id}`, `/sources/{passage_id}`, `/query`, and `/mcp` (including `/mcp/`, all MCP tools `oracle`, `oracle_sources`, `search`, `fetch`, and MCP resources `oracle://sources`, `oracle://passages/{id}`).
  - Exempt (public operational/schema endpoints): `/health`, `/docs`, `/openapi.json`.
  - Unauthenticated requests receive `401 Unauthorized` with `WWW-Authenticate: Bearer` and an `X-Correlation-ID` response header.
- **Source Filters Are NOT ACLs**: `Query.source` (HTTP `/query` and MCP `oracle` tool) is a caller-supplied case-insensitive substring filter over source paths used for narrowing search scope. It provides **zero** authorization or tenant isolation; callers can omit or change `source` at will.

### Future Document-Level ACL Strategy

If Oracle is evolved to serve multi-tenant or permissioned enterprise corpora, implement ACLs before claiming tenant isolation:

1. **Authenticated Principal Context**: Replace the shared static bearer token with verifiable principal identities (e.g., OIDC/JWT claims or mTLS client identities) resolved in middleware into an immutable request principal context.
2. **Ingestion-Time Security Metadata**: Attach immutable ACL metadata (`tenant_id`, `allowed_principals` / `security_groups`) to each ingested `Chunk` and persist it in the index manifest schema.
3. **Pre-Retrieval Filtering (Defense in Depth)**: Filter candidate chunks by principal authorization **before** BM25 scoring, vector similarity ranking, deduplication, and citation URL issuance — and enforce the identical ACL check on `/documents/{passage_id}`, `/sources/{passage_id}`, `/sources`, and MCP `fetch` / resource reads so passage IDs cannot act as bearer capabilities.
4. **Cache & Telemetry Isolation**: Partition any query/embedding/preparation caches by authorization policy hash, and never emit principal identifiers or document paths in metric labels.

---

## 5. Telemetry, Logging & Privacy Contract

### Default-Deny Sensitive Content Policy (`src/runtime/telemetry.py`, `src/runtime/runtime.py`, `src/embeddings/*`, `src/runtime/server.py`)

By default across `INFO`, `WARNING`, `ERROR`, and `DEBUG` levels:

- **Forbidden by default**: Raw user queries, raw document/passage excerpts, `Authorization: Bearer ...` / `Basic ...` headers, API keys (`sk-...`, `ghp_...`, `xox...`), AWS access/secret/session keys (`AKIA...`, `ASIA...`, `aws_secret_access_key`), embedded URL credentials, signed URL query parameters (`X-Amz-Signature`, `X-Amz-Credential`, `X-Amz-Security-Token`, `Signature`, `token`, `api_key`, `key`, `secret`, `password`), and absolute user home paths (`/Users/<user>/...`, `/home/<user>/...`, `C:\Users\<user>\...`).
- **Handler-Level Redaction**: `setup_logging()` attaches `RedactingLogFilter` to every configured handler (both terminal and rotating file handlers). The filter scrubs log messages, formatted arguments, exception tracebacks (`exc_info`/`exc_text`), and `stack_info`.
- **Wire-Level Suppression**: Third-party HTTP/SDK wire loggers (`boto3`, `botocore`, `urllib3`, `httpx`, `httpcore`) are clamped to `WARNING`.
- **Provider & Index Input Logging**: `BedrockEmbedding`, `OpenAIEmbedding`, and `MemoryIndex` log only structural metadata (`model`, `query_fingerprint`, `chars=<len>`, token counts, durations, scores, redacted endpoints) at `DEBUG` by default. Raw query/passage content logging requires explicit opt-in (`allow_content_debug_logs=True`).
- **HTTP/MCP Error Sanitization**: `EmbeddingError`, `IndexNotReady`, and unhandled exceptions are redacted via `redact_text()` before logging or returning JSON/MCP error details; unhandled internal errors return a generic `"Internal server error."` body with the correlation ID.
- **Structured Diagnostic Events & Retrieval Traces**: `DiagnosticEvent` is a frozen dataclass with an explicit schema (`schema_version`, `event_type`, `stage`, `correlation_id`, `duration_ms`, `status`, `error_category`, `counts`, `index_generation_id`, `retrieval_profile_id`) and runtime validation rejecting forbidden raw-content/secret keys in `counts`. `RetrievalTrace` / `CandidateTrace` (`src/models/provenance.py`, `src/data/index.py`) record only hashed `query_fingerprint`, passage IDs, safe relative `source_name`s, and bounded per-stage ranks/scores — zero raw queries or passage excerpts.

### Low-Cardinality Metric Label Policy

`MetricsRegistry` enforces an explicit allowlist (`ALLOWED_METRIC_LABELS`):

- **Allowed labels**: `stage`, `status`, `error_category`, `route`, `method`, `cache_outcome`.
- **Rejected labels**: Any unlisted or unbounded label (including `query`, `passage_id`, `source_path`, `correlation_id`, `url`) raises `ValueError`.
- **Route normalization**: Dynamic URLs (`/documents/<id>`, `/sources/<id>`, arbitrary 404 paths) are normalized to bounded route templates (`/documents/{passage_id}`, `/sources/{passage_id}`, `/other`) before recording metrics.
- **Error taxonomy**: `ErrorCategory` bounds failures to 9 stable categories: `authentication`, `authorization`, `transport`, `rate_limit`, `parser`, `corrupt_index`, `incompatible_profile`, `insufficient_evidence`, `internal`.
- **Correlation IDs**: `X-Correlation-ID` is propagated on every HTTP response, sanitized against `^[A-Za-z0-9._-]{1,128}$` (invalid values are replaced with a newly minted UUID hex ID to prevent header/log injection), and kept strictly out of metric labels.

### Opt-In Content Debugging & Retention Guidance

- Enable raw content debug logging (`allow_content_debug_logs=True`) **only** on isolated local developer machines while reproducing a parser or embedding defect.
- Write debug logs to a user-private directory (`0o700` / `0o600`), keep rotating log budgets bounded (`logging.max_bytes`, `logging.backup_count`), and delete debug logs immediately after diagnosis (recommended maximum retention: **24 hours** locally; **0 days / disallowed** in shared or CI artifacts).
