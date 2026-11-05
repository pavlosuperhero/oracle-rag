# Ingestion preparation, resume, and VDI connection checks

## What is saved

With the default `oracle ingest --resume`, two sidecars sit beside the destination
index:

- `<index>.prepared.sqlite3` stores extracted sections and chunk lists, including
  titles, text, page/line citations and ordering.
- `<index>.embeddings.sqlite3` stores validated successful embedding vectors.

Each file's extraction is committed only after extraction finishes and a second
content hash confirms that the file did not change during extraction. Chunk lists
are committed separately after chunking that file finishes. Both artifacts have
checksums, schema validation and expected item counts, and are written atomically.
An interruption cannot turn a partly written artifact into a complete file.

If 20% of files were extracted before a later file failed, the completed files'
sections remain reusable. Files still being extracted are retried. If extraction
completed but embedding failed, the next run can reuse both the prepared documents
and the successful vectors. The final `.npz` is still replaced only after all
current passages have valid embeddings; the previous usable index survives failure.

Discovery and content hashing still happen on every run. This checks the current
input set rather than trusting an old file count or modification timestamp.
Unchanged files skip parsing and chunking; modified files are prepared again.
Removed files are excluded, new files are prepared, and source/citation order is
preserved. Parser module code and pypdf/Beautiful Soup versions participate in
invalidation. Changing chunk size/overlap keeps extraction reusable and rebuilds
the affected chunk lists. Changing providers does not require extracting again.

Progress reports reused files on the Extract row and reused passages on the Chunk
and Embed rows. File completion means all sections returned by the extractor were
saved; it does not prove that OCR-free extraction captured every intended PDF page.

`--no-resume` bypasses both preparation and embedding caches for that run, without
erasing or refreshing them. `oracle clean` includes both sidecar types in its
confirmed cleanup. Preparation caches contain document text and should be protected
like the saved index; new cache files use owner-only permissions where supported.
An invalid artifact is recomputed. An unreadable database produces a storage error;
it is not silently deleted. Old unreferenced artifacts remain until cleanup.

## Checks on the Windows VDI

1. **Verify persistent storage and reuse.** Confirm that the destination index and
   both sidecars use a VDI/profile location retained across logoff or restart.
   Rerun the same command/configuration/destination. Unchanged inputs should show
   reused extraction and chunking; previously saved vectors should show reused
   embeddings. A different destination intentionally uses different sidecars.

2. **Correlate the connection reset with VDI infrastructure logs.** Record the
   failure timestamp with timezone and SDK attempt numbers from `oracle-ingest.log`.
   Ask the VDI/network team to correlate guest adapter/route events, NAT/firewall,
   outbound proxy and TLS-inspection logs for the Bedrock hostname. The client-to-VDI
   desktop session and VDI-to-AWS traffic are separate paths; a desktop disconnect
   alone does not identify the cause of an AWS socket reset.

3. **Check DNS and TCP connectivity inside the VDI.** In Windows PowerShell:

   ```powershell
   Resolve-DnsName bedrock-runtime.us-east-1.amazonaws.com
   Test-NetConnection -ComputerName bedrock-runtime.us-east-1.amazonaws.com -Port 443 -InformationLevel Detailed
   ```

   These are point-in-time checks, not proof that sustained signed requests work.
   If a proxy is required, a direct TCP test may not represent the application's
   actual route. Compare effective proxy and CA settings used by the AWS CLI,
   boto3 and aiobotocore with the VDI team. Keep proxy credentials out of reports.

4. **Compare sustained transports with controlled settings.** On the same VDI,
   compare synchronous boto3 and aiobotocore using the same small synthetic inputs,
   model, region, credentials, worker count, timeouts and total-attempt budget.
   Start with one worker and run beyond the observed embedding failure window,
   then compare with two workers. Capture elapsed request time, attempt count and
   complete nested exception. If both fail, investigate the shared network path;
   if only async fails, investigate the Windows/aiohttp transport and connection
   pooling. A short successful test or laptop comparison cannot settle this.

5. **Measure the embedding phase independently.** In the supplied runs, embedding
   elapsed times were 143.90s and 26.43s; most of the roughly ten-minute command
   duration was document preparation. Those observations do not establish a fixed
   ten-minute connection timeout. `WinError 64` records a failed network operation,
   but does not identify which device or software component caused it. Increasing
   retry counts or shortening a connection timeout alone is not a diagnosis.

References: Microsoft's [system error codes](https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-),
[Resolve-DnsName](https://learn.microsoft.com/en-us/powershell/module/dnsclient/resolve-dnsname),
and [Test-NetConnection](https://learn.microsoft.com/en-us/powershell/module/nettcpip/test-netconnection).

No live Windows/VDI network diagnosis is implied by the repository regression tests.
