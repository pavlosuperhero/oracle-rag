# Slate message queue
`slate.flush()` waits for all acknowledged messages to reach the durable log.
If the queue is saturated, the dispatcher responds with SLATE_317 and retries must honor the response backoff.
Audit retained messages age out after seven days.
