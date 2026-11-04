<!--
Welcome! This template guides preparation and review of a contribution.
Read https://github.com/pavlosuperhero/oracle-rag/blob/main/CONTRIBUTING.md first.

Keep the explanation proportional to the patch. A typo fix may need only a few
sentences. Mark irrelevant sections N/A with a brief reason; do not invent tests
or claim checks passed when you did not run them. Replace prompts with answers.

Before requesting review:
- Start from main and submit one logical change; separate unrelated cleanup.
- Discuss substantial features or compatibility changes in an issue first.
- Use a specific title describing the resulting behavior, not "update files".
- Review your complete diff and sign off each contribution commit yourself with
  git commit -s. A sign-off certifies the DCO; it is not a review approval.
- Open a draft PR if the approach is uncertain or the work is unfinished.
-->

## Why this is needed

<!-- Explain the user problem and why it matters. Link the issue or prior design
discussion when available. Use "Closes #..." only if this fully resolves it. -->

## Behavior and reproduction

<!-- Give a concrete before/after example. For a bug, provide minimal reproduction
steps, expected behavior, and actual behavior. For docs or refactors, explain what
becomes clearer or easier to maintain and whether runtime behavior changes. -->

## Approach and scope

<!-- Explain why this solution fits, its material tradeoffs, and alternatives you
considered when useful. Identify anything deliberately deferred. Help a reviewer
understand the decision instead of listing changed filenames. -->

## How to verify

<!-- List exact commands and observed results, including relevant regression or
failure cases. Add a short manual reproduction if needed. See README's contributor
verification commands; choose checks relevant to this patch. Report skipped or
failing checks and limitations. For a docs-only patch, checking links and examples
may be enough. Do not assume later review replaces your own testing. -->

## Compatibility and operational impact

<!-- If applicable, explain changes to CLI/HTTP/MCP behavior, configuration, saved
indexes, dependencies, authentication, or resource usage. Give migration or rollback
steps when needed. Otherwise write N/A. Use synthetic or redistributable documents
in examples; keep credentials and private corpus contents out of the PR. -->

## Reviewer attention

<!-- Point to uncertain decisions or areas needing particular expertise. Disclose
material AI assistance and how you verified its output. You remain responsible
for the entire patch. Otherwise write "No specific concerns." -->

## Ready for review

- [ ] The PR is focused, and I have reviewed the complete diff.
- [ ] I reported verification results and any known limitations above.
- [ ] Relevant tests and documentation are updated, or I explained why no updates are needed.
- [ ] I have the right to submit this work under AGPL-3.0-only and have preserved applicable third-party notices.
- [ ] Each contribution commit has my own DCO `Signed-off-by` trailer (`git commit -s`).

<!--
During review:
- Respond to feedback with reasoning, questions, or a revision; summarize material
  updates so reviewers know what to revisit. Keep discussion technical and respectful.
- Do not add someone else's sign-off or Reviewed-by/Acked-by/Tested-by endorsement
  without their consent. New commits require review of the latest revision.
- If review stalls, a polite follow-up in this PR after about a week is welcome.

For reviewers: examine correctness, failure cases, compatibility, maintainability,
verification, documentation, license provenance, and DCO trailers. Distinguish
required fixes from optional suggestions and explain the reason for each request.
Public reviews are welcome; the maintainer makes the final integration decision.
-->
