# Contributing to Oracle RAG

Anyone may report problems, propose changes, review patches, and submit pull
requests. Development follows a maintainer-led review process inspired by the
Linux kernel, using GitHub pull requests as the public review record.

## Submit a focused patch

1. Discuss substantial features, architecture changes, and compatibility breaks
   in an issue before implementing them.
2. Fork the repository and create a topic branch from `main`. Use a purpose-based
   prefix such as `fix/`, `feature/`, `docs/`, `refactor/`, `test/`, or `chore/`,
   followed by a short descriptive name. Do not use agent, assistant, or tool names
   as branch prefixes (for example, `codex/`, `claude/`, or `copilot/`).
3. Keep each commit focused on one logical change. Explain the problem and why
   the solution works in its commit message. Keep unrelated cleanup separate.
4. Follow the contributor setup and verification commands in [README.md](README.md).
   Run checks appropriate to the change; report the commands and results in the
   pull request. Add regression coverage for behavior changes when appropriate.
5. Sign off each contribution commit with `git commit -s`, then open a pull
   request against `main`. This adds a `Signed-off-by: Your Name <your@email>`
   trailer; it is a DCO certification, not a cryptographic signature.
6. Respond to review, update the patch, and explain material revisions. The
   maintainer decides whether the change is ready and integrates it.

The default [pull request template](.github/PULL_REQUEST_TEMPLATE.md) includes
prompts for motivation, reproduction, design choices, verification, and reviewer
attention. Keep answers proportional to the change and mark inapplicable sections
with a reason. A small documentation fix does not need a full design proposal.
Use a draft pull request when you want early feedback on unfinished work.

Keep discussion technical and respectful. Critique the patch with concrete
evidence; explain disagreements and address review feedback. Approval is based
on correctness, maintainability, scope, and the project's direction.

## Design checklist

Before requesting review, compare your diff with this checklist and report the
result in the pull request:

- A small `Protocol` port sits at each boundary, and tests inject a fake.
- Tests use public names; they do not import or patch `_private` names from `src/`.
- Each function has one purpose.
- Timeouts, retries and sizes are named constants.
- Function names and docstrings state their side effects.

`tests/test_conventions.py` enforces the private-name rule.

## Licensing and provenance

Contributions to project-owned code and documentation are accepted under
**AGPL-3.0-only**, unless a file explicitly retains separate third-party terms.
Contributors retain copyright in their work; no copyright assignment is required.
By signing off, you certify the [Developer Certificate of Origin 1.1](DCO).
Only sign off when you can make that certification yourself. Do not add another
person's sign-off or review endorsement on their behalf.

Preserve third-party copyright and license notices. Identify the origin and
license of copied code, and ensure it is compatible before submitting it. Oracle's
license does not grant rights to downloaded books, documents, model weights,
or other third-party material. Do not submit such material without permission
to redistribute it under its applicable terms.

You are responsible for the entire patch, including AI-assisted changes. Review
and verify them before submission and disclose material assistance in the pull
request. An AI tool cannot make your DCO certification for you.

## Review and integration

The maintainer reviews the technical change, verification results, licensing,
and DCO trailers before integration. Sign-offs are currently checked manually;
there is no automated DCO status check. A sign-off establishes provenance and
does not constitute technical approval.

Public review is welcome, but only the maintainer grants integration approval.
Do not add `Reviewed-by`, `Acked-by`, or `Tested-by` trailers without the named
person's explicit consent. Review approval must cover the latest revision.
See [GOVERNANCE.md](GOVERNANCE.md) for authority and repository permissions.

If a submission receives no response, a polite follow-up in the same pull request
after about a week is welcome; this is guidance, not a promised response time.
Reviewers should explain requested changes and distinguish required fixes from
optional suggestions. The maintainer should leave each submission with a clear
next step, integration decision, or explanation for closure.

This guidance draws on Karl Fogel's *Producing Open Source Software*, retrieved
through Oracle: author self-review and testing (indexed PDF page 38), and following
patches through review to a clear outcome (indexed PDF pages 158–159). The PR
structure and follow-up interval are this project's application of those principles.
The book is available at [producingoss.com](https://producingoss.com/).
