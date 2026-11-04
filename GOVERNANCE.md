# Governance and repository permissions

Oracle RAG uses a maintainer-led model inspired by Linux kernel development.
Pavlo Fabuliak ([@pavlosuperhero](https://github.com/pavlosuperhero)) is the
project maintainer and final integrator. GitHub issues and pull requests are the
public channels for proposals and review; a kernel-style mailing list and
subsystem trees are not required for this project's current size.

## Roles

| Role | Authority |
| --- | --- |
| Anyone | Use, fork, modify, and redistribute under the license; open issues and pull requests; participate in review. |
| Contributor | Submit focused, DCO-signed patches and respond to review; retain copyright in their contributions. |
| Reviewer | Give technical feedback and testing results; public feedback alone does not grant merge authority. |
| Maintainer | Set direction, accept or reject changes, integrate patches, publish releases, and manage repository access. |

The maintainer is the code owner for every path in `.github/CODEOWNERS`, including
the license and governance files. Ownership routes review; it does not transfer
contributors' copyright. Trusted subsystem maintainers may be appointed as the
project grows, with scope recorded in CODEOWNERS and this document.

## GitHub access and main branch

The repository is public. Public access permits reading and proposing changes;
it does not grant write access. Contributors normally work from forks. The
current maintainer is the only administrator. Additional collaborator privileges
are granted explicitly by the maintainer as needed, not automatically on contribution.

Branch protection for `main` requires a pull request with at least one approval,
code-owner review, dismissal of stale approvals after new commits, and resolution
of review conversations. Force pushes and branch deletion are disabled.

Administrator bypass is retained for the sole maintainer to integrate their own
work, bootstrap repository policy, and handle exceptional maintenance. GitHub
does not allow authors to approve their own pull requests. This bypass is a
trusted administrative capability; it means review is enforced for ordinary
write-access contributors, not technically mandatory for the administrator.
The maintainer should normally use public pull requests and explain exceptions
in the commit or pull request. No automated CI or DCO status check is currently
required by branch protection.

Reviewers may request changes, but the maintainer makes the final integration
decision. A license permits reuse and redistribution; it does not require the
maintainer to accept a contribution or grant repository permissions.

## License scope

Project-owned code and documentation are licensed under **AGPL-3.0-only**, as
specified in [LICENSE](LICENSE). Separate third-party notices take precedence
for the material they cover. Dependencies, downloaded documents and books,
datasets, and model weights retain their own licenses. Cataloging or retrieving
a source through Oracle does not relicense that source.

License changes affecting contributed work must respect the rights and existing
grants of all relevant copyright holders. Maintainer authority alone does not
grant permission to relicense other contributors' work.
