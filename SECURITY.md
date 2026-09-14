# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities in a public GitHub issue.
Instead, email the maintainer at [eb2819@bath.ac.uk](mailto:eb2819@bath.ac.uk)
with:

- a description of the vulnerability and its impact;
- the affected ParaNMR version and environment;
- reliable reproduction steps or a minimal proof of concept;
- any proposed mitigation, if available.

We will acknowledge reports when practical and coordinate disclosure with the
reporter. No fixed response or remediation SLA is currently provided.

## Supported versions

ParaNMR is an actively developed research software project without an LTS
release line. Security fixes target the latest published release and the
current `main` branch.

| Version | Security support |
| --- | --- |
| Latest published release | Supported |
| Current `main` branch | Best effort during development |
| Older releases | Not supported; backports are discretionary |

The supported Python baseline is Python 3.10 and newer. Continuous integration
currently verifies Python 3.10, 3.11, and 3.12.
