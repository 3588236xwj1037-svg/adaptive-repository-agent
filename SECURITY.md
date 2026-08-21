# Security policy

## Supported version

Security fixes are applied to the latest code on the default branch.

## Reporting a vulnerability

Do not open a public GitHub Issue for a suspected vulnerability, leaked key, or unsafe repository-mutation path. Use the repository's private reporting channel when one is enabled, or contact the repository owner directly with:

- a concise impact description;
- reproduction steps or a minimal proof of concept;
- affected files and versions; and
- any mitigation already tested.

Do not include live credentials in a report. Acknowledge that this is a local-first educational project, not a hosted service with a response-time SLA.

## Operational guidance

- Keep LLM credentials in `.streamlit/secrets.toml` or environment variables only.
- Review the exact diff before approving every patch.
- Use a disposable or non-sensitive repository copy for experiments.
- Treat external repositories and model output as untrusted input.
- Do not disable the pytest-target allowlist or approval interruption in shared environments.
