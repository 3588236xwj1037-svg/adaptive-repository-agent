# Contributing

Thanks for improving Adaptive Repository Agent. Small, focused pull requests
are easiest to review and keep the project useful as a portfolio reference.

## Development setup

```powershell
uv sync --all-groups
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

An API key is optional for benchmark work and required only for a real local
repository task. Never commit `.streamlit/secrets.toml` or a key in code, test
fixtures, commits, screenshots, or pull-request text.

## Before opening a pull request

Run the full local check set:

```powershell
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src app.py
```

When changing workflow behavior, include a regression test that demonstrates
the changed state transition or safety guarantee. When changing the dashboard,
keep the proposal, applied diff, and before/after test evidence consistent with
the graph result.

## Scope and design principles

- Preserve the human approval gate before workspace mutation.
- Keep source repositories read-only; all work happens in disposable copies.
- Do not execute an LLM-generated shell command.
- Keep the Q-learning action space finite and explainable.
- Do not hide baseline test failures as successful post-patch verification.
- Avoid committing generated `.adaptive_agent/` run data or model credentials.

## Pull request guidance

Describe the behavior problem, the design decision, verification command and
result, and any behavior intentionally left unchanged. Keep unrelated formatting
or refactoring out of a functional pull request.
