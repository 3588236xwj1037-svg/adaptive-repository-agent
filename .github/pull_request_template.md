## Summary

Describe the behavior change and why it is needed.

## Verification

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check .`
- [ ] `uv run python -m compileall -q src app.py`
- [ ] Added or updated regression coverage when behavior changed

## Safety impact

- [ ] This change preserves source-repository isolation
- [ ] This change preserves approval before mutation
- [ ] This change does not execute model-generated shell commands

Describe any safety boundary affected by this pull request.
