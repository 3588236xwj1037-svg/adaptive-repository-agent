from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import BaseTool, tool


def _safe_path(root: str, relative_path: str) -> Path:
    root_path = Path(root).resolve()
    candidate = (root_path / relative_path).resolve()
    if candidate != root_path and root_path not in candidate.parents:
        raise ValueError("Path escapes the sandbox workspace")
    return candidate


def build_repository_tools(root: Path) -> dict[str, BaseTool]:
    """Create LangChain tools scoped to one disposable repository workspace."""

    @tool("inspect_repository")
    def inspect_repository() -> str:
        """List source and test files in the sandboxed repository."""

        files = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and ".pytest_cache" not in path.parts
            and "__pycache__" not in path.parts
        ]
        return "\n".join(sorted(files))

    @tool("read_repository_file")
    def read_repository_file(relative_path: str) -> str:
        """Read a UTF-8 text file inside the sandboxed repository."""

        path = _safe_path(str(root), relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        if path.stat().st_size > 250_000:
            raise ValueError("Refusing to read files larger than 250 KB")
        return path.read_text(encoding="utf-8")

    @tool("search_repository")
    def search_repository(term: str) -> str:
        """Find relevant matches, prioritizing source definitions over test references."""

        normalized_term = term.strip()
        if not normalized_term:
            return "No search term supplied"
        definition = None
        if re.fullmatch(r"[A-Za-z_]\w*", normalized_term):
            definition = re.compile(
                rf"^\s*(?:async\s+)?def\s+{re.escape(normalized_term)}\s*\("
            )
        matches: list[tuple[int, int, str, int, str]] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".toml", ".txt"}:
                continue
            if ".pytest_cache" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            relative_path = path.relative_to(root).as_posix()
            is_test_file = "tests" in path.parts or path.name.startswith("test_")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if normalized_term.lower() not in line.lower():
                    continue
                is_definition = bool(definition and definition.match(line))
                matches.append(
                    (
                        0 if is_definition else 1,
                        1 if is_test_file else 0,
                        relative_path,
                        line_number,
                        f"{relative_path}:{line_number}: {line.strip()}",
                    )
                )
        matches.sort()
        return "\n".join(match[-1] for match in matches[:40]) or "No matches found"

    return {
        "inspect_repository": inspect_repository,
        "read_repository_file": read_repository_file,
        "search_repository": search_repository,
    }
