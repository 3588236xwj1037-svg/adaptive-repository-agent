from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import tomllib
import venv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from adaptive_agent.models import TaskSpec
from adaptive_agent.tools import build_repository_tools

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures" / "calculator"
RUNS_ROOT = PROJECT_ROOT / ".adaptive_agent" / "runs"


@dataclass
class TestResult:
    passed: bool
    output: str


class BenchmarkWorkspace:
    """Creates disposable workspaces and exposes only allowlisted operations."""

    def create(self, run_id: str) -> Path:
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        destination = (RUNS_ROOT / run_id).resolve()
        if RUNS_ROOT.resolve() not in destination.parents:
            raise ValueError("Invalid run workspace")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(FIXTURE_ROOT, destination)
        return destination

    def create_from_path(self, run_id: str, source_path: Path) -> Path:
        """Copy a user-selected repository into a disposable worktree."""

        source = source_path.expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Repository path is not a directory: {source}")
        if source == RUNS_ROOT.resolve() or RUNS_ROOT.resolve() in source.parents:
            raise ValueError("Select the source repository, not an existing run workspace")
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        destination = (RUNS_ROOT / run_id).resolve()
        ignored = shutil.ignore_patterns(
            ".git",
            ".adaptive_agent",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        )
        shutil.copytree(source, destination, ignore=ignored)
        return destination

    def tools(self, root: Path):
        return build_repository_tools(root)

    def apply_patch(
        self,
        root: Path,
        task: TaskSpec,
        *,
        required_source_targets: Iterable[str] = (),
    ) -> str:
        self.validate_patch(root, task, required_source_targets=required_source_targets)
        applied: list[str] = []
        already_applied: list[str] = []
        for operation in task.patch_operations():
            path = self._patch_path(root, operation.target_file)
            content = path.read_text(encoding="utf-8")
            if operation.old_text not in content:
                already_applied.append(operation.target_file)
                continue
            path.write_text(content.replace(operation.old_text, operation.new_text, 1), encoding="utf-8")
            applied.append(operation.target_file)
        if not applied and already_applied:
            return f"Patch already applied in {', '.join(already_applied)}"
        return f"Applied {len(applied)} safe replacement(s): {', '.join(applied)}"

    def validate_patch(
        self,
        root: Path,
        task: TaskSpec,
        *,
        required_source_targets: Iterable[str] = (),
    ) -> None:
        """Validate a proposed replacement without changing the workspace."""

        operations = task.patch_operations()
        if not operations or any(
            not operation.target_file or not operation.old_text
            for operation in operations
        ):
            raise ValueError("No complete structured patch proposal is available")
        required = {
            Path(target).as_posix().lstrip("./") for target in required_source_targets if target
        }
        targets = {Path(operation.target_file).as_posix().lstrip("./") for operation in operations}
        if required and not required.intersection(targets):
            raise ValueError(
                "Patch must include the Issue's source file; expected one of: "
                + ", ".join(sorted(required))
            )
        working_content: dict[Path, str] = {}
        for operation in operations:
            path = self._patch_path(root, operation.target_file)
            content = working_content.setdefault(path, path.read_text(encoding="utf-8"))
            if operation.old_text not in content:
                if operation.new_text not in content:
                    raise ValueError(f"Expected patch context not found in {operation.target_file}")
                continue
            if content.count(operation.old_text) != 1:
                raise ValueError(
                    f"Patch context is ambiguous in {operation.target_file}; expected exactly one match"
                )
            working_content[path] = content.replace(operation.old_text, operation.new_text, 1)

    @staticmethod
    def _patch_path(root: Path, relative_path: str) -> Path:
        path = (root / relative_path).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Patch target escapes the workspace")
        if not path.is_file():
            raise ValueError(f"Patch target does not exist: {relative_path}")
        return path

    @staticmethod
    def _dependency_specs(root: Path) -> tuple[list[str], list[str]]:
        """Read dependency declarations without importing or executing repository code."""

        packages = ["pytest"]
        if sys.platform == "win32":
            packages.append("tzdata")
        requirement_files: list[str] = []
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                data = {}
            project = data.get("project", {})
            packages.extend(item for item in project.get("dependencies", []) if isinstance(item, str))
            for name, dependencies in project.get("optional-dependencies", {}).items():
                if "test" in name.lower() or "dev" in name.lower():
                    packages.extend(item for item in dependencies if isinstance(item, str))
            groups = data.get("dependency-groups", {})

            def add_group(name: str, seen: set[str]) -> None:
                """Expand PEP 735 include-group entries without executing project code."""

                if name in seen or not isinstance(groups.get(name), list):
                    return
                seen.add(name)
                for dependency in groups[name]:
                    if isinstance(dependency, str):
                        packages.append(dependency)
                    elif isinstance(dependency, dict):
                        included = dependency.get("include-group")
                        if isinstance(included, str):
                            add_group(included, seen)

            test_groups = [name for name in groups if "test" in name.lower()]
            # Some projects expose test dependencies only through a development group.
            for name in test_groups or [name for name in groups if "dev" in name.lower()]:
                add_group(name, set())
        for filename in ("requirements.txt", "requirements-test.txt", "requirements-dev.txt"):
            if (root / filename).is_file():
                requirement_files.append(filename)
        return list(dict.fromkeys(packages)), requirement_files

    @staticmethod
    def _test_python(root: Path) -> Path:
        env_dir = root / ".adaptive_agent_test_env"
        return env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def prepare_test_environment(self, root: Path) -> tuple[Path | None, str]:
        """Create an isolated venv in the disposable copy and install its test deps."""

        python = self._test_python(root)
        if not python.exists():
            try:
                venv.EnvBuilder(with_pip=True, clear=False).create(root / ".adaptive_agent_test_env")
            except OSError as exc:
                return None, f"Could not create isolated test environment: {exc}"
        installer = shutil.which("uv")
        packages, requirement_files = self._dependency_specs(root)
        commands: list[list[str]] = []
        if installer:
            base = [installer, "pip", "install", "--python", str(python)]
        else:
            base = [str(python), "-m", "pip", "install"]
        if packages:
            commands.append([*base, *packages])
        for filename in requirement_files:
            commands.append([*base, "-r", filename])
        install_project = (root / "setup.py").is_file()
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            try:
                install_project = install_project or bool(
                    tomllib.loads(pyproject.read_text(encoding="utf-8")).get("build-system")
                )
            except (OSError, tomllib.TOMLDecodeError):
                pass
        if install_project:
            commands.append([*base, "-e", "."])
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                output = (completed.stdout + "\n" + completed.stderr).strip()[-2500:]
                return None, f"Dependency setup failed in the disposable workspace:\n{output}"
        return python, "Installed project and test dependencies in the disposable workspace."

    def run_targeted_test(self, root: Path, task: TaskSpec, *, fast: bool = False) -> TestResult:
        """Run only the task's known test selector, never an arbitrary command."""

        if fast:
            passed = True
            for operation in task.patch_operations():
                source = (root / operation.target_file).read_text(encoding="utf-8")
                compile(source, operation.target_file, "exec")
                passed = passed and operation.new_text in source
            detail = (
                "Fast verifier confirmed all approved patches and valid Python syntax."
                if passed
                else "Fast verifier could not find every expected patch."
            )
            return TestResult(passed=passed, output=detail)

        python, setup_detail = self.prepare_test_environment(root)
        if python is None:
            return TestResult(passed=False, output=setup_detail)

        test_args = shlex.split(task.test_target, posix=False)
        if not test_args or any(arg.startswith("-") for arg in test_args):
            return TestResult(
                passed=False,
                output="Pytest target must contain one or more relative test paths or selectors.",
            )
        completed = subprocess.run(
            [str(python), "-m", "pytest", "-q", *test_args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        output = setup_detail + "\n" + (completed.stdout + "\n" + completed.stderr).strip()
        return TestResult(passed=completed.returncode == 0, output=output[-4000:])
