import sys
from pathlib import Path

import pytest
from langgraph.types import Command

from adaptive_agent import workspace as workspace_module
from adaptive_agent.graph import RepositoryAgentGraph
from adaptive_agent.models import PatchOperation, TaskSpec
from adaptive_agent.planner import FilePatch, PatchProposal, TaskPlanner
from adaptive_agent.rl import QLearningPolicy
from adaptive_agent.runtime import AgentRuntime
from adaptive_agent.tools import build_repository_tools
from adaptive_agent.workspace import BenchmarkWorkspace
from adaptive_agent.workspace import TestResult as WorkspaceTestResult


def test_auto_approved_task_completes_in_disposable_workspace(tmp_path) -> None:
    runtime = AgentRuntime(policy_path=tmp_path / "policy.json")

    outcome = runtime.run_task("division-by-zero", auto_approve=True, epsilon=0.0)
    result = outcome["result"]

    assert result["status"] == "completed"
    assert result["test_passed"] is True
    assert result["steps"] == 5
    assert "Targeted test passed" in result["events"][-1]["detail"]
    assert result["patches"] == [
        {
            "target_file": "src/calculator.py",
            "old_text": "    if denominator == 0:\n        return 0\n",
            "new_text": '    if denominator == 0:\n        raise ValueError("denominator cannot be zero")\n',
        }
    ]


def test_benchmark_result_retains_the_exact_patch_for_dashboard_review(tmp_path) -> None:
    runtime = AgentRuntime(policy_path=tmp_path / "policy.json")

    result = runtime.run_task("decimal-average", auto_approve=True, epsilon=0.0)["result"]

    assert result["status"] == "completed"
    assert result["patches"] == [
        {
            "target_file": "src/calculator.py",
            "old_text": "    return sum(values) // len(values)\n",
            "new_text": "    return sum(values) / len(values)\n",
        }
    ]
    assert "Applied 1 safe replacement" in result["events"][3]["detail"]


def test_graph_interrupts_before_mutating_without_auto_approval(tmp_path) -> None:
    runtime = AgentRuntime(policy_path=tmp_path / "policy.json")

    outcome = runtime.run_task("decimal-average", auto_approve=False, epsilon=0.0)
    result = outcome["result"]

    assert "__interrupt__" in result
    resumed = runtime.resume(outcome["config"], approved=True)["result"]
    assert resumed["status"] == "completed"
    assert resumed["test_passed"] is True


def test_local_repository_is_copied_without_source_mutation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(workspace_module, "RUNS_ROOT", tmp_path / "runs")
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("private", encoding="utf-8")
    (source / "app.py").write_text("print('hello')\n", encoding="utf-8")

    workspace = BenchmarkWorkspace().create_from_path("local-copy-test", source)

    assert (workspace / "app.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert not (workspace / ".git").exists()
    assert (source / "app.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_provider_json_proposal_parser_accepts_fenced_json() -> None:
    proposal = TaskPlanner._parse_json_proposal(
        "```json\n"
        '{"plan":"fix it","target_file":"src/app.py",'
        '"old_text":"old","new_text":"new","test_target":"tests"}'
        "\n```"
    )

    assert isinstance(proposal, PatchProposal)
    assert proposal.target_file == "src/app.py"


def test_targeted_test_accepts_multiple_pytest_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(workspace_module, "RUNS_ROOT", tmp_path / "runs")
    workspace_manager = BenchmarkWorkspace()
    monkeypatch.setattr(
        workspace_manager,
        "prepare_test_environment",
        lambda _root: (Path(sys.executable), "Prepared test environment."),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_one.py").write_text("def test_one():\n    assert True\n", encoding="utf-8")
    (workspace / "tests" / "test_two.py").write_text("def test_two():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    task = TaskSpec(
        task_id="multi-test-target",
        title="Run multiple tests",
        objective="Run the calculator tests",
        target_file="tests/test_one.py",
        search_term="divide",
        old_text="",
        new_text="",
        test_target="tests/test_one.py tests/test_two.py",
        acceptance_note="tests pass",
    )

    result = workspace_manager.run_targeted_test(workspace, task)

    assert result.passed is True


def test_patch_validation_rejects_missing_or_ambiguous_context(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "module.py"
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    manager = BenchmarkWorkspace()

    missing = TaskSpec(
        task_id="missing-context",
        title="Missing context",
        objective="Replace a line",
        target_file="module.py",
        search_term="value",
        old_text="value = 2\n",
        new_text="value = 3\n",
        test_target="tests",
        acceptance_note="tests pass",
    )
    ambiguous = TaskSpec(
        task_id="ambiguous-context",
        title="Ambiguous context",
        objective="Replace a line",
        target_file="module.py",
        search_term="value",
        old_text="value = 1\n",
        new_text="value = 3\n",
        test_target="tests",
        acceptance_note="tests pass",
    )

    with pytest.raises(ValueError, match="Expected patch context not found"):
        manager.validate_patch(workspace, missing)
    with pytest.raises(ValueError, match="Patch context is ambiguous"):
        manager.validate_patch(workspace, ambiguous)


def test_multi_file_patch_updates_source_and_regression_test(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src"
    tests = workspace / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "logic.py").write_text("return False\n", encoding="utf-8")
    (tests / "test_logic.py").write_text("assert False\n", encoding="utf-8")
    task = TaskSpec(
        task_id="multi-file",
        title="Change source and test",
        objective="Fix the result and add a regression test",
        target_file="src/logic.py",
        search_term="return False",
        old_text="return False\n",
        new_text="return True\n",
        test_target="tests/test_logic.py",
        acceptance_note="tests pass",
        patches=(
            PatchOperation("src/logic.py", "return False\n", "return True\n"),
            PatchOperation("tests/test_logic.py", "assert False\n", "assert True\n"),
        ),
    )
    manager = BenchmarkWorkspace()

    manager.validate_patch(workspace, task, required_source_targets=["src/logic.py"])
    detail = manager.apply_patch(workspace, task, required_source_targets=["src/logic.py"])

    assert "2 safe replacement" in detail
    assert (source / "logic.py").read_text(encoding="utf-8") == "return True\n"
    assert (tests / "test_logic.py").read_text(encoding="utf-8") == "assert True\n"


def test_patch_can_delete_an_exact_source_block(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "module.py"
    source.write_text(
        "value = float(value)\nif value < 0:\n    raise ValueError('invalid')\nreturn value\n",
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="delete-guard",
        title="Delete a guard",
        objective="Allow negative values",
        target_file="module.py",
        search_term="value < 0",
        old_text="if value < 0:\n    raise ValueError('invalid')\n",
        new_text="",
        test_target="tests",
        acceptance_note="The guard is removed.",
    )

    detail = BenchmarkWorkspace().apply_patch(
        workspace, task, required_source_targets=["module.py"]
    )

    assert "1 safe replacement" in detail
    assert source.read_text(encoding="utf-8") == "value = float(value)\nreturn value\n"


def test_patch_parser_allows_an_empty_replacement_for_a_deletion() -> None:
    proposal = TaskPlanner._parse_json_proposal(
        '{"plan":"Remove the obsolete guard.","patches":[{'
        '"target_file":"src/module.py","old_text":"if blocked: raise ValueError()\\n",'
        '"new_text":""}],"test_target":"tests/test_module.py"}'
    )

    assert proposal.patches[0].new_text == ""


def test_patch_that_only_changes_tests_is_rejected_before_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src"
    tests = workspace / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "logic.py").write_text("return False\n", encoding="utf-8")
    (tests / "test_logic.py").write_text("assert False\n", encoding="utf-8")
    task = TaskSpec(
        task_id="tests-only",
        title="Tests only",
        objective="Fix the result",
        target_file="tests/test_logic.py",
        search_term="return False",
        old_text="assert False\n",
        new_text="assert True\n",
        test_target="tests/test_logic.py",
        acceptance_note="tests pass",
        patches=(PatchOperation("tests/test_logic.py", "assert False\n", "assert True\n"),),
    )

    with pytest.raises(ValueError, match="must include the Issue's source file"):
        BenchmarkWorkspace().validate_patch(
            workspace, task, required_source_targets=["src/logic.py"]
        )


def test_dependency_specs_include_test_group_and_requirements_files(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = [\"runtime-lib>=1\"]\n"
        "[dependency-groups]\ntests = [\"test-lib>=2\"]\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements-test.txt").write_text("extra-test-lib>=3\n", encoding="utf-8")

    packages, requirement_files = BenchmarkWorkspace._dependency_specs(tmp_path)

    expected = ["pytest"]
    if sys.platform == "win32":
        expected.append("tzdata")
    expected.extend(["runtime-lib>=1", "test-lib>=2"])
    assert packages == expected
    assert requirement_files == ["requirements-test.txt"]


def test_dependency_specs_expand_test_dependencies_in_nested_dev_group(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[dependency-groups]\ntests = [\"simplejson\"]\n"
        "dev = [{ include-group = \"tests\" }, \"ruff\"]\n",
        encoding="utf-8",
    )

    packages, _ = BenchmarkWorkspace._dependency_specs(tmp_path)

    assert "simplejson" in packages


def test_target_file_is_inferred_from_search_evidence() -> None:
    evidence = "src/marshmallow/utils.py:40: def from_timestamp(value):\n"

    assert RepositoryAgentGraph._target_from_search(evidence) == "src/marshmallow/utils.py"


def test_target_file_prefers_function_definition_over_call_site() -> None:
    evidence = (
        "src/marshmallow/fields.py:1340: \"timestamp\": utils.from_timestamp,\n"
        "src/marshmallow/utils.py:40: def from_timestamp(value: typing.Any) -> dt.datetime:\n"
    )

    assert (
        RepositoryAgentGraph._target_from_search(evidence, "from_timestamp")
        == "src/marshmallow/utils.py"
    )


def test_search_repository_prioritizes_definition_before_test_references(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "marshmallow").mkdir(parents=True)
    (workspace / "tests" / "test_utils.py").write_text(
        "\n".join(f"assert utils.from_timestamp({index})" for index in range(50)),
        encoding="utf-8",
    )
    (workspace / "src" / "marshmallow" / "fields.py").write_text(
        'TIMESTAMP = utils.from_timestamp\n', encoding="utf-8"
    )
    (workspace / "src" / "marshmallow" / "utils.py").write_text(
        "def from_timestamp(value):\n    return value\n", encoding="utf-8"
    )

    evidence = build_repository_tools(workspace)["search_repository"].invoke(
        {"term": "from_timestamp"}
    )

    assert evidence.splitlines()[0].startswith(
        "src/marshmallow/utils.py:1: def from_timestamp"
    )


def test_retrieve_requires_the_selected_definition_file(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src" / "marshmallow").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "src" / "marshmallow" / "fields.py").write_text(
        'TIMESTAMP = utils.from_timestamp\n', encoding="utf-8"
    )
    (workspace / "src" / "marshmallow" / "utils.py").write_text(
        "def from_timestamp(value):\n    if value < 0:\n        raise ValueError()\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_utils.py").write_text("assert True\n", encoding="utf-8")
    graph = RepositoryAgentGraph(QLearningPolicy(), BenchmarkWorkspace())

    retrieved = graph.retrieve(
        {
            "task_id": "custom-target-selection",
            "task_kind": "custom",
            "task_title": "Allow negative timestamps",
            "objective": "Allow negative values in from_timestamp.",
            "search_term": "from_timestamp",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "test_target": "tests/test_utils.py",
            "workspace": str(workspace),
        }
    )

    assert retrieved["target_file"] == "src/marshmallow/utils.py"
    assert retrieved["source_targets"] == ["src/marshmallow/utils.py"]


def test_legacy_patch_parser_allows_an_empty_replacement_for_a_deletion() -> None:
    proposal = TaskPlanner._parse_json_proposal(
        '{"plan":"delete guard","target_file":"src/utils.py",'
        '"old_text":"if value < 0: raise ValueError()\\n","new_text":"",'
        '"test_target":"tests/test_utils.py"}'
    )

    assert proposal.patches[0].new_text == ""


def test_baseline_comparison_allows_only_resolved_failures() -> None:
    baseline = (
        "FAILED tests/test_utils.py::test_from_timestamp_with_overflow_value - AssertionError\n"
        "1 failed, 127 passed\n"
    )
    current = (
        "FAILED tests/test_utils.py::test_from_timestamp_with_overflow_value - AssertionError\n"
        "1 failed, 128 passed\n"
    )

    assert RepositoryAgentGraph._only_preexisting_failures(baseline, current)
    assert not RepositoryAgentGraph._only_preexisting_failures(current, baseline)


def test_baseline_comparison_accepts_equal_pass_count_when_failure_set_is_unchanged() -> None:
    output = "FAILED tests/test_utils.py::test_overflow - AssertionError\n1 failed, 127 passed\n"

    assert RepositoryAgentGraph._only_preexisting_failures(output, output)


def test_review_plan_explains_delegated_pre_epoch_timestamp_support() -> None:
    task = TaskSpec(
        task_id="negative-timestamp",
        title="Allow dates before 1970",
        objective="Allow negative timestamps and timestamp_ms values before 1970.",
        target_file="src/marshmallow/utils.py",
        search_term="from_timestamp",
        old_text="if value < 0:\n",
        new_text="",
        test_target="tests/test_utils.py tests/test_fields.py",
        acceptance_note="Pre-1970 dates round-trip through both timestamp formats.",
    )
    proposal = PatchProposal(
        plan="Remove the negative-value rejection and add regression coverage.",
        patches=[
            FilePatch(
                target_file="src/marshmallow/utils.py",
                old_text="if value < 0:\n",
                new_text="",
            ),
            FilePatch(
                target_file="tests/test_utils.py",
                old_text="assert True\n",
                new_text="assert True\n# regression\n",
            ),
        ],
    )
    source = (
        "def from_timestamp(value):\n    return value\n\n"
        "def from_timestamp_ms(value):\n    return from_timestamp(value / 1000)\n"
    )

    plan = RepositoryAgentGraph._review_plan(
        proposal.plan, task, proposal, task.test_target, source
    )

    assert "Implementation and verification scope (derived from the approved patch)" in plan
    assert "Production files changed: src/marshmallow/utils.py" in plan
    assert "from_timestamp_ms" in plan
    assert "Regression tests changed: tests/test_utils.py" in plan
    assert "pytest -q tests/test_utils.py tests/test_fields.py" in plan


def test_review_plan_has_contiguous_steps_for_cross_platform_assertion_tasks() -> None:
    task = TaskSpec(
        task_id="cross-platform-message",
        title="Portable assertion",
        objective="Make the overflow timestamp error-message assertion cross-platform.",
        target_file="tests/test_utils.py",
        search_term="from_timestamp",
        old_text="with pytest.raises(ValueError, match=r'platform detail'):\n",
        new_text="with pytest.raises(ValueError):\n",
        test_target="tests/test_utils.py::test_from_timestamp_with_overflow_value",
        acceptance_note="The assertion is portable.",
    )
    proposal = PatchProposal(
        plan="Identify the selected assertion.",
        patches=[
            FilePatch(
                target_file="tests/test_utils.py",
                old_text="with pytest.raises(ValueError, match=r'platform detail'):\n",
                new_text="with pytest.raises(ValueError):\n",
            )
        ],
    )

    plan = RepositoryAgentGraph._review_plan(
        proposal.plan, task, proposal, task.test_target, ""
    )

    assert "1. Production files changed: none." in plan
    assert "2. Assertion portability: remove only the `match=` constraint" in plan
    assert "3. Regression tests changed: tests/test_utils.py." in plan
    assert "4. Verification command: pytest -q " in plan


def test_review_plan_does_not_invent_delegated_behavior_for_unrelated_tasks() -> None:
    task = TaskSpec(
        task_id="simple-value",
        title="Replace a value",
        objective="Replace the incorrect value.",
        target_file="module.py",
        search_term="value",
        old_text="value = 1\n",
        new_text="value = 2\n",
        test_target="tests/test_module.py",
        acceptance_note="The focused test passes.",
    )
    proposal = PatchProposal(
        plan="Replace the incorrect value and test it.",
        patches=[
            FilePatch(
                target_file="module.py",
                old_text="value = 1\n",
                new_text="value = 2\n",
            )
        ],
    )

    plan = RepositoryAgentGraph._review_plan(
        proposal.plan, task, proposal, task.test_target, "def value():\n    return 1\n"
    )

    assert "from_timestamp_ms" not in plan
    assert "Verification command: pytest -q tests/test_module.py" in plan


def test_full_graph_completes_when_the_only_remaining_failure_matches_baseline(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "marshmallow"
    tests = workspace / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "utils.py").write_text(
        "def from_timestamp(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('invalid timestamp')\n"
        "    return value\n",
        encoding="utf-8",
    )
    (tests / "test_utils.py").write_text("assert True\n", encoding="utf-8")

    class TimestampPlanner:
        def invoke(self, _payload):
            return PatchProposal(
                plan="Remove the negative timestamp guard and add coverage.",
                patches=[
                    FilePatch(
                        target_file="src/marshmallow/utils.py",
                        old_text="    if value < 0:\n        raise ValueError('invalid timestamp')\n",
                        new_text="",
                    ),
                    FilePatch(
                        target_file="tests/test_utils.py",
                        old_text="assert True\n",
                        new_text="assert True\n# pre-1970 regression\n",
                    ),
                ],
                # The agent must ignore this narrowed selector and retain the
                # task's two-file baseline selector below.
                test_target="tests/test_utils.py",
            )

    class BaselineFailureWorkspace(BenchmarkWorkspace):
        def __init__(self):
            self.test_calls = 0
            self.targets: list[str] = []

        def run_targeted_test(self, root, task, *, fast=False):
            self.test_calls += 1
            self.targets.append(task.test_target)
            passed = 127 if self.test_calls == 1 else 129
            return WorkspaceTestResult(
                False,
                "FAILED tests/test_utils.py::test_from_timestamp_with_overflow_value "
                f"- AssertionError\n1 failed, {passed} passed",
            )

    manager = BaselineFailureWorkspace()
    graph = RepositoryAgentGraph(QLearningPolicy(), manager, planner=TimestampPlanner())
    result = graph.graph.invoke(
        {
            "run_id": "baseline-only-failure",
            "task_id": "custom-baseline-only-failure",
            "task_kind": "custom",
            "task_title": "Allow timestamps before 1970",
            "objective": "Timestamp deserialization rejects negative values before 1970.",
            "search_term": "from_timestamp",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": "tests/test_utils.py tests/test_fields.py",
            "workspace": str(workspace),
            "baseline_test": True,
            "auto_approve": True,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": "baseline-only-failure"}},
    )

    assert manager.test_calls == 2
    assert manager.targets == [
        "tests/test_utils.py tests/test_fields.py",
        "tests/test_utils.py tests/test_fields.py",
    ]
    assert result["status"] == "completed"
    assert result["test_passed"] is True
    assert result["baseline_failures_remain"] is True
    assert "Patch verification passed relative to the pre-patch baseline" in result["events"][-1]["detail"]
    assert [event["action"] for event in result["events"]] == [
        "retrieve",
        "plan",
        "request_approval",
        "apply_patch",
        "run_test",
    ]


def test_plan_rejects_overflow_only_patch_when_issue_requires_negative_timestamp_guard(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "marshmallow"
    tests = workspace / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "utils.py").write_text(
        "def from_timestamp(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('timestamp must be positive')\n"
        "    try:\n"
        "        return value\n"
        "    except OSError:\n"
        "        raise ValueError('invalid timestamp')\n",
        encoding="utf-8",
    )
    (tests / "test_utils.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")

    class OverflowOnlyPlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, _payload):
            self.calls += 1
            return PatchProposal(
                plan="Change the overflow error message.",
                patches=[
                    FilePatch(
                        target_file="src/marshmallow/utils.py",
                        old_text="    except OSError:\n        raise ValueError('invalid timestamp')\n",
                        new_text="    except OSError:\n        raise ValueError('timestamp out of range')\n",
                    )
                ],
                test_target="tests/test_utils.py",
            )

    planner = OverflowOnlyPlanner()
    graph = RepositoryAgentGraph(QLearningPolicy(), BenchmarkWorkspace(), planner=planner)
    result = graph.graph.invoke(
        {
            "run_id": "negative-timestamp-alignment",
            "task_id": "custom-negative-timestamp-alignment",
            "task_kind": "custom",
            "task_title": "Allow timestamps before 1970",
            "objective": "Timestamp deserialization rejects negative values before 1970.",
            "search_term": "from_timestamp",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": "tests/test_utils.py",
            "workspace": str(workspace),
            "auto_approve": False,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": "negative-timestamp-alignment"}},
    )

    assert planner.calls == 3
    assert result["status"] == "failed"
    assert "does not address the negative-value guard" in result["error"]
    assert "__interrupt__" not in result


def test_retrieve_preserves_the_original_baseline_on_a_revision(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 1\n", encoding="utf-8")

    class BaselineWorkspace(BenchmarkWorkspace):
        def __init__(self):
            self.calls = 0

        def run_targeted_test(self, root, task, *, fast=False):
            self.calls += 1
            return WorkspaceTestResult(False, "1 failed, 1 passed")

    manager = BaselineWorkspace()
    graph = RepositoryAgentGraph(QLearningPolicy(), manager)
    existing_baseline = "FAILED tests/test_utils.py::test_existing - AssertionError\n1 failed, 4 passed"
    retrieved = graph.retrieve(
        {
            "task_id": "custom-preserve-baseline",
            "task_kind": "custom",
            "task_title": "Keep the original baseline",
            "objective": "Replace the value.",
            "search_term": "value",
            "target_file": "module.py",
            "patch_old": "",
            "patch_new": "",
            "test_target": "tests",
            "workspace": str(workspace),
            "baseline_test": True,
            "baseline_test_output": existing_baseline,
        }
    )

    assert manager.calls == 0
    assert retrieved["baseline_test_output"] == existing_baseline


def test_revision_feedback_does_not_target_preexisting_failures() -> None:
    graph = RepositoryAgentGraph(QLearningPolicy(), BenchmarkWorkspace())
    baseline = "FAILED tests/test_utils.py::test_overflow - AssertionError\n1 failed, 127 passed"
    revised = graph.revise(
        {
            "last_action": "revise",
            "task_id": "custom-baseline-feedback",
            "task_kind": "custom",
            "objective": "Fix the timestamp deserialization guard.",
            "failures": 1,
            "revisions": 0,
            "baseline_test_output": baseline,
            "last_test_output": "FAILED tests/test_utils.py::test_overflow - AssertionError\n1 failed, 127 passed",
        }
    )

    assert "already present in the unmodified baseline" in revised["planner_feedback"]
    assert "Do not change production exception messages" in revised["planner_feedback"]


def test_invalid_model_patch_stops_before_human_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 1\n", encoding="utf-8")

    class InvalidPlanner:
        def invoke(self, _payload):
            return PatchProposal(
                plan="Replace the value.",
                target_file="module.py",
                old_text="value = 2\n",
                new_text="value = 3\n",
                test_target="tests",
            )

    graph = RepositoryAgentGraph(QLearningPolicy(), BenchmarkWorkspace(), planner=InvalidPlanner())
    result = graph.graph.invoke(
        {
            "run_id": "invalid-model-patch",
            "task_id": "custom-invalid-model-patch",
            "task_kind": "custom",
            "task_title": "Replace a value",
            "objective": "Replace the value.",
            "search_term": "value",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "test_target": "tests",
            "workspace": str(workspace),
            "auto_approve": False,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": "invalid-model-patch"}},
    )

    assert result["status"] == "failed"
    assert "Generated patch rejected safely" in result["error"]
    assert "__interrupt__" not in result
    assert [event["action"] for event in result["events"]] == [
        "retrieve",
        "plan",
        "plan",
        "plan",
    ]
    assert result["plan_attempts"] == 3


def test_invalid_model_patch_is_repaired_before_human_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 1\n", encoding="utf-8")

    class RepairingPlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, _payload):
            self.calls += 1
            if self.calls == 1:
                return PatchProposal(
                    plan="Replace the value.",
                    patches=[
                        FilePatch(
                            target_file="module.py",
                            old_text="value = 2\n",
                            new_text="value = 3\n",
                        )
                    ],
                    test_target="tests",
                )
            return PatchProposal(
                plan="Replace the value.",
                patches=[
                    FilePatch(
                        target_file="module.py",
                        old_text="value = 1\n",
                        new_text="value = 2\n",
                    )
                ],
                test_target="tests",
            )

    planner = RepairingPlanner()
    graph = RepositoryAgentGraph(QLearningPolicy(), BenchmarkWorkspace(), planner=planner)
    result = graph.graph.invoke(
        {
            "run_id": "repairing-model-patch",
            "task_id": "custom-repairing-model-patch",
            "task_kind": "custom",
            "task_title": "Replace a value",
            "objective": "Replace the value.",
            "search_term": "value",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": "tests",
            "workspace": str(workspace),
            "auto_approve": True,
            "fast_verification": True,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": "repairing-model-patch"}},
    )

    assert planner.calls == 2
    assert result["status"] == "completed"
    assert result["test_passed"] is True


def test_test_only_model_patch_stops_before_human_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "src" / "logic.py").write_text("return False\n", encoding="utf-8")
    (workspace / "tests" / "test_logic.py").write_text("assert False\n", encoding="utf-8")

    class TestOnlyPlanner:
        def invoke(self, _payload):
            return PatchProposal(
                plan="Update the test.",
                patches=[
                    FilePatch(
                        target_file="tests/test_logic.py",
                        old_text="assert False\n",
                        new_text="assert True\n",
                    )
                ],
                test_target="tests/test_logic.py",
            )

    graph = RepositoryAgentGraph(QLearningPolicy(), BenchmarkWorkspace(), planner=TestOnlyPlanner())
    result = graph.graph.invoke(
        {
            "run_id": "tests-only-model-patch",
            "task_id": "custom-tests-only-model-patch",
            "task_kind": "custom",
            "task_title": "Fix the logic",
            "objective": "Change return False to return True.",
            "search_term": "return False",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": "tests/test_logic.py",
            "workspace": str(workspace),
            "auto_approve": False,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": "tests-only-model-patch"}},
    )

    assert result["status"] == "failed"
    assert "must include the Issue's source file" in result["error"]
    assert "__interrupt__" not in result


@pytest.mark.parametrize(
    "objective",
    [
        "Make the overflow timestamp error-message assertion cross-platform.",
        "修 overflow timestamp 的跨平台错误消息断言。",
    ],
)
def test_cross_platform_error_message_assertion_issue_allows_a_test_only_patch(
    tmp_path, objective, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "marshmallow"
    tests = workspace / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "utils.py").write_text(
        "def from_timestamp(value):\n"
        "    raise ValueError('platform-specific timestamp error')\n",
        encoding="utf-8",
    )
    test_file = tests / "test_utils.py"
    test_file.write_text(
        "import pytest\n\n"
        "def test_from_timestamp_with_overflow_value():\n"
        "    with pytest.raises(ValueError, match=r'out of range|year must be in 1\\.\\.9999'):\n"
        "        raise ValueError('platform-specific timestamp error')\n",
        encoding="utf-8",
    )

    class PortabilityPlanner:
        def invoke(self, _payload):
            return PatchProposal(
                plan="Remove the platform-specific error-message match.",
                patches=[
                    FilePatch(
                        target_file="tests/test_utils.py",
                        old_text=(
                            "    with pytest.raises(ValueError, "
                            "match=r'out of range|year must be in 1\\.\\.9999'):\n"
                        ),
                        new_text="    with pytest.raises(ValueError):\n",
                    )
                ],
                test_target="tests/test_utils.py",
            )

    workspace_manager = BenchmarkWorkspace()
    monkeypatch.setattr(
        workspace_manager,
        "prepare_test_environment",
        lambda _root: (Path(sys.executable), "Prepared test environment."),
    )
    graph = RepositoryAgentGraph(QLearningPolicy(), workspace_manager, planner=PortabilityPlanner())
    result = graph.graph.invoke(
        {
            "run_id": "portable-timestamp-assertion",
            "task_id": "custom-portable-timestamp-assertion",
            "task_kind": "custom",
            "task_title": "Fix the overflow timestamp assertion",
            "objective": objective,
            "search_term": "from_timestamp",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": "tests/test_utils.py",
            "workspace": str(workspace),
            "auto_approve": True,
            "fast_verification": False,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": f"portable-timestamp-{tmp_path.name}"}},
    )

    assert result["status"] == "completed"
    assert result["test_passed"] is True
    assert "1 passed" in result["events"][-1]["detail"]
    assert "Fast verifier" not in result["events"][-1]["detail"]
    assert "match=" not in test_file.read_text(encoding="utf-8")
    assert "platform-specific timestamp error" in (source / "utils.py").read_text(
        encoding="utf-8"
    )


def test_cross_platform_assertion_fallback_recovers_after_invalid_model_proposals(
    tmp_path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "marshmallow"
    tests = workspace / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    source_file = source / "utils.py"
    source_file.write_text(
        "def from_timestamp(value):\n"
        "    raise ValueError('Error converting value to datetime')\n",
        encoding="utf-8",
    )
    test_file = tests / "test_utils.py"
    test_file.write_text(
        "import pytest\n\n"
        "def test_from_timestamp_with_overflow_value():\n"
        "    with pytest.raises(ValueError, match=r'out of range|year must be in 1\\.\\.9999'):\n"
        "        raise ValueError('Error converting value to datetime')\n\n"
        "def test_unrelated_error_message_contract():\n"
        "    with pytest.raises(ValueError, match=r'unrelated detail'):\n"
        "        raise ValueError('unrelated detail')\n",
        encoding="utf-8",
    )

    class UnsafePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, _payload):
            self.calls += 1
            return PatchProposal(
                plan="Replace the assertion with a trivial pass.",
                patches=[
                    FilePatch(
                        target_file="tests/test_utils.py",
                        old_text=(
                            "    with pytest.raises(ValueError, "
                            "match=r'out of range|year must be in 1\\.\\.9999'):\n"
                        ),
                        new_text="    assert True\n",
                    )
                ],
                test_target="tests/test_utils.py",
            )

    workspace_manager = BenchmarkWorkspace()
    monkeypatch.setattr(
        workspace_manager,
        "prepare_test_environment",
        lambda _root: (Path(sys.executable), "Prepared test environment."),
    )
    planner = UnsafePlanner()
    graph = RepositoryAgentGraph(QLearningPolicy(), workspace_manager, planner=planner)
    test_target = "tests/test_utils.py::test_from_timestamp_with_overflow_value"
    result = graph.graph.invoke(
        {
            "run_id": "portable-assertion-fallback",
            "task_id": "custom-portable-assertion-fallback",
            "task_kind": "custom",
            "task_title": "Fix the overflow timestamp assertion",
            "objective": (
                "Fix the overflow timestamp cross-platform error-message assertion in "
                "tests/test_utils.py::test_from_timestamp_with_overflow_value. Keep "
                "production code unchanged. Assert ValueError without requiring a "
                "platform-specific exception message."
            ),
            "search_term": "from_timestamp",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": test_target,
            "workspace": str(workspace),
            "auto_approve": True,
            "fast_verification": False,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": f"portable-fallback-{tmp_path.name}"}},
    )

    assert planner.calls == 3
    assert result["status"] == "completed"
    assert result["test_passed"] is True
    assert any(
        "constrained test-only fallback plan" in event["detail"]
        for event in result["events"]
    )
    assert (
        "    with pytest.raises(ValueError):\n"
        "        raise ValueError('Error converting value to datetime')"
    ) in test_file.read_text(encoding="utf-8")
    assert "match=r'unrelated detail'" in test_file.read_text(encoding="utf-8")
    assert source_file.read_text(encoding="utf-8") == (
        "def from_timestamp(value):\n"
        "    raise ValueError('Error converting value to datetime')\n"
    )


def test_cross_platform_error_message_assertion_issue_rejects_a_source_patch() -> None:
    task = TaskSpec(
        task_id="cross-platform-message",
        title="Portable assertion",
        objective="Make the overflow timestamp error-message assertion cross-platform.",
        target_file="src/marshmallow/utils.py",
        search_term="from_timestamp",
        old_text="raise ValueError('old')\n",
        new_text="raise ValueError('new')\n",
        test_target="tests/test_utils.py",
        acceptance_note="The assertion is portable.",
    )

    with pytest.raises(ValueError, match="must modify only test files"):
        RepositoryAgentGraph._required_source_targets(
            task, ["src/marshmallow/utils.py"]
        )


@pytest.mark.parametrize(
    ("old_text", "new_text"),
    [
        (
            "    with pytest.raises(ValueError, match=r'platform detail'):\n",
            "    assert True\n",
        ),
        (
            "def test_from_timestamp_with_overflow_value():\n"
            "    with pytest.raises(ValueError, match=r'platform detail'):\n"
            "        raise ValueError('platform detail')\n",
            "",
        ),
    ],
)
def test_cross_platform_error_message_assertion_issue_rejects_unrelated_test_edits(
    old_text, new_text
) -> None:
    task = TaskSpec(
        task_id="cross-platform-message",
        title="Portable assertion",
        objective="Make the overflow timestamp error-message assertion cross-platform.",
        target_file="tests/test_utils.py",
        search_term="from_timestamp",
        old_text=old_text,
        new_text=new_text,
        test_target="tests/test_utils.py",
        acceptance_note="The assertion is portable.",
    )

    with pytest.raises(ValueError, match="must only remove a match"):
        RepositoryAgentGraph._required_source_targets(task, ["src/marshmallow/utils.py"])


def test_apply_patch_failure_stops_after_one_attempt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 1\n", encoding="utf-8")

    class ValidPlanner:
        def invoke(self, _payload):
            return PatchProposal(
                plan="Replace the value.",
                target_file="module.py",
                old_text="value = 1\n",
                new_text="value = 2\n",
                test_target="tests",
            )

    class FailingWorkspace(BenchmarkWorkspace):
        def apply_patch(self, root, task, **_kwargs):
            raise ValueError("simulated write failure")

    graph = RepositoryAgentGraph(QLearningPolicy(), FailingWorkspace(), planner=ValidPlanner())
    config = {"configurable": {"thread_id": "write-failure"}}
    paused = graph.graph.invoke(
        {
            "run_id": "write-failure",
            "task_id": "custom-write-failure",
            "task_kind": "custom",
            "task_title": "Replace a value",
            "objective": "Replace the value.",
            "search_term": "value",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "test_target": "tests",
            "workspace": str(workspace),
            "auto_approve": False,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        config,
    )
    assert "__interrupt__" in paused

    result = graph.graph.invoke(Command(resume={"approved": True}), config)

    assert result["status"] == "failed"
    assert result["error"] == "Patch failed safely: simulated write failure"
    assert [event["action"] for event in result["events"]].count("apply_patch") == 1


def test_failed_test_replans_from_current_workspace_before_a_new_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 1\n", encoding="utf-8")

    class RevisingPlanner:
        def __init__(self):
            self.calls = 0
            self.feedback: list[str] = []

        def invoke(self, payload):
            self.calls += 1
            self.feedback.append(payload["feedback"])
            old, new = (
                ("value = 1\n", "value = 2\n")
                if self.calls == 1
                else ("value = 2\n", "value = 3\n")
            )
            return PatchProposal(
                plan=f"Set {new.strip()}.",
                patches=[FilePatch(target_file="module.py", old_text=old, new_text=new)],
                test_target="tests",
            )

    class FailingThenPassingWorkspace(BenchmarkWorkspace):
        def __init__(self):
            self.test_calls = 0

        def run_targeted_test(self, root, task, *, fast=False):
            self.test_calls += 1
            if self.test_calls == 1:
                return WorkspaceTestResult(False, "AssertionError: expected value == 3")
            return WorkspaceTestResult(True, "1 passed")

    planner = RevisingPlanner()
    graph = RepositoryAgentGraph(
        QLearningPolicy(), FailingThenPassingWorkspace(), planner=planner
    )
    result = graph.graph.invoke(
        {
            "run_id": "replan-after-test-failure",
            "task_id": "custom-replan-after-test-failure",
            "task_kind": "custom",
            "task_title": "Replace a value",
            "objective": "Change the value to 3.",
            "search_term": "value",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": "tests",
            "workspace": str(workspace),
            "auto_approve": True,
            "epsilon": 0.0,
            "max_steps": 14,
        },
        {"configurable": {"thread_id": "replan-after-test-failure"}},
    )

    assert result["status"] == "completed"
    assert result["test_passed"] is True
    assert result["revisions"] == 1
    assert planner.calls == 2
    assert "AssertionError" in planner.feedback[1]
    assert [event["action"] for event in result["events"]].count("request_approval") == 2
    assert (workspace / "module.py").read_text(encoding="utf-8") == "value = 3\n"


def test_test_environment_failure_stops_without_proposing_another_patch(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 1\n", encoding="utf-8")

    class ValidPlanner:
        def invoke(self, _payload):
            return PatchProposal(
                plan="Replace the value.",
                patches=[
                    FilePatch(
                        target_file="module.py",
                        old_text="value = 1\n",
                        new_text="value = 2\n",
                    )
                ],
                test_target="tests",
            )

    class EnvironmentFailureWorkspace(BenchmarkWorkspace):
        def run_targeted_test(self, root, task, *, fast=False):
            return WorkspaceTestResult(
                False, "ImportError while loading conftest: No module named 'simplejson'"
            )

    planner = ValidPlanner()
    graph = RepositoryAgentGraph(
        QLearningPolicy(), EnvironmentFailureWorkspace(), planner=planner
    )
    result = graph.graph.invoke(
        {
            "run_id": "environment-failure",
            "task_id": "custom-environment-failure",
            "task_kind": "custom",
            "task_title": "Replace a value",
            "objective": "Replace the value.",
            "search_term": "value",
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": "tests",
            "workspace": str(workspace),
            "auto_approve": True,
            "epsilon": 0.0,
            "max_steps": 8,
        },
        {"configurable": {"thread_id": "environment-failure"}},
    )

    assert result["status"] == "failed"
    assert result["test_environment_failed"] is True
    assert "test environment" in result["error"]
    assert [event["action"] for event in result["events"]].count("plan") == 1
