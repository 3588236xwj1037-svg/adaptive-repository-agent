from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

ActionName = Literal[
    "retrieve",
    "plan",
    "request_approval",
    "apply_patch",
    "run_test",
    "revise",
]


@dataclass(frozen=True)
class PatchOperation:
    """One exact, repository-relative replacement; ``new_text`` may be empty."""

    target_file: str
    old_text: str
    new_text: str


@dataclass(frozen=True)
class TaskSpec:
    """A benchmark task with one known-good patch and focused test."""

    task_id: str
    title: str
    objective: str
    target_file: str
    search_term: str
    old_text: str
    new_text: str
    test_target: str
    acceptance_note: str
    patches: tuple[PatchOperation, ...] = ()

    def patch_operations(self) -> tuple[PatchOperation, ...]:
        """Return the normalized patch set, including legacy benchmark fields."""

        if self.patches:
            return self.patches
        return (
            PatchOperation(
                target_file=self.target_file,
                old_text=self.old_text,
                new_text=self.new_text,
            ),
        )


@dataclass
class RunSummary:
    task_id: str
    status: str
    total_reward: float
    steps: int
    test_passed: bool
    events: list[dict[str, object]] = field(default_factory=list)
    workspace: Path | None = None


class AgentState(TypedDict, total=False):
    run_id: str
    task_id: str
    task_kind: str
    task_title: str
    objective: str
    source_repository: str
    search_term: str
    target_file: str
    patch_old: str
    patch_new: str
    patches: list[dict[str, str]]
    source_targets: list[str]
    test_target: str
    workspace: str
    status: str
    auto_approve: bool
    fast_verification: bool
    epsilon: float
    max_steps: int
    steps: int
    context: str
    target_source: str
    issue_anchor_addressed: bool
    context_ready: bool
    plan_attempts: int
    planner_feedback: str
    baseline_test: bool
    baseline_test_output: str
    baseline_failures_remain: bool
    last_test_output: str
    test_environment_failed: bool
    plan: str
    plan_ready: bool
    approved: bool
    patch_ready: bool
    patch_applied: bool
    test_passed: bool
    failures: int
    revisions: int
    last_action: ActionName
    last_observation: str
    last_reward: float
    total_reward: float
    events: list[dict[str, object]]
    error: str
