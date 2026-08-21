from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from adaptive_agent.fixtures import get_task
from adaptive_agent.models import AgentState, PatchOperation, TaskSpec
from adaptive_agent.planner import FilePatch, PatchProposal, TaskPlanner
from adaptive_agent.rl import QLearningPolicy
from adaptive_agent.workspace import BenchmarkWorkspace


def _append_event(
    state: AgentState, name: str, detail: str, reward: float
) -> list[dict[str, object]]:
    events = list(state.get("events", []))
    events.append(
        {
            "step": state.get("steps", 0),
            "action": name,
            "detail": detail,
            "reward": round(reward, 3),
        }
    )
    return events


class RepositoryAgentGraph:
    MAX_PLAN_ATTEMPTS = 3
    MAX_REVISIONS = 2

    def __init__(
        self,
        policy: QLearningPolicy,
        workspace: BenchmarkWorkspace,
        planner: TaskPlanner | None = None,
    ) -> None:
        self.policy = policy
        self.workspace = workspace
        self.planner = planner or TaskPlanner()
        self.checkpointer = MemorySaver()
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("bootstrap", self.bootstrap)
        builder.add_node("decide", self.decide)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("plan", self.plan)
        builder.add_node("request_approval", self.request_approval)
        builder.add_node("apply_patch", self.apply_patch)
        builder.add_node("run_test", self.run_test)
        builder.add_node("revise", self.revise)
        builder.add_node("learn", self.learn)
        builder.add_edge(START, "bootstrap")
        builder.add_edge("bootstrap", "decide")
        builder.add_conditional_edges("decide", self.route_decision)
        for action in ("retrieve", "plan", "request_approval", "apply_patch", "run_test", "revise"):
            builder.add_edge(action, "learn")
        builder.add_conditional_edges("learn", self.route_after_learning)
        return builder.compile(checkpointer=self.checkpointer)

    def bootstrap(self, state: AgentState) -> dict[str, object]:
        return {
            "status": "running",
            "steps": 0,
            "failures": 0,
            "revisions": 0,
            "total_reward": 0.0,
            "events": [],
            "context_ready": False,
            "issue_anchor_addressed": False,
            "plan_ready": False,
            "approved": False,
            "patch_applied": False,
            "test_passed": False,
            "baseline_test_output": "",
            "baseline_failures_remain": False,
            "last_test_output": "",
            "test_environment_failed": False,
            "planner_feedback": "",
            "error": "",
        }

    def decide(self, state: AgentState) -> dict[str, object]:
        if state.get("steps", 0) >= state.get("max_steps", 12):
            return {"status": "failed", "error": "Maximum workflow steps exceeded"}
        observation = self.policy.observation(state)
        action = self.policy.choose(state, float(state.get("epsilon", 0.0)))
        return {
            "last_observation": observation,
            "last_action": action,
            "steps": state.get("steps", 0) + 1,
        }

    def route_decision(self, state: AgentState) -> str:
        if state.get("status") in {"completed", "failed"}:
            return END
        return str(state["last_action"])

    def retrieve(self, state: AgentState) -> dict[str, object]:
        if state.get("context_ready"):
            reward = -0.18
            return self._result(
                state, reward, "Context was already available; redundant retrieval."
            )
        task = self._task(state)
        root = self._root(state)
        tools = self.workspace.tools(root)
        files = tools["inspect_repository"].invoke({})
        evidence = (
            tools["search_repository"].invoke({"term": task.search_term})
            if task.search_term
            else "No search term supplied; the planner must infer relevant files from the listing."
        )
        target_file = task.target_file or self._target_from_search(
            evidence, task.search_term
        )
        source = (
            tools["read_repository_file"].invoke({"relative_path": target_file})
            if target_file
            else "No target file was identified from the search results."
        )
        test_context = self._test_context(root, task.test_target)
        context = (
            f"Target file: {target_file or 'not identified'}\n\n"
            f"Target source:\n{source[:12000]}\n\n"
            f"Focused test source:\n{test_context}\n\n"
            f"Search evidence:\n{evidence}\n\n"
            f"Repository files:\n{files[:6000]}"
        )
        # The baseline represents the unmodified workspace. On a revision the
        # workspace already contains the approved patch, so never overwrite it.
        baseline_output = state.get("baseline_test_output", "")
        detail = "Repository inspected and target evidence collected."
        if state.get("baseline_test") and not baseline_output:
            baseline = self.workspace.run_targeted_test(root, task)
            baseline_output = baseline.output
            if self._is_test_environment_failure(baseline.output):
                message = (
                    "Baseline test could not start because the disposable test environment "
                    "is not ready. No patch will be proposed."
                )
                return self._result(
                    state,
                    -0.42,
                    f"{message}\n{baseline.output}",
                    status="failed",
                    error=message,
                    last_test_output=baseline.output,
                    test_environment_failed=True,
                )
            detail += " Baseline targeted test was recorded before planning."

        reward = -0.02
        return self._result(
            state,
            reward,
            detail,
            context=context,
            # Keep the first inspection as immutable Issue evidence. A revision
            # runs against an already patched workspace, so current source must
            # not erase the original guard identified by the Issue.
            target_source=state.get("target_source") or source,
            context_ready=True,
            target_file=target_file,
            # The first matching file is often a call site. Require the patch to
            # touch the selected definition file, rather than any search result.
            source_targets=[target_file] if target_file else [],
            baseline_test_output=baseline_output,
        )

    def plan(self, state: AgentState) -> dict[str, object]:
        if not state.get("context_ready"):
            return self._result(
                state, -0.28, "Planning was attempted before repository inspection."
            )
        if state.get("plan_ready"):
            return self._result(state, -0.16, "Plan already exists; redundant planning.")
        task = self._task(state)
        evidence = state.get("context", "")[:800]
        attempt = state.get("plan_attempts", 0) + 1
        planner = (
            self.planner.invoke
            if state.get("task_kind") == "custom"
            else self.planner.deterministic.invoke
        )
        try:
            plan = planner(
                {
                    "objective": task.objective,
                    "evidence": evidence,
                    "context": state.get("context", ""),
                    "target_file": state.get("target_file", task.target_file),
                    "acceptance": task.acceptance_note,
                    "feedback": state.get("planner_feedback", ""),
                    "safety_instruction": self._cross_platform_planning_instruction(task),
                }
            )
        except (OSError, ValueError) as exc:
            return self._planning_retry(state, attempt, exc)
        if isinstance(plan, PatchProposal):
            try:
                return self._store_plan(state, task, plan, attempt)
            except (OSError, ValueError) as exc:
                return self._planning_retry(state, attempt, exc)
        if state.get("task_kind") == "custom":
            return self._result(
                state,
                -0.6,
                "A custom repository task needs OPENAI_API_KEY for structured patch generation.",
                status="failed",
                error="No structured patch model is configured",
            )
        # Benchmark tasks keep their known-good patch in TaskSpec. Convert that
        # patch into the same structured proposal used by model-backed tasks so
        # approval, application, and the dashboard all share one audit trail.
        benchmark_plan = PatchProposal(
            plan=str(plan),
            patches=[
                FilePatch(
                    target_file=operation.target_file,
                    old_text=operation.old_text,
                    new_text=operation.new_text,
                )
                for operation in task.patch_operations()
            ],
            test_target=task.test_target,
        )
        try:
            return self._store_plan(
                state,
                task,
                benchmark_plan,
                attempt,
                detail="A deterministic benchmark patch plan was generated.",
            )
        except (OSError, ValueError) as exc:
            return self._planning_retry(state, attempt, exc)

    def _store_plan(
        self,
        state: AgentState,
        task: TaskSpec,
        plan: PatchProposal,
        attempt: int,
        *,
        detail: str = "A structured patch plan was generated.",
    ) -> dict[str, object]:
        """Validate a proposal and retain it for human approval."""

        primary_patch = plan.patches[0]
        # The test selector comes from the user-facing task configuration.
        # A model may suggest a narrower test file, but allowing it to replace
        # the baseline selector makes pass-count comparisons invalid.
        verification_target = task.test_target
        proposed_task = TaskSpec(
            task_id=task.task_id,
            title=task.title,
            objective=task.objective,
            target_file=primary_patch.target_file,
            search_term=task.search_term,
            old_text=primary_patch.old_text,
            new_text=primary_patch.new_text,
            test_target=verification_target,
            acceptance_note=task.acceptance_note,
            patches=plan.patch_operations(),
        )
        self._validate_issue_alignment(task, plan, state)
        required_source_targets = self._required_source_targets(
            proposed_task, state.get("source_targets", [])
        )
        self.workspace.validate_patch(
            self._root(state),
            proposed_task,
            required_source_targets=required_source_targets,
        )
        return self._result(
            state,
            -0.02,
            detail,
            plan=self._review_plan(
                plan.plan,
                task,
                plan,
                verification_target,
                state.get("target_source", ""),
            ),
            plan_ready=True,
            target_file=primary_patch.target_file,
            patch_old=primary_patch.old_text,
            patch_new=primary_patch.new_text,
            patches=[operation.__dict__ for operation in plan.patch_operations()],
            test_target=verification_target,
            plan_attempts=attempt,
            planner_feedback="",
            issue_anchor_addressed=state.get("issue_anchor_addressed", False)
            or self._proposal_addresses_negative_guard(plan),
        )

    def _planning_retry(
        self, state: AgentState, attempt: int, error: Exception
    ) -> dict[str, object]:
        message = f"Generated patch rejected safely: {error}"
        if attempt >= self.MAX_PLAN_ATTEMPTS:
            task = self._task(state)
            fallback = self._cross_platform_assertion_fallback(state, task)
            if fallback is not None:
                try:
                    return self._store_plan(
                        state,
                        task,
                        fallback,
                        attempt,
                        detail=(
                            "Model proposals were invalid; a constrained test-only fallback "
                            "plan was created from the selected pytest assertion."
                        ),
                    )
                except (OSError, ValueError) as fallback_error:
                    message = f"{message} Safe fallback rejected: {fallback_error}"
            return self._result(
                state,
                -0.6,
                f"{message} No valid proposal after {self.MAX_PLAN_ATTEMPTS} attempts.",
                status="failed",
                error=f"{message} No valid proposal after {self.MAX_PLAN_ATTEMPTS} attempts.",
                plan_attempts=attempt,
            )
        task = self._task(state)
        feedback = f"Previous proposal failed validation: {error}. "
        if self._is_cross_platform_error_message_assertion_issue(task.objective):
            feedback += (
                "This Issue permits exactly one test-only change: remove only the `match=` "
                "keyword from `with pytest.raises(ValueError, match=...):`. Return one patch "
                "in the selected test file; do not change production code, a test function, "
                "or any other assertion."
            )
        else:
            feedback += (
                "Re-read the exact Target source and return a contiguous old_text substring "
                "with original whitespace; do not paraphrase or normalize it."
            )
        return self._result(
            state,
            -0.35,
            f"{message} Retrying proposal ({attempt + 1}/{self.MAX_PLAN_ATTEMPTS}).",
            plan_attempts=attempt,
            planner_feedback=feedback,
        )

    @classmethod
    def _cross_platform_planning_instruction(cls, task: TaskSpec) -> str:
        """Return the narrow prompt exception for portable pytest assertions."""

        if not cls._is_cross_platform_error_message_assertion_issue(task.objective):
            return ""
        return (
            "This is an explicit cross-platform error-message assertion task. Return exactly "
            "one test-only patch in the selected pytest file. Its old_text must be the exact "
            "single-line `with pytest.raises(ValueError, match=...)` header from that selected "
            "test, and its new_text must remove only the `match=` keyword and value. Do not "
            "modify production code, test bodies, test names, other assertions, or the pytest "
            "target."
        )

    def _cross_platform_assertion_fallback(
        self, state: AgentState, task: TaskSpec
    ) -> PatchProposal | None:
        """Build one auditable test patch when model output misses this narrow task.

        This fallback deliberately supports one Issue shape only. It derives the replacement
        from the exact function named by the user's pytest node selector, then the normal
        proposal validation, approval, patch, and pytest steps still run unchanged.
        """

        if not self._is_cross_platform_error_message_assertion_issue(task.objective):
            return None
        selector_parts = shlex.split(task.test_target, posix=False)
        if len(selector_parts) != 1:
            return None
        selector = selector_parts[0]
        selector_path, separator, function_name = selector.partition("::")
        if (
            not separator
            or "::" in function_name
            or not function_name.isidentifier()
            or not self._is_test_path(selector_path)
        ):
            return None

        root = self._root(state).resolve()
        test_path = (root / selector_path).resolve()
        if root not in test_path.parents or not test_path.is_file():
            return None
        try:
            source = test_path.read_text(encoding="utf-8")
            module = ast.parse(source, filename=selector_path)
        except (OSError, SyntaxError, UnicodeDecodeError):
            return None

        functions = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ]
        if len(functions) != 1:
            return None
        candidates = [
            node
            for node in ast.walk(functions[0])
            if isinstance(node, (ast.With, ast.AsyncWith))
            and self._is_eligible_pytest_error_message_with(node)
        ]
        if len(candidates) != 1:
            return None

        header = source.splitlines(keepends=True)[candidates[0].lineno - 1]
        old_text = header
        indentation = header[: len(header) - len(header.lstrip(" \t"))]
        newline = "\r\n" if header.endswith("\r\n") else "\n" if header.endswith("\n") else ""
        new_text = f"{indentation}with pytest.raises(ValueError):{newline}"
        patch = FilePatch(
            target_file=Path(selector_path).as_posix(),
            old_text=old_text,
            new_text=new_text,
        )
        operation = PatchOperation(**patch.model_dump())
        if not self._is_pytest_error_message_match_relaxation(operation):
            return None
        return PatchProposal(
            plan=(
                "1. Keep production code unchanged because the Issue is limited to a "
                "platform-dependent exception-message assertion.\n"
                "2. In the selected overflow timestamp test, remove only the `match=` "
                "constraint from `pytest.raises(ValueError)` so every valid ValueError "
                "message is accepted.\n"
                f"3. Run `pytest -q {task.test_target}` to verify the selected regression."
            ),
            patches=[patch],
            test_target=task.test_target,
        )

    @staticmethod
    def _is_eligible_pytest_error_message_with(
        statement: ast.With | ast.AsyncWith,
    ) -> bool:
        """Return whether a with statement is the one fallback may relax."""

        if len(statement.items) != 1:
            return False
        call = statement.items[0].context_expr
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pytest"
            and call.func.attr == "raises"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "ValueError"
        ):
            return False
        return len(call.keywords) == 1 and call.keywords[0].arg == "match"

    def request_approval(self, state: AgentState) -> dict[str, object]:
        if not state.get("plan_ready"):
            return self._result(state, -0.3, "Approval requested before a plan existed.")
        if state.get("approved"):
            return self._result(state, -0.14, "Approval was already granted.")
        if state.get("auto_approve"):
            decision: Any = {"approved": True, "reason": "automatic benchmark approval"}
        else:
            decision = interrupt(
                {
                    "kind": "patch_approval",
                    "title": self._task(state).title,
                    "plan": state.get("plan", ""),
                    "patches": state.get("patches", []),
                    "message": "Approve the minimal patch before it is written to the disposable workspace.",
                }
            )
        if not isinstance(decision, dict) or not decision.get("approved"):
            return self._result(
                state,
                -0.55,
                "Patch was rejected by the human reviewer.",
                status="failed",
                error="Patch approval rejected",
            )
        return self._result(state, -0.01, "Patch approval granted.", approved=True)

    def apply_patch(self, state: AgentState) -> dict[str, object]:
        if not state.get("approved") or not state.get("plan_ready"):
            return self._result(
                state, -0.35, "Patch attempt blocked because approval or planning is missing."
            )
        task = self._task(state)
        try:
            required_source_targets = self._required_source_targets(
                task, state.get("source_targets", [])
            )
            detail = self.workspace.apply_patch(
                self._root(state),
                task,
                required_source_targets=required_source_targets,
            )
        except (OSError, ValueError) as exc:
            message = f"Patch failed safely: {exc}"
            return self._result(
                state,
                -0.45,
                message,
                status="failed",
                error=message,
                failures=state.get("failures", 0) + 1,
            )
        return self._result(state, -0.04, detail, patch_applied=True)

    def run_test(self, state: AgentState) -> dict[str, object]:
        if not state.get("patch_applied"):
            return self._result(state, -0.32, "Test was attempted before a patch was applied.")
        result = self.workspace.run_targeted_test(
            self._root(state),
            self._task(state),
            fast=bool(state.get("fast_verification", False)),
        )
        if result.passed:
            return self._result(
                state,
                1.0,
                f"Targeted test passed.\n{result.output}",
                status="completed",
                test_passed=True,
                last_test_output=result.output,
            )
        if self._is_test_environment_failure(result.output):
            message = (
                "Targeted test could not start because the disposable test environment "
                "is not ready. No new patch will be proposed for an infrastructure error."
            )
            return self._result(
                state,
                -0.42,
                f"{message}\n{result.output}",
                status="failed",
                error=message,
                last_test_output=result.output,
                test_environment_failed=True,
            )
        if self._only_preexisting_failures(
            state.get("baseline_test_output", ""), result.output
        ):
            remaining = sorted(self._pytest_failure_ids(result.output))
            return self._result(
                state,
                0.82,
                "Patch verification passed relative to the pre-patch baseline: no new "
                "failing node IDs were introduced and the passing-test count did not decrease. "
                "Remaining baseline failures: "
                + ", ".join(remaining),
                status="completed",
                test_passed=True,
                last_test_output=result.output,
                baseline_failures_remain=True,
            )
        return self._result(
            state,
            -0.42,
            f"Targeted test failed.\n{result.output}",
            failures=state.get("failures", 0) + 1,
            last_test_output=result.output,
        )

    def revise(self, state: AgentState) -> dict[str, object]:
        if state.get("test_passed"):
            return self._result(state, -0.12, "Revision was unnecessary after a passing test.")
        if state.get("test_environment_failed"):
            return self._result(
                state,
                -0.2,
                "Revision skipped because the test environment failed before collection.",
                status="failed",
            )
        if not state.get("failures"):
            return self._result(state, -0.27, "Revision was attempted before a test failure.")
        revisions = state.get("revisions", 0) + 1
        if revisions > self.MAX_REVISIONS:
            return self._result(
                state,
                -0.5,
                f"No passing patch after {self.MAX_REVISIONS} revisions.",
                status="failed",
                error=f"No passing patch after {self.MAX_REVISIONS} revisions.",
                revisions=revisions,
            )
        test_output = state.get("last_test_output", "")
        if self._failure_ids_are_preexisting(
            state.get("baseline_test_output", ""), test_output
        ):
            remaining = ", ".join(sorted(self._pytest_failure_ids(test_output)))
            feedback = (
                "The remaining failing pytest node IDs were already present in the "
                f"unmodified baseline: {remaining}. They are evaluation evidence, not a "
                "new repair target. Do not change production exception messages or tests "
                "solely to make those pre-existing failures pass. Re-inspect the current "
                "workspace, retain the approved changes, and propose only a patch that "
                "directly addresses the original Issue's unmet acceptance criteria. Use "
                "exact current old_text and include the production source file."
            )
        else:
            feedback = (
                "The previous approved patch was applied in the disposable workspace, but "
                "the targeted test introduced or retained an Issue-related failure. "
                "Re-inspect the current workspace before proposing a new patch. Keep "
                "already-applied changes, include the production source file, and use "
                "exact current old_text. Test output:\n"
                f"{test_output[-3500:]}"
            )
        return self._result(
            state,
            -0.1,
            "Test failure invalidated the old plan; the graph will re-inspect the workspace and generate a new patch.",
            approved=False,
            patch_applied=False,
            plan_ready=False,
            context_ready=False,
            plan="",
            patches=[],
            patch_old="",
            patch_new="",
            plan_attempts=0,
            planner_feedback=feedback,
            # The revision count preserves the failure history; clearing this flag lets
            # the policy test the newly applied patch instead of revising immediately.
            failures=0,
            revisions=revisions,
        )

    def learn(self, state: AgentState) -> dict[str, object]:
        action = state.get("last_action")
        previous_observation = state.get("last_observation")
        reward = float(state.get("last_reward", 0.0))
        done = state.get("status") in {"completed", "failed"}
        if action and previous_observation:
            self.policy.update(
                previous_observation, action, reward, self.policy.observation(state), done
            )
        if not done and state.get("steps", 0) >= state.get("max_steps", 12):
            timeout_reward = -1.0
            return {
                "status": "failed",
                "error": "Maximum workflow steps exceeded",
                "last_reward": reward + timeout_reward,
                "total_reward": float(state.get("total_reward", 0.0)) + reward + timeout_reward,
            }
        return {"total_reward": float(state.get("total_reward", 0.0)) + reward}

    def route_after_learning(self, state: AgentState) -> str:
        return END if state.get("status") in {"completed", "failed"} else "decide"

    def _root(self, state: AgentState):
        from pathlib import Path

        return Path(state["workspace"])

    @staticmethod
    def _targets_from_search(evidence: str) -> list[str]:
        """Extract unique repository-relative paths from search tool output."""

        targets: list[str] = []
        for line in evidence.splitlines():
            match = re.match(r"^(.+?):\d+:", line)
            if match and match.group(1) not in targets:
                targets.append(match.group(1))
        return targets

    @staticmethod
    def _target_from_search(evidence: str, search_term: str = "") -> str:
        """Prefer a searched Python function's definition over call sites."""

        targets = RepositoryAgentGraph._targets_from_search(evidence)
        if re.fullmatch(r"[A-Za-z_]\w*", search_term):
            definition = re.compile(
                rf"^(?P<target>.+?):\d+:\s*(?:async\s+)?def\s+{re.escape(search_term)}\s*\("
            )
            for line in evidence.splitlines():
                match = definition.match(line)
                if match:
                    return match.group("target")
        return targets[0] if targets else ""

    @staticmethod
    def _is_test_path(relative_path: str) -> bool:
        parts = Path(relative_path).as_posix().lower().split("/")
        return "tests" in parts or parts[-1].startswith("test_")

    def _test_context(self, root: Path, test_target: str) -> str:
        """Provide exact focused-test text so the model can propose safe test edits."""

        snippets: list[str] = []
        for selector in shlex.split(test_target, posix=False):
            relative_path = selector.split("::", 1)[0]
            if not relative_path or relative_path.startswith("-"):
                continue
            path = (root / relative_path).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                continue
            snippets.append(f"{relative_path}:\n{path.read_text(encoding='utf-8')[:8000]}")
        return "\n\n".join(snippets) or "No focused test file was supplied."

    @staticmethod
    def _review_plan(
        plan_text: str,
        task: TaskSpec,
        proposal: PatchProposal,
        verification_target: str,
        target_source: str,
    ) -> str:
        """Build a complete review plan from the accepted patch, not LLM prose alone."""

        source_files = sorted(
            {
                patch.target_file
                for patch in proposal.patches
                if not RepositoryAgentGraph._is_test_path(patch.target_file)
            }
        )
        test_files = sorted(
            {
                patch.target_file
                for patch in proposal.patches
                if RepositoryAgentGraph._is_test_path(patch.target_file)
            }
        )
        scope = ["Implementation and verification scope (derived from the approved patch):"]
        step = 1
        if source_files:
            scope.append(f"{step}. Production files changed: {', '.join(source_files)}.")
        else:
            scope.append(f"{step}. Production files changed: none.")
        step += 1
        if RepositoryAgentGraph._is_cross_platform_error_message_assertion_issue(
            task.objective
        ):
            scope.append(
                f"{step}. Assertion portability: remove only the `match=` constraint from "
                "the selected `pytest.raises(ValueError)` assertion; preserve the test body "
                "and the required `ValueError`."
            )
            step += 1
        else:
            scope.append(
                f"{step}. Apply {len(proposal.patches)} exact approved replacement(s); "
                "the displayed patch preview is the reviewable source of truth."
            )
            step += 1
        if RepositoryAgentGraph._is_pre_epoch_timestamp_issue(task.objective, target_source):
            scope.append(
                f"{step}. Timestamp behavior: remove the negative-value rejection in "
                "`from_timestamp`. `from_timestamp_ms` converts milliseconds and delegates "
                "to `from_timestamp`, so the same change enables pre-1970 values on both paths."
            )
            step += 1
        if test_files:
            scope.append(f"{step}. Regression tests changed: {', '.join(test_files)}.")
            step += 1
        scope.append(f"{step}. Verification command: pytest -q {verification_target}.")
        model_summary = plan_text.strip() or "No model rationale was provided."
        return f"Model proposal summary:\n{model_summary}\n\n" + "\n".join(scope)

    @staticmethod
    def _is_pre_epoch_timestamp_issue(objective: str, target_source: str) -> bool:
        normalized = objective.lower()
        return (
            "timestamp" in normalized
            and any(term in normalized for term in ("negative", "before 1970", "pre-1970"))
            and "def from_timestamp" in target_source
            and "def from_timestamp_ms" in target_source
            and "return from_timestamp(" in target_source
        )

    @staticmethod
    def _is_cross_platform_error_message_assertion_issue(objective: str) -> bool:
        """Return whether the Issue explicitly asks to make a test assertion portable."""

        normalized = objective.lower()
        assertion_terms = (
            "assertion",
            "assert ",
            "match=",
            "error message",
            "exception message",
            "错误消息",
            "异常消息",
            "断言",
        )
        portability_terms = (
            "cross-platform",
            "cross platform",
            "platform-specific",
            "platform dependent",
            "跨平台",
            "平台相关",
        )
        return any(term in normalized for term in assertion_terms) and any(
            term in normalized for term in portability_terms
        )

    @classmethod
    def _required_source_targets(
        cls, task: TaskSpec, source_targets: list[str]
    ) -> list[str]:
        """Allow test-only edits only for explicit cross-platform assertion fixes."""

        if not cls._is_cross_platform_error_message_assertion_issue(task.objective):
            return source_targets
        if any(not cls._is_test_path(operation.target_file) for operation in task.patch_operations()):
            raise ValueError(
                "A cross-platform error-message assertion patch must modify only test files"
            )
        if not all(
            cls._is_pytest_error_message_match_relaxation(operation)
            for operation in task.patch_operations()
        ):
            raise ValueError(
                "A cross-platform error-message assertion patch must only remove a "
                "match= argument from a pytest.raises(ValueError) assertion"
            )
        return []

    @staticmethod
    def _is_pytest_error_message_match_relaxation(operation: PatchOperation) -> bool:
        """Allow only a precise relaxation of one pytest error-message assertion."""

        old_statement = RepositoryAgentGraph._pytest_with_statement(operation.old_text)
        new_statement = RepositoryAgentGraph._pytest_with_statement(operation.new_text)
        if old_statement is None or new_statement is None:
            return False
        old_call = old_statement.items[0].context_expr
        new_call = new_statement.items[0].context_expr
        if not (
            isinstance(old_call, ast.Call)
            and isinstance(new_call, ast.Call)
            and isinstance(old_call.func, ast.Attribute)
            and isinstance(new_call.func, ast.Attribute)
            and isinstance(old_call.func.value, ast.Name)
            and isinstance(new_call.func.value, ast.Name)
            and old_call.func.value.id == new_call.func.value.id == "pytest"
            and old_call.func.attr == new_call.func.attr == "raises"
            and len(old_call.args) == len(new_call.args) == 1
            and isinstance(old_call.args[0], ast.Name)
            and isinstance(new_call.args[0], ast.Name)
            and old_call.args[0].id == new_call.args[0].id == "ValueError"
        ):
            return False
        old_match_keywords = [keyword for keyword in old_call.keywords if keyword.arg == "match"]
        new_match_keywords = [keyword for keyword in new_call.keywords if keyword.arg == "match"]
        if len(old_match_keywords) != 1 or new_match_keywords:
            return False
        if any(keyword.arg is None for keyword in old_call.keywords + new_call.keywords):
            return False
        old_without_match = ast.Call(
            func=old_call.func,
            args=old_call.args,
            keywords=[keyword for keyword in old_call.keywords if keyword.arg != "match"],
        )
        return (
            ast.dump(old_without_match, include_attributes=False)
            == ast.dump(new_call, include_attributes=False)
            and RepositoryAgentGraph._ast_nodes_match(
                old_statement.items[0].optional_vars,
                new_statement.items[0].optional_vars,
            )
        )

    @staticmethod
    def _ast_nodes_match(left: ast.AST | None, right: ast.AST | None) -> bool:
        """Compare AST nodes while preserving an absent ``as`` binding."""

        if left is None or right is None:
            return left is right
        return ast.dump(left, include_attributes=False) == ast.dump(
            right, include_attributes=False
        )

    @staticmethod
    def _pytest_with_statement(text: str) -> ast.With | None:
        """Parse exactly one ``with pytest.raises(...)`` header from a patch fragment."""

        lines = text.strip().splitlines()
        if len(lines) != 1:
            return None
        header = lines[0].strip()
        if not header.startswith("with ") or not header.endswith(":"):
            return None
        try:
            parsed = ast.parse(f"{header}\n    pass\n")
        except SyntaxError:
            return None
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.With):
            return None
        statement = parsed.body[0]
        return statement if len(statement.items) == 1 else None

    @staticmethod
    def _is_test_environment_failure(output: str) -> bool:
        """Identify failures that a source patch cannot repair."""

        lowered = output.lower()
        if "dependency setup failed" in lowered or "could not create isolated test environment" in lowered:
            return True
        missing_module = "no module named" in lowered
        collection_import = "importerror while loading conftest" in lowered
        return missing_module and collection_import

    @staticmethod
    def _pytest_failure_ids(output: str) -> set[str]:
        """Return pytest node IDs from its stable summary lines."""

        return set(re.findall(r"^FAILED\s+(\S+?)(?:\s+-\s+|$)", output, re.MULTILINE))

    @classmethod
    def _only_preexisting_failures(cls, baseline_output: str, current_output: str) -> bool:
        """Accept a regression test that improves passing coverage without new failures."""

        baseline_passed = cls._pytest_passed_count(baseline_output)
        current_passed = cls._pytest_passed_count(current_output)
        return cls._failure_ids_are_preexisting(
            baseline_output, current_output
        ) and current_passed >= baseline_passed

    @classmethod
    def _failure_ids_are_preexisting(cls, baseline_output: str, current_output: str) -> bool:
        """Return whether every current pytest failure was present before patching."""

        baseline = cls._pytest_failure_ids(baseline_output)
        current = cls._pytest_failure_ids(current_output)
        return bool(baseline and current and current.issubset(baseline))

    @staticmethod
    def _validate_issue_alignment(
        task: TaskSpec, proposal: PatchProposal, state: AgentState
    ) -> None:
        """Reject a syntactically valid patch that misses a known Issue guard.

        Repository validation can prove that a replacement applies, but it cannot
        prove that a model kept the Issue's causal source in scope. This targeted
        rule covers timestamp Issues where the inspected function explicitly
        rejects negative POSIX values.
        """

        objective = task.objective.lower()
        source = state.get("target_source", "")
        timestamp_rejection_issue = "timestamp" in objective and any(
            phrase in objective
            for phrase in ("negative", "before 1970", "pre-1970", "reject", "rejection")
        )
        negative_guard = "if value < 0:"
        if not (timestamp_rejection_issue and negative_guard in source):
            return
        if any(
            negative_guard in patch.old_text or negative_guard in patch.new_text
            for patch in proposal.patches
        ):
            return
        # A later correction may legitimately modify another location only
        # after the approved patch has already removed this Issue's guard.
        if state.get("issue_anchor_addressed"):
            return
        raise ValueError(
            "Proposal does not address the negative-value guard identified in the Issue source"
        )

    @staticmethod
    def _proposal_addresses_negative_guard(proposal: PatchProposal) -> bool:
        """Return whether a proposal changes the timestamp negative-value guard."""

        return any(
            "if value < 0:" in patch.old_text or "if value < 0:" in patch.new_text
            for patch in proposal.patches
        )

    @staticmethod
    def _pytest_passed_count(output: str) -> int:
        # Pytest emits either a comma-separated summary or a standalone line
        # such as ``1 passed in 0.05s`` for a single selected node.
        match = re.search(
            r"(?m)(?:^|,|\n)[ \t]*(\d+)[ \t]+passed(?:,|[ \t]|$)", output
        )
        return int(match.group(1)) if match else 0

    @staticmethod
    def _task(state: AgentState) -> TaskSpec:
        if state.get("task_kind") != "custom":
            return get_task(state["task_id"])
        return TaskSpec(
            task_id=state["task_id"],
            title=state.get("task_title", "Custom repository task"),
            objective=state.get("objective", ""),
            target_file=state.get("target_file", ""),
            search_term=state.get("search_term", ""),
            old_text=state.get("patch_old", ""),
            new_text=state.get("patch_new", ""),
            test_target=state.get("test_target", "tests"),
            acceptance_note="The selected test target must pass after the patch.",
            patches=tuple(
                PatchOperation(
                    target_file=patch["target_file"],
                    old_text=patch["old_text"],
                    new_text=patch["new_text"],
                )
                for patch in state.get("patches", [])
            ),
        )

    @staticmethod
    def _result(
        state: AgentState, reward: float, detail: str, **updates: object
    ) -> dict[str, object]:
        action = state.get("last_action", "unknown")
        result: dict[str, object] = {
            "last_reward": reward,
            "events": _append_event(state, str(action), detail, reward),
        }
        result.update(updates)
        return result
