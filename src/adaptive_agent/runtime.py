from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command

from adaptive_agent.fixtures import TASKS, get_task
from adaptive_agent.graph import RepositoryAgentGraph
from adaptive_agent.models import AgentState
from adaptive_agent.planner import TaskPlanner
from adaptive_agent.rl import QLearningPolicy
from adaptive_agent.workspace import PROJECT_ROOT, BenchmarkWorkspace


class AgentRuntime:
    """Application service shared by the CLI and Streamlit dashboard."""

    def __init__(
        self,
        policy_path: Path | None = None,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
        timeout: float | None = None,
    ) -> None:
        self.policy_path = policy_path or (PROJECT_ROOT / ".adaptive_agent" / "q_table.json")
        self.policy = QLearningPolicy()
        self.policy.load(self.policy_path)
        self.workspace = BenchmarkWorkspace()
        self.provider = provider
        self.planner = TaskPlanner(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            provider=provider,
            timeout=timeout,
        )
        self.agent = RepositoryAgentGraph(self.policy, self.workspace, planner=self.planner)

    @property
    def model_enabled(self) -> bool:
        return self.planner.model_enabled

    @property
    def tasks(self):
        return TASKS

    def reset_policy(self) -> None:
        self.policy.reset()
        if self.policy_path.exists():
            self.policy_path.unlink()

    def initial_state(
        self,
        task_id: str,
        *,
        auto_approve: bool,
        epsilon: float,
        fast_verification: bool = False,
        max_steps: int = 12,
    ) -> tuple[AgentState, dict[str, dict[str, str]]]:
        task = get_task(task_id)
        run_id = uuid.uuid4().hex[:12]
        workspace = self.workspace.create(run_id)
        state: AgentState = {
            "run_id": run_id,
            "task_id": task.task_id,
            "task_title": task.title,
            "objective": task.objective,
            "task_kind": "benchmark",
            "source_repository": str(PROJECT_ROOT / "fixtures" / "calculator"),
            "search_term": task.search_term,
            "target_file": task.target_file,
            "patch_old": task.old_text,
            "patch_new": task.new_text,
            "test_target": task.test_target,
            "workspace": str(workspace),
            "baseline_test": False,
            "auto_approve": auto_approve,
            "fast_verification": fast_verification,
            "epsilon": epsilon,
            "max_steps": max_steps,
        }
        config = {"configurable": {"thread_id": run_id}}
        return state, config

    def run_task(
        self,
        task_id: str,
        *,
        auto_approve: bool = True,
        epsilon: float = 0.0,
        fast_verification: bool = False,
        max_steps: int = 12,
    ) -> dict[str, Any]:
        state, config = self.initial_state(
            task_id,
            auto_approve=auto_approve,
            epsilon=epsilon,
            fast_verification=fast_verification,
            max_steps=max_steps,
        )
        result = self.agent.graph.invoke(state, config)
        self.policy.save(self.policy_path)
        return {"result": result, "config": config, "run_id": state["run_id"]}

    def run_custom_task(
        self,
        repository_path: str,
        issue: str,
        *,
        search_term: str = "",
        test_target: str = "tests",
        auto_approve: bool = False,
        epsilon: float = 0.0,
        max_steps: int = 14,
    ) -> dict[str, Any]:
        """Run an Issue against a copied local repository.

        A structured-output model is required to produce an arbitrary patch. The
        graph still performs repository inspection before it calls the model.
        """

        source = Path(repository_path).expanduser().resolve()
        run_id = uuid.uuid4().hex[:12]
        workspace = self.workspace.create_from_path(run_id, source)
        state: AgentState = {
            "run_id": run_id,
            "task_id": f"custom-{run_id}",
            "task_kind": "custom",
            "task_title": issue.strip()[:100] or "Custom repository task",
            "objective": issue.strip(),
            "source_repository": str(source),
            "search_term": search_term.strip(),
            "target_file": "",
            "patch_old": "",
            "patch_new": "",
            "patches": [],
            "source_targets": [],
            "test_target": test_target.strip() or "tests",
            "workspace": str(workspace),
            "baseline_test": True,
            "auto_approve": auto_approve,
            "epsilon": epsilon,
            "max_steps": max_steps,
        }
        config = {"configurable": {"thread_id": run_id}}
        result = self.agent.graph.invoke(state, config)
        self.policy.save(self.policy_path)
        return {"result": result, "config": config, "run_id": run_id}

    def resume(self, config: dict[str, dict[str, str]], approved: bool) -> dict[str, Any]:
        result = self.agent.graph.invoke(Command(resume={"approved": approved}), config)
        self.policy.save(self.policy_path)
        return {"result": result, "config": config}

    def train(self, episodes: int = 120) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        for episode in range(episodes):
            task_id = list(TASKS)[episode % len(TASKS)]
            progress = episode / max(episodes - 1, 1)
            epsilon = max(0.05, 0.55 * (1.0 - progress))
            outcome = self.run_task(
                task_id,
                auto_approve=True,
                epsilon=epsilon,
                fast_verification=True,
            )
            result = outcome["result"]
            if "__interrupt__" in result:
                raise RuntimeError("Training unexpectedly requested human approval")
            history.append(
                {
                    "episode": episode + 1,
                    "task_id": task_id,
                    "epsilon": round(epsilon, 3),
                    "status": result.get("status", "unknown"),
                    "reward": round(float(result.get("total_reward", 0.0)), 3),
                    "steps": result.get("steps", 0),
                    "test_passed": result.get("test_passed", False),
                }
            )
        return history

    def policy_rows(self) -> list[dict[str, object]]:
        return self.policy.export_rows()
