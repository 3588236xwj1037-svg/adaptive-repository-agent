from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from adaptive_agent.models import ActionName, AgentState

ALL_ACTIONS: tuple[ActionName, ...] = (
    "retrieve",
    "plan",
    "request_approval",
    "apply_patch",
    "run_test",
    "revise",
)


class QLearningPolicy:
    """Small, inspectable Q-learning policy for graph routing decisions.

    The policy optimizes workflow choices, not language-model token generation.
    That makes the action space finite and keeps training reproducible on a local
    benchmark.
    """

    def __init__(
        self,
        alpha: float = 0.22,
        gamma: float = 0.9,
        seed: int = 7,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self._rng = random.Random(seed)
        self._table: dict[str, dict[str, float]] = defaultdict(dict)

    @staticmethod
    def observation(state: AgentState) -> str:
        """Compress workflow state into a stable, finite RL observation."""

        return "|".join(
            (
                f"context={int(state.get('context_ready', False))}",
                f"plan={int(state.get('plan_ready', False))}",
                f"approved={int(state.get('approved', False))}",
                f"patch={int(state.get('patch_applied', False))}",
                f"test={int(state.get('test_passed', False))}",
                f"failures={min(state.get('failures', 0), 2)}",
                f"revisions={min(state.get('revisions', 0), 2)}",
            )
        )

    def choose(self, state: AgentState, epsilon: float) -> ActionName:
        observation = self.observation(state)
        legal_actions = self.legal_actions(state)
        if self._rng.random() < epsilon:
            return self._rng.choice(legal_actions)

        values = self._table[observation]
        if not values or max(values.values(), default=0.0) == 0.0:
            return self._bootstrap_action(state)
        best_value = max(values.get(action, 0.0) for action in legal_actions)
        for action in legal_actions:
            if values.get(action, 0.0) == best_value:
                return action
        return "retrieve"

    @staticmethod
    def legal_actions(state: AgentState) -> tuple[ActionName, ...]:
        """Mask impossible transitions while leaving meaningful choices to RL."""

        if not state.get("context_ready"):
            return ("retrieve",)
        if not state.get("plan_ready"):
            return ("plan",)
        if not state.get("approved"):
            return ("request_approval",)
        if not state.get("patch_applied"):
            return ("apply_patch",)
        if state.get("failures", 0):
            # A failed test must invalidate the current patch before another test run.
            return ("revise",)
        return ("run_test", "revise")

    @staticmethod
    def _bootstrap_action(state: AgentState) -> ActionName:
        """A deterministic tie-breaker before the policy has experience."""

        if not state.get("context_ready"):
            return "retrieve"
        if not state.get("plan_ready"):
            return "plan"
        if not state.get("approved"):
            return "request_approval"
        if not state.get("patch_applied"):
            return "apply_patch"
        if state.get("failures", 0):
            return "revise"
        return "run_test"

    def update(
        self,
        previous_observation: str,
        action: ActionName,
        reward: float,
        next_observation: str,
        done: bool,
    ) -> None:
        current = self._table[previous_observation].get(action, 0.0)
        next_max = 0.0 if done else max(self._table[next_observation].values(), default=0.0)
        target = reward + self.gamma * next_max
        self._table[previous_observation][action] = current + self.alpha * (target - current)

    def reset(self) -> None:
        self._table.clear()

    def export_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for observation, actions in sorted(self._table.items()):
            for action, value in sorted(actions.items()):
                rows.append(
                    {"observation": observation, "action": action, "q_value": round(value, 4)}
                )
        return rows

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._table, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._table = defaultdict(dict, {key: dict(value) for key, value in raw.items()})
