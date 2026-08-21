from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, model_validator

from adaptive_agent.models import PatchOperation


class FilePatch(BaseModel):
    """One exact replacement in a repository file."""

    target_file: str = Field(description="Repository-relative file to edit.")
    old_text: str = Field(description="Exact existing text to replace once.")
    new_text: str = Field(
        description="Replacement text, preserving indentation. Use an empty string only to delete old_text."
    )

    @model_validator(mode="after")
    def require_target_and_context(self) -> FilePatch:
        if not self.target_file.strip():
            raise ValueError("target_file must not be empty")
        if not self.old_text:
            raise ValueError("old_text must not be empty")
        return self


class PatchProposal(BaseModel):
    """Structured output required before a patch can reach the approval gate."""

    plan: str = Field(
        description=(
            "A concise, ordered implementation plan. State the direct code change, "
            "any relevant delegated behavior, regression coverage, and verification scope."
        )
    )
    patches: list[FilePatch] = Field(
        default_factory=list,
        description="All exact file replacements needed for the issue, including regression tests.",
    )
    # These fields keep old provider responses readable while the prompt migrates to patches.
    target_file: str | None = Field(default=None, description="Legacy single-file target.")
    old_text: str | None = Field(default=None, description="Legacy single-file context.")
    new_text: str | None = Field(default=None, description="Legacy single-file replacement.")
    test_target: str = Field(
        default="tests",
        description="A pytest path or node selector to run after the patch.",
    )

    @model_validator(mode="after")
    def normalize_legacy_patch(self) -> PatchProposal:
        if (
            not self.patches
            and self.target_file
            and self.old_text
            and self.new_text is not None
        ):
            self.patches = [
                FilePatch(
                    target_file=self.target_file,
                    old_text=self.old_text,
                    new_text=self.new_text,
                )
            ]
        if not self.patches:
            raise ValueError("At least one file patch is required")
        return self

    def patch_operations(self) -> tuple[PatchOperation, ...]:
        return tuple(
            PatchOperation(
                target_file=patch.target_file,
                old_text=patch.old_text,
                new_text=patch.new_text,
            )
            for patch in self.patches
        )


class DeterministicPlanner:
    """A zero-key planner for the local demo.

    The RunnableLambda is intentionally shaped like a future structured chat-model
    chain, so it can be replaced by `ChatOpenAI.with_structured_output(...)` without
    changing the graph contract.
    """

    def __init__(self) -> None:
        self.chain = RunnableLambda(self._make_plan).with_config({"run_name": "task_planner"})

    @staticmethod
    def _make_plan(payload: dict[str, str]) -> str:
        return (
            f"Objective: {payload['objective']}\n"
            f"Evidence: {payload['evidence']}\n"
            f"Patch target: {payload['target_file']}\n"
            "Steps: inspect the target function, apply the approved minimal change, "
            "then run the focused regression test.\n"
            f"Acceptance: {payload['acceptance']}"
        )

    def invoke(self, payload: dict[str, str]) -> str:
        return self.chain.invoke(payload)


class TaskPlanner:
    """Use an OpenAI-compatible model when configured, deterministic text otherwise."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
        timeout: float | None = None,
    ) -> None:
        self.deterministic = DeterministicPlanner()
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.provider = provider.lower()
        self._llm_chain = None
        self._llm = None
        request_timeout = timeout or (30.0 if self.provider == "openai" else 90.0)
        configured_key = api_key or os.getenv("OPENAI_API_KEY")
        if configured_key:
            from langchain_openai import ChatOpenAI

            model_kwargs: dict[str, object] = {
                "model": self.model_name,
                "api_key": configured_key,
                "temperature": 0,
                "timeout": request_timeout,
                "max_retries": 0,
            }
            configured_base_url = base_url or os.getenv("OPENAI_BASE_URL")
            if configured_base_url:
                model_kwargs["base_url"] = configured_base_url
            self._llm = ChatOpenAI(**model_kwargs)
            if self.provider == "openai":
                self._llm_chain = self._llm.with_structured_output(PatchProposal)

    @property
    def model_enabled(self) -> bool:
        return self._llm is not None

    def invoke(self, payload: dict[str, str]) -> PatchProposal | str:
        if self._llm is None:
            return self.deterministic.invoke(payload)
        context = payload.get("context", "")
        prompt = (
            "You are a cautious repository maintenance agent. Analyze the issue and the "
            "repository context below. Return one minimal, reviewable patch proposal. "
            "Do not invent files or shell commands. old_text must match the repository "
            "exactly, and every target_file must be relative to the repository root.\n\n"
            f"Issue:\n{payload['objective']}\n\nRepository context:\n{context[:16000]}\n\n"
            f"Acceptance criteria:\n{payload.get('acceptance', 'The focused tests must pass.')}\n\n"
            f"Task-specific safety instruction:\n{payload.get('safety_instruction') or 'None'}\n\n"
            "The proposal will be checked and shown to a human before it is applied.\n\n"
            f"Correction feedback from an earlier attempt:\n{payload.get('feedback', 'None')}\n\n"
            "Copy every old_text exactly from the provided source, including whitespace and indentation. "
            "new_text may be an empty JSON string only when deleting a complete, contiguous block. "
            "Treat test diagnostics as evaluation evidence, not as a request to repair unrelated "
            "pre-existing failures. The patch must directly address the Issue. Follow any "
            "task-specific safety instruction when it explicitly defines an allowed test-only "
            "assertion change. "
            "If correction feedback says a previous patch was applied, re-read the current files "
            "and propose a fresh patch; do not blindly repeat stale context. Include the production "
            "source change and any required regression-test changes in patches. For an Issue "
            "explicitly limited to a cross-platform error-message assertion, a test-only patch is "
            "appropriate; do not change production exception text solely to satisfy that assertion. "
            "Write plan as an ordered 3-5 step review summary. It must distinguish the function "
            "changed directly from any behavior reached through delegation, name each Issue behavior "
            "covered by the regression tests, and state how the selected pytest target verifies it. "
            "Return only a JSON object with exactly these keys: plan, patches, test_target. "
            "Each patches item must have target_file, old_text, and new_text. Do not leave target_file "
            "or old_text blank. Do not wrap it in Markdown fences."
        )
        if self._llm_chain is not None:
            return self._llm_chain.invoke(prompt)
        response = self._llm.invoke(prompt)
        return self._parse_json_proposal(response.content)

    @staticmethod
    def _parse_json_proposal(content: Any) -> PatchProposal:
        text = str(content).strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            return PatchProposal.model_validate_json(text)
        except Exception as exc:
            raise ValueError(
                "The selected model did not return a valid patch JSON object: "
                f"{exc}"
            ) from exc
