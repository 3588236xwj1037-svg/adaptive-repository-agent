from __future__ import annotations

import difflib
import re
from typing import Any

import streamlit as st

from adaptive_agent.graph import RepositoryAgentGraph
from adaptive_agent.runtime import AgentRuntime

APP_VERSION = "0.20-benchmark-patch-audit"

_PROVIDER_CONFIG = {
    "openai": {
        "key_names": ("OPENAI_API_KEY",),
        "model_name": "OPENAI_MODEL",
        "base_url": "OPENAI_BASE_URL",
        "default_model": "gpt-4.1-mini",
        "default_base_url": None,
    },
    "deepseek": {
        "key_names": ("DEEPSEEK_API_KEY",),
        "model_name": "DEEPSEEK_MODEL",
        "base_url": "DEEPSEEK_BASE_URL",
        "default_model": "deepseek-chat",
        "default_base_url": "https://api.deepseek.com/v1",
    },
    "qwen": {
        "key_names": ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        "model_name": "QWEN_MODEL",
        "base_url": "QWEN_BASE_URL",
        "default_model": "qwen-plus",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}

_PROVIDER_ALIASES = {
    "dashscope": "qwen",
    "tongyi": "qwen",
    "通义千问": "qwen",
}


def _streamlit_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def _selected_provider() -> str:
    configured = (_streamlit_secret("LLM_PROVIDER") or "").strip().lower()
    configured = _PROVIDER_ALIASES.get(configured, configured)
    if configured in _PROVIDER_CONFIG:
        return configured
    for provider, settings in _PROVIDER_CONFIG.items():
        if any(_streamlit_secret(name) for name in settings["key_names"]):
            return provider
    return "openai"


def _new_runtime() -> AgentRuntime:
    provider = _selected_provider()
    settings = _PROVIDER_CONFIG[provider]
    api_key = next(
        (_streamlit_secret(name) for name in settings["key_names"] if _streamlit_secret(name)),
        None,
    )
    model_name = _streamlit_secret(settings["model_name"]) or settings["default_model"]
    base_url = _streamlit_secret(settings["base_url"]) or settings["default_base_url"]
    timeout_value = _streamlit_secret("LLM_TIMEOUT")
    timeout = float(timeout_value) if timeout_value else None
    return AgentRuntime(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        provider=provider,
        timeout=timeout,
    )


def _initialize_session() -> None:
    if st.session_state.get("_app_version") != APP_VERSION:
        st.session_state.runtime = _new_runtime()
        st.session_state.last_outcome = None
        st.session_state.pending_config = None
        st.session_state.training_history = []
        st.session_state._app_version = APP_VERSION
    if "runtime" not in st.session_state:
        st.session_state.runtime = _new_runtime()
    st.session_state.setdefault("last_outcome", None)
    st.session_state.setdefault("pending_config", None)
    st.session_state.setdefault("training_history", [])


def _runtime() -> AgentRuntime:
    return st.session_state.runtime


def _set_result(outcome: dict[str, Any]) -> None:
    st.session_state.last_outcome = outcome
    st.session_state.pending_config = (
        outcome["config"] if "__interrupt__" in outcome["result"] else None
    )


def _outcome() -> dict[str, Any] | None:
    return st.session_state.get("last_outcome")


def _render_events(result: dict[str, Any]) -> None:
    events = result.get("events", [])
    if not events:
        st.info("The graph has not completed an action yet.")
        return
    st.dataframe(events, hide_index=True)
    with st.expander("Latest action detail", expanded=False):
        st.code(str(events[-1].get("detail", "")), language="text")


def _patch_preview(patch: dict[str, Any]) -> str:
    """Render an exact, compact unified diff for an approved replacement."""

    target = str(patch.get("target_file", "unknown file"))
    old_text = str(patch.get("old_text", ""))
    new_text = str(patch.get("new_text", ""))
    return "\n".join(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
            lineterm="",
        )
    ) or "No textual change was recorded."


def _render_patch_review(patches: list[object], *, title: str) -> None:
    """Show the exact, already validated patch before or after approval."""

    valid_patches = [patch for patch in patches if isinstance(patch, dict)]
    if not valid_patches:
        return
    with st.container(border=True):
        st.caption(title)
        st.dataframe(
            [
                {
                    "file": patch.get("target_file", ""),
                    "removed lines": len(str(patch.get("old_text", "")).splitlines()),
                    "added lines": len(str(patch.get("new_text", "")).splitlines()),
                }
                for patch in valid_patches
            ],
            hide_index=True,
        )
        for index, patch in enumerate(valid_patches, start=1):
            st.markdown(f"**Change {index}: `{patch.get('target_file', 'unknown file')}`**")
            st.code(_patch_preview(patch), language="diff")


def _pytest_summary(output: str) -> dict[str, str]:
    """Extract compact comparison data while keeping the full pytest output available."""

    passed = RepositoryAgentGraph._pytest_passed_count(output)
    failed_match = re.search(r"(?m)(?:^|,)\s*(\d+) failed(?:,|\s|$)", output)
    failed = failed_match.group(1) if failed_match else "0"
    failure_ids = sorted(RepositoryAgentGraph._pytest_failure_ids(output))
    return {
        "passed": str(passed),
        "failed": failed,
        "failure IDs": ", ".join(failure_ids) if failure_ids else "None",
    }


def _render_run_result() -> None:
    outcome = _outcome()
    if not outcome:
        st.info("Select a task and run the graph to inspect its state transitions.")
        return

    result = outcome["result"]
    interrupted = "__interrupt__" in result
    status = "Awaiting human approval" if interrupted else result.get("status", "Unknown")
    test_status = "Passed" if result.get("test_passed") else "Pending"
    if result.get("baseline_failures_remain"):
        test_status = "Verified vs baseline"
    with st.container(horizontal=True):
        st.metric("Status", status.replace("_", " ").title(), border=True)
        st.metric("Reward", f"{float(result.get('total_reward', 0.0)):.2f}", border=True)
        st.metric("Steps", result.get("steps", 0), border=True)
        st.metric("Test", test_status, border=True)

    if interrupted:
        st.warning("The LangGraph run is paused at `interrupt()` before applying a patch.")
        pending = result["__interrupt__"][0]
        payload = getattr(pending, "value", pending)
        if isinstance(payload, dict):
            st.markdown(f"**{payload.get('title', 'Proposed repository change')}**")
            st.caption(payload.get("message", "Review the proposed change before continuing."))
            # Keep the approval surface flat. Nested expanders can cause a
            # React removeChild error when Streamlit reruns after approval.
            with st.container(border=True):
                st.caption("Proposed implementation plan")
                st.code(payload.get("plan", ""), language="text")
                patches = payload.get("patches", [])
            _render_patch_review(patches, title="Proposed exact file changes")
        else:
            st.code(str(payload), language="text")
        with st.container(horizontal=True):
            if st.button("Approve patch", type="primary", key="approve_patch"):
                resumed = _runtime().resume(st.session_state.pending_config, approved=True)
                _set_result(resumed)
                st.rerun()
            if st.button("Reject patch", key="reject_patch"):
                resumed = _runtime().resume(st.session_state.pending_config, approved=False)
                _set_result(resumed)
                st.rerun()

    _render_events(result)

    if result.get("error"):
        st.error(result["error"])
    elif result.get("status") == "completed":
        patches = result.get("patches", [])
        task_is_portable_assertion = RepositoryAgentGraph._is_cross_platform_error_message_assertion_issue(
            str(result.get("objective", ""))
        )
        only_test_files = bool(patches) and all(
            RepositoryAgentGraph._is_test_path(str(patch.get("target_file", "")))
            for patch in patches
            if isinstance(patch, dict)
        )
        if task_is_portable_assertion and only_test_files:
            st.success(
                "Verified cross-platform assertion fix: only the selected test assertion "
                "changed; production code was not modified."
            )
        _render_patch_review(patches, title="Applied exact file changes")
    if result.get("baseline_test_output") and result.get("last_test_output"):
        baseline_summary = _pytest_summary(str(result["baseline_test_output"]))
        patched_summary = _pytest_summary(str(result["last_test_output"]))
        st.caption("Targeted-test comparison")
        st.dataframe(
            [
                {"run": "Before patch", **baseline_summary},
                {"run": "After patch", **patched_summary},
            ],
            hide_index=True,
        )
    if result.get("last_test_output"):
        with st.expander("Test diagnostics", expanded=bool(result.get("error"))):
            st.code(str(result["last_test_output"]), language="text")
    if result.get("baseline_test_output"):
        with st.expander("Baseline test diagnostics", expanded=False):
            st.code(str(result["baseline_test_output"]), language="text")
    if result.get("baseline_failures_remain"):
        patched_summary = _pytest_summary(str(result["last_test_output"]))
        remaining = patched_summary["failure IDs"]
        st.warning(
            "The current Issue was verified relative to the pre-patch baseline, but the "
            "selected pytest target is not fully green. No new failures were introduced; "
            f"{patched_summary['failed']} pre-existing failure(s) remain: {remaining}."
        )
    if result.get("test_environment_failed"):
        st.warning(
            "The test environment failed before the selected tests could run. "
            "The Agent stopped instead of proposing an unrelated source-code change."
        )
    if result.get("revisions"):
        st.caption(
            f"Patch revision attempts: {result['revisions']}/{RepositoryAgentGraph.MAX_REVISIONS}"
        )

    if result.get("plan"):
        with st.expander("Generated plan and verification scope", expanded=False):
            st.code(result["plan"], language="text")
    if result.get("workspace"):
        st.caption(f"Disposable workspace: `{result['workspace']}`")
    if result.get("source_repository"):
        st.caption(f"Source repository: `{result['source_repository']}`")


def _render_training() -> None:
    runtime = _runtime()
    st.subheader("Policy training")
    st.info(
        "This trains the Q-learning routing table for workflow actions. "
        "It does not read a GitHub Issue and does not train the language model."
    )
    st.caption(
        "Training uses the disposable fixture, automated approval, and a fast patch verifier. "
        "Task runs always execute the real targeted pytest command."
    )
    episode_count = st.slider("Episodes", min_value=20, max_value=300, value=120, step=20)
    with st.container(horizontal=True):
        if st.button("Train routing policy", type="primary", key="train_policy"):
            with st.spinner("Running benchmark episodes and updating Q-values..."):
                st.session_state.training_history = runtime.train(episode_count)
            st.success("Training completed. The Q-table was saved locally.")
        if st.button("Reset policy", key="reset_policy"):
            runtime.reset_policy()
            st.session_state.training_history = []
            st.rerun()

    history = st.session_state.get("training_history", [])
    if history:
        st.line_chart({"reward": [item["reward"] for item in history]})
        completed = sum(item["status"] == "completed" for item in history)
        st.caption(f"Completed {completed}/{len(history)} training episodes.")
        st.dataframe(history[-20:], hide_index=True)

    rows = runtime.policy_rows()
    if rows:
        st.subheader("Learned policy")
        st.dataframe(rows, hide_index=True)
    else:
        st.info("No learned Q-values yet. Run a training cycle to populate the policy table.")


def render_app() -> None:
    st.set_page_config(
        page_title="Adaptive Repository Agent",
        page_icon=":material/account_tree:",
        layout="wide",
    )
    _initialize_session()
    runtime = _runtime()
    st.title("Adaptive Repository Agent")
    st.caption(
        "Give it a local repository and an Issue. It will inspect, plan, request approval, "
        "patch a disposable copy, and run tests."
    )

    with st.sidebar:
        st.subheader("Task run")
        run_mode = st.segmented_control(
            "Run mode",
            ["Local repository", "Benchmark demo"],
            default="Local repository",
            key="run_mode",
        )
        max_steps = st.slider(
            "Maximum graph steps", min_value=6, max_value=20, value=12, key="max_steps"
        )
        approval_mode = st.toggle("Pause for human approval", value=True, key="approval_mode")

        if run_mode == "Local repository":
            repository_path = st.text_input(
                "Repository path",
                placeholder=r"C:\projects\your-repository",
                key="repository_path",
            )
            issue = st.text_area(
                "Issue or task",
                placeholder="Fix the bug described in the GitHub Issue...",
                height=130,
                key="issue_text",
            )
            search_term = st.text_input(
                "Search term (optional)",
                placeholder="function name, class name, or error text",
                key="search_term",
            )
            test_target = st.text_input(
                "Pytest target",
                value="tests",
                help="A relative pytest path or node selector, for example tests/test_api.py::test_login.",
                key="test_target",
            )
            st.caption(
                f"Model provider: {runtime.provider} | Model: {runtime.planner.model_name}"
            )
            if not runtime.model_enabled:
                st.info(
                    "Set the selected provider's API key in .streamlit/secrets.toml "
                    "to generate structured patches."
                )
        else:
            task_id = st.selectbox(
                "Benchmark task",
                options=list(runtime.tasks),
                format_func=lambda item: runtime.tasks[item].title,
                key="selected_task",
            )
            st.info(
                "Benchmark demo: this uses the built-in calculator fixture. "
                "No GitHub repository or Issue is involved."
            )

        if st.button("Run Agent", type="primary", key="run_agent"):
            try:
                if run_mode == "Local repository":
                    if not repository_path.strip() or not issue.strip():
                        st.error("Enter a repository path and an Issue before running the Agent.")
                    else:
                        with st.spinner(
                            "Running Agent: inspecting the repository and asking the model for a patch..."
                        ):
                            outcome = runtime.run_custom_task(
                                repository_path,
                                issue,
                                search_term=search_term,
                                test_target=test_target,
                                auto_approve=not approval_mode,
                                max_steps=max_steps,
                            )
                        _set_result(outcome)
                else:
                    with st.spinner("Running benchmark workflow..."):
                        outcome = runtime.run_task(
                            task_id,
                            auto_approve=not approval_mode,
                            epsilon=0.0,
                            max_steps=max_steps,
                        )
                    _set_result(outcome)
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(
                    f"Agent failed while calling the {runtime.provider} model. Check its API key, "
                    f"model/base URL, and network access. "
                    f"Details: {type(exc).__name__}: {exc}"
                )

        if run_mode == "Benchmark demo":
            task = runtime.tasks[task_id]
            st.caption(task.objective)
            st.caption(f"Acceptance: {task.acceptance_note}")

    run_tab, train_tab, design_tab = st.tabs(["Task run", "RL training", "Design"])
    with run_tab:
        _render_run_result()
    with train_tab:
        _render_training()
    with design_tab:
        st.subheader("Responsibility boundary")
        st.markdown(
            """
            - **LangGraph:** explicit task state, conditional routing, retry loops, checkpoints, and approval interrupts.
            - **LangChain:** repository tools and structured-output patch proposals for local repositories.
            - **Q-learning:** selects the next workflow action from a finite action space; it does not train the language model.
            - **Safety:** every run copies a repository into a disposable workspace. Writes happen only after approval, and tests run from an allowlisted pytest entry point.
            - **Recovery:** a code-test failure invalidates the old plan, refreshes repository evidence, and requests a newly generated patch. Test-environment failures stop with diagnostics instead of repeating a stale patch.
            """
        )
