from __future__ import annotations

from adaptive_agent.models import TaskSpec

TASKS: dict[str, TaskSpec] = {
    "division-by-zero": TaskSpec(
        task_id="division-by-zero",
        title="Reject division by zero",
        objective="Make divide() reject a zero denominator with a ValueError.",
        target_file="src/calculator.py",
        search_term="def divide",
        old_text="""    if denominator == 0:\n        return 0\n""",
        new_text="""    if denominator == 0:\n        raise ValueError(\"denominator cannot be zero\")\n""",
        test_target="tests/test_calculator.py::test_divide_by_zero_raises",
        acceptance_note="The targeted test must observe a ValueError instead of a silent zero.",
    ),
    "decimal-average": TaskSpec(
        task_id="decimal-average",
        title="Preserve decimal precision in average",
        objective="Make average() return an exact arithmetic mean instead of flooring it.",
        target_file="src/calculator.py",
        search_term="def average",
        old_text="""    return sum(values) // len(values)\n""",
        new_text="""    return sum(values) / len(values)\n""",
        test_target="tests/test_calculator.py::test_average_preserves_decimal_precision",
        acceptance_note="The targeted test must return 1.5 for [1, 2].",
    ),
}


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark task: {task_id}") from exc
