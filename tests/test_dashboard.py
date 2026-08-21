from adaptive_agent.dashboard import _patch_preview, _pytest_summary


def test_pytest_summary_preserves_multiline_failure_count() -> None:
    output = (
        "Installed project and test dependencies in the disposable workspace.\n"
        "FAILED tests/test_utils.py::test_from_timestamp_with_overflow_value "
        "- AssertionError\n"
        "1 failed, 128 passed in 0.28s\n"
    )

    assert _pytest_summary(output) == {
        "passed": "128",
        "failed": "1",
        "failure IDs": "tests/test_utils.py::test_from_timestamp_with_overflow_value",
    }


def test_pytest_summary_counts_a_standalone_single_test_pass() -> None:
    output = (
        "Installed project and test dependencies in the disposable workspace.\n"
        ".                                                                        [100%]\n"
        "1 passed in 0.05s\n"
    )

    assert _pytest_summary(output) == {
        "passed": "1",
        "failed": "0",
        "failure IDs": "None",
    }


def test_pytest_summary_counts_passes_after_a_failure_in_the_same_summary() -> None:
    output = "FAILED tests/test_utils.py::test_overflow - AssertionError\n1 failed, 128 passed\n"

    assert _pytest_summary(output)["passed"] == "128"


def test_patch_preview_shows_the_exact_replacement_as_a_unified_diff() -> None:
    preview = _patch_preview(
        {
            "target_file": "tests/test_utils.py",
            "old_text": "    with pytest.raises(ValueError, match=r'platform detail'):\n",
            "new_text": "    with pytest.raises(ValueError):\n",
        }
    )

    assert "--- a/tests/test_utils.py" in preview
    assert "+++ b/tests/test_utils.py" in preview
    assert "-    with pytest.raises(ValueError, match=r'platform detail'):" in preview
    assert "+    with pytest.raises(ValueError):" in preview
