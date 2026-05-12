"""Tests for the fixed-query search quality runner."""

from __future__ import annotations

from types import SimpleNamespace

from cocoindex_code.search_quality import QueryCase, evaluate_case, format_markdown


def test_evaluate_case_uses_exact_terms_when_present() -> None:
    case = QueryCase(
        id="Q7",
        query="where kwargs are constructed",
        expected_file="sample_project/src/adapters/native.py",
        expected_rank_max=2,
        exact_terms=("plugins", "plugin_selection", "concurrent_plugin_calls"),
    )
    results = [
        SimpleNamespace(
            file_path="sample_project/src/adapters/native.py",
            content="def build_request_options(): pass",
        ),
        SimpleNamespace(
            file_path="sample_project/src/adapters/native.py",
            content=(
                "options['plugins'] options['plugin_selection'] "
                "options['concurrent_plugin_calls']"
            ),
        ),
    ]

    evaluation = evaluate_case(case, results)

    assert evaluation.expected_rank == 1
    assert evaluation.exact_rank == 2
    assert evaluation.passed is True


def test_format_markdown_includes_pass_count() -> None:
    case = QueryCase(id="Q1", query="auth", expected_file="sample_project/src/auth.py")
    evaluation = evaluate_case(
        case,
        [SimpleNamespace(file_path="sample_project/src/auth.py", content="middleware")],
    )

    report = format_markdown([evaluation])

    assert "Passed: `1/1`" in report
    assert "`sample_project/src/auth.py`" in report
