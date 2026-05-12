"""Fixed-query search quality runner.

This is intentionally small and file-based so model/index changes can be
compared without encoding expectations in ad hoc shell history.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from . import client


@dataclass(frozen=True)
class QueryCase:
    id: str
    query: str
    expected_file: str
    expected_rank_max: int | None = None
    exact_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryEvaluation:
    id: str
    query: str
    expected_file: str
    expected_rank_max: int | None
    expected_rank: int | None
    exact_rank: int | None
    passed: bool
    top_files: tuple[str, ...]


def load_query_cases(path: Path) -> list[QueryCase]:
    data = yaml.safe_load(path.read_text()) or {}
    raw_cases = data.get("queries")
    if not isinstance(raw_cases, list):
        raise ValueError(f"{path}: expected a top-level 'queries' list")

    cases: list[QueryCase] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: each query case must be a mapping")
        cases.append(
            QueryCase(
                id=str(raw["id"]),
                query=str(raw["query"]),
                expected_file=str(raw["expected_file"]),
                expected_rank_max=(
                    int(raw["expected_rank_max"])
                    if raw.get("expected_rank_max") is not None
                    else None
                ),
                exact_terms=tuple(str(term) for term in raw.get("exact_terms", ())),
            )
        )
    return cases


def evaluate_case(case: QueryCase, results: list[Any]) -> QueryEvaluation:
    expected_rank: int | None = None
    exact_rank: int | None = None
    exact_terms = tuple(term.lower() for term in case.exact_terms)

    for index, result in enumerate(results, start=1):
        file_path = str(result.file_path)
        if file_path == case.expected_file and expected_rank is None:
            expected_rank = index
        if file_path == case.expected_file and exact_rank is None and exact_terms:
            content = str(result.content).lower()
            if all(term in content for term in exact_terms):
                exact_rank = index

    threshold = case.expected_rank_max
    rank_to_check = exact_rank if exact_terms else expected_rank
    passed = rank_to_check is not None and (threshold is None or rank_to_check <= threshold)

    return QueryEvaluation(
        id=case.id,
        query=case.query,
        expected_file=case.expected_file,
        expected_rank_max=case.expected_rank_max,
        expected_rank=expected_rank,
        exact_rank=exact_rank,
        passed=passed,
        top_files=tuple(str(result.file_path) for result in results[:5]),
    )


def run_benchmark(
    *,
    project_root: Path,
    cases_path: Path,
    limit: int,
    refresh: bool = False,
) -> list[QueryEvaluation]:
    cases = load_query_cases(cases_path)
    if refresh:
        index_response = client.index(str(project_root))
        if not index_response.success:
            raise RuntimeError(index_response.message)

    evaluations: list[QueryEvaluation] = []
    for case in cases:
        search_response = client.search(str(project_root), case.query, limit=limit)
        if not search_response.success:
            raise RuntimeError(search_response.message)
        evaluations.append(evaluate_case(case, list(search_response.results)))
    return evaluations


def format_markdown(evaluations: list[QueryEvaluation]) -> str:
    passed = sum(1 for evaluation in evaluations if evaluation.passed)
    lines = [
        "# ccc Search Quality Benchmark",
        "",
        f"Passed: `{passed}/{len(evaluations)}`",
        "",
        "| ID | Expected file | Expected rank | Exact rank | Threshold | Pass | Top files |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for evaluation in evaluations:
        top_files = "<br>".join(evaluation.top_files)
        lines.append(
            "| "
            f"{evaluation.id} | "
            f"`{evaluation.expected_file}` | "
            f"{evaluation.expected_rank or '-'} | "
            f"{evaluation.exact_rank or '-'} | "
            f"{evaluation.expected_rank_max or '-'} | "
            f"{'yes' if evaluation.passed else 'no'} | "
            f"{top_files} |"
        )
    lines.append("")
    return "\n".join(lines)


def _json_report(evaluations: list[QueryEvaluation]) -> str:
    return json.dumps([asdict(evaluation) for evaluation in evaluations], indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed-query ccc search quality checks.")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    evaluations = run_benchmark(
        project_root=args.project_root,
        cases_path=args.cases,
        limit=args.limit,
        refresh=args.refresh,
    )
    output = _json_report(evaluations) if args.format == "json" else format_markdown(evaluations)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0 if all(evaluation.passed for evaluation in evaluations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
