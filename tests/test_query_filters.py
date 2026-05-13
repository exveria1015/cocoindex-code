"""Tests for query-side metadata and path filters."""

from __future__ import annotations

import sqlite3

from cocoindex_code.query import (
    _expand_path_patterns,
    _extract_language_selectors,
    _filter_conditions,
    _fuse_results,
    _lexical_query,
    _query_terms,
)
from cocoindex_code.settings import SearchFilterSettings, SearchRankingSettings


def test_plain_path_filter_expands_as_directory() -> None:
    assert _expand_path_patterns(["src"]) == ["src", "src/*"]
    assert _expand_path_patterns(["./src/"]) == ["src", "src/*"]


def test_globstar_path_filter_includes_root_level_variant() -> None:
    assert _expand_path_patterns(["**/*.py"]) == ["**/*.py", "*.py"]
    assert _expand_path_patterns(["**/src/**"]) == ["**/src/**", "src/**"]


def test_dot_path_filter_means_project_scope() -> None:
    assert _expand_path_patterns(["."]) is None


def test_language_filter_normalizes_aliases() -> None:
    params: list[object] = []

    conditions = _filter_conditions(languages=["kt", "TS,md"], paths=None, params=params)

    assert conditions == ["language IN (?,?,?)"]
    assert params == ["kotlin", "typescript", "markdown"]


def test_language_selector_is_extracted_from_query() -> None:
    query, languages = _extract_language_selectors("lang:kotlin coroutine language:ts,md parser")

    assert query == "coroutine parser"
    assert languages == ["kotlin", "ts", "md"]


def test_language_terms_match_language_metadata() -> None:
    terms = {term.value: term for term in _query_terms("kotlin coroutine")}

    assert terms["kotlin"].language_weight > 0


def test_lexical_query_matches_language_metadata() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE code_chunks_vec (
            id INTEGER,
            file_path TEXT,
            language TEXT,
            content TEXT,
            start_line INTEGER,
            end_line INTEGER,
            symbols TEXT,
            enclosing_symbols TEXT,
            file_role TEXT,
            basename TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO code_chunks_vec
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "src/Main.kt",
            "kotlin",
            "fun launchScope() = Unit",
            1,
            1,
            "",
            "",
            "implementation",
            "Main.kt",
        ),
    )

    rows = _lexical_query(conn, "kotlin", limit=10)

    assert rows[0][1] == "src/Main.kt"


def test_filter_conditions_apply_search_excludes() -> None:
    params: list[object] = []

    conditions = _filter_conditions(
        languages=None,
        paths=None,
        search_exclude=SearchFilterSettings(
            languages=["kt"],
            paths=["legacy"],
            keywords=["DO NOT USE"],
        ),
        params=params,
    )

    assert conditions[0] == "language NOT IN (?)"
    assert conditions[1] == "NOT (file_path GLOB ? OR file_path GLOB ?)"
    assert conditions[2].startswith("NOT (")
    assert params[:3] == ["kotlin", "legacy", "legacy/*"]


def test_fuse_results_demotes_search_matches() -> None:
    rows = [
        (
            1,
            "legacy/old.py",
            "python",
            "def duplicated_impl(): pass",
            1,
            1,
            "",
            "",
            "implementation",
            "old.py",
            0.01,
        ),
        (
            2,
            "src/new.py",
            "python",
            "def duplicated_impl(): pass",
            1,
            1,
            "",
            "",
            "implementation",
            "new.py",
            0.02,
        ),
    ]

    results = _fuse_results(
        semantic_rows=rows,
        lexical_rows=[],
        query="duplicated_impl",
        limit=2,
        offset=0,
        search_ranking=SearchRankingSettings(
            demote=SearchFilterSettings(paths=["legacy"]),
            demote_score_multiplier=0.1,
        ),
    )

    assert [result.file_path for result in results] == ["src/new.py", "legacy/old.py"]
