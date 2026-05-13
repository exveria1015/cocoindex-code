"""Tests for project file path matching."""

from __future__ import annotations

from pathlib import Path, PurePath

from cocoindex.resources.file import PatternFilePathMatcher

from cocoindex_code.indexer import GitignoreAwareMatcher


def test_directory_exclude_pattern_excludes_descendant_files(tmp_path: Path) -> None:
    delegate = PatternFilePathMatcher(["**/*.py"], ["**/build"])
    matcher = GitignoreAwareMatcher(delegate, root_spec=None, project_root=tmp_path)

    assert not matcher.is_dir_included(PurePath("build"))
    assert not matcher.is_file_included(PurePath("build/generated.py"))
    assert not matcher.is_dir_included(PurePath("src/build"))
    assert not matcher.is_file_included(PurePath("src/build/generated.py"))
    assert matcher.is_file_included(PurePath("src/app.py"))
