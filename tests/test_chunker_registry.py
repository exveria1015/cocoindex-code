"""Tests for the pluggable chunker registry.

Uses Project.create() directly with a mock embedder so no real embedding model
is needed.  Each test writes files to a temp directory, indexes them, and
queries the resulting SQLite database to verify chunk content and language.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np
import pytest
from cocoindex.connectors import localfs as coco_localfs
from cocoindex.connectors import sqlite as coco_sqlite
from cocoindex.resources.schema import VectorSchema
from example_toml_chunker import toml_chunker

import cocoindex_code.indexer as _indexer
from cocoindex_code.chunking import CHUNKER_REGISTRY, Chunk, TextPosition
from cocoindex_code.daemon import ProjectRegistry
from cocoindex_code.project import Project
from cocoindex_code.settings import ProjectSettings, save_project_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 4  # tiny dimension — enough to satisfy the vector table schema


class _StubEmbedder:
    """Minimal embedder stub satisfying cocoindex memo-key and vector-schema requirements."""

    def __coco_memo_key__(self) -> str:
        return "stub-embedder"

    async def __coco_vector_schema__(self) -> VectorSchema:
        return VectorSchema(dtype=np.dtype("float32"), size=_EMBED_DIM)

    async def embed(self, text: str, **kwargs: Any) -> np.ndarray:
        return np.zeros(_EMBED_DIM, dtype=np.float32)


class _RecordingEmbedder(_StubEmbedder):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __coco_memo_key__(self) -> str:
        return "recording-embedder"

    async def embed(self, text: str, **kwargs: Any) -> np.ndarray:
        self.texts.append(text)
        return np.zeros(_EMBED_DIM, dtype=np.float32)


async def _index_project(
    project_root: Path,
    embedder: Any | None = None,
    **create_kwargs: Any,
) -> Project:
    """Create a Project and run a full index pass."""
    settings = ProjectSettings(include_patterns=["**/*.*"], exclude_patterns=["**/.cocoindex_code"])
    stub = embedder or _StubEmbedder()
    save_project_settings(project_root, settings)
    project = await Project.create(
        project_root,
        stub,
        indexing_params={},
        query_params={},
        **create_kwargs,
    )
    await project.run_index()
    return project


def _query_chunks(project_root: Path) -> list[dict[str, Any]]:
    """Read all stored chunks from the target SQLite database."""
    db_path = project_root / ".cocoindex_code" / "target_sqlite.db"
    conn = coco_sqlite.connect(str(db_path), load_vec=True)
    try:
        with conn.readonly() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT file_path, language, content, start_line, end_line FROM code_chunks_vec"
            ).fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()


def _pos(line: int) -> TextPosition:
    """TextPosition with only line number set; suitable for line-granularity chunkers."""
    return TextPosition(byte_offset=0, char_offset=0, line=line, column=0)


# ---------------------------------------------------------------------------
# TOML fixture content
# ---------------------------------------------------------------------------

_TOML_CONTENT = """\
[section_one]
key = "value"
answer = 42

[section_two]
other = "hello"
flag = true
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_default_registry_is_empty(tmp_path: Path) -> None:
    """CHUNKER_REGISTRY is an empty dict when no registry is passed."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "hello.py").write_text("x = 1\n")

    project = await _index_project(tmp_path)
    registry = project.env.get_context(CHUNKER_REGISTRY)
    assert isinstance(registry, dict)
    assert registry == {}


async def test_unregistered_suffix_uses_splitter(tmp_path: Path) -> None:
    """Files with no registered chunker are processed by RecursiveSplitter."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "sample.py").write_text("def foo():\n    return 1\n")

    await _index_project(tmp_path)
    chunks = _query_chunks(tmp_path)

    assert len(chunks) >= 1
    assert all(c["language"] == "python" for c in chunks)
    assert any("foo" in c["content"] for c in chunks)


async def test_registered_chunker_is_called(tmp_path: Path) -> None:
    """A registered ChunkerFn splits files and may override the language."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "config.toml").write_text(_TOML_CONTENT)

    await _index_project(tmp_path, chunker_registry={".toml": toml_chunker})
    chunks = _query_chunks(tmp_path)

    assert len(chunks) == 2
    contents = {c["content"] for c in chunks}
    assert any("section_one" in c for c in contents)
    assert any("section_two" in c for c in contents)
    assert all(c["language"] == "toml" for c in chunks)


async def test_chunker_language_none_preserves_detected(tmp_path: Path) -> None:
    """When ChunkerFn returns language=None, detect_code_language() is used."""

    def _passthrough_chunker(path: Path, content: str) -> tuple[str | None, list[Chunk]]:
        lines = content.splitlines()
        return None, [Chunk(text=content, start=_pos(1), end=_pos(len(lines)))]

    (tmp_path / ".git").mkdir()
    (tmp_path / "script.py").write_text("x = 1\n")

    await _index_project(tmp_path, chunker_registry={".py": _passthrough_chunker})
    chunks = _query_chunks(tmp_path)

    assert all(c["language"] == "python" for c in chunks)


async def test_registry_does_not_affect_other_suffixes(tmp_path: Path) -> None:
    """Registering a chunker for .toml does not affect .py files."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "config.toml").write_text(_TOML_CONTENT)
    (tmp_path / "code.py").write_text("def bar():\n    pass\n")

    await _index_project(tmp_path, chunker_registry={".toml": toml_chunker})
    chunks = _query_chunks(tmp_path)

    toml_chunks = [c for c in chunks if c["language"] == "toml"]
    py_chunks = [c for c in chunks if c["language"] == "python"]

    assert len(toml_chunks) == 2
    assert len(py_chunks) >= 1
    assert any("bar" in c["content"] for c in py_chunks)


async def test_indexer_does_not_use_mount_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Indexing should no longer rely on mount_each fan-out."""

    def _fail_mount_each(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("mount_each should not be used")

    monkeypatch.setattr(coco, "mount_each", _fail_mount_each)
    (tmp_path / ".git").mkdir()
    (tmp_path / "sample.py").write_text("def foo():\n    return 1\n")

    await _index_project(tmp_path)
    chunks = _query_chunks(tmp_path)

    assert len(chunks) >= 1
    assert any("foo" in c["content"] for c in chunks)


async def test_large_files_use_streaming_chunker_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large files should bypass RecursiveSplitter to avoid list materialization."""

    def _fail_split(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("RecursiveSplitter should not be used for large files")

    monkeypatch.setenv(_indexer.MAX_RECURSIVE_SPLIT_BYTES_ENV, "1")
    monkeypatch.setattr(_indexer.splitter, "split", _fail_split)

    (tmp_path / ".git").mkdir()
    large_python = "".join(f"def func_{i}():\n    return {i}\n\n" for i in range(256))
    (tmp_path / "big.py").write_text(large_python)

    await _index_project(tmp_path)
    chunks = _query_chunks(tmp_path)

    assert len(chunks) > 1
    assert all(c["language"] == "python" for c in chunks)
    assert any("func_0" in c["content"] for c in chunks)


async def test_file_reads_are_bounded_and_do_not_use_read_text_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Indexing should skip oversized files and avoid FileLike full-read caching."""

    async def _fail_read_text(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("read_text should not be used")

    original_read_impl = coco_localfs.File._read_impl

    async def _bounded_read_impl(self: coco_localfs.File, size: int = -1) -> bytes:
        assert 0 <= size <= 65, f"unexpected unbounded read size: {size}"
        return await original_read_impl(self, size)

    monkeypatch.setenv(_indexer.MAX_FILE_READ_BYTES_ENV, "64")
    monkeypatch.setattr(coco_localfs.File, "read_text", _fail_read_text)
    monkeypatch.setattr(coco_localfs.File, "_read_impl", _bounded_read_impl)

    (tmp_path / ".git").mkdir()
    (tmp_path / "small.py").write_text("def small():\n    return 1\n")
    (tmp_path / "large.py").write_text("def large():\n" + ("#" * 256) + "\n")

    await _index_project(tmp_path)
    chunks = _query_chunks(tmp_path)

    assert any("small" in c["content"] for c in chunks)
    assert not any("large" in c["content"] for c in chunks)


async def test_embedding_text_includes_metadata_but_stores_raw_content(tmp_path: Path) -> None:
    """Embedding input should include metadata while the result content stays raw."""
    (tmp_path / ".git").mkdir()
    source = "class SearchService:\n    def find_user(self):\n        return 1\n"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "search.py").write_text(source)

    embedder = _RecordingEmbedder()
    await _index_project(tmp_path, embedder=embedder)
    chunks = _query_chunks(tmp_path)

    assert chunks
    assert any("SearchService" in c["content"] for c in chunks)
    assert all("path: src/search.py" not in c["content"] for c in chunks)
    indexed_text = "\n".join(embedder.texts)
    assert "path: src/search.py" in indexed_text
    assert "basename: search.py" in indexed_text
    assert "role: implementation" in indexed_text
    assert "language: python" in indexed_text
    assert (
        "symbols: find_user, SearchService" in indexed_text
        or "symbols: SearchService, find_user" in indexed_text
    )
    assert (
        "context: SearchService, SearchService.find_user" in indexed_text
        or "context: SearchService.find_user, SearchService" in indexed_text
    )
    assert "code:\nclass SearchService" in indexed_text
    assert indexed_text.index("code:\nclass SearchService") < indexed_text.index("metadata:")


async def test_hybrid_search_uses_lexical_identifier_matches(tmp_path: Path) -> None:
    """Exact identifier matches should rank highly even when vectors are indistinguishable."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "alpha.py").write_text("def common_alpha():\n    return 'alpha'\n")
    (tmp_path / "needle.py").write_text("def specialIdentifier():\n    return 'needle'\n")

    project = await _index_project(tmp_path)
    results = await project.search("specialIdentifier", limit=2)

    assert results
    assert results[0].file_path == "needle.py"
    assert "specialIdentifier" in results[0].content


async def test_hybrid_search_prioritizes_exact_identifier_over_metadata_words(
    tmp_path: Path,
) -> None:
    """Metadata label words should not outrank an exact symbol match."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "indexer.py").write_text(
        "def _embedding_text_for_chunk():\n    return 'metadata'\n"
    )
    (tmp_path / "tests" / "test_indexer.py").write_text(
        "def test_metadata_labels():\n"
        "    assert 'path: src/search.py'\n"
        "    assert 'language: python'\n"
        "    assert 'symbols: SearchService'\n"
    )

    project = await _index_project(tmp_path)
    results = await project.search("_embedding_text_for_chunk path language symbols", limit=3)

    assert results
    assert results[0].file_path == "src/indexer.py"


async def test_hybrid_search_uses_identifier_subtokens_without_generic_drift(
    tmp_path: Path,
) -> None:
    """Identifier-like prose should beat generic settings/env matches."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "settings.py").write_text(
        "def save_environment_settings_variables():\n"
        "    return 'settings environment variables settings'\n"
    )
    (tmp_path / "tests" / "test_backward_compat.py").write_text(
        "def test_legacy_entry_creates_settings_from_env_vars():\n"
        "    assert True\n"
    )

    project = await _index_project(tmp_path)
    results = await project.search(
        "legacy entry creates settings from environment variables",
        limit=3,
    )

    assert results
    assert results[0].file_path == "tests/test_backward_compat.py"


async def test_hybrid_search_prefers_enclosing_implementation_over_docs_and_tests(
    tmp_path: Path,
) -> None:
    """Implementation-intent queries should use symbol context to beat docs/tests."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "src" / "adapters").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text(
        "Adapter notes: request options plugins plugin_selection "
        "concurrent_plugin_calls are supported.\n"
    )
    (tmp_path / "tests" / "test_native_adapter.py").write_text(
        "def test_options_are_forwarded():\n"
        "    assert options['plugins'] == request.plugins\n"
        "    assert options['plugin_selection'] == 'required'\n"
        "    assert options['concurrent_plugin_calls'] is True\n"
    )
    (tmp_path / "src" / "adapters" / "native.py").write_text(
        "class NativeAdapter:\n"
        "    def build_request_options(self, request, stream):\n"
        "        options = {'stream': stream}\n"
        "        if request.plugins:\n"
        "            options['plugins'] = request.plugins\n"
        "        if request.plugin_selection is not None:\n"
        "            options['plugin_selection'] = request.plugin_selection\n"
        "        if request.concurrent_plugin_calls:\n"
        "            options['concurrent_plugin_calls'] = request.concurrent_plugin_calls\n"
        "        return options\n"
    )

    project = await _index_project(tmp_path)
    results = await project.search(
        "where are adapter request options constructed for plugins "
        "plugin_selection concurrent_plugin_calls",
        limit=5,
    )

    assert results
    assert results[0].file_path == "src/adapters/native.py"
    assert "build_request_options" in results[0].content


async def test_project_aclose_cancels_background_index_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a project should not leave load-time indexing tasks alive."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "sample.py").write_text("def foo():\n    return 1\n")
    project = await _index_project(tmp_path)

    started = asyncio.Event()
    cancelled = asyncio.Event()
    never_finish = asyncio.Event()

    async def _never_finish(
        on_progress: Any | None = None,
    ) -> None:
        started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    project._initial_index_done.clear()
    monkeypatch.setattr(project, "_run_index_inner", _never_finish)

    await project.ensure_indexing_started()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert project._index_tasks

    await project.aclose()

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert not project._index_tasks


async def test_project_registry_evicts_idle_projects_over_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon registry should not retain an unbounded set of idle projects."""
    monkeypatch.setenv("COCOINDEX_CODE_MAX_LOADED_PROJECTS", "1")
    project_roots = [tmp_path / "one", tmp_path / "two"]
    for root in project_roots:
        root.mkdir()
        save_project_settings(root, ProjectSettings(include_patterns=["**/*.py"]))

    registry = ProjectRegistry(_StubEmbedder())
    first = await registry.get_project(str(project_roots[0]))
    second = await registry.get_project(str(project_roots[1]))

    listed = registry.list_projects()
    assert [p.project_root for p in listed] == [str(project_roots[1])]
    assert first._closed
    assert not second._closed

    await registry.close_all()
