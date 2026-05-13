"""CocoIndex app for indexing codebases."""

from __future__ import annotations

import ast
import asyncio
import bisect
import codecs
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, cast

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from cocoindex.connectors.sqlite import Vec0TableDef
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk, TextPosition
from cocoindex.resources.file import FilePathMatcher, PatternFilePathMatcher
from cocoindex.resources.id import IdGenerator
from pathspec import GitIgnoreSpec

from .chunking import CHUNKER_REGISTRY, ChunkerFn
from .settings import load_gitignore_spec, load_project_settings
from .shared import (
    CODEBASE_DIR,
    EMBEDDER,
    INDEXING_EMBED_PARAMS,
    SQLITE_DB,
    CodeChunk,
    Embedder,
)

# Chunking configuration
CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 250
CHUNK_OVERLAP = 150
DEFAULT_CHUNK_EMBED_BATCH_SIZE = 64
DEFAULT_MAX_RECURSIVE_SPLIT_BYTES = 1_000_000
DEFAULT_MAX_FILE_READ_BYTES = 5_000_000
CHUNK_EMBED_BATCH_SIZE_ENV = "COCOINDEX_CODE_CHUNK_EMBED_BATCH_SIZE"
MAX_RECURSIVE_SPLIT_BYTES_ENV = "COCOINDEX_CODE_MAX_RECURSIVE_SPLIT_BYTES"
MAX_FILE_READ_BYTES_ENV = "COCOINDEX_CODE_MAX_FILE_READ_BYTES"

# Chunking splitter (stateless, can be module-level)
splitter = RecursiveSplitter()

_SYMBOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)\b", re.MULTILINE),
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\b", re.MULTILINE),
    re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
        re.MULTILINE,
    ),
    re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
)


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an integer env var, falling back to a validated default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _chunk_embed_batch_size() -> int:
    """Maximum number of chunk embeds to schedule at once."""
    return _int_env(CHUNK_EMBED_BATCH_SIZE_ENV, DEFAULT_CHUNK_EMBED_BATCH_SIZE, minimum=1)


def _max_recursive_split_bytes() -> int:
    """Maximum file size for syntax-aware RecursiveSplitter."""
    return _int_env(
        MAX_RECURSIVE_SPLIT_BYTES_ENV,
        DEFAULT_MAX_RECURSIVE_SPLIT_BYTES,
        minimum=1,
    )


def _max_file_read_bytes() -> int:
    """Maximum source file bytes to read into memory for indexing."""
    return _int_env(MAX_FILE_READ_BYTES_ENV, DEFAULT_MAX_FILE_READ_BYTES, minimum=1)


_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)


@dataclass(frozen=True)
class IndexableFile:
    """Lightweight file reference used for memoization without content fingerprinting."""

    rel_path: str
    abs_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _SymbolSpan:
    """A symbol declaration span used to attach enclosing context to chunks."""

    name: str
    start_line: int
    end_line: int


async def _indexable_file_from_local(file: localfs.File) -> IndexableFile | None:
    """Build a stat-based file reference without reading file content."""
    path = file.file_path.resolve()
    try:
        stat = await asyncio.to_thread(os.stat, path)
    except OSError:
        return None
    return IndexableFile(
        rel_path=file.file_path.path.as_posix(),
        abs_path=str(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _decode_file_bytes(data: bytes, encoding: str | None = None, errors: str = "replace") -> str:
    """Decode file bytes like FileLike.read_text without caching full content."""
    if encoding is not None:
        return data.decode(encoding, errors)
    for bom, enc in _BOM_ENCODINGS:
        if data.startswith(bom):
            return data.decode(enc, errors)
    return data.decode("utf-8", errors)


def _read_path_bytes(path: Path, size: int) -> bytes:
    with path.open("rb") as f:
        return f.read(size)


async def _read_file_text_bounded(file: IndexableFile) -> str | None:
    """Read file text with a hard byte cap; return None if too large or unavailable."""
    max_bytes = _max_file_read_bytes()
    if file.size > max_bytes:
        return None

    path = Path(file.abs_path)
    try:
        data = await asyncio.to_thread(_read_path_bytes, path, max_bytes + 1)
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    return _decode_file_bytes(data)


def _normalize_gitignore_lines(lines: Iterable[str], directory: PurePath) -> list[str]:
    """Normalize .gitignore lines to root-relative gitignore patterns."""
    if directory in (PurePath("."), PurePath("")):
        prefix = ""
    else:
        prefix = f"{directory.as_posix().rstrip('/')}/"

    normalized: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("\\#") or line.startswith("\\!"):
            line = line[1:]
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        body = line.strip()
        if not body:
            continue
        anchor = body.startswith("/")
        if anchor:
            body = body.lstrip("/")
            pattern = f"{prefix}{body}" if prefix else body
        else:
            contains_slash = "/" in body
            base = prefix
            if contains_slash:
                pattern = f"{base}{body}"
            else:
                if base:
                    pattern = f"{base}**/{body}"
                else:
                    pattern = f"**/{body}"
        if negated:
            pattern = f"!{pattern}"
        normalized.append(pattern)
    return normalized


class GitignoreAwareMatcher(FilePathMatcher):
    """Wraps another matcher and applies .gitignore filtering."""

    def __init__(
        self,
        delegate: FilePathMatcher,
        root_spec: GitIgnoreSpec | None,
        project_root: Path,
    ) -> None:
        self._delegate = delegate
        self._root = project_root
        self._spec_cache: dict[PurePath, GitIgnoreSpec | None] = {PurePath("."): root_spec}

    def _spec_for(self, directory: PurePath) -> GitIgnoreSpec | None:
        if directory in self._spec_cache:
            return self._spec_cache[directory]

        parent_dir = directory.parent if directory != PurePath(".") else PurePath(".")
        parent_spec = self._spec_for(parent_dir)
        spec = parent_spec

        gitignore_path = (self._root / directory) / ".gitignore"
        if gitignore_path.is_file():
            try:
                lines = gitignore_path.read_text().splitlines()
            except (OSError, UnicodeDecodeError):
                lines = []
            normalized = _normalize_gitignore_lines(lines, directory)
            if normalized:
                new_spec = GitIgnoreSpec.from_lines(normalized)
                spec = new_spec if spec is None else spec + new_spec

        self._spec_cache[directory] = spec
        return spec

    def _is_ignored(self, path: PurePath, is_dir: bool) -> bool:
        directory = path if is_dir else path.parent
        if directory == PurePath(""):
            directory = PurePath(".")
        spec = self._spec_for(directory)
        if spec is None:
            return False
        match_path = path.as_posix()
        if is_dir and not match_path.endswith("/"):
            match_path = f"{match_path}/"
        return spec.match_file(match_path)

    def is_dir_included(self, path: PurePath) -> bool:
        if self._is_ignored(path, True):
            return False
        return self._delegate.is_dir_included(path)

    def is_file_included(self, path: PurePath) -> bool:
        if self._is_ignored(path, False):
            return False
        for parent in path.parents:
            if parent in (PurePath("."), PurePath("")):
                break
            if not self.is_dir_included(parent):
                return False
        return self._delegate.is_file_included(path)


def _line_start_offsets(text: str) -> list[int]:
    """Return the character offsets for the start of each line."""
    offsets = [0]
    pos = text.find("\n")
    while pos != -1:
        offsets.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return offsets


def _position_at_offset(line_starts: list[int], offset: int) -> TextPosition:
    """Return a TextPosition for a character offset."""
    line_index = bisect.bisect_right(line_starts, offset) - 1
    line_start = line_starts[line_index]
    return TextPosition(
        byte_offset=offset,
        char_offset=offset,
        line=line_index + 1,
        column=(offset - line_start) + 1,
    )


def _fallback_split_large_text(
    text: str,
    *,
    chunk_size: int,
    min_chunk_size: int,
    chunk_overlap: int,
) -> Iterator[Chunk]:
    """Yield chunks from large files without materializing the entire split result."""
    if not text:
        return

    line_starts = _line_start_offsets(text)
    text_len = len(text)
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    min_size = min(min_chunk_size, chunk_size)
    start = 0

    while start < text_len:
        target_end = min(text_len, start + chunk_size)
        end = target_end
        if target_end < text_len:
            min_end = min(text_len, start + min_size)
            newline = text.rfind("\n", min_end, target_end)
            if newline != -1:
                end = newline + 1

        if end <= start:
            end = min(text_len, start + chunk_size)
        if end <= start:
            break

        start_pos = _position_at_offset(line_starts, start)
        end_pos = _position_at_offset(line_starts, max(start, end - 1))
        yield Chunk(text=text[start:end], start=start_pos, end=end_pos)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)


def _split_file_content(
    file_path: Path,
    content: str,
    *,
    language: str,
    chunker: ChunkerFn | None = None,
) -> tuple[str, Iterable[Chunk]]:
    """Return language and chunk iterable for a file."""
    if chunker is not None:
        language_override, chunks = chunker(file_path, content)
        return language_override or language, chunks

    if len(content) > _max_recursive_split_bytes():
        return language, _fallback_split_large_text(
            content,
            chunk_size=CHUNK_SIZE,
            min_chunk_size=MIN_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    return language, splitter.split(
        content,
        chunk_size=CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        language=language,
    )


def _iter_chunk_batches(chunks: Iterable[Chunk], batch_size: int) -> Iterator[list[Chunk]]:
    """Yield chunks in bounded batches to avoid unbounded task fan-out."""
    batch: list[Chunk] = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _symbol_names_for_chunk(text: str) -> list[str]:
    """Return likely symbol names declared in a chunk."""
    names: list[str] = []
    seen: set[str] = set()
    for pattern in _SYMBOL_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def _line_for_offset(line_starts: list[int], offset: int) -> int:
    return bisect.bisect_right(line_starts, offset)


def _python_symbol_spans(content: str) -> list[_SymbolSpan]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    spans: list[_SymbolSpan] = []

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        is_symbol = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        next_parents = parents
        if is_symbol:
            symbol_node = cast(ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, node)
            name = ".".join((*parents, symbol_node.name))
            end_line = getattr(symbol_node, "end_lineno", None) or symbol_node.lineno
            spans.append(
                _SymbolSpan(
                    name=name,
                    start_line=symbol_node.lineno,
                    end_line=end_line,
                )
            )
            next_parents = (*parents, symbol_node.name)

        for child in ast.iter_child_nodes(node):
            visit(child, next_parents)

    visit(tree, ())
    return spans


def _regex_symbol_spans(content: str) -> list[_SymbolSpan]:
    line_starts = _line_start_offsets(content)
    found: list[tuple[str, int]] = []
    seen_at_line: set[tuple[str, int]] = set()
    for pattern in _SYMBOL_PATTERNS:
        for match in pattern.finditer(content):
            name = match.group(1)
            line = _line_for_offset(line_starts, match.start())
            key = (name, line)
            if key in seen_at_line:
                continue
            seen_at_line.add(key)
            found.append((name, line))

    found.sort(key=lambda item: item[1])
    spans: list[_SymbolSpan] = []
    total_lines = max(1, content.count("\n") + 1)
    for index, (name, line) in enumerate(found):
        next_line = found[index + 1][1] if index + 1 < len(found) else total_lines + 1
        spans.append(_SymbolSpan(name=name, start_line=line, end_line=max(line, next_line - 1)))
    return spans


def _symbol_spans_for_file(content: str, language: str) -> list[_SymbolSpan]:
    if language == "python":
        spans = _python_symbol_spans(content)
        if spans:
            return spans
    return _regex_symbol_spans(content)


def _enclosing_symbol_names(chunk: Chunk, spans: list[_SymbolSpan]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    chunk_start = chunk.start.line
    chunk_end = chunk.end.line
    for span in spans:
        if span.end_line < chunk_start or span.start_line > chunk_end:
            continue
        if span.name in seen:
            continue
        seen.add(span.name)
        names.append(span.name)
    return names


def _file_role(file_path: str) -> str:
    """Classify a file so retrieval can distinguish docs/tests from implementation."""
    path = PurePath(file_path)
    parts = tuple(part.lower() for part in path.parts)
    basename = path.name.lower()
    suffix = path.suffix.lower()

    if basename.startswith("test_") or basename.endswith("_test.py") or "tests" in parts:
        return "test"
    if basename in {"readme.md", "readme.mdx", "changelog.md", "license"} or suffix in {
        ".md",
        ".mdx",
        ".rst",
        ".txt",
    }:
        return "docs"
    if basename in {
        "pyproject.toml",
        "package.json",
        "tsconfig.json",
        "dockerfile",
        "docker-compose.yml",
    } or suffix in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}:
        return "config"
    return "implementation"


def _join_metadata(values: list[str]) -> str:
    return ", ".join(values)


def _embedding_text_for_chunk(
    *,
    file_path: str,
    language: str,
    chunk: Chunk,
    symbols: list[str],
    enclosing_symbols: list[str],
    file_role: str,
    basename: str,
) -> str:
    """Build embedding input with code metadata while preserving raw stored content."""
    metadata = [
        f"path: {file_path}",
        f"basename: {basename}",
        f"role: {file_role}",
        f"language: {language}",
        f"lines: {chunk.start.line}-{chunk.end.line}",
    ]
    if symbols:
        metadata.append(f"symbols: {_join_metadata(symbols)}")
    if enclosing_symbols:
        metadata.append(f"context: {_join_metadata(enclosing_symbols)}")
    return "\n".join(["code:", chunk.text, "", "metadata:", *metadata])


async def _declare_chunk_batch(
    *,
    chunks: list[Chunk],
    file_path: str,
    language: str,
    symbol_spans: list[_SymbolSpan],
    file_role: str,
    basename: str,
    id_gen: IdGenerator,
    embedder: Embedder,
    indexing_params: dict[str, Any],
    table: sqlite.TableTarget[CodeChunk],
) -> None:
    """Embed and declare a bounded batch of chunks."""
    pending_chunks: list[tuple[int, Chunk, str]] = []
    for chunk in chunks:
        symbols = _symbol_names_for_chunk(chunk.text)
        enclosing_symbols = _enclosing_symbol_names(chunk, symbol_spans)
        embedding_text = _embedding_text_for_chunk(
            file_path=file_path,
            language=language,
            chunk=chunk,
            symbols=symbols,
            enclosing_symbols=enclosing_symbols,
            file_role=file_role,
            basename=basename,
        )
        pending_chunks.append((await id_gen.next_id(chunk.text), chunk, embedding_text))

    embeddings = await asyncio.gather(
        *(
            embedder.embed(embedding_text, **indexing_params)
            for _, _, embedding_text in pending_chunks
        )
    )

    for (chunk_id, chunk, _), embedding in zip(pending_chunks, embeddings, strict=True):
        symbols = _symbol_names_for_chunk(chunk.text)
        enclosing_symbols = _enclosing_symbol_names(chunk, symbol_spans)
        table.declare_row(
            row=CodeChunk(
                id=chunk_id,
                file_path=file_path,
                language=language,
                content=chunk.text,
                start_line=chunk.start.line,
                end_line=chunk.end.line,
                symbols=_join_metadata(symbols),
                enclosing_symbols=_join_metadata(enclosing_symbols),
                file_role=file_role,
                basename=basename,
                embedding=embedding,
            )
        )


@coco.fn(memo=True, version=3)
async def process_file(
    file: IndexableFile,
    table: sqlite.TableTarget[CodeChunk],
) -> None:
    """Process a single file: chunk, embed, and store."""
    embedder = coco.use_context(EMBEDDER)
    indexing_params = coco.use_context(INDEXING_EMBED_PARAMS)

    try:
        content = await _read_file_text_bounded(file)
    except UnicodeDecodeError:
        return
    if content is None:
        return

    if not content.strip():
        return

    rel_path = Path(file.rel_path)
    suffix = rel_path.suffix
    project_root = coco.use_context(CODEBASE_DIR)
    ps = load_project_settings(project_root)
    ext_lang_map = {f".{lo.ext}": lo.lang for lo in ps.language_overrides}
    language = (
        ext_lang_map.get(suffix)
        or detect_code_language(filename=rel_path.name)
        or "text"
    )

    chunker_registry = coco.use_context(CHUNKER_REGISTRY)
    chunker = chunker_registry.get(suffix)
    language, chunks = _split_file_content(
        rel_path,
        content,
        language=language,
        chunker=chunker,
    )
    symbol_spans = _symbol_spans_for_file(content, language)
    file_role = _file_role(file.rel_path)
    basename = rel_path.name

    id_gen = IdGenerator()
    for chunk_batch in _iter_chunk_batches(chunks, _chunk_embed_batch_size()):
        await _declare_chunk_batch(
            chunks=chunk_batch,
            file_path=file.rel_path,
            language=language,
            symbol_spans=symbol_spans,
            file_role=file_role,
            basename=basename,
            id_gen=id_gen,
            embedder=embedder,
            indexing_params=indexing_params,
            table=table,
        )


@coco.fn
async def indexer_main() -> None:
    """Main indexing function - walks files and processes each."""
    project_root = coco.use_context(CODEBASE_DIR)
    ps = load_project_settings(project_root)
    gitignore_spec = load_gitignore_spec(project_root)

    table = await sqlite.mount_table_target(
        db=SQLITE_DB,
        table_name="code_chunks_vec",
        table_schema=await sqlite.TableSchema.from_class(
            CodeChunk,
            primary_key=["id"],
        ),
        virtual_table_def=Vec0TableDef(
            partition_key_columns=["language"],
            auxiliary_columns=[
                "file_path",
                "content",
                "start_line",
                "end_line",
                "symbols",
                "enclosing_symbols",
                "file_role",
                "basename",
            ],
        ),
    )

    base_matcher = PatternFilePathMatcher(
        included_patterns=ps.include_patterns,
        excluded_patterns=ps.exclude_patterns,
    )
    matcher: FilePathMatcher = GitignoreAwareMatcher(base_matcher, gitignore_spec, project_root)

    files = localfs.walk_dir(
        CODEBASE_DIR,
        recursive=True,
        path_matcher=matcher,
    )

    with coco.component_subpath(coco.Symbol("process_file")):
        async for key, file in files.items():
            file_ref = await _indexable_file_from_local(file)
            if file_ref is None:
                continue
            await coco.use_mount(coco.component_subpath(key), process_file, file_ref, table)
