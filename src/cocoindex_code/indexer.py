"""CocoIndex app for indexing codebases."""

from __future__ import annotations

import asyncio
import bisect
import os
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePath

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
CHUNK_EMBED_BATCH_SIZE_ENV = "COCOINDEX_CODE_CHUNK_EMBED_BATCH_SIZE"
MAX_RECURSIVE_SPLIT_BYTES_ENV = "COCOINDEX_CODE_MAX_RECURSIVE_SPLIT_BYTES"

# Chunking splitter (stateless, can be module-level)
splitter = RecursiveSplitter()


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


async def _declare_chunk_batch(
    *,
    chunks: list[Chunk],
    file_path: str,
    language: str,
    id_gen: IdGenerator,
    embedder: Embedder,
    table: sqlite.TableTarget[CodeChunk],
) -> None:
    """Embed and declare a bounded batch of chunks."""
    pending_chunks: list[tuple[int, Chunk]] = []
    for chunk in chunks:
        pending_chunks.append((await id_gen.next_id(chunk.text), chunk))

    embeddings = await asyncio.gather(*(embedder.embed(chunk.text) for _, chunk in pending_chunks))

    for (chunk_id, chunk), embedding in zip(pending_chunks, embeddings, strict=True):
        table.declare_row(
            row=CodeChunk(
                id=chunk_id,
                file_path=file_path,
                language=language,
                content=chunk.text,
                start_line=chunk.start.line,
                end_line=chunk.end.line,
                embedding=embedding,
            )
        )


@coco.fn(memo=True)
async def process_file(
    file: localfs.File,
    table: sqlite.TableTarget[CodeChunk],
) -> None:
    """Process a single file: chunk, embed, and store."""
    embedder = coco.use_context(EMBEDDER)

    try:
        content = await file.read_text()
    except UnicodeDecodeError:
        return

    if not content.strip():
        return

    suffix = file.file_path.path.suffix
    project_root = coco.use_context(CODEBASE_DIR)
    ps = load_project_settings(project_root)
    ext_lang_map = {f".{lo.ext}": lo.lang for lo in ps.language_overrides}
    language = (
        ext_lang_map.get(suffix)
        or detect_code_language(filename=file.file_path.path.name)
        or "text"
    )

    chunker_registry = coco.use_context(CHUNKER_REGISTRY)
    chunker = chunker_registry.get(suffix)
    language, chunks = _split_file_content(
        Path(file.file_path.path),
        content,
        language=language,
        chunker=chunker,
    )

    id_gen = IdGenerator()
    file_path = file.file_path.path.as_posix()
    for chunk_batch in _iter_chunk_batches(chunks, _chunk_embed_batch_size()):
        await _declare_chunk_batch(
            chunks=chunk_batch,
            file_path=file_path,
            language=language,
            id_gen=id_gen,
            embedder=embedder,
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
            auxiliary_columns=["file_path", "content", "start_line", "end_line"],
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
            await coco.use_mount(coco.component_subpath(key), process_file, file, table)
