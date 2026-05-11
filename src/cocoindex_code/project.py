"""Project management: wraps a CocoIndex Environment + App."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectors import sqlite as coco_sqlite

from .chunking import CHUNKER_REGISTRY, ChunkerFn
from .indexer import indexer_main
from .protocol import (
    IndexingProgress,
    IndexProgressUpdate,
    IndexResponse,
    IndexStreamResponse,
    IndexWaitingNotice,
    ProjectStatusResponse,
    SearchResult,
)
from .query import query_codebase
from .settings import (
    cocoindex_db_path as _cocoindex_db_path,
)
from .settings import (
    resolve_db_dir,
)
from .settings import (
    target_sqlite_db_path as _target_sqlite_db_path,
)
from .shared import (
    CODEBASE_DIR,
    EMBEDDER,
    INDEXING_EMBED_PARAMS,
    QUERY_EMBED_PARAMS,
    SQLITE_DB,
    Embedder,
)


class Project:
    _env: coco.Environment
    _app: coco.App[[], None]
    _project_root: Path
    _index_lock: asyncio.Lock
    _initial_index_done: asyncio.Event
    _index_tasks: set[asyncio.Task[None]]
    _initial_index_task: asyncio.Task[None] | None
    _initial_index_started: asyncio.Event | None
    _closed: bool
    _indexing_stats: IndexingProgress | None = None

    def close(self) -> None:
        """Synchronously close project resources.

        Prefer :meth:`aclose` from async daemon paths so background indexing
        tasks can be cancelled and awaited before file handles are released.
        This fallback is intentionally best-effort for synchronous callers.
        """
        self._closed = True
        for task in list(self._index_tasks):
            if not task.done():
                task.cancel()
        self._close_db()

    async def aclose(self) -> None:
        """Cancel background indexing and close project resources."""
        self._closed = True
        tasks = [task for task in self._index_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._index_tasks.clear()
        self._initial_index_task = None
        self._initial_index_started = None
        self._close_db()

    def _close_db(self) -> None:
        """Close the target SQLite connection if it has been created."""
        try:
            db = self._env.get_context(SQLITE_DB)
            db.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def run_index(
        self,
        on_progress: Callable[[IndexingProgress], None] | None = None,
        on_started: asyncio.Event | None = None,
    ) -> None:
        """Acquire the index lock, run indexing, and release.

        If *on_started* is provided, it is set once the lock is acquired
        (i.e. indexing has truly begun).  On completion (success or failure)
        ``_initial_index_done`` is set.
        """
        if self._closed:
            raise RuntimeError("Project is closed")

        async with self._index_lock:
            self._indexing_stats = IndexingProgress(
                num_execution_starts=0,
                num_unchanged=0,
                num_adds=0,
                num_deletes=0,
                num_reprocesses=0,
                num_errors=0,
            )
            if on_started is not None:
                on_started.set()
            await self._run_index_inner(on_progress=on_progress)

    async def _run_index_inner(
        self,
        on_progress: Callable[[IndexingProgress], None] | None = None,
    ) -> None:
        """Run indexing (lock must already be held)."""
        try:
            handle = self._app.update()
            async for snapshot in handle.watch():
                file_stats = snapshot.stats.by_component.get("process_file")
                if file_stats is not None:
                    progress = IndexingProgress(
                        num_execution_starts=file_stats.num_execution_starts,
                        num_unchanged=file_stats.num_unchanged,
                        num_adds=file_stats.num_adds,
                        num_deletes=file_stats.num_deletes,
                        num_reprocesses=file_stats.num_reprocesses,
                        num_errors=file_stats.num_errors,
                    )
                    self._indexing_stats = progress
                    if on_progress is not None:
                        on_progress(progress)
                    await asyncio.sleep(0.1)
        finally:
            self._initial_index_done.set()
            self._indexing_stats = None

    def _create_index_task(
        self,
        *,
        on_progress: Callable[[IndexingProgress], None] | None = None,
        on_started: asyncio.Event | None = None,
    ) -> asyncio.Task[None]:
        """Create and track a background indexing task."""
        task = asyncio.create_task(self.run_index(on_progress=on_progress, on_started=on_started))
        self._index_tasks.add(task)

        def _done(done: asyncio.Task[None]) -> None:
            self._index_tasks.discard(done)
            if on_started is not None and not on_started.is_set():
                on_started.set()

        task.add_done_callback(_done)
        return task

    async def ensure_indexing_started(self) -> None:
        """Kick off background indexing and wait until it has actually started.

        Returns once the indexing task holds the lock.  Safe to call multiple
        times — only the first call spawns a task; subsequent calls return
        immediately.
        """
        if self._closed:
            raise RuntimeError("Project is closed")
        if self._initial_index_done.is_set() or self._index_lock.locked():
            return
        task = self._initial_index_task
        started = self._initial_index_started
        if task is None or task.done():
            started = asyncio.Event()
            task = self._create_index_task(on_started=started)
            self._initial_index_task = task
            self._initial_index_started = started

            def _clear_initial(done: asyncio.Task[None]) -> None:
                if self._initial_index_task is done:
                    self._initial_index_task = None
                if self._initial_index_started is started:
                    self._initial_index_started = None

            task.add_done_callback(_clear_initial)

        if started is None:
            return
        await started.wait()

    async def stream_index(self) -> AsyncIterator[IndexStreamResponse]:
        """Run indexing, streaming progress updates and a final IndexResponse.

        If the lock is already held, yields ``IndexWaitingNotice`` first.
        The actual indexing runs in a separate task so that client disconnects
        (``GeneratorExit``) do not abort the indexing.
        """
        if self._closed:
            yield IndexResponse(success=False, message="Project is closed")
            return

        if self._index_lock.locked():
            yield IndexWaitingNotice()

        progress_queue: asyncio.Queue[IndexingProgress] = asyncio.Queue(maxsize=128)
        stream_open = True

        def _on_progress(progress: IndexingProgress) -> None:
            if not stream_open:
                return
            try:
                progress_queue.put_nowait(progress)
            except asyncio.QueueFull:
                pass

        index_task = self._create_index_task(on_progress=_on_progress)

        try:
            while not index_task.done():
                try:
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    yield IndexProgressUpdate(progress=progress)
                except TimeoutError:
                    continue

            while not progress_queue.empty():
                yield IndexProgressUpdate(progress=progress_queue.get_nowait())

            index_task.result()
            yield IndexResponse(success=True)
        except GeneratorExit:
            return
        except Exception as e:
            yield IndexResponse(success=False, message=str(e))
        finally:
            stream_open = False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @property
    def should_wait_for_indexing(self) -> bool:
        """True if indexing has been started but not yet completed."""
        return not self._initial_index_done.is_set()

    async def wait_for_indexing_done(self) -> None:
        """Wait until initial indexing is complete and no indexing is running."""
        await self._initial_index_done.wait()
        if self._index_lock.locked():
            async with self._index_lock:
                pass

    async def search(
        self,
        query: str,
        languages: list[str] | None = None,
        paths: list[str] | None = None,
        limit: int = 5,
        offset: int = 0,
    ) -> list[SearchResult]:
        """Search within this project."""
        target_db = _target_sqlite_db_path(self._project_root)
        results = await query_codebase(
            query=query,
            target_sqlite_db_path=target_db,
            env=self._env,
            limit=limit,
            offset=offset,
            languages=languages,
            paths=paths,
        )
        return [
            SearchResult(
                file_path=r.file_path,
                language=r.language,
                content=r.content,
                start_line=r.start_line,
                end_line=r.end_line,
                score=r.score,
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> ProjectStatusResponse:
        """Get index stats by querying the SQLite database."""
        db = self._env.get_context(SQLITE_DB)
        index_exists = True
        try:
            with db.readonly() as conn:
                total_chunks = conn.execute("SELECT COUNT(*) FROM code_chunks_vec").fetchone()[0]
                total_files = conn.execute(
                    "SELECT COUNT(DISTINCT file_path) FROM code_chunks_vec"
                ).fetchone()[0]
                lang_rows = conn.execute(
                    "SELECT language, COUNT(*) as cnt FROM code_chunks_vec"
                    " GROUP BY language ORDER BY cnt DESC"
                ).fetchall()
        except sqlite3.OperationalError:
            index_exists = False
            total_chunks = 0
            total_files = 0
            lang_rows = []

        is_indexing = self._index_lock.locked()
        progress = self._indexing_stats if is_indexing else None
        return ProjectStatusResponse(
            indexing=is_indexing,
            total_chunks=total_chunks,
            total_files=total_files,
            languages={lang: cnt for lang, cnt in lang_rows},
            progress=progress,
            index_exists=index_exists,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_indexing(self) -> bool:
        return self._index_lock.locked() or any(not task.done() for task in self._index_tasks)

    @property
    def indexing_stats(self) -> IndexingProgress | None:
        return self._indexing_stats

    @property
    def env(self) -> coco.Environment:
        return self._env

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    async def create(
        project_root: Path,
        embedder: Embedder,
        indexing_params: dict[str, Any],
        query_params: dict[str, Any],
        chunker_registry: dict[str, ChunkerFn] | None = None,
    ) -> Project:
        """Create a project with explicit embedder and per-call params.

        Project-level settings and .gitignore are NOT cached here — the
        indexer loads them fresh from disk on every run so that user edits
        take effect without restarting the daemon.

        Args:
            project_root: Root directory of the codebase to index.
            embedder: Embedding model instance.
            indexing_params: Extra kwargs spread into ``embedder.embed()`` during
                indexing (e.g. ``{"prompt_name": "passage"}``).  Pass ``{}`` for
                no extras.
            query_params: Extra kwargs spread into ``embedder.embed()`` for the
                query side.
            chunker_registry: Optional mapping of file suffix (e.g. ``".toml"``)
                to a ``ChunkerFn``. When a suffix matches, the registered
                chunker is called instead of the built-in splitter.
        """
        settings_dir = project_root / ".cocoindex_code"
        settings_dir.mkdir(parents=True, exist_ok=True)

        db_dir = resolve_db_dir(project_root)
        db_dir.mkdir(parents=True, exist_ok=True)

        cocoindex_db = _cocoindex_db_path(project_root)
        target_sqlite_db = _target_sqlite_db_path(project_root)

        settings = coco.Settings.from_env(cocoindex_db)

        context = coco.ContextProvider()
        context.provide(CODEBASE_DIR, project_root)
        context.provide(SQLITE_DB, coco_sqlite.connect(str(target_sqlite_db), load_vec=True))
        context.provide(EMBEDDER, embedder)
        context.provide(INDEXING_EMBED_PARAMS, dict(indexing_params))
        context.provide(QUERY_EMBED_PARAMS, dict(query_params))
        context.provide(CHUNKER_REGISTRY, dict(chunker_registry) if chunker_registry else {})

        env = coco.Environment(settings, context_provider=context)
        app = coco.App(
            coco.AppConfig(
                name="CocoIndexCode",
                environment=env,
            ),
            indexer_main,
        )

        result = Project.__new__(Project)
        result._env = env
        result._app = app
        result._project_root = project_root
        result._index_lock = asyncio.Lock()
        result._initial_index_done = asyncio.Event()
        result._index_tasks = set()
        result._initial_index_task = None
        result._initial_index_started = None
        result._closed = False
        return result
