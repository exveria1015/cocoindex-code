"""Query implementation for codebase search."""

from __future__ import annotations

import heapq
import re
import sqlite3
from pathlib import Path
from typing import Any

from .schema import QueryResult
from .shared import EMBEDDER, QUERY_EMBED_PARAMS, SQLITE_DB

_HYBRID_FETCH_MIN = 50
_HYBRID_FETCH_MAX = 200
_RRF_K = 60.0
_SEMANTIC_WEIGHT = 1.0
_LEXICAL_WEIGHT = 1.0


def _l2_to_score(distance: float) -> float:
    """Convert L2 distance to cosine similarity (exact for unit vectors)."""
    return 1.0 - distance * distance / 2.0


def _candidate_limit(limit: int, offset: int) -> int:
    """Number of candidates to fetch from each retrieval side before fusion."""
    requested = max(1, limit + offset)
    return max(requested, min(_HYBRID_FETCH_MAX, max(_HYBRID_FETCH_MIN, requested * 4)))


def _path_conditions(paths: list[str] | None, params: list[Any]) -> str | None:
    if not paths:
        return None
    path_clauses = " OR ".join("file_path GLOB ?" for _ in paths)
    params.extend(paths)
    return f"({path_clauses})"


def _filter_conditions(
    *,
    languages: list[str] | None,
    paths: list[str] | None,
    params: list[Any],
) -> list[str]:
    conditions: list[str] = []
    if languages:
        placeholders = ",".join("?" for _ in languages)
        conditions.append(f"language IN ({placeholders})")
        params.extend(languages)
    path_condition = _path_conditions(paths, params)
    if path_condition is not None:
        conditions.append(path_condition)
    return conditions


def _knn_query(
    conn: sqlite3.Connection,
    embedding_bytes: bytes,
    k: int,
    language: str | None = None,
) -> list[tuple[Any, ...]]:
    """Run a vec0 KNN query, optionally constrained to a language partition."""
    if language is not None:
        return conn.execute(
            """
            SELECT id, file_path, language, content, start_line, end_line, distance
            FROM code_chunks_vec
            WHERE embedding MATCH ? AND k = ? AND language = ?
            ORDER BY distance
            """,
            (embedding_bytes, k, language),
        ).fetchall()
    return conn.execute(
        """
        SELECT id, file_path, language, content, start_line, end_line, distance
        FROM code_chunks_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (embedding_bytes, k),
    ).fetchall()


def _full_scan_query(
    conn: sqlite3.Connection,
    embedding_bytes: bytes,
    limit: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Full scan with SQL-level distance computation and filtering."""
    params: list[Any] = [embedding_bytes]
    conditions = _filter_conditions(languages=languages, paths=paths, params=params)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    return conn.execute(
        f"""
        SELECT id, file_path, language, content, start_line, end_line,
               vec_distance_L2(embedding, ?) as distance
        FROM code_chunks_vec
        {where}
        ORDER BY distance
        LIMIT ?
        """,
        params,
    ).fetchall()


def _query_tokens(query: str) -> list[str]:
    """Tokenize query for lexical retrieval, including identifier subwords."""
    tokens: list[str] = []
    seen: set[str] = set()
    raw_parts = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", query)
    for part in raw_parts:
        pieces = [part]
        pieces.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part.replace("_", " ")))
        for piece in pieces:
            token = piece.lower()
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens[:12]


def _lexical_query(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Fetch lexical candidates with lightweight identifier/path scoring."""
    tokens = _query_tokens(query)
    if not tokens:
        return []

    filter_params: list[Any] = []
    conditions = _filter_conditions(languages=languages, paths=paths, params=filter_params)

    match_clauses: list[str] = []
    match_params: list[Any] = []
    score_terms: list[str] = []
    score_params: list[Any] = []
    for token in tokens:
        like = f"%{token}%"
        match_clauses.append("(lower(content) LIKE ? OR lower(file_path) LIKE ?)")
        match_params.extend([like, like])
        score_terms.extend(
            [
                "CASE WHEN lower(file_path) LIKE ? THEN 4.0 ELSE 0.0 END",
                "CASE WHEN lower(content) LIKE ? THEN 1.0 ELSE 0.0 END",
                "CASE WHEN lower(language) = ? THEN 0.5 ELSE 0.0 END",
            ]
        )
        score_params.extend([like, like, token])

    conditions.append(f"({' OR '.join(match_clauses)})")
    where = f"WHERE {' AND '.join(conditions)}"
    score_expr = " + ".join(score_terms)

    return conn.execute(
        f"""
        SELECT id, file_path, language, content, start_line, end_line,
               ({score_expr}) as lexical_score
        FROM code_chunks_vec
        {where}
        ORDER BY lexical_score DESC, file_path, start_line
        LIMIT ?
        """,
        [*score_params, *filter_params, *match_params, limit],
    ).fetchall()


def _semantic_candidates(
    conn: sqlite3.Connection,
    embedding_bytes: bytes,
    fetch_k: int,
    languages: list[str] | None,
    paths: list[str] | None,
) -> list[tuple[Any, ...]]:
    if paths:
        return _full_scan_query(conn, embedding_bytes, fetch_k, languages, paths)
    if not languages or len(languages) == 1:
        lang = languages[0] if languages else None
        return _knn_query(conn, embedding_bytes, fetch_k, lang)
    return heapq.nsmallest(
        fetch_k,
        (
            row
            for lang in languages
            for row in _knn_query(conn, embedding_bytes, fetch_k, lang)
        ),
        key=lambda r: r[6],
    )


def _fuse_results(
    semantic_rows: list[tuple[Any, ...]],
    lexical_rows: list[tuple[Any, ...]],
    limit: int,
    offset: int,
) -> list[QueryResult]:
    combined: dict[int, dict[str, Any]] = {}

    for rank, row in enumerate(semantic_rows, start=1):
        chunk_id, file_path, language, content, start_line, end_line, distance = row
        entry = combined.setdefault(
            chunk_id,
            {
                "row": (file_path, language, content, start_line, end_line),
                "score": 0.0,
            },
        )
        entry["score"] += _SEMANTIC_WEIGHT / (_RRF_K + rank)
        entry["semantic_score"] = _l2_to_score(distance)

    for rank, row in enumerate(lexical_rows, start=1):
        chunk_id, file_path, language, content, start_line, end_line, lexical_score = row
        entry = combined.setdefault(
            chunk_id,
            {
                "row": (file_path, language, content, start_line, end_line),
                "score": 0.0,
            },
        )
        lexical_weight = min(2.0, max(0.5, float(lexical_score)))
        entry["score"] += (_LEXICAL_WEIGHT * lexical_weight) / (_RRF_K + rank)

    ranked = sorted(
        combined.values(),
        key=lambda item: (
            -float(item["score"]),
            item["row"][0],
            item["row"][3],
        ),
    )
    page = ranked[offset : offset + limit]
    max_score = float(ranked[0]["score"]) if ranked else 1.0
    results: list[QueryResult] = []
    for item in page:
        file_path, language, content, start_line, end_line = item["row"]
        results.append(
            QueryResult(
                file_path=file_path,
                language=language,
                content=content,
                start_line=start_line,
                end_line=end_line,
                score=float(item["score"]) / max_score if max_score else 0.0,
            )
        )
    return results


async def query_codebase(
    query: str,
    target_sqlite_db_path: Path,
    env: Any,
    limit: int = 10,
    offset: int = 0,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[QueryResult]:
    """
    Perform hybrid code search using semantic vector search plus lexical matching.

    Semantic candidates come from sqlite-vec. Lexical candidates use identifier/path
    matching over the same managed table so deleted chunks cannot become stale.
    Both candidate sets are merged with reciprocal-rank fusion.
    """
    if not target_sqlite_db_path.exists():
        raise RuntimeError(
            f"Index database not found at {target_sqlite_db_path}. "
            "Please run a query with refresh_index=True first."
        )

    db = env.get_context(SQLITE_DB)
    embedder = env.get_context(EMBEDDER)
    query_params = env.get_context(QUERY_EMBED_PARAMS)

    query_embedding = await embedder.embed(query, **query_params)
    embedding_bytes = query_embedding.astype("float32").tobytes()
    fetch_k = _candidate_limit(limit, offset)

    with db.readonly() as conn:
        semantic_rows = _semantic_candidates(conn, embedding_bytes, fetch_k, languages, paths)
        lexical_rows = _lexical_query(conn, query, fetch_k, languages, paths)

    return _fuse_results(semantic_rows, lexical_rows, limit, offset)
