"""Query implementation for codebase search."""

from __future__ import annotations

import heapq
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import QueryResult
from .shared import EMBEDDER, QUERY_EMBED_PARAMS, SQLITE_DB

_HYBRID_FETCH_MIN = 50
_HYBRID_FETCH_MAX = 200
_RRF_K = 60.0
_SEMANTIC_WEIGHT = 1.0
_LEXICAL_WEIGHT = 0.55
_COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "use",
    "used",
    "uses",
    "using",
    "when",
    "where",
    "with",
}
_METADATA_LABEL_STOPWORDS = {
    "code",
    "file",
    "files",
    "language",
    "languages",
    "line",
    "lines",
    "path",
    "paths",
    "symbol",
    "symbols",
}
_TOKEN_SYNONYMS = {
    "cpp": ("gguf", "llama_cpp"),
    "environment": ("env",),
    "environments": ("envs",),
    "kwargs": ("request_kwargs",),
    "llama": ("gguf", "llama_cpp"),
    "parameter": ("param",),
    "parameters": ("params",),
    "variable": ("var",),
    "variables": ("vars",),
}

_IMPLEMENTATION_INTENT_TERMS = {
    "apply",
    "build",
    "completion",
    "completions",
    "construct",
    "constructed",
    "defined",
    "directories",
    "discover",
    "discovered",
    "does",
    "fallback",
    "handle",
    "handled",
    "handles",
    "how",
    "implementation",
    "implemented",
    "infer",
    "logic",
    "merge",
    "normalize",
    "reject",
    "stream",
    "streams",
    "where",
}
_TEST_INTENT_TERMS = {
    "assert",
    "fixture",
    "fixtures",
    "pytest",
    "regression",
    "test",
    "testing",
    "tests",
    "unittest",
}
_DOC_INTENT_TERMS = {
    "doc",
    "docs",
    "documentation",
    "guide",
    "markdown",
    "readme",
    "rst",
}
_DOC_EXTENSION_INTENT_TERMS = {"md", "mdx", "rst"}
_DEFAULT_TEST_ROLE_PENALTY = 0.020
_DEFAULT_DOCS_ROLE_PENALTY = 0.018
_INTENT_ROLE_BONUS = 0.020


@dataclass(frozen=True)
class _LexicalTerm:
    value: str
    content_weight: float
    path_weight: float
    symbol_weight: float = 0.0
    enclosing_weight: float = 0.0
    basename_weight: float = 0.0
    language_weight: float = 0.0


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
            SELECT id, file_path, language, content, start_line, end_line,
                   symbols, enclosing_symbols, file_role, basename, distance
            FROM code_chunks_vec
            WHERE embedding MATCH ? AND k = ? AND language = ?
            ORDER BY distance
            """,
            (embedding_bytes, k, language),
        ).fetchall()
    return conn.execute(
        """
        SELECT id, file_path, language, content, start_line, end_line,
               symbols, enclosing_symbols, file_role, basename, distance
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
               symbols, enclosing_symbols, file_role, basename,
               vec_distance_L2(embedding, ?) as distance
        FROM code_chunks_vec
        {where}
        ORDER BY distance
        LIMIT ?
        """,
        params,
    ).fetchall()


def _is_identifier_like(part: str) -> bool:
    """Return whether a query part looks like a symbol, env var, or path."""
    return (
        "_" in part
        or "/" in part
        or "." in part
        or "$" in part
        or any(c.isdigit() for c in part)
        or part.isupper()
        or bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", part))
    )


def _identifier_subtokens(part: str) -> list[str]:
    """Split code identifiers and paths into searchable sub-tokens."""
    subtokens: list[str] = []
    for segment in re.split(r"[_./$:-]+", part):
        subtokens.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", segment))
    return [token.lower() for token in subtokens]


def _add_lexical_term(terms: dict[str, _LexicalTerm], term: _LexicalTerm) -> None:
    existing = terms.get(term.value)
    if existing is None:
        terms[term.value] = term
        return
    terms[term.value] = _LexicalTerm(
        value=term.value,
        content_weight=max(existing.content_weight, term.content_weight),
        path_weight=max(existing.path_weight, term.path_weight),
        symbol_weight=max(existing.symbol_weight, term.symbol_weight),
        enclosing_weight=max(existing.enclosing_weight, term.enclosing_weight),
        basename_weight=max(existing.basename_weight, term.basename_weight),
        language_weight=max(existing.language_weight, term.language_weight),
    )


def _query_terms(query: str) -> list[_LexicalTerm]:
    """Tokenize query into weighted lexical terms."""
    raw_parts = re.findall(r"[A-Za-z_][A-Za-z0-9_$./-]*|\d+", query)
    has_identifier = any(_is_identifier_like(part) for part in raw_parts)
    stopwords = set(_COMMON_STOPWORDS)
    if has_identifier:
        stopwords.update(_METADATA_LABEL_STOPWORDS)

    terms: dict[str, _LexicalTerm] = {}
    for part in raw_parts:
        value = part.strip("._-/").lower()
        if len(value) < 2 or value in stopwords:
            continue

        identifier_like = _is_identifier_like(part)
        if identifier_like:
            _add_lexical_term(
                terms,
                _LexicalTerm(
                    value=value,
                    content_weight=8.0,
                    path_weight=10.0,
                    symbol_weight=14.0,
                    enclosing_weight=16.0,
                    basename_weight=12.0,
                ),
            )
        elif len(value) >= 4:
            _add_lexical_term(
                terms,
                _LexicalTerm(
                    value=value,
                    content_weight=0.35,
                    path_weight=0.6,
                    symbol_weight=0.8,
                    enclosing_weight=0.8,
                    basename_weight=0.7,
                ),
            )

        for synonym in _TOKEN_SYNONYMS.get(value, ()):
            _add_lexical_term(
                terms,
                _LexicalTerm(
                    value=synonym,
                    content_weight=0.35,
                    path_weight=0.8,
                    symbol_weight=1.2,
                    enclosing_weight=1.4,
                    basename_weight=1.0,
                ),
            )

        if identifier_like:
            for subtoken in _identifier_subtokens(part):
                if len(subtoken) < 3 or subtoken in stopwords:
                    continue
                _add_lexical_term(
                    terms,
                    _LexicalTerm(
                        value=subtoken,
                        content_weight=0.45,
                        path_weight=0.7,
                        symbol_weight=1.4,
                        enclosing_weight=1.8,
                        basename_weight=1.0,
                        language_weight=0.35,
                    ),
                )

    return list(terms.values())[:16]


def _like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _lexical_query(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Fetch lexical candidates with lightweight identifier/path scoring."""
    terms = _query_terms(query)
    if not terms:
        return []

    filter_params: list[Any] = []
    conditions = _filter_conditions(languages=languages, paths=paths, params=filter_params)

    match_clauses: list[str] = []
    match_params: list[Any] = []
    score_terms: list[str] = []
    score_params: list[Any] = []
    for term in terms:
        like = _like_pattern(term.value)
        match_clauses.append(
            "("
            "lower(content) LIKE ? ESCAPE '\\' OR "
            "lower(file_path) LIKE ? ESCAPE '\\' OR "
            "lower(symbols) LIKE ? ESCAPE '\\' OR "
            "lower(enclosing_symbols) LIKE ? ESCAPE '\\' OR "
            "lower(basename) LIKE ? ESCAPE '\\'"
            ")"
        )
        match_params.extend([like, like, like, like, like])
        score_terms.extend(
            [
                "CASE WHEN lower(file_path) LIKE ? ESCAPE '\\' THEN ? ELSE 0.0 END",
                "CASE WHEN lower(content) LIKE ? ESCAPE '\\' THEN ? ELSE 0.0 END",
                "CASE WHEN lower(symbols) LIKE ? ESCAPE '\\' THEN ? ELSE 0.0 END",
                "CASE WHEN lower(enclosing_symbols) LIKE ? ESCAPE '\\' THEN ? ELSE 0.0 END",
                "CASE WHEN lower(basename) LIKE ? ESCAPE '\\' THEN ? ELSE 0.0 END",
                "CASE WHEN lower(language) = ? THEN ? ELSE 0.0 END",
            ]
        )
        score_params.extend(
            [
                like,
                term.path_weight,
                like,
                term.content_weight,
                like,
                term.symbol_weight,
                like,
                term.enclosing_weight,
                like,
                term.basename_weight,
                term.value,
                term.language_weight,
            ]
        )

    conditions.append(f"({' OR '.join(match_clauses)})")
    where = f"WHERE {' AND '.join(conditions)}"
    score_expr = " + ".join(score_terms)

    return conn.execute(
        f"""
        SELECT id, file_path, language, content, start_line, end_line,
               symbols, enclosing_symbols, file_role, basename,
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
        key=lambda r: r[10],
    )


def _query_word_set(query: str) -> set[str]:
    words: set[str] = set()
    for part in re.findall(r"[A-Za-z_][A-Za-z0-9_$./-]*|\d+", query):
        value = part.lower()
        stripped = value.strip("._-/")
        if stripped:
            words.add(stripped)
        if value in _DOC_EXTENSION_INTENT_TERMS or any(
            value.endswith(f".{ext}") for ext in _DOC_EXTENSION_INTENT_TERMS
        ):
            words.add("markdown")
        words.update(_identifier_subtokens(part))
    return words


def _metadata_text(*values: str) -> str:
    return " ".join(values).lower()


def _has_implementation_intent(words: set[str]) -> bool:
    return bool(words & _IMPLEMENTATION_INTENT_TERMS)


def _has_test_intent(words: set[str]) -> bool:
    return bool(words & _TEST_INTENT_TERMS)


def _has_doc_intent(words: set[str]) -> bool:
    return bool(words & _DOC_INTENT_TERMS)


def _rerank_bonus(
    *,
    query: str,
    file_path: str,
    content: str,
    symbols: str,
    enclosing_symbols: str,
    file_role: str,
    basename: str,
) -> float:
    """Small metadata-aware adjustment after RRF fusion.

    This deliberately stays lightweight: semantic/vector rank and lexical rank
    still do the heavy lifting, while file role and symbol context resolve
    close calls between implementation chunks, tests, and documentation.
    """
    words = _query_word_set(query)
    query_terms = _query_terms(query)
    content_lower = content.lower()
    metadata = _metadata_text(file_path, symbols, enclosing_symbols, basename)

    bonus = 0.0
    identifier_hits = 0
    for term in query_terms:
        value = term.value.lower()
        if not value:
            continue
        if term.content_weight >= 8.0 and (value in content_lower or value in metadata):
            identifier_hits += 1
        if value in enclosing_symbols.lower():
            bonus += 0.012
        elif value in symbols.lower():
            bonus += 0.008
        elif value in basename.lower():
            bonus += 0.008
        elif value in file_path.lower():
            bonus += 0.006
        elif value in metadata:
            bonus += 0.002

    bonus += min(0.030, identifier_hits * 0.006)
    if ({"llama", "cpp"} & words) and "gguf" in file_path.lower():
        bonus += 0.060

    implementation_intent = _has_implementation_intent(words)
    test_intent = _has_test_intent(words)
    doc_intent = _has_doc_intent(words)
    if not test_intent and file_role == "test":
        bonus -= _DEFAULT_TEST_ROLE_PENALTY
    if not doc_intent and file_role == "docs":
        bonus -= _DEFAULT_DOCS_ROLE_PENALTY
    if test_intent and file_role == "test":
        bonus += _INTENT_ROLE_BONUS
    if doc_intent and file_role == "docs":
        bonus += _INTENT_ROLE_BONUS

    if implementation_intent and not test_intent and not doc_intent:
        if file_role == "implementation":
            bonus += 0.035
        elif file_role == "config":
            bonus -= 0.010

    return bonus


def _fuse_results(
    semantic_rows: list[tuple[Any, ...]],
    lexical_rows: list[tuple[Any, ...]],
    query: str,
    limit: int,
    offset: int,
) -> list[QueryResult]:
    combined: dict[int, dict[str, Any]] = {}

    for rank, row in enumerate(semantic_rows, start=1):
        (
            chunk_id,
            file_path,
            language,
            content,
            start_line,
            end_line,
            symbols,
            enclosing_symbols,
            file_role,
            basename,
            distance,
        ) = row
        entry = combined.setdefault(
            chunk_id,
            {
                "row": (
                    file_path,
                    language,
                    content,
                    start_line,
                    end_line,
                    symbols,
                    enclosing_symbols,
                    file_role,
                    basename,
                ),
                "score": 0.0,
            },
        )
        entry["score"] += _SEMANTIC_WEIGHT / (_RRF_K + rank)
        entry["semantic_score"] = _l2_to_score(distance)

    for rank, row in enumerate(lexical_rows, start=1):
        (
            chunk_id,
            file_path,
            language,
            content,
            start_line,
            end_line,
            symbols,
            enclosing_symbols,
            file_role,
            basename,
            lexical_score,
        ) = row
        entry = combined.setdefault(
            chunk_id,
            {
                "row": (
                    file_path,
                    language,
                    content,
                    start_line,
                    end_line,
                    symbols,
                    enclosing_symbols,
                    file_role,
                    basename,
                ),
                "score": 0.0,
            },
        )
        lexical_weight = min(1.8, max(0.0, float(lexical_score) / 4.0))
        entry["score"] += (_LEXICAL_WEIGHT * lexical_weight) / (_RRF_K + rank)

    for item in combined.values():
        (
            file_path,
            _language,
            _content,
            _start_line,
            _end_line,
            symbols,
            enclosing_symbols,
            file_role,
            basename,
        ) = item["row"]
        item["score"] += _rerank_bonus(
            query=query,
            file_path=file_path,
            content=_content,
            symbols=symbols,
            enclosing_symbols=enclosing_symbols,
            file_role=file_role,
            basename=basename,
        )

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
        file_path, language, content, start_line, end_line, *_metadata = item["row"]
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

    return _fuse_results(semantic_rows, lexical_rows, query, limit, offset)
