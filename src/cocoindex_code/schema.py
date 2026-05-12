"""Data models for CocoIndex Code."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CodeChunk:
    """Represents an indexed code chunk stored in SQLite."""

    id: int
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    symbols: str
    enclosing_symbols: str
    file_role: str
    basename: str
    embedding: Any  # NDArray - type hint relaxed for compatibility


@dataclass
class QueryResult:
    """Result from a vector similarity query."""

    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    score: float
