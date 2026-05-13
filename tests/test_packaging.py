"""Packaging metadata regression tests."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_directml_extra_keeps_qwen3_compatible_transformers_stack() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())

    directml = pyproject["project"]["optional-dependencies"]["directml"]

    assert "cocoindex[sentence-transformers]>=1.0.0,<1.1.0" in directml
    assert "peft>=0.10.0" in directml
    assert "torch-directml>=0.2.5.dev240914" in directml
    assert "sentence-transformers>=3.0,<4" in directml
    assert "transformers>=4.51,<5" in directml


def test_directml_extra_is_locked_separately_from_default_local_embeddings() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())

    conflicts = pyproject["tool"]["uv"]["conflicts"]

    assert [{"extra": "directml"}, {"extra": "embeddings-local"}] in conflicts
    assert [{"extra": "directml"}, {"extra": "full"}] in conflicts
    assert [{"extra": "directml"}, {"group": "dev"}] in conflicts
