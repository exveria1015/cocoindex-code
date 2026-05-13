"""Tests for embedder creation helpers."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from cocoindex_code.litellm_embedder import PacedLiteLLMEmbedder
from cocoindex_code.settings import EmbeddingSettings
from cocoindex_code.shared import (
    _RuntimeVectorSchemaEmbedder,
    check_embedding,
    create_embedder,
    is_sentence_transformers_installed,
)


def test_create_embedder_uses_default_litellm_pacing() -> None:
    embedder = create_embedder(
        EmbeddingSettings(
            provider="litellm",
            model="text-embedding-3-small",
        )
    )
    assert isinstance(embedder, PacedLiteLLMEmbedder)
    assert embedder._min_request_interval_seconds == 0.005


def test_create_embedder_uses_paced_litellm_embedder() -> None:
    embedder = create_embedder(
        EmbeddingSettings(
            provider="litellm",
            model="text-embedding-3-small",
            min_interval_ms=300,
        )
    )
    assert isinstance(embedder, PacedLiteLLMEmbedder)
    assert embedder._min_request_interval_seconds == 0.3


def test_create_embedder_litellm_passes_indexing_params_as_constructor_default() -> None:
    """Indexing params become default kwargs forwarded into every litellm call —
    covering paths that don't go through INDEXING_EMBED_PARAMS (dim probe, etc.).
    """
    embedder = create_embedder(
        EmbeddingSettings(provider="litellm", model="cohere/embed-english-v3.0"),
        indexing_params={"input_type": "search_document"},
    )
    assert isinstance(embedder, PacedLiteLLMEmbedder)
    assert embedder._kwargs == {"input_type": "search_document"}


def test_create_embedder_sentence_transformers_ignores_indexing_params() -> None:
    """The SentenceTransformer constructor doesn't accept arbitrary kwargs;
    indexing_params is silently ignored for that provider.
    """
    embedder = create_embedder(
        EmbeddingSettings(
            provider="sentence-transformers", model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        indexing_params={"prompt_name": "passage"},
    )
    # No exception, and prompt_name is not stashed on the constructor —
    # it's a per-call argument supplied via the embed() call site.
    assert not isinstance(embedder, PacedLiteLLMEmbedder)


def test_create_embedder_sentence_transformers_resolves_directml_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_device = object()
    fake_directml = types.ModuleType("torch_directml")
    fake_directml.device = lambda: fake_device  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_directml", fake_directml)

    captured: dict[str, Any] = {}

    class FakeSentenceTransformerEmbedder:
        def __init__(
            self,
            model_name_or_path: str,
            *,
            device: Any = None,
            trust_remote_code: bool = False,
        ) -> None:
            captured["model_name_or_path"] = model_name_or_path
            captured["device"] = device
            captured["trust_remote_code"] = trust_remote_code

        async def embed(self, text: str, **kwargs: Any) -> Any:
            return np.zeros(384, dtype=np.float32)

        async def __coco_vector_schema__(self) -> Any:
            from cocoindex.resources import schema as _schema

            return _schema.VectorSchema(dtype=np.dtype(np.float32), size=384)

        def __coco_memo_key__(self) -> object:
            return ("fake-sentence-transformer",)

    import cocoindex.ops.sentence_transformers as st_module

    monkeypatch.setattr(
        st_module,
        "SentenceTransformerEmbedder",
        FakeSentenceTransformerEmbedder,
    )

    create_embedder(
        EmbeddingSettings(
            provider="sentence-transformers",
            model="sbert/sentence-transformers/all-MiniLM-L6-v2",
            device="directml",
        )
    )

    assert captured["model_name_or_path"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["device"] is fake_device
    assert captured["trust_remote_code"] is True


def test_create_embedder_sentence_transformers_resolves_directml_device_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_directml = types.ModuleType("torch_directml")
    fake_directml.device = lambda device_id=None: ("dml-device", device_id)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_directml", fake_directml)

    captured: dict[str, Any] = {}

    class FakeSentenceTransformerEmbedder:
        def __init__(
            self,
            model_name_or_path: str,
            *,
            device: Any = None,
            trust_remote_code: bool = False,
        ) -> None:
            captured["device"] = device

        async def embed(self, text: str, **kwargs: Any) -> Any:
            return np.zeros(384, dtype=np.float32)

        async def __coco_vector_schema__(self) -> Any:
            from cocoindex.resources import schema as _schema

            return _schema.VectorSchema(dtype=np.dtype(np.float32), size=384)

        def __coco_memo_key__(self) -> object:
            return ("fake-sentence-transformer",)

    import cocoindex.ops.sentence_transformers as st_module

    monkeypatch.setattr(
        st_module,
        "SentenceTransformerEmbedder",
        FakeSentenceTransformerEmbedder,
    )

    create_embedder(
        EmbeddingSettings(
            provider="sentence-transformers",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="dml:1",
        )
    )

    assert captured["device"] == ("dml-device", 1)


def test_create_embedder_sentence_transformers_directml_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> types.ModuleType:
        if name == "torch_directml":
            raise ModuleNotFoundError(name)
        return __import__(name)

    monkeypatch.setattr("cocoindex_code.shared.importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="torch-directml"):
        create_embedder(
            EmbeddingSettings(
                provider="sentence-transformers",
                model="sentence-transformers/all-MiniLM-L6-v2",
                device="dml",
            )
        )


def test_create_embedder_qwen3_transformers_error_has_directml_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_directml = types.ModuleType("torch_directml")
    fake_directml.device = lambda: "dml-device"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_directml", fake_directml)
    monkeypatch.setattr("cocoindex_code.shared.version", lambda name: "4.46.3")

    class FakeSentenceTransformerEmbedder:
        def __init__(
            self,
            model_name_or_path: str,
            *,
            device: Any = None,
            trust_remote_code: bool = False,
        ) -> None:
            raise ValueError(
                "The checkpoint you are trying to load has model type `qwen3` "
                "but Transformers does not recognize this architecture."
            )

    import cocoindex.ops.sentence_transformers as st_module

    monkeypatch.setattr(
        st_module,
        "SentenceTransformerEmbedder",
        FakeSentenceTransformerEmbedder,
    )

    with pytest.raises(RuntimeError) as exc_info:
        create_embedder(
            EmbeddingSettings(
                provider="sentence-transformers",
                model="Qwen/Qwen3-Embedding-0.6B",
                device="directml",
            )
        )

    message = str(exc_info.value)
    assert "Qwen3 embedding checkpoints require Transformers >= 4.51.0" in message
    assert "transformers 4.46.3" in message
    assert 'uv tool install --python 3.12 --upgrade "cocoindex-code[directml]"' in message


def test_is_sentence_transformers_installed_true_in_dev() -> None:
    # Dev env pulls in sentence-transformers via the `dev` extras group.
    assert is_sentence_transformers_installed() is True


def test_is_sentence_transformers_installed_false_when_find_spec_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert is_sentence_transformers_installed() is False


class _StubOkEmbedder:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def embed(self, text: str, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        return np.zeros(384, dtype=np.float32)


class _StubErrEmbedder:
    async def embed(self, text: str, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


class _StubMismatchedSchemaEmbedder:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def embed(self, text: str, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        return np.zeros(640, dtype=np.float32)

    async def __coco_vector_schema__(self) -> Any:
        from cocoindex.resources import schema as _schema

        return _schema.VectorSchema(dtype=np.dtype(np.float32), size=1024)

    def __coco_memo_key__(self) -> object:
        return ("mismatched-schema-stub",)


async def test_check_embedding_ok() -> None:
    result = await check_embedding(_StubOkEmbedder())
    assert result.error is None
    assert result.dim == 384


async def test_check_embedding_error() -> None:
    result = await check_embedding(_StubErrEmbedder())
    assert result.dim is None
    assert result.error is not None
    assert result.error.startswith("RuntimeError:")
    assert "boom" in result.error


async def test_check_embedding_forwards_params() -> None:
    stub = _StubOkEmbedder()
    await check_embedding(stub, {"prompt_name": "passage"})
    assert stub.last_kwargs == {"prompt_name": "passage"}


async def test_check_embedding_no_params_forwards_empty() -> None:
    stub = _StubOkEmbedder()
    await check_embedding(stub)
    assert stub.last_kwargs == {}


async def test_runtime_vector_schema_uses_actual_embedding_length() -> None:
    stub = _StubMismatchedSchemaEmbedder()
    embedder = _RuntimeVectorSchemaEmbedder(stub)

    schema = await embedder.__coco_vector_schema__()

    assert schema.size == 640
    assert schema.dtype == np.dtype(np.float32)


async def test_runtime_vector_schema_embed_forwards_kwargs() -> None:
    stub = _StubMismatchedSchemaEmbedder()
    embedder = _RuntimeVectorSchemaEmbedder(stub)

    vector = await embedder.embed("hello", prompt_name="query")

    assert len(vector) == 640
    assert stub.last_kwargs == {"prompt_name": "query"}
