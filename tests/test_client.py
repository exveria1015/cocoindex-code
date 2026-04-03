"""Tests for client connection handling."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from cocoindex_code import client


def test_client_connect_refuses_when_no_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sock_dir = Path(tempfile.mkdtemp(prefix="ccc_noconn_"))
    sock_path = str(sock_dir / "d.sock")
    monkeypatch.setattr("cocoindex_code.client.daemon_socket_path", lambda: sock_path)

    with pytest.raises(ConnectionRefusedError):
        client._raw_connect_and_handshake()


def test_start_daemon_uses_module_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cocoindex_code.daemon as daemon_mod

    daemon_dir = tmp_path / "daemon"
    log_path = daemon_dir / "daemon.log"
    daemon_dir.mkdir()

    monkeypatch.setattr(daemon_mod, "daemon_dir", lambda: daemon_dir)
    monkeypatch.setattr(daemon_mod, "daemon_log_path", lambda: log_path)

    captured: dict[str, object] = {}

    class DummyProc:
        def poll(self) -> None:
            return None

    def _fake_popen(cmd: list[str], **kwargs: object) -> DummyProc:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return DummyProc()

    monkeypatch.setattr(client.subprocess, "Popen", _fake_popen)

    proc = client.start_daemon()

    assert isinstance(proc, DummyProc)
    assert captured["cmd"] == [sys.executable, "-m", "cocoindex_code.daemon_entry"]
