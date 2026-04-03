"""End-to-end tests for MCP stdio lifecycle."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from cocoindex_code.client import stop_daemon
from cocoindex_code.settings import (
    default_project_settings,
    default_user_settings,
    save_project_settings,
    save_user_settings,
)


def test_mcp_exits_on_stdin_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stdio MCP server should exit promptly once stdin is closed."""
    base_dir = Path(tempfile.mkdtemp(prefix="ccc_mcp_"))
    project_dir = base_dir / "proj"
    project_dir.mkdir()
    (project_dir / "main.py").write_text(
        'def greet(name: str) -> str:\n    return f"hello, {name}"\n'
    )

    monkeypatch.setenv("COCOINDEX_CODE_DIR", str(base_dir))
    save_user_settings(default_user_settings())
    save_project_settings(project_dir, default_project_settings())

    proc = subprocess.Popen(
        [sys.executable, "-m", "cocoindex_code.cli", "mcp"],
        cwd=project_dir,
        env=os.environ.copy(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(
            "ccc mcp did not exit after stdin EOF.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    finally:
        try:
            stop_daemon()
        except Exception:
            pass

    assert proc.returncode == 0, f"ccc mcp exited with {proc.returncode}.\nstderr:\n{stderr}"
