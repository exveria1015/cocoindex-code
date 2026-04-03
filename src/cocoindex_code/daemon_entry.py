"""Dedicated module entry point for the background daemon process."""

from __future__ import annotations

from .daemon import run_daemon


def main() -> None:
    """Start the daemon process."""
    run_daemon()


if __name__ == "__main__":
    main()
