"""Run the checked-out ParaNMR CLI from subprocess-based tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run_paranmr(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    """Run the current checkout's CLI with deterministic import resolution."""
    process_env = {**os.environ, **(env or {})}
    repository_path = str(_REPOSITORY_ROOT)
    existing_path = process_env.get("PYTHONPATH")
    process_env["PYTHONPATH"] = (
        f"{repository_path}{os.pathsep}{existing_path}"
        if existing_path
        else repository_path
    )
    command = [
        sys.executable,
        "-c",
        "from paranmr.cli.main import interface; interface()",
        *arguments,
    ]
    return subprocess.run(
        command,
        cwd=cwd,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
