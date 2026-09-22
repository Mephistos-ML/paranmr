"""Shared pytest configuration."""

import os
import tempfile
from pathlib import Path

# Plotting tests must be headless by default.  Respect an explicit backend
# selected by a developer, while avoiding the macOS GUI backend in CI/local
# test runs where no event loop is available.
os.environ.setdefault("MPLBACKEND", "Agg")

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_config_dir = Path(tempfile.gettempdir()) / "simpnmr_x-matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)
