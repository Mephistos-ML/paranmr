"""Susceptibility policy helpers.

This module centralizes application-level decisions for loading susceptibility
sources. In particular, it provides a single source of truth for:

- Determining the susceptibility backend/source (CSV vs ORCA).
- Resolving the ORCA QDPT section to read, honoring legacy overrides when
  provided and otherwise selecting the best available method by priority.

The functions here intentionally return simple primitives (strings/tuples)
so pipelines and loaders do not need to embed backend-specific heuristics.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from simpnmr.io.qc.backends.orca.detect import detect_susc_methods
from simpnmr.io.qc.detect import detect_backend

SusceptibilityBackend = Literal["csv", "orca"]

_ORCA_PREFIX: Final[str] = "orca_"

# Default ORCA susceptibility method priority (best -> fallback).
# Extend this tuple as new ORCA QDPT methods are supported.
ORCA_SUSC_PRIORITY: Final[tuple[str, ...]] = (
    "nevpt2",
    "casscf",
)


def resolve_susceptibility_backend(susceptibility_file: str) -> SusceptibilityBackend:
    """Resolve which backend should be used to read susceptibility.

    Args:
        susceptibility_file: Path to the susceptibility source.

    Returns:
        Backend identifier ("csv" or "orca").

    Raises:
        ValueError: If the backend is unsupported.
    """

    path = Path(susceptibility_file)
    if path.suffix.lower() == ".csv":
        return "csv"

    backend = detect_backend(susceptibility_file)

    if backend == "orca":
        return "orca"

    if backend == "molcas":
        raise ValueError("Susceptibility Molcas support is in active development")

    raise ValueError(
        "Unsupported QC backend for "
        f"susceptibility_file='{susceptibility_file}': {backend}"
    )


def resolve_orca_section(
    susceptibility_file: str,
    susceptibility_format: str | None,
    *,
    priority: tuple[str, ...] = ORCA_SUSC_PRIORITY,
) -> str:
    """Resolve the ORCA QDPT section label to read.

    Resolution order:
      1) Legacy explicit override via `susceptibility_format` (e.g. "orca_nevpt2").
      2) Autodetect available methods in the ORCA output and select by priority.

    Args:
        susceptibility_file: Path to the ORCA output file.
        susceptibility_format: Optional legacy override (e.g. "orca_nevpt2"). If
            None or "orca", autodetection is used.
        priority: Ordered preference list (best -> fallback).

    Returns:
        Selected section label (lowercase, e.g. "nevpt2", "casscf").

    Raises:
        ValueError: If no supported methods are present in the output.
    """

    fmt = _normalize_format(susceptibility_format)

    # Honor explicit legacy override if present.
    explicit = _extract_orca_section_from_format(fmt)
    if explicit is not None:
        return explicit

    # Otherwise autodetect and select by priority.
    return _resolve_orca_section_autodetect(
        susceptibility_file,
        priority=priority,
    )


def resolve_susceptibility_source(
    susceptibility_file: str,
    susceptibility_format: str | None,
) -> tuple[SusceptibilityBackend, str | None]:
    """Resolve backend and (if ORCA) the section to read.

    Args:
        susceptibility_file: Path to the susceptibility source.
        susceptibility_format: Optional legacy override.

    Returns:
        Tuple (backend, section). For CSV, section is None. For ORCA, section is
        a lowercase method label.
    """

    backend = resolve_susceptibility_backend(susceptibility_file)
    if backend == "csv":
        return backend, None

    section = resolve_orca_section(susceptibility_file, susceptibility_format)
    return backend, section


def _normalize_format(fmt: str | None) -> str | None:
    """Normalize a user-provided susceptibility format string."""

    if fmt is None:
        return None

    cleaned = fmt.strip().lower()
    return cleaned or None


def _extract_orca_section_from_format(fmt: str | None) -> str | None:
    """Extract an explicit ORCA section from a legacy format string."""

    if fmt is None:
        return None

    # Accept "orca" as 'auto'.
    if fmt == "orca":
        return None

    if fmt.startswith(_ORCA_PREFIX) and len(fmt) > len(_ORCA_PREFIX):
        return fmt.split(_ORCA_PREFIX, 1)[1]

    return None


@lru_cache(maxsize=32)
def _resolve_orca_section_autodetect(
    susceptibility_file: str,
    *,
    priority: tuple[str, ...],
) -> str:
    """Autodetect ORCA susceptibility methods and pick by priority."""

    methods = detect_susc_methods(susceptibility_file)
    method_set = set(methods)

    for method in priority:
        if method in method_set:
            return method

    raise ValueError(
        "No supported ORCA susceptibility methods found. "
        f"Detected methods: {sorted(method_set)}"
    )
