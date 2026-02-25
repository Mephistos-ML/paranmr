# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Policies related to hyperfine coupling (HFC) handling.

This module is intentionally free of IO and QC-backend specifics. It defines
small policy helpers that can be reused across loaders and pipelines.

Key invariants:
- Orbital hyperfine contributions (A(ORB)) must not be combined with
  g-tensor–corrected isotropic susceptibility (g_corr) to avoid double counting.
"""

from __future__ import annotations

from typing import Final

ORBITAL_CONTRIBUTION_ON: Final[str] = "on"
ORBITAL_CONTRIBUTION_OFF: Final[str] = "off"
ORBITAL_CONTRIBUTION_AUTO: Final[str] = "auto"

_ALLOWED_ORBITAL_CONTRIBUTIONS: Final[set[str]] = {
    ORBITAL_CONTRIBUTION_AUTO,
    ORBITAL_CONTRIBUTION_ON,
    ORBITAL_CONTRIBUTION_OFF,
}


def normalise_orbital_contribution(mode: str) -> str:
    """Normalise the orbital contribution mode.

    Args:
        mode: Orbital contribution mode. Expected values are 'auto', 'on', 'off'
            (case-insensitive; surrounding whitespace is ignored).

    Returns:
        Normalised mode ('auto', 'on', or 'off').

    Raises:
        ValueError: If `mode` is not one of the accepted values.
    """

    value = mode.strip().lower()
    if value not in _ALLOWED_ORBITAL_CONTRIBUTIONS:
        raise ValueError(
            "Unknown hyperfine.orbital_contribution value "
            f"{mode!r}; expected one of: auto, on, off."
        )

    return value


def is_orbital_hyperfine_used(effective_mode: str) -> bool:
    """Return whether A(ORB) is used in the effective hyperfine model.

    Args:
        effective_mode: Effective orbital contribution mode ('on' or 'off').
            'auto' is accepted for convenience and treated as 'on'.

    Returns:
        True if A(ORB) is used, otherwise False.
    """

    mode = normalise_orbital_contribution(effective_mode)
    return mode != ORBITAL_CONTRIBUTION_OFF


def assert_orbital_incompatible_with_g_corr(
    *,
    hyperfine_orb_used: bool,
    iso_mode: str,
) -> None:
    """Enforce that A(ORB) is not combined with g_corr susceptibility.

    Args:
        hyperfine_orb_used: Whether A(ORB) is used in the hyperfine model.
        iso_mode: Requested isotropic susceptibility mode (e.g., 'g_corr').

    Raises:
        ValueError: If `hyperfine_orb_used` is True and `iso_mode` requests
            g-tensor correction.
    """

    if hyperfine_orb_used and iso_mode.strip().lower() == "g_corr":
        raise ValueError(
            "A(ORB) hyperfine contributions must not be combined with g-tensor "
            "correction (g_corr)."
        )
