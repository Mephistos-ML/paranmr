# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Policies related to hyperfine coupling (HFC) handling.

This module is intentionally free of IO and QC-backend specifics. It defines
small policy helpers that can be reused across loaders and pipelines.

Key notes:
- Orbital hyperfine contributions (A(ORB)) affect the effective hyperfine
  operator used in PCS calculations and do not directly modify the magnetic
  susceptibility tensor.
"""

from __future__ import annotations

from enum import Enum


class OrbitalContribution(str, Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


def normalise_orbital_contribution(mode: str) -> OrbitalContribution:
    """Normalise the orbital contribution mode.

    Args:
        mode: Orbital contribution mode. Expected values are 'auto', 'on', 'off'
            (case-insensitive; surrounding whitespace is ignored).

    Returns:
        Normalised orbital contribution mode.

    Raises:
        ValueError: If `mode` is not one of the accepted values.
    """

    try:
        return OrbitalContribution(mode.strip().lower())
    except ValueError as exc:
        raise ValueError(
            "Unknown hyperfine.orbital_contribution value "
            f"{mode!r}; expected one of: auto, on, off."
        ) from exc


def is_orbital_hyperfine_used(mode: OrbitalContribution) -> bool:
    """Return whether A(ORB) is used in the effective hyperfine model.

    Args:
        mode: Normalised orbital contribution mode.

    Returns:
        True if A(ORB) is used, otherwise False.
    """

    return mode is not OrbitalContribution.OFF
