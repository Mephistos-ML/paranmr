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

from collections.abc import Mapping
from enum import Enum

from paranmr.core.domain.mol import Molecule


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


def validate_pdip_xyz_labels(labels: list[str]) -> None:
    """Validate XYZ labels required by point-dipole HFC.

    Args:
        labels: Atomic labels read from the XYZ structure file.

    Raises:
        ValueError: If any atomic label is missing an index suffix.
    """

    if any(not any(char.isdigit() for char in label) for label in labels):
        raise ValueError(
            "Point-dipole HFC requires indexed XYZ atom labels such as Fe1, H1, H2, C1."
        )


def has_missing_selected_chem_labels(
    molecule: Molecule,
    labels_by_atom_label: Mapping[str, str],
) -> bool:
    """Return whether selected nuclei are missing chemical labels.

    Args:
        molecule: Molecule containing selected runtime nuclei.
        labels_by_atom_label: Mapping from atom labels to chemical labels.

    Returns:
        True if any selected nucleus is absent from `labels_by_atom_label`.
    """

    return any(nuc.label not in labels_by_atom_label for nuc in molecule.nuclei)
