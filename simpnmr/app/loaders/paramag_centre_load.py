# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Application-layer loader for canonical paramagnetic-centre coordinates.

This module validates and transfers paramagnetic-centre coordinates from parsed
application input into the `Molecule` domain container. The loader requires the
provided centre to resolve unambiguously against the canonical molecule
geometry; otherwise it fails without mutating the domain object. It performs no
user-input parsing and triggers no downstream calculations.
"""

import logging

import numpy as np

from simpnmr.core.domain.mol import Molecule

logger = logging.getLogger(__name__)


def load_paramagnetic_centre(
    molecule: Molecule,
    paramagnetic_centre: list[float] | None,
) -> Molecule:
    """Load the canonical paramagnetic centre into a Molecule.

    The loader accepts a canonical paramagnetic-centre coordinate and first
    validates that it matches exactly one coordinate already present in the
    molecule geometry. If no match or multiple matches are found, the loader
    raises an error and does not mutate the domain object.

    Args:
        molecule: Molecule domain object to enrich.
        paramagnetic_centre: Canonical paramagnetic-centre coordinates or
            `None`.

    Returns:
        The same molecule with `paramagnetic_centre` attached when provided.

    Raises:
        ValueError: If the provided centre does not match exactly one molecule
            coordinate.
    """
    if paramagnetic_centre is None:
        logger.info("No paramagnetic centre provided; skipping load.")
        return molecule

    centre = np.asarray(paramagnetic_centre, dtype=float)
    matches = [
        coord
        for coord in molecule.coords
        if np.allclose(np.asarray(coord, dtype=float), centre, atol=1e-8)
    ]

    if len(matches) == 0:
        raise ValueError(
            "Paramagnetic centre coordinates do not match any "
            "coordinate in the molecule geometry"
        )
    if len(matches) > 1:
        raise ValueError(
            "Paramagnetic centre coordinates match multiple "
            "coordinates in the molecule geometry"
        )

    molecule.paramagnetic_centre = centre
    return molecule
