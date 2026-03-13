# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Application-layer loader for canonical paramagnetic-centre coordinates.

This module transfers paramagnetic-centre coordinates from parsed application
input into the `Molecule` domain container. It performs no user-input parsing
and triggers no downstream calculations.
"""

import logging

from simpnmr.core.domain.mol import Molecule

logger = logging.getLogger(__name__)


def load_paramagnetic_centre(
    molecule: Molecule,
    paramagnetic_centre: list[float] | None,
) -> Molecule:
    """Load the canonical paramagnetic centre into a Molecule.

    This loader transfers canonical paramagnetic-centre coordinates into the
    domain container on `Molecule`. It does not trigger any downstream
    calculations.

    Args:
        molecule: Molecule domain object to enrich.
        paramagnetic_centre: Canonical paramagnetic-centre coordinates or
            `None`.

    Returns:
        The same molecule with `paramagnetic_centre` attached when provided.
    """
    if paramagnetic_centre is None:
        logger.info("No paramagnetic centre provided; skipping load.")
        return molecule

    molecule.paramagnetic_centre = paramagnetic_centre
    return molecule
