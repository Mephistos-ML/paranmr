# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Resolve ``r^-6`` linewidth inputs for fitting workflows."""

from __future__ import annotations

from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.linewidth import mean_inv_r6_by_label
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
)


def resolve_r6_linewidth_inputs(
    *,
    molecule: Molecule,
    isotope_filter: str | None,
    label_kind: str = "atom_label",
) -> SusceptibilityLinewidthInputs:
    """Return resolved ``r^-6`` linewidth inputs for fitting."""
    if molecule.paramagnetic_centre is None:
        raise ValueError("linewidth:method 'r6' requires a paramagnetic centre")

    mean_inv_r6 = mean_inv_r6_by_label(
        nuclei=molecule.nuclei,
        paramagnetic_centre=molecule.paramagnetic_centre,
        isotope_filter=isotope_filter,
        label_kind=label_kind,
    )

    return SusceptibilityLinewidthInputs(
        mean_inv_r6_by_atom_label=mean_inv_r6,
    )
