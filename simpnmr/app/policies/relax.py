# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Workflow-level policies for relaxation-rate post-processing.

This module contains policy helpers that transform canonical per-nucleus
relaxation outputs into workflow-specific representations. These helpers do not
change the underlying relaxation physics; they define how application-level
workflows choose to aggregate, project, or otherwise consume the evaluated
rates.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from simpnmr.core.domain.mol import Molecule


def average_relaxation_rates_by_chem_label(
    molecule: Molecule,
    rates_by_label: dict[str, float] | None,
) -> dict[str, float] | None:
    """Average per-nucleus relaxation rates by chemical label.

    This helper projects atom-label-indexed relaxation rates onto the
    chemical-label representation used by higher-level workflows such as
    linewidth prediction and correlation-time fitting.

    Args:
        molecule: Molecule providing the ``label`` to ``chem_label`` mapping.
        rates_by_label: Optional mapping from atom label to relaxation rate.

    Returns:
        Mapping from chemical label to averaged relaxation rate, or ``None``
        when no channel data is available.
    """
    if rates_by_label is None:
        return None

    grouped_rates = defaultdict(list)
    for nuc in molecule.nuclei:
        if nuc.label in rates_by_label:
            grouped_rates[nuc.chem_label].append(rates_by_label[nuc.label])

    return {
        chem_label: np.mean(rate_list)
        for chem_label, rate_list in grouped_rates.items()
    }
