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


def resolve_relaxation_conditions(
    config,
    experiment,
) -> tuple[float | None, float | None]:
    """Resolve workflow-level relaxation temperature and magnetic field.

    Application workflows may provide optional relaxation overrides in the
    config. When present, these values take precedence. Otherwise, the values
    are taken from the paired experiment object when available.

    Args:
        config: Workflow config object that may define
            ``relaxation_temperature`` and
            ``relaxation_magnetic_field_tesla`` overrides.
        experiment: Optional experiment object that may provide
            ``temperature`` and ``magnetic_field`` values.

    Returns:
        Tuple ``(temperature, magnetic_field_tesla)`` resolved using the
        precedence ``config override > experiment value > None``.
    """
    temperature = getattr(config, "relaxation_temperature", None)
    magnetic_field_tesla = getattr(config, "relaxation_magnetic_field_tesla", None)

    if temperature is None and experiment is not None:
        temperature = experiment.temperature
    if magnetic_field_tesla is None and experiment is not None:
        magnetic_field_tesla = experiment.magnetic_field

    return temperature, magnetic_field_tesla
