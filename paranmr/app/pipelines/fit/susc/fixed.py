# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Assigned shift fitting branch for susceptibility workflows."""

from __future__ import annotations

from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.fitters.shifts import (
    fit_model_to_shifts,
)
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel


def fit_assigned_shifts(
    *,
    model: SusceptibilityModel,
    molecule: Molecule,
    experiment: Experiment,
    average_labels: list[list[str]],
) -> None:
    """Fit an assigned experimental shift dataset."""

    fit_model_to_shifts(
        model=model,
        molecule=molecule,
        experiment=experiment,
        average_labels=average_labels,
    )
