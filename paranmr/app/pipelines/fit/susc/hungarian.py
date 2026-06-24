# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Hungarian assignment branch for susceptibility workflows."""

from __future__ import annotations

import logging
import os

from paranmr.app.loaders.exp_load import save_experiment
from paranmr.app.policies.assignment import resolve_assignment_search_settings
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.assignment.hungarian import (
    fit_with_hungarian_assignment,
)
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel

logger = logging.getLogger(__name__)


def fit_hungarian_assignment(
    *,
    molecule: Molecule,
    susc_model: SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
    assignment_search: str,
    n_attempts: int | None,
    max_iter: int | None,
    r2_threshold: float | None,
    project_name: str,
    delimiter: str,
) -> tuple[float, list[str]]:
    """Run Hungarian assignment search and persist the selected experiment."""

    search_settings = resolve_assignment_search_settings(
        mode=assignment_search,
        n_attempts=n_attempts,
        max_iter=max_iter,
        r2_threshold=r2_threshold,
    )
    logger.info(
        "Hungarian search policy resolved: mode=%s, n_attempts=%d, "
        "max_iter=%d, r2_threshold=%.6f",
        search_settings.mode,
        search_settings.n_attempts,
        search_settings.max_iter,
        search_settings.r2_threshold,
    )

    opt_r2, assignment = fit_with_hungarian_assignment(
        molecule=molecule,
        susc_model=susc_model,
        experiment=experiment,
        average_labels=average_labels,
        n_attempts=search_settings.n_attempts,
        max_iter=search_settings.max_iter,
        r2_threshold=search_settings.r2_threshold,
    )
    logger.info("Hungarian completed: best R² = %.6f", opt_r2)

    save_experiment(
        experiment,
        file_name=os.path.join(
            project_name,
            f"assigned_experiment_{experiment.temperature:.2f}_K.csv",
        ),
        delimiter=delimiter,
        comment=(
            f"# Optimal Assignment (Hungarian)\n"
            f"# r2 = {opt_r2:f}\n"
            f"# T = {experiment.temperature:.2f} K"
        ),
    )

    return opt_r2, assignment
