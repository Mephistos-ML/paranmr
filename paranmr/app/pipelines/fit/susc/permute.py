# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Brute-force permutation assignment branch for susceptibility workflows."""

from __future__ import annotations

import copy
import logging
import os

import numpy as np
from pathos import multiprocessing as mp

from paranmr.app.loaders.exp_load import save_experiment
from paranmr.app.pipelines.fit.susc.fixed import fit_assigned_shifts
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.assignment.permutations import (
    generate_assignment_permutations,
)
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel

logger = logging.getLogger(__name__)


def fit_permuted_assignments(
    *,
    molecule: Molecule,
    experiment: Experiment,
    model: SusceptibilityModel,
    average_labels: list[list[str]],
    assignment_groups: list[list[str]],
    num_threads: int | str,
    project_name: str,
    delimiter: str,
) -> tuple[float, list[str]]:
    """Search assignment permutations and keep the best adjusted R^2."""

    if not len(assignment_groups):
        assignment_groups = [list({nuc.signal_label for nuc in molecule.nuclei})]

    permed_assignments = generate_assignment_permutations(
        experiment=experiment,
        groups=assignment_groups,
    )
    logger.info("There are %s possible permutations", len(permed_assignments))

    if num_threads == "auto":
        worker_count = mp.cpu_count() - 1
    else:
        worker_count = int(num_threads)
    if worker_count > len(permed_assignments):
        worker_count = len(permed_assignments)

    pool = mp.Pool(worker_count)
    logger.info("Parallel permutation search: %s worker processes", worker_count)

    iterables = [
        (
            molecule,
            permed_assgn,
            model,
            copy.deepcopy(experiment),
            average_labels,
        )
        for permed_assgn in permed_assignments
    ]

    try:
        results = pool.starmap(_obtain_r2a, iterables)
    finally:
        pool.close()
        pool.join()

    assignment = permed_assignments[np.nanargmax(results)]
    opt_r2 = float(np.nanmax(results))
    logger.info("Optimal assignment with adj R² = %.6f", opt_r2)

    for it, new in enumerate(assignment):
        experiment.signals[it].signal_label = new

    save_experiment(
        experiment,
        file_name=os.path.join(
            project_name,
            f"assigned_experiment_{experiment.temperature:.2f}_K.csv",
        ),
        delimiter=delimiter,
        comment=(
            f"Optimal Assignment\n"
            f"r2 = {opt_r2:f}\n"
            f"T = {experiment.temperature:.2f} K"
        ),
    )

    return opt_r2, assignment


def _obtain_r2a(
    molecule: Molecule,
    assignment: list[str],
    model: SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
):
    """Fit a susceptibility model for a proposed assignment."""

    for it, new in enumerate(assignment):
        experiment.signals[it].signal_label = new

    fit_assigned_shifts(
        model=model,
        molecule=molecule,
        experiment=experiment,
        average_labels=average_labels,
    )

    return model.adj_r2
