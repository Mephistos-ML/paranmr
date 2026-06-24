# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Hungarian assignment search for susceptibility fitting."""

from __future__ import annotations

import copy
import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.fitters.shifts import fit_model_to_shifts
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel

logger = logging.getLogger(__name__)


def fit_with_hungarian_assignment(
    molecule: Molecule,
    susc_model: SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
    n_attempts: int,
    max_iter: int,
    r2_threshold: float,
) -> tuple[float, list[str]]:
    """Fit assignments with alternating Hungarian reassignment and chi refits.

    This routine alternates between fitting the susceptibility model to a
    temporary experiment assignment and recomputing the assignment with the
    Hungarian algorithm from predicted shifts. The first attempt uses the
    assignment already present in the experiment as a warm start. Subsequent
    attempts use random permutations to reduce the risk of converging to a
    poor local optimum. Intermediate attempts are evaluated on temporary
    copies; only the selected best assignment is written back to the caller's
    experiment and model.

    Args:
        molecule: Molecule containing nuclei and tensor data used for fitting.
        susc_model: Susceptibility model instance to fit against the experiment.
        experiment: Experiment containing observed shifts and current
            assignments.
        average_labels: Groups of labels whose predicted shifts are averaged
            before solving the assignment problem.
        n_attempts: Maximum number of restart attempts.
        max_iter: Maximum number of alternating fit-assignment iterations per
            attempt.
        r2_threshold: Early-stop threshold for the best adjusted R^2 found across
            attempts.

    Returns:
        The adjusted R^2 after restoring and re-fitting the best assignment,
        together with that assignment.

    Raises:
        RuntimeError: If no attempt converges within the allowed number of
            iterations and restart attempts.
    """
    logger.info(
        "Starting Hungarian assignment optimization for temperature %.4f K",
        experiment.temperature,
    )
    logger.info(
        "Parameters: n_attempts=%d, max_iter=%d, R² threshold=%.6f",
        n_attempts,
        max_iter,
        r2_threshold,
    )

    def _apply_assignment(exp: Experiment, assignment: list[str]) -> None:
        for i, signal_label in enumerate(assignment):
            exp.signals[i].signal_label = signal_label

    def _current_assignment(exp: Experiment) -> list[str]:
        return [sig.signal_label for sig in exp.signals]

    best_r2 = -np.inf
    best_assignment: list[str] | None = None
    attempt = 0

    while best_r2 < r2_threshold and attempt < n_attempts:
        logger.debug(
            "Attempt %d out of maximum number of attempts %d: "
            "starting Hungarian optimisation",
            attempt + 1,
            n_attempts,
        )

        trial_experiment = copy.deepcopy(experiment)
        trial_model = copy.deepcopy(susc_model)

        initial = _current_assignment(experiment)
        if attempt == 0:
            current_assignment = list(initial)
        else:
            perm = np.random.permutation(len(initial))
            current_assignment = [initial[i] for i in perm]
        _apply_assignment(trial_experiment, current_assignment)

        converged = False
        final_assignment = list(current_assignment)
        for iteration in range(max_iter):
            fit_model_to_shifts(
                model=trial_model,
                molecule=molecule,
                experiment=trial_experiment,
                average_labels=average_labels,
            )
            logger.debug(
                "  Iteration %d/%d: R² = %.6f",
                iteration + 1,
                max_iter,
                trial_model.adj_r2,
            )

            pred_para = trial_model.model(
                trial_model.final_var_values,
                molecule.nuclei,
            )

            dia = {nuc.label: nuc.shift.dia for nuc in molecule.nuclei}
            pred_total = {label: pred_para[label] + dia[label] for label in pred_para}

            signal_to_group: dict[str, list[str]] = {}
            for group in average_labels:
                nuc = next(n for n in molecule.nuclei if n.label == group[0])
                signal_to_group[nuc.signal_label] = group

            avg_pred: dict[str, float] = {}
            for signal_label, group in signal_to_group.items():
                shifts = [pred_total[lbl] for lbl in group]
                avg_pred[signal_label] = float(np.mean(shifts))

            labels_ordered = sorted(avg_pred)
            n_sig = len(trial_experiment.signals)
            n_lbl = len(labels_ordered)
            cost = np.zeros((n_sig, n_lbl))
            for i, sig in enumerate(trial_experiment.signals):
                for j, signal_label in enumerate(labels_ordered):
                    cost[i, j] = abs(sig.shift - avg_pred[signal_label])

            _, col_idx = linear_sum_assignment(cost)
            new_assignment = [labels_ordered[j] for j in col_idx]
            final_assignment = list(new_assignment)

            if new_assignment == current_assignment:
                converged = True
                logger.debug(
                    "  Converged at iteration %d",
                    iteration + 1,
                )
                break

            current_assignment = new_assignment
            _apply_assignment(trial_experiment, current_assignment)

        if converged:
            attempt_r2 = trial_model.adj_r2
            if best_assignment is None or attempt_r2 > best_r2:
                best_r2 = attempt_r2
                best_assignment = list(final_assignment)
            logger.info("Attempt %d converged, R² = %.6f", attempt + 1, attempt_r2)
        else:
            logger.info(
                "Attempt %d did not converge within %d iterations, discarding",
                attempt + 1,
                max_iter,
            )

        attempt += 1

    if best_assignment is None:
        raise RuntimeError(
            f"Hungarian assignment: no attempt converged within {max_iter} "
            f"iterations across {n_attempts} restarts."
            "Fallback to an alternative method if feasible, "
            "or consider increasing max_iter and/or n_attempts."
        )

    _apply_assignment(experiment, best_assignment)
    fit_model_to_shifts(
        model=susc_model,
        molecule=molecule,
        experiment=experiment,
        average_labels=average_labels,
    )

    return susc_model.adj_r2, best_assignment
