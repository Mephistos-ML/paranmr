# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generate constrained assignment permutations.

Provides helpers to enumerate assignment label permutations subject to grouping
constraints for downstream fitting workflows.
"""

import logging

from itertools import chain, permutations, product

import numpy as np
from scipy.optimize import linear_sum_assignment

# Core / domain
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.fitting import models

logger = logging.getLogger(__name__)

def generate_assignment_permutations(
    experiment: Experiment,
    groups: list[list[str]] | None = None,
) -> list[list[str]]:
    """Generate assignment permutations consistent with grouping constraints.

    Args:
        experiment: Reference experiment whose assignment labels are permuted.
        groups: Groups of assignment labels that may be permuted within each group.
            Labels not present in any group are treated as fixed (singletons).

    Returns:
        List of permuted assignment label lists, each ordered to match the
        original experiment signal ordering.

    Raises:
        ValueError: If a grouped label is not present in the experiment.
    """

    exp_labels = list(experiment.keys())

    # Normalise input and avoid mutating caller-provided lists.
    if groups is None:
        group_list: list[list[str]] = []
    else:
        group_list = [list(g) for g in groups]

    grouped_labels: list[str]
    if group_list:
        grouped_labels = list(np.concatenate(group_list))
    else:
        grouped_labels = []

    # Validate grouped labels.
    missing = [lab for lab in grouped_labels if lab not in exp_labels]
    if missing:
        raise ValueError(
            "Grouped assignment label(s) not present in experiment: "
            + ", ".join(missing)
        )

    # Add fixed assignments as singleton groups.
    fixed = [[lab] for lab in exp_labels if lab not in grouped_labels]
    group_list = group_list + fixed

    # Generate all permutations subject to grouping constraints.
    per_group = [permutations(group) for group in group_list]
    perms = [list(chain.from_iterable(e)) for e in product(*per_group, repeat=1)]

    # Map each label in group_list order to its position in the experiment.
    l2i = {label: idx for idx, label in enumerate(exp_labels)}
    group_to_exp = [
        l2i[lab] for lab in (list(np.concatenate(group_list)) if group_list else [])
    ]

    # Reorder each permuted assignment list to match the original experiment ordering.
    order = np.argsort(group_to_exp)
    all_new_assignments = [[new_assgn[o] for o in order] for new_assgn in perms]

    return all_new_assignments


def _fit_with_hungarian_assignment(
    molecule: Molecule,
    susc_model: models.SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
    n_attempts: int = 10,
    max_iter: int = 100,
    r2_threshold: float = 0.99,
) -> tuple[float, list[str]]:
    """Performs Hungarian assignment with alternating optimization.

    Iteratively optimizes the signal-to-label assignment through the following steps:
    1. Fit the χ tensor to the current assignment.
    2. Predict chemical shifts from the fitted χ tensor.
    3. Use the Hungarian algorithm to find the optimal assignment of predictions to
    observed signals.
    4. Repeat until the assignment converges or `max_iter` is reached.

    Multiple random restarts (controlled by `n_attempts`) are used to reduce the
    risk of converging to a local minimum.

    Args:
        molecule (Molecule): Molecule object containing hyperfine tensors.
        susc_model (SuscModel): The susceptibility model to fit.
        experiment (Experiment): Experiment object with observed signals.
        average_labels (list of list of str): Groups of labels to average during
            fitting (e.g., methyl protons).
        n_attempts (int): Number of random restarts to perform.
        max_iter (int): Maximum number of iterations allowed per attempt.

    Returns:
        tuple: A tuple containing:
            - float: The best adjusted $R^2$ value achieved.
            - list of str: The best assignment, represented as a list of chemical
            labels.
    """
    logger.info("Starting Hungarian assignment optimization for temperature %.4f K", experiment.temperature)
    logger.info("Parameters: n_attempts=%d, max_iter=%d, R² threshold=%.6f", n_attempts, max_iter, r2_threshold)
    
    r2_records: list[float] = []
    assignment_records: list[list[str]] = []

    def _apply_assignment(exp: Experiment, assignment: list[str]) -> None:
        for i, chem_label in enumerate(assignment):
            exp.signals[i].assignment = chem_label

    def _current_assignment(exp: Experiment) -> list[str]:
        return [sig.assignment for sig in exp.signals]

    best_r2 = 0.0
    attempt = 0
    while best_r2 < r2_threshold and attempt < n_attempts:
        logger.debug(
            "Attempt %d out of maximum number of attempts %d: "
            "starting Hungarian optimisation",
            attempt + 1,
            n_attempts,
        )

        # Attempt 0: warm start using the assignment already in the exp file.
        # Subsequent attempts: random permutation to escape local minima.
        initial = _current_assignment(experiment)
        if attempt == 0:
            current_assignment = list(initial)
        else:
            perm = np.random.permutation(len(initial))
            current_assignment = [initial[i] for i in perm]
        _apply_assignment(experiment, current_assignment)

        converged = False
        for iteration in range(max_iter):
            # Fit susceptibility model to current assignment
            susc_model.fit_to(
                molecule,
                experiment,
                average_labels=average_labels,
            )
            logger.debug(
                "  Iteration %d/%d: R² = %.6f",
                iteration + 1,
                max_iter,
                susc_model.adj_r2,
            )

            # Predict paramagnetic shifts from fitted model
            pred_para = susc_model.model(
                susc_model.final_var_values, molecule.nuclei
            )

            # Add diamagnetic contribution
            dia = {
                nuc.label: nuc.shift.dia
                for nuc in molecule.nuclei
            }
            pred_total = {
                label: pred_para[label] + dia[label]
                for label in pred_para
            }

            # Map chem_labels to their averaging groups
            cl_to_group: dict[str, list[str]] = {}
            for group in average_labels:
                nuc = next(
                    n
                    for n in molecule.nuclei
                    if n.label == group[0]
                )
                cl_to_group[nuc.chem_label] = group

            # Average predicted shifts within each group
            avg_pred: dict[str, float] = {}
            for cl, group in cl_to_group.items():
                shifts = [pred_total[lbl] for lbl in group]
                avg_pred[cl] = float(np.mean(shifts))

            # Build cost matrix and solve with Hungarian algorithm
            labels_ordered = sorted(avg_pred)
            n_sig = len(experiment.signals)
            n_lbl = len(labels_ordered)
            cost = np.zeros((n_sig, n_lbl))
            for i, sig in enumerate(experiment.signals):
                for j, cl in enumerate(labels_ordered):
                    cost[i, j] = abs(sig.shift - avg_pred[cl])

            _, col_idx = linear_sum_assignment(cost)
            new_assignment = [
                labels_ordered[j] for j in col_idx
            ]

            # Check for convergence
            if new_assignment == current_assignment:
                converged = True
                logger.debug(
                    "  Converged at iteration %d",
                    iteration + 1,
                )
                break

            current_assignment = new_assignment
            _apply_assignment(experiment, current_assignment)

        # Only record and update best_r2 when the attempt converged.
        # If max_iter was hit without convergence, susc_model state is
        # unreliable so we discard this attempt entirely.
        if converged:
            attempt_r2 = susc_model.adj_r2
            r2_records.append(attempt_r2)
            assignment_records.append(new_assignment)
            if attempt_r2 > best_r2:
                best_r2 = attempt_r2
            logger.info(
                "Attempt %d converged, "
                "R² = %.6f",
                attempt + 1,
                attempt_r2
            )
        else:
            logger.info("Attempt %d did not converge within %d iterations, discarding",
                attempt + 1,
                max_iter
            )

        attempt += 1

    if not r2_records:
        raise RuntimeError(
            f"Hungarian assignment: no attempt converged within {max_iter} "
            f"iterations across {n_attempts} restarts."
            "Fallback to an alternative method if feasible, or consider increasing max_iter and/or n_attempts."
        )

    best_idx = int(np.argmax(r2_records))
    best_r2 = r2_records[best_idx]
    best_assignment = assignment_records[best_idx]
    _apply_assignment(experiment, best_assignment)
    
    return best_r2, best_assignment