# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load and save Experiment objects via CSV.

Reads external data and constructs domain Experiment instances, or serializes
them back to CSV.
"""

from __future__ import annotations

import logging
from typing import Iterable

from paranmr.core.domain.exp import Experiment

logger = logging.getLogger(__name__)


def load_experiments(file_names: str | Iterable[str]) -> list[Experiment]:
    """
    Load Experiment objects from one or more CSV files.

    This function is the application-level entry point for experiment IO.
    It delegates CSV parsing to the IO layer and returns domain objects.

    Args:
        file_names: Path or iterable of paths to CSV files.

    Returns:
        List of Experiment objects.
    """
    from paranmr.io.csv.exp import load_experiments_from_csv

    experiments = load_experiments_from_csv(file_names)
    for experiment in experiments:
        if any(signal.l_to_g != 0.0 for signal in experiment.signals):
            logger.warning(
                "Gaussian peak projection at %.2f K ignores nonzero L/G values.",
                experiment.temperature,
            )
    return experiments


def save_experiment(
    experiment: Experiment,
    file_name: str,
    *,
    delimiter: str = ",",
    comment: str | None = None,
) -> None:
    """
    Save a single Experiment object to CSV.

    Args:
        experiment: Experiment instance to save.
        file_name: Output CSV path.
        delimiter: CSV delimiter.
        comment: Optional comment written to the file header.
    """
    from paranmr.io.csv.exp import write_experiment_to_csv

    write_experiment_to_csv(
        experiment,
        file_name=file_name,
        delimiter=delimiter,
        comment=comment,
    )
