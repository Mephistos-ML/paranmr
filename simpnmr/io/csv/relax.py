# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write relaxation analysis outputs as CSV.

Provides helpers to export relaxation decompositions and correlation-time fit data.
"""

import logging

import numpy as np
import pandas as pd

from simpnmr.io.csv.csv_util import write_csv_safe

logger = logging.getLogger(__name__)


def save_relaxation_decomposition(
    avg_r1_by_chem_label: dict[str, float],
    avg_r2_by_chem_label: dict[str, float],
    avg_lw_by_chem_label: dict[str, float],
    file_name: str,
    avg_dipolar_by_chem_label: dict[str, float] | None = None,
    avg_contact_by_chem_label: dict[str, float] | None = None,
    avg_curie_by_chem_label: dict[str, float] | None = None,
    delimiter: str = ",",
    comment: str = "",
    verbose: bool = True,
) -> None:
    """Writes relaxation-rate decompositions and linewidths to a CSV file.

    The function writes per-chemical-label averages for total R1/R2 rates and
    linewidths, and optionally includes decomposed contributions.

    Args:
        avg_r1_by_chem_label: Average total R1 rates by chemical label (s^-1).
        avg_r2_by_chem_label: Average total R2 rates by chemical label (s^-1).
        avg_lw_by_chem_label: Average linewidths by chemical label (Hz).
        file_name: Output CSV file path.
        avg_dipolar_by_chem_label: Optional average SBM dipolar R1 contribution (s^-1).
        avg_contact_by_chem_label: Optional average SBM contact R1 contribution (s^-1).
        avg_curie_by_chem_label: Optional average Curie R1 contribution (s^-1).
        delimiter: CSV delimiter.
        comment: Optional comment line appended to the file header. If provided,
            it must begin with ``#`` (or will be prefixed automatically).
        verbose: If ``True``, prints the output file path.

    Returns:
        None.
    """

    # Collect the union of all chemical labels that appear in any dict
    chem_labels: set[str] = set(avg_r1_by_chem_label.keys())
    chem_labels |= set(avg_r2_by_chem_label.keys())
    chem_labels |= set(avg_lw_by_chem_label.keys())

    if avg_dipolar_by_chem_label is not None:
        chem_labels |= set(avg_dipolar_by_chem_label.keys())
    if avg_contact_by_chem_label is not None:
        chem_labels |= set(avg_contact_by_chem_label.keys())
    if avg_curie_by_chem_label is not None:
        chem_labels |= set(avg_curie_by_chem_label.keys())

    chem_labels = sorted(chem_labels)

    # Base columns
    out: dict[str, list] = {
        "chem_label": chem_labels,
        "R1_total (s^-1)": [
            avg_r1_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
        "R2_total (s^-1)": [
            avg_r2_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
        "linewidth (Hz)": [
            avg_lw_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ],
    }

    # Optional decompositions
    if avg_dipolar_by_chem_label is not None:
        out["R1_sbm_dipolar (s^-1)"] = [
            avg_dipolar_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_contact_by_chem_label is not None:
        out["R1_sbm_contact (s^-1)"] = [
            avg_contact_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]
    if avg_curie_by_chem_label is not None:
        out["R1_curie (s^-1)"] = [
            avg_curie_by_chem_label.get(lbl, np.nan) for lbl in chem_labels
        ]

    df = pd.DataFrame(data=out)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Relaxation decomposition written to %s", file_name)

    return


def save_corr_time_fit_data(
    exp_r1: np.ndarray,
    theory_r1: np.ndarray,
    chem_labels: np.ndarray,
    file_name: str,
    fitted_tau_r: float | None = None,
    fitted_tau_e: float | None = None,
    rsquared: float | None = None,
    comment: str = "",
    verbose: bool = True,
) -> None:
    """Write correlation-time fit diagnostics to a CSV file.

    Per-point experimental and theoretical R1 values are written
    row-wise, while global fit diagnostics are repeated across rows as explicit
    columns for easier downstream parsing.

    Args:
        exp_r1: Experimental R1 values (s^-1).
        theory_r1: Theoretical fitted R1 values (s^-1).
        chem_labels: Chemical labels corresponding to each data point.
        file_name: Output CSV file path.
        fitted_tau_r: Optional fitted rotational correlation time ``tau_R`` (s).
        fitted_tau_e: Optional fitted electronic correlation time ``tau_E`` (s).
        rsquared: Optional coefficient of determination ``R^2`` for the fit.
        comment: Optional comment line appended to the file header. If provided,
            it must begin with ``#`` (or will be prefixed automatically).
        verbose: If ``True``, logs the output file path.

    Returns:
        None.
    """
    n_rows = len(chem_labels)

    fitted_tau_r_value = (
        f"{float(fitted_tau_r):.6e}" if fitted_tau_r is not None else np.nan
    )
    fitted_tau_e_value = (
        f"{float(fitted_tau_e):.6e}" if fitted_tau_e is not None else np.nan
    )
    rsquared_value = float(rsquared) if rsquared is not None else np.nan

    out: dict[str, list] = {
        "chem_label": list(chem_labels),
        "R1_exp (s^-1)": list(exp_r1),
        "R1_theory (s^-1)": list(theory_r1),
        "R^2": [rsquared_value] * n_rows,
    }
    if fitted_tau_r is not None:
        out["fitted_tau_R (s)"] = [fitted_tau_r_value] * n_rows
    if fitted_tau_e is not None:
        out["fitted_tau_E (s)"] = [fitted_tau_e_value] * n_rows

    df = pd.DataFrame(data=out)
    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Correlation time fit data written to %s", file_name)

    return
