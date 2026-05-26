# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write correlation-time fit outputs as CSV.

Provides helpers to export correlation-time fit diagnostics and optional
per-point relaxation decompositions.
"""

import logging

import numpy as np
import pandas as pd

from simpnmr.io.csv.csv_util import write_csv_safe

logger = logging.getLogger(__name__)


def save_corr_time_fit_data(
    exp_r1: np.ndarray,
    theory_r1: np.ndarray,
    chem_labels: np.ndarray,
    file_name: str,
    fitted_tau_r: float | None = None,
    fitted_tau_e: float | None = None,
    rsquared: float | None = None,
    theory_r1_dipolar: np.ndarray | None = None,
    theory_r1_contact: np.ndarray | None = None,
    theory_r1_curie: np.ndarray | None = None,
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
        theory_r1_dipolar: Optional fitted dipolar contribution to ``R1``
            written row-wise in the same order as ``chem_labels``.
        theory_r1_contact: Optional fitted contact contribution to ``R1``
            written row-wise in the same order as ``chem_labels``.
        theory_r1_curie: Optional fitted Curie contribution to ``R1``
            written row-wise in the same order as ``chem_labels``.
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
    if theory_r1_dipolar is not None:
        out["R1_dipolar (s^-1)"] = list(theory_r1_dipolar)
    if theory_r1_contact is not None:
        out["R1_contact (s^-1)"] = list(theory_r1_contact)
    if theory_r1_curie is not None:
        out["R1_curie (s^-1)"] = list(theory_r1_curie)
    if fitted_tau_r is not None:
        out["fitted_tau_R (s)"] = [fitted_tau_r_value] * n_rows
    if fitted_tau_e is not None:
        out["fitted_tau_E (s)"] = [fitted_tau_e_value] * n_rows

    df = pd.DataFrame(data=out)
    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Correlation time fit data written to %s", file_name)

    return
