# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Statistical helpers for fitting routines."""

import numpy as np
import numpy.linalg as la
from scipy.optimize._optimize import OptimizeResult


def svd_stdev(curr_fit: OptimizeResult) -> tuple[list[float], list[bool]]:
    """Estimates standard deviations of fit parameters from the Jacobian.

    Uses an SVD of the Jacobian to identify near-singular directions. Singular values
    below a numerical threshold are discarded.

    Args:
        curr_fit: Result object returned by `scipy.optimize.least_squares` or a
            compatible object exposing ``jac``, ``fun``, and ``x``.

    Returns:
        A tuple ``(stdev, has_stdev)`` where:

        - `stdev` is the per-parameter standard deviation estimate.
        - `has_stdev` is a boolean list indicating whether each standard deviation
          is numerically meaningful.
    """

    # SVD of jacobian
    _, s, VT = la.svd(curr_fit.jac, full_matrices=False)
    # Zero threshold as multiple of machine precision
    threshold = np.finfo(float).eps * max(curr_fit.jac.shape) * s[0]
    # Find singular values = 0.
    nonzero_sing = s > threshold
    # Truncate to remove these values
    s = s[nonzero_sing]
    VT = VT[: s.size]
    # Calculate covariance of each parameter using truncated arrays
    pcov = VT.T / s**2 @ VT
    # Scale by reduced chi**2 to remove influence of input sigma (if present)
    # and just obtain standard deviation of fit
    chi2dof = np.sum(curr_fit.fun**2)
    chi2dof /= curr_fit.fun.size - curr_fit.x.size
    pcov *= chi2dof
    stdev = np.sqrt(np.diag(pcov))

    no_stdev = stdev > threshold

    if sum(nonzero_sing) == len(nonzero_sing):
        no_stdev = [True] * len(nonzero_sing)

    return stdev, no_stdev
