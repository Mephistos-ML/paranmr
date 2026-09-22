# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Utilities for uncertainty propagation.

This module provides helper routines for propagating parameter uncertainties
to derived quantities using the delta method under the assumption of
independent input variables (diagonal covariance).
"""

import numpy as np


def delta_method_sigma(jac: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Return 1σ output uncertainties via delta method."""
    return np.sqrt((jac**2) @ (sig**2))
