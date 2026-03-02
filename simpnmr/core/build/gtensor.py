# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Builders for assembling g-tensors from parsed QC components."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.const.physics import GE


def build_g_tensor_from_dft_components(
    g_rmc: NDArray[np.floating],
    g_dso: NDArray[np.floating],
    g_pso: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Assemble a full DFT-derived g-tensor from decomposed components.

    The input tensors are the decomposed DFT g-tensor contribution matrices
    parsed from QC output. This builder assembles the full physical g-tensor as
    the free-electron contribution plus the reconstructed component tensors.

    Args:
        g_rmc: Relativistic mass correction contribution tensor as a (3, 3)
            array.
        g_dso: Diamagnetic spin-orbit contribution tensor as a (3, 3) array.
        g_pso: Paramagnetic spin-orbit contribution tensor as a (3, 3) array.

    Returns:
        The assembled DFT-derived g-tensor as a (3, 3) ndarray.

    Raises:
        ValueError: If any component is not a (3, 3) matrix.
    """
    g_rmc = np.asarray(g_rmc, dtype=float)
    g_dso = np.asarray(g_dso, dtype=float)
    g_pso = np.asarray(g_pso, dtype=float)

    for name, tensor in (
        ("g_rmc", g_rmc),
        ("g_dso", g_dso),
        ("g_pso", g_pso),
    ):
        if tensor.shape != (3, 3):
            raise ValueError(
                f"Invalid DFT g-tensor component {name}: expected a (3, 3) matrix."
            )

    g_tensor_dft = GE * np.eye(3) + g_rmc + g_dso + g_pso

    return g_tensor_dft
