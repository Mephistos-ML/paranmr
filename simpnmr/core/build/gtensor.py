# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Builders for assembling g-tensors from parsed QC components."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.const.physics import GE
from simpnmr.core.domain.mol import Molecule


def build_g_tensor_ab_initio(
    molecule: Molecule,
    g_tensor: NDArray[np.floating] | None,
) -> Molecule:
    """Build ab-initio g-tensor data and attach it to a molecule.

    This builder extracts the isotropic, axial, and rhombic components from a
    3x3 g-tensor expressed in the target working frame. The axial and rhombic
    components are evaluated from the diagonal tensor elements, so callers must
    rotate/project the tensor into the intended principal frame before calling
    this builder. The full tensor and derived scalar components are stored on
    ``molecule.sh``.

    Args:
        molecule: Existing molecule domain object to enrich.
        g_tensor: Full g-tensor matrix with shape (3, 3), or None when no
            ab-initio g-tensor is available.

    Returns:
        The same molecule object enriched with ab-initio g-tensor data.

    Raises:
        ValueError: If ``g_tensor`` is not a (3, 3) matrix.
    """
    if g_tensor is None:
        molecule.sh.g_tensor_ab_initio = None
        molecule.sh.g_tensor_ab_initio_iso = None
        molecule.sh.g_tensor_ab_initio_ax = None
        molecule.sh.g_tensor_ab_initio_rho = None
        return molecule

    g_tensor = np.asarray(g_tensor, dtype=float)
    if g_tensor.shape != (3, 3):
        raise ValueError("Invalid g-tensor: expected a (3, 3) matrix.")

    g_iso = float(np.trace(g_tensor) / 3.0)
    g_ax = float(1.5 * (g_tensor[2, 2] - g_iso))
    g_rho = float(0.5 * (g_tensor[0, 0] - g_tensor[1, 1]))

    molecule.sh.g_tensor_ab_initio = g_tensor
    molecule.sh.g_tensor_ab_initio_iso = g_iso
    molecule.sh.g_tensor_ab_initio_ax = g_ax
    molecule.sh.g_tensor_ab_initio_rho = g_rho

    return molecule


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
