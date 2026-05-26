# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse g-tensor data from ORCA outputs.

Provides helpers to extract g-tensor components from ORCA quantum-chemistry
calculation files.
"""

import numpy as np

from paranmr.io.qc.errors import ParseError


def read_g_tensor_ab_initio(file_name: str, section: str) -> np.ndarray | None:
    """Extract an ab initio electronic g-tensor from an ORCA output file.

    This reader parses the spin-Hamiltonian g-tensor from an ORCA QDPT block.
    The `ab initio` qualifier is intentional: this tensor represents the
    electronic-structure-derived g-tensor and should remain distinct from any
    other g-tensor variants used elsewhere in the workflow, such as DFT-derived
    tensors with different physical meaning.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        A 3x3 ab initio g-tensor as a NumPy array if found, otherwise None.
    """

    g_tensor = None

    try:
        with open(file_name, "r") as f:
            for line in f:
                # Find the correct QDPT section
                if f"QDPT WITH {section.upper()}" in line:
                    # Go down to the G-matrix header
                    for line in f:
                        if "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN" in line:
                            break
                    # Find "g-matrix:"
                    for line in f:
                        if "g-matrix:" in line:
                            # Next three lines are the rows of the tensor
                            row_1 = [float(val) for val in next(f).split()]
                            row_2 = [float(val) for val in next(f).split()]
                            row_3 = [float(val) for val in next(f).split()]
                            g_tensor = np.array([row_1, row_2, row_3])
                            break
                    break
    except Exception as e:
        raise ParseError(
            message=(
                f"g-tensor could not be parsed from ORCA output "
                f"inside the QDPT {section.upper()} block"
            ),
            path=file_name,
            backend="orca",
            kind="gtensor",
            section=f"QDPT WITH {section.upper()}",
        ) from e

    return g_tensor


def read_g_tensor_dft(
    file_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract decomposed DFT g-tensor contributions from an ORCA output file.

    This reader parses the DFT-level ``ELECTRONIC G-MATRIX`` block reported by
    ORCA and reconstructs the full 3x3 tensors for the ``gRMC``, ``gDSO(tot)``,
    and ``gPSO(tot)`` contributions. ORCA reports these contributions as
    principal values together with an ``Orientation`` matrix. The full tensors
    are reconstructed as ``R.T @ diag(vals) @ R``, where ``R`` is the
    orientation matrix whose rows correspond to the printed X/Y/Z axes.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple ``(g_rmc, g_dso, g_pso)`` of 3x3 NumPy arrays if the DFT
        electronic g-matrix block is found, otherwise None.

    Raises:
        ParseError: If the DFT g-matrix block is found but cannot be parsed.
    """

    g_rmc_vals = None
    g_dso_vals = None
    g_pso_vals = None
    orientation = None
    found_block = False

    try:
        with open(file_name, "r") as f:
            for line in f:
                if line.strip() == "ELECTRONIC G-MATRIX":
                    found_block = True
                    for line in f:
                        stripped = line.strip()

                        if stripped.startswith("gRMC"):
                            g_rmc_vals = np.array(
                                [float(val) for val in stripped.split()[1:4]]
                            )
                        elif stripped.startswith("gDSO(tot)"):
                            g_dso_vals = np.array(
                                [float(val) for val in stripped.split()[1:4]]
                            )
                        elif stripped.startswith("gPSO(tot)"):
                            g_pso_vals = np.array(
                                [float(val) for val in stripped.split()[1:4]]
                            )
                        elif stripped.startswith("Orientation:"):
                            row_x = [float(val) for val in next(f).split()[1:4]]
                            row_y = [float(val) for val in next(f).split()[1:4]]
                            row_z = [float(val) for val in next(f).split()[1:4]]
                            orientation = np.array([row_x, row_y, row_z])
                            break

                    break
    except Exception as e:
        raise ParseError(
            message="DFT g-tensor contributions could not be parsed from ORCA output",
            path=file_name,
            backend="orca",
            kind="gtensor",
            section="ELECTRONIC G-MATRIX",
        ) from e

    if not found_block:
        return None

    if (
        g_rmc_vals is None
        or g_dso_vals is None
        or g_pso_vals is None
        or orientation is None
    ):
        raise ParseError(
            message=(
                "Incomplete DFT g-tensor contribution block in ORCA output: "
                "expected gRMC, gDSO(tot), gPSO(tot), and Orientation"
            ),
            path=file_name,
            backend="orca",
            kind="gtensor",
            section="ELECTRONIC G-MATRIX",
        )

    g_rmc = orientation @ np.diag(g_rmc_vals) @ orientation.T
    g_dso = orientation @ np.diag(g_dso_vals) @ orientation.T
    g_pso = orientation @ np.diag(g_pso_vals) @ orientation.T

    return g_rmc, g_dso, g_pso
