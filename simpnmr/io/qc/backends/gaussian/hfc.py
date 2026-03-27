# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse Gaussian hyperfine coupling components.

Provides helpers to extract isotropic Fermi-contact values and traceless
spin-dipolar tensors from Gaussian quantum-chemistry calculation files.
"""

import numpy as np
import numpy.linalg as la
import numpy.typing as npt

from simpnmr.io.qc.errors import MissingSectionError


def _read_gaussian_log_fc_values(file_name: str, n_atoms: int) -> npt.NDArray:
    """Read isotropic Fermi-contact values from a Gaussian log file.

    Args:
        file_name: Path to the Gaussian log file.
        n_atoms: Number of atoms expected in the parsed section.

    Returns:
        Array of shape ``(n_atoms,)`` containing isotropic Fermi-contact values
        in MHz.
    """
    a_fc_tensors = np.zeros(n_atoms)

    with open(file_name, "r") as f:
        for line in f:
            if "Isotropic Fermi Contact Couplings" not in line:
                continue

            next(f)
            for atom_idx in range(n_atoms):
                data_line = next(f)
                a_fc_tensors[atom_idx] = float(data_line.split()[3])  # MHz

    return a_fc_tensors


def _read_gaussian_log_sd_tensors(file_name: str, n_atoms: int) -> npt.NDArray:
    """Read traceless spin-dipolar tensors from a Gaussian log file.

    Gaussian outputs may contain multiple ``Anisotropic Spin Dipole Couplings``
    sections depending on corrections and printout details. The final matching
    block is used.

    Args:
        file_name: Path to the Gaussian log file.
        n_atoms: Number of atoms expected in the parsed section.

    Returns:
        Array of shape ``(n_atoms, 3, 3)`` containing traceless spin-dipolar
        tensors in MHz.

    Raises:
        MissingSectionError: If no spin-dipole block is found in the Gaussian
            log.
    """
    a_sd_tensors = np.zeros((n_atoms, 3, 3))
    n_sd_blocks = 0

    with open(file_name, "r") as f:
        for line in f:
            if "Anisotropic Spin Dipole Couplings" not in line:
                continue

            n_sd_blocks += 1
            candidate_tensors = np.zeros((n_atoms, 3, 3))

            for line in f:
                parts = line.split()
                if len(parts) >= 8 and parts[0] == "Baa":
                    first_data_row = line
                    break
            else:
                raise MissingSectionError(
                    message=(
                        "Anisotropic (traceless) hyperfine tensor data rows "
                        "not found in Gaussian log "
                    ),
                    path=file_name,
                    backend="gaussian",
                    kind="hfc",
                    section="Anisotropic Spin Dipole Couplings",
                )

            current_row = first_data_row
            for atom_idx in range(n_atoms):
                row_baa = current_row.split()
                val_1 = float(row_baa[2])  # MHz
                vecs_1 = [float(value) for value in row_baa[-3:]]

                row_bbb = next(f).split()
                val_2 = float(row_bbb[4])  # MHz
                vecs_2 = [float(value) for value in row_bbb[-3:]]

                row_bcc = next(f).split()
                val_3 = float(row_bcc[2])  # MHz
                vecs_3 = [float(value) for value in row_bcc[-3:]]

                vals = np.array([val_1, val_2, val_3])
                vecs = np.array([vecs_1, vecs_2, vecs_3]).T

                # Transform back to coordinate frame in MHz.
                candidate_tensors[atom_idx, :, :] = vecs @ np.diag(vals) @ la.inv(vecs)

                if atom_idx < n_atoms - 1:
                    for next_line in f:
                        next_parts = next_line.split()
                        if len(next_parts) >= 8 and next_parts[0] == "Baa":
                            current_row = next_line
                            break
                    else:
                        raise ValueError(
                            "Gaussian spin-dipole block ended before all atoms "
                            "were parsed. The log format may differ from the "
                            "expected layout."
                        )

            a_sd_tensors = candidate_tensors

    if n_sd_blocks == 0:
        raise MissingSectionError(
            message=(
                "Anisotropic (traceless) hyperfine tensor block "
                "not found in Gaussian log "
            ),
            path=file_name,
            backend="gaussian",
            kind="hfc",
            section="Anisotropic Spin Dipole Couplings",
        )

    return a_sd_tensors


def read_gaussian_log_a_tensors(
    file_name: str,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Extract Gaussian hyperfine components from a log file.

    Gaussian outputs provide isotropic Fermi-contact values separately from
    anisotropic spin-dipolar tensors. This reader returns those raw backend
    quantities without converting the isotropic values into 3x3 tensors.

    Args:
        file_name: Path to the Gaussian log file.

    Returns:
        A tuple `(a_fc_tensors, a_sd_tensors)` where:
            * `a_fc_tensors` is an array of shape `(n_atoms,)` containing isotropic
              Fermi-contact values in MHz.
            * `a_sd_tensors` is an array of shape `(n_atoms, 3, 3)` containing
              traceless spin-dipolar tensors in MHz.
    """

    # Read number of atoms
    with open(file_name, "r") as f:
        for line in f:
            if "NAtoms=" in line:
                spl_line = line.split()
                n_atoms = int(spl_line[spl_line.index("NAtoms=") + 1])
                break

    a_fc_tensors = _read_gaussian_log_fc_values(file_name, n_atoms)
    a_sd_tensors = _read_gaussian_log_sd_tensors(file_name, n_atoms)

    return a_fc_tensors, a_sd_tensors
