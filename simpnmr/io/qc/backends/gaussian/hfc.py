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

    a_fc_tensors = np.zeros(n_atoms)

    # Read isotropic part
    with open(file_name, "r") as f:
        for line in f:
            if "Isotropic Fermi Contact Couplings" in line:
                line = next(f)
                for it in range(n_atoms):
                    line = next(f)
                    a_fc_tensors[it] = float(line.split()[3])  # MHz

    a_sd_tensors = np.zeros([n_atoms, 3, 3])
    # Read traceless tensor as eigenvalues and eigenvectors
    track = 0
    with open(file_name, "r") as f:
        for line in f:
            if "Anisotropic Spin Dipole Couplings" in line:
                track += 1
                # Make sure in spin density part!
            if "Anisotropic Spin Dipole Couplings" in line and track == 2:
                line = next(f)
                line = next(f)
                line = next(f)
                line = next(f)
                for it in range(n_atoms):
                    line = next(f)
                    val_1 = float(line.split()[2])  # MHz
                    vecs_1 = [float(val) for val in line.split()[-3:]]
                    line = next(f)
                    val_2 = float(line.split()[4])  # MHz
                    vecs_2 = [float(val) for val in line.split()[-3:]]
                    line = next(f)
                    val_3 = float(line.split()[2])  # MHz
                    vecs_3 = [float(val) for val in line.split()[-3:]]
                    vals = np.array([val_1, val_2, val_3])
                    vecs = np.array([vecs_1, vecs_2, vecs_3]).T

                    # Transform back to coordinate frame in MHz
                    a_sd_tensors[it, :, :] = vecs @ np.diag(vals) @ la.inv(vecs)
                    line = next(f)

    if track != 2:
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

    return a_fc_tensors, a_sd_tensors
