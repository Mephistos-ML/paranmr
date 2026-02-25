# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse hyperfine coupling data from ORCA outputs.

Provides helpers to extract isotropic and deviatoric (traceless) hyperfine tensors from
ORCA quantum-chemistry calculation files.
"""

import logging

import numpy as np
import numpy.typing as npt

from simpnmr.core.util.strings import remove_letters, remove_numbers

logger = logging.getLogger(__name__)


def read_orca5_property_a_tensors(
    file_name: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Read hyperfine coupling tensors from an ORCA property file.

    Args:
        file_name: Path to the ORCA property file.

    Returns:
        A tuple `(a_iso, a_dtensor)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dtensor` maps atom labels to 3x3 deviatoric (traceless) tensors in MHz.
    """

    a_iso = {}
    a_dtensor = {}

    with open(file_name, "r") as f:
        for line in f:
            if "EPRNMR_ATensor" in line:
                while "Number of stored nuclei" not in line:
                    line = next(f)
                n_calcd = int(line.split()[4])
                while "Nucleus:" not in line:
                    line = next(f)
                for _ in range(n_calcd):
                    label = "{}{}".format(line.split()[2], line.split()[1])
                    for _ in range(6):
                        line = next(f)
                    # Raw values
                    row_1 = [float(val) for val in line.split()[1:]]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()[1:]]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()[1:]]
                    a_dtensor[label] = np.array([row_1, row_2, row_3])
                    for _ in range(9):
                        line = next(f)
                    # Isotropic value
                    a_iso[label] = float(line.split()[-1])
                    a_dtensor[label] -= np.eye(3) * a_iso[label]
                    line = next(f)

    return a_iso, a_dtensor


def read_orca5_output_a_tensors(
    file_name: str,
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract hyperfine (A) tensors from an ORCA 5 output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(a_iso, a_dtensor)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dtensor` maps atom labels to 3x3 deviatoric (traceless) tensors in MHz.
    """

    # Find how many nuclei have been calculated
    with open(file_name, "r") as f:
        for line in f:
            if "Number of nuclei for epr/nmr" in line:
                n_calcd = int(line.split()[-1])

    a_iso = {}
    a_dtensor = {}

    # Read hyperfine data
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                line = next(f)
                line = next(f)
                line = next(f)
                for it in range(n_calcd):
                    line = next(f)
                    tmp = line.split()[1]
                    label = "{}{}".format(remove_numbers(tmp), remove_letters(tmp))
                    for _ in range(5):
                        line = next(f)

                    # Raw matrix in MHz
                    row_1 = [float(val) for val in line.split()]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()]

                    for _ in range(5):
                        line = next(f)
                    a_iso[label] = float(line.split()[-1])

                    for _ in range(9):
                        line = next(f)

                    full = np.array([row_1, row_2, row_3])

                    a_dtensor[label] = full - np.eye(3) * a_iso[label]

    return a_iso, a_dtensor


def read_orca6_output_a_tensors(
    file_name: str,
    orbital_contribution: str = "off",
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract hyperfine (A) tensors from an ORCA 6 output file.

    Args:
        file_name: Path to the ORCA output file.
        orbital_contribution: Controls whether the ORCA A(ORB) principal values
            are included when reconstructing the full A tensor.
            Accepted values are 'on' and 'off'.

    Returns:
        A tuple `(a_iso, a_dtensor)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dtensor` maps atom labels to 3x3 deviatoric (traceless) tensors in MHz.
    """

    # Find how many nuclei have been calculated
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                n_calcd = int(line.split()[5][1:])

    a_iso = {}
    a_dtensor = {}

    # Read hyperfine data
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                for _ in range(n_calcd):
                    while "Nucleus" not in line:
                        line = next(f)
                    tmp = line.split()[1]
                    label = "{}{}".format(remove_numbers(tmp), remove_letters(tmp))
                    # Reconstruct the full tensor from principal components and
                    # orientation
                    a_fc: list[float] | None = None
                    a_sd: list[float] | None = None
                    a_orb: list[float] | None = None
                    r_rows: list[list[float]] = []

                    # Collect principal values reported by ORCA (PAS)
                    # Advance until we reach the Orientation block, collecting
                    # principal values on the way.
                    while "Orientation:" not in line:
                        line = next(f)
                        stripped = line.strip()
                        if stripped.startswith("A(FC)"):
                            parts = stripped.split()
                            a_fc = [float(parts[1]), float(parts[2]), float(parts[3])]
                        elif stripped.startswith("A(SD)"):
                            parts = stripped.split()
                            a_sd = [float(parts[1]), float(parts[2]), float(parts[3])]
                        elif stripped.startswith("A(ORB)"):
                            parts = stripped.split()
                            a_orb = [float(parts[1]), float(parts[2]), float(parts[3])]

                    if a_fc is None or a_sd is None:
                        raise ValueError(
                            "Could not find A(FC)/A(SD) principal values "
                            f"for nucleus {label}"
                        )

                    # FC + SD define the non-orbital hyperfine principal values
                    a_principal_fc_sd = np.array(a_fc) + np.array(a_sd)

                    # Rotation matrix from PAS to the laboratory frame
                    for _axis in ("X", "Y", "Z"):
                        line = next(f)
                        parts = line.split()
                        if not parts or parts[0] != _axis:
                            raise ValueError(
                                f"Unexpected Orientation format for nucleus {label}"
                            )
                        r_rows.append(
                            [float(parts[1]), float(parts[2]), float(parts[3])]
                        )

                    r_mat = np.array(r_rows)

                    if orbital_contribution == "on":
                        if a_orb is None:
                            raise ValueError(
                                "A(ORB) contribution was requested, "
                                "but no A(ORB) block was found in the ORCA output."
                            )
                        a_principal_total = a_principal_fc_sd + np.array(a_orb)
                    else:
                        a_principal_total = a_principal_fc_sd

                    a_pas = np.diag(a_principal_total)

                    # Note: keep the isotropic part defined by FC+SD only
                    # Orbital contributions (when enabled) are added only to the
                    # reconstructed full tensor used for anisotropic/deviatoric part.
                    full = r_mat @ a_pas @ r_mat.T

                    a_iso_fc_sd = float(np.mean(a_principal_fc_sd))
                    a_iso[label] = a_iso_fc_sd
                    a_dtensor[label] = full - np.eye(3) * a_iso[label]

                # Summary log for hyperfine contributions used in this calculation
                if orbital_contribution == "on":
                    logger.info("Hyperfine contributions used: A(FC), A(SD), A(ORB)")
                else:
                    logger.info("Hyperfine contributions used: A(FC), A(SD)")

    return a_iso, a_dtensor
