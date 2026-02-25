# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse hyperfine coupling data from ORCA outputs.

Provides helpers to extract isotropic and anisotropic hyperfine tensors from
ORCA quantum-chemistry calculation files.
"""

import numpy as np
import numpy.typing as npt

from simpnmr.core.util.strings import remove_letters, remove_numbers


def read_orca5_property_a_tensors(
    file_name: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Read hyperfine coupling tensors from an ORCA property file.

    Args:
        file_name: Path to the ORCA property file.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dip` maps atom labels to 3x3 traceless dipolar tensors in MHz.
    """

    a_dip = {}
    a_iso = {}

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
                    a_dip[label] = np.array([row_1, row_2, row_3])
                    for _ in range(9):
                        line = next(f)
                    # Isotropic value
                    a_iso[label] = float(line.split()[-1])
                    a_dip[label] -= np.eye(3) * a_iso[label]
                    line = next(f)

    return a_iso, a_dip


def read_orca5_output_a_tensors(
    file_name: str,
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract hyperfine (A) tensors from an ORCA 5 output file.

    Args:
        file_name: Path to the ORCA output file.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dip` maps atom labels to 3x3 traceless dipolar tensors in MHz.
    """

    # Find how many nuclei have been calculated
    with open(file_name, "r") as f:
        for line in f:
            if "Number of nuclei for epr/nmr" in line:
                n_calcd = int(line.split()[-1])

    a_iso = {}
    a_dip = {}

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

                    a_dip[label] = full - np.eye(3) * a_iso[label]

    return a_iso, a_dip


def read_orca6_output_a_tensors(
    file_name: str,
    orbital_contribution: str = "auto",
) -> tuple[dict[str, float], dict[str, npt.NDArray]]:
    """Extract hyperfine (A) tensors from an ORCA 6 output file.

    Args:
        file_name: Path to the ORCA output file.
        orbital_contribution: Controls whether the ORCA A(ORB) principal values
            are included when reconstructing the full A tensor.
            Accepted values are 'auto', 'on', and 'off'.

    Returns:
        A tuple `(a_iso, a_dip)` where:
            * `a_iso` maps atom labels to isotropic couplings in MHz.
            * `a_dip` maps atom labels to 3x3 traceless dipolar tensors in MHz.
    """

    # Find how many nuclei have been calculated
    with open(file_name, "r") as f:
        for line in f:
            if "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE" in line:
                n_calcd = int(line.split()[5][1:])

    a_iso = {}
    a_dip = {}

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
                    a_principal = np.array(a_fc) + np.array(a_sd)
                    if orbital_contribution == "on" or (
                        orbital_contribution == "auto" and a_orb is not None
                    ):
                        a_principal = a_principal + np.array(a_orb)
                    a_pas = np.diag(a_principal)

                    full = r_mat @ a_pas @ r_mat.T

                    a_iso[label] = 1 / 3 * np.trace(full)
                    a_dip[label] = full - np.eye(3) * a_iso[label]

    return a_iso, a_dip
