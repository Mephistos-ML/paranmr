# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Parse magnetic susceptibility data from ORCA outputs.

Provides helpers to extract susceptibility tensors from ORCA
quantum-chemistry calculation files.
"""

import re

import numpy as np

_CHI_T_ROW = re.compile(
    r"^\s*[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+"
    r"(?:----|[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s*$"
)


def read_orca_susceptibility(file_name: str, section: str) -> dict[float, np.ndarray]:
    """Extract temperature-dependent molar magnetic susceptibility tensors.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read (e.g., "casscf" or "nevpt2").

    Returns:
        Dictionary mapping temperature in K to a 3x3 susceptibility tensor.
    """

    susceptibilities = {}

    with open(file_name, "r") as f:
        for line in f:
            if f"QDPT WITH {section.upper()}" in line:
                while (
                    "TEMPERATURE DEPENDENT MOLAR MAGNETIC SUSCEPTIBILITY TENSOR"
                    not in line
                ):
                    line = next(f)
                # Move down until we reach the first temperature header line
                while "TEMPERATURE/K" not in line:
                    line = next(f)
                while "TEMPERATURE/K" in line:
                    _temp = float(line.split("TEMPERATURE/K:")[1])
                    line = next(f)
                    line = next(f)
                    # Read tensor
                    row_1 = [float(val) for val in line.split()]
                    line = next(f)
                    row_2 = [float(val) for val in line.split()]
                    line = next(f)
                    row_3 = [float(val) for val in line.split()]
                    susceptibilities[_temp] = np.array([row_1, row_2, row_3])
                    line = next(f)
                    line = next(f)

    return susceptibilities


def read_orca_chi_t(file_name: str, section: str) -> dict[float, float]:
    """Extract scalar ``chi*T`` values from one ORCA QDPT section.

    Args:
        file_name: Path to the ORCA output file.
        section: Label of the QDPT section to read.

    Returns:
        Dictionary mapping temperature in K to ``chi*T`` in cm³ K mol⁻¹.

    Raises:
        ValueError: If the requested section or scalar susceptibility table is
            not present.
    """

    with open(file_name, "r") as stream:
        lines = stream.readlines()

    section_marker = f"QDPT WITH {section.upper()}"
    section_start = next(
        (index for index, line in enumerate(lines) if section_marker in line),
        None,
    )
    if section_start is None:
        raise ValueError(f"QDPT section {section.upper()!r} not found")

    section_end = next(
        (
            index
            for index in range(section_start + 1, len(lines))
            if "QDPT WITH " in lines[index]
        ),
        len(lines),
    )
    section_lines = lines[section_start:section_end]

    header_index = next(
        (
            index
            for index, line in enumerate(section_lines)
            if "TEMPERATURE DEPENDENT MAGNETIC SUSCEPTIBILITY" in line
            and "TENSOR" not in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError(
            "Scalar temperature-dependent magnetic susceptibility table not found"
        )

    values: dict[float, float] = {}
    table_started = False
    for line in section_lines[header_index + 1 :]:
        match = _CHI_T_ROW.match(line)
        if match is None:
            if table_started:
                break
            continue
        table_started = True
        temperature = float(match.group(1).replace("D", "E").replace("d", "e"))
        chi_t = float(match.group(2).replace("D", "E").replace("d", "e"))
        values[temperature] = chi_t

    if not values:
        raise ValueError(
            "Scalar temperature-dependent magnetic susceptibility table is empty"
        )
    return dict(sorted(values.items()))
