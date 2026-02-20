# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load magnetic susceptibility tensors from CSV or QC output.

Reads external data and returns Susceptibility domain objects.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simpnmr.app.policies.susc import resolve_susceptibility_source
from simpnmr.core.build.susc import susc_from_orca_xt
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.io.csv.susc import read_susceptibilities_csv
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def load_susceptibilities(
    susceptibility_file: str,
    susceptibility_format: str | None = None,
    *,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility tensors from a given file/format.

    Args:
        susceptibility_file: Path to the susceptibility source (CSV or ORCA output).
        susceptibility_format: Optional format identifier. If not provided, the
            format is detected from ``susceptibility_file``.
            - CSV: any string containing ``"csv"``.
            - ORCA: strings starting with ``"orca_"`` (e.g., ``"orca_cas"``)``.
            - Molcas: any string containing ``"molcas"`` (unsupported).
        electronic: Optional electronic-state context passed through to the factory.

    Returns:
        List of :class:`~simpnmr.core.domain.tensors.Susceptibility`.

    Raises:
        ValueError: If the format is unsupported or no data is found.
    """

    backend, section = resolve_susceptibility_source(
        susceptibility_file,
        susceptibility_format,
    )

    if backend == "csv":
        # Returns (tensor, temperature)
        rows = read_susceptibilities_csv(susceptibility_file)
        return [Susceptibility(tensor, temperature=t) for tensor, t in rows]

    logger.info(
        "Susceptibility source: %s (%s)",
        backend.upper(),
        section.upper(),
    )

    return _load_orca_susceptibilities(
        susceptibility_file,
        section=section,
        electronic=electronic,
        g_tensor=g_tensor,
    )


def _load_orca_susceptibilities(
    susceptibility_file: str,
    *,
    section: str | None,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility tensors from an ORCA output.

    Args:
        susceptibility_file: Path to the ORCA output file.
        section: Resolved ORCA QDPT section label to read (e.g. "nevpt2",
        "casscf"). Provided by susceptibility policy.
        electronic: Optional electronic-state context passed through to the factory.
        g_tensor: Optional g-tensor used for g-corrected isotropic susceptibility.

    Returns:
        List of susceptibility tensors as domain objects.

    Raises:
        ValueError: If no supported methods are found or no data is parsed.
    """

    if section is None:
        raise ValueError("ORCA susceptibility section is required but was None")

    # ORCA reader returns temperature -> tensor (XT), typically in cm^3 mol^-1 K.
    tensors = rdrs.read_orca_susceptibility(susceptibility_file, section)

    if not tensors:
        raise ValueError("No susceptibility data found in ORCA output")

    # Derive iso handling mode from available electronic-state context.
    # Priority: g-corr iso -> spin-only -> raw.
    spin = electronic.spin_S if electronic is not None else None
    orbit = electronic.orbit_L if electronic is not None else None
    total_J = electronic.total_J if electronic is not None else None

    # g-correction requires a g-tensor and at least one quantum-number handle.
    if g_tensor is not None and (
        spin is not None or orbit is not None or total_J is not None
    ):
        iso_mode = "g_corr"
        logger.info("Using g-tensor–corrected isotropic magnetic susceptibility")
    elif spin is not None:
        iso_mode = "spin_only"
        logger.info("Using spin-only isotropic magnetic susceptibility")
    else:
        iso_mode = "raw"

    suscs: list[Susceptibility] = []
    for temperature, tensor_xt in tensors.items():
        suscs.append(
            susc_from_orca_xt(
                temperature=float(temperature),
                tensor_xt=tensor_xt,
                iso_mode=iso_mode,
                electronic=electronic,
                g_tensor=g_tensor,
            )
        )

    return suscs
